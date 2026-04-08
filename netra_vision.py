import argparse
import sys
import threading
import time
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import serial
import serial.tools.list_ports
import torch
import torch.nn as nn
from mediapipe.tasks.python import vision as mp_vision

try:
    from train_lstm import DrowsinessLSTM
except ImportError:
    DrowsinessLSTM = None

from netra_common import (
    AdaptiveEARBaseline,
    CONSEC_FRAMES_DROWSY,
    DrowsinessState,
    LEFT_EYE_IDX,
    ModelLoadError,
    RIGHT_EYE_IDX,
    apply_adaptive_thresholds_to_state,
    compute_ear_from_result,
    create_face_landmarker,
    detection_to_np,
    get_head_pose_from_result,
)

try:
    import winsound
except ImportError:
    winsound = None


# ---------------------------------------------------------------------------
# EAR scrolling graph
# ---------------------------------------------------------------------------

def draw_ear_graph(overlay: np.ndarray, ear_history: deque, threshold: float, state: str) -> None:
    if len(ear_history) < 2:
        return
    g_w, g_h = 240, 80
    h, w = overlay.shape[:2]
    bottom_offset = 60 + 10 if state == "drowsy" else 10
    g_x = w - g_w - 10
    g_y = h - g_h - bottom_offset

    sub = overlay[g_y:g_y + g_h, g_x:g_x + g_w]
    overlay[g_y:g_y + g_h, g_x:g_x + g_w] = cv2.addWeighted(
        sub, 0.5, np.zeros_like(sub), 0.5, 0
    )
    cv2.rectangle(overlay, (g_x, g_y), (g_x + g_w, g_y + g_h), (200, 200, 200), 1)

    max_ear = max(0.40, max(ear_history))
    maxlen  = ear_history.maxlen or 100
    pts = []
    for i, v in enumerate(ear_history):
        x = g_x + g_w - int(((len(ear_history) - 1 - i) / (maxlen - 1)) * g_w)
        y = g_y + g_h - int((min(v, max_ear) / max_ear) * g_h)
        pts.append([x, y])
    cv2.polylines(overlay, [np.array(pts, np.int32).reshape((-1, 1, 2))],
                  isClosed=False, color=(0, 255, 255), thickness=2)

    if threshold > 0:
        y_thr = g_y + g_h - int((min(threshold, max_ear) / max_ear) * g_h)
        if g_y <= y_thr <= g_y + g_h:
            cv2.line(overlay, (g_x, y_thr), (g_x + g_w, y_thr), (0, 0, 255), 1)


# ---------------------------------------------------------------------------
# Serial Notifier (ESP32)
# ---------------------------------------------------------------------------

@dataclass
class SerialNotifier:
    port: Optional[str]
    baudrate: int = 9600
    _ser: Optional[serial.Serial] = None
    _enabled: bool = True
    _last_sent: Optional[str] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.port:
            self._enabled = False
            print("[Serial] No port specified — no-serial mode.")
            return
        try:
            self._ser = serial.Serial(
                port=self.port, baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE, timeout=0.05,
                write_timeout=None, dsrdtr=False, rtscts=False, xonxoff=False,
            )
            print(f"[Serial] Connected to {self.port} at {self.baudrate} baud.")
            time.sleep(0.2)
            self._write_raw(b"0", force_set_last="awake")
        except serial.SerialException as exc:
            print(f"[Serial] WARNING: {exc}. Continuing without ESP32.")
            self._enabled = False
            self._ser = None

    def _write_raw(self, byte_val: bytes, force_set_last: Optional[str] = None) -> None:
        if not self._enabled or self._ser is None:
            if force_set_last:
                self._last_sent = force_set_last
            return
        for attempt in range(3):
            try:
                if self._ser.is_open:
                    self._ser.write(byte_val)
                if force_set_last:
                    self._last_sent = force_set_last
                return
            except (serial.SerialException, OSError):
                time.sleep(0.08 * (attempt + 1))

    def send_state_byte(self, state: str) -> None:
        if self._last_sent == state:
            return
        self._write_raw(b"0" if state == "awake" else b"1", force_set_last=state)

    def close(self) -> None:
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Netra — Real-Time Drowsiness Detection (MediaPipe FaceLandmarker)"
    )
    parser.add_argument("--source", choices=["webcam", "file"], default="webcam")
    parser.add_argument("--video-path", type=str)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--serial-port", type=str)
    parser.add_argument("--no-serial", action="store_true")
    parser.add_argument("--show-landmarks", action="store_true",
                        help="Draw eye landmark points on screen.")
    parser.add_argument("--beep", action="store_true")
    parser.add_argument("--fixed-thresholds", action="store_true")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to face_landmarker.task (default: models/face_landmarker.task).")
    parser.add_argument("--use-lstm", action="store_true",
                        help="Use the custom-trained PyTorch LSTM instead of heuristic state machine.")
    parser.add_argument("--fullscreen", action="store_true",
                        help="Open the camera window in fullscreen mode.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Video Capture
# ---------------------------------------------------------------------------

def release_capture(cap: Optional[cv2.VideoCapture]) -> None:
    if cap:
        try:
            cap.release()
        except Exception:
            pass


def init_video_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    if args.source == "webcam":
        cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW if sys.platform == "win32" else 0)
    else:
        if not args.video_path:
            print("ERROR: --video-path required with --source file.")
            sys.exit(1)
        cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open {'webcam' if args.source == 'webcam' else args.video_path}.")
        sys.exit(1)
    return cap


def resize_frame(frame: np.ndarray, width: int = 640) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == width:
        return frame
    return cv2.resize(frame, (width, int(h * (width / w))))


# ---------------------------------------------------------------------------
# CameraStream — dedicated I/O thread (Producer-Consumer)
# ---------------------------------------------------------------------------

class CameraStream:
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.ret, self.frame = self.cap.read()
        self.started = False
        self.lock = threading.Lock()

    def start(self):
        self.started = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while self.started:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame if ret else None

    def read(self):
        with self.lock:
            return (True, self.frame.copy()) if self.frame is not None else (False, None)

    def stop(self):
        self.started = False
        if hasattr(self, "thread"):
            self.thread.join()


# ---------------------------------------------------------------------------
# Beep
# ---------------------------------------------------------------------------

def _beep_worker(freq: int, ms: int) -> None:
    if winsound:
        winsound.Beep(freq, ms)
    else:
        print("[Beep] Drowsiness detected.")


def play_beep() -> None:
    threading.Thread(target=_beep_worker, args=(1500, 200), daemon=True).start()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def draw_landmarks(overlay: np.ndarray, result: mp_vision.FaceLandmarkerResult) -> None:
    """
    Draws eye landmarks onto the overlay.
    """
    if not result.face_landmarks:
        return
    h, w = overlay.shape[:2]
    coords = detection_to_np(result, w, h)
    for idx in RIGHT_EYE_IDX + LEFT_EYE_IDX:
        cv2.circle(overlay, tuple(coords[idx]), 2, (0, 255, 255), -1)

def draw_modern_hud(overlay, state, warn_reason, hud_data, use_lstm):
    """
    Renders an ultra-minimal, glassmorphic telemetry dashboard.
    """
    h, w = overlay.shape[:2]
    
    # 1. Ultra-compact side panel
    panel_w = 140
    panel_h = 115
    cv2.rectangle(overlay, (5, 5), (panel_w, panel_h), (10, 10, 10), -1)
    cv2.rectangle(overlay, (5, 5), (panel_w, panel_h), (50, 50, 50), 1)

    # 2. Status Header and Reason
    header_color = (0, 255, 0) if state == "awake" else (0, 0, 255)
    header_txt = state.upper()
    
    # If drowsy, prioritize the specific reason (Drowsiness/Microsleep) over the generic state
    if state == "drowsy" and warn_reason:
        header_txt = warn_reason.split("(")[0].strip().upper()

    cv2.putText(overlay, header_txt, (12, 18), 
                cv2.FONT_HERSHEY_DUPLEX, 0.42, header_color, 1)

    # 3. AI Risk Bar
    prob = hud_data.get("prob", 0)
    bar_x, bar_y, bar_w, bar_h = 12, 28, 110, 8
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 30, 30), -1)
    fill_w = int((prob / 100) * bar_w)
    bar_color = (255, 120, 0) if prob < 50 else (0, 80, 255)
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
    cv2.putText(overlay, f"AI {prob:.0f}%", (bar_x, bar_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

    # 4. Eye Density Bar
    low_count = hud_data.get("low_count", 0)
    cv2.rectangle(overlay, (bar_x, 58), (bar_x + bar_w, 58 + bar_h), (30, 30, 30), -1)
    fill_density = int((low_count / 30) * bar_w)
    cv2.rectangle(overlay, (bar_x, 58), (bar_x + fill_density, 58 + bar_h), (0, 180, 180), -1)
    cv2.putText(overlay, f"EYE {low_count}/30", (bar_x, 58 + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

    # 5. Baseline / Thresh (Bottom Row)
    thresh = hud_data.get("thresh", 0.0)
    cv2.putText(overlay, f"BASE: {hud_data.get('base', 0):.3f}", (12, 95), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
    cv2.putText(overlay, f"THR: {thresh:.3f}", (12, 107), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

def main() -> None:
    args = parse_args()

    if args.no_serial:
        serial_notifier = None
        print("[Serial] no-serial mode.")
    else:
        serial_notifier = SerialNotifier(port=args.serial_port)

    # Load MediaPipe FaceLandmarker (Deep Learning model)
    try:
        face_landmarker = create_face_landmarker(model_path=args.model_path)
    except ModelLoadError as exc:
        print(f"Model load error: {exc}")
        sys.exit(1)
    print("[MediaPipe] FaceLandmarker DNN loaded successfully.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm_model = None
    scaler_mean = None
    scaler_std = None
    lstm_queue = deque(maxlen=30)
    lstm_current_consec = 0
    
    if args.use_lstm:
        if DrowsinessLSTM is None:
            print("ERROR: Could not import DrowsinessLSTM from train_lstm.py")
            sys.exit(1)
        
        lstm_path = os.path.join("models", "drowsiness_lstm.pt")
        mean_path = os.path.join("models", "scaler_mean.npy")
        std_path  = os.path.join("models", "scaler_std.npy")

        if not os.path.exists(lstm_path) or not os.path.exists(mean_path):
            print(f"ERROR: LSTM assets missing in 'models/'. Have you run train_lstm.py yet?")
            sys.exit(1)

        lstm_model = DrowsinessLSTM().to(device)
        lstm_model.load_state_dict(torch.load(lstm_path, map_location=device, weights_only=True))
        lstm_model.eval()
        
        scaler_mean = np.load(mean_path)
        scaler_std  = np.load(std_path)
        print("[PyTorch] Temporal LSTM sequence classifier loaded successfully.")
    
    cap: Optional[cv2.VideoCapture] = None
    stream: Optional[CameraStream] = None
    frame_ts_ms = 0  # VIDEO mode timestamp counter

    try:
        cap = init_video_capture(args)
        stream = CameraStream(cap).start()

        drowsiness_state = DrowsinessState()
        ear_history = deque(maxlen=100)
        adaptive = None if args.fixed_thresholds else AdaptiveEARBaseline()
        if adaptive:
            print("[Adaptive] EAR thresholds personalise to your open-eye baseline.")

        last_state = "awake"
        window_title = "Netra — Drowsiness Detection  |  MediaPipe"
        frame_idx = 0

        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        if args.fullscreen:
            cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            ret, frame = stream.read()
            if not ret:
                if cap.isOpened() and args.source == "webcam":
                    time.sleep(0.01)
                    continue
                print("End of stream.")
                break

            frame = resize_frame(frame, width=640)
            h_img, w_img = frame.shape[:2]

            # Inference via MediaPipe FaceLandmarker (VIDEO mode needs a timestamp)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = face_landmarker.detect_for_video(mp_image, frame_ts_ms)
            frame_ts_ms += 33  # simulate ~30 fps timestamps

            face_detected = bool(result.face_landmarks)

            if face_detected:
                ear   = compute_ear_from_result(result, w_img, h_img)
                pitch = get_head_pose_from_result(result, w_img, h_img)
            else:
                ear, pitch = 0.0, 0.0

            ear_history.append(ear)

            # Ensure the EAR baseline continues to adapt to the user's face even in LSTM mode
            if adaptive and face_detected:
                adaptive.observe(ear, True, last_state == "awake")
                apply_adaptive_thresholds_to_state(adaptive, drowsiness_state)

            warn_reason = ""
            if args.use_lstm:
                # Provide temporal sequence to PyTorch LSTM
                lstm_queue.append([ear, pitch])
                
                drowsy_prob = 0.0
                n_low_30 = 0
                
                # We need 30 frames for a valid prediction sequence
                if len(lstm_queue) < 30:
                    state = "awake"
                else:
                    seq_arr = np.array(lstm_queue, dtype=np.float32) # (30, 2)
                    # Use .squeeze() on scaler values to ensure correct broadcasting to (30, 2)
                    seq_arr = (seq_arr - scaler_mean.squeeze()) / scaler_std.squeeze()
                    
                    seq_tensor = torch.tensor(seq_arr).unsqueeze(0).to(device) # Shape: (1, 30, 2)
                    
                    with torch.no_grad():
                        outputs = lstm_model(seq_tensor)
                        probs = torch.softmax(outputs, dim=1)
                        drowsy_prob = probs[0][1].item()
                    
                    # 1. Smart Trigger Logic (AI Primary)
                    n_low_30 = sum(1 for f in lstm_queue if f[0] < drowsiness_state.ear_threshold_sleep)
                    
                    # - Scenario A (High AI Confidence): If model is > 85% certain, trigger regardless of density
                    # - Scenario B (Medium Confidence): If model is > 40% certain, trigger if eyes have been shut for 33% of window
                    is_ml_certain = (drowsy_prob > 0.85)
                    is_ml_potential = (drowsy_prob > 0.40) and (n_low_30 >= 10)

                    if is_ml_certain or is_ml_potential:
                        lstm_current_consec += 1
                        reason_prefix = "MICROSLEEP" if is_ml_certain else "DROWSINESS"
                        warn_reason = f"{reason_prefix} ({drowsy_prob*100:.0f}%)"
                    else:
                        lstm_current_consec = 0

                    if lstm_current_consec >= 10: # Faster trigger response (~0.3s)
                        state = "drowsy"
                    else:
                        state = "awake"
                        # Explicit debugging for "Almost Drowsy"
                        if (not is_ml_certain and not is_ml_potential) and drowsy_prob > 0.20:
                            warn_reason = f"MONITORING ({drowsy_prob*100:.0f}%)"

                hud_data = {
                    "prob": drowsy_prob * 100,
                    "low_count": n_low_30,
                    "thresh": drowsiness_state.ear_threshold_sleep,
                    "base": (adaptive.baseline_open_ear() if adaptive else 0.0) or 0.0
                }
            else:
                hud_data = {
                    "prob": 0,
                    "low_count": 0, 
                    "thresh": drowsiness_state.ear_threshold_sleep,
                    "base": (adaptive.baseline_open_ear() if adaptive else 0.0) or 0.0
                }

            if serial_notifier:
                serial_notifier.send_state_byte(state)

            if args.beep:
                if state == "drowsy" and last_state != "drowsy":
                    play_beep()
                elif not face_detected:
                    # Low-pitch beep for missing face
                    winsound.Beep(400, 100) if winsound else None

            last_state = state

            # ------------------------------------------------------------------
            # Overlay Rendering
            # ------------------------------------------------------------------
            # Create a semi-transparent combined view for the modern HUD
            overlay = frame.copy()
            overlay.fill(0) # Clear overlay per-frame for fresh HUD
            combined_view = cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)
            
            # 1. Draw Modern Glass HUD
            draw_modern_hud(combined_view, state, warn_reason, hud_data, args.use_lstm)
            
            # 2. Draw extra status messages below the HUD
            y_extra = 125
            
            if not face_detected:
                cv2.putText(combined_view, "FACE NOT FOUND",
                            (15, y_extra), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                y_extra += 20

            # Optional eye landmark dots
            if args.show_landmarks and face_detected:
                draw_landmarks(combined_view, result)

            # 3. Handle Alert Visuals
            if state == "drowsy":
                h_f, w_f = combined_view.shape[:2]
                # Red bottom alert bar
                cv2.rectangle(combined_view, (0, h_f - 40), (w_f, h_f), (0, 0, 255), -1)
                label_txt = f"ALARM: {warn_reason.upper()}" if warn_reason else "DROWSINESS DETECTED"
                cv2.putText(combined_view, label_txt,
                            (20, h_f - 12), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

            # 4. EAR History Graph
            draw_ear_graph(combined_view, ear_history, drowsiness_state.ear_threshold_sleep, state)
            
            # Final Show
            cv2.imshow(window_title, combined_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            frame_idx += 1
            if frame_idx > 15:
                try:
                    if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    pass

    finally:
        if stream:
            stream.stop()
        release_capture(cap)
        face_landmarker.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        if serial_notifier:
            serial_notifier.close()


if __name__ == "__main__":
    main()
