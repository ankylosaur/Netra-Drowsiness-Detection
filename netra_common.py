import os
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from scipy.spatial import distance as dist


# ---------------------------------------------------------------------------
# MediaPipe Face Landmarker model path
# ---------------------------------------------------------------------------
FACE_LANDMARKER_MODEL = os.path.join("models", "face_landmarker.task")

# ---------------------------------------------------------------------------
# MediaPipe landmark indices for EAR (478-point output with irises)
# We use the same 6 points per eye as the classic Soukupova formulation,
# mapped to the MediaPipe 468-point topology.
# ---------------------------------------------------------------------------
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]

# ---------------------------------------------------------------------------
# Default drowsiness parameters
# ---------------------------------------------------------------------------
EAR_THRESHOLD_SLEEP = 0.26
EAR_THRESHOLD_WAKE  = 0.29

PITCH_THRESHOLD_NOD = -20.0
CONSEC_FRAMES_NOD   = 10

CONSEC_FRAMES_DROWSY = 20
CONSEC_FRAMES_WAKE   = 12

# Adaptive baseline settings
ADAPTIVE_WINDOW_SIZE            = 90
ADAPTIVE_BASELINE_LOW_TRIM_PCT  = 40.0
ADAPTIVE_BASELINE_HIGH_TRIM_PCT = 78.0
ADAPTIVE_SLEEP_RATIO            = 0.72
ADAPTIVE_WAKE_RATIO             = 0.84
ADAPTIVE_MIN_BASELINE           = 0.10
ADAPTIVE_MAX_BASELINE           = 0.50
ADAPTIVE_MIN_SAMPLES            = 15
ADAPTIVE_MIN_EAR_SAMPLE         = 0.05
ADAPTIVE_CALIB_SLEEP            = 0.05
ADAPTIVE_CALIB_WAKE             = 0.10

# ---------------------------------------------------------------------------
# 3D generic face model for head pose (6 stable points)
# MediaPipe indices: Nose tip (1), Chin (152), Left eye corner (263),
# Right eye corner (33), Left mouth corner (287), Right mouth corner (57)
# ---------------------------------------------------------------------------
FACE_3D_POINTS = np.array([
    (0.0,    0.0,    0.0),
    (0.0,   -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0,  170.0, -135.0),
    (-150.0,-150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)
FACE_KEY_LANDMARK_IDX = [1, 152, 263, 33, 287, 57]


class ModelLoadError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# MediaPipe FaceLandmarker factory
# ---------------------------------------------------------------------------

def create_face_landmarker(
    model_path: str | None = None,
    num_faces: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> mp_vision.FaceLandmarker:
    """
    Instantiate a MediaPipe FaceLandmarker using the new Tasks API.
    Returns a FaceLandmarker ready to call `.detect()`.
    """
    path = model_path or FACE_LANDMARKER_MODEL
    if not os.path.isfile(path):
        raise ModelLoadError(
            f"MediaPipe model not found at '{path}'. "
            "Run the download command or place face_landmarker.task in models/."
        )

    base_options = mp_python.BaseOptions(model_asset_path=path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=num_faces,
        min_face_detection_confidence=min_detection_confidence,
        min_face_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)"""
    if eye_points.shape != (6, 2):
        raise ValueError(f"eye_points must have shape (6,2), got {eye_points.shape}")
    p1, p2, p3, p4, p5, p6 = eye_points
    d26 = dist.euclidean(p2, p6)
    d35 = dist.euclidean(p3, p5)
    d14 = dist.euclidean(p1, p4)
    return 1.0 if d14 == 0 else float((d26 + d35) / (2.0 * d14))


def detection_to_np(detection_result, img_w: int, img_h: int) -> np.ndarray:
    """
    Convert the first face in a FaceLandmarkerResult to a (N, 2) pixel array.
    Returns empty array if no face.
    """
    if not detection_result.face_landmarks:
        return np.empty((0, 2), dtype=np.int32)
    lms = detection_result.face_landmarks[0]
    coords = np.array(
        [(int(lm.x * img_w), int(lm.y * img_h)) for lm in lms],
        dtype=np.int32,
    )
    return coords


def compute_ear_from_result(
    detection_result,
    img_w: int,
    img_h: int,
) -> float:
    """Compute average EAR over both eyes from a FaceLandmarkerResult."""
    coords = detection_to_np(detection_result, img_w, img_h)
    if len(coords) == 0:
        return 0.0
    right_eye = coords[RIGHT_EYE_IDX]
    left_eye  = coords[LEFT_EYE_IDX]
    return (eye_aspect_ratio(right_eye) + eye_aspect_ratio(left_eye)) / 2.0


def get_head_pose_from_result(
    detection_result,
    img_w: int,
    img_h: int,
) -> float:
    """Return head pitch (degrees) from a FaceLandmarkerResult; negative = looking down."""
    coords = detection_to_np(detection_result, img_w, img_h)
    if len(coords) == 0:
        return 0.0

    image_points = np.array(
        [coords[i] for i in FACE_KEY_LANDMARK_IDX], dtype="double"
    )
    focal_length = img_w
    camera_matrix = np.array([
        [focal_length, 0,            img_w / 2.0],
        [0,            focal_length, img_h / 2.0],
        [0,            0,            1           ],
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rvec, _ = cv2.solvePnP(
        FACE_3D_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    proj_matrix = np.hstack((rmat, np.zeros((3, 1))))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_matrix)
    return float(euler[0][0])


# ---------------------------------------------------------------------------
# Adaptive EAR Baseline
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveEARBaseline:
    window_size:            int   = ADAPTIVE_WINDOW_SIZE
    baseline_low_trim_pct:  float = ADAPTIVE_BASELINE_LOW_TRIM_PCT
    baseline_high_trim_pct: float = ADAPTIVE_BASELINE_HIGH_TRIM_PCT
    sleep_ratio:  float = ADAPTIVE_SLEEP_RATIO
    wake_ratio:   float = ADAPTIVE_WAKE_RATIO
    min_baseline: float = ADAPTIVE_MIN_BASELINE
    max_baseline: float = ADAPTIVE_MAX_BASELINE
    min_samples:  int   = ADAPTIVE_MIN_SAMPLES
    min_ear_sample: float = ADAPTIVE_MIN_EAR_SAMPLE
    fallback_sleep: float = EAR_THRESHOLD_SLEEP
    fallback_wake:  float = EAR_THRESHOLD_WAKE
    _buf: deque = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._buf = deque(maxlen=self.window_size)

    def reset(self) -> None:
        self._buf.clear()

    def observe(self, ear: float, face_present: bool, prev_state_awake: bool) -> None:
        if not face_present or ear < self.min_ear_sample or not prev_state_awake:
            return
        self._buf.append(float(ear))

    def baseline_open_ear(self) -> Optional[float]:
        if len(self._buf) < self.min_samples:
            return None
        arr = np.array(self._buf, dtype=np.float64)
        lo  = float(np.percentile(arr, self.baseline_low_trim_pct))
        hi  = float(np.percentile(arr, self.baseline_high_trim_pct))
        if hi <= lo:
            b = float(np.median(arr))
        else:
            mid = arr[(arr >= lo) & (arr <= hi)]
            b   = float(np.median(mid)) if len(mid) >= 5 else float(np.median(arr))
        return float(np.clip(b, self.min_baseline, self.max_baseline))

    def thresholds(self) -> Tuple[float, float]:
        baseline = self.baseline_open_ear()
        if baseline is None:
            return (self.fallback_sleep, self.fallback_wake)
        sleep_t = baseline * self.sleep_ratio
        wake_t  = baseline * self.wake_ratio
        if wake_t < sleep_t + 0.015:
            wake_t = sleep_t + 0.015
        wake_t  = min(wake_t, baseline * 0.98)
        sleep_t = max(sleep_t, 0.05)
        return (float(sleep_t), float(wake_t))


# ---------------------------------------------------------------------------
# Drowsiness State Machine
# ---------------------------------------------------------------------------

@dataclass
class DrowsinessState:
    low_ear_frames:       int   = 0
    high_ear_frames:      int   = 0
    low_pitch_frames:     int   = 0
    ear_threshold_sleep:  float = EAR_THRESHOLD_SLEEP
    ear_threshold_wake:   float = EAR_THRESHOLD_WAKE
    consec_frames_drowsy: int   = CONSEC_FRAMES_DROWSY
    consec_frames_wake:   int   = CONSEC_FRAMES_WAKE
    state:          str = "awake"
    trigger_reason: str = ""

    def update(self, ear: float, pitch: float = 0.0) -> Tuple[str, str, int]:
        """
        Update state given current EAR and head pitch.
        Returns (state, trigger_reason, low_ear_frames).
        """
        if ear <= self.ear_threshold_sleep:
            self.low_ear_frames  += 1
            self.high_ear_frames  = 0
        elif ear >= self.ear_threshold_wake:
            self.high_ear_frames += 1
            self.low_ear_frames   = 0
        else:
            self.low_ear_frames  = max(0, self.low_ear_frames  - 1)
            self.high_ear_frames = max(0, self.high_ear_frames - 1)

        if pitch <= PITCH_THRESHOLD_NOD:
            self.low_pitch_frames += 1
        else:
            self.low_pitch_frames = max(0, self.low_pitch_frames - 1)

        if self.state == "awake":
            if self.low_ear_frames >= self.consec_frames_drowsy:
                self.state = "drowsy"
                self.trigger_reason = "MICRO-SLEEP"
            elif self.low_pitch_frames >= CONSEC_FRAMES_NOD:
                self.state = "drowsy"
                self.trigger_reason = "HEAD NODDING"
        elif self.state == "drowsy":
            if self.high_ear_frames >= self.consec_frames_wake and self.low_pitch_frames == 0:
                self.state = "awake"
                self.trigger_reason = ""

        return self.state, self.trigger_reason, self.low_ear_frames


def apply_adaptive_thresholds_to_state(
    adaptive: Optional[AdaptiveEARBaseline],
    drowsiness_state: DrowsinessState,
) -> bool:
    """Push personalised thresholds into the state machine. Returns False during calibration."""
    if adaptive is None:
        return True
    if adaptive.baseline_open_ear() is None:
        drowsiness_state.ear_threshold_sleep = ADAPTIVE_CALIB_SLEEP
        drowsiness_state.ear_threshold_wake  = ADAPTIVE_CALIB_WAKE
        return False
    sleep_t, wake_t = adaptive.thresholds()
    drowsiness_state.ear_threshold_sleep = sleep_t
    drowsiness_state.ear_threshold_wake  = wake_t
    return True
