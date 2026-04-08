"""
evaluate_datasets.py
====================
Headless evaluation of the Netra drowsiness pipeline on labelled video datasets.

Computes a full confusion matrix, Precision, Recall and F1-Score, then saves
a professional Seaborn/Matplotlib confusion matrix heatmap to disk.

Usage:
    python evaluate_datasets.py --root-dir datasets
"""

import argparse
import os
import random
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import seaborn as sns
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

from netra_common import (
    AdaptiveEARBaseline,
    DrowsinessState,
    ModelLoadError,
    apply_adaptive_thresholds_to_state,
    compute_ear_from_result,
    create_face_landmarker,
    get_head_pose_from_result,
)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def collect_videos(root_dir: str) -> List[Tuple[str, int]]:
    """
    Walk root_dir and collect (video_path, label) pairs.

    Convention:
        path containing 'microsleeps', 'yawning', or 'drowsy'  → label 1
        path containing 'normal'                               → label 0
    """
    video_label_pairs: List[Tuple[str, int]] = []

    for dirpath, _, filenames in os.walk(root_dir):
        parts_lower = {p.lower() for p in dirpath.split(os.sep)}

        label = None
        if "microsleeps" in parts_lower or "yawning" in parts_lower or "drowsy" in parts_lower:
            label = 1
        elif "normal" in parts_lower:
            label = 0

        if label is None:
            continue

        for fname in filenames:
            if not fname.lower().endswith(".mp4"):
                continue
            video_label_pairs.append((os.path.join(dirpath, fname), label))

    return video_label_pairs


def _resize_frame(frame, width: int = 640):
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale = width / float(w)
    return cv2.resize(frame, (width, int(h * scale)))


# ---------------------------------------------------------------------------
# Per-video prediction
# ---------------------------------------------------------------------------

def predict_video_label(
    video_path: str,
    face_landmarker: mp_vision.FaceLandmarker,
    fixed_thresholds: bool = False,
) -> int:
    """Run the full EAR + Pitch pipeline headless. Returns 1 if drowsy ever entered."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Could not open '{video_path}'. Skipping.")
        return 0

    drowsiness_state = DrowsinessState()
    adaptive = None if fixed_thresholds else AdaptiveEARBaseline()
    last_state = "awake"
    entered_drowsy = False
    frame_idx = 0
    frame_ts_ms = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 3 != 0:
            frame_idx += 1
            frame_ts_ms += 33
            continue

        frame = _resize_frame(frame, width=640)
        h_img, w_img = frame.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = face_landmarker.detect_for_video(mp_image, frame_ts_ms)

        face_detected = bool(result.face_landmarks)

        if face_detected:
            ear   = compute_ear_from_result(result, w_img, h_img)
            pitch = get_head_pose_from_result(result, w_img, h_img)
            if adaptive:
                adaptive.observe(ear, True, last_state == "awake")
        else:
            ear, pitch = 0.0, 0.0

        calibrated = True
        if adaptive:
            calibrated = apply_adaptive_thresholds_to_state(adaptive, drowsiness_state)

        if face_detected:
            state, _, _ = drowsiness_state.update(ear, pitch)
        elif adaptive and not calibrated:
            state, _, _ = drowsiness_state.update(ear=1.0)
        else:
            state, _, _ = drowsiness_state.update(ear=0.0)

        last_state = state
        if state == "drowsy":
            entered_drowsy = True
            break

        frame_idx  += 1
        frame_ts_ms += 33

    cap.release()
    return 1 if entered_drowsy else 0


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    tp: int, fn: int, fp: int, tn: int,
    output_path: str = "confusion_matrix.png",
) -> None:
    """
    Render a Seaborn Confusion Matrix heatmap and save it to disk.
    """
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.set_theme(style="whitegrid", font_scale=1.2)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 18, "weight": "bold"},
        xticklabels=["Predicted: Awake", "Predicted: Drowsy"],
        yticklabels=["Actual: Awake",    "Actual: Drowsy"],
        cbar_kws={"label": "Count"},
    )
    ax.set_title("Netra — Drowsiness Detection\nConfusion Matrix", fontsize=15, fontweight="bold", pad=16)
    ax.set_ylabel("Ground Truth", fontsize=12)
    ax.set_xlabel("Prediction",   fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[Metrics] Confusion matrix saved → {os.path.abspath(output_path)}")


def print_metrics(tp: int, fn: int, fp: int, tn: int) -> None:
    total = tp + fn + fp + tn

    print("\n" + "=" * 56)
    print("  NETRA  —  Evaluation Results")
    print("=" * 56)
    print(f"  Total videos evaluated : {total}")
    print()
    print("  Confusion Matrix")
    print(f"    TP (drowsy, correct)  : {tp}")
    print(f"    FN (drowsy, missed)   : {fn}")
    print(f"    FP (awake, false alarm): {fp}")
    print(f"    TN (awake, correct)   : {tn}")
    print()

    accuracy    = (tp + tn) / total if total > 0 else 0.0
    precision   = tp / (tp + fp)   if (tp + fp)  > 0 else 0.0
    recall      = tp / (tp + fn)   if (tp + fn)  > 0 else 0.0
    specificity = tn / (tn + fp)   if (tn + fp)  > 0 else 0.0
    f1          = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else 0.0)

    print(f"  Accuracy    : {accuracy:.4f}  ({(accuracy*100):.1f}%)")
    print(f"  Precision   : {precision:.4f}")
    print(f"  Recall      : {recall:.4f}")
    print(f"  Specificity : {specificity:.4f}")
    print(f"  F1-Score    : {f1:.4f}")
    print("=" * 56)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    root_dir: str,
    max_videos: int | None = None,
    fixed_thresholds: bool = False,
    output_cm: str = "confusion_matrix.png",
) -> None:
    videos = collect_videos(root_dir)
    if not videos:
        print(f"No labeled .mp4 videos found under '{root_dir}'.")
        return

    if max_videos is not None and max_videos > 0 and len(videos) > max_videos:
        random.shuffle(videos)
        videos = videos[:max_videos]

    print(f"[Eval] Running on {len(videos)} videos using MediaPipe FaceLandmarker.\n")

    try:
        face_landmarker = create_face_landmarker()
    except ModelLoadError as exc:
        print(f"Model load error: {exc}")
        return

    tp = fn = fp = tn = 0

    for video_path, true_label in tqdm(videos, desc="Evaluating"):
        pred_label = predict_video_label(video_path, face_landmarker, fixed_thresholds)

        if true_label == 1 and pred_label == 1:
            tp += 1
        elif true_label == 1 and pred_label == 0:
            fn += 1
        elif true_label == 0 and pred_label == 1:
            fp += 1
        elif true_label == 0 and pred_label == 0:
            tn += 1

    face_landmarker.close()

    print_metrics(tp, fn, fp, tn)
    plot_confusion_matrix(tp, fn, fp, tn, output_path=output_cm)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Netra on labeled video datasets using MediaPipe."
    )
    parser.add_argument("--root-dir",   required=True,
                        help="Root directory of labeled datasets.")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Randomly cap the number of videos to evaluate.")
    parser.add_argument("--fixed-thresholds", action="store_true",
                        help="Use static global thresholds instead of adaptive baseline.")
    parser.add_argument("--output-cm",  type=str, default="confusion_matrix.png",
                        help="Output path for the confusion matrix PNG.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        root_dir=args.root_dir,
        max_videos=args.max_videos,
        fixed_thresholds=args.fixed_thresholds,
        output_cm=args.output_cm,
    )


if __name__ == "__main__":
    main()
