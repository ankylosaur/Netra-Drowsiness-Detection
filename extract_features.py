"""
extract_features.py
===================
Extracts temporal features (EAR and Pitch) from the video datasets using MediaPipe.
Saves the sequences to `.npy` arrays for training the LSTM temporal classifier.

It creates an `extracted_features/` directory containing:
  - `features.npy`: A numpy array of shape (N_videos, max_frames, 2)
  - `labels.npy`: A numpy array of shape (N_videos, 1)

Usage:
    python extract_features.py --root-dir datasets --max-frames 30
"""

import argparse
import os
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

from evaluate_datasets import collect_videos
from netra_common import (
    ModelLoadError,
    compute_ear_from_result,
    create_face_landmarker,
    get_head_pose_from_result,
)


def _resize_frame(frame, width: int = 640):
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale = width / float(w)
    return cv2.resize(frame, (width, int(h * scale)))


def extract_video_features(
    video_path: str,
    face_landmarker: mp_vision.FaceLandmarker,
    start_ts_ms: int,
    max_frames: int = 30,
    stride: int = 3,
) -> Tuple[np.ndarray, int]:
    """
    Runs MediaPipe over a video and returns an array of shape (max_frames, 2)
    where each row is [EAR, Pitch], and the final timestamp.
    If the video has fewer frames, it is padded with the last observed values.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros((max_frames, 2)), start_ts_ms

    features = []
    frame_idx = 0
    frame_ts_ms = start_ts_ms

    while len(features) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Subsample frames
        if frame_idx % stride != 0:
            frame_idx += 1
            frame_ts_ms += 33
            continue

        frame = _resize_frame(frame, width=320)  # Lower res for speed
        h_img, w_img = frame.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = face_landmarker.detect_for_video(mp_image, frame_ts_ms)

        face_detected = bool(result.face_landmarks)

        if face_detected:
            ear   = compute_ear_from_result(result, w_img, h_img)
            pitch = get_head_pose_from_result(result, w_img, h_img)
        else:
            ear, pitch = 0.0, 0.0

        features.append([ear, pitch])
        frame_idx += 1
        frame_ts_ms += 33

    cap.release()

    # Convert to numpy and pad if necessary
    features = np.array(features, dtype=np.float32)
    if len(features) == 0:
        features = np.zeros((max_frames, 2), dtype=np.float32)
    elif len(features) < max_frames:
        # Pad with the last observed feature
        pad_len = max_frames - len(features)
        last_val = features[-1:]
        padding = np.repeat(last_val, pad_len, axis=0)
        features = np.vstack([features, padding])

    return features, frame_ts_ms


def extract_features(
    root_dir: str,
    output_dir: str,
    max_frames: int = 30,
    stride: int = 3,
    max_videos: int | None = None,
) -> None:
    videos = collect_videos(root_dir)
    if not videos:
        print(f"No labeled .mp4 videos found under '{root_dir}'.")
        return

    if max_videos is not None and max_videos > 0 and len(videos) > max_videos:
        import random
        random.shuffle(videos)
        videos = videos[:max_videos]

    print(f"Discovered {len(videos)} videos. Beginning feature extraction...")
    
    try:
        face_landmarker = create_face_landmarker()
    except ModelLoadError as exc:
        print(f"Model load error: {exc}")
        return

    all_features = []
    all_labels = []
    global_ts_ms = 0

    for video_path, label in tqdm(videos, desc="Extracting features"):
        feats, global_ts_ms = extract_video_features(
            video_path,
            face_landmarker,
            start_ts_ms=global_ts_ms,
            max_frames=max_frames,
            stride=stride,
        )
        all_features.append(feats)
        all_labels.append(label)

    face_landmarker.close()

    # Convert to standard numpy tensor shapes
    X = np.array(all_features, dtype=np.float32)  # Shape: (N, max_frames, 2)
    y = np.array(all_labels, dtype=np.int64)      # Shape: (N,)

    os.makedirs(output_dir, exist_ok=True)
    features_path = os.path.join(output_dir, "features.npy")
    labels_path = os.path.join(output_dir, "labels.npy")

    np.save(features_path, X)
    np.save(labels_path, y)

    print("\n" + "=" * 50)
    print("Feature Extraction Complete!")
    print("=" * 50)
    print(f"Total videos processed : {len(X)}")
    print(f"Extracted shape X      : {X.shape} (N_videos, Timesteps, Features)")
    print(f"Extracted shape Y      : {y.shape} (N_videos,)")
    print(f"Saved to               : {features_path}")
    print(f"                         {labels_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract temporal sequences for LSTM training.")
    parser.add_argument("--root-dir", required=True, help="Root directory of labeled dataset.")
    parser.add_argument("--out-dir", default="extracted_features", help="Output directory for .npy arrays.")
    parser.add_argument("--max-frames", type=int, default=30, help="Max timesteps per video.")
    parser.add_argument("--stride", type=int, default=3, help="Subsample frame stride.")
    parser.add_argument("--max-videos", type=int, default=None, help="Cap randomly for testing.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_features(
        root_dir=args.root_dir,
        output_dir=args.out_dir,
        max_frames=args.max_frames,
        stride=args.stride,
        max_videos=args.max_videos,
    )
