"""
tune_thresholds.py
==================
Bayesian hyperparameter optimisation (via Optuna) for the Netra drowsiness
detection pipeline.

Instead of the old brute-force grid search, Optuna uses a Tree-structured
Parzen Estimator (TPE) — a Bayesian surrogate model — to intelligently
explore the hyperparameter space and converge on the threshold values that
maximise the F1-Score on your dataset.

Usage:
    python tune_thresholds.py --root-dir datasets --n-trials 100
"""

import argparse
import logging

import cv2
import optuna
from tqdm import tqdm

from evaluate_datasets import collect_videos
from netra_common import (
    AdaptiveEARBaseline,
    DrowsinessState,
    ModelLoadError,
    apply_adaptive_thresholds_to_state,
    compute_ear_from_result,
    create_face_landmarker,
    get_head_pose_from_result,
)

# Suppress verbose Optuna logs; we print our own summary
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _resize_frame(frame, width: int = 640):
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale = width / float(w)
    return cv2.resize(frame, (width, int(h * scale)))


def evaluate_with_params(
    videos,
    sleep_ratio: float,
    wake_offset: float,
    consec_frames_drowsy: int,
    consec_frames_nod: int,
) -> float:
    """
    Run the EAR/Pitch pipeline headless across all videos with the given
    hyperparameters. Returns the F1-Score (harmonic mean of precision & recall).
    """
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision

    try:
        face_landmarker = create_face_landmarker()
    except ModelLoadError as exc:
        print(f"[Optuna] Model load error: {exc}")
        return 0.0

    tp = fn = fp = tn = 0
    frame_ts_ms = 0

    for video_path, true_label in videos:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        state_machine = DrowsinessState(consec_frames_drowsy=consec_frames_drowsy)
        adaptive = AdaptiveEARBaseline(sleep_ratio=sleep_ratio, wake_ratio=sleep_ratio + wake_offset)
        last_state = "awake"
        entered_drowsy = False
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 3 != 0:
                frame_idx += 1
                frame_ts_ms += 33
                continue

            frame = _resize_frame(frame, width=320)
            h_img, w_img = frame.shape[:2]

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = face_landmarker.detect_for_video(mp_image, frame_ts_ms)
            face_detected = bool(result.face_landmarks)

            if face_detected:
                ear   = compute_ear_from_result(result, w_img, h_img)
                pitch = get_head_pose_from_result(result, w_img, h_img)
                adaptive.observe(ear, True, last_state == "awake")
            else:
                ear, pitch = 0.0, 0.0

            apply_adaptive_thresholds_to_state(adaptive, state_machine)
            calibrated = adaptive.baseline_open_ear() is not None

            if face_detected:
                state, _, _ = state_machine.update(ear, pitch)
            elif not calibrated:
                state, _, _ = state_machine.update(ear=1.0)
            else:
                state, _, _ = state_machine.update(ear=0.0)

            last_state = state
            if state == "drowsy":
                entered_drowsy = True
                break

            frame_idx  += 1
            frame_ts_ms += 33

        cap.release()

        pred_label = 1 if entered_drowsy else 0
        if true_label == 1 and pred_label == 1:
            tp += 1
        elif true_label == 1 and pred_label == 0:
            fn += 1
        elif true_label == 0 and pred_label == 1:
            fp += 1
        elif true_label == 0 and pred_label == 0:
            tn += 1

    face_landmarker.close()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return f1


def build_objective(videos, args):
    """Return an Optuna objective function closed over the dataset and args."""

    def objective(trial: optuna.Trial) -> float:
        sleep_ratio          = trial.suggest_float("sleep_ratio",          0.60, 0.82)
        wake_offset          = trial.suggest_float("wake_offset",          0.05, 0.20)
        consec_frames_drowsy = trial.suggest_int(  "consec_frames_drowsy", 3,    20)
        consec_frames_nod    = trial.suggest_int(  "consec_frames_nod",    5,    20)

        return evaluate_with_params(
            videos,
            sleep_ratio=sleep_ratio,
            wake_offset=wake_offset,
            consec_frames_drowsy=consec_frames_drowsy,
            consec_frames_nod=consec_frames_nod,
        )

    return objective


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter search for Netra EAR/Pitch thresholds."
    )
    parser.add_argument("--root-dir",  required=True,
                        help="Dataset root directory (e.g., datasets).")
    parser.add_argument("--n-trials",  type=int, default=50,
                        help="Number of Optuna trials (default 50).")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Cap on number of videos per trial (speeds up tuning).")
    args = parser.parse_args()

    videos = collect_videos(args.root_dir)
    if not videos:
        print(f"No labeled .mp4 videos found under '{args.root_dir}'.")
        return

    if args.max_videos and len(videos) > args.max_videos:
        import random
        random.shuffle(videos)
        videos = videos[:args.max_videos]

    print(f"[Optuna] Bayesian optimisation — {args.n_trials} trials on {len(videos)} videos.")
    print("[Optuna] Objective: maximise F1-Score over the labeled dataset.\n")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(build_objective(videos, args), n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_trial
    print("\n" + "=" * 60)
    print(f"  Best F1-Score : {best.value:.4f}")
    print(f"  Best params   :")
    for k, v in best.params.items():
        print(f"    {k:30s} = {v}")
    print("=" * 60)
    print("\nCopy these values into netra_common.py to apply the optimised thresholds.")


if __name__ == "__main__":
    main()
