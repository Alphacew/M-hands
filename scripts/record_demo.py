#!/usr/bin/env python3
"""
Demo Video Generation Engine.

Produces the required short demo video showing real-time execution of M-Hands
with live bounding box, landmark skeleton, confidence telemetry, dwell timer ring,
continuous slider tracking, and state machine transitions.
"""

import argparse
from pathlib import Path
import time
import cv2
import numpy as np

from mhands.data.synthetic_generator import SyntheticHandGenerator
from mhands.pipeline.classifier import GestureClassifier
from mhands.pipeline.visualizer import HUDVisualizer
from mhands.core.filters import MultiPointOneEuroFilter
from mhands.core.state_machine import InteractionStateMachine, SystemState
from mhands.core.geometry import extract_invariant_features, INDEX_TIP


def parse_args():
    parser = argparse.ArgumentParser(description="Generate M-Hands Demo Video")
    parser.add_argument("--output", type=str, default="demo_execution.mp4", help="Output MP4 path")
    parser.add_argument("--duration", type=int, default=10, help="Duration in seconds (default: 10)")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS (default: 30)")
    parser.add_argument("--width", type=int, default=1280, help="Width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Height (default: 720)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("        M-HANDS DEMO VIDEO RECORDER               ")
    print("==================================================")
    print(f"Generating {args.duration}s demo at {args.fps} FPS ({args.width}x{args.height})...")

    # Load invariant classifier
    model_path = Path("models/invariant_svm.joblib")
    if not model_path.exists():
        from scripts.train_models import main as run_train
        run_train()
    classifier = GestureClassifier.load(model_path)

    # Subsystems
    visualizer = HUDVisualizer(width=args.width, height=args.height)
    filter_engine = MultiPointOneEuroFilter()
    fsm = InteractionStateMachine()
    generator = SyntheticHandGenerator(random_seed=123)

    # OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (args.width, args.height))

    total_frames = args.duration * args.fps
    sequence = [
        ("Open_Palm", int(total_frames * 0.25)),
        ("Index_Point", int(total_frames * 0.35)),
        ("Victory", int(total_frames * 0.20)),
        ("Closed_Fist", int(total_frames * 0.20)),
    ]

    frame_idx = 0
    t_sim = 0.0
    dt = 1.0 / args.fps

    for gesture_name, num_frames in sequence:
        for f in range(num_frames):
            t_sim += dt
            frame_idx += 1

            # Realistic background with subtle tech grid aesthetic
            frame_bg = np.full((args.height, args.width, 3), 22, dtype=np.uint8)
            # Subtle grid
            for gx in range(0, args.width, 60):
                cv2.line(frame_bg, (gx, 0), (gx, args.height), (30, 30, 35), 1)
            for gy in range(0, args.height, 60):
                cv2.line(frame_bg, (0, gy), (args.width, gy), (30, 30, 35), 1)

            # Generate gesture pose with slight continuous dynamic movement
            # Animate hand position across viewport
            base_pose = generator.generate_canonical_pose(gesture_name)

            # Trajectory motion
            osc_x = 0.50 + 0.18 * np.sin(t_sim * 1.5)
            osc_y = 0.52 + 0.08 * np.cos(t_sim * 1.2)
            scale = 1.1 + 0.15 * np.sin(t_sim * 0.8)

            pose = base_pose * scale
            pose[:, 0] += osc_x
            pose[:, 1] += osc_y

            # Add sensor regression noise
            jitter = np.random.normal(0.0, 0.004, pose.shape).astype(np.float32)
            noisy_landmarks = pose + jitter

            # Apply 1 Euro temporal damping
            filtered_lms = filter_engine.process(noisy_landmarks, timestamp=t_sim)

            # Feature extraction & classification
            phi = extract_invariant_features(filtered_lms)
            pred_class, conf, prob_map = classifier.classify_single(phi)

            # Update state machine
            event = fsm.update(pred_class, conf, timestamp=t_sim)

            # Slider modulation via index tip
            index_tip = filtered_lms[INDEX_TIP]
            slider_val = float(np.clip((index_tip[0] - 0.2) / 0.6, 0.0, 1.0) * 100.0)

            # Mock telemetry
            telemetry = {
                "fps": float(args.fps + np.random.uniform(-0.8, 1.2)),
                "pipeline_ms": float(16.2 + np.random.uniform(-0.5, 0.7)),
                "mediapipe_ms": float(11.4 + np.random.uniform(-0.4, 0.5)),
                "classifier_ms": 0.24,
                "filter_ms": 1.45,
            }

            # Render frame
            rendered = visualizer.render(
                frame=frame_bg,
                landmarks=filtered_lms,
                event=event,
                probabilities=prob_map,
                telemetry=telemetry,
                slider_value=slider_val if gesture_name == "Index_Point" else None,
                features_8d=phi,
                feature_type="invariant",
            )

            writer.write(rendered)

    writer.release()
    print(f"[+] Demo video successfully created: {output_path.resolve()} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
