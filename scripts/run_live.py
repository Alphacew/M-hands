#!/usr/bin/env python3
"""
Live Real-Time Gesture Recognition Deployment.

Streams from local webcam (e.g. /dev/video0), performs CLAHE illumination normalization,
MediaPipe keypoint inference, invariant feature transformation, adaptive 1 Euro damping,
deterministic hysteresis FSM, and cyber-medical HUD visualization.
"""

import argparse
from pathlib import Path
import sys
import cv2

from mhands.pipeline.classifier import GestureClassifier
from mhands.pipeline.threaded_stream import DecoupledPipeline, ThreadedCamera


def parse_args():
    parser = argparse.ArgumentParser(description="M-Hands Real-Time Gesture Recognition Pipeline")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--model-path", type=str, default="models/invariant_svm.joblib", help="Path to trained model")
    parser.add_argument("--feature-type", type=str, default="invariant", choices=["invariant", "raw"], help="Feature representation")
    parser.add_argument("--tremor-mode", action="store_true", help="Enable high-damping tremor suppression profile")
    parser.add_argument("--no-clahe", action="store_true", help="Disable adaptive CLAHE illumination normalization")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"[!] Model not found at {model_path}. Training default models first...")
        from scripts.train_models import main as run_train
        run_train()

    print(f"[+] Loading classifier from: {model_path}")
    classifier = GestureClassifier.load(model_path)

    filter_mode = "tremor_suppression" if args.tremor_mode else "standard"
    pipeline = DecoupledPipeline(
        classifier=classifier,
        feature_type=args.feature_type,
        camera_id=args.camera,
        enable_clahe=not args.no_clahe,
        filter_mode=filter_mode,
    )

    print(f"[+] Initializing video capture on /dev/video{args.camera}...")
    cam = ThreadedCamera(src=args.camera)
    cam.start()

    window_name = "M-Hands // Production Real-Time Hand Gesture Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("\n=======================================================")
    print("           M-HANDS LIVE PIPELINE RUNNING               ")
    print("=======================================================")
    print("Controls:")
    print("  'q' / ESC : Exit application")
    print("  't'       : Toggle feature representation (invariant <-> raw)")
    print("  'm'       : Toggle tremor suppression mode")
    print("  's'       : Capture screenshot")
    print("=======================================================\n")

    current_feature_type = args.feature_type
    current_filter_mode = filter_mode

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            rendered_canvas, event, telemetry = pipeline.process_frame(frame)

            cv2.imshow(window_name, rendered_canvas)
            key = cv2.waitKey(1) & 0xFF

            if key in [ord("q"), 27]:
                break
            elif key == ord("t"):
                current_feature_type = "raw" if current_feature_type == "invariant" else "invariant"
                pipeline.feature_type = current_feature_type
                print(f"[*] Switched feature representation to: {current_feature_type}")
            elif key == ord("m"):
                current_filter_mode = "tremor_suppression" if current_filter_mode == "standard" else "standard"
                pipeline.filter = MultiPointOneEuroFilter(mode=current_filter_mode)
                print(f"[*] Switched filter profile to: {current_filter_mode}")
            elif key == ord("s"):
                snap_path = f"screenshot_{int(time.time())}.png"
                cv2.imwrite(snap_path, rendered_canvas)
                print(f"[*] Screenshot saved: {snap_path}")

    except KeyboardInterrupt:
        print("\n[!] User interrupted execution.")
    finally:
        cam.stop()
        pipeline.close()
        cv2.destroyAllWindows()
        print("[+] Pipeline shutdown cleanly.")


if __name__ == "__main__":
    main()
