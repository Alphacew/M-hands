#!/usr/bin/env python3
"""
The "Jedi" 2D Physics Sandbox Application.

Transforms the webcam into an interactive telekinetic chamber where physical
rigid bodies, articulated ragdolls, and particle fluids react directly to natural
hand kinematics with zero perceptible input lag.
"""

import argparse
from pathlib import Path
import time
import cv2
import numpy as np

from mhands.pipeline.classifier import GestureClassifier
from mhands.pipeline.landmarker import HandLandmarkerEngine
from mhands.pipeline.threaded_stream import ThreadedCamera
from mhands.sandbox.physics_space import PhysicsSpace
from mhands.sandbox.telekinesis import TelekineticController
from mhands.sandbox.environments import SandboxEnvironments
from mhands.sandbox.visualizer_2d import SandboxVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Jedi 2D Physics Sandbox")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--model-path", type=str, default="models/invariant_svm.joblib", help="Classifier model path")
    parser.add_argument("--environment", type=str, default="castle", choices=["castle", "ragdoll", "fluid", "zerog", "bridge"])
    parser.add_argument("--ar", action="store_true", help="Start in Augmented Reality mode with camera feed")
    return parser.parse_args()


def main():
    args = parse_args()
    w, h = 1280, 720

    print("==================================================")
    print("           JEDI 2D PHYSICS SANDBOX                ")
    print("==================================================")

    # 1. Load Classifier
    model_path = Path(args.model_path)
    if not model_path.exists():
        from scripts.train_models import main as run_train
        run_train()
    classifier = GestureClassifier.load(model_path)

    # 2. Initialize Subsystems
    physics = PhysicsSpace(width=w, height=h)
    controller = TelekineticController(physics=physics, classifier=classifier)
    landmarker = HandLandmarkerEngine(max_num_hands=2)
    visualizer = SandboxVisualizer(width=w, height=h)

    # 3. Load Initial Environment
    env_map = {
        "castle": (SandboxEnvironments.load_jenga_castle, "Jenga_Castle"),
        "ragdoll": (SandboxEnvironments.load_ragdoll_arena, "Ragdoll_Arena"),
        "fluid": (SandboxEnvironments.load_fluid_vat, "Fluid_Vat"),
        "zerog": (SandboxEnvironments.load_zero_g_asteroids, "ZeroG_Asteroids"),
        "bridge": (SandboxEnvironments.load_seismic_bridge, "Seismic_Bridge"),
    }
    load_func, current_env_name = env_map[args.environment]
    load_func(physics)

    # 4. Video Stream
    print(f"[+] Starting video capture on /dev/video{args.camera}...")
    cam = ThreadedCamera(src=args.camera, width=w, height=h)
    cam.start()

    window_name = "Jedi 2D Physics Sandbox // M-Hands"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, w, h)

    print("\nControls:")
    print("  '1' - '5' : Switch Environment (Castle, Ragdolls, Fluids, Zero-G, Bridge)")
    print("  'r'       : Reset current environment")
    print("  'a'       : Toggle AR Camera View / Sci-Fi Holo-Chamber")
    print("  's'       : Trigger Slow-Motion Time Dilation (0.2x)")
    print("  'p'       : Capture Screenshot")
    print("  'q' / ESC : Exit application\n")

    ar_mode = args.ar
    last_t = time.perf_counter()
    fps = 60.0

    try:
        while True:
            t_now = time.perf_counter()
            dt = max(1e-4, t_now - last_t)
            last_t = t_now
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            # Process both hands
            hands = landmarker.process_multi(frame)

            # Update telekinesis for each detected hand
            for hand in hands:
                controller.update_hand(hand, screen_w=w, screen_h=h, timestamp=t_now)

            # Update animations & step physics simulation (120 Hz internal)
            controller.update_effects(dt)
            physics.step(dt)

            # Render
            rendered = visualizer.render(
                physics=physics,
                controller=controller,
                hands=hands,
                camera_frame=frame,
                ar_mode=ar_mode,
                environment_name=current_env_name,
                fps=fps,
            )

            cv2.imshow(window_name, rendered)
            key = cv2.waitKey(1) & 0xFF

            if key in [ord("q"), 27]:
                break
            elif key == ord("1"):
                load_func, current_env_name = env_map["castle"]
                load_func(physics)
            elif key == ord("2"):
                load_func, current_env_name = env_map["ragdoll"]
                load_func(physics)
            elif key == ord("3"):
                load_func, current_env_name = env_map["fluid"]
                load_func(physics)
            elif key == ord("4"):
                load_func, current_env_name = env_map["zerog"]
                load_func(physics)
            elif key == ord("5"):
                load_func, current_env_name = env_map["bridge"]
                load_func(physics)
            elif key == ord("r"):
                load_func(physics)
            elif key == ord("a"):
                ar_mode = not ar_mode
            elif key == ord("s"):
                physics.trigger_slow_motion(duration=2.5, speed_factor=0.2)
            elif key == ord("p"):
                snap_path = f"jedi_snapshot_{int(time.time())}.png"
                cv2.imwrite(snap_path, rendered)
                print(f"[*] Snapshot saved: {snap_path}")

    except KeyboardInterrupt:
        print("\n[!] User interrupted execution.")
    finally:
        cam.stop()
        landmarker.close()
        cv2.destroyAllWindows()
        print("[+] Jedi Sandbox closed cleanly.")


if __name__ == "__main__":
    main()
