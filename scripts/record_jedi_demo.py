#!/usr/bin/env python3
"""
Jedi Sandbox Demo Video Generator.

Produces a cinematic 12-second high-definition 720p 60 FPS video demonstration
showcasing telekinetic spring gripping, momentum flinging, Force Push shockwave,
Force Pull singularity, Victory plasma slicing, and slow-motion destruction.
"""

import argparse
from pathlib import Path
import time
import cv2
import numpy as np
import pymunk
from pymunk import Vec2d

from mhands.pipeline.classifier import GestureClassifier
from mhands.pipeline.landmarker import HandData
from mhands.data.synthetic_generator import SyntheticHandGenerator
from mhands.sandbox.physics_space import PhysicsSpace
from mhands.sandbox.telekinesis import TelekineticController
from mhands.sandbox.environments import SandboxEnvironments
from mhands.sandbox.visualizer_2d import SandboxVisualizer
from mhands.sandbox.polygon_cutter import slice_convex_polygon
from mhands.core.geometry import INDEX_TIP, THUMB_TIP, WRIST


def parse_args():
    parser = argparse.ArgumentParser(description="Record Jedi Sandbox Demo Video")
    parser.add_argument("--output", type=str, default="jedi_sandbox_demo.mp4", help="Output video path")
    parser.add_argument("--duration", type=int, default=15, help="Duration in seconds (default: 15)")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS (default: 30)")
    return parser.parse_args()


def main():
    args = parse_args()
    w, h = 1280, 720
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print("      JEDI SANDBOX DEMO VIDEO GENERATOR           ")
    print("==================================================")
    print(f"Rendering {args.duration}s cinematic demo at {args.fps} FPS ({w}x{h})...")

    # Load classifier
    model_path = Path("models/invariant_svm.joblib")
    if not model_path.exists():
        from scripts.train_models import main as run_train
        run_train()
    classifier = GestureClassifier.load(model_path)

    # Initialize subsystems
    physics = PhysicsSpace(width=w, height=h)
    controller = TelekineticController(physics=physics, classifier=classifier)
    visualizer = SandboxVisualizer(width=w, height=h)
    generator = SyntheticHandGenerator(random_seed=777)

    # Load Jenga Castle environment
    SandboxEnvironments.load_jenga_castle(physics)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (w, h))

    total_frames = args.duration * args.fps
    dt = 1.0 / args.fps
    t_sim = 0.0

    # Camera background frame (subtle laboratory room gradient to showcase transparent AR mode)
    bg_frame = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        val = int(22 + 25 * (1.0 - y / h))
        bg_frame[y, :] = (val, val + 4, val + 10)
    bg_frame[::40, ::40] = (70, 75, 85)

    # Locate arch keystone or a prominent block to grab
    keystone_body = None
    for b in physics.space.bodies:
        if b.body_type == pymunk.Body.DYNAMIC and 400 < b.position.x < 500 and 450 < b.position.y < 600:
            keystone_body = b
            break
    if keystone_body is None:
        keystone_body = [b for b in physics.space.bodies if b.body_type == pymunk.Body.DYNAMIC][0]

    def translate_hand(lms_in: np.ndarray, target_x: float, target_y: float) -> np.ndarray:
        lms = lms_in.copy()
        cur_center = np.mean(lms[[0, 5, 9, 13, 17], :2], axis=0)
        lms[:, 0] += (target_x - cur_center[0])
        lms[:, 1] += (target_y - cur_center[1])
        return lms

    for frame_idx in range(total_frames):
        t_sim += dt
        phase = t_sim / args.duration  # [0.0, 1.0]

        # Stage 1: (0.00 - 0.20) -> Telekinetic Pinch & Elastic Spring Wrecking Ball
        if phase < 0.20:
            progress = phase / 0.20
            target_pt = keystone_body.position
            hand_x = 0.35 + 0.15 * np.sin(progress * np.pi)
            hand_y = 0.65 - 0.35 * progress

            raw_lms = generator.generate_canonical_pose("Open_Palm")
            lms = translate_hand(raw_lms, hand_x, hand_y)
            # Pinch: move thumb tip directly adjacent to index tip
            lms[THUMB_TIP, :2] = lms[INDEX_TIP, :2] + np.array([0.008, 0.008])

            hand = HandData(landmarks=lms, handedness="Right", confidence=0.98)
            controller.update_hand(hand, screen_w=w, screen_h=h, timestamp=t_sim)
            env_name = "Telekinetic Spring Grip & Tension Control"

        # Stage 2: (0.20 - 0.40) -> Momentum Fling & Explosive Force Push
        elif phase < 0.40:
            progress = (phase - 0.20) / 0.20
            # Hand whips across to x = 0.75 and snaps open
            hand_x = 0.50 + 0.30 * progress
            hand_y = 0.35 + 0.10 * np.sin(progress * np.pi)

            raw_lms = generator.generate_canonical_pose("Open_Palm")
            lms = translate_hand(raw_lms, hand_x, hand_y)

            hand = HandData(landmarks=lms, handedness="Right", confidence=0.99)
            controller.update_hand(hand, screen_w=w, screen_h=h, timestamp=t_sim)

            # Trigger a massive force push halfway through
            if 0.28 < phase < 0.30:
                from mhands.sandbox.telekinesis import ActiveShockwave
                palm_pt = Vec2d(hand_x * w, hand_y * h)
                for b in physics.space.bodies:
                    if b.body_type == pymunk.Body.DYNAMIC:
                        diff = b.position - palm_pt
                        b.apply_impulse_at_world_point(diff.normalized() * (900.0 * b.mass), b.position)
                controller.active_shockwaves.append(
                    ActiveShockwave(center=palm_pt, radius=20, max_radius=420, intensity=1200, duration=0.45)
                )
                physics.trigger_slow_motion(duration=1.8, speed_factor=0.25)

            env_name = "Kinetic Momentum Fling & Force Push Blast"

        # Stage 3: (0.40 - 0.60) -> Gravitational Singularity (Closed Fist)
        elif phase < 0.60:
            progress = (phase - 0.40) / 0.20
            hand_x = 0.50 + 0.08 * np.sin(progress * 2 * np.pi)
            hand_y = 0.45

            raw_lms = generator.generate_canonical_pose("Closed_Fist")
            lms = translate_hand(raw_lms, hand_x, hand_y)

            hand = HandData(landmarks=lms, handedness="Right", confidence=0.98)
            controller.update_hand(hand, screen_w=w, screen_h=h, timestamp=t_sim)
            env_name = "Gravitational Singularity Vortex (Closed Fist Gravity)"

        # Stage 4: (0.60 - 0.80) -> Thumbs-Up On-Screen Guide
        elif phase < 0.80:
            progress = (phase - 0.60) / 0.20
            hand_x = 0.16
            hand_y = 0.55

            raw_fist = generator.generate_canonical_pose("Closed_Fist")
            lms = translate_hand(raw_fist, hand_x, hand_y)
            # Extend thumb upward for Thumbs-Up
            lms[THUMB_TIP] = lms[WRIST] + np.array([-0.03, -0.18, 0.0])
            lms[2] = lms[WRIST] + np.array([-0.02, -0.08, 0.0])
            lms[3] = lms[WRIST] + np.array([-0.025, -0.13, 0.0])

            hand = HandData(landmarks=lms, handedness="Right", confidence=0.99)
            controller.update_hand(hand, screen_w=w, screen_h=h, timestamp=t_sim)
            env_name = "On-Screen Holographic Manual (Thumbs-Up Pose)"

        # Stage 5: (0.80 - 1.00) -> Victory Plasma Bisection Laser Blade
        else:
            progress = (phase - 0.80) / 0.20
            hand_x = 0.30 + 0.45 * progress
            hand_y = 0.40 + 0.15 * np.cos(progress * np.pi)

            raw_lms = generator.generate_canonical_pose("Victory")
            lms = translate_hand(raw_lms, hand_x, hand_y)

            hand = HandData(landmarks=lms, handedness="Right", confidence=0.99)
            controller.update_hand(hand, screen_w=w, screen_h=h, timestamp=t_sim)
            env_name = "Plasma Bisection Blade & Convex Polygon Slicing"

        # Step physics and visual effects
        controller.update_effects(dt)
        physics.step(dt)

        # Render frame
        rendered = visualizer.render(
            physics=physics,
            controller=controller,
            hands=[hand],
            camera_frame=bg_frame,
            ar_mode=True,
            environment_name=env_name,
            fps=float(args.fps),
        )

        writer.write(rendered)

    writer.release()
    print(f"[+] Demo video written to: {output_path.resolve()} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
