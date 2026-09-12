#!/usr/bin/env python3
"""
Jedi Sandbox Demo Video Generator.

Produces a cinematic 21-second high-definition 720p 30 FPS video demonstration
showcasing ALL telekinetic features, starting with the On-Screen Holographic Manual:
1. Holographic Manual (Thumbs-Up Summon & Immediate Zero-Lag Release)
2. Telekinetic Spring Grip & Variable Tension (Natural Human Pinch)
3. Kinetic Momentum Fling & Explosive Force Push Shockwave
4. Closed Fist Universal Gravitational Singularity (All-Block Accretion Disk)
5. Dual-Hand Asymmetric Telekinesis (Deflector Shield & Spatial Stasis)
6. Victory Sign Plasma Bisection Blade & Convex Polygon Slicing
"""

import argparse
from pathlib import Path
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
from mhands.core.geometry import INDEX_TIP, THUMB_TIP, WRIST


def parse_args():
    parser = argparse.ArgumentParser(description="Record Jedi Sandbox Demo Video")
    parser.add_argument("--output", type=str, default="jedi_sandbox_demo.mp4", help="Output video path")
    parser.add_argument("--duration", type=int, default=21, help="Duration in seconds (default: 21)")
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

        hands_to_render = []

        # =========================================================================
        # Stage 1: (0.0s - 3.5s) -> Holographic Guide (Thumbs-Up Summon & Immediate Release)
        # =========================================================================
        if t_sim < 3.5:
            env_name = "1. Holographic Manual (Thumbs-Up Summon & Immediate Release)"
            hand_x, hand_y = 0.16, 0.52

            if t_sim < 2.6:
                # Deliberate Thumbs-Up pose
                raw_fist = generator.generate_canonical_pose("Closed_Fist")
                lms = translate_hand(raw_fist, hand_x, hand_y)
                lms[THUMB_TIP] = lms[WRIST] + np.array([-0.03, -0.18, 0.0])
                lms[2] = lms[WRIST] + np.array([-0.02, -0.08, 0.0])
                lms[3] = lms[WRIST] + np.array([-0.025, -0.13, 0.0])
                hand_r = HandData(landmarks=lms, handedness="Right", confidence=0.99)
            else:
                # Immediate release: open hand relaxes -> guide instantly vanishes!
                raw_open = generator.generate_canonical_pose("Open_Palm")
                lms = translate_hand(raw_open, hand_x, hand_y)
                hand_r = HandData(landmarks=lms, handedness="Right", confidence=0.99)

            controller.update_hands([hand_r], screen_w=w, screen_h=h, timestamp=t_sim)
            hands_to_render = [hand_r]

        # =========================================================================
        # Stage 2: (3.5s - 7.0s) -> Natural Telekinetic Pinch Grip & Variable Spring
        # =========================================================================
        elif t_sim < 7.0:
            env_name = "2. Telekinetic Spring Grip & Variable Tension (Pinch)"
            progress = (t_sim - 3.5) / 3.5  # [0, 1]

            # Move towards block then swing it up like a wrecking ball
            hand_x = 0.35 + 0.16 * np.sin(progress * np.pi)
            hand_y = 0.65 - 0.38 * progress

            raw_point = generator.generate_canonical_pose("Index_Point")
            lms = translate_hand(raw_point, hand_x, hand_y)
            # Natural pinch: thumb tip meets index tip with curled middle/ring/pinky
            lms[THUMB_TIP, :2] = lms[INDEX_TIP, :2] + np.array([0.008, 0.008])

            hand_r = HandData(landmarks=lms, handedness="Right", confidence=0.98)
            controller.update_hands([hand_r], screen_w=w, screen_h=h, timestamp=t_sim)
            hands_to_render = [hand_r]

        # =========================================================================
        # Stage 3: (7.0s - 10.5s) -> Kinetic Momentum Fling & Force Push Blast
        # =========================================================================
        elif t_sim < 10.5:
            env_name = "3. Kinetic Momentum Fling & Explosive Force Push"
            progress = (t_sim - 7.0) / 3.5

            hand_x = 0.50 + 0.30 * progress
            hand_y = 0.35 + 0.10 * np.sin(progress * np.pi)

            raw_open = generator.generate_canonical_pose("Open_Palm")
            lms = translate_hand(raw_open, hand_x, hand_y)
            hand_r = HandData(landmarks=lms, handedness="Right", confidence=0.99)
            controller.update_hands([hand_r], screen_w=w, screen_h=h, timestamp=t_sim)
            hands_to_render = [hand_r]

            # Trigger explosive shockwave at t = 8.3s
            if 8.25 <= t_sim <= 8.32 and len(controller.active_shockwaves) == 0:
                from mhands.sandbox.telekinesis import ActiveShockwave
                palm_pt = Vec2d(hand_x * w, hand_y * h)
                for b in physics.space.bodies:
                    if b.body_type == pymunk.Body.DYNAMIC:
                        diff = b.position - palm_pt
                        b.apply_impulse_at_world_point(diff.normalized() * (1000.0 * b.mass), b.position)
                controller.active_shockwaves.append(
                    ActiveShockwave(center=palm_pt, radius=20, max_radius=440, intensity=1400, duration=0.45)
                )
                physics.trigger_slow_motion(duration=1.8, speed_factor=0.25)

        # =========================================================================
        # Stage 4: (10.5s - 14.5s) -> Closed Fist Gravitational Singularity
        # =========================================================================
        elif t_sim < 14.5:
            env_name = "4. Closed Fist Universal Gravitational Singularity"
            progress = (t_sim - 10.5) / 4.0

            hand_x = 0.50 + 0.08 * np.sin(progress * 2 * np.pi)
            hand_y = 0.42

            raw_fist = generator.generate_canonical_pose("Closed_Fist")
            lms = translate_hand(raw_fist, hand_x, hand_y)
            hand_r = HandData(landmarks=lms, handedness="Right", confidence=0.98)
            controller.update_hands([hand_r], screen_w=w, screen_h=h, timestamp=t_sim)
            hands_to_render = [hand_r]

        # =========================================================================
        # Stage 5: (14.5s - 17.5s) -> Dual-Hand Asymmetry: Deflector Shield & Spatial Stasis
        # =========================================================================
        elif t_sim < 17.5:
            env_name = "5. Dual-Hand Asymmetry: Deflector Shield & Spatial Stasis"
            # Right hand holds flat stationary palm (Spatial Stasis)
            rx, ry = 0.65, 0.45
            raw_r = generator.generate_canonical_pose("Open_Palm")
            lms_r = translate_hand(raw_r, rx, ry)
            hand_r = HandData(landmarks=lms_r, handedness="Right", confidence=0.99)

            # Left hand projects hexagonal Deflector Shield barrier
            lx, ly = 0.28, 0.48
            raw_l = generator.generate_canonical_pose("Open_Palm")
            lms_l = translate_hand(raw_l, lx, ly)
            hand_l = HandData(landmarks=lms_l, handedness="Left", confidence=0.99)

            controller.update_hands([hand_r, hand_l], screen_w=w, screen_h=h, timestamp=t_sim)
            hands_to_render = [hand_r, hand_l]

        # =========================================================================
        # Stage 6: (17.5s - 21.0s) -> Victory Plasma Bisection Laser Blade
        # =========================================================================
        else:
            env_name = "6. Victory Sign Plasma Bisection Blade & Slicing"
            progress = (t_sim - 17.5) / 3.5

            # Unfreeze stasis if still held
            if physics.is_stasis:
                physics.set_stasis(False)

            hand_x = 0.28 + 0.48 * progress
            hand_y = 0.38 + 0.14 * np.cos(progress * np.pi)

            raw_vic = generator.generate_canonical_pose("Victory")
            lms = translate_hand(raw_vic, hand_x, hand_y)
            hand_r = HandData(landmarks=lms, handedness="Right", confidence=0.99)
            controller.update_hands([hand_r], screen_w=w, screen_h=h, timestamp=t_sim)
            hands_to_render = [hand_r]

        # Step physics and visual effects
        controller.update_effects(dt)
        physics.step(dt)

        # Render frame
        rendered = visualizer.render(
            physics=physics,
            controller=controller,
            hands=hands_to_render,
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
