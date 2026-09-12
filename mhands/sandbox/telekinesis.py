"""
Telekinetic Force Controller & Kinematic Coupling Engine.

Translates human hand kinematics into physical force fields:
- Variable-tension elastic springs (pinch aperture modulation)
- Zero-lag momentum flinging
- Kinetic shockwave (Force Push)
- Gravitational singularity (Force Pull)
- Spatial stasis (Force Freeze)
- Plasma bisection laser cutter
- Dual-handed asymmetric mechanics (Deflector Shield, The Tear)
"""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pymunk
from pymunk import Vec2d

from mhands.core.geometry import (
    WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP,
    extract_invariant_features,
)
from mhands.pipeline.classifier import GestureClassifier
from mhands.pipeline.landmarker import HandData
from mhands.sandbox.physics_space import PhysicsSpace
from mhands.sandbox.polygon_cutter import slice_convex_polygon


@dataclass
class HandSpringGrip:
    body: pymunk.Body
    local_anchor: Vec2d
    stiffness: float
    damping: float
    tension: float = 0.0


@dataclass
class ActiveShockwave:
    center: Vec2d
    radius: float
    max_radius: float
    intensity: float
    duration: float
    elapsed: float = 0.0


@dataclass
class LaserBlade:
    p1: Vec2d
    p2: Vec2d
    sparks: List[Vec2d] = field(default_factory=list)


class TelekineticController:
    """
    Couples MediaPipe hand kinematics to Pymunk physical simulation.
    """

    def __init__(
        self,
        physics: PhysicsSpace,
        classifier: GestureClassifier,
        pinch_threshold: float = 0.40,
        grab_radius: float = 90.0,
    ):
        self.physics = physics
        self.classifier = classifier
        self.pinch_threshold = pinch_threshold
        self.grab_radius = grab_radius

        # Per-hand tracking state
        self.grips: Dict[str, Optional[HandSpringGrip]] = {"Right": None, "Left": None}
        self.hand_positions: Dict[str, Vec2d] = {"Right": Vec2d(0, 0), "Left": Vec2d(0, 0)}
        self.hand_velocities: Dict[str, Vec2d] = {"Right": Vec2d(0, 0), "Left": Vec2d(0, 0)}
        self.prev_positions: Dict[str, Vec2d] = {"Right": Vec2d(0, 0), "Left": Vec2d(0, 0)}
        self.prev_times: Dict[str, float] = {"Right": 0.0, "Left": 0.0}

        # Kinetic Push detection state
        self.prev_curl: Dict[str, float] = {"Right": 0.5, "Left": 0.5}

        # Stasis dwell tracker
        self.stasis_dwell: Dict[str, float] = {"Right": 0.0, "Left": 0.0}

        # Active visual effects
        self.active_shockwaves: List[ActiveShockwave] = []
        self.active_lasers: Dict[str, Optional[LaserBlade]] = {"Right": None, "Left": None}
        self.deflector_shields: Dict[str, Optional[Tuple[Vec2d, float]]] = {"Right": None, "Left": None}

    def update_hand(
        self,
        hand_data: HandData,
        screen_w: int,
        screen_h: int,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Processes single hand kinematics, executes telekinetic interactions, and updates physics.
        """
        if timestamp is None:
            timestamp = time.perf_counter()

        handedness = hand_data.handedness
        lms = hand_data.landmarks  # (21, 3)

        # 1. Map normalized landmarks to screen coordinates
        index_tip_px = Vec2d(lms[INDEX_TIP][0] * screen_w, lms[INDEX_TIP][1] * screen_h)
        thumb_tip_px = Vec2d(lms[THUMB_TIP][0] * screen_w, lms[THUMB_TIP][1] * screen_h)
        wrist_px = Vec2d(lms[WRIST][0] * screen_w, lms[WRIST][1] * screen_h)
        palm_center = (index_tip_px + thumb_tip_px + wrist_px) / 3.0

        # Estimate instantaneous hand velocity
        dt = max(1e-4, timestamp - self.prev_times[handedness])
        v_instant = (index_tip_px - self.prev_positions[handedness]) / dt
        # Exponential smoothing on velocity
        alpha_v = 0.4
        self.hand_velocities[handedness] = alpha_v * v_instant + (1.0 - alpha_v) * self.hand_velocities[handedness]
        self.hand_positions[handedness] = index_tip_px
        self.prev_positions[handedness] = index_tip_px
        self.prev_times[handedness] = timestamp

        # 2. Extract Invariant Features & Classify
        phi = extract_invariant_features(lms)
        pred_class, conf, _ = self.classifier.classify_single(phi)

        r_aperture = float(phi[7])
        r_spread = float(phi[6])
        mean_flexion = float(np.mean(phi[0:5]))

        pinch_center = (index_tip_px + thumb_tip_px) / 2.0
        is_pinching = r_aperture < self.pinch_threshold

        action_summary = {
            "gesture": pred_class,
            "confidence": conf,
            "is_pinching": is_pinching,
            "aperture": r_aperture,
            "grip_active": False,
            "grip_tension": 0.0,
            "stasis_active": False,
        }

        # -------------------------------------------------------------
        # Action 1: Elastic Force Grip & Momentum Flinging
        # -------------------------------------------------------------
        if is_pinching:
            grip = self.grips[handedness]
            if grip is None:
                # Query nearest dynamic body
                target_body = self.physics.query_nearest_body((pinch_center.x, pinch_center.y), self.grab_radius)
                if target_body:
                    local_anchor = target_body.world_to_local(pinch_center)
                    grip = HandSpringGrip(
                        body=target_body,
                        local_anchor=local_anchor,
                        stiffness=500.0,
                        damping=40.0,
                    )
                    self.grips[handedness] = grip

            if grip is not None:
                # Variable tension scaling with pinch tightness
                norm_pinch = np.clip(1.0 - (r_aperture / self.pinch_threshold), 0.0, 1.0)
                # Exponential stiffness scaling from 300 to 12000
                stiffness = 300.0 + (norm_pinch ** 2) * 11700.0
                damping = 2.0 * np.sqrt(stiffness * max(1.0, grip.body.mass)) * 0.85
                grip.stiffness = stiffness
                grip.damping = damping

                # Compute spring force
                world_anchor = grip.body.local_to_world(grip.local_anchor)
                delta = pinch_center - world_anchor
                dist = delta.length
                grip.tension = min(1.0, dist / 150.0)

                if dist > 1e-4:
                    spring_dir = delta.normalized()
                    force_mag = stiffness * dist
                    rel_vel = self.hand_velocities[handedness] - grip.body.velocity_at_world_point(world_anchor)
                    damping_force = rel_vel * damping
                    total_force = spring_dir * force_mag + damping_force

                    # Apply force to target body
                    grip.body.apply_force_at_world_point(total_force, world_anchor)

                action_summary["grip_active"] = True
                action_summary["grip_tension"] = grip.tension
        else:
            # Pinch released: sever spring and transfer momentum
            grip = self.grips[handedness]
            if grip is not None:
                # Momentum flinging: add hand velocity impulse
                fling_vel = self.hand_velocities[handedness]
                if fling_vel.length > 80.0:
                    grip.body.velocity += fling_vel * 0.7
                self.grips[handedness] = None

        # -------------------------------------------------------------
        # Action 2: Kinetic Shockwave ("Force Push")
        # -------------------------------------------------------------
        d_curl = self.prev_curl[handedness] - mean_flexion  # Positive when rapidly opening
        self.prev_curl[handedness] = mean_flexion

        if pred_class == "Open_Palm" and conf >= 0.85 and d_curl > 0.08:
            # Force Push impulse
            push_power = min(1500.0, d_curl * 8000.0 + self.hand_velocities[handedness].length * 1.5)
            bodies_near = self.physics.query_bodies_in_radius((palm_center.x, palm_center.y), 320.0)
            for b in bodies_near:
                diff = b.position - palm_center
                dist = max(10.0, diff.length)
                impulse = diff.normalized() * (push_power * b.mass * (320.0 / dist))
                b.apply_impulse_at_world_point(impulse, b.position)

            self.active_shockwaves.append(
                ActiveShockwave(
                    center=palm_center,
                    radius=20.0,
                    max_radius=320.0,
                    intensity=push_power,
                    duration=0.35,
                )
            )

        # -------------------------------------------------------------
        # Action 3: Gravitational Singularity ("Force Pull")
        # -------------------------------------------------------------
        if pred_class == "Closed_Fist" and conf >= 0.85:
            pull_radius = 450.0
            pull_bodies = self.physics.query_bodies_in_radius((palm_center.x, palm_center.y), pull_radius)
            for b in pull_bodies:
                diff = palm_center - b.position
                dist = max(20.0, diff.length)
                # Inward gravitational pull
                g_force = 120000.0 * b.mass / (dist ** 1.3)
                b.apply_force_at_world_point(diff.normalized() * g_force, b.position)

        # -------------------------------------------------------------
        # Action 4: Spatial Stasis ("Force Freeze")
        # -------------------------------------------------------------
        if pred_class == "Open_Palm" and conf >= 0.90 and self.hand_velocities[handedness].length < 40.0:
            self.stasis_dwell[handedness] += dt
            if self.stasis_dwell[handedness] >= 0.25:
                self.physics.set_stasis(True)
                action_summary["stasis_active"] = True
        else:
            self.stasis_dwell[handedness] = 0.0
            if self.physics.is_stasis:
                self.physics.set_stasis(False)

        # -------------------------------------------------------------
        # Action 5: Plasma Bisection Blade (Laser Cutter)
        # -------------------------------------------------------------
        if pred_class == "Victory" and conf >= 0.85:
            # Vector along extended fingers
            finger_dir = (index_tip_px - wrist_px).normalized()
            blade_start = palm_center
            blade_end = palm_center + finger_dir * 380.0
            blade_sparks = []

            # Check polygon intersection across dynamic bodies
            for body in list(self.physics.space.bodies):
                if body.body_type == pymunk.Body.DYNAMIC:
                    for shape in list(body.shapes):
                        if isinstance(shape, pymunk.Poly):
                            res = slice_convex_polygon(shape, blade_start, blade_end)
                            if res is not None:
                                (b_a, s_a), (b_b, s_b), cut_pts = res
                                # Replace old shape and body
                                self.physics.space.remove(shape, body)
                                self.physics.space.add(b_a, s_a, b_b, s_b)
                                blade_sparks.extend(cut_pts)
                                self.physics.trigger_slow_motion(duration=1.2, speed_factor=0.3)

            self.active_lasers[handedness] = LaserBlade(p1=blade_start, p2=blade_end, sparks=blade_sparks)
        else:
            self.active_lasers[handedness] = None

        # -------------------------------------------------------------
        # Action 6: Deflector Shield (Left Hand Open Palm)
        # -------------------------------------------------------------
        if handedness == "Left" and pred_class == "Open_Palm" and conf >= 0.80:
            shield_radius = 110.0
            self.deflector_shields["Left"] = (palm_center, shield_radius)
            # Deflect any bodies penetrating shield perimeter
            shield_bodies = self.physics.query_bodies_in_radius((palm_center.x, palm_center.y), shield_radius + 20.0)
            for b in shield_bodies:
                diff = b.position - palm_center
                if diff.length < shield_radius:
                    rebound = diff.normalized() * (400.0 * b.mass)
                    b.apply_impulse_at_world_point(rebound, b.position)
        else:
            self.deflector_shields["Left"] = None

        return action_summary

    def update_effects(self, dt: float) -> None:
        """Advances animations for shockwaves and plasma sparks."""
        remaining = []
        for sw in self.active_shockwaves:
            sw.elapsed += dt
            progress = sw.elapsed / sw.duration
            sw.radius = 20.0 + progress * (sw.max_radius - 20.0)
            if sw.elapsed < sw.duration:
                remaining.append(sw)
        self.active_shockwaves = remaining
