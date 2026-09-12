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
    WRIST, THUMB_MCP, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP,
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
        pinch_threshold: float = 0.45,
        pinch_release_threshold: float = 0.58,
        grab_radius: float = 160.0,
        guide_dwell_threshold: float = 0.25,
    ):
        self.physics = physics
        self.classifier = classifier
        self.pinch_threshold = pinch_threshold
        self.pinch_release_threshold = pinch_release_threshold
        self.grab_radius = grab_radius
        self.guide_dwell_threshold = guide_dwell_threshold

        # Per-hand tracking state
        self.grips: Dict[str, Optional[HandSpringGrip]] = {"Right": None, "Left": None}
        self.is_pinching: Dict[str, bool] = {"Right": False, "Left": False}
        self.apertures: Dict[str, float] = {"Right": 1.0, "Left": 1.0}
        self.hand_positions: Dict[str, Vec2d] = {"Right": Vec2d(0, 0), "Left": Vec2d(0, 0)}
        self.hand_velocities: Dict[str, Vec2d] = {"Right": Vec2d(0, 0), "Left": Vec2d(0, 0)}
        self.prev_positions: Dict[str, Vec2d] = {"Right": Vec2d(0, 0), "Left": Vec2d(0, 0)}
        self.prev_times: Dict[str, float] = {"Right": 0.0, "Left": 0.0}

        # Kinetic Push detection state
        self.prev_curl: Dict[str, float] = {"Right": 0.5, "Left": 0.5}

        # Stasis dwell tracker
        self.stasis_dwell: Dict[str, float] = {"Right": 0.0, "Left": 0.0}

        # Laser blade slice rate limiter & cooldown
        self.last_slice_time: float = 0.0
        self.slice_cooldown: float = 0.35

        # Thumbs-Up On-Screen Guide state & dwell tracker
        self.guide_active: Dict[str, bool] = {"Right": False, "Left": False}
        self.guide_dwell: Dict[str, float] = {"Right": 0.0, "Left": 0.0}
        self.manual_guide: bool = False

        # Gravitational Singularity state
        self.singularity_active: Dict[str, bool] = {"Right": False, "Left": False}
        self.vortex_anim_time: float = 0.0

        # Active visual effects
        self.active_shockwaves: List[ActiveShockwave] = []
        self.active_lasers: Dict[str, Optional[LaserBlade]] = {"Right": None, "Left": None}
        self.deflector_shields: Dict[str, Optional[Tuple[Vec2d, float]]] = {"Right": None, "Left": None}

    @property
    def is_guide_active(self) -> bool:
        """Returns True if any hand has sustained Thumbs-Up or manual override is active."""
        return self.manual_guide or any(self.guide_active.values())

    @property
    def is_singularity_active(self) -> bool:
        """Returns True if any hand has active Gravitational Singularity."""
        return any(self.singularity_active.values())

    def reset_hand_state(self, handedness: str) -> None:
        """Immediately resets telekinetic states for a hand that left the sensor field."""
        self.guide_active[handedness] = False
        self.guide_dwell[handedness] = 0.0
        self.singularity_active[handedness] = False
        self.is_pinching[handedness] = False
        self.active_lasers[handedness] = None
        self.deflector_shields[handedness] = None
        if self.grips[handedness] is not None:
            self.grips[handedness] = None

    def update_hands(
        self,
        hands: List[HandData],
        screen_w: int,
        screen_h: int,
        timestamp: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Processes all detected hands, updating kinematics and immediately resetting absent hands.
        """
        seen_handedness = {h.handedness for h in hands}
        for h_name in ["Right", "Left"]:
            if h_name not in seen_handedness:
                self.reset_hand_state(h_name)

        summaries = []
        for h in hands:
            summaries.append(self.update_hand(h, screen_w, screen_h, timestamp))
        return summaries

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

        # Skeletal scale metric & distance ratios
        d_ref = float(np.linalg.norm(lms[MIDDLE_MCP] - lms[WRIST]))
        if d_ref < 1e-5:
            d_ref = 1.0
        idx_dist = float(np.linalg.norm(lms[INDEX_TIP] - lms[WRIST]) / d_ref)
        th_dy = float((lms[THUMB_TIP][1] - lms[THUMB_MCP][1]) / d_ref)

        pinch_center = (index_tip_px + thumb_tip_px) / 2.0

        # Check Thumbs-Up gesture: 4 fingers curled into palm, thumb extended upward
        # Stricter criteria: thumb extended, pointing distinctly UP, higher than knuckles, 4 digits curled
        is_thumbs_up = (
            (phi[0] < 0.70)
            and (th_dy < -0.25)
            and (lms[THUMB_TIP][1] < lms[INDEX_MCP][1] - 0.02)
            and (np.mean(phi[1:5]) > 1.20)
            and all(phi[i] > 0.90 for i in range(1, 5))
            and (r_aperture > 0.40)
        )

        # Check Closed Fist: all fingers curled into tight fist, thumb tucked
        is_fist = (
            (not is_thumbs_up)
            and (
                (pred_class == "Closed_Fist" and conf >= 0.70 and idx_dist < 1.10)
                or (np.mean(phi[1:5]) > 1.30 and phi[0] > 1.15 and idx_dist < 1.10)
            )
        )

        # True pinch: thumb & index fingertips touching or adjacent, decoupled from fist via reach & hysteresis
        is_gripping = (self.grips[handedness] is not None)
        active_aperture_thresh = self.pinch_release_threshold if is_gripping else self.pinch_threshold
        is_pinching = (
            (not is_thumbs_up)
            and (not is_fist)
            and (r_aperture < active_aperture_thresh)
            and (idx_dist >= 1.05 or phi[0] < 1.15 or pred_class != "Closed_Fist")
        )

        self.is_pinching[handedness] = is_pinching
        self.apertures[handedness] = r_aperture

        action_summary = {
            "gesture": "Thumbs_Up" if is_thumbs_up else pred_class,
            "confidence": conf,
            "is_pinching": is_pinching,
            "is_thumbs_up": is_thumbs_up,
            "is_fist": is_fist,
            "aperture": r_aperture,
            "grip_active": False,
            "grip_tension": 0.0,
            "singularity_active": False,
            "guide_active": False,
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

                if dist > 500.0:
                    # Sever overstretched grip to prevent catapulting
                    self.grips[handedness] = None
                else:
                    action_summary["grip_active"] = True
                    action_summary["grip_tension"] = grip.tension

                    if dist > 1e-4:
                        spring_dir = delta.normalized()
                        force_mag = min(18000.0, stiffness * dist)
                        rel_vel = self.hand_velocities[handedness] - grip.body.velocity_at_world_point(world_anchor)
                        damping_force = rel_vel * damping
                        total_force = spring_dir * force_mag + damping_force
                        if total_force.length > 25000.0:
                            total_force = total_force.normalized() * 25000.0

                        # Apply force to target body
                        grip.body.apply_force_at_world_point(total_force, world_anchor)

                        # Cap peak velocity to prevent tunneling or infinite coordinates
                        if grip.body.velocity.length > 2500.0:
                            grip.body.velocity = grip.body.velocity.normalized() * 2500.0
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

        if (not is_pinching) and pred_class == "Open_Palm" and conf >= 0.85 and d_curl > 0.08:
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
        # Action 3: Gravitational Singularity (Attract ALL blocks to fist)
        # -------------------------------------------------------------
        if is_fist:
            self.singularity_active[handedness] = True
            action_summary["singularity_active"] = True

            # Attract ALL dynamic bodies across the entire chamber toward the closed fist
            for b in self.physics.space.bodies:
                if b.body_type == pymunk.Body.DYNAMIC:
                    diff = palm_center - b.position
                    dist = max(25.0, diff.length)
                    dir_vec = diff.normalized()

                    # Gravitational acceleration: strong pull from anywhere on screen
                    g_acc = min(2400.0, 850000.0 / (dist + 100.0))
                    f_pull = dir_vec * (g_acc * b.mass)

                    # Orbital capture and damping within 160px of fist
                    if dist < 160.0:
                        radial_vel = b.velocity.dot(dir_vec) * dir_vec
                        tangent_dir = Vec2d(-dir_vec.y, dir_vec.x)
                        f_pull += tangent_dir * (260.0 * b.mass) - radial_vel * (9.0 * b.mass)

                    b.apply_force_at_world_point(f_pull, b.position)
        else:
            self.singularity_active[handedness] = False

        # -------------------------------------------------------------
        # Action 4: Spatial Stasis ("Force Freeze")
        # -------------------------------------------------------------
        if (not is_pinching) and (not is_thumbs_up) and pred_class == "Open_Palm" and conf >= 0.90 and self.hand_velocities[handedness].length < 40.0:
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
        # Strictly ensure not pinching, fingers extended, ring/pinky curled
        is_victory = (
            (not is_pinching)
            and (r_aperture > 0.50)
            and (pred_class == "Victory")
            and (conf >= 0.85)
            and (phi[1] < 0.65)
            and (phi[2] < 0.65)
            and (phi[3] > 0.60)
            and (phi[4] > 0.60)
        )

        if is_victory:
            # Vector along extended fingers
            finger_dir = (index_tip_px - wrist_px).normalized()
            blade_start = palm_center
            blade_end = palm_center + finger_dir * 380.0
            blade_sparks = []

            # Cooldown check: only slice if cooldown has elapsed
            if (timestamp - self.last_slice_time) >= self.slice_cooldown:
                for body in list(self.physics.space.bodies):
                    if body.body_type == pymunk.Body.DYNAMIC:
                        # CRITICAL CHIPMUNK SAFETY: Never slice a body attached to constraints
                        if len(body.constraints) > 0:
                            continue
                        # Never slice a body currently gripped by any hand
                        if any(g is not None and g.body == body for g in self.grips.values()):
                            continue

                        cut_done = False
                        for shape in list(body.shapes):
                            if isinstance(shape, pymunk.Poly):
                                res = slice_convex_polygon(shape, blade_start, blade_end)
                                if res is not None:
                                    (b_a, s_a), (b_b, s_b), cut_pts = res
                                    try:
                                        self.physics.space.remove(shape, body)
                                        self.physics.space.add(b_a, s_a, b_b, s_b)
                                        blade_sparks.extend(cut_pts)
                                        self.physics.trigger_slow_motion(duration=1.2, speed_factor=0.3)
                                        self.last_slice_time = timestamp
                                        cut_done = True
                                        break
                                    except Exception:
                                        pass
                        if cut_done:
                            break  # Slice only 1 body per trigger

            self.active_lasers[handedness] = LaserBlade(p1=blade_start, p2=blade_end, sparks=blade_sparks)
        else:
            self.active_lasers[handedness] = None

        # -------------------------------------------------------------
        # Action 6: Deflector Shield (Left Hand Open Palm)
        # -------------------------------------------------------------
        if (not is_pinching) and handedness == "Left" and pred_class == "Open_Palm" and conf >= 0.80:
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

        # -------------------------------------------------------------
        # Action 7: On-Screen Holocron Guide (Thumbs Up with Immediate Release)
        # -------------------------------------------------------------
        if is_thumbs_up:
            self.guide_dwell[handedness] = min(
                self.guide_dwell_threshold + 0.1,
                self.guide_dwell[handedness] + dt,
            )
            if self.guide_dwell[handedness] >= self.guide_dwell_threshold:
                self.guide_active[handedness] = True
        else:
            # Immediate, zero-lag dismissal the exact instant thumbs up is released
            self.guide_dwell[handedness] = 0.0
            self.guide_active[handedness] = False

        action_summary["guide_active"] = self.is_guide_active

        return action_summary

    def update_effects(self, dt: float) -> None:
        """Advances animations for shockwaves, plasma sparks, and gravitational vortex."""
        self.vortex_anim_time += dt

        remaining = []
        for sw in self.active_shockwaves:
            sw.elapsed += dt
            progress = sw.elapsed / sw.duration
            sw.radius = 20.0 + progress * (sw.max_radius - 20.0)
            if sw.elapsed < sw.duration:
                remaining.append(sw)
        self.active_shockwaves = remaining

