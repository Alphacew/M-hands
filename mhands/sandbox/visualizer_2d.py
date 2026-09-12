"""
Jedi Sandbox 2D Renderer & Visualizer.

Composites physical simulation bodies, FEA stress heatmaps, luminous stress-vector
tether beams, atmospheric shockwave distortion rings, plasma cutting blades,
and slow-motion visual indicators over video or holographic chamber backgrounds.
"""

from typing import List, Tuple, Dict, Optional, Any
import cv2
import numpy as np
import pymunk
from pymunk import Vec2d

from mhands.sandbox.physics_space import PhysicsSpace
from mhands.sandbox.telekinesis import TelekineticController
from mhands.pipeline.landmarker import HandData
from mhands.core.geometry import (
    WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP,
)

# MediaPipe Hand Skeletal Connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky
    (0, 17)                                   # Palm base
]

# Sci-Fi Color Palette (BGR)
COLOR_CHAMBER_BG = (16, 18, 24)
COLOR_GRID_LINE = (28, 32, 42)
COLOR_CYAN_GLOW = (240, 210, 0)
COLOR_AMBER_GLOW = (30, 160, 255)
COLOR_CRIMSON_GLOW = (50, 50, 240)
COLOR_MAGENTA_STRESS = (220, 40, 220)
COLOR_LASER_CORE = (255, 255, 255)
COLOR_LASER_EDGE = (60, 255, 120)
COLOR_SHIELD = (240, 180, 40)
COLOR_WHITE = (245, 245, 245)
COLOR_BONE = (120, 220, 120)
COLOR_EMERALD = (80, 220, 100)


class SandboxVisualizer:
    """
    Renders 2D physics sandbox with advanced visual effects.
    """

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height

    def render(
        self,
        physics: PhysicsSpace,
        controller: TelekineticController,
        hands: List[HandData],
        camera_frame: Optional[np.ndarray] = None,
        ar_mode: bool = False,
        environment_name: str = "Jenga_Castle",
        fps: float = 60.0,
    ) -> np.ndarray:
        """
        Renders complete frame with physics bodies, telekinetic forces, and HUD.
        """
        w, h = self.width, self.height

        # 1. Base Canvas: Transparent Camera Feed (AR Mode)
        if ar_mode and camera_frame is not None:
            # Clean transparent webcam background - user sees hands and environment directly
            if camera_frame.shape[1] != w or camera_frame.shape[0] != h:
                canvas = cv2.resize(camera_frame, (w, h))
            else:
                canvas = camera_frame.copy()
        else:
            # Holographic Sci-Fi Chamber with Isometric Grid
            canvas = np.full((h, w, 3), COLOR_CHAMBER_BG, dtype=np.uint8)
            # Perspective floor grid
            grid_spacing = 48
            for x in range(0, w, grid_spacing):
                cv2.line(canvas, (x, 0), (x, h), COLOR_GRID_LINE, 1)
            for y in range(0, h, grid_spacing):
                cv2.line(canvas, (0, y), (w, y), COLOR_GRID_LINE, 1)

        # 2. Render Semi-Transparent Physics Bodies (visible hands behind pieces)
        self._render_physics_bodies(canvas, physics, alpha=0.65 if ar_mode else 0.95)

        # 3. Render Atmospheric Shockwaves
        self._render_shockwaves(canvas, controller.active_shockwaves)

        # 4. Render Deflector Shields
        for handedness, shield_data in controller.deflector_shields.items():
            if shield_data is not None:
                center, radius = shield_data
                self._render_shield(canvas, center, radius)

        # 5. Render Stress-Vector Tether Lines
        for handedness, grip in controller.grips.items():
            if grip is not None:
                hand_pos = controller.hand_positions[handedness]
                world_anchor = grip.body.local_to_world(grip.local_anchor)
                self._render_tether(canvas, hand_pos, world_anchor, grip.tension)

        # 6. Render Plasma Bisection Blades
        for handedness, laser in controller.active_lasers.items():
            if laser is not None:
                self._render_laser(canvas, laser.p1, laser.p2, laser.sparks)

        # 7. Render Hand Skeletons & Pinch Rings
        self._render_hands(canvas, hands, controller)

        # 8. Render Top Telemetry & Environment HUD
        self._render_hud(canvas, physics, controller, environment_name, fps)

        # 9. Render Holographic On-Screen Guide if Thumbs-Up Gesture is Active
        if controller.is_guide_active:
            self._render_guide(canvas)

        return canvas

    def _render_physics_bodies(self, img: np.ndarray, physics: PhysicsSpace, alpha: float = 0.72) -> None:
        """Draws all rigid bodies with semi-transparent crystal fills and crisp neon outlines."""
        overlay = img.copy()

        for body in physics.space.bodies:
            is_static = (body.body_type == pymunk.Body.STATIC)
            v_len = body.velocity.length
            stress_ratio = min(1.0, v_len / 450.0) if not is_static else 0.0

            for shape in body.shapes:
                if shape in physics.boundaries:
                    continue  # Don't draw outer bounding boundaries

                base_color = getattr(shape, "color", (140, 180, 210))

                # Interpolate color with magenta stress
                if stress_ratio > 0.15:
                    r = int((1 - stress_ratio) * base_color[2] + stress_ratio * COLOR_MAGENTA_STRESS[2])
                    g = int((1 - stress_ratio) * base_color[1] + stress_ratio * COLOR_MAGENTA_STRESS[1])
                    b = int((1 - stress_ratio) * base_color[0] + stress_ratio * COLOR_MAGENTA_STRESS[0])
                    color = (b, g, r)
                else:
                    color = base_color

                if isinstance(shape, pymunk.Poly):
                    verts = [body.local_to_world(v) for v in shape.get_vertices()]
                    pts_px = np.array([[int(v.x), int(v.y)] for v in verts], dtype=np.int32)
                    if len(pts_px) >= 3:
                        cv2.fillPoly(overlay, [pts_px], color)

                elif isinstance(shape, pymunk.Circle):
                    center_px = (int(body.position.x), int(body.position.y))
                    rad_px = int(shape.radius)
                    cv2.circle(overlay, center_px, rad_px, color, -1, cv2.LINE_AA)

                elif isinstance(shape, pymunk.Segment):
                    p1 = body.local_to_world(shape.a)
                    p2 = body.local_to_world(shape.b)
                    p1_px = (int(p1.x), int(p1.y))
                    p2_px = (int(p2.x), int(p2.y))
                    radius = max(2, int(shape.radius))
                    cv2.line(overlay, p1_px, p2_px, color, radius * 2, cv2.LINE_AA)

        # Alpha blend fills onto base canvas
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

        # Draw crisp high-contrast outlines and orientation spokes
        for body in physics.space.bodies:
            is_static = (body.body_type == pymunk.Body.STATIC)
            for shape in body.shapes:
                if shape in physics.boundaries:
                    continue

                if isinstance(shape, pymunk.Poly):
                    verts = [body.local_to_world(v) for v in shape.get_vertices()]
                    pts_px = np.array([[int(v.x), int(v.y)] for v in verts], dtype=np.int32)
                    if len(pts_px) >= 3:
                        border_col = (255, 255, 255) if is_static else (220, 245, 255)
                        cv2.polylines(img, [pts_px], True, border_col, 2, cv2.LINE_AA)

                elif isinstance(shape, pymunk.Circle):
                    center_px = (int(body.position.x), int(body.position.y))
                    rad_px = int(shape.radius)
                    cv2.circle(img, center_px, rad_px, (220, 245, 255), 2, cv2.LINE_AA)
                    angle = body.angle
                    spoke_end = (
                        int(center_px[0] + rad_px * np.cos(angle)),
                        int(center_px[1] + rad_px * np.sin(angle)),
                    )
                    cv2.line(img, center_px, spoke_end, (255, 255, 255), 1, cv2.LINE_AA)

                elif isinstance(shape, pymunk.Segment):
                    p1 = body.local_to_world(shape.a)
                    p2 = body.local_to_world(shape.b)
                    p1_px = (int(p1.x), int(p1.y))
                    p2_px = (int(p2.x), int(p2.y))
                    radius = max(2, int(shape.radius))
                    cv2.line(img, p1_px, p2_px, color, radius * 2, cv2.LINE_AA)

    def _render_tether(self, img: np.ndarray, p1: Vec2d, p2: Vec2d, tension: float) -> None:
        """Renders luminous lightning stress-vector tether from hand to object."""
        # Color interpolation based on tension (Cyan -> Amber -> Crimson)
        if tension < 0.5:
            r = tension / 0.5
            color = (
                int((1 - r) * COLOR_CYAN_GLOW[0] + r * COLOR_AMBER_GLOW[0]),
                int((1 - r) * COLOR_CYAN_GLOW[1] + r * COLOR_AMBER_GLOW[1]),
                int((1 - r) * COLOR_CYAN_GLOW[2] + r * COLOR_AMBER_GLOW[2]),
            )
        else:
            r = (tension - 0.5) / 0.5
            color = (
                int((1 - r) * COLOR_AMBER_GLOW[0] + r * COLOR_CRIMSON_GLOW[0]),
                int((1 - r) * COLOR_AMBER_GLOW[1] + r * COLOR_CRIMSON_GLOW[1]),
                int((1 - r) * COLOR_AMBER_GLOW[2] + r * COLOR_CRIMSON_GLOW[2]),
            )

        pt1 = (int(np.clip(p1.x, -3000, 4000)), int(np.clip(p1.y, -3000, 4000)))
        pt2 = (int(np.clip(p2.x, -3000, 4000)), int(np.clip(p2.y, -3000, 4000)))

        # Core line
        cv2.line(img, pt1, pt2, color, 3, cv2.LINE_AA)
        # Inner energy beam
        cv2.line(img, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA)

        # Connection anchors
        cv2.circle(img, pt1, 6, color, -1, cv2.LINE_AA)
        cv2.circle(img, pt2, 6, color, -1, cv2.LINE_AA)

    def _render_shockwaves(self, img: np.ndarray, shockwaves: List[Any]) -> None:
        """Renders expanding concentric shockwave ripples."""
        for sw in shockwaves:
            cx, cy = int(sw.center.x), int(sw.center.y)
            r = int(sw.radius)
            progress = sw.elapsed / sw.duration
            alpha = max(0.0, 1.0 - progress)
            color = (int(240 * alpha), int(220 * alpha), int(100 * alpha))

            cv2.circle(img, (cx, cy), r, color, 3, cv2.LINE_AA)
            if r > 30:
                cv2.circle(img, (cx, cy), int(r * 0.75), color, 1, cv2.LINE_AA)

    def _render_shield(self, img: np.ndarray, center: Vec2d, radius: float) -> None:
        """Renders dynamic deflector shield barrier."""
        cx, cy = int(center.x), int(center.y)
        r = int(radius)
        cv2.circle(img, (cx, cy), r, COLOR_SHIELD, 2, cv2.LINE_AA)
        # Concentric hexagon shield lattice
        for a in range(0, 360, 60):
            rad1 = np.radians(a)
            rad2 = np.radians(a + 60)
            p1 = (int(cx + r * np.cos(rad1)), int(cy + r * np.sin(rad1)))
            p2 = (int(cx + r * np.cos(rad2)), int(cy + r * np.sin(rad2)))
            cv2.line(img, p1, p2, (255, 220, 100), 2, cv2.LINE_AA)

    def _render_laser(self, img: np.ndarray, p1: Vec2d, p2: Vec2d, sparks: List[Vec2d]) -> None:
        """Renders plasma bisection laser cutter blade with spark particles."""
        pt1 = (int(np.clip(p1.x, -3000, 4000)), int(np.clip(p1.y, -3000, 4000)))
        pt2 = (int(np.clip(p2.x, -3000, 4000)), int(np.clip(p2.y, -3000, 4000)))

        # Outer green/cyan glow
        cv2.line(img, pt1, pt2, COLOR_LASER_EDGE, 6, cv2.LINE_AA)
        # Inner white plasma core
        cv2.line(img, pt1, pt2, COLOR_LASER_CORE, 2, cv2.LINE_AA)

        # Render sparks
        for sp in sparks:
            spx, spy = int(sp.x), int(sp.y)
            for _ in range(5):
                ox = int(spx + np.random.uniform(-14, 14))
                oy = int(spy + np.random.uniform(-14, 14))
                cv2.circle(img, (ox, oy), 2, (80, 180, 255), -1, cv2.LINE_AA)

    def _render_hands(self, img: np.ndarray, hands: List[HandData], controller: TelekineticController) -> None:
        w, h = self.width, self.height
        tips_indices = {THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP}

        for hand in hands:
            lms = hand.landmarks
            pts_px = [(int(np.clip(lm[0], 0, 1) * w), int(np.clip(lm[1], 0, 1) * h)) for lm in lms]

            # 1. MediaPipe Full 21-Joint Skeletal Connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(img, pts_px[start_idx], pts_px[end_idx], COLOR_BONE, 2, cv2.LINE_AA)

            # 2. Distinct Color-Coded Landmarks
            for idx, pt in enumerate(pts_px):
                if idx in tips_indices:
                    # White-ringed amber fingertips
                    cv2.circle(img, pt, 7, COLOR_AMBER_GLOW, -1, cv2.LINE_AA)
                    cv2.circle(img, pt, 9, COLOR_WHITE, 1, cv2.LINE_AA)
                elif idx == WRIST:
                    # White-ringed cyan wrist root
                    cv2.circle(img, pt, 8, COLOR_CYAN_GLOW, -1, cv2.LINE_AA)
                    cv2.circle(img, pt, 11, COLOR_WHITE, 1, cv2.LINE_AA)
                else:
                    # Emerald knuckles and intermediate phalanx joints
                    cv2.circle(img, pt, 4, COLOR_EMERALD, -1, cv2.LINE_AA)

            # 3. High-Tech Cyberpunk Corner Bounding Box
            xs = [p[0] for p in pts_px]
            ys = [p[1] for p in pts_px]
            x_min, x_max = max(0, min(xs) - 18), min(w - 1, max(xs) + 18)
            y_min, y_max = max(0, min(ys) - 18), min(h - 1, max(ys) + 18)

            is_grip = controller.grips[hand.handedness] is not None
            is_laser = controller.active_lasers[hand.handedness] is not None
            is_shield = controller.deflector_shields.get(hand.handedness) is not None
            is_guide = controller.guide_active.get(hand.handedness, False)
            is_singularity = controller.singularity_active.get(hand.handedness, False)

            # Accent color based on telekinetic action
            if is_grip:
                bracket_color = COLOR_CRIMSON_GLOW
            elif is_singularity:
                bracket_color = COLOR_MAGENTA_STRESS
            elif is_guide:
                bracket_color = COLOR_AMBER_GLOW
            elif is_laser:
                bracket_color = COLOR_LASER_EDGE
            elif is_shield:
                bracket_color = COLOR_SHIELD
            else:
                bracket_color = COLOR_CYAN_GLOW

            # Outer subtle boundary
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (50, 55, 65), 1, cv2.LINE_AA)

            # Cyberpunk corner brackets (16px)
            c_len = 16
            # Top-left
            cv2.line(img, (x_min, y_min), (x_min + c_len, y_min), bracket_color, 2, cv2.LINE_AA)
            cv2.line(img, (x_min, y_min), (x_min, y_min + c_len), bracket_color, 2, cv2.LINE_AA)
            # Top-right
            cv2.line(img, (x_max, y_min), (x_max - c_len, y_min), bracket_color, 2, cv2.LINE_AA)
            cv2.line(img, (x_max, y_min), (x_max, y_min + c_len), bracket_color, 2, cv2.LINE_AA)
            # Bottom-left
            cv2.line(img, (x_min, y_max), (x_min + c_len, y_max), bracket_color, 2, cv2.LINE_AA)
            cv2.line(img, (x_min, y_max), (x_min, y_max - c_len), bracket_color, 2, cv2.LINE_AA)
            # Bottom-right
            cv2.line(img, (x_max, y_max), (x_max - c_len, y_max), bracket_color, 2, cv2.LINE_AA)
            cv2.line(img, (x_max, y_max), (x_max, y_max - c_len), bracket_color, 2, cv2.LINE_AA)

            # 4. Pinch Midpoint & Magnetic Target Highlight
            p_idx = pts_px[INDEX_TIP]
            p_thb = pts_px[THUMB_TIP]
            mid_p = ((p_idx[0] + p_thb[0]) // 2, (p_idx[1] + p_thb[1]) // 2)

            if is_grip:
                # Active telekinetic spring grip
                cv2.circle(img, mid_p, 10, COLOR_CRIMSON_GLOW, 2, cv2.LINE_AA)
                cv2.circle(img, mid_p, 4, COLOR_WHITE, -1, cv2.LINE_AA)
                cv2.line(img, p_idx, p_thb, COLOR_CRIMSON_GLOW, 2, cv2.LINE_AA)
                badge_text = f"[{hand.handedness.upper()} // TELEKINETIC GRIP]"
            elif is_singularity:
                # Active Gravitational Singularity
                badge_text = f"[{hand.handedness.upper()} // GRAVITATIONAL SINGULARITY]"
                cx, cy = int(mid_p[0]), int(mid_p[1])
                t_anim = controller.vortex_anim_time * 3.0
                for ring_i in range(3):
                    phase_r = (t_anim + ring_i * 0.33) % 1.0
                    r_cur = int(25 + (1.0 - phase_r) * 130)
                    alpha_r = max(0.0, min(1.0, phase_r * 1.5))
                    c_ring = (int(220 * alpha_r), int(40 * alpha_r), int(220 * alpha_r))
                    cv2.circle(img, (cx, cy), r_cur, c_ring, 2, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), 10, COLOR_MAGENTA_STRESS, -1, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), 13, COLOR_WHITE, 1, cv2.LINE_AA)
            elif is_guide:
                # Thumbs-Up on-screen guide active
                badge_text = f"[{hand.handedness.upper()} // ON-SCREEN GUIDE]"
                p_thb_pt = pts_px[THUMB_TIP]
                cv2.circle(img, p_thb_pt, 12, COLOR_AMBER_GLOW, 2, cv2.LINE_AA)
                cv2.circle(img, p_thb_pt, 16, (60, 220, 255), 1, cv2.LINE_AA)
            else:
                is_hand_pinching = controller.is_pinching.get(hand.handedness, False)
                if is_hand_pinching:
                    cv2.circle(img, mid_p, 7, COLOR_AMBER_GLOW, 2, cv2.LINE_AA)
                    cv2.line(img, p_idx, p_thb, COLOR_AMBER_GLOW, 2, cv2.LINE_AA)
                    badge_text = f"[{hand.handedness.upper()} // PINCH]"
                elif is_laser:
                    badge_text = f"[{hand.handedness.upper()} // PLASMA CUTTER]"
                elif is_shield:
                    badge_text = f"[{hand.handedness.upper()} // DEFLECTOR SHIELD]"
                else:
                    badge_text = f"[{hand.handedness.upper()} // READY]"

                # Subtle magnetic reticle to nearest body within grab radius
                nearest_b = controller.physics.query_nearest_body(mid_p, controller.grab_radius)
                if nearest_b is not None:
                    target_pos = (int(nearest_b.position.x), int(nearest_b.position.y))
                    cv2.line(img, mid_p, target_pos, (100, 240, 240), 1, cv2.LINE_AA)
                    cv2.circle(img, target_pos, 8, (100, 240, 240), 1, cv2.LINE_AA)

            # 5. Hand State Pill Badge (Above Bounding Box)
            badge_y = max(24, y_min - 8)
            (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            bx1 = x_min
            by1 = badge_y - bh - 6
            bx2 = bx1 + bw + 12
            by2 = badge_y + 2
            # Dark pill backdrop
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (18, 20, 28), -1)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), bracket_color, 1, cv2.LINE_AA)
            cv2.putText(
                img,
                badge_text,
                (bx1 + 6, badge_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                bracket_color,
                1,
                cv2.LINE_AA,
            )

    def _render_hud(
        self,
        img: np.ndarray,
        physics: PhysicsSpace,
        controller: TelekineticController,
        environment_name: str,
        fps: float,
    ) -> None:
        w, h = self.width, self.height

        # Top Bar
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (18, 20, 28), -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        cv2.line(img, (0, 42), (w, 42), (50, 55, 70), 1)

        title = f"JEDI 2D PHYSICS SANDBOX // {environment_name.upper()}"
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        font_scale = 0.58 if (20 + tw < w - 430) else 0.46
        cv2.putText(img, title, (20, 27), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_CYAN_GLOW, 2, cv2.LINE_AA)

        telem = f"FPS: {fps:4.1f} | BODIES: {len(physics.space.bodies):2d} | JOINTS: {len(physics.space.constraints):2d}"
        cv2.putText(img, telem, (w - 410, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 230, 140), 1, cv2.LINE_AA)

        # Stasis Banner
        if physics.is_stasis:
            cv2.putText(
                img,
                "[SPATIAL STASIS // FROZEN]",
                (w // 2 - 160, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 240, 100),
                2,
                cv2.LINE_AA,
            )

        # Slow-Motion Banner
        if physics.time_scale < 0.9:
            cv2.putText(
                img,
                f"[TIME DILATION: {int(physics.time_scale * 100)}% SLO-MO]",
                (w // 2 - 180, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                COLOR_AMBER_GLOW,
                2,
                cv2.LINE_AA,
            )

        # Controls hint overlay bar (bottom)
        overlay_b = img.copy()
        cv2.rectangle(overlay_b, (0, h - 26), (w, h), (14, 16, 22), -1)
        cv2.addWeighted(overlay_b, 0.85, img, 0.15, 0, img)
        cv2.line(img, (0, h - 26), (w, h - 26), (40, 45, 58), 1)

        hints = "Hotkeys: 1-5 Environments | 'r' Reset | 'a' AR Toggle | 's' Slow-Motion | 'p' Snapshot | 'q' Exit"
        cv2.putText(img, hints, (20, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 190, 210), 1, cv2.LINE_AA)

    def _render_guide(self, img: np.ndarray) -> None:
        """Renders cyberpunk holographic on-screen manual / holocron guide."""
        w, h = self.width, self.height
        card_w, card_h = 880, 520
        cx, cy = w // 2, h // 2
        x1, y1 = cx - card_w // 2, cy - card_h // 2
        x2, y2 = x1 + card_w, y1 + card_h

        # 1. Semi-transparent backdrop
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (12, 14, 20), -1)
        cv2.addWeighted(overlay, 0.90, img, 0.10, 0, img)

        # 2. Cyberpunk borders & glowing corner brackets
        cv2.rectangle(img, (x1, y1), (x2, y2), (50, 55, 70), 1, cv2.LINE_AA)
        c_len = 26
        bracket_color = COLOR_AMBER_GLOW
        # Top-left
        cv2.line(img, (x1, y1), (x1 + c_len, y1), bracket_color, 3, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + c_len), bracket_color, 3, cv2.LINE_AA)
        # Top-right
        cv2.line(img, (x2, y1), (x2 - c_len, y1), bracket_color, 3, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + c_len), bracket_color, 3, cv2.LINE_AA)
        # Bottom-left
        cv2.line(img, (x1, y2), (x1 + c_len, y2), bracket_color, 3, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - c_len), bracket_color, 3, cv2.LINE_AA)
        # Bottom-right
        cv2.line(img, (x2, y2), (x2 - c_len, y2), bracket_color, 3, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - c_len), bracket_color, 3, cv2.LINE_AA)

        # 3. Header Ribbon
        cv2.rectangle(img, (x1 + 1, y1 + 1), (x2 - 1, y1 + 54), (20, 24, 34), -1)
        cv2.line(img, (x1, y1 + 54), (x2, y1 + 54), (70, 80, 100), 1)

        title = "JEDI 2D PHYSICS SANDBOX // ON-SCREEN INTERFACE MANUAL"
        cv2.putText(img, title, (x1 + 24, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, COLOR_AMBER_GLOW, 2, cv2.LINE_AA)
        sub = "[ HOLD THUMBS-UP GESTURE TO DISPLAY GUIDE  |  RELEASE TO DISMISS ]"
        cv2.putText(img, sub, (x1 + 24, y1 + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 220, 140), 1, cv2.LINE_AA)

        # 4. Column Layout: Left Column = Powers, Right Column = Hotkeys
        col1_x = x1 + 24
        col2_x = x1 + 460
        sep_x = x1 + 440

        # Vertical divider line
        cv2.line(img, (sep_x, y1 + 65), (sep_x, y2 - 35), (45, 50, 65), 1)

        # Left Header: Telekinetic Powers
        cv2.putText(img, "TELEKINETIC POWERS & KINEMATICS", (col1_x, y1 + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_CYAN_GLOW, 2, cv2.LINE_AA)

        powers = [
            ("[ PINCH ]", "Elastic Force Grip", "Pinch index & thumb near a block to latch a variable spring.", "Tighten to lock; loosen to swing; release to fling."),
            ("[ CLOSED FIST ]", "Gravitational Singularity", "Curl into a fist to attract ALL blocks across chamber.", "Blocks gather and orbit smoothly around your fist."),
            ("[ OPEN PALM ]", "Kinetic Force Push", "Rapidly snap open hand to blast nearby blocks away", "with an explosive expanding shockwave ripple."),
            ("[ FLAT PALM ]", "Spatial Stasis (Freeze)", "Hold flat palm stationary for 250ms to freeze", "physical time and all body velocities mid-air."),
            ("[ VICTORY SIGN ]", "Plasma Cutter Blade", "Extend index + middle fingers to cast a plasma beam", "that cleanly bisects convex polygons in two."),
            ("[ LEFT PALM ]", "Deflector Shield Barrier", "Left hand open palm deploys dynamic hexagonal", "deflector barrier that repels incoming debris."),
        ]

        curr_y = y1 + 108
        for tag, name, desc1, desc2 in powers:
            cv2.putText(img, tag, (col1_x, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f": {name}", (col1_x + 90, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_WHITE, 1, cv2.LINE_AA)
            curr_y += 18
            cv2.putText(img, desc1, (col1_x + 10, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (170, 175, 185), 1, cv2.LINE_AA)
            curr_y += 16
            cv2.putText(img, desc2, (col1_x + 10, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (140, 145, 155), 1, cv2.LINE_AA)
            curr_y += 24

        # Right Header: Environments & Hotkeys
        cv2.putText(img, "CHAMBER CONTROLS & HOTKEYS", (col2_x, y1 + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_CYAN_GLOW, 2, cv2.LINE_AA)

        hotkeys = [
            ("'1'", "Jenga Castle & Keystone Arch", "Masonry arch under balanced gravity load."),
            ("'2'", "Articulated Ragdoll Arena", "Humanoid dummies with biological joint limits."),
            ("'3'", "Granular Fluid Vat", "120+ viscous fluid particles with splash physics."),
            ("'4'", "Zero-G Orbital Asteroids", "Microgravity chamber with orbiting satellites."),
            ("'5'", "Seismic Truss Bridge", "Pinned bridge span under heavy vehicular load."),
            ("'r'", "Reset Simulation", "Reloads current toybox in initial state."),
            ("'a'", "Toggle Transparent AR Feed", "Switch between transparent webcam & holo-chamber."),
            ("'s'", "Slow-Motion Dilation", "Triggers 0.2x cinematic slow-motion effect."),
            ("'p'", "Capture HD Snapshot", "Saves high-res snapshot PNG to current directory."),
            ("'q'", "Exit Application", "Closes physics engine and camera feed cleanly."),
        ]

        curr_y2 = y1 + 108
        for key, name, desc in hotkeys:
            cv2.putText(img, f"Key {key}", (col2_x, curr_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 240, 150), 1, cv2.LINE_AA)
            cv2.putText(img, f": {name}", (col2_x + 60, curr_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_WHITE, 1, cv2.LINE_AA)
            curr_y2 += 18
            cv2.putText(img, desc, (col2_x + 10, curr_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (140, 145, 155), 1, cv2.LINE_AA)
            curr_y2 += 21

        # 5. Bottom Status Banner
        cv2.line(img, (x1, y2 - 30), (x2, y2 - 30), (50, 55, 70), 1)
        cv2.putText(
            img,
            "SYSTEM: ACTIVE // INVARIANT R8 COMPUTER VISION // ZERO PERCEPTIBLE INPUT LAG",
            (x1 + 24, y2 - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (120, 130, 150),
            1,
            cv2.LINE_AA,
        )
