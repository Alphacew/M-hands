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
from mhands.core.geometry import INDEX_TIP, THUMB_TIP, WRIST


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
        self._render_physics_bodies(canvas, physics, alpha=0.72 if ar_mode else 0.95)

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

        pt1 = (int(p1.x), int(p1.y))
        pt2 = (int(p2.x), int(p2.y))

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
        pt1 = (int(p1.x), int(p1.y))
        pt2 = (int(p2.x), int(p2.y))

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
        for hand in hands:
            lms = hand.landmarks
            pts_px = [(int(lm[0] * w), int(lm[1] * h)) for lm in lms]

            # Index & Thumb tips
            p_idx = pts_px[INDEX_TIP]
            p_thb = pts_px[THUMB_TIP]

            # Highlight pinch midpoint
            mid_p = ((p_idx[0] + p_thb[0]) // 2, (p_idx[1] + p_thb[1]) // 2)
            is_grip = controller.grips[hand.handedness] is not None
            grip_color = COLOR_CRIMSON_GLOW if is_grip else COLOR_CYAN_GLOW

            cv2.circle(img, p_idx, 6, grip_color, -1, cv2.LINE_AA)
            cv2.circle(img, p_thb, 6, grip_color, -1, cv2.LINE_AA)
            cv2.line(img, p_idx, p_thb, grip_color, 1, cv2.LINE_AA)

            # Handedness label
            cv2.putText(
                img,
                hand.handedness.upper(),
                (pts_px[WRIST][0] - 20, pts_px[WRIST][1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
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
        cv2.putText(img, title, (20, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_CYAN_GLOW, 2, cv2.LINE_AA)

        telem = f"FPS: {fps:4.1f} | BODIES: {len(physics.space.bodies):2d} | JOINTS: {len(physics.space.constraints):2d}"
        cv2.putText(img, telem, (w - 420, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 230, 140), 1, cv2.LINE_AA)

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
                (w // 2 - 180, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                COLOR_AMBER_GLOW,
                2,
                cv2.LINE_AA,
            )

        # Controls hint (bottom left)
        hints = "Keys: 1-5 Environment | 'r' Reset | 'a' AR Toggle | 's' Slo-Mo | 'q' Quit"
        cv2.putText(img, hints, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140, 140, 160), 1, cv2.LINE_AA)
