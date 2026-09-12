"""
Cyber-Medical Interactive HUD & Telemetry Visualizer.

Renders real-time hand skeleton annotations, bounding boxes, state machine badges,
circular dwell progress rings, continuous spatial slider gauges, and microsecond
latency telemetry.
"""

from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from mhands.core.geometry import (
    WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP,
    HAND_LANDMARK_NAMES,
)
from mhands.core.state_machine import SystemState, GestureEvent

# MediaPipe Hand Skeletal Connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky
    (0, 17)                                   # Palm base
]

# Color Palette (BGR)
COLOR_BG_DARK = (18, 18, 24)
COLOR_CYAN = (240, 210, 0)
COLOR_EMERALD = (80, 220, 100)
COLOR_AMBER = (30, 160, 255)
COLOR_CRIMSON = (60, 60, 230)
COLOR_WHITE = (245, 245, 245)
COLOR_GRAY = (140, 140, 150)
COLOR_PURPLE = (220, 120, 180)


class HUDVisualizer:
    """
    Renders high-performance interactive HUD on OpenCV video frames.
    """

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height

    def render(
        self,
        frame: np.ndarray,
        landmarks: Optional[np.ndarray],
        event: Optional[GestureEvent],
        probabilities: Optional[Dict[str, float]] = None,
        telemetry: Optional[Dict[str, float]] = None,
        slider_value: Optional[float] = None,
        features_8d: Optional[np.ndarray] = None,
        feature_type: str = "invariant",
    ) -> np.ndarray:
        """
        Composites HUD annotations onto frame.
        """
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # 1. Draw Hand Skeleton & Bounding Box if landmarks present
        if landmarks is not None:
            self._draw_skeleton(canvas, landmarks, w, h)
            self._draw_bounding_box(canvas, landmarks, w, h)

        # 2. Draw Top Telemetry Bar
        self._draw_telemetry_bar(canvas, telemetry, w)

        # 3. Draw Left Gesture & FSM Status Panel
        self._draw_status_panel(canvas, event, probabilities, feature_type, h)

        # 4. Draw Circular Dwell Progress on Hand if engaging
        if landmarks is not None and event is not None:
            if event.dwell_progress > 0.0 and event.state == SystemState.DWELL_ENGAGING:
                self._draw_dwell_indicator(canvas, landmarks, event.dwell_progress, w, h)

        # 5. Draw Continuous Interactive Slider if active
        if slider_value is not None:
            self._draw_slider(canvas, slider_value, w, h)

        # 6. Draw Invariant Feature Histogram (Bottom Right)
        if features_8d is not None and feature_type == "invariant":
            self._draw_feature_panel(canvas, features_8d, w, h)

        return canvas

    def _draw_skeleton(self, img: np.ndarray, landmarks: np.ndarray, w: int, h: int) -> None:
        pts_px = [(int(np.clip(lm[0], 0, 1) * w), int(np.clip(lm[1], 0, 1) * h)) for lm in landmarks]

        # Draw bones
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = pts_px[start_idx]
            pt2 = pts_px[end_idx]
            cv2.line(img, pt1, pt2, (120, 220, 120), 2, cv2.LINE_AA)

        # Draw joints
        tips = {THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP}
        for idx, pt in enumerate(pts_px):
            if idx in tips:
                cv2.circle(img, pt, 7, COLOR_AMBER, -1, cv2.LINE_AA)
                cv2.circle(img, pt, 9, COLOR_WHITE, 1, cv2.LINE_AA)
            elif idx == WRIST:
                cv2.circle(img, pt, 8, COLOR_CYAN, -1, cv2.LINE_AA)
                cv2.circle(img, pt, 11, COLOR_WHITE, 1, cv2.LINE_AA)
            else:
                cv2.circle(img, pt, 4, COLOR_EMERALD, -1, cv2.LINE_AA)

    def _draw_bounding_box(self, img: np.ndarray, landmarks: np.ndarray, w: int, h: int) -> None:
        xs = [int(np.clip(lm[0], 0, 1) * w) for lm in landmarks]
        ys = [int(np.clip(lm[1], 0, 1) * h) for lm in landmarks]
        x_min, x_max = max(0, min(xs) - 20), min(w - 1, max(xs) + 20)
        y_min, y_max = max(0, min(ys) - 20), min(h - 1, max(ys) + 20)

        # Cyberpunk corner brackets
        corner_len = 18
        color = COLOR_CYAN
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (50, 50, 60), 1, cv2.LINE_AA)

        # Top-left
        cv2.line(img, (x_min, y_min), (x_min + corner_len, y_min), color, 3)
        cv2.line(img, (x_min, y_min), (x_min, y_min + corner_len), color, 3)
        # Top-right
        cv2.line(img, (x_max, y_min), (x_max - corner_len, y_min), color, 3)
        cv2.line(img, (x_max, y_min), (x_max, y_min + corner_len), color, 3)
        # Bottom-left
        cv2.line(img, (x_min, y_max), (x_min + corner_len, y_max), color, 3)
        cv2.line(img, (x_min, y_max), (x_min, y_max - corner_len), color, 3)
        # Bottom-right
        cv2.line(img, (x_max, y_max), (x_max - corner_len, y_max), color, 3)
        cv2.line(img, (x_max, y_max), (x_max, y_max - corner_len), color, 3)

    def _draw_dwell_indicator(self, img: np.ndarray, landmarks: np.ndarray, progress: float, w: int, h: int) -> None:
        index_tip = landmarks[INDEX_TIP]
        cx = int(np.clip(index_tip[0], 0, 1) * w)
        cy = int(np.clip(index_tip[1], 0, 1) * h)

        radius = 28
        # Background track
        cv2.circle(img, (cx, cy), radius, (60, 60, 70), 3, cv2.LINE_AA)
        # Animated progress arc
        end_angle = int(360.0 * progress)
        cv2.ellipse(img, (cx, cy), (radius, radius), -90, 0, end_angle, COLOR_CYAN, 4, cv2.LINE_AA)
        # Progress text
        pct_text = f"{int(progress * 100)}%"
        cv2.putText(img, pct_text, (cx - 14, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1, cv2.LINE_AA)

    def _draw_telemetry_bar(self, img: np.ndarray, telemetry: Optional[Dict[str, float]], w: int) -> None:
        # Header bar
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        cv2.line(img, (0, 42), (w, 42), (60, 60, 75), 1)

        title = "M-HANDS // PRODUCTION CV PIPELINE"
        cv2.putText(img, title, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_CYAN, 2, cv2.LINE_AA)

        if telemetry:
            fps = telemetry.get("fps", 0.0)
            pipe_ms = telemetry.get("pipeline_ms", 0.0)
            clf_ms = telemetry.get("classifier_ms", 0.0)
            mp_ms = telemetry.get("mediapipe_ms", 0.0)

            telem_text = f"FPS: {fps:5.1f} | TOTAL: {pipe_ms:4.1f}ms | MP: {mp_ms:4.1f}ms | CLF: {clf_ms:4.2f}ms"
            cv2.putText(img, telem_text, (w - 530, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_EMERALD, 1, cv2.LINE_AA)

    def _draw_status_panel(
        self,
        img: np.ndarray,
        event: Optional[GestureEvent],
        probabilities: Optional[Dict[str, float]],
        feature_type: str,
        h: int,
    ) -> None:
        panel_x = 20
        panel_y = 60
        panel_w = 320
        panel_h = 240

        overlay = img.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        cv2.rectangle(img, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (60, 60, 75), 1)

        # Mode Badge
        mode_text = f"REPRESENTATION: {feature_type.upper()}"
        mode_col = COLOR_EMERALD if feature_type == "invariant" else COLOR_AMBER
        cv2.putText(img, mode_text, (panel_x + 15, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_col, 1, cv2.LINE_AA)

        # FSM State
        state_str = event.state.value if event else "IDLE"
        if state_str == "GESTURE_ACTIVE":
            state_color = COLOR_EMERALD
        elif state_str == "DWELL_ENGAGING":
            state_color = COLOR_CYAN
        elif state_str == "ACTION_RELEASED":
            state_color = COLOR_AMBER
        else:
            state_color = COLOR_GRAY

        cv2.putText(img, f"STATE: {state_str}", (panel_x + 15, panel_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2, cv2.LINE_AA)

        # Active Gesture
        active_name = (event.gesture_class if event and event.gesture_class else "None").replace("_", " ")
        conf = (event.confidence if event else 0.0) * 100.0
        cv2.putText(img, f"GESTURE: {active_name}", (panel_x + 15, panel_y + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 2, cv2.LINE_AA)

        # Confidence Bar
        bar_x = panel_x + 15
        bar_y = panel_y + 95
        bar_w = panel_w - 30
        bar_h = 10
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 50), -1)
        fill_w = int((conf / 100.0) * bar_w)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), state_color, -1)
        cv2.putText(img, f"CONF: {conf:4.1f}%", (bar_x, bar_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GRAY, 1, cv2.LINE_AA)

        # Probability distribution breakdown
        if probabilities:
            line_y = bar_y + 44
            for cls_name, prob in probabilities.items():
                short_name = cls_name.replace("_", " ")[:12]
                cv2.putText(img, f"{short_name:12s}: {prob * 100:4.1f}%", (panel_x + 15, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WHITE, 1, cv2.LINE_AA)
                line_y += 18

    def _draw_slider(self, img: np.ndarray, value: float, w: int, h: int) -> None:
        """Renders continuous slider control at bottom-center of frame."""
        slider_w = 460
        slider_h = 24
        x0 = (w - slider_w) // 2
        y0 = h - 60

        # Background track
        overlay = img.copy()
        cv2.rectangle(overlay, (x0 - 10, y0 - 25), (x0 + slider_w + 10, y0 + slider_h + 10), COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        cv2.rectangle(img, (x0, y0), (x0 + slider_w, y0 + slider_h), (50, 50, 60), -1)

        # Fill
        fill_w = int(np.clip(value / 100.0, 0.0, 1.0) * slider_w)
        cv2.rectangle(img, (x0, y0), (x0 + fill_w, y0 + slider_h), COLOR_CYAN, -1)
        # Thumb handle
        cv2.circle(img, (x0 + fill_w, y0 + slider_h // 2), 14, COLOR_WHITE, -1, cv2.LINE_AA)
        cv2.circle(img, (x0 + fill_w, y0 + slider_h // 2), 14, COLOR_CYAN, 2, cv2.LINE_AA)

        label = f"CONTINUOUS SPATIAL CONTROL: {value:5.1f}%"
        cv2.putText(img, label, (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_CYAN, 1, cv2.LINE_AA)

    def _draw_feature_panel(self, img: np.ndarray, feats: np.ndarray, w: int, h: int) -> None:
        """Renders the 8D invariant features as dynamic horizontal bars."""
        panel_w = 260
        panel_h = 190
        x0 = w - panel_w - 20
        y0 = h - panel_h - 20

        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (60, 60, 75), 1)

        cv2.putText(img, "R^8 INVARIANT MANIFOLD", (x0 + 12, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_CYAN, 1, cv2.LINE_AA)

        names = ["Thmb", "Indx", "Midd", "Ring", "Pnk", "Abdc", "Sprd", "Aper"]
        max_vals = [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, 2.5, 2.5]

        bar_y = y0 + 36
        bar_len = 130
        for i in range(min(8, len(feats))):
            val = feats[i]
            ratio = float(np.clip(val / max_vals[i], 0.0, 1.0))
            cv2.putText(img, names[i], (x0 + 12, bar_y + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_GRAY, 1, cv2.LINE_AA)
            cv2.rectangle(img, (x0 + 55, bar_y), (x0 + 55 + bar_len, bar_y + 10), (40, 40, 50), -1)
            cv2.rectangle(img, (x0 + 55, bar_y), (x0 + 55 + int(ratio * bar_len), bar_y + 10), COLOR_PURPLE, -1)
            cv2.putText(img, f"{val:3.2f}", (x0 + 195, bar_y + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_WHITE, 1, cv2.LINE_AA)
            bar_y += 18
