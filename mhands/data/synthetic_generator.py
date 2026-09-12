"""
Anatomical Synthetic Hand Landmark Generator.

Synthesizes biomechanically authentic 21-landmark 3D hand poses for gesture classes:
Open_Palm, Closed_Fist, Index_Point, and Victory.
Simulates realistic multi-session domain shifts including extreme translation,
sensor distance variation (scale shifts), 3D perspective tilts, and monocular noise.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np

from mhands.core.geometry import (
    WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP,
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP,
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP,
    RING_MCP, RING_PIP, RING_DIP, RING_TIP,
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP,
)


class SyntheticHandGenerator:
    """
    Kinematic hand generator producing 21 3D landmarks for benchmark gestures.
    """

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.default_rng(random_seed)

    def _rotation_matrix(self, rx: float, ry: float, rz: float) -> np.ndarray:
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

        return Rz @ Ry @ Rx

    def generate_canonical_pose(self, gesture_class: str) -> np.ndarray:
        """
        Builds a canonical normalized 3D hand skeleton (21, 3) centered at origin.
        """
        pts = np.zeros((21, 3), dtype=np.float32)

        # Base palm proportions
        pts[WRIST] = [0.0, 0.0, 0.0]

        # MCP Base positions (relative to wrist)
        pts[THUMB_CMC] = [-0.06, 0.05, -0.01]
        pts[THUMB_MCP] = [-0.11, 0.12, -0.02]

        pts[INDEX_MCP]  = [-0.05, 0.19, 0.0]
        pts[MIDDLE_MCP] = [ 0.00, 0.20, 0.0]  # Reference d_ref = 0.20
        pts[RING_MCP]   = [ 0.05, 0.18, 0.0]
        pts[PINKY_MCP]  = [ 0.09, 0.16, 0.0]

        # Articulation state parameters per gesture
        # (is_extended: True/False, spread_offset)
        if gesture_class == "Open_Palm":
            curl = {"thumb": 0.1, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0}
            spread = {"thumb": -0.05, "index": -0.02, "middle": 0.0, "ring": 0.02, "pinky": 0.05}
        elif gesture_class == "Closed_Fist":
            curl = {"thumb": 0.8, "index": 0.9, "middle": 0.9, "ring": 0.9, "pinky": 0.9}
            spread = {"thumb": 0.02, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0}
        elif gesture_class == "Index_Point":
            curl = {"thumb": 0.7, "index": 0.0, "middle": 0.9, "ring": 0.9, "pinky": 0.9}
            spread = {"thumb": 0.01, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0}
        elif gesture_class == "Victory":
            curl = {"thumb": 0.7, "index": 0.0, "middle": 0.0, "ring": 0.9, "pinky": 0.9}
            spread = {"thumb": 0.01, "index": -0.04, "middle": 0.04, "ring": 0.0, "pinky": 0.0}
        else:
            raise ValueError(f"Unknown gesture class: {gesture_class}")

        # Construct finger joint chains
        # Thumb chain
        c_th = curl["thumb"]
        s_th = spread["thumb"]
        pts[THUMB_IP] = pts[THUMB_MCP] + np.array([-0.04 * (1.0 - c_th) + s_th, 0.05 * (1.0 - c_th), 0.04 * c_th])
        pts[THUMB_TIP] = pts[THUMB_IP] + np.array([-0.03 * (1.0 - c_th) + s_th, 0.04 * (1.0 - c_th), 0.04 * c_th])

        # Digit segments configuration
        digits = [
            ("index", INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, 0.07, 0.05, 0.04),
            ("middle", MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, 0.08, 0.06, 0.04),
            ("ring", RING_MCP, RING_PIP, RING_DIP, RING_TIP, 0.07, 0.05, 0.04),
            ("pinky", PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, 0.06, 0.04, 0.03),
        ]

        for name, mcp_i, pip_i, dip_i, tip_i, l1, l2, l3 in digits:
            c = curl[name]
            sp = spread[name]

            # Extended extends upwards in +y; curled curls inwards towards palm (+z and -y)
            if c < 0.3:
                # Extended digit
                pts[pip_i] = pts[mcp_i] + np.array([sp * 0.3, l1, 0.0])
                pts[dip_i] = pts[pip_i] + np.array([sp * 0.6, l2, 0.0])
                pts[tip_i] = pts[dip_i] + np.array([sp * 1.0, l3, 0.0])
            else:
                # Curled digit
                pts[pip_i] = pts[mcp_i] + np.array([sp * 0.2, l1 * 0.5, l1 * 0.6])
                pts[dip_i] = pts[pip_i] + np.array([0.0, -l2 * 0.5, l2 * 0.7])
                pts[tip_i] = pts[dip_i] + np.array([0.0, -l3 * 0.8, -l3 * 0.2])

        return pts

    def generate_sample(
        self,
        gesture_class: str,
        session_type: str = "same_session",
    ) -> np.ndarray:
        """
        Generates a 21-landmark hand sample with domain characteristics:
        - 'same_session': Centered, fixed operational distance, minimal tilt.
        - 'cross_session': Arbitrary translation, large scale changes (distance), perspective rotation, sensor noise.
        """
        base_pose = self.generate_canonical_pose(gesture_class)

        # Small biometric jitter on joint positions
        joint_jitter = self.rng.normal(0.0, 0.003, base_pose.shape)
        pose = base_pose + joint_jitter

        if session_type == "same_session":
            # Controlled setup: hand roughly center of frame, scale ~ 1.0, tilt within +/- 8 degrees
            scale = self.rng.uniform(0.95, 1.05)
            rx = np.radians(self.rng.uniform(-5.0, 5.0))
            ry = np.radians(self.rng.uniform(-5.0, 5.0))
            rz = np.radians(self.rng.uniform(-8.0, 8.0))
            # Screen translation centered around (0.5, 0.5)
            tx = self.rng.uniform(0.46, 0.54)
            ty = self.rng.uniform(0.48, 0.56)
            tz = self.rng.uniform(-0.02, 0.02)
        else:
            # Independent cross-session setup: extreme operational diversity
            # Scale variation (distance shift from near 1.6 to far 0.5)
            scale = self.rng.uniform(0.45, 1.60)
            # 3D Tilt variation up to +/- 35 degrees
            rx = np.radians(self.rng.uniform(-25.0, 25.0))
            ry = np.radians(self.rng.uniform(-30.0, 30.0))
            rz = np.radians(self.rng.uniform(-35.0, 35.0))
            # Arbitrary spatial translation across viewport
            tx = self.rng.uniform(0.18, 0.82)
            ty = self.rng.uniform(0.20, 0.80)
            tz = self.rng.uniform(-0.10, 0.10)

        # Apply 3D Rotation
        R = self._rotation_matrix(rx, ry, rz)
        transformed = (pose @ R.T) * scale

        # Apply Translation
        transformed[:, 0] += tx
        transformed[:, 1] += ty
        transformed[:, 2] += tz

        # Monocular landmark regression noise
        noise_std = 0.003 if session_type == "same_session" else 0.008
        sensor_noise = self.rng.normal(0.0, noise_std, transformed.shape)
        final_landmarks = transformed + sensor_noise

        return final_landmarks.astype(np.float32)

    def generate_dataset(
        self,
        samples_per_class: int = 150,
        session_type: str = "same_session",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates complete dataset across all 4 gesture classes.
        Returns:
            landmarks_batch: np.ndarray of shape (N, 21, 3)
            labels: np.ndarray of string class names of shape (N,)
        """
        classes = ["Open_Palm", "Closed_Fist", "Index_Point", "Victory"]
        all_landmarks = []
        all_labels = []

        for cls_name in classes:
            for _ in range(samples_per_class):
                lm = self.generate_sample(cls_name, session_type=session_type)
                all_landmarks.append(lm)
                all_labels.append(cls_name)

        indices = np.arange(len(all_labels))
        self.rng.shuffle(indices)

        return np.array(all_landmarks, dtype=np.float32)[indices], np.array(all_labels)[indices]
