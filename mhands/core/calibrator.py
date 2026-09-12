"""
Dynamic Anatomical Calibration Engine.

Captures individual joint articulation limits (open hand to closed fist) during
a 2-second initialization sequence to normalize feature ranges across diverse
anatomical profiles and mobility constraints.
"""

import time
from typing import Optional, Dict, Tuple
import numpy as np


class DynamicAnatomicalCalibrator:
    """
    Tracks min and max flexion angles during user calibration phase
    and normalizes joint angles dynamically.
    """

    def __init__(self, duration_sec: float = 2.0):
        self.duration_sec = duration_sec
        self.start_time: Optional[float] = None
        self.is_calibrating = False
        self.is_calibrated = False

        # Baseline bounds for 8D feature vector
        self.min_bounds = np.full(8, np.inf, dtype=np.float32)
        self.max_bounds = np.full(8, -np.inf, dtype=np.float32)

    def start(self, timestamp: Optional[float] = None) -> None:
        self.start_time = timestamp if timestamp is not None else time.perf_counter()
        self.is_calibrating = True
        self.is_calibrated = False
        self.min_bounds = np.full(8, np.inf, dtype=np.float32)
        self.max_bounds = np.full(8, -np.inf, dtype=np.float32)

    def update(self, features_8d: np.ndarray, timestamp: Optional[float] = None) -> Tuple[bool, float]:
        """
        Updates calibration bounds with incoming 8D features.
        Returns (is_finished, progress_ratio).
        """
        if not self.is_calibrating:
            return self.is_calibrated, 1.0 if self.is_calibrated else 0.0

        if timestamp is None:
            timestamp = time.perf_counter()

        elapsed = timestamp - (self.start_time or timestamp)
        progress = min(1.0, elapsed / self.duration_sec)

        # Update min/max observed envelopes
        self.min_bounds = np.minimum(self.min_bounds, features_8d)
        self.max_bounds = np.maximum(self.max_bounds, features_8d)

        if elapsed >= self.duration_sec:
            self.is_calibrating = False
            self.is_calibrated = True
            # Protect against zero range
            span = self.max_bounds - self.min_bounds
            for idx in range(len(span)):
                if span[idx] < 1e-4:
                    self.max_bounds[idx] = self.min_bounds[idx] + 1.0
            return True, 1.0

        return False, progress

    def normalize(self, features_8d: np.ndarray) -> np.ndarray:
        """
        Applies min-max range normalization to 8D invariant features:
        theta_norm = (theta - theta_min) / (theta_max - theta_min)
        """
        if not self.is_calibrated:
            return features_8d
        span = np.maximum(self.max_bounds - self.min_bounds, 1e-5)
        normalized = (features_8d - self.min_bounds) / span
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)
