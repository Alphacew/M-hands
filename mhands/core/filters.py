"""
Adaptive 1 Euro Filter and Temporal Damping Module.

Eliminates monocular landmark jitter during static postures while maintaining
zero-lag dynamic responsiveness. Includes specialized tremor-suppression presets
for motor conditions (e.g. Parkinsonian / essential tremor).
"""

import time
from typing import Optional, Union, Tuple
import numpy as np


class LowPassFilter:
    """First-order low-pass exponential filter supporting scalar or vectorized signals."""

    def __init__(self, alpha: Union[float, np.ndarray] = 1.0, initval: Optional[np.ndarray] = None):
        self.alpha = alpha
        self.y: Optional[np.ndarray] = None
        self.s: Optional[np.ndarray] = None
        if initval is not None:
            self.set_value(initval)

    def set_value(self, val: np.ndarray) -> None:
        self.y = np.array(val, dtype=np.float32, copy=True)
        self.s = np.array(val, dtype=np.float32, copy=True)

    def filter(self, value: np.ndarray, alpha: Optional[Union[float, np.ndarray]] = None) -> np.ndarray:
        if alpha is not None:
            self.alpha = alpha
        if self.y is None or self.s is None:
            self.set_value(value)
            return self.y
        self.y = np.array(value, dtype=np.float32)
        self.s = self.alpha * self.y + (1.0 - self.alpha) * self.s
        return self.s


class OneEuroFilter:
    """
    Adaptive 1 Euro Filter for continuous multi-dimensional signals.
    Dynamically adjusts cutoff frequency based on instantaneous signal velocity.

    fc = min_cutoff + beta * |dx/dt|
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time: Optional[float] = None

    def _compute_alpha(self, cutoff: Union[float, np.ndarray], dt: float) -> Union[float, np.ndarray]:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        """
        Filters input array x (1D or multi-dimensional array).
        If timestamp is None, system monotonic clock is sampled.
        """
        x = np.asarray(x, dtype=np.float32)
        if timestamp is None:
            timestamp = time.perf_counter()

        if self.last_time is None:
            self.last_time = timestamp
            self.x_filter.set_value(x)
            self.dx_filter.set_value(np.zeros_like(x))
            return x

        dt = max(1e-4, timestamp - self.last_time)
        self.last_time = timestamp

        # 1. Estimate instantaneous derivative
        prev_x = self.x_filter.s if self.x_filter.s is not None else x
        dx = (x - prev_x) / dt

        # 2. Filter derivative with constant d_cutoff
        alpha_d = self._compute_alpha(self.d_cutoff, dt)
        edx = self.dx_filter.filter(dx, alpha=alpha_d)

        # 3. Adaptively compute dynamic cutoff frequency
        cutoff = self.min_cutoff + self.beta * np.abs(edx)

        # 4. Filter signal with adaptive cutoff
        alpha = self._compute_alpha(cutoff, dt)
        return self.x_filter.filter(x, alpha=alpha)

    def reset(self) -> None:
        """Resets filter memory."""
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time = None


class MultiPointOneEuroFilter:
    """
    Dedicated temporal filter managing 21 3D hand keypoints (shape (21, 3)).
    Includes accessibility presets for motor tremors.
    """

    def __init__(
        self,
        mode: str = "standard",
        min_cutoff: Optional[float] = None,
        beta: Optional[float] = None,
        d_cutoff: float = 1.0,
    ):
        if mode == "tremor_suppression":
            # Preset for Parkinsonian or essential tremors (4-7 Hz damping)
            self.min_cutoff = min_cutoff if min_cutoff is not None else 0.4
            self.beta = beta if beta is not None else 0.002
        else:
            # Standard responsive interactive preset
            self.min_cutoff = min_cutoff if min_cutoff is not None else 1.0
            self.beta = beta if beta is not None else 0.007

        self.d_cutoff = d_cutoff
        self.filter = OneEuroFilter(
            min_cutoff=self.min_cutoff,
            beta=self.beta,
            d_cutoff=self.d_cutoff,
        )

    def process(self, landmarks: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        """
        Accepts landmarks array of shape (21, 3) and applies adaptive damping.
        Returns filtered array with identical shape (21, 3).
        """
        shape = landmarks.shape
        flat = landmarks.flatten()
        filtered_flat = self.filter.filter(flat, timestamp=timestamp)
        return filtered_flat.reshape(shape)

    def reset(self) -> None:
        self.filter.reset()
