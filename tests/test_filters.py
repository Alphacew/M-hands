"""
Unit Tests for 1 Euro Filter and Temporal Damping.
"""

import numpy as np
import pytest

from mhands.core.filters import OneEuroFilter, MultiPointOneEuroFilter


def test_one_euro_filter_initialization():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.007)
    x0 = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    out0 = f.filter(x0, timestamp=0.0)
    assert np.allclose(out0, x0)


def test_jitter_smoothing_on_static_signal():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.007)
    np.random.seed(42)

    # Static signal around 10.0 with high-frequency noise
    t = 0.0
    dt = 0.016  # ~60 FPS
    raw_vals = []
    filtered_vals = []

    for _ in range(100):
        t += dt
        noisy = 10.0 + np.random.normal(0.0, 0.5)
        raw_vals.append(noisy)
        filtered = f.filter(np.array([noisy]), timestamp=t)[0]
        filtered_vals.append(filtered)

    # Filtered signal must have substantially lower standard deviation than raw
    std_raw = np.std(raw_vals[20:])
    std_flt = np.std(filtered_vals[20:])
    assert std_flt < std_raw * 0.5, f"Expected smoothing: std_flt={std_flt}, std_raw={std_raw}"


def test_multipoint_filter_shape_and_reset():
    mp_filter = MultiPointOneEuroFilter(mode="standard")
    lms = np.random.uniform(0.0, 1.0, (21, 3)).astype(np.float32)

    res = mp_filter.process(lms, timestamp=1.0)
    assert res.shape == (21, 3)
    assert np.allclose(res, lms)

    mp_filter.reset()
    assert mp_filter.filter.last_time is None
