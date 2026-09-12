"""
Unit Tests for Geometrical Invariance Engine.
"""

import numpy as np
import pytest

from mhands.core.geometry import (
    extract_raw_features,
    extract_invariant_features,
    verify_geometric_invariance,
)
from mhands.data.synthetic_generator import SyntheticHandGenerator


@pytest.fixture
def sample_hand():
    generator = SyntheticHandGenerator(random_seed=42)
    return generator.generate_canonical_pose("Open_Palm")


def test_raw_features_shape(sample_hand):
    raw = extract_raw_features(sample_hand)
    assert raw.shape == (63,)
    assert raw.dtype == np.float32


def test_invariant_features_shape(sample_hand):
    inv = extract_invariant_features(sample_hand)
    assert inv.shape == (8,)
    assert inv.dtype == np.float32


def test_translation_invariance(sample_hand):
    phi_base = extract_invariant_features(sample_hand)

    # Shift by large arbitrary 3D translation vector
    shifts = [
        np.array([10.0, -5.0, 3.0], dtype=np.float32),
        np.array([-50.0, 100.0, -20.0], dtype=np.float32),
        np.array([0.5, 0.5, 0.0], dtype=np.float32),
    ]

    for t in shifts:
        shifted_hand = sample_hand + t
        phi_shifted = extract_invariant_features(shifted_hand)
        diff = np.linalg.norm(phi_base - phi_shifted)
        assert diff < 1e-4, f"Translation invariance violated: diff={diff}"


def test_scale_invariance(sample_hand):
    phi_base = extract_invariant_features(sample_hand)

    scales = [0.25, 0.5, 1.8, 3.5, 10.0]
    for s in scales:
        scaled_hand = sample_hand * s
        phi_scaled = extract_invariant_features(scaled_hand)
        diff = np.linalg.norm(phi_base - phi_scaled)
        assert diff < 1e-4, f"Scale invariance violated for scale={s}: diff={diff}"


def test_zero_division_protection():
    # Degenerate all-zero landmarks
    zeros = np.zeros((21, 3), dtype=np.float32)
    phi = extract_invariant_features(zeros)
    assert phi.shape == (8,)
    assert not np.isnan(phi).any()
    assert not np.isinf(phi).any()


def test_verification_helper(sample_hand):
    result = verify_geometric_invariance(sample_hand)
    assert result["translation_invariant"] is True
    assert result["scale_invariant"] is True
    assert result["affine_invariant"] is True
