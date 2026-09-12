"""
Core geometric and signal processing modules for M-Hands.
"""

from mhands.core.geometry import (
    extract_raw_features,
    extract_invariant_features,
    HAND_LANDMARK_NAMES,
)
from mhands.core.filters import OneEuroFilter, MultiPointOneEuroFilter
from mhands.core.state_machine import InteractionStateMachine, SystemState, GestureEvent
from mhands.core.preprocessor import AdaptiveIlluminationNormalizer, BilateralHandMirror
from mhands.core.calibrator import DynamicAnatomicalCalibrator

__all__ = [
    "extract_raw_features",
    "extract_invariant_features",
    "HAND_LANDMARK_NAMES",
    "OneEuroFilter",
    "MultiPointOneEuroFilter",
    "InteractionStateMachine",
    "SystemState",
    "GestureEvent",
    "AdaptiveIlluminationNormalizer",
    "BilateralHandMirror",
    "DynamicAnatomicalCalibrator",
]
