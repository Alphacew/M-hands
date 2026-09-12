"""
Unit Tests for Telekinetic Controller and Spring Forces.
"""

from pathlib import Path
import numpy as np
import pytest
from pymunk import Vec2d

from mhands.sandbox.physics_space import PhysicsSpace
from mhands.sandbox.telekinesis import TelekineticController
from mhands.pipeline.classifier import GestureClassifier
from mhands.pipeline.landmarker import HandData
from mhands.data.synthetic_generator import SyntheticHandGenerator


@pytest.fixture
def controller():
    ps = PhysicsSpace(width=800, height=600)
    # Load or train classifier
    model_path = Path("models/invariant_svm.joblib")
    clf = GestureClassifier.load(model_path)
    return TelekineticController(physics=ps, classifier=clf, pinch_threshold=0.40)


def test_telekinetic_spring_grip_and_release(controller):
    # Add a dynamic test block at (400, 300)
    target_b, _ = controller.physics.add_box(400, 300, 50, 50, mass=2.0)

    gen = SyntheticHandGenerator(random_seed=42)
    # Generate canonical Open Palm and force pinch coordinates
    lms = gen.generate_canonical_pose("Open_Palm")
    # Position index tip and thumb tip close together at (400, 300) -> Screen coords: (0.5, 0.5)
    lms[:, 0] = 0.5
    lms[:, 1] = 0.5

    hand = HandData(landmarks=lms, handedness="Right", confidence=0.95)

    # Frame 1: Hand pinching near block
    summary = controller.update_hand(hand, screen_w=800, screen_h=600, timestamp=0.0)

    assert controller.grips["Right"] is not None
    assert controller.grips["Right"].body == target_b
    assert summary["grip_active"] is True

    # Frame 2: Hand moves and releases pinch
    gen_open = gen.generate_sample("Open_Palm", "same_session")
    hand_open = HandData(landmarks=gen_open, handedness="Right", confidence=0.95)
    summary_open = controller.update_hand(hand_open, screen_w=800, screen_h=600, timestamp=0.05)

    # Spring severed
    assert controller.grips["Right"] is None
    assert summary_open["grip_active"] is False


def test_shockwave_expansion(controller):
    controller.active_shockwaves.clear()
    controller.active_shockwaves.append(
        from_sw := controller.active_shockwaves.__class__
    ) if False else None

    # Simulate shockwave update
    from mhands.sandbox.telekinesis import ActiveShockwave
    sw = ActiveShockwave(center=Vec2d(400, 300), radius=20, max_radius=200, intensity=500, duration=0.2)
    controller.active_shockwaves.append(sw)

    controller.update_effects(0.1)
    assert len(controller.active_shockwaves) == 1
    assert controller.active_shockwaves[0].radius > 20

    # Advance beyond duration
    controller.update_effects(0.15)
    assert len(controller.active_shockwaves) == 0
