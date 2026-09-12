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
from mhands.core.geometry import INDEX_TIP, THUMB_TIP, WRIST


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


def test_pinch_strictly_prevents_laser(controller):
    # Construct a hand where index and thumb are pinched
    gen = SyntheticHandGenerator(random_seed=123)
    lms = gen.generate_canonical_pose("Open_Palm")
    # Force pinch
    lms[THUMB_TIP, :2] = lms[INDEX_TIP, :2] + np.array([0.005, 0.005])

    hand = HandData(landmarks=lms, handedness="Right", confidence=0.99)
    # Even if classifier were to return "Victory", is_pinching MUST block laser
    controller.classifier.classify_single = lambda phi: ("Victory", 0.99, {})

    controller.update_hand(hand, screen_w=800, screen_h=600, timestamp=0.0)

    # Laser blade must NOT be active while pinching
    assert controller.active_lasers["Right"] is None


def test_laser_blade_ignores_constrained_bodies(controller):
    import pymunk
    # Add a dynamic box with an attached PinJoint constraint
    target_b, _ = controller.physics.add_box(400, 300, 60, 60, mass=3.0)
    anchor_b = controller.physics.space.static_body
    joint = pymunk.PinJoint(target_b, anchor_b, (0, 0), (400, 300))
    controller.physics.space.add(joint)

    assert len(target_b.constraints) > 0

    # Generate Victory pose directed across (400, 300)
    gen = SyntheticHandGenerator(random_seed=456)
    lms = gen.generate_canonical_pose("Victory")
    # Position palm below the box so blade cuts across it
    cur_center = np.mean(lms[[0, 5, 9, 13, 17], :2], axis=0)
    lms[:, 0] += (0.5 - cur_center[0])
    lms[:, 1] += (0.6 - cur_center[1])

    hand = HandData(landmarks=lms, handedness="Right", confidence=0.99)
    controller.classifier.classify_single = lambda phi: ("Victory", 0.99, {})

    # Execute update - must safely ignore constrained body without crashing
    controller.update_hand(hand, screen_w=800, screen_h=600, timestamp=0.0)

    # Target body must still exist in space and have its constraint
    assert target_b in controller.physics.space.bodies
    assert joint in controller.physics.space.constraints


def test_laser_blade_slice_cooldown(controller):
    # Add two dynamic boxes in the cut path
    b1, _ = controller.physics.add_box(400, 300, 60, 60, mass=3.0)
    b2, _ = controller.physics.add_box(400, 200, 60, 60, mass=3.0)

    gen = SyntheticHandGenerator(random_seed=789)
    lms = gen.generate_canonical_pose("Victory")
    cur_center = np.mean(lms[[0, 5, 9, 13, 17], :2], axis=0)
    lms[:, 0] += (0.5 - cur_center[0])
    lms[:, 1] += (0.7 - cur_center[1])

    hand = HandData(landmarks=lms, handedness="Right", confidence=0.99)
    controller.classifier.classify_single = lambda phi: ("Victory", 0.99, {})

    # Trigger slice at t=0.0
    controller.update_hand(hand, screen_w=800, screen_h=600, timestamp=0.0)
    t1 = controller.last_slice_time
    assert t1 == 0.0

    # Trigger slice at t=0.1 (within 0.35s cooldown) -> should be blocked
    controller.update_hand(hand, screen_w=800, screen_h=600, timestamp=0.1)
    assert controller.last_slice_time == t1

    # Trigger slice after cooldown expires at t=0.5
    controller.update_hand(hand, screen_w=800, screen_h=600, timestamp=0.5)
    assert controller.last_slice_time == 0.5


def test_closed_fist_attracts_all_blocks_across_chamber(controller):
    # Add dynamic blocks in 4 distinct corners of the room
    b_nw, _ = controller.physics.add_box(100, 100, 40, 40, mass=2.0)
    b_ne, _ = controller.physics.add_box(700, 100, 40, 40, mass=2.0)
    b_sw, _ = controller.physics.add_box(100, 500, 40, 40, mass=2.0)
    b_se, _ = controller.physics.add_box(700, 500, 40, 40, mass=2.0)

    # Hand at center: (400, 300) -> (0.5, 0.5)
    gen = SyntheticHandGenerator(random_seed=111)
    lms = gen.generate_canonical_pose("Closed_Fist")
    cur_center = np.mean(lms[[0, 5, 9, 13, 17], :2], axis=0)
    lms[:, 0] += (0.5 - cur_center[0])
    lms[:, 1] += (0.5 - cur_center[1])

    hand = HandData(landmarks=lms, handedness="Right", confidence=0.98)
    summary = controller.update_hand(hand, screen_w=800, screen_h=600, timestamp=0.0)

    assert controller.is_singularity_active is True
    assert summary["singularity_active"] is True

    # Advance one simulation step
    controller.physics.step(0.016)

    # Verify every dynamic body across the entire chamber has moved/accelerated towards center
    center = Vec2d(400, 300)
    for b in [b_nw, b_ne, b_sw, b_se]:
        # Vector towards center should have positive dot product with velocity
        diff = center - b.position
        assert b.velocity.dot(diff) > 0.0, f"Body at {b.position} failed to accelerate toward {center}"


def test_thumbs_up_summons_on_screen_guide(controller):
    # Build Thumbs-Up pose: 4 fingers curled, thumb extended pointing upward
    gen = SyntheticHandGenerator(random_seed=222)
    lms = gen.generate_canonical_pose("Closed_Fist")
    # Position in center (screen coords: 0.5, 0.5)
    cur_center = np.mean(lms[[0, 5, 9, 13, 17], :2], axis=0)
    lms[:, 0] += (0.5 - cur_center[0])
    lms[:, 1] += (0.5 - cur_center[1])

    # Extend thumb pointing upward (-y in screen coordinates)
    lms[THUMB_TIP] = lms[WRIST] + np.array([-0.04, -0.16, 0.0])
    lms[2] = lms[WRIST] + np.array([-0.03, -0.06, 0.0])  # THUMB_MCP

    hand_tu = HandData(landmarks=lms, handedness="Right", confidence=0.99)

    # 1. Transient exposure (e.g. 0.10s) must NOT trigger guide (sensitivity protection)
    controller.update_hand(hand_tu, screen_w=800, screen_h=600, timestamp=0.0)
    controller.update_hand(hand_tu, screen_w=800, screen_h=600, timestamp=0.10)
    assert controller.is_guide_active is False, "Guide must not trigger on transient hand posture"

    # 2. Sustained Thumbs-Up (>= 0.25s) activates guide
    summary = controller.update_hand(hand_tu, screen_w=800, screen_h=600, timestamp=0.30)
    assert controller.is_guide_active is True
    assert summary["guide_active"] is True
    assert summary["gesture"] == "Thumbs_Up"

    # 3. Now open hand into Open Palm -> Guide dismisses IMMEDIATELY on frame 1
    lms_open = gen.generate_canonical_pose("Open_Palm")
    hand_open = HandData(landmarks=lms_open, handedness="Right", confidence=0.99)
    summary_open = controller.update_hand(hand_open, screen_w=800, screen_h=600, timestamp=0.35)

    assert controller.is_guide_active is False, "Guide must dismiss immediately in the very first frame after release"
    assert summary_open["guide_active"] is False


def test_absent_hand_immediately_dismisses_guide(controller):
    # Activate guide with sustained Thumbs-Up
    gen = SyntheticHandGenerator(random_seed=333)
    lms = gen.generate_canonical_pose("Closed_Fist")
    cur_center = np.mean(lms[[0, 5, 9, 13, 17], :2], axis=0)
    lms[:, 0] += (0.5 - cur_center[0])
    lms[:, 1] += (0.5 - cur_center[1])
    lms[THUMB_TIP] = lms[WRIST] + np.array([-0.04, -0.16, 0.0])
    lms[2] = lms[WRIST] + np.array([-0.03, -0.06, 0.0])

    hand_tu = HandData(landmarks=lms, handedness="Right", confidence=0.99)
    controller.update_hands([hand_tu], screen_w=800, screen_h=600, timestamp=0.0)
    controller.update_hands([hand_tu], screen_w=800, screen_h=600, timestamp=0.30)
    assert controller.is_guide_active is True

    # User drops their hand out of camera view (hands list is empty)
    controller.update_hands([], screen_w=800, screen_h=600, timestamp=0.35)
    assert controller.is_guide_active is False, "Guide must dismiss immediately when hand drops out of frame"


def test_natural_human_pinch_with_curled_fingers(controller):
    # Human pinching index and thumb while middle, ring, pinky are naturally curled into palm
    target_b, _ = controller.physics.add_box(400, 300, 50, 50, mass=2.0)
    gen = SyntheticHandGenerator(random_seed=123)
    lms = gen.generate_canonical_pose("Index_Point")

    # Position index tip and thumb tip touching near block at (400, 300)
    cur_center = np.mean(lms[[0, 5, 9, 13, 17], :2], axis=0)
    lms[:, 0] += (0.5 - cur_center[0])
    lms[:, 1] += (0.5 - cur_center[1])
    lms[THUMB_TIP, :2] = lms[INDEX_TIP, :2] + np.array([0.01, 0.01])

    hand = HandData(landmarks=lms, handedness="Right", confidence=0.95)
    summary = controller.update_hand(hand, screen_w=800, screen_h=600, timestamp=0.0)

    # Must successfully pinch and grip target body despite curled middle/ring/pinky fingers
    assert summary["is_pinching"] is True
    assert controller.grips["Right"] is not None
    assert controller.grips["Right"].body == target_b


def test_pinch_release_hysteresis(controller):
    # Verify grip latches at <= 0.45 and stays held up to 0.58 aperture before releasing
    target_b, _ = controller.physics.add_box(400, 300, 50, 50, mass=2.0)
    gen = SyntheticHandGenerator(random_seed=456)
    lms = gen.generate_canonical_pose("Index_Point")
    cur_center = np.mean(lms[[0, 5, 9, 13, 17], :2], axis=0)
    lms[:, 0] += (0.5 - cur_center[0])
    lms[:, 1] += (0.5 - cur_center[1])

    # 1. Close pinch (aperture ~0.10) -> Latches grip
    lms[THUMB_TIP, :2] = lms[INDEX_TIP, :2] + np.array([0.01, 0.01])
    hand = HandData(landmarks=lms, handedness="Right", confidence=0.95)
    s1 = controller.update_hand(hand, screen_w=800, screen_h=600, timestamp=0.0)
    assert s1["is_pinching"] is True
    assert controller.grips["Right"] is not None

    # 2. Relax pinch slightly (aperture ~0.50, above initial threshold 0.45, but under release threshold 0.58)
    lms_relaxed = lms.copy()
    d_ref = np.linalg.norm(lms_relaxed[9] - lms_relaxed[0])
    lms_relaxed[THUMB_TIP, :2] = lms_relaxed[INDEX_TIP, :2] + np.array([0.50 * d_ref, 0.0])
    hand_rel = HandData(landmarks=lms_relaxed, handedness="Right", confidence=0.95)
    s2 = controller.update_hand(hand_rel, screen_w=800, screen_h=600, timestamp=0.05)
    assert s2["is_pinching"] is True
    assert controller.grips["Right"] is not None, "Hysteresis must maintain grip while aperture < 0.58"

    # 3. Open fingers wide (aperture ~0.70) -> Sever grip & fling
    lms_wide = lms.copy()
    lms_wide[THUMB_TIP, :2] = lms_wide[INDEX_TIP, :2] + np.array([0.70 * d_ref, 0.0])
    hand_wide = HandData(landmarks=lms_wide, handedness="Right", confidence=0.95)
    s3 = controller.update_hand(hand_wide, screen_w=800, screen_h=600, timestamp=0.10)
    assert s3["is_pinching"] is False
    assert controller.grips["Right"] is None, "Grip must release when aperture exceeds release threshold"


