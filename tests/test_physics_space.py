"""
Unit Tests for PhysicsSpace Simulation Engine.
"""

import pytest
import pymunk
from pymunk import Vec2d
from mhands.sandbox.physics_space import PhysicsSpace


def test_physics_space_initialization():
    ps = PhysicsSpace(width=800, height=600, gravity=(0, 900))
    assert ps.space.gravity == (0, 900)
    assert len(ps.boundaries) == 4


def test_add_box_and_circle():
    ps = PhysicsSpace(width=800, height=600)
    box_b, box_s = ps.add_box(400, 300, 50, 50, mass=2.0)
    assert box_b.body_type == pymunk.Body.DYNAMIC
    assert box_b.mass == 2.0

    circle_b, circle_s = ps.add_circle(200, 100, radius=20, mass=1.0)
    assert circle_b.body_type == pymunk.Body.DYNAMIC
    assert circle_s.radius == 20.0


def test_spatial_query_nearest():
    ps = PhysicsSpace(width=800, height=600)
    box_b, _ = ps.add_box(400, 300, 40, 40)

    # Query directly over box
    found = ps.query_nearest_body((405, 305), max_radius=50.0)
    assert found == box_b

    # Query far away
    not_found = ps.query_nearest_body((100, 100), max_radius=50.0)
    assert not_found is None


def test_stasis_freeze_and_resume():
    ps = PhysicsSpace(width=800, height=600)
    box_b, _ = ps.add_box(400, 300, 40, 40)
    box_b.velocity = Vec2d(150, -80)

    # Freeze
    ps.set_stasis(True)
    assert ps.is_stasis is True
    assert box_b.velocity == (0, 0)
    assert ps.space.gravity == (0, 0)

    # Unfreeze
    ps.set_stasis(False)
    assert ps.is_stasis is False
    assert box_b.velocity.x == 150
    assert box_b.velocity.y == -80
    assert ps.space.gravity == ps.default_gravity


def test_slow_motion_decay():
    ps = PhysicsSpace(width=800, height=600)
    ps.trigger_slow_motion(duration=0.5, speed_factor=0.2)
    assert ps.time_scale == 0.2
    assert ps.slo_mo_timer == 0.5

    # Advance 0.6 seconds
    ps.step(0.6)
    assert ps.time_scale == 1.0


def test_sandbox_visualizer_rendering():
    import numpy as np
    from pathlib import Path
    from mhands.sandbox.visualizer_2d import SandboxVisualizer
    from mhands.sandbox.telekinesis import TelekineticController
    from mhands.pipeline.classifier import GestureClassifier
    from mhands.pipeline.landmarker import HandData
    from mhands.data.synthetic_generator import SyntheticHandGenerator

    ps = PhysicsSpace(width=800, height=600)
    ps.add_box(400, 300, 50, 50, mass=2.0)

    clf = GestureClassifier.load(Path("models/invariant_svm.joblib"))
    ctrl = TelekineticController(physics=ps, classifier=clf)
    viz = SandboxVisualizer(width=800, height=600)

    gen = SyntheticHandGenerator(random_seed=10)
    lms = gen.generate_canonical_pose("Open_Palm")
    hands = [HandData(landmarks=lms, handedness="Right", confidence=0.98)]

    # Test AR mode transparent overlay with camera frame
    fake_cam = np.full((600, 800, 3), 120, dtype=np.uint8)
    rendered = viz.render(
        physics=ps,
        controller=ctrl,
        hands=hands,
        camera_frame=fake_cam,
        ar_mode=True,
        environment_name="Test_Env",
        fps=60.0,
    )

    assert rendered.shape == (600, 800, 3)
    assert rendered.dtype == np.uint8
    # Ensure visualizer drew onto the frame
    assert not np.array_equal(rendered, fake_cam)

