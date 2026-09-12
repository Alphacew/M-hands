"""
Articulated Humanoid Ragdoll Generator.

Constructs realistic physical test dummies composed of 10 constrained rigid bodies
(head, torso, arms, legs) coupled via Chipmunk2D pivot and rotary limit joints.
"""

from typing import List, Tuple, Dict
import pymunk
from pymunk import Vec2d


class Ragdoll:
    """
    Articulated multi-joint humanoid dummy.
    """

    def __init__(
        self,
        space: pymunk.Space,
        origin_x: float,
        origin_y: float,
        scale: float = 1.0,
        group_id: int = 1,
    ):
        self.space = space
        self.group_id = group_id
        self.scale = scale
        self.bodies: List[pymunk.Body] = []
        self.shapes: List[pymunk.Shape] = []
        self.joints: List[pymunk.Constraint] = []

        self._build(origin_x, origin_y)

    def _build(self, ox: float, oy: float) -> None:
        s = self.scale
        filt = pymunk.ShapeFilter(group=self.group_id)

        # 1. Torso
        torso_w, torso_h = 32 * s, 64 * s
        torso_m = 10.0
        torso_b = pymunk.Body(torso_m, pymunk.moment_for_box(torso_m, (torso_w, torso_h)))
        torso_b.position = Vec2d(ox, oy)
        torso_s = pymunk.Poly.create_box(torso_b, (torso_w, torso_h))
        torso_s.filter = filt
        torso_s.friction = 0.8
        torso_s.color = (180, 100, 50)
        self.torso = torso_b

        # 2. Head
        head_r = 18 * s
        head_m = 3.0
        head_b = pymunk.Body(head_m, pymunk.moment_for_circle(head_m, 0, head_r))
        head_b.position = Vec2d(ox, oy - (torso_h / 2 + head_r + 4 * s))
        head_s = pymunk.Circle(head_b, head_r)
        head_s.filter = filt
        head_s.friction = 0.6
        head_s.color = (220, 180, 120)
        self.head = head_b

        # Neck Joint
        neck_pivot = pymunk.PivotJoint(torso_b, head_b, (0, -torso_h / 2), (0, head_r))
        neck_limit = pymunk.RotaryLimitJoint(torso_b, head_b, -0.4, 0.4)

        # 3. Limbs: Arms (Upper + Forearm)
        arm_w, arm_h = 12 * s, 34 * s
        arm_m = 2.0
        forearm_w, forearm_h = 10 * s, 30 * s
        forearm_m = 1.5

        # Left Arm
        l_upper_b = pymunk.Body(arm_m, pymunk.moment_for_box(arm_m, (arm_w, arm_h)))
        l_upper_b.position = Vec2d(ox - (torso_w / 2 + arm_w / 2), oy - 14 * s)
        l_upper_s = pymunk.Poly.create_box(l_upper_b, (arm_w, arm_h))
        l_upper_s.filter = filt
        l_upper_s.color = (160, 90, 40)

        l_fore_b = pymunk.Body(forearm_m, pymunk.moment_for_box(forearm_m, (forearm_w, forearm_h)))
        l_fore_b.position = Vec2d(ox - (torso_w / 2 + arm_w / 2), oy + 24 * s)
        l_fore_s = pymunk.Poly.create_box(l_fore_b, (forearm_w, forearm_h))
        l_fore_s.filter = filt
        l_fore_s.color = (210, 170, 110)

        l_shoulder = pymunk.PivotJoint(torso_b, l_upper_b, (-torso_w / 2, -18 * s), (0, -arm_h / 2))
        l_elbow = pymunk.PivotJoint(l_upper_b, l_fore_b, (0, arm_h / 2), (0, -forearm_h / 2))
        l_elbow_limit = pymunk.RotaryLimitJoint(l_upper_b, l_fore_b, -2.4, 0.2)

        # Right Arm
        r_upper_b = pymunk.Body(arm_m, pymunk.moment_for_box(arm_m, (arm_w, arm_h)))
        r_upper_b.position = Vec2d(ox + (torso_w / 2 + arm_w / 2), oy - 14 * s)
        r_upper_s = pymunk.Poly.create_box(r_upper_b, (arm_w, arm_h))
        r_upper_s.filter = filt
        r_upper_s.color = (160, 90, 40)

        r_fore_b = pymunk.Body(forearm_m, pymunk.moment_for_box(forearm_m, (forearm_w, forearm_h)))
        r_fore_b.position = Vec2d(ox + (torso_w / 2 + arm_w / 2), oy + 24 * s)
        r_fore_s = pymunk.Poly.create_box(r_fore_b, (forearm_w, forearm_h))
        r_fore_s.filter = filt
        r_fore_s.color = (210, 170, 110)

        r_shoulder = pymunk.PivotJoint(torso_b, r_upper_b, (torso_w / 2, -18 * s), (0, -arm_h / 2))
        r_elbow = pymunk.PivotJoint(r_upper_b, r_fore_b, (0, arm_h / 2), (0, -forearm_h / 2))
        r_elbow_limit = pymunk.RotaryLimitJoint(r_upper_b, r_fore_b, -0.2, 2.4)

        # 4. Legs (Thigh + Shin)
        thigh_w, thigh_h = 14 * s, 42 * s
        thigh_m = 3.5
        shin_w, shin_h = 12 * s, 38 * s
        shin_m = 2.5

        # Left Leg
        l_thigh_b = pymunk.Body(thigh_m, pymunk.moment_for_box(thigh_m, (thigh_w, thigh_h)))
        l_thigh_b.position = Vec2d(ox - 10 * s, oy + torso_h / 2 + thigh_h / 2)
        l_thigh_s = pymunk.Poly.create_box(l_thigh_b, (thigh_w, thigh_h))
        l_thigh_s.filter = filt
        l_thigh_s.color = (40, 70, 150)

        l_shin_b = pymunk.Body(shin_m, pymunk.moment_for_box(shin_m, (shin_w, shin_h)))
        l_shin_b.position = Vec2d(ox - 10 * s, oy + torso_h / 2 + thigh_h + shin_h / 2)
        l_shin_s = pymunk.Poly.create_box(l_shin_b, (shin_w, shin_h))
        l_shin_s.filter = filt
        l_shin_s.color = (30, 50, 110)

        l_hip = pymunk.PivotJoint(torso_b, l_thigh_b, (-10 * s, torso_h / 2), (0, -thigh_h / 2))
        l_hip_limit = pymunk.RotaryLimitJoint(torso_b, l_thigh_b, -1.2, 0.8)
        l_knee = pymunk.PivotJoint(l_thigh_b, l_shin_b, (0, thigh_h / 2), (0, -shin_h / 2))
        l_knee_limit = pymunk.RotaryLimitJoint(l_thigh_b, l_shin_b, 0.0, 2.4)

        # Right Leg
        r_thigh_b = pymunk.Body(thigh_m, pymunk.moment_for_box(thigh_m, (thigh_w, thigh_h)))
        r_thigh_b.position = Vec2d(ox + 10 * s, oy + torso_h / 2 + thigh_h / 2)
        r_thigh_s = pymunk.Poly.create_box(r_thigh_b, (thigh_w, thigh_h))
        r_thigh_s.filter = filt
        r_thigh_s.color = (40, 70, 150)

        r_shin_b = pymunk.Body(shin_m, pymunk.moment_for_box(shin_m, (shin_w, shin_h)))
        r_shin_b.position = Vec2d(ox + 10 * s, oy + torso_h / 2 + thigh_h + shin_h / 2)
        r_shin_s = pymunk.Poly.create_box(r_shin_b, (shin_w, shin_h))
        r_shin_s.filter = filt
        r_shin_s.color = (30, 50, 110)

        r_hip = pymunk.PivotJoint(torso_b, r_thigh_b, (10 * s, torso_h / 2), (0, -thigh_h / 2))
        r_hip_limit = pymunk.RotaryLimitJoint(torso_b, r_thigh_b, -0.8, 1.2)
        r_knee = pymunk.PivotJoint(r_thigh_b, r_shin_b, (0, thigh_h / 2), (0, -shin_h / 2))
        r_knee_limit = pymunk.RotaryLimitJoint(r_thigh_b, r_shin_b, 0.0, 2.4)

        self.bodies = [
            torso_b, head_b,
            l_upper_b, l_fore_b, r_upper_b, r_fore_b,
            l_thigh_b, l_shin_b, r_thigh_b, r_shin_b,
        ]
        self.shapes = [
            torso_s, head_s,
            l_upper_s, l_fore_s, r_upper_s, r_fore_s,
            l_thigh_s, l_shin_s, r_thigh_s, r_shin_s,
        ]
        self.joints = [
            neck_pivot, neck_limit,
            l_shoulder, l_elbow, l_elbow_limit,
            r_shoulder, r_elbow, r_elbow_limit,
            l_hip, l_hip_limit, l_knee, l_knee_limit,
            r_hip, r_hip_limit, r_knee, r_knee_limit,
        ]

        # Register into Pymunk space
        for b, s in zip(self.bodies, self.shapes):
            self.space.add(b, s)
        for j in self.joints:
            self.space.add(j)
