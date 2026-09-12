"""
Pymunk 2D Physics Space Engine.

High-performance physical simulation managing rigid bodies, boundary constraints,
sub-stepped integration (120 Hz), slow-motion dilation, spatial stasis freeze,
and kinetic energy monitoring.
"""

from typing import List, Tuple, Optional, Dict, Any
import time
import numpy as np
import pymunk
from pymunk import Vec2d


class PhysicsSpace:
    """
    Encapsulates the 2D Chipmunk physics universe for the Jedi Sandbox.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        gravity: Tuple[float, float] = (0.0, 900.0),
        sub_steps: int = 2,
    ):
        self.width = width
        self.height = height
        self.default_gravity = gravity
        self.sub_steps = sub_steps

        self.space = pymunk.Space()
        self.space.gravity = gravity
        self.space.damping = 0.98

        self.time_scale = 1.0
        self.slo_mo_timer = 0.0
        self.is_stasis = False
        self._stasis_cache: Dict[pymunk.Body, Tuple[Vec2d, float]] = {}

        self.boundaries: List[pymunk.Shape] = []
        self._build_boundaries()

    def _build_boundaries(self, thickness: float = 40.0) -> None:
        """Constructs outer barrier walls."""
        w, h = float(self.width), float(self.height)
        t = thickness

        # Floor, Left Wall, Right Wall, Ceiling
        segments = [
            ((-t, h), (w + t, h)),        # Floor
            ((0, -t), (0, h + t)),        # Left Wall
            ((w, -t), (w, h + t)),        # Right Wall
            ((-t, 0), (w + t, 0)),        # Ceiling
        ]

        for p1, p2 in segments:
            seg = pymunk.Segment(self.space.static_body, p1, p2, t / 2)
            seg.elasticity = 0.5
            seg.friction = 0.7
            seg.filter = pymunk.ShapeFilter(categories=0x1)
            self.space.add(seg)
            self.boundaries.append(seg)

    def add_box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        mass: float = 2.0,
        friction: float = 0.7,
        elasticity: float = 0.3,
        color: Tuple[int, int, int] = (80, 180, 240),
        is_static: bool = False,
    ) -> Tuple[pymunk.Body, pymunk.Poly]:
        """Creates and registers a dynamic rectangular box."""
        if is_static:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = Vec2d(x, y)
        else:
            moment = pymunk.moment_for_box(mass, (w, h))
            body = pymunk.Body(mass, moment)
            body.position = Vec2d(x, y)

        poly = pymunk.Poly.create_box(body, (w, h))
        poly.friction = friction
        poly.elasticity = elasticity
        poly.color = color

        if is_static:
            self.space.add(body, poly)
        else:
            self.space.add(body, poly)

        return body, poly

    def add_circle(
        self,
        x: float,
        y: float,
        radius: float = 15.0,
        mass: float = 1.0,
        friction: float = 0.5,
        elasticity: float = 0.7,
        color: Tuple[int, int, int] = (250, 180, 40),
    ) -> Tuple[pymunk.Body, pymunk.Circle]:
        """Creates and registers a dynamic circular body."""
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)
        body.position = Vec2d(x, y)

        circle = pymunk.Circle(body, radius)
        circle.friction = friction
        circle.elasticity = elasticity
        circle.color = color

        self.space.add(body, circle)
        return body, circle

    def step(self, dt: float = 1.0 / 60.0) -> None:
        """
        Advances the simulation by dt, modulated by slow-mo time dilation.
        """
        if self.is_stasis:
            return  # Physical simulation frozen

        # Manage dynamic slo-mo timer
        if self.slo_mo_timer > 0.0:
            self.slo_mo_timer -= dt
            if self.slo_mo_timer <= 0.0:
                self.time_scale = 1.0

        effective_dt = dt * self.time_scale
        sub_dt = effective_dt / float(self.sub_steps)

        for _ in range(self.sub_steps):
            self.space.step(sub_dt)

    def trigger_slow_motion(self, duration: float = 2.5, speed_factor: float = 0.2) -> None:
        """Initiates dramatic slow-motion destruction."""
        self.time_scale = speed_factor
        self.slo_mo_timer = duration

    def set_stasis(self, enable: bool) -> None:
        """
        Freezes or unfreezes all dynamic rigid bodies in space.
        """
        if enable == self.is_stasis:
            return

        self.is_stasis = enable
        if enable:
            self._stasis_cache.clear()
            for body in self.space.bodies:
                if body.body_type == pymunk.Body.DYNAMIC:
                    self._stasis_cache[body] = (body.velocity, body.angular_velocity)
                    body.velocity = Vec2d(0, 0)
                    body.angular_velocity = 0.0
            self.space.gravity = (0.0, 0.0)
        else:
            for body, (vel, avel) in self._stasis_cache.items():
                if body in self.space.bodies and body.body_type == pymunk.Body.DYNAMIC:
                    body.velocity = vel
                    body.angular_velocity = avel
            self._stasis_cache.clear()
            self.space.gravity = self.default_gravity

    def query_nearest_body(self, point: Tuple[float, float], max_radius: float = 80.0) -> Optional[pymunk.Body]:
        """
        Finds the nearest non-static body within max_radius of point.
        """
        p = Vec2d(point[0], point[1])
        info = self.space.point_query_nearest(p, max_radius, pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS()))
        if info and info.shape and info.shape.body and info.shape.body.body_type == pymunk.Body.DYNAMIC:
            return info.shape.body
        return None

    def query_bodies_in_radius(self, point: Tuple[float, float], radius: float) -> List[pymunk.Body]:
        """Returns all dynamic bodies located within a given circle."""
        p = Vec2d(point[0], point[1])
        queries = self.space.point_query(p, radius, pymunk.ShapeFilter())
        bodies = set()
        for q in queries:
            if q.shape and q.shape.body and q.shape.body.body_type == pymunk.Body.DYNAMIC:
                bodies.add(q.shape.body)
        return list(bodies)

    def compute_total_kinetic_energy(self) -> float:
        """Computes system-wide kinetic energy for collapse detection."""
        ke = 0.0
        for body in self.space.bodies:
            if body.body_type == pymunk.Body.DYNAMIC:
                v_sq = body.velocity.length_squared
                ke += 0.5 * body.mass * v_sq
        return ke

    def clear_dynamic_bodies(self) -> None:
        """Removes all dynamic bodies and joints, keeping boundaries."""
        for c in list(self.space.constraints):
            self.space.remove(c)
        for b in list(self.space.bodies):
            if b.body_type != pymunk.Body.STATIC:
                for s in list(b.shapes):
                    self.space.remove(s)
                self.space.remove(b)
