"""
Unit Tests for Polygon Cutter and Bisection Engine.
"""

import pytest
import pymunk
from pymunk import Vec2d
from mhands.sandbox.polygon_cutter import polygon_area_and_centroid, slice_convex_polygon


def test_polygon_area_and_centroid():
    # 40x40 Square centered at (100, 100)
    verts = [
        Vec2d(80, 80),
        Vec2d(120, 80),
        Vec2d(120, 120),
        Vec2d(80, 120),
    ]
    area, centroid = polygon_area_and_centroid(verts)
    assert abs(area - 1600.0) < 1e-3
    assert abs(centroid.x - 100.0) < 1e-3
    assert abs(centroid.y - 100.0) < 1e-3


def test_slice_convex_polygon_horizontal():
    space = pymunk.Space()
    # Box 100x100 at (200, 200)
    body = pymunk.Body(10.0, pymunk.moment_for_box(10.0, (100, 100)))
    body.position = Vec2d(200, 200)
    shape = pymunk.Poly.create_box(body, (100, 100))
    space.add(body, shape)

    # Slice through y = 200 (horizontal line from x=50 to x=350)
    p1 = Vec2d(50, 200)
    p2 = Vec2d(350, 200)

    res = slice_convex_polygon(shape, p1, p2, min_area=20.0)
    assert res is not None

    (body_a, shape_a), (body_b, shape_b), intersections = res

    # Two cut intersection points
    assert len(intersections) == 2

    # Mass conservation: mass_a + mass_b == original mass 10.0
    total_daughter_mass = body_a.mass + body_b.mass
    assert abs(total_daughter_mass - 10.0) < 1e-3

    # Equal halves
    assert abs(body_a.mass - 5.0) < 0.2
    assert abs(body_b.mass - 5.0) < 0.2


def test_slice_misses_polygon():
    space = pymunk.Space()
    body = pymunk.Body(1.0, 10.0)
    body.position = Vec2d(100, 100)
    shape = pymunk.Poly.create_box(body, (40, 40))
    space.add(body, shape)

    # Line far above polygon
    p1 = Vec2d(0, 10)
    p2 = Vec2d(200, 10)

    res = slice_convex_polygon(shape, p1, p2)
    assert res is None
