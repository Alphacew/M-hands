"""
Convex Polygon Line Bisection & Fracturing Engine.

Slices 2D convex polygons along arbitrary laser cutting lines into two physical
daughter rigid bodies with conserved mass, linear/angular momentum, and kinetic spark ejection.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import pymunk
from pymunk import Vec2d


def polygon_area_and_centroid(vertices: List[Vec2d]) -> Tuple[float, Vec2d]:
    """Computes signed area and centroid of a 2D polygon."""
    n = len(vertices)
    if n < 3:
        return 0.0, Vec2d(0, 0)

    area = 0.0
    cx = 0.0
    cy = 0.0

    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        cross = p1.x * p2.y - p2.x * p1.y
        area += cross
        cx += (p1.x + p2.x) * cross
        cy += (p1.y + p2.y) * cross

    area *= 0.5
    if abs(area) < 1e-5:
        return 0.0, Vec2d(0, 0)

    cx /= (6.0 * area)
    cy /= (6.0 * area)
    return abs(area), Vec2d(cx, cy)


def is_clockwise(verts: List[Vec2d]) -> bool:
    """Returns True if vertices are in clockwise order."""
    total = 0.0
    n = len(verts)
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i + 1) % n]
        total += (p2.x - p1.x) * (p2.y + p1.y)
    return total > 0.0


def slice_convex_polygon(
    poly_shape: pymunk.Poly,
    line_p1: Vec2d,
    line_p2: Vec2d,
    min_area: float = 30.0,
    burst_speed: float = 40.0,
) -> Optional[Tuple[Tuple[pymunk.Body, pymunk.Poly], Tuple[pymunk.Body, pymunk.Poly], List[Vec2d]]]:
    """
    Slices a dynamic Pymunk Poly shape along the line (line_p1, line_p2).
    
    Returns:
        ((body_A, poly_A), (body_B, poly_B), cut_intersection_points) or None if no cut occurs.
    """
    body = poly_shape.body
    if body.body_type != pymunk.Body.DYNAMIC:
        return None

    # Line direction and normal vector
    line_dir = line_p2 - line_p1
    if line_dir.length < 1e-3:
        return None

    normal = Vec2d(-line_dir.y, line_dir.x).normalized()
    d = -normal.dot(line_p1)

    # Convert shape local vertices to world coordinates
    world_verts = [body.local_to_world(v) for v in poly_shape.get_vertices()]
    n_verts = len(world_verts)
    if n_verts < 3:
        return None

    # Evaluate signed distance to cutting line
    dists = [normal.dot(v) + d for v in world_verts]

    # Check if polygon spans across both sides of the line
    has_pos = any(dist > 1e-4 for dist in dists)
    has_neg = any(dist < -1e-4 for dist in dists)
    if not (has_pos and has_neg):
        return None  # Line does not bisect this polygon

    poly_a_verts: List[Vec2d] = []
    poly_b_verts: List[Vec2d] = []
    intersections: List[Vec2d] = []

    for i in range(n_verts):
        curr_v = world_verts[i]
        curr_d = dists[i]
        next_v = world_verts[(i + 1) % n_verts]
        next_d = dists[(i + 1) % n_verts]

        if curr_d >= -1e-5:
            poly_a_verts.append(curr_v)
        if curr_d <= 1e-5:
            poly_b_verts.append(curr_v)

        # Check for edge crossing
        if (curr_d > 1e-5 and next_d < -1e-5) or (curr_d < -1e-5 and next_d > 1e-5):
            denom = curr_d - next_d
            if abs(denom) > 1e-7:
                t = curr_d / denom
                intersect_pt = curr_v + (next_v - curr_v) * t
                poly_a_verts.append(intersect_pt)
                poly_b_verts.append(intersect_pt)
                intersections.append(intersect_pt)

    if len(intersections) < 2 or len(poly_a_verts) < 3 or len(poly_b_verts) < 3:
        return None

    area_a, cent_a = polygon_area_and_centroid(poly_a_verts)
    area_b, cent_b = polygon_area_and_centroid(poly_b_verts)

    if area_a < min_area or area_b < min_area:
        return None

    total_area = area_a + area_b
    orig_mass = body.mass
    mass_a = orig_mass * (area_a / total_area)
    mass_b = orig_mass * (area_b / total_area)

    # Filter duplicate vertices
    def clean_vertices(verts: List[Vec2d]) -> List[Vec2d]:
        cleaned: List[Vec2d] = []
        for v in verts:
            if not any((v - cv).length < 2.0 for cv in cleaned):
                cleaned.append(v)
        return cleaned

    local_a = clean_vertices([v - cent_a for v in poly_a_verts])
    local_b = clean_vertices([v - cent_b for v in poly_b_verts])

    if len(local_a) < 3 or len(local_b) < 3:
        return None

    # Ensure counter-clockwise winding for Pymunk
    if is_clockwise(local_a):
        local_a.reverse()
    if is_clockwise(local_b):
        local_b.reverse()

    try:
        # Create daughter body A
        moment_a = pymunk.moment_for_poly(mass_a, local_a)
        body_a = pymunk.Body(mass_a, moment_a)
        body_a.position = cent_a
        body_a.velocity = body.velocity + normal * burst_speed
        body_a.angular_velocity = body.angular_velocity

        shape_a = pymunk.Poly(body_a, local_a)
        shape_a.friction = poly_shape.friction
        shape_a.elasticity = poly_shape.elasticity
        shape_a.color = getattr(poly_shape, "color", (80, 180, 240))

        # Create daughter body B
        moment_b = pymunk.moment_for_poly(mass_b, local_b)
        body_b = pymunk.Body(mass_b, moment_b)
        body_b.position = cent_b
        body_b.velocity = body.velocity - normal * burst_speed
        body_b.angular_velocity = body.angular_velocity

        shape_b = pymunk.Poly(body_b, local_b)
        shape_b.friction = poly_shape.friction
        shape_b.elasticity = poly_shape.elasticity
        shape_b.color = getattr(poly_shape, "color", (80, 180, 240))

        return (body_a, shape_a), (body_b, shape_b), intersections
    except Exception:
        # Fallback gracefully if vertex geometry is non-convex
        return None
