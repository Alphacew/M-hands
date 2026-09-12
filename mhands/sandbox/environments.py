"""
Pre-Built Physics Toyboxes & Interactive Environments.

Constructs 5 rich interactive sandbox scenarios:
1. Jenga_Castle: Masonry arches, keystones, structural columns, domino monoliths.
2. Ragdoll_Arena: Articulated humanoids on platforms with kinetic pendulums.
3. Fluid_Vat: Viscous particle nodes exhibiting splash dynamics.
4. ZeroG_Asteroids: Microgravity celestial playground with orbital bodies.
5. Seismic_Bridge: Structural truss bridge spanning a chasm under heavy load.
"""

from typing import List, Tuple
import numpy as np
import pymunk
from pymunk import Vec2d

from mhands.sandbox.physics_space import PhysicsSpace
from mhands.sandbox.ragdoll import Ragdoll


class SandboxEnvironments:
    """
    Spawns curated physical sandbox worlds.
    """

    @staticmethod
    def load_jenga_castle(physics: PhysicsSpace) -> str:
        """Constructs a towering destructible masonry fortress and Jenga tower."""
        physics.clear_dynamic_bodies()
        physics.space.gravity = (0, 900)
        w, h = float(physics.width), float(physics.height)

        # 1. Base Platform
        physics.add_box(w * 0.5, h - 25, w * 0.85, 20, is_static=True, color=(60, 60, 75))

        # 2. Central Arch & Keystone
        arch_x = w * 0.35
        # Pillars
        physics.add_box(arch_x - 60, h - 85, 30, 100, mass=15, color=(140, 180, 210))
        physics.add_box(arch_x + 60, h - 85, 30, 100, mass=15, color=(140, 180, 210))
        # Arch lintel
        physics.add_box(arch_x, h - 145, 170, 22, mass=10, color=(180, 210, 230))
        # Keystone & Top Tower
        physics.add_box(arch_x, h - 175, 45, 36, mass=6, color=(240, 120, 60))
        physics.add_box(arch_x - 45, h - 215, 24, 44, mass=4, color=(120, 160, 200))
        physics.add_box(arch_x + 45, h - 215, 24, 44, mass=4, color=(120, 160, 200))
        physics.add_box(arch_x, h - 245, 120, 16, mass=5, color=(160, 200, 220))

        # 3. Balanced Jenga Monolith (Right side)
        jenga_x = w * 0.70
        block_w, block_h = 100, 18
        for layer in range(12):
            ly = (h - 45) - layer * (block_h + 2)
            physics.add_box(jenga_x, ly, block_w, block_h, mass=3.0, color=(210, 160, 90))

        # 4. Domino Run (Left side)
        for i in range(8):
            dx = w * 0.10 + i * 26
            physics.add_box(dx, h - 60, 12, 50, mass=1.5, color=(220, 70, 70))

        return "Destructible Jenga Castle & Keystone Arch Loaded."

    @staticmethod
    def load_ragdoll_arena(physics: PhysicsSpace) -> str:
        """Spawns articulated ragdoll test dummies across multi-tier platforms."""
        physics.clear_dynamic_bodies()
        physics.space.gravity = (0, 850)
        w, h = float(physics.width), float(physics.height)

        # Platforms
        physics.add_box(w * 0.30, h * 0.55, 280, 18, is_static=True, color=(70, 80, 95))
        physics.add_box(w * 0.70, h * 0.40, 280, 18, is_static=True, color=(70, 80, 95))

        # Ragdoll 1 on left platform
        _ = Ragdoll(physics.space, origin_x=w * 0.30, origin_y=h * 0.38, scale=1.0, group_id=1)

        # Ragdoll 2 on right platform
        _ = Ragdoll(physics.space, origin_x=w * 0.70, origin_y=h * 0.22, scale=1.1, group_id=2)

        # Kinetic Pendulum Wrecking Ball
        pivot_pos = Vec2d(w * 0.50, h * 0.15)
        ball_pos = Vec2d(w * 0.50 + 120, h * 0.25)
        ball_b, ball_s = physics.add_circle(ball_pos.x, ball_pos.y, radius=24, mass=12, color=(240, 60, 60))
        rod = pymunk.PinJoint(physics.space.static_body, ball_b, pivot_pos, (0, 0))
        physics.space.add(rod)

        # Scatter dynamic obstacle crates
        for i in range(5):
            physics.add_box(w * 0.45 + i * 30, h - 55, 25, 25, mass=2.5, color=(140, 190, 240))

        return "Articulated Ragdoll Arena & Kinetic Pendulum Loaded."

    @staticmethod
    def load_fluid_vat(physics: PhysicsSpace) -> str:
        """Spawns 120+ viscous particle nodes exhibiting splash dynamics."""
        physics.clear_dynamic_bodies()
        physics.space.gravity = (0, 750)
        w, h = float(physics.width), float(physics.height)

        # Tank container
        vat_x = w * 0.50
        vat_w = 460
        vat_h = 280
        vat_y = h - vat_h / 2 - 20

        # Tank walls
        physics.add_box(vat_x - vat_w / 2, vat_y, 18, vat_h, is_static=True, color=(50, 70, 90))
        physics.add_box(vat_x + vat_w / 2, vat_y, 18, vat_h, is_static=True, color=(50, 70, 90))
        physics.add_box(vat_x, vat_y + vat_h / 2, vat_w, 18, is_static=True, color=(50, 70, 90))

        # Floating buoy crate
        physics.add_box(vat_x, vat_y - 40, 70, 35, mass=2.0, color=(240, 160, 40))

        # Particle fluid cluster
        p_radius = 8.5
        spacing = p_radius * 2.2
        cols = 14
        rows = 9
        start_x = vat_x - (cols * spacing) / 2
        start_y = vat_y + 10

        for r in range(rows):
            for c in range(cols):
                px = start_x + c * spacing + (r % 2) * (spacing / 2)
                py = start_y + r * spacing
                physics.add_circle(
                    px, py,
                    radius=p_radius,
                    mass=0.35,
                    friction=0.1,
                    elasticity=0.4,
                    color=(40, 180, 240),
                )

        return "Granular Viscous Fluid Vat Loaded."

    @staticmethod
    def load_zero_g_asteroids(physics: PhysicsSpace) -> str:
        """Spawns a microgravity orbital playground with floating asteroids."""
        physics.clear_dynamic_bodies()
        physics.space.gravity = (0, 0)  # Microgravity
        w, h = float(physics.width), float(physics.height)

        # Central Star / Anchor body
        center_x, center_y = w * 0.50, h * 0.50
        star_b, star_s = physics.add_circle(center_x, center_y, radius=38, mass=50, color=(255, 210, 40))

        # Orbital asteroids
        n_asteroids = 14
        orbit_r = 220.0
        for i in range(n_asteroids):
            angle = (i / n_asteroids) * 2 * np.pi
            ax = center_x + orbit_r * np.cos(angle)
            ay = center_y + orbit_r * np.sin(angle)
            # Tangential velocity for stable orbit
            v_mag = 120.0
            vx = -v_mag * np.sin(angle)
            vy = v_mag * np.cos(angle)

            radius = float(np.random.uniform(12.0, 22.0))
            mass = radius * 0.4
            ab, ashape = physics.add_circle(ax, ay, radius=radius, mass=mass, color=(140, 140, 160))
            ab.velocity = Vec2d(vx, vy)

        # Floating space debris blocks
        for _ in range(8):
            rx = np.random.uniform(w * 0.15, w * 0.85)
            ry = np.random.uniform(h * 0.15, h * 0.85)
            rw = np.random.uniform(20, 50)
            rh = np.random.uniform(20, 50)
            b, s = physics.add_box(rx, ry, rw, rh, mass=3.0, color=(100, 170, 220))
            b.angular_velocity = np.random.uniform(-2.0, 2.0)

        return "Zero-G Orbital Asteroid Field Loaded."

    @staticmethod
    def load_seismic_bridge(physics: PhysicsSpace) -> str:
        """Spawns a structural truss bridge spanning a central chasm."""
        physics.clear_dynamic_bodies()
        physics.space.gravity = (0, 900)
        w, h = float(physics.width), float(physics.height)

        # Chasm cliffs
        cliff_w = w * 0.28
        cliff_h = 240
        physics.add_box(cliff_w / 2, h - cliff_h / 2, cliff_w, cliff_h, is_static=True, color=(60, 65, 75))
        physics.add_box(w - cliff_w / 2, h - cliff_h / 2, cliff_w, cliff_h, is_static=True, color=(60, 65, 75))

        # Bridge segments connected by pivot joints
        span_start = cliff_w - 20
        span_end = w - cliff_w + 20
        n_segments = 7
        seg_len = (span_end - span_start) / n_segments
        bridge_y = h - cliff_h + 10

        prev_body = physics.space.static_body
        prev_anchor = Vec2d(span_start, bridge_y)

        for i in range(n_segments):
            seg_x = span_start + (i + 0.5) * seg_len
            seg_b, seg_s = physics.add_box(seg_x, bridge_y, seg_len, 14, mass=4.0, color=(180, 150, 90))

            # Joint with previous segment
            joint = pymunk.PivotJoint(prev_body, seg_b, prev_anchor, (-seg_len / 2, 0))
            physics.space.add(joint)

            prev_body = seg_b
            prev_anchor = (seg_len / 2, 0)

        # Anchor last segment to right cliff
        end_joint = pymunk.PivotJoint(prev_body, physics.space.static_body, (seg_len / 2, 0), (span_end, bridge_y))
        physics.space.add(end_joint)

        # Heavy rolling carts on top of bridge
        cart_b, cart_s = physics.add_box(span_start + 40, bridge_y - 30, 48, 24, mass=12.0, color=(240, 70, 70))
        wheel1_b, wheel1_s = physics.add_circle(span_start + 26, bridge_y - 14, radius=10, mass=2.0, color=(40, 40, 50))
        wheel2_b, wheel2_s = physics.add_circle(span_start + 54, bridge_y - 14, radius=10, mass=2.0, color=(40, 40, 50))
        physics.space.add(pymunk.PivotJoint(cart_b, wheel1_b, (-14, 16), (0, 0)))
        physics.space.add(pymunk.PivotJoint(cart_b, wheel2_b, (14, 16), (0, 0)))

        return "Seismic Truss Bridge Loaded."
