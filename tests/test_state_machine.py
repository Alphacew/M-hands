"""
Unit Tests for Interaction State Machine (Hysteresis & Dwell Integration).
"""

import pytest
from mhands.core.state_machine import InteractionStateMachine, SystemState


def test_state_machine_dwell_and_activation():
    fsm = InteractionStateMachine(
        engage_threshold=0.85,
        release_threshold=0.40,
        dwell_time_ms=250.0,
        release_timeout_ms=150.0,
    )

    t = 0.0
    # Frame 1: Low confidence -> IDLE
    ev = fsm.update("Open_Palm", 0.70, timestamp=t)
    assert ev.state == SystemState.IDLE
    assert not ev.is_active

    # Frame 2: High confidence (0.90) -> Enters DWELL_ENGAGING
    t += 0.05
    ev = fsm.update("Open_Palm", 0.90, timestamp=t)
    assert ev.state == SystemState.DWELL_ENGAGING
    assert not ev.is_active
    assert ev.dwell_progress < 1.0

    # Frame 3: Continue holding for 200 ms total -> still DWELL_ENGAGING
    t += 0.15
    ev = fsm.update("Open_Palm", 0.92, timestamp=t)
    assert ev.state == SystemState.DWELL_ENGAGING
    assert not ev.is_active

    # Frame 4: Crosses 250 ms threshold -> GESTURE_ACTIVE
    t += 0.10  # Total elapsed > 250 ms
    ev = fsm.update("Open_Palm", 0.92, timestamp=t)
    assert ev.state == SystemState.GESTURE_ACTIVE
    assert ev.is_active
    assert ev.is_triggered


def test_hysteresis_and_release():
    fsm = InteractionStateMachine(
        engage_threshold=0.85,
        release_threshold=0.40,
        dwell_time_ms=200.0,
        release_timeout_ms=100.0,
    )

    t = 0.0
    # Fast track to GESTURE_ACTIVE
    fsm.update("Victory", 0.95, timestamp=t)
    t += 0.25
    ev = fsm.update("Victory", 0.95, timestamp=t)
    assert ev.state == SystemState.GESTURE_ACTIVE

    # Confidence drops to 0.60 (below engage 0.85, but above release 0.40) -> Stays ACTIVE!
    t += 0.05
    ev = fsm.update("Victory", 0.60, timestamp=t)
    assert ev.state == SystemState.GESTURE_ACTIVE
    assert ev.is_active

    # Confidence drops below 0.40 -> starts release timeout
    t += 0.05
    ev = fsm.update("Victory", 0.30, timestamp=t)
    assert ev.state == SystemState.GESTURE_ACTIVE

    # Grace period passes (100 ms) -> ACTION_RELEASED
    t += 0.12
    ev = fsm.update("Victory", 0.30, timestamp=t)
    assert ev.state == SystemState.ACTION_RELEASED
    assert ev.is_released

    # Next frame reverts to IDLE
    t += 0.02
    ev = fsm.update(None, 0.0, timestamp=t)
    assert ev.state == SystemState.IDLE
