"""
Deterministic Finite State Machine (FSM) with Confidence Hysteresis & Dwell Integration.

Eliminates the "Midas Touch" problem by requiring sustained confidence (P >= 0.85)
over a temporal dwell interval (t_dwell >= 250ms), and preventing rapid flickering
via dual-threshold hysteresis (release threshold P < 0.40).
"""

from dataclasses import dataclass
from enum import Enum
import time
from typing import Optional, Dict, Any


class SystemState(str, Enum):
    IDLE = "IDLE"
    DWELL_ENGAGING = "DWELL_ENGAGING"
    GESTURE_ACTIVE = "GESTURE_ACTIVE"
    ACTION_RELEASED = "ACTION_RELEASED"


@dataclass
class GestureEvent:
    state: SystemState
    gesture_class: Optional[str]
    confidence: float
    dwell_progress: float  # [0.0, 1.0] for HUD circular progress display
    is_triggered: bool     # True on first frame entering GESTURE_ACTIVE
    is_active: bool        # True throughout GESTURE_ACTIVE
    is_released: bool      # True on transition to ACTION_RELEASED


class InteractionStateMachine:
    """
    State machine implementing dual-threshold confidence hysteresis
    and temporal dwell integration.
    """

    def __init__(
        self,
        engage_threshold: float = 0.85,
        release_threshold: float = 0.40,
        dwell_time_ms: float = 250.0,
        release_timeout_ms: float = 150.0,
    ):
        self.engage_threshold = engage_threshold
        self.release_threshold = release_threshold
        self.dwell_time_sec = dwell_time_ms / 1000.0
        self.release_timeout_sec = release_timeout_ms / 1000.0

        self.current_state = SystemState.IDLE
        self.candidate_class: Optional[str] = None
        self.active_class: Optional[str] = None

        self.dwell_start_time: Optional[float] = None
        self.dropout_start_time: Optional[float] = None

    def update(
        self,
        predicted_class: Optional[str],
        confidence: float,
        timestamp: Optional[float] = None,
    ) -> GestureEvent:
        """
        Updates state machine with latest classifier prediction and confidence.
        Returns a GestureEvent representing current system interaction status.
        """
        if timestamp is None:
            timestamp = time.perf_counter()

        dwell_progress = 0.0
        is_triggered = False
        is_active = False
        is_released = False

        # State Transitions
        if self.current_state == SystemState.IDLE:
            if predicted_class is not None and confidence >= self.engage_threshold:
                # Candidate gesture recognized above engage threshold
                self.current_state = SystemState.DWELL_ENGAGING
                self.candidate_class = predicted_class
                self.dwell_start_time = timestamp
                dwell_progress = 0.0
            else:
                self.candidate_class = None
                self.dwell_start_time = None

        elif self.current_state == SystemState.DWELL_ENGAGING:
            if predicted_class == self.candidate_class and confidence >= self.engage_threshold:
                start_t = self.dwell_start_time if self.dwell_start_time is not None else timestamp
                elapsed = timestamp - start_t
                dwell_progress = min(1.0, elapsed / self.dwell_time_sec)

                if elapsed >= self.dwell_time_sec:
                    # Dwell requirement fulfilled
                    self.current_state = SystemState.GESTURE_ACTIVE
                    self.active_class = self.candidate_class
                    self.dropout_start_time = None
                    is_triggered = True
                    is_active = True
                    dwell_progress = 1.0
            else:
                # Confidence dropped or gesture changed before dwell elapsed -> cancel
                self.current_state = SystemState.IDLE
                self.candidate_class = None
                self.dwell_start_time = None
                dwell_progress = 0.0

        elif self.current_state == SystemState.GESTURE_ACTIVE:
            dwell_progress = 1.0
            is_active = True

            # Check if active class maintains above release threshold
            if predicted_class == self.active_class and confidence >= self.release_threshold:
                self.dropout_start_time = None
            else:
                # Dropout detected, enter grace period
                if self.dropout_start_time is None:
                    self.dropout_start_time = timestamp
                
                dropout_duration = timestamp - self.dropout_start_time
                if dropout_duration >= self.release_timeout_sec:
                    self.current_state = SystemState.ACTION_RELEASED
                    is_released = True
                    is_active = False

        elif self.current_state == SystemState.ACTION_RELEASED:
            # Action released, transition unconditionally to IDLE
            is_released = True
            self.current_state = SystemState.IDLE
            self.active_class = None
            self.candidate_class = None
            self.dwell_start_time = None
            self.dropout_start_time = None
            dwell_progress = 0.0

        displayed_class = self.active_class if is_active else self.candidate_class

        return GestureEvent(
            state=self.current_state,
            gesture_class=displayed_class,
            confidence=confidence,
            dwell_progress=dwell_progress,
            is_triggered=is_triggered,
            is_active=is_active,
            is_released=is_released,
        )

    def reset(self) -> None:
        self.current_state = SystemState.IDLE
        self.candidate_class = None
        self.active_class = None
        self.dwell_start_time = None
        self.dropout_start_time = None
