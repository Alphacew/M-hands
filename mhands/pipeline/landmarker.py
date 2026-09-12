"""
MediaPipe Hands Landmarker Engine.

Encapsulates the monocular hand keypoint inference pipeline with optimized
frame conversion, handedness classification, and landmark extraction.
"""

from typing import Optional, Tuple, Any, Dict
import cv2
import mediapipe as mp
import numpy as np

from mhands.core.geometry import landmarks_to_numpy


class HandLandmarkerEngine:
    """
    High-performance wrapper for MediaPipe Hands pipeline.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
        model_complexity: int = 1,
    ):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
        )

    def process(
        self,
        frame_bgr: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[str], float]:
        """
        Executes landmark inference on an input BGR frame.
        
        Returns:
            landmarks: (21, 3) float32 NumPy array or None if no hand detected.
            handedness: 'Right' or 'Left' (or None)
            confidence: float confidence score [0.0, 1.0]
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None, None, 0.0

        # Convert BGR to RGB (MediaPipe requires RGB)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Mark image as not writeable to pass by reference for speed
        frame_rgb.flags.writeable = False

        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return None, None, 0.0

        primary_hand_landmarks = results.multi_hand_landmarks[0]
        landmarks_np = landmarks_to_numpy(primary_hand_landmarks)

        handedness = "Right"
        confidence = 0.90

        if results.multi_handedness and len(results.multi_handedness) > 0:
            classification = results.multi_handedness[0].classification[0]
            handedness = classification.label
            confidence = float(classification.score)

        return landmarks_np, handedness, confidence

    def close(self) -> None:
        self.hands.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
