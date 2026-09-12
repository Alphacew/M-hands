"""
MediaPipe Hands Landmarker Engine.

Encapsulates the monocular hand keypoint inference pipeline with optimized
frame conversion, handedness classification, and landmark extraction.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict, List
import cv2
import mediapipe as mp
import numpy as np

from mhands.core.geometry import landmarks_to_numpy


@dataclass
class HandData:
    landmarks: np.ndarray  # (21, 3)
    handedness: str        # 'Right' or 'Left'
    confidence: float      # [0.0, 1.0]


class HandLandmarkerEngine:
    """
    High-performance wrapper for MediaPipe Hands pipeline supporting single and dual hands.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.65,
        min_tracking_confidence: float = 0.55,
        model_complexity: int = 1,
    ):
        self.max_num_hands = max_num_hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
        )

    def process_multi(self, frame_bgr: np.ndarray) -> List[HandData]:
        """
        Executes landmark inference, returning all detected hands (up to max_num_hands).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        results = self.hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return []

        hands_list = []
        for i, lm_container in enumerate(results.multi_hand_landmarks):
            lms_np = landmarks_to_numpy(lm_container)
            handedness = "Right"
            confidence = 0.90

            if results.multi_handedness and i < len(results.multi_handedness):
                cls_info = results.multi_handedness[i].classification[0]
                handedness = cls_info.label
                confidence = float(cls_info.score)

            hands_list.append(HandData(landmarks=lms_np, handedness=handedness, confidence=confidence))

        return hands_list

    def process(
        self,
        frame_bgr: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[str], float]:
        """
        Backward compatible single-hand extraction (returns primary detected hand).
        """
        hands = self.process_multi(frame_bgr)
        if not hands:
            return None, None, 0.0
        primary = hands[0]
        return primary.landmarks, primary.handedness, primary.confidence

    def close(self) -> None:
        self.hands.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
