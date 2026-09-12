"""
Interactive Webcam Gesture Data Collector.

Provides guided multi-session recording to capture physical hand samples
under both controlled and unconstrained conditions.
"""

import time
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

from mhands.pipeline.landmarker import HandLandmarkerEngine
from mhands.pipeline.visualizer import COLOR_CYAN, COLOR_EMERALD, COLOR_WHITE, COLOR_BG_DARK


class GestureDataCollector:
    """
    Guides the user through capturing samples for each gesture class via webcam.
    """

    def __init__(
        self,
        classes: Optional[List[str]] = None,
        samples_per_class: int = 100,
        output_path: str = "data/webcam_dataset.npz",
    ):
        self.classes = classes or ["Open_Palm", "Closed_Fist", "Index_Point", "Victory"]
        self.samples_per_class = samples_per_class
        self.output_path = Path(output_path)
        self.landmarker = HandLandmarkerEngine()

    def run_guided_collection(self, camera_id: int = 0) -> None:
        """Runs interactive OpenCV window guiding data capture."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Unable to open camera {camera_id}")
            return

        all_landmarks = []
        all_labels = []

        print("=== M-HANDS GUIDED DATA COLLECTION ===")
        print("Controls: SPACE to start recording current gesture, 'q' to quit.")

        for cls_name in self.classes:
            print(f"\n[NEXT GESTURE]: {cls_name}")
            print("Position your hand and press SPACE when ready...")

            # 1. Waiting for user readiness
            ready = False
            while not ready:
                ret, frame = cap.read()
                if not ret:
                    break

                cv2.putText(
                    frame,
                    f"PREPARE: {cls_name} (Press SPACE)",
                    (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    COLOR_CYAN,
                    2,
                )
                cv2.imshow("M-Hands Data Collector", frame)
                key = cv2.waitKey(10) & 0xFF
                if key == ord(" "):
                    ready = True
                elif key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # 2. Recording loop
            collected = 0
            while collected < self.samples_per_class:
                ret, frame = cap.read()
                if not ret:
                    break

                lms, _, _ = self.landmarker.process(frame)
                if lms is not None:
                    all_landmarks.append(lms)
                    all_labels.append(cls_name)
                    collected += 1

                progress_pct = int((collected / self.samples_per_class) * 100)
                cv2.putText(
                    frame,
                    f"RECORDING: {cls_name} [{progress_pct}%]",
                    (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    COLOR_EMERALD,
                    2,
                )
                cv2.imshow("M-Hands Data Collector", frame)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

            print(f"Captured {collected} samples for {cls_name}.")

        cap.release()
        cv2.destroyAllWindows()

        if all_landmarks:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.output_path,
                landmarks=np.array(all_landmarks, dtype=np.float32),
                labels=np.array(all_labels),
            )
            print(f"\nDataset successfully saved to: {self.output_path}")
