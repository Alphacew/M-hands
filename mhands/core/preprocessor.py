"""
Biomechanical & Illumination Preprocessor.

Normalizes environmental lighting variations (particularly for Fitzpatrick V-VI
complexions in dim or uneven illumination) and ensures bilateral mirror symmetry
between left and right hands.
"""

from typing import Tuple, Optional, Any
import cv2
import numpy as np


class AdaptiveIlluminationNormalizer:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    strictly to the luminance channel (Y) in YCrCb color space.
    Preserves chromaticity while restoring gradient contrast across hand contours.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Enhances frame luminance contrast without color cast distortion.
        Returns contrast-normalized BGR image.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr

        # Convert to YCrCb space
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        # Apply CLAHE exclusively to luminance
        y_eq = self.clahe.apply(y)

        # Re-merge and convert back to BGR
        ycrcb_eq = cv2.merge([y_eq, cr, cb])
        return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)


class BilateralHandMirror:
    """
    Applies coordinate reflection across the horizontal axis for left-handed captures
    to leverage anatomical bilateral symmetry.
    """

    @staticmethod
    def normalize_handedness(
        landmarks: np.ndarray,
        handedness_label: str = "Right",
    ) -> np.ndarray:
        """
        If handedness is 'Left', reflects horizontal x-coordinate:
        x_symmetric = 1.0 - x
        """
        coords = np.array(landmarks, copy=True)
        if handedness_label.lower() == "left":
            coords[:, 0] = 1.0 - coords[:, 0]
        return coords
