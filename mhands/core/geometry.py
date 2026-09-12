"""
Geometrically Invariant Hand Representation Engine.

Implements translation, scale, and planar rotation invariant feature extraction
from 21 monocular 3D hand landmarks according to rigorous biomechanical kinematics.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Landmark Indices conforming to MediaPipe Hands topology
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

HAND_LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

DIGIT_JOINTS = {
    "thumb": (THUMB_MCP, THUMB_TIP),
    "index": (INDEX_MCP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_TIP),
}

EPSILON = 1e-7


def landmarks_to_numpy(landmarks: Any) -> np.ndarray:
    """
    Converts various landmark representations (MediaPipe NormalizedLandmarkList,
    list of dicts, or nested lists) into a standardized (21, 3) float32 NumPy array.
    """
    if isinstance(landmarks, np.ndarray):
        if landmarks.ndim == 2 and landmarks.shape == (21, 3):
            return landmarks.astype(np.float32)
        elif landmarks.ndim == 1 and landmarks.shape[0] == 63:
            return landmarks.reshape((21, 3)).astype(np.float32)
        raise ValueError(f"Unexpected NumPy landmark shape: {landmarks.shape}")

    coords = []
    # MediaPipe landmark container check
    if hasattr(landmarks, "landmark"):
        landmarks = landmarks.landmark

    for lm in landmarks:
        if hasattr(lm, "x") and hasattr(lm, "y"):
            z = getattr(lm, "z", 0.0)
            coords.append([lm.x, lm.y, z])
        elif isinstance(lm, (list, tuple)) and len(lm) >= 2:
            z = lm[2] if len(lm) >= 3 else 0.0
            coords.append([lm[0], lm[1], z])
        elif isinstance(lm, dict) and "x" in lm and "y" in lm:
            coords.append([lm["x"], lm["y"], lm.get("z", 0.0)])
        else:
            raise TypeError(f"Unsupported landmark item type: {type(lm)}")

    arr = np.array(coords, dtype=np.float32)
    if arr.shape != (21, 3):
        raise ValueError(f"Expected (21, 3) landmark array, got {arr.shape}")
    return arr


def extract_raw_features(landmarks: Any) -> np.ndarray:
    """
    Extracts raw 63-dimensional Cartesian coordinate array x_raw in R^63.
    x_raw = [x_0, y_0, z_0, ..., x_20, y_20, z_20]^T
    """
    pts = landmarks_to_numpy(landmarks)
    return pts.flatten().astype(np.float32)


def compute_angle_between_vectors(u: np.ndarray, v: np.ndarray) -> float:
    """
    Calculates angle in radians between two 3D vectors:
    theta = arccos((u . v) / (||u|| * ||v||))
    Clamped to [-1.0, 1.0] to avoid numerical instabilities.
    """
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u < EPSILON or norm_v < EPSILON:
        return 0.0
    cosine = np.dot(u, v) / (norm_u * norm_v)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.arccos(cosine))


def extract_invariant_features(landmarks: Any) -> np.ndarray:
    """
    Transforms 21 monocular 3D landmarks into a canonical 8-dimensional
    geometrically invariant feature vector Phi in R^8:
    
    Phi = [
        theta_thumb,
        theta_index,
        theta_middle,
        theta_ring,
        theta_pinky,
        theta_abduct,
        r_spread,
        r_aperture
    ]^T

    Guarantees:
    - Translation Invariance: Re-centered onto wrist joint p_0.
    - Scale Invariance: Normalized by palm skeletal metric d_ref = ||p_9 - p_0||_2.
    - Planar Rotation Robustness: Intrinsic angles and normalized distance ratios.
    """
    pts = landmarks_to_numpy(landmarks)  # (21, 3)

    p0 = pts[WRIST]
    p9 = pts[MIDDLE_MCP]

    # Skeletal scale reference length (wrist to middle MCP)
    d_ref = float(np.linalg.norm(p9 - p0))
    if d_ref < EPSILON:
        d_ref = 1.0  # Fallback to prevent divide-by-zero on degenerate inputs

    # 1. Primary MCP flexion angles for all 5 digits
    angles = []
    for digit_name in ["thumb", "index", "middle", "ring", "pinky"]:
        mcp_idx, tip_idx = DIGIT_JOINTS[digit_name]
        p_mcp = pts[mcp_idx]
        p_tip = pts[tip_idx]
        u = p_mcp - p0
        v = p_tip - p_mcp
        theta = compute_angle_between_vectors(u, v)
        angles.append(theta)

    # 2. Radial abduction angle between thumb and index:
    # w_thumb = p_4 - p_2, w_index = p_8 - p_5
    w_thumb = pts[THUMB_TIP] - pts[THUMB_MCP]
    w_index = pts[INDEX_TIP] - pts[INDEX_MCP]
    theta_abduct = compute_angle_between_vectors(w_thumb, w_index)

    # 3. Inter-finger spread ratio: ||p_8 - p_12||_2 / d_ref
    p8 = pts[INDEX_TIP]
    p12 = pts[MIDDLE_TIP]
    r_spread = float(np.linalg.norm(p8 - p12) / d_ref)

    # 4. Pinch aperture ratio: ||p_4 - p_8||_2 / d_ref
    p4 = pts[THUMB_TIP]
    r_aperture = float(np.linalg.norm(p4 - p8) / d_ref)

    phi = np.array([
        angles[0],       # theta_thumb
        angles[1],       # theta_index
        angles[2],       # theta_middle
        angles[3],       # theta_ring
        angles[4],       # theta_pinky
        theta_abduct,    # theta_abduct
        r_spread,        # r_spread
        r_aperture       # r_aperture
    ], dtype=np.float32)

    return phi


def verify_geometric_invariance(landmarks: np.ndarray) -> Dict[str, bool]:
    """
    Mathematical self-test proving translation and scale invariance.
    Returns boolean status for each geometric invariance condition.
    """
    base_phi = extract_invariant_features(landmarks)

    # 1. Translation Invariance Test: P_shifted = P + t
    t = np.array([12.5, -45.2, 8.7], dtype=np.float32)
    shifted_landmarks = landmarks + t
    shifted_phi = extract_invariant_features(shifted_landmarks)
    translation_delta = float(np.linalg.norm(base_phi - shifted_phi))

    # 2. Scale Invariance Test: P_scaled = s * P
    s = 2.75
    scaled_landmarks = landmarks * s
    scaled_phi = extract_invariant_features(scaled_landmarks)
    scale_delta = float(np.linalg.norm(base_phi - scaled_phi))

    # 3. Combined Affine Invariance: P_affine = s * P + t
    affine_landmarks = landmarks * 0.45 + t
    affine_phi = extract_invariant_features(affine_landmarks)
    affine_delta = float(np.linalg.norm(base_phi - affine_phi))

    tolerance = 1e-4
    return {
        "translation_invariant": translation_delta < tolerance,
        "scale_invariant": scale_delta < tolerance,
        "affine_invariant": affine_delta < tolerance,
        "translation_delta": translation_delta,
        "scale_delta": scale_delta,
        "affine_delta": affine_delta,
    }
