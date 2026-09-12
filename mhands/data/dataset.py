"""
Dataset Management & Benchmark Splitting Engine.

Handles loading, transforming, and partitioning landmark datasets into
Raw R^63 and Invariant R^8 representations across Same-Session and Cross-Session splits.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import numpy as np
from sklearn.model_selection import train_test_split

from mhands.core.geometry import extract_raw_features, extract_invariant_features
from mhands.data.synthetic_generator import SyntheticHandGenerator


class GestureDataset:
    """
    Manages collection of hand landmark arrays and associated class labels.
    """

    def __init__(self, landmarks: np.ndarray, labels: np.ndarray):
        """
        landmarks: shape (N, 21, 3)
        labels: shape (N,)
        """
        self.landmarks = np.asarray(landmarks, dtype=np.float32)
        self.labels = np.asarray(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def extract_raw_features(self) -> np.ndarray:
        """Transforms all samples into raw R^63 feature matrix."""
        N = len(self.landmarks)
        return self.landmarks.reshape((N, 63)).astype(np.float32)

    def extract_invariant_features(self) -> np.ndarray:
        """Transforms all samples into R^8 invariant feature matrix."""
        N = len(self.landmarks)
        feats = np.zeros((N, 8), dtype=np.float32)
        for i in range(N):
            feats[i] = extract_invariant_features(self.landmarks[i])
        return feats

    def save_npz(self, file_path: Union[str, Path]) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, landmarks=self.landmarks, labels=self.labels)

    @classmethod
    def load_npz(cls, file_path: Union[str, Path]) -> "GestureDataset":
        path = Path(file_path)
        data = np.load(path, allow_pickle=True)
        return cls(landmarks=data["landmarks"], labels=data["labels"])


def load_or_generate_benchmark_datasets(
    data_dir: Union[str, Path] = "data",
    samples_per_class: int = 200,
    force_regenerate: bool = False,
) -> Tuple[GestureDataset, GestureDataset]:
    """
    Loads or generates the standard Same-Session and Cross-Session benchmark datasets.
    
    Returns:
        same_session_ds: Controlled domain capture
        cross_session_ds: Independent operational test capture
    """
    data_path = Path(data_dir)
    same_path = data_path / "same_session.npz"
    cross_path = data_path / "cross_session.npz"

    if not force_regenerate and same_path.exists() and cross_path.exists():
        same_ds = GestureDataset.load_npz(same_path)
        cross_ds = GestureDataset.load_npz(cross_path)
        return same_ds, cross_ds

    generator = SyntheticHandGenerator(random_seed=42)

    same_lms, same_lbls = generator.generate_dataset(
        samples_per_class=samples_per_class,
        session_type="same_session",
    )
    cross_lms, cross_lbls = generator.generate_dataset(
        samples_per_class=samples_per_class,
        session_type="cross_session",
    )

    same_ds = GestureDataset(same_lms, same_lbls)
    cross_ds = GestureDataset(cross_lms, cross_lbls)

    same_ds.save_npz(same_path)
    cross_ds.save_npz(cross_path)

    return same_ds, cross_ds


def load_benchmark_splits(
    data_dir: Union[str, Path] = "data",
    test_size: float = 0.25,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Produces partitioned splits ready for model fitting and 2x2 matrix benchmarking.
    """
    same_ds, cross_ds = load_or_generate_benchmark_datasets(data_dir=data_dir)

    # 1. Split same-session dataset into train and same-session test
    train_idx, test_same_idx = train_test_split(
        np.arange(len(same_ds)),
        test_size=test_size,
        stratify=same_ds.labels,
        random_state=random_state,
    )

    train_lms = same_ds.landmarks[train_idx]
    y_train = same_ds.labels[train_idx]

    test_same_lms = same_ds.landmarks[test_same_idx]
    y_test_same = same_ds.labels[test_same_idx]

    # Cross-session test set is the entirety of independent cross_ds
    test_cross_lms = cross_ds.landmarks
    y_test_cross = cross_ds.labels

    train_ds = GestureDataset(train_lms, y_train)
    test_same_ds = GestureDataset(test_same_lms, y_test_same)
    test_cross_ds = GestureDataset(test_cross_lms, y_test_cross)

    return {
        # Raw features (R^63)
        "X_train_raw": train_ds.extract_raw_features(),
        "X_test_same_raw": test_same_ds.extract_raw_features(),
        "X_test_cross_raw": test_cross_ds.extract_raw_features(),
        # Invariant features (R^8)
        "X_train_inv": train_ds.extract_invariant_features(),
        "X_test_same_inv": test_same_ds.extract_invariant_features(),
        "X_test_cross_inv": test_cross_ds.extract_invariant_features(),
        # Ground truth labels
        "y_train": y_train,
        "y_test_same": y_test_same,
        "y_test_cross": y_test_cross,
    }
