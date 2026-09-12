"""
Gesture Classification Engine.

Implements model training, hyperparameter configuration, serialization,
and calibrated probability inference for SVM and Random Forest estimators
on both raw R^63 and invariant R^8 feature manifolds.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DEFAULT_GESTURES = ["Open_Palm", "Closed_Fist", "Index_Point", "Victory"]


class GestureClassifier:
    """
    Wrapper for training and real-time inference with scikit-learn estimators.
    """

    def __init__(
        self,
        model_type: str = "svm_rbf",
        feature_type: str = "invariant",
        classes: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.model_type = model_type
        self.feature_type = feature_type
        self.classes = classes if classes is not None else list(DEFAULT_GESTURES)
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        if self.model_type == "svm_rbf":
            estimator = SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                probability=True,
                random_state=self.random_state,
            )
            self.pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", estimator),
            ])
        elif self.model_type == "svm_linear":
            estimator = SVC(
                kernel="linear",
                C=1.0,
                probability=True,
                random_state=self.random_state,
            )
            self.pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", estimator),
            ])
        elif self.model_type == "random_forest":
            estimator = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=4,
                random_state=self.random_state,
                n_jobs=-1,
            )
            # Tree ensembles don't strictly need scaling, but uniform pipeline interface
            self.pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", estimator),
            ])
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GestureClassifier":
        """Fits the pipeline on feature matrix X and label vector y."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels for feature array X."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities for feature array X."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.pipeline.predict_proba(X)

    def classify_single(self, feature_vector: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Infers gesture class and confidence for a single feature vector.
        
        Returns:
            predicted_class: Name of highest probability gesture
            confidence: Probability score in [0.0, 1.0]
            class_probabilities: Mapping from each class name to its probability
        """
        feats = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        probs = self.pipeline.predict_proba(feats)[0]
        classes = self.pipeline.classes_

        class_prob_map = {str(c): float(p) for c, p in zip(classes, probs)}
        best_idx = int(np.argmax(probs))
        best_class = str(classes[best_idx])
        best_conf = float(probs[best_idx])

        return best_class, best_conf, class_prob_map

    def save(self, file_path: Union[str, Path]) -> None:
        """Serializes model to disk using joblib."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": self.model_type,
            "feature_type": self.feature_type,
            "classes": self.classes,
            "pipeline": self.pipeline,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "GestureClassifier":
        """Loads serialized model from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        payload = joblib.load(path)
        instance = cls(
            model_type=payload["model_type"],
            feature_type=payload["feature_type"],
            classes=payload["classes"],
        )
        instance.pipeline = payload["pipeline"]
        return instance
