"""
Generalization Evaluator & Four-Cell Matrix Engine.

Empirically benchmarks model generalization under Same-Session and Cross-Session
domain conditions for both Raw R^63 coordinates and Invariant R^8 features.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from mhands.pipeline.classifier import GestureClassifier


@dataclass
class EvaluationCell:
    accuracy: float
    f1_macro: float
    precision_macro: float
    recall_macro: float
    confusion: np.ndarray


@dataclass
class GeneralizationResult:
    model_name: str
    raw_same: EvaluationCell
    raw_cross: EvaluationCell
    inv_same: EvaluationCell
    inv_cross: EvaluationCell
    raw_delta: float
    inv_delta: float
    classes: List[str]


class GeneralizationEvaluator:
    """
    Computes four-cell empirical generalization matrix:
    [Raw Coordinates, Invariant Features] x [Same-Session Test, Cross-Session Test]
    """

    def __init__(self, classes: Optional[List[str]] = None):
        self.classes = classes or ["Open_Palm", "Closed_Fist", "Index_Point", "Victory"]

    def _evaluate_cell(self, classifier: GestureClassifier, X: np.ndarray, y: np.ndarray) -> EvaluationCell:
        preds = classifier.predict(X)
        acc = float(accuracy_score(y, preds))
        f1 = float(f1_score(y, preds, average="macro", zero_division=0))
        prec = float(precision_score(y, preds, average="macro", zero_division=0))
        rec = float(recall_score(y, preds, average="macro", zero_division=0))
        cm = confusion_matrix(y, preds, labels=self.classes)

        return EvaluationCell(
            accuracy=acc,
            f1_macro=f1,
            precision_macro=prec,
            recall_macro=rec,
            confusion=cm,
        )

    def evaluate(
        self,
        raw_classifier: GestureClassifier,
        inv_classifier: GestureClassifier,
        splits: Dict[str, np.ndarray],
    ) -> GeneralizationResult:
        """
        Evaluates raw and invariant classifiers across both same-session and cross-session test sets.
        """
        # 1. Raw Coordinates Evaluation
        raw_same = self._evaluate_cell(raw_classifier, splits["X_test_same_raw"], splits["y_test_same"])
        raw_cross = self._evaluate_cell(raw_classifier, splits["X_test_cross_raw"], splits["y_test_cross"])

        # 2. Invariant Features Evaluation
        inv_same = self._evaluate_cell(inv_classifier, splits["X_test_same_inv"], splits["y_test_same"])
        inv_cross = self._evaluate_cell(inv_classifier, splits["X_test_cross_inv"], splits["y_test_cross"])

        raw_delta = raw_cross.accuracy - raw_same.accuracy
        inv_delta = inv_cross.accuracy - inv_same.accuracy

        return GeneralizationResult(
            model_name=raw_classifier.model_type,
            raw_same=raw_same,
            raw_cross=raw_cross,
            inv_same=inv_same,
            inv_cross=inv_cross,
            raw_delta=raw_delta,
            inv_delta=inv_delta,
            classes=self.classes,
        )
