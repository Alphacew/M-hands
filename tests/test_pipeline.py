"""
End-to-End Pipeline and Classifier Integration Tests.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from mhands.data.synthetic_generator import SyntheticHandGenerator
from mhands.data.dataset import load_benchmark_splits
from mhands.pipeline.classifier import GestureClassifier
from mhands.benchmark.evaluator import GeneralizationEvaluator


def test_classifier_fit_and_serialization():
    generator = SyntheticHandGenerator(random_seed=42)
    lms, lbls = generator.generate_dataset(samples_per_class=20, session_type="same_session")

    # Invariant features
    from mhands.core.geometry import extract_invariant_features
    X = np.array([extract_invariant_features(lm) for lm in lms])

    clf = GestureClassifier(model_type="svm_rbf", feature_type="invariant")
    clf.fit(X, lbls)

    pred, conf, prob_map = clf.classify_single(X[0])
    assert pred in clf.classes
    assert 0.0 <= conf <= 1.0
    assert len(prob_map) == 4

    # Test serialization
    with tempfile.TemporaryDirectory() as tmpdir:
        model_file = Path(tmpdir) / "test_model.joblib"
        clf.save(model_file)
        assert model_file.exists()

        loaded_clf = GestureClassifier.load(model_file)
        pred2, conf2, _ = loaded_clf.classify_single(X[0])
        assert pred == pred2
        assert np.isclose(conf, conf2)


def test_evaluator_generalization_gap():
    splits = load_benchmark_splits(data_dir="data")

    clf_inv = GestureClassifier(model_type="svm_rbf", feature_type="invariant")
    clf_inv.fit(splits["X_train_inv"], splits["y_train"])

    clf_raw = GestureClassifier(model_type="svm_rbf", feature_type="raw")
    clf_raw.fit(splits["X_train_raw"], splits["y_train"])

    evaluator = GeneralizationEvaluator()
    result = evaluator.evaluate(clf_raw, clf_inv, splits)

    # Invariant model should outperform raw model on cross-session test
    print(f"Raw cross: {result.raw_cross.accuracy}, Inv cross: {result.inv_cross.accuracy}")
    assert result.inv_cross.accuracy > result.raw_cross.accuracy
    assert result.inv_cross.accuracy >= 0.90
