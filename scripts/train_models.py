#!/usr/bin/env python3
"""
Model Training Script for M-Hands.

Fits SVM (RBF) and Random Forest classifiers on both Raw R^63 and Invariant R^8
feature representations using the same-session training partition.
Serializes trained models to models/ directory.
"""

from pathlib import Path
import sys

from mhands.data.dataset import load_benchmark_splits
from mhands.pipeline.classifier import GestureClassifier


def main():
    print("==================================================")
    print("        M-HANDS MODEL TRAINING PIPELINE           ")
    print("==================================================")

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading benchmark dataset splits...")
    splits = load_benchmark_splits(data_dir="data")

    X_train_raw = splits["X_train_raw"]
    X_train_inv = splits["X_train_inv"]
    y_train = splits["y_train"]

    print(f"      Training samples: {len(y_train)} across 4 gesture classes.")
    print(f"      Raw features shape: {X_train_raw.shape}")
    print(f"      Invariant features shape: {X_train_inv.shape}")

    # 1. Train Invariant SVM (RBF) - Primary Robust Model
    print("\n[2/4] Training Invariant R^8 SVM Classifier (RBF Kernel)...")
    clf_inv_svm = GestureClassifier(model_type="svm_rbf", feature_type="invariant")
    clf_inv_svm.fit(X_train_inv, y_train)
    clf_inv_svm.save(models_dir / "invariant_svm.joblib")
    print(f"      Saved: {models_dir / 'invariant_svm.joblib'}")

    # 2. Train Raw SVM (RBF) - Baseline Model for Generalization Comparison
    print("\n[3/4] Training Raw R^63 SVM Classifier (RBF Kernel)...")
    clf_raw_svm = GestureClassifier(model_type="svm_rbf", feature_type="raw")
    clf_raw_svm.fit(X_train_raw, y_train)
    clf_raw_svm.save(models_dir / "raw_svm.joblib")
    print(f"      Saved: {models_dir / 'raw_svm.joblib'}")

    # 3. Train Invariant Random Forest & Raw Random Forest
    print("\n[4/4] Training Random Forest Classifiers...")
    clf_inv_rf = GestureClassifier(model_type="random_forest", feature_type="invariant")
    clf_inv_rf.fit(X_train_inv, y_train)
    clf_inv_rf.save(models_dir / "invariant_rf.joblib")

    clf_raw_rf = GestureClassifier(model_type="random_forest", feature_type="raw")
    clf_raw_rf.fit(X_train_raw, y_train)
    clf_raw_rf.save(models_dir / "raw_rf.joblib")

    print(f"      Saved: {models_dir / 'invariant_rf.joblib'}")
    print(f"      Saved: {models_dir / 'raw_rf.joblib'}")

    print("\nModel training complete! All 4 models serialized successfully.")


if __name__ == "__main__":
    main()
