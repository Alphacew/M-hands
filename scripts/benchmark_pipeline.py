#!/usr/bin/env python3
"""
Empirical Benchmark Runner for M-Hands.

Executes:
1. 2x2 Generalization Matrix evaluation (Same-Session vs Cross-Session on Raw vs Invariant).
2. Subsystem latency profiling (capturing, MediaPipe, invariant transform, classifier, HUD).
3. Formats and exports results into Markdown and LaTeX reports.
"""

from pathlib import Path
import sys

from mhands.data.dataset import load_benchmark_splits
from mhands.pipeline.classifier import GestureClassifier
from mhands.benchmark.evaluator import GeneralizationEvaluator
from mhands.benchmark.profiler import PipelineProfiler
from mhands.benchmark.reporter import BenchmarkReporter


def main():
    print("==================================================")
    print("      M-HANDS COMPREHENSIVE BENCHMARK SUITE       ")
    print("==================================================")

    models_dir = Path("models")
    inv_svm_path = models_dir / "invariant_svm.joblib"
    raw_svm_path = models_dir / "raw_svm.joblib"

    if not inv_svm_path.exists() or not raw_svm_path.exists():
        print("[!] Trained models not found. Running train_models.py first...")
        from scripts.train_models import main as run_train
        run_train()

    print("\n[1/3] Loading trained models and dataset splits...")
    clf_inv_svm = GestureClassifier.load(inv_svm_path)
    clf_raw_svm = GestureClassifier.load(raw_svm_path)
    splits = load_benchmark_splits(data_dir="data")

    # 1. Evaluate Generalization Matrix
    print("\n[2/3] Evaluating 2x2 Generalization Matrix across domain splits...")
    evaluator = GeneralizationEvaluator()
    result_svm = evaluator.evaluate(clf_raw_svm, clf_inv_svm, splits)

    four_cell_md = BenchmarkReporter.format_markdown_four_cell(result_svm)
    four_cell_tex = BenchmarkReporter.format_latex_four_cell(result_svm)

    print("\n" + four_cell_md)

    # 2. Subsystem Latency Profiling
    print("\n[3/3] Profiling subsystem execution latencies (200 iterations)...")
    profiler = PipelineProfiler(classifier=clf_inv_svm, iterations=200, warmup=20)
    latency_profile = profiler.run_benchmark()

    latency_md = BenchmarkReporter.format_markdown_latency(latency_profile)
    print("\n" + latency_md)

    # Save consolidated benchmark reports
    full_report = f"""# M-Hands Empirical Benchmark Report

{four_cell_md}

---

{latency_md}

---

### LaTeX Publication Source

```latex
{four_cell_tex}
```
"""

    report_path = Path("BENCHMARK_REPORT.md")
    report_path.write_text(full_report)
    print(f"\nConsolidated benchmark report saved to: {report_path.resolve()}")


if __name__ == "__main__":
    main()
