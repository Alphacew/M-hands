"""
Subsystem Latency Profiler & Throughput Benchmark Engine.

Disaggregates execution latency across individual pipeline stages:
- Frame I/O & BGR-to-RGB Preprocessing
- MediaPipe Neural Hand Landmarker Inference
- R^8 Invariant Vector Transformation
- scikit-learn Classifier Inference (SVM vs Random Forest)
- 1 Euro Temporal Damping & HUD Annotation
- End-to-End Cumulative Latency & FPS
"""

from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from mhands.core.geometry import extract_invariant_features, extract_raw_features
from mhands.core.filters import MultiPointOneEuroFilter
from mhands.core.preprocessor import AdaptiveIlluminationNormalizer
from mhands.core.state_machine import InteractionStateMachine
from mhands.pipeline.landmarker import HandLandmarkerEngine
from mhands.pipeline.classifier import GestureClassifier
from mhands.pipeline.visualizer import HUDVisualizer


@dataclass
class StageMetric:
    name: str
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    peak_ms: float
    bottleneck_profile: str


@dataclass
class LatencyProfile:
    stages: List[StageMetric]
    cumulative_mean_ms: float
    cumulative_peak_ms: float
    pipeline_fps: float
    iterations: int


class PipelineProfiler:
    """
    Executes microsecond-resolution latency profiling across all CV pipeline subsystems.
    """

    def __init__(
        self,
        classifier: GestureClassifier,
        iterations: int = 300,
        warmup: int = 30,
        frame_width: int = 1280,
        frame_height: int = 720,
    ):
        self.classifier = classifier
        self.iterations = iterations
        self.warmup = warmup
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.normalizer = AdaptiveIlluminationNormalizer()
        self.landmarker = HandLandmarkerEngine()
        self.filter = MultiPointOneEuroFilter()
        self.fsm = InteractionStateMachine()
        self.visualizer = HUDVisualizer()

    def run_benchmark(self, sample_frame: Optional[np.ndarray] = None) -> LatencyProfile:
        """
        Runs comprehensive benchmark over specified iterations.
        """
        if sample_frame is None:
            # Synthetic 720p frame with realistic color gradient
            sample_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            sample_frame[:, :, 0] = 60
            sample_frame[:, :, 1] = 80
            sample_frame[:, :, 2] = 100

        # Synthetic landmark array for invariant & classifier testing
        from mhands.data.synthetic_generator import SyntheticHandGenerator
        generator = SyntheticHandGenerator(random_seed=42)
        test_landmarks = generator.generate_sample("Open_Palm", "same_session")

        latencies_pre = []
        latencies_mp = []
        latencies_inv = []
        latencies_clf = []
        latencies_hud = []
        latencies_total = []

        total_runs = self.warmup + self.iterations

        for i in range(total_runs):
            t_start = time.perf_counter()

            # Stage 1: Frame Preprocessing (CLAHE)
            t0 = time.perf_counter()
            processed_frame = self.normalizer.process(sample_frame)
            t1 = time.perf_counter()
            d_pre = (t1 - t0) * 1000.0

            # Stage 2: MediaPipe Inference
            t0 = time.perf_counter()
            # Feed processed frame to landmarker
            _ = self.landmarker.process(processed_frame)
            t1 = time.perf_counter()
            d_mp = (t1 - t0) * 1000.0

            # Stage 3: Invariant Feature Transformation
            t0 = time.perf_counter()
            phi = extract_invariant_features(test_landmarks)
            t1 = time.perf_counter()
            d_inv = (t1 - t0) * 1000.0

            # Stage 4: Classifier Inference
            t0 = time.perf_counter()
            pred, conf, prob_map = self.classifier.classify_single(phi)
            t1 = time.perf_counter()
            d_clf = (t1 - t0) * 1000.0

            # Stage 5: Signal Damping & HUD Rendering
            t0 = time.perf_counter()
            filtered_lms = self.filter.process(test_landmarks)
            event = self.fsm.update(pred, conf)
            _ = self.visualizer.render(
                frame=sample_frame,
                landmarks=filtered_lms,
                event=event,
                probabilities=prob_map,
                telemetry={"fps": 60.0, "pipeline_ms": 16.0, "classifier_ms": d_clf, "mediapipe_ms": d_mp},
                slider_value=50.0,
                features_8d=phi,
            )
            t1 = time.perf_counter()
            d_hud = (t1 - t0) * 1000.0

            t_end = time.perf_counter()
            d_total = (t_end - t_start) * 1000.0

            if i >= self.warmup:
                latencies_pre.append(d_pre)
                latencies_mp.append(d_mp)
                latencies_inv.append(d_inv)
                latencies_clf.append(d_clf)
                latencies_hud.append(d_hud)
                latencies_total.append(d_total)

        stages_data = [
            ("Frame Capture & BGR-to-RGB Preprocessing (OpenCV)", latencies_pre, "I/O & Memory Copy Bound"),
            ("Neural Hand Landmarker Inference (MediaPipe)", latencies_mp, "Compute/Inference Bound (Primary Bottleneck)"),
            ("Invariant Vector Transformation (R^8)", latencies_inv, "Negligible CPU Bound (Vectorized NumPy)"),
            ("Classifier Inference (scikit-learn SVM/RF)", latencies_clf, "Negligible CPU Bound (Vectorized)"),
            ("Signal Damping & HUD Annotation (1 Euro / OpenCV)", latencies_hud, "Graphic Render & Blit Bound"),
        ]

        stage_metrics = []
        for name, data, note in stages_data:
            arr = np.array(data)
            stage_metrics.append(
                StageMetric(
                    name=name,
                    mean_ms=float(np.mean(arr)),
                    std_ms=float(np.std(arr)),
                    p50_ms=float(np.percentile(arr, 50)),
                    p95_ms=float(np.percentile(arr, 95)),
                    peak_ms=float(np.max(arr)),
                    bottleneck_profile=note,
                )
            )

        total_arr = np.array(latencies_total)
        mean_total = float(np.mean(total_arr))
        peak_total = float(np.max(total_arr))
        fps = 1000.0 / mean_total if mean_total > 0 else 0.0

        return LatencyProfile(
            stages=stage_metrics,
            cumulative_mean_ms=mean_total,
            cumulative_peak_ms=peak_total,
            pipeline_fps=fps,
            iterations=self.iterations,
        )
