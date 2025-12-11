"""
Metrics Tracking for Benchmark Evaluation.

Provides comprehensive metrics collection for comparing model configurations:
- Accuracy / task-specific metrics (binary, MCQ, free-text)
- Frames used per input
- Tokens per QA
- GPU time per QA
- Memory usage

Outputs:
- Per-sample metrics
- Aggregate statistics per model config
- Comparison tables
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class SampleMetrics:
    """Metrics for a single evaluation sample."""

    sample_id: str
    benchmark: str
    task_type: str
    model_config: str

    # Accuracy
    correct: bool
    predicted: str
    ground_truth: str
    confidence: float = 0.0

    # Compute metrics
    num_frames: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Timing (seconds)
    perception_time: float = 0.0
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0

    # Memory (GB)
    peak_vram_gb: float = 0.0

    # Optional metadata
    video_duration_sec: float = 0.0
    question: str = ""
    answer_format: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sample_id": self.sample_id,
            "benchmark": self.benchmark,
            "task_type": self.task_type,
            "model_config": self.model_config,
            "correct": self.correct,
            "predicted": self.predicted,
            "ground_truth": self.ground_truth,
            "confidence": self.confidence,
            "num_frames": self.num_frames,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "perception_time": self.perception_time,
            "retrieval_time": self.retrieval_time,
            "generation_time": self.generation_time,
            "total_time": self.total_time,
            "peak_vram_gb": self.peak_vram_gb,
            "video_duration_sec": self.video_duration_sec,
            "question": self.question,
            "answer_format": self.answer_format,
        }


@dataclass
class AggregateMetrics:
    """Aggregate metrics for a model configuration on a benchmark."""

    model_config: str
    benchmark: str
    num_samples: int = 0

    # Accuracy
    accuracy: float = 0.0
    correct_count: int = 0

    # Compute averages
    avg_frames: float = 0.0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    avg_total_tokens: float = 0.0

    # Timing averages (seconds)
    avg_perception_time: float = 0.0
    avg_retrieval_time: float = 0.0
    avg_generation_time: float = 0.0
    avg_total_time: float = 0.0

    # Memory
    max_peak_vram_gb: float = 0.0

    # Timing totals
    total_time_sec: float = 0.0

    # Task-specific breakdown
    task_accuracy: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_config": self.model_config,
            "benchmark": self.benchmark,
            "num_samples": self.num_samples,
            "accuracy": self.accuracy,
            "correct_count": self.correct_count,
            "avg_frames": self.avg_frames,
            "avg_input_tokens": self.avg_input_tokens,
            "avg_output_tokens": self.avg_output_tokens,
            "avg_total_tokens": self.avg_total_tokens,
            "avg_perception_time": self.avg_perception_time,
            "avg_retrieval_time": self.avg_retrieval_time,
            "avg_generation_time": self.avg_generation_time,
            "avg_total_time": self.avg_total_time,
            "max_peak_vram_gb": self.max_peak_vram_gb,
            "total_time_sec": self.total_time_sec,
            "task_accuracy": self.task_accuracy,
        }


class MetricsTracker:
    """
    Comprehensive metrics tracking for benchmark evaluation.

    Collects per-sample metrics and computes aggregate statistics
    for model comparison.

    Example:
        tracker = MetricsTracker(output_dir="results/eval")

        # Track each sample
        metrics = tracker.start_sample("sample_001", "GlitchBench", "glitch_detection", "GVP-Light")
        # ... run inference ...
        tracker.end_sample(metrics, correct=True, predicted="yes", ground_truth="yes",
                          num_frames=64, input_tokens=1024, output_tokens=32)

        # Get aggregates
        aggregates = tracker.compute_aggregates()
        tracker.save_results()
    """

    def __init__(
        self,
        output_dir: str = "results/eval",
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize metrics tracker.

        Args:
            output_dir: Directory for saving results
            experiment_name: Optional experiment name for organizing results
        """
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Storage
        self._samples: list[SampleMetrics] = []
        self._current_sample: Optional[SampleMetrics] = None
        self._start_time: Optional[float] = None

        # GPU memory tracking
        self._gpu_available = False
        try:
            import torch
            if torch.cuda.is_available():
                self._gpu_available = True
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass

        logger.info(f"MetricsTracker initialized: {self.output_dir / self.experiment_name}")

    def start_sample(
        self,
        sample_id: str,
        benchmark: str,
        task_type: str,
        model_config: str,
        video_duration_sec: float = 0.0,
        question: str = "",
        answer_format: str = "",
    ) -> SampleMetrics:
        """
        Start tracking a new sample.

        Args:
            sample_id: Unique sample identifier
            benchmark: Benchmark name (e.g., "GlitchBench")
            task_type: Task type (e.g., "glitch_detection")
            model_config: Model configuration name (e.g., "GVP-Light")
            video_duration_sec: Video duration in seconds
            question: Question text
            answer_format: Answer format (e.g., "binary", "mcq")

        Returns:
            SampleMetrics object to populate
        """
        self._start_time = time.perf_counter()

        # Reset GPU memory tracking
        if self._gpu_available:
            import torch
            torch.cuda.reset_peak_memory_stats()

        self._current_sample = SampleMetrics(
            sample_id=sample_id,
            benchmark=benchmark,
            task_type=task_type,
            model_config=model_config,
            correct=False,
            predicted="",
            ground_truth="",
            video_duration_sec=video_duration_sec,
            question=question,
            answer_format=answer_format,
        )

        return self._current_sample

    def end_sample(
        self,
        sample: SampleMetrics,
        correct: bool,
        predicted: str,
        ground_truth: str,
        num_frames: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        perception_time: float = 0.0,
        retrieval_time: float = 0.0,
        generation_time: float = 0.0,
        confidence: float = 0.0,
    ) -> SampleMetrics:
        """
        Complete sample tracking with results.

        Args:
            sample: The SampleMetrics object from start_sample
            correct: Whether prediction was correct
            predicted: Predicted answer
            ground_truth: Ground truth answer
            num_frames: Number of frames used
            input_tokens: Input token count
            output_tokens: Output token count
            perception_time: Time for perception (seconds)
            retrieval_time: Time for retrieval (seconds)
            generation_time: Time for LLM generation (seconds)
            confidence: Model confidence score

        Returns:
            Updated SampleMetrics
        """
        # Update results
        sample.correct = correct
        sample.predicted = predicted
        sample.ground_truth = ground_truth
        sample.confidence = confidence

        # Update compute metrics
        sample.num_frames = num_frames
        sample.input_tokens = input_tokens
        sample.output_tokens = output_tokens
        sample.total_tokens = input_tokens + output_tokens

        # Update timing
        sample.perception_time = perception_time
        sample.retrieval_time = retrieval_time
        sample.generation_time = generation_time

        if self._start_time is not None:
            sample.total_time = time.perf_counter() - self._start_time
        else:
            sample.total_time = perception_time + retrieval_time + generation_time

        # Get peak GPU memory
        if self._gpu_available:
            import torch
            sample.peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

        # Store sample
        self._samples.append(sample)
        self._current_sample = None
        self._start_time = None

        return sample

    def add_sample(self, sample: SampleMetrics) -> None:
        """Add a pre-populated sample."""
        self._samples.append(sample)

    def get_samples(
        self,
        benchmark: Optional[str] = None,
        model_config: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> list[SampleMetrics]:
        """Get samples with optional filtering."""
        samples = self._samples

        if benchmark:
            samples = [s for s in samples if s.benchmark == benchmark]
        if model_config:
            samples = [s for s in samples if s.model_config == model_config]
        if task_type:
            samples = [s for s in samples if s.task_type == task_type]

        return samples

    def compute_aggregates(self) -> dict[str, dict[str, AggregateMetrics]]:
        """
        Compute aggregate metrics per model config and benchmark.

        Returns:
            Nested dict: {model_config: {benchmark: AggregateMetrics}}
        """
        # Group samples
        groups: dict[tuple[str, str], list[SampleMetrics]] = {}
        for sample in self._samples:
            key = (sample.model_config, sample.benchmark)
            if key not in groups:
                groups[key] = []
            groups[key].append(sample)

        # Compute aggregates
        results: dict[str, dict[str, AggregateMetrics]] = {}

        for (model_config, benchmark), samples in groups.items():
            if model_config not in results:
                results[model_config] = {}

            agg = AggregateMetrics(
                model_config=model_config,
                benchmark=benchmark,
                num_samples=len(samples),
            )

            # Accuracy
            agg.correct_count = sum(1 for s in samples if s.correct)
            agg.accuracy = agg.correct_count / len(samples) if samples else 0.0

            # Averages
            agg.avg_frames = statistics.mean(s.num_frames for s in samples)
            agg.avg_input_tokens = statistics.mean(s.input_tokens for s in samples)
            agg.avg_output_tokens = statistics.mean(s.output_tokens for s in samples)
            agg.avg_total_tokens = statistics.mean(s.total_tokens for s in samples)

            agg.avg_perception_time = statistics.mean(s.perception_time for s in samples)
            agg.avg_retrieval_time = statistics.mean(s.retrieval_time for s in samples)
            agg.avg_generation_time = statistics.mean(s.generation_time for s in samples)
            agg.avg_total_time = statistics.mean(s.total_time for s in samples)

            agg.max_peak_vram_gb = max(s.peak_vram_gb for s in samples)
            agg.total_time_sec = sum(s.total_time for s in samples)

            # Task-specific accuracy
            task_groups: dict[str, list[SampleMetrics]] = {}
            for s in samples:
                if s.task_type not in task_groups:
                    task_groups[s.task_type] = []
                task_groups[s.task_type].append(s)

            for task_type, task_samples in task_groups.items():
                correct = sum(1 for s in task_samples if s.correct)
                agg.task_accuracy[task_type] = {
                    "accuracy": correct / len(task_samples),
                    "correct": correct,
                    "total": len(task_samples),
                }

            results[model_config][benchmark] = agg

        return results

    def print_summary(self) -> None:
        """Print summary of evaluation results."""
        aggregates = self.compute_aggregates()

        print()
        print("=" * 80)
        print("EVALUATION RESULTS SUMMARY")
        print("=" * 80)
        print()

        for model_config, benchmarks in aggregates.items():
            print(f"Model: {model_config}")
            print("-" * 60)

            for benchmark, agg in benchmarks.items():
                print(f"\n  Benchmark: {benchmark}")
                print(f"    Samples: {agg.num_samples}")
                print(f"    Accuracy: {agg.accuracy:.1%} ({agg.correct_count}/{agg.num_samples})")
                print(f"    Avg Frames: {agg.avg_frames:.1f}")
                print(f"    Avg Tokens: {agg.avg_total_tokens:.0f} (in: {agg.avg_input_tokens:.0f}, out: {agg.avg_output_tokens:.0f})")
                print(f"    Avg Time: {agg.avg_total_time:.2f}s (perception: {agg.avg_perception_time:.2f}s, gen: {agg.avg_generation_time:.2f}s)")
                print(f"    Peak VRAM: {agg.max_peak_vram_gb:.1f} GB")

                if agg.task_accuracy:
                    print("    Task Breakdown:")
                    for task, stats in agg.task_accuracy.items():
                        print(f"      - {task}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")

            print()

        print("=" * 80)

    def print_comparison_table(self) -> None:
        """Print comparison table across model configurations."""
        aggregates = self.compute_aggregates()

        # Get all benchmarks
        all_benchmarks = set()
        for benchmarks in aggregates.values():
            all_benchmarks.update(benchmarks.keys())

        print()
        print("=" * 100)
        print("MODEL COMPARISON")
        print("=" * 100)

        for benchmark in sorted(all_benchmarks):
            print(f"\nBenchmark: {benchmark}")
            print("-" * 90)
            print(f"{'Model':<20} {'Accuracy':<12} {'Frames':<10} {'Tokens':<12} {'Time (s)':<10} {'VRAM (GB)':<10}")
            print("-" * 90)

            for model_config in sorted(aggregates.keys()):
                if benchmark in aggregates[model_config]:
                    agg = aggregates[model_config][benchmark]
                    print(
                        f"{model_config:<20} "
                        f"{agg.accuracy:>10.1%} "
                        f"{agg.avg_frames:>8.1f} "
                        f"{agg.avg_total_tokens:>10.0f} "
                        f"{agg.avg_total_time:>8.2f} "
                        f"{agg.max_peak_vram_gb:>8.1f}"
                    )

        print("=" * 100)

    def save_results(self) -> Path:
        """
        Save all results to disk.

        Returns:
            Path to results directory
        """
        results_dir = self.output_dir / self.experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save per-sample results
        samples_path = results_dir / "samples.json"
        with open(samples_path, "w") as f:
            json.dump([s.to_dict() for s in self._samples], f, indent=2)

        # Save aggregates
        aggregates = self.compute_aggregates()
        aggregates_dict = {
            model: {bench: agg.to_dict() for bench, agg in benchmarks.items()}
            for model, benchmarks in aggregates.items()
        }

        aggregates_path = results_dir / "aggregates.json"
        with open(aggregates_path, "w") as f:
            json.dump(aggregates_dict, f, indent=2)

        # Save summary
        summary_path = results_dir / "summary.txt"
        with open(summary_path, "w") as f:
            import sys
            from io import StringIO

            # Capture print output
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            self.print_summary()
            self.print_comparison_table()

            f.write(sys.stdout.getvalue())
            sys.stdout = old_stdout

        logger.info(f"Results saved to {results_dir}")
        return results_dir

    def load_results(self, experiment_name: str) -> None:
        """Load results from a previous experiment."""
        results_dir = self.output_dir / experiment_name

        samples_path = results_dir / "samples.json"
        if samples_path.exists():
            with open(samples_path, "r") as f:
                samples_data = json.load(f)

            self._samples = []
            for s in samples_data:
                sample = SampleMetrics(
                    sample_id=s["sample_id"],
                    benchmark=s["benchmark"],
                    task_type=s["task_type"],
                    model_config=s["model_config"],
                    correct=s["correct"],
                    predicted=s["predicted"],
                    ground_truth=s["ground_truth"],
                    confidence=s.get("confidence", 0.0),
                    num_frames=s.get("num_frames", 0),
                    input_tokens=s.get("input_tokens", 0),
                    output_tokens=s.get("output_tokens", 0),
                    total_tokens=s.get("total_tokens", 0),
                    perception_time=s.get("perception_time", 0.0),
                    retrieval_time=s.get("retrieval_time", 0.0),
                    generation_time=s.get("generation_time", 0.0),
                    total_time=s.get("total_time", 0.0),
                    peak_vram_gb=s.get("peak_vram_gb", 0.0),
                    video_duration_sec=s.get("video_duration_sec", 0.0),
                    question=s.get("question", ""),
                    answer_format=s.get("answer_format", ""),
                )
                self._samples.append(sample)

            logger.info(f"Loaded {len(self._samples)} samples from {experiment_name}")


def create_metrics_tracker(
    output_dir: str = "results/eval",
    experiment_name: Optional[str] = None,
) -> MetricsTracker:
    """
    Factory function for creating a metrics tracker.

    Args:
        output_dir: Directory for results
        experiment_name: Optional experiment name

    Returns:
        Configured MetricsTracker
    """
    return MetricsTracker(output_dir=output_dir, experiment_name=experiment_name)
