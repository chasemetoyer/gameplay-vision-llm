#!/usr/bin/env python3
"""
Evaluation Harness for Gameplay Vision LLM.

This module provides a standardized evaluation framework for benchmarking
the system against various video game QA benchmarks.

Supported benchmark formats:
- VideoGameQA-Bench style (JSON with video paths and MCQ questions)
- PhysGame style (physical commonsense MCQ)
- Custom JSON format for internal evaluation

Usage:
    # Run evaluation on a benchmark
    python benchmarks/eval_harness.py \
        --benchmark data/benchmarks/videogameqa_subset.json \
        --output results/eval_results.json \
        --preset standard

    # Run with specific model configuration
    python benchmarks/eval_harness.py \
        --benchmark data/benchmarks/test_set.json \
        --preset light \
        --max-samples 100

Benchmark JSON format:
    {
        "name": "VideoGameQA-Test",
        "version": "1.0",
        "samples": [
            {
                "id": "sample_001",
                "video_path": "videos/gameplay_01.mp4",
                "question": "What weapon is the player using?",
                "options": ["Sword", "Bow", "Staff", "Axe"],
                "answer": "Sword",
                "category": "visual_recognition"
            }
        ]
    }
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    """A single evaluation sample."""

    id: str
    video_path: str
    question: str
    options: list[str]  # For MCQ questions
    answer: str  # Ground truth answer
    category: str = "general"
    timestamp: Optional[float] = None  # Optional specific timestamp
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result for a single evaluation sample."""

    sample_id: str
    predicted: str
    ground_truth: str
    correct: bool
    confidence: float
    latency_sec: float
    category: str
    error: Optional[str] = None


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""

    benchmark_name: str
    benchmark_version: str
    model_config: dict
    timestamp: str
    total_samples: int
    correct_count: int
    accuracy: float
    avg_latency_sec: float
    results_by_category: dict[str, dict]
    individual_results: list[EvalResult]
    errors: list[dict]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "benchmark_name": self.benchmark_name,
            "benchmark_version": self.benchmark_version,
            "model_config": self.model_config,
            "timestamp": self.timestamp,
            "summary": {
                "total_samples": self.total_samples,
                "correct_count": self.correct_count,
                "accuracy": self.accuracy,
                "avg_latency_sec": self.avg_latency_sec,
            },
            "results_by_category": self.results_by_category,
            "individual_results": [
                {
                    "sample_id": r.sample_id,
                    "predicted": r.predicted,
                    "ground_truth": r.ground_truth,
                    "correct": r.correct,
                    "confidence": r.confidence,
                    "latency_sec": r.latency_sec,
                    "category": r.category,
                    "error": r.error,
                }
                for r in self.individual_results
            ],
            "errors": self.errors,
        }

    def save(self, path: str | Path) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Results saved to {path}")

    def print_summary(self) -> None:
        """Print a summary of results to stdout."""
        print()
        print("=" * 60)
        print(f"EVALUATION RESULTS: {self.benchmark_name}")
        print("=" * 60)
        print(f"Timestamp: {self.timestamp}")
        print(f"Total Samples: {self.total_samples}")
        print(f"Correct: {self.correct_count}")
        print(f"Accuracy: {self.accuracy:.2%}")
        print(f"Avg Latency: {self.avg_latency_sec:.2f}s")
        print()
        print("Results by Category:")
        for cat, stats in self.results_by_category.items():
            print(f"  {cat}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
        print()
        if self.errors:
            print(f"Errors: {len(self.errors)}")
        print("=" * 60)


class BenchmarkLoader:
    """Loads benchmark data from various formats."""

    @staticmethod
    def load(path: str | Path) -> tuple[str, str, list[EvalSample]]:
        """
        Load benchmark from JSON file.

        Args:
            path: Path to benchmark JSON file

        Returns:
            Tuple of (benchmark_name, version, samples)
        """
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)

        name = data.get("name", path.stem)
        version = data.get("version", "1.0")

        samples = []
        for item in data.get("samples", []):
            sample = EvalSample(
                id=item.get("id", str(len(samples))),
                video_path=item["video_path"],
                question=item["question"],
                options=item.get("options", []),
                answer=item["answer"],
                category=item.get("category", "general"),
                timestamp=item.get("timestamp"),
                metadata=item.get("metadata", {}),
            )
            samples.append(sample)

        logger.info(f"Loaded {len(samples)} samples from {path}")
        return name, version, samples

    @staticmethod
    def create_example_benchmark(output_path: str | Path) -> None:
        """
        Create an example benchmark file for testing.

        Args:
            output_path: Where to save the example benchmark
        """
        example = {
            "name": "Example Gameplay QA Benchmark",
            "version": "1.0",
            "description": "Example benchmark for testing the evaluation harness",
            "samples": [
                {
                    "id": "example_001",
                    "video_path": "data/raw_videos/test_clip.mp4",
                    "question": "What type of game is being played?",
                    "options": ["Action RPG", "Puzzle", "Racing", "Sports"],
                    "answer": "Action RPG",
                    "category": "game_recognition"
                },
                {
                    "id": "example_002",
                    "video_path": "data/raw_videos/test_clip.mp4",
                    "question": "Is the player character visible on screen?",
                    "options": ["Yes", "No"],
                    "answer": "Yes",
                    "category": "visual_recognition"
                },
                {
                    "id": "example_003",
                    "video_path": "data/raw_videos/test_clip.mp4",
                    "question": "What action is the player performing?",
                    "options": ["Attacking", "Walking", "Jumping", "Standing"],
                    "answer": "Attacking",
                    "category": "action_recognition"
                },
            ]
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(example, f, indent=2)
        logger.info(f"Example benchmark created at {output_path}")


class EvaluationHarness:
    """
    Main evaluation harness for running benchmarks.

    This class coordinates loading benchmarks, running inference,
    and computing metrics.
    """

    def __init__(
        self,
        preset: str = "light",
        model_path: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        Initialize the evaluation harness.

        Args:
            preset: Configuration preset ('light', 'standard', 'full')
            model_path: Optional path to custom model weights
            device: Compute device ('cuda' or 'cpu')
        """
        self.preset = preset
        self.model_path = model_path
        self.device = device
        self._inference_fn = None
        self._config = None

    def _load_inference_pipeline(self) -> None:
        """Load the inference pipeline (lazy initialization)."""
        if self._inference_fn is not None:
            return

        try:
            from src.config.presets import load_preset
            self._config = load_preset(self.preset)
            logger.info(f"Loaded preset: {self.preset}")

            # For now, use a mock inference function
            # In production, this would load the actual model
            self._inference_fn = self._mock_inference

            logger.info("Inference pipeline initialized (mock mode)")
        except Exception as e:
            logger.warning(f"Could not load full pipeline: {e}")
            self._inference_fn = self._mock_inference

    def _mock_inference(
        self,
        video_path: str,
        question: str,
        options: list[str],
        timestamp: Optional[float] = None,
    ) -> tuple[str, float]:
        """
        Mock inference function for testing.

        Returns:
            Tuple of (predicted_answer, confidence)
        """
        # Simple mock: return first option with low confidence
        import random
        predicted = random.choice(options) if options else "Unknown"
        confidence = random.uniform(0.3, 0.7)
        return predicted, confidence

    def run_sample(
        self,
        sample: EvalSample,
    ) -> EvalResult:
        """
        Run evaluation on a single sample.

        Args:
            sample: EvalSample to evaluate

        Returns:
            EvalResult with prediction and metrics
        """
        self._load_inference_pipeline()

        start_time = time.time()
        error = None

        try:
            predicted, confidence = self._inference_fn(
                video_path=sample.video_path,
                question=sample.question,
                options=sample.options,
                timestamp=sample.timestamp,
            )
        except Exception as e:
            predicted = ""
            confidence = 0.0
            error = str(e)

        latency = time.time() - start_time

        # Determine correctness
        correct = predicted.lower().strip() == sample.answer.lower().strip()

        return EvalResult(
            sample_id=sample.id,
            predicted=predicted,
            ground_truth=sample.answer,
            correct=correct,
            confidence=confidence,
            latency_sec=latency,
            category=sample.category,
            error=error,
        )

    def run_benchmark(
        self,
        benchmark_path: str | Path,
        max_samples: Optional[int] = None,
        categories: Optional[list[str]] = None,
    ) -> BenchmarkResults:
        """
        Run evaluation on a full benchmark.

        Args:
            benchmark_path: Path to benchmark JSON file
            max_samples: Optional limit on number of samples
            categories: Optional list of categories to include

        Returns:
            BenchmarkResults with aggregated metrics
        """
        # Load benchmark
        name, version, samples = BenchmarkLoader.load(benchmark_path)

        # Filter by categories if specified
        if categories:
            samples = [s for s in samples if s.category in categories]

        # Limit samples if specified
        if max_samples and len(samples) > max_samples:
            samples = samples[:max_samples]

        logger.info(f"Running evaluation on {len(samples)} samples")

        # Run evaluation
        results = []
        errors = []

        for i, sample in enumerate(samples):
            logger.info(f"Processing sample {i+1}/{len(samples)}: {sample.id}")
            result = self.run_sample(sample)
            results.append(result)

            if result.error:
                errors.append({
                    "sample_id": sample.id,
                    "error": result.error,
                })

        # Compute metrics
        correct_count = sum(1 for r in results if r.correct)
        total = len(results)
        accuracy = correct_count / total if total > 0 else 0.0
        avg_latency = sum(r.latency_sec for r in results) / total if total > 0 else 0.0

        # Compute metrics by category
        results_by_category = {}
        for result in results:
            cat = result.category
            if cat not in results_by_category:
                results_by_category[cat] = {"correct": 0, "total": 0}
            results_by_category[cat]["total"] += 1
            if result.correct:
                results_by_category[cat]["correct"] += 1

        for cat in results_by_category:
            stats = results_by_category[cat]
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0

        # Build results object
        return BenchmarkResults(
            benchmark_name=name,
            benchmark_version=version,
            model_config={
                "preset": self.preset,
                "model_path": self.model_path,
                "device": self.device,
            },
            timestamp=datetime.utcnow().isoformat() + "Z",
            total_samples=total,
            correct_count=correct_count,
            accuracy=accuracy,
            avg_latency_sec=avg_latency,
            results_by_category=results_by_category,
            individual_results=results,
            errors=errors,
        )


def run_baseline_comparison(
    benchmark_path: str,
    output_dir: str,
    presets: list[str] = None,
) -> dict[str, BenchmarkResults]:
    """
    Run evaluation with multiple presets for baseline comparison.

    Args:
        benchmark_path: Path to benchmark JSON
        output_dir: Directory for output files
        presets: List of presets to compare (default: ['light', 'standard'])

    Returns:
        Dictionary of preset name to BenchmarkResults
    """
    if presets is None:
        presets = ["light", "standard"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for preset in presets:
        logger.info(f"Running evaluation with preset: {preset}")
        harness = EvaluationHarness(preset=preset)
        results = harness.run_benchmark(benchmark_path)

        # Save individual results
        results.save(output_dir / f"results_{preset}.json")
        all_results[preset] = results

    # Create comparison summary
    comparison = {
        "presets": {},
        "benchmark": benchmark_path,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    for preset, results in all_results.items():
        comparison["presets"][preset] = {
            "accuracy": results.accuracy,
            "avg_latency": results.avg_latency_sec,
            "total_samples": results.total_samples,
        }

    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Print comparison
    print()
    print("=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)
    for preset, results in all_results.items():
        print(f"{preset:12} | Accuracy: {results.accuracy:.2%} | Latency: {results.avg_latency_sec:.2f}s")
    print("=" * 60)

    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluation harness for Gameplay Vision LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/eval_results.json",
        help="Path for output results JSON",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="light",
        choices=["light", "standard", "full"],
        help="Configuration preset to use",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        help="Categories to include (space-separated)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison across multiple presets",
    )
    parser.add_argument(
        "--create-example",
        type=str,
        help="Create example benchmark at specified path",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create example benchmark if requested
    if args.create_example:
        BenchmarkLoader.create_example_benchmark(args.create_example)
        return

    # Run evaluation
    if args.benchmark:
        if args.compare:
            run_baseline_comparison(
                benchmark_path=args.benchmark,
                output_dir=Path(args.output).parent,
            )
        else:
            harness = EvaluationHarness(preset=args.preset)
            results = harness.run_benchmark(
                benchmark_path=args.benchmark,
                max_samples=args.max_samples,
                categories=args.categories,
            )
            results.save(args.output)
            results.print_summary()
    else:
        parser.print_help()
        print()
        print("To create an example benchmark:")
        print("  python benchmarks/eval_harness.py --create-example data/benchmarks/example.json")


if __name__ == "__main__":
    main()
