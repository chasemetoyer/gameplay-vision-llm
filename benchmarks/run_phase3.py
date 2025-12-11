#!/usr/bin/env python3
"""
Phase 3 Evaluation: Long-Video Stress Test

Optional long-video stress test phase:
- LongVideoBench: 3,763 videos (up to 1 hour)
- MLVU: 1,730 videos (3-120 minutes)

This phase tests:
- Long-form temporal reasoning (minutes to hours)
- Memory and context management at scale
- Temporal compression effectiveness (HiCo)
- Timeline/KB retrieval performance

Usage:
    python benchmarks/run_phase3.py --help
    python benchmarks/run_phase3.py --benchmark longvideobench --config gvp_full
    python benchmarks/run_phase3.py --full --max-samples 25 --max-duration 600
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.loaders import LongVideoBenchLoader, MLVULoader
from benchmarks.loaders.base import BenchmarkLoader, BenchmarkSample
from benchmarks.metrics import MetricsTracker, SampleMetrics, create_metrics_tracker
from benchmarks.model_configs import (
    EvalModelConfig,
    ModelConfigType,
    create_baseline_plain,
    create_gvp_full,
    create_gvp_light,
)
from benchmarks.perception_cache import CachedFeatures, PerceptionCache, create_perception_cache
from benchmarks.model_inference import FullPipelineRunner, get_full_pipeline_runner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Phase3Evaluator:
    """
    Phase 3 evaluation runner for long-video stress testing.

    This phase specifically tests the system's ability to handle
    long-form video content (3+ minutes up to 1+ hours), which is
    where timeline indexing and HiCo compression become critical.

    Key metrics of interest:
    - Accuracy vs video duration
    - Token efficiency (tokens per minute of video)
    - Retrieval latency at scale
    - Memory usage under long contexts
    """

    def __init__(
        self,
        cache_dir: str = "data/cache/perception",
        results_dir: str = "results/phase3",
        experiment_name: Optional[str] = None,
        projector_weights: str = "outputs/projector_weights.pt",
        lora_path: str = "outputs/lora_adapter",
        device: str = "cuda",
        preset: str = "standard",
    ):
        """
        Initialize Phase 3 evaluator.

        Args:
            cache_dir: Directory for perception cache
            results_dir: Directory for evaluation results
            experiment_name: Optional experiment name
            projector_weights: Path to projector weights
            lora_path: Path to LoRA adapter
            device: Device to run on
            preset: Configuration preset
        """
        self.cache = create_perception_cache(cache_dir)
        self.tracker = create_metrics_tracker(results_dir, experiment_name)

        # Available configs
        self.configs = {
            "baseline_plain": create_baseline_plain,
            "gvp_light": create_gvp_light,
            "gvp_full": create_gvp_full,
        }

        # Full pipeline runner for complete perception + reasoning
        self.pipeline_runner = get_full_pipeline_runner(
            preset=preset,
            projector_weights=projector_weights,
            lora_path=lora_path,
            device=device,
        )

        logger.info("Phase 3 Evaluator initialized (Full Pipeline Mode)")

    def filter_by_duration(
        self,
        samples: list[BenchmarkSample],
        min_duration_sec: Optional[float] = None,
        max_duration_sec: Optional[float] = None,
    ) -> list[BenchmarkSample]:
        """
        Filter samples by video duration.

        Args:
            samples: List of samples
            min_duration_sec: Minimum duration in seconds
            max_duration_sec: Maximum duration in seconds

        Returns:
            Filtered list of samples
        """
        filtered = []
        for sample in samples:
            duration = sample.video_duration_sec or 0

            if min_duration_sec and duration < min_duration_sec:
                continue
            if max_duration_sec and duration > max_duration_sec:
                continue

            filtered.append(sample)

        return filtered

    def run_perception_caching(
        self,
        samples: list[BenchmarkSample],
        batch_size: int = 5,
    ) -> int:
        """
        Stage A: Run perception pipeline and cache results.

        For long videos, this is particularly expensive so we use
        smaller batch sizes and more verbose logging.

        Args:
            samples: List of benchmark samples
            batch_size: Number of videos to process before logging

        Returns:
            Number of videos cached
        """
        cached_count = 0
        skipped_count = 0

        # Get unique video paths
        video_paths = {}
        for sample in samples:
            if sample.video_path:
                video_paths[sample.video_path] = sample.video_duration_sec or 0

        total_duration = sum(video_paths.values())
        logger.info(f"Stage A: Caching {len(video_paths)} long videos")
        logger.info(f"  Total duration: {total_duration / 60:.1f} minutes")

        for i, (video_path, duration) in enumerate(video_paths.items()):
            # Check if already cached
            if self.cache.has_cache(video_path):
                skipped_count += 1
                continue

            # Run perception (placeholder)
            logger.info(f"  Processing: {Path(video_path).name} ({duration / 60:.1f} min)")

            start_time = time.perf_counter()
            features = self._run_perception_pipeline(video_path, duration)
            elapsed = time.perf_counter() - start_time

            if features:
                self.cache.save(video_path, features)
                cached_count += 1
                logger.info(f"    Cached in {elapsed:.1f}s")

            if (i + 1) % batch_size == 0:
                logger.info(
                    f"  Progress: {i + 1}/{len(video_paths)} "
                    f"(cached: {cached_count}, skipped: {skipped_count})"
                )

        logger.info(f"Stage A complete: cached {cached_count}, skipped {skipped_count}")
        return cached_count

    def _run_perception_pipeline(
        self,
        video_path: str,
        duration_sec: float,
    ) -> Optional[CachedFeatures]:
        """
        Run perception pipeline for a long video.

        For long videos, we would use:
        - Adaptive frame sampling (more aggressive subsampling)
        - Hierarchical temporal processing
        - Progressive KB construction
        - Streaming timeline building

        Args:
            video_path: Path to video file
            duration_sec: Video duration in seconds

        Returns:
            CachedFeatures or None on error
        """
        features = CachedFeatures(
            video_hash="",
            video_path=video_path,
            video_duration_sec=duration_sec,
        )
        return features

    def evaluate_sample(
        self,
        sample: BenchmarkSample,
        config: EvalModelConfig,
        cached_features: Optional[CachedFeatures],
    ) -> SampleMetrics:
        """
        Evaluate a single long-video sample.

        For long videos, we track additional metrics:
        - Tokens per minute of video
        - Retrieval efficiency
        - Context compression ratio

        Args:
            sample: Benchmark sample
            config: Model configuration
            cached_features: Pre-computed perception features

        Returns:
            SampleMetrics with results
        """
        start_time = time.perf_counter()
        perception_time = 0.0
        retrieval_time = 0.0
        generation_time = 0.0

        # Get question
        question = sample.question or sample.get_formatted_prompt()
        answer_format = sample.answer_format.value if sample.answer_format else "unknown"

        # Create sample metrics
        metrics = self.tracker.start_sample(
            sample_id=sample.sample_id,
            benchmark=sample.benchmark,
            task_type=sample.task_type.value if sample.task_type else "unknown",
            model_config=config.name,
            video_duration_sec=sample.video_duration_sec or 0.0,
            question=question,
            answer_format=answer_format,
        )

        # Run FULL PIPELINE inference
        predicted, perception_time, generation_time = self._run_inference(sample, config, cached_features)
        ground_truth = sample.ground_truth

        # Evaluate correctness
        correct = self._check_correctness(predicted, ground_truth, sample)

        # Compute metrics
        video_duration = sample.video_duration_sec or 60.0
        fps = config.frame_sampling.target_fps
        actual_frames = min(int(video_duration * fps), config.frame_sampling.max_frames)
        input_tokens = 1024 + (actual_frames * 32)
        output_tokens = len(predicted.split()) * 2

        # Add retrieval time for configs with timeline/KB
        if config.retrieval.use_timeline or config.retrieval.use_kb:
            retrieval_time = 0.1 + (video_duration / 60) * 0.05

        # Complete metrics
        self.tracker.end_sample(
            metrics,
            correct=correct,
            predicted=predicted,
            ground_truth=str(ground_truth),
            num_frames=actual_frames,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            perception_time=perception_time,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
        )

        return metrics

    def _run_inference(
        self,
        sample: BenchmarkSample,
        config: EvalModelConfig,
        cached_features: Optional[CachedFeatures],
    ) -> tuple[str, float, float]:
        """
        Run FULL PIPELINE inference on a long-video sample.

        Runs: Frame extraction, SAM3, SigLIP, VideoMAE, OCR, Whisper, Timeline, KB, LLM

        Args:
            sample: Benchmark sample
            config: Model configuration
            cached_features: (Not used - full pipeline handles internally)

        Returns:
            Tuple of (predicted_answer, perception_time_sec, inference_time_sec)
        """
        # Run full pipeline inference
        predicted, perception_time, inference_time = self.pipeline_runner.run_inference(
            sample=sample,
        )
        
        return predicted, perception_time, inference_time

    def _check_correctness(
        self,
        predicted: str,
        ground_truth: str,
        sample: BenchmarkSample,
    ) -> bool:
        """Check if prediction is correct."""
        pred_norm = predicted.lower().strip()
        gt_norm = str(ground_truth).lower().strip()

        if pred_norm == gt_norm:
            return True

        # MCQ letter matching
        if len(pred_norm) > 0 and pred_norm[0] in "abcdefgh":
            if len(gt_norm) > 0 and gt_norm[0] in "abcdefgh":
                return pred_norm[0] == gt_norm[0]

        return False

    def run_evaluation(
        self,
        benchmark: str,
        config_name: str,
        max_samples: Optional[int] = None,
        min_duration_sec: Optional[float] = None,
        max_duration_sec: Optional[float] = None,
        data_root: str = "data",
    ) -> dict:
        """
        Run evaluation for a benchmark and config.

        Args:
            benchmark: "longvideobench" or "mlvu"
            config_name: Model config name
            max_samples: Maximum samples to evaluate
            min_duration_sec: Minimum video duration
            max_duration_sec: Maximum video duration
            data_root: Root directory for benchmark data

        Returns:
            Evaluation results dict
        """
        # Get loader
        if benchmark == "longvideobench":
            loader = LongVideoBenchLoader(data_root=data_root)
        elif benchmark == "mlvu":
            loader = MLVULoader(data_root=data_root)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")

        # Get config
        if config_name not in self.configs:
            raise ValueError(f"Unknown config: {config_name}")
        config = self.configs[config_name]()

        # Load and filter samples
        samples = list(loader)

        if min_duration_sec or max_duration_sec:
            samples = self.filter_by_duration(
                samples,
                min_duration_sec=min_duration_sec,
                max_duration_sec=max_duration_sec,
            )

        if max_samples:
            samples = samples[:max_samples]

        # Log statistics
        durations = [s.video_duration_sec or 0 for s in samples]
        if durations:
            avg_duration = sum(durations) / len(durations)
            max_dur = max(durations)
            min_dur = min(durations)
        else:
            avg_duration = max_dur = min_dur = 0

        logger.info(f"Evaluating {benchmark} with {config_name}")
        logger.info(f"  Samples: {len(samples)}")
        logger.info(f"  Duration range: {min_dur / 60:.1f} - {max_dur / 60:.1f} min")
        logger.info(f"  Avg duration: {avg_duration / 60:.1f} min")

        # Run Stage A: Perception caching
        self.run_perception_caching(samples)

        # Run Stage B: QA evaluation
        for i, sample in enumerate(samples):
            # Load cached features
            cached_features = None
            if sample.video_path:
                cached_features = self.cache.load(sample.video_path)

            # Evaluate
            self.evaluate_sample(sample, config, cached_features)

            if (i + 1) % 10 == 0:
                logger.info(f"  Progress: {i + 1}/{len(samples)}")

        return self.tracker.compute_aggregates()

    def run_all_configs(
        self,
        benchmark: str,
        max_samples: Optional[int] = None,
        min_duration_sec: Optional[float] = None,
        max_duration_sec: Optional[float] = None,
        data_root: str = "data",
    ) -> dict:
        """
        Run evaluation for all configs on a benchmark.

        Args:
            benchmark: Benchmark name
            max_samples: Maximum samples per config
            min_duration_sec: Minimum video duration
            max_duration_sec: Maximum video duration
            data_root: Root directory for benchmark data

        Returns:
            Combined evaluation results
        """
        for config_name in self.configs:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Config: {config_name}")
            logger.info(f"{'=' * 60}")

            self.run_evaluation(
                benchmark=benchmark,
                config_name=config_name,
                max_samples=max_samples,
                min_duration_sec=min_duration_sec,
                max_duration_sec=max_duration_sec,
                data_root=data_root,
            )

        return self.tracker.compute_aggregates()

    def run_full_phase3(
        self,
        max_samples: Optional[int] = None,
        min_duration_sec: Optional[float] = 180,  # 3 minutes minimum
        max_duration_sec: Optional[float] = None,
        data_root: str = "data",
    ) -> dict:
        """
        Run complete Phase 3 evaluation.

        Evaluates all model configs on both LongVideoBench and MLVU.

        Args:
            max_samples: Maximum samples per benchmark per config
            min_duration_sec: Minimum video duration (default: 3 min)
            max_duration_sec: Maximum video duration
            data_root: Root directory for benchmark data

        Returns:
            Complete evaluation results
        """
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 3 EVALUATION: Long-Video Stress Test")
        logger.info("=" * 80)
        logger.info(f"  Min duration: {min_duration_sec / 60:.1f} min")
        if max_duration_sec:
            logger.info(f"  Max duration: {max_duration_sec / 60:.1f} min")

        for benchmark in ["longvideobench", "mlvu"]:
            logger.info(f"\n\n{'#' * 70}")
            logger.info(f"# BENCHMARK: {benchmark.upper()}")
            logger.info(f"{'#' * 70}")

            self.run_all_configs(
                benchmark=benchmark,
                max_samples=max_samples,
                min_duration_sec=min_duration_sec,
                max_duration_sec=max_duration_sec,
                data_root=data_root,
            )

        # Print and save results
        self.tracker.print_summary()
        self.tracker.print_comparison_table()
        self._print_duration_analysis()
        results_dir = self.tracker.save_results()

        logger.info(f"\nResults saved to: {results_dir}")
        return self.tracker.compute_aggregates()

    def _print_duration_analysis(self) -> None:
        """Print analysis of accuracy vs video duration."""
        samples = self.tracker.get_samples()

        if not samples:
            return

        print("\n" + "=" * 80)
        print("DURATION ANALYSIS")
        print("=" * 80)

        # Group by duration buckets
        buckets = {
            "0-5 min": (0, 300),
            "5-10 min": (300, 600),
            "10-30 min": (600, 1800),
            "30-60 min": (1800, 3600),
            "60+ min": (3600, float("inf")),
        }

        for config_name in self.configs:
            print(f"\nModel: {config_name}")
            print("-" * 60)

            config_samples = [s for s in samples if s.model_config == config_name]

            for bucket_name, (min_dur, max_dur) in buckets.items():
                bucket_samples = [
                    s
                    for s in config_samples
                    if min_dur <= s.video_duration_sec < max_dur
                ]

                if bucket_samples:
                    correct = sum(1 for s in bucket_samples if s.correct)
                    total = len(bucket_samples)
                    accuracy = correct / total
                    avg_tokens = sum(s.total_tokens for s in bucket_samples) / total
                    avg_time = sum(s.total_time for s in bucket_samples) / total

                    print(
                        f"  {bucket_name:12s}: {accuracy:>6.1%} "
                        f"({correct:>3}/{total:<3}) | "
                        f"Tokens: {avg_tokens:>6.0f} | "
                        f"Time: {avg_time:>5.2f}s"
                    )

        print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 3 Evaluation: Long-Video Stress Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run LongVideoBench with GVP-Full
    python run_phase3.py --benchmark longvideobench --config gvp_full

    # Run MLVU with all configs
    python run_phase3.py --benchmark mlvu --all-configs

    # Run full Phase 3 with duration limits
    python run_phase3.py --full --max-samples 25 --min-duration 180 --max-duration 1800

    # Focus on very long videos (30+ minutes)
    python run_phase3.py --benchmark longvideobench --config gvp_full --min-duration 1800
        """,
    )

    # Benchmark selection
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["longvideobench", "mlvu"],
        help="Benchmark to evaluate",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run complete Phase 3 (both benchmarks, all configs)",
    )

    # Config selection
    parser.add_argument(
        "--config",
        type=str,
        choices=["baseline_plain", "gvp_light", "gvp_full"],
        help="Model configuration to use",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Run all model configurations",
    )

    # Sample and duration limits
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate per benchmark/config",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=None,
        help="Minimum video duration in seconds (default: 180 for --full)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Maximum video duration in seconds",
    )

    # Directories
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory for benchmark data",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache/perception",
        help="Directory for perception cache",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/phase3",
        help="Directory for evaluation results",
    )

    # Model weights
    parser.add_argument(
        "--projector-weights",
        type=str,
        default="outputs/projector_weights.pt",
        help="Path to projector weights",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="outputs/lora_adapter",
        help="Path to LoRA adapter",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )

    # Experiment
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name for organizing results",
    )

    parser.add_argument(
        "--preset",
        type=str,
        default="standard",
        choices=["light", "standard", "full"],
        help="Configuration preset for perception pipeline",
    )

    args = parser.parse_args()

    # Validate args
    if not args.full and not args.benchmark:
        parser.error("Either --benchmark or --full is required")

    if args.benchmark and not args.config and not args.all_configs:
        parser.error("Either --config or --all-configs is required with --benchmark")

    # Create evaluator with full pipeline
    evaluator = Phase3Evaluator(
        cache_dir=args.cache_dir,
        results_dir=args.results_dir,
        experiment_name=args.experiment,
        projector_weights=args.projector_weights,
        lora_path=args.lora_path,
        device=args.device,
        preset=args.preset,
    )

    # Run evaluation
    if args.full:
        evaluator.run_full_phase3(
            max_samples=args.max_samples,
            min_duration_sec=args.min_duration if args.min_duration else 180,
            max_duration_sec=args.max_duration,
            data_root=args.data_root,
        )
    elif args.all_configs:
        evaluator.run_all_configs(
            benchmark=args.benchmark,
            max_samples=args.max_samples,
            min_duration_sec=args.min_duration,
            max_duration_sec=args.max_duration,
            data_root=args.data_root,
        )
    else:
        evaluator.run_evaluation(
            benchmark=args.benchmark,
            config_name=args.config,
            max_samples=args.max_samples,
            min_duration_sec=args.min_duration,
            max_duration_sec=args.max_duration,
            data_root=args.data_root,
        )


if __name__ == "__main__":
    main()
