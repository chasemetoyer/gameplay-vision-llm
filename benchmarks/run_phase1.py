#!/usr/bin/env python3
"""
Phase 1 Evaluation: GlitchBench + PhysGame

Cheap, game-specific evaluation phase:
- GlitchBench: 593 glitches from gaming videos (CVPR 2024)
- PhysGame: 880 physics anomaly videos across 4 domains

This phase tests:
- Visual glitch/anomaly detection
- Physics understanding
- Short-form video comprehension (5-30 seconds)

Usage:
    python benchmarks/run_phase1.py --help
    python benchmarks/run_phase1.py --benchmark glitchbench --config baseline_plain
    python benchmarks/run_phase1.py --all-configs --max-samples 100
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

from benchmarks.loaders import GlitchBenchLoader, PhysGameLoader
from benchmarks.loaders.base import BenchmarkLoader, BenchmarkSample
from benchmarks.metrics import MetricsTracker, SampleMetrics, create_metrics_tracker
from benchmarks.model_configs import (
    EvalModelConfig,
    ModelConfigType,
    create_baseline_plain,
    create_gvp_full,
    create_gvp_light,
    get_model_config,
)
from benchmarks.perception_cache import CachedFeatures, PerceptionCache, create_perception_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Phase1Evaluator:
    """
    Phase 1 evaluation runner.

    Handles:
    - Two-stage evaluation (cache perception, then run QA)
    - Multiple model configuration comparison
    - Comprehensive metrics tracking
    """

    def __init__(
        self,
        cache_dir: str = "data/cache/perception",
        results_dir: str = "results/phase1",
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize Phase 1 evaluator.

        Args:
            cache_dir: Directory for perception cache
            results_dir: Directory for evaluation results
            experiment_name: Optional experiment name
        """
        self.cache = create_perception_cache(cache_dir)
        self.tracker = create_metrics_tracker(results_dir, experiment_name)

        # Available configs
        self.configs = {
            "baseline_plain": create_baseline_plain,
            "gvp_light": create_gvp_light,
            "gvp_full": create_gvp_full,
        }

        logger.info("Phase 1 Evaluator initialized")

    def run_perception_caching(
        self,
        loader: BenchmarkLoader,
        max_samples: Optional[int] = None,
    ) -> int:
        """
        Stage A: Run perception pipeline and cache results.

        Args:
            loader: Benchmark data loader
            max_samples: Maximum samples to process (None for all)

        Returns:
            Number of videos cached
        """
        samples = list(loader)
        if max_samples:
            samples = samples[:max_samples]

        cached_count = 0
        skipped_count = 0

        logger.info(f"Stage A: Caching perception for {len(samples)} samples")

        for i, sample in enumerate(samples):
            video_path = sample.video_path or sample.image_path

            if not video_path:
                logger.warning(f"Sample {sample.sample_id} has no video/image path")
                continue

            # Check if already cached
            if self.cache.has_cache(video_path):
                skipped_count += 1
                continue

            # Run perception (placeholder - actual implementation would call perception pipeline)
            features = self._run_perception_pipeline(sample)

            if features:
                self.cache.save(video_path, features)
                cached_count += 1

            if (i + 1) % 50 == 0:
                logger.info(f"  Progress: {i + 1}/{len(samples)} (cached: {cached_count}, skipped: {skipped_count})")

        logger.info(f"Stage A complete: cached {cached_count}, skipped {skipped_count} (already cached)")
        return cached_count

    def _run_perception_pipeline(self, sample: BenchmarkSample) -> Optional[CachedFeatures]:
        """
        Run perception pipeline for a sample.

        This is a placeholder - actual implementation would:
        1. Load video/image
        2. Extract frames
        3. Run visual encoders (SigLIP, SAM3, VideoMAE)
        4. Run OCR/ASR
        5. Build timeline and KB

        Args:
            sample: Benchmark sample

        Returns:
            CachedFeatures or None on error
        """
        video_path = sample.video_path or sample.image_path
        if not video_path:
            return None

        # Placeholder features
        features = CachedFeatures(
            video_hash="",
            video_path=video_path,
            video_duration_sec=sample.video_duration_sec or 10.0,
        )

        return features

    def evaluate_sample(
        self,
        sample: BenchmarkSample,
        config: EvalModelConfig,
        cached_features: Optional[CachedFeatures],
    ) -> SampleMetrics:
        """
        Evaluate a single sample with a model configuration.

        Args:
            sample: Benchmark sample
            config: Model configuration
            cached_features: Pre-computed perception features

        Returns:
            SampleMetrics with results
        """
        # Start timing
        start_time = time.perf_counter()
        perception_time = 0.0
        retrieval_time = 0.0
        generation_time = 0.0

        # Get question and answer format
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

        # Placeholder inference
        # Actual implementation would:
        # 1. Load cached features or run perception
        # 2. Build context from timeline/KB
        # 3. Run LLM inference
        # 4. Parse and evaluate answer

        predicted = self._run_inference(sample, config, cached_features)
        ground_truth = sample.ground_truth

        # Evaluate correctness
        correct = self._check_correctness(predicted, ground_truth, sample)

        # Estimate compute metrics (placeholder values)
        num_frames = config.frame_sampling.max_frames
        input_tokens = 1024  # Placeholder
        output_tokens = 32  # Placeholder

        generation_time = time.perf_counter() - start_time

        # Complete metrics
        self.tracker.end_sample(
            metrics,
            correct=correct,
            predicted=predicted,
            ground_truth=str(ground_truth),
            num_frames=num_frames,
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
    ) -> str:
        """
        Run model inference on a sample.

        This is a placeholder - actual implementation would:
        1. Load model with config
        2. Build input from cached features
        3. Generate response
        4. Parse answer

        Args:
            sample: Benchmark sample
            config: Model configuration
            cached_features: Cached perception features

        Returns:
            Predicted answer string
        """
        # Placeholder - return dummy answer
        # In real implementation, this would call the model
        return "yes" if sample.ground_truth else "no"

    def _check_correctness(
        self,
        predicted: str,
        ground_truth: str,
        sample: BenchmarkSample,
    ) -> bool:
        """
        Check if prediction is correct.

        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            sample: Original sample for format info

        Returns:
            True if correct
        """
        # Normalize strings
        pred_norm = predicted.lower().strip()
        gt_norm = str(ground_truth).lower().strip()

        # Direct match
        if pred_norm == gt_norm:
            return True

        # Binary answer normalization
        yes_variants = {"yes", "true", "1", "correct", "glitch", "anomaly"}
        no_variants = {"no", "false", "0", "incorrect", "normal", "none"}

        pred_is_yes = pred_norm in yes_variants
        pred_is_no = pred_norm in no_variants
        gt_is_yes = gt_norm in yes_variants
        gt_is_no = gt_norm in no_variants

        if pred_is_yes and gt_is_yes:
            return True
        if pred_is_no and gt_is_no:
            return True

        return False

    def run_evaluation(
        self,
        benchmark: str,
        config_name: str,
        max_samples: Optional[int] = None,
        data_root: str = "data",
    ) -> dict:
        """
        Run evaluation for a single benchmark and config.

        Args:
            benchmark: "glitchbench" or "physgame"
            config_name: "baseline_plain", "gvp_light", or "gvp_full"
            max_samples: Maximum samples to evaluate
            data_root: Root directory for benchmark data

        Returns:
            Evaluation results dict
        """
        # Get loader
        if benchmark == "glitchbench":
            loader = GlitchBenchLoader(data_root=data_root)
        elif benchmark == "physgame":
            loader = PhysGameLoader(data_root=data_root)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")

        # Get config
        if config_name not in self.configs:
            raise ValueError(f"Unknown config: {config_name}")
        config = self.configs[config_name]()

        # Load samples
        samples = list(loader)
        if max_samples:
            samples = samples[:max_samples]

        logger.info(f"Evaluating {benchmark} with {config_name} ({len(samples)} samples)")

        # Run Stage A: Perception caching
        self.run_perception_caching(loader, max_samples)

        # Run Stage B: QA evaluation
        for i, sample in enumerate(samples):
            video_path = sample.video_path or sample.image_path

            # Load cached features
            cached_features = None
            if video_path:
                cached_features = self.cache.load(video_path)

            # Evaluate
            self.evaluate_sample(sample, config, cached_features)

            if (i + 1) % 50 == 0:
                logger.info(f"  Progress: {i + 1}/{len(samples)}")

        return self.tracker.compute_aggregates()

    def run_all_configs(
        self,
        benchmark: str,
        max_samples: Optional[int] = None,
        data_root: str = "data",
    ) -> dict:
        """
        Run evaluation for all model configs on a benchmark.

        Args:
            benchmark: "glitchbench" or "physgame"
            max_samples: Maximum samples per config
            data_root: Root directory for benchmark data

        Returns:
            Combined evaluation results
        """
        for config_name in self.configs:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Running {config_name} on {benchmark}")
            logger.info(f"{'=' * 60}")

            self.run_evaluation(
                benchmark=benchmark,
                config_name=config_name,
                max_samples=max_samples,
                data_root=data_root,
            )

        return self.tracker.compute_aggregates()

    def run_full_phase1(
        self,
        max_samples: Optional[int] = None,
        data_root: str = "data",
    ) -> dict:
        """
        Run complete Phase 1 evaluation.

        Evaluates all model configs on both GlitchBench and PhysGame.

        Args:
            max_samples: Maximum samples per benchmark per config
            data_root: Root directory for benchmark data

        Returns:
            Complete evaluation results
        """
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1 EVALUATION: GlitchBench + PhysGame")
        logger.info("=" * 80)

        for benchmark in ["glitchbench", "physgame"]:
            self.run_all_configs(
                benchmark=benchmark,
                max_samples=max_samples,
                data_root=data_root,
            )

        # Print and save results
        self.tracker.print_summary()
        self.tracker.print_comparison_table()
        results_dir = self.tracker.save_results()

        logger.info(f"\nResults saved to: {results_dir}")
        return self.tracker.compute_aggregates()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 1 Evaluation: GlitchBench + PhysGame",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run GlitchBench with baseline
    python run_phase1.py --benchmark glitchbench --config baseline_plain

    # Run all configs on PhysGame
    python run_phase1.py --benchmark physgame --all-configs

    # Run full Phase 1 with sample limit
    python run_phase1.py --full --max-samples 100

    # Custom cache and results directories
    python run_phase1.py --full --cache-dir /data/cache --results-dir /data/results
        """,
    )

    # Benchmark selection
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["glitchbench", "physgame"],
        help="Benchmark to evaluate",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run complete Phase 1 (both benchmarks, all configs)",
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

    # Sample limits
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate per benchmark/config",
    )

    # Directories
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root directory for benchmark data (default: data)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache/perception",
        help="Directory for perception cache (default: data/cache/perception)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/phase1",
        help="Directory for evaluation results (default: results/phase1)",
    )

    # Experiment
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name for organizing results",
    )

    args = parser.parse_args()

    # Validate args
    if not args.full and not args.benchmark:
        parser.error("Either --benchmark or --full is required")

    if args.benchmark and not args.config and not args.all_configs:
        parser.error("Either --config or --all-configs is required with --benchmark")

    # Create evaluator
    evaluator = Phase1Evaluator(
        cache_dir=args.cache_dir,
        results_dir=args.results_dir,
        experiment_name=args.experiment,
    )

    # Run evaluation
    if args.full:
        evaluator.run_full_phase1(
            max_samples=args.max_samples,
            data_root=args.data_root,
        )
    elif args.all_configs:
        evaluator.run_all_configs(
            benchmark=args.benchmark,
            max_samples=args.max_samples,
            data_root=args.data_root,
        )
    else:
        evaluator.run_evaluation(
            benchmark=args.benchmark,
            config_name=args.config,
            max_samples=args.max_samples,
            data_root=args.data_root,
        )


if __name__ == "__main__":
    main()
