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

from benchmarks.loaders import GlitchBenchLoader, PhysGameLoader, BenchmarkConfig
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
from benchmarks.model_inference import FullPipelineRunner, get_full_pipeline_runner

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
        projector_weights: str = "outputs/projector_weights.pt",
        lora_path: str = "outputs/lora_adapter",
        device: str = "cuda",
        preset: str = "standard",
    ):
        """
        Initialize Phase 1 evaluator.

        Args:
            cache_dir: Directory for perception cache
            results_dir: Directory for evaluation results
            experiment_name: Optional experiment name
            projector_weights: Path to projector weights
            lora_path: Path to LoRA adapter
            device: Device to run on
            preset: Configuration preset ("light", "standard", "full")
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

        logger.info("Phase 1 Evaluator initialized (Full Pipeline Mode)")

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
        Evaluate a single sample with full perception pipeline.

        Args:
            sample: Benchmark sample
            config: Model configuration
            cached_features: (Not used - full pipeline handles internally)

        Returns:
            SampleMetrics with results
        """
        retrieval_time = 0.0

        # Get question and answer format
        question = sample.question or sample.get_formatted_prompt()
        answer_format = sample.answer_format.value if sample.answer_format else "unknown"

        # Create sample metrics
        metrics = self.tracker.start_sample(
            sample_id=sample.sample_id,
            benchmark=sample.benchmark_name,
            task_type=sample.task_type.value if sample.task_type else "unknown",
            model_config=config.name,
            video_duration_sec=sample.video_duration_sec or 0.0,
            question=question,
            answer_format=answer_format,
        )

        # Run FULL PIPELINE inference (SAM3, SigLIP, VideoMAE, OCR, Whisper, Timeline, KB, LLM)
        predicted, perception_time, generation_time = self._run_inference(sample, config, cached_features)
        ground_truth = sample.ground_truth

        # Evaluate correctness
        correct = self._check_correctness(predicted, ground_truth, sample)
        
        # Log prediction details
        pred_preview = predicted[:150].replace('\n', ' ') if predicted else ''
        logger.info(f"[{sample.sample_id}] GT='{ground_truth}' | Pred='{pred_preview}...' | Correct={correct}")

        # Compute metrics
        num_frames = config.frame_sampling.max_frames
        input_tokens = 1024  # Estimated
        output_tokens = len(predicted.split()) * 2 if predicted else 0

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
    ) -> tuple[str, float]:
        """
        Run model inference on a sample using the FULL perception pipeline.

        This runs the complete pipeline:
        - Frame extraction, SAM3, SigLIP, VideoMAE, OCR, Whisper
        - Timeline building, KB construction
        - LLM reasoning with projectors + LoRA

        Args:
            sample: Benchmark sample
            config: Model configuration
            cached_features: (Not used - full pipeline handles caching internally)

        Returns:
            Tuple of (predicted_answer, perception_time_sec, inference_time_sec)
        """
        # Run full pipeline inference (same as realtime_inference.py)
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

        # Debug logging
        logger.debug(f"Checking: pred='{pred_norm[:100]}...' vs gt='{gt_norm}'")

        # Direct match
        if pred_norm == gt_norm:
            return True

        # Binary answer normalization - check for keywords in response
        yes_keywords = {"yes", "glitch", "bug", "error", "anomaly", "issue", "problem"}
        no_keywords = {"no", "normal", "correct", "fine", "nothing wrong"}

        # Check if pred contains yes/no keywords
        pred_has_yes = any(kw in pred_norm for kw in yes_keywords)
        pred_has_no = any(kw in pred_norm for kw in no_keywords) and "not" not in pred_norm[:50]
        
        # Check ground truth
        gt_is_yes = gt_norm in {"yes", "true", "1", "glitch", "a"}
        gt_is_no = gt_norm in {"no", "false", "0", "normal", "b"}

        # Match based on sentiment
        if pred_has_yes and not pred_has_no and gt_is_yes:
            return True
        if pred_has_no and not pred_has_yes and gt_is_no:
            return True
        
        # For MCQ, check if letter matches
        if sample.options and len(pred_norm) >= 1:
            pred_letter = pred_norm[0].upper()
            if pred_letter in "ABCD":
                gt_letter = gt_norm[0].upper() if gt_norm else ""
                if pred_letter == gt_letter:
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
            loader = GlitchBenchLoader(BenchmarkConfig(data_dir=data_root))
        elif benchmark == "physgame":
            loader = PhysGameLoader(BenchmarkConfig(data_dir=data_root))
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

    # Model weights
    parser.add_argument(
        "--projector-weights",
        type=str,
        default="outputs/projector_weights.pt",
        help="Path to projector weights (default: outputs/projector_weights.pt)",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="outputs/lora_adapter",
        help="Path to LoRA adapter (default: outputs/lora_adapter)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)",
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
        help="Configuration preset for perception pipeline (default: standard)",
    )

    args = parser.parse_args()

    # Validate args
    if not args.full and not args.benchmark:
        parser.error("Either --benchmark or --full is required")

    if args.benchmark and not args.config and not args.all_configs:
        parser.error("Either --config or --all-configs is required with --benchmark")

    # Create evaluator with full pipeline
    evaluator = Phase1Evaluator(
        cache_dir=args.cache_dir,
        results_dir=args.results_dir,
        experiment_name=args.experiment,
        projector_weights=args.projector_weights,
        lora_path=args.lora_path,
        device=args.device,
        preset=args.preset,
    )

    # Run evaluation
    results = None
    if args.full:
        results = evaluator.run_full_phase1(
            max_samples=args.max_samples,
            data_root=args.data_root,
        )
    elif args.all_configs:
        results = evaluator.run_all_configs(
            benchmark=args.benchmark,
            max_samples=args.max_samples,
            data_root=args.data_root,
        )
    else:
        results = evaluator.run_evaluation(
            benchmark=args.benchmark,
            config_name=args.config,
            max_samples=args.max_samples,
            data_root=args.data_root,
        )
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    if results:
        # results is {model_config: {benchmark: AggregateMetrics}}
        for model_config, benchmarks in results.items():
            for benchmark, agg in benchmarks.items():
                print(f"\n  Model: {model_config}")
                print(f"  Benchmark: {benchmark}")
                print(f"  Total samples: {agg.num_samples}")
                print(f"  Correct: {agg.correct_count}")
                print(f"  Accuracy: {agg.accuracy:.1%}")
                print(f"  Avg inference time: {agg.avg_total_time:.2f}s per sample")
                
                # Task breakdown if available
                if agg.task_accuracy:
                    print("\n  By task type:")
                    for task, stats in agg.task_accuracy.items():
                        task_acc = stats.get("accuracy", 0.0)
                        task_n = stats.get("total", 0)
                        print(f"    {task}: {task_acc:.1%} ({task_n} samples)")
    else:
        print("  No results available")
    
    print("=" * 60)
    print(f"Results saved to: {args.results_dir}")
    
    # Also save results to file
    evaluator.tracker.save_results()


if __name__ == "__main__":
    main()
