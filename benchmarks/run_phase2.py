#!/usr/bin/env python3
"""
Phase 2 Evaluation: VideoGameQA-Bench Subset

Mid-cost game QA evaluation phase:
- VideoGameQA-Bench: Comprehensive gaming video understanding benchmark
- Tasks: visual_unit_test, needle_haystack, glitch_detection, bug_report

This phase tests:
- Game-specific visual understanding
- Temporal reasoning over gameplay
- Game knowledge retrieval
- Multi-step reasoning

Usage:
    python benchmarks/run_phase2.py --help
    python benchmarks/run_phase2.py --task visual_unit_test --config gvp_light
    python benchmarks/run_phase2.py --all-tasks --max-samples 50
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

from benchmarks.loaders import VideoGameQALoader
from benchmarks.loaders.base import BenchmarkLoader, BenchmarkSample, TaskType
from benchmarks.loaders.videogameqa import TASK_MAPPING
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

# Available VideoGameQA-Bench tasks
VIDEOGAMEQA_TASKS = list(TASK_MAPPING.keys())


class Phase2Evaluator:
    """
    Phase 2 evaluation runner for VideoGameQA-Bench.

    Handles:
    - Task-specific evaluation subsets
    - Two-stage evaluation (cache perception, then run QA)
    - Multiple model configuration comparison
    - Comprehensive metrics tracking
    """

    def __init__(
        self,
        cache_dir: str = "data/cache/perception",
        results_dir: str = "results/phase2",
        experiment_name: Optional[str] = None,
        projector_weights: str = "outputs/projector_weights.pt",
        lora_path: str = "outputs/lora_adapter",
        device: str = "cuda",
        preset: str = "standard",
    ):
        """
        Initialize Phase 2 evaluator.

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

        # Available tasks from VideoGameQA-Bench
        self.available_tasks = VIDEOGAMEQA_TASKS

        logger.info("Phase 2 Evaluator initialized (Full Pipeline Mode)")
        logger.info(f"  Available tasks: {self.available_tasks}")

    def run_perception_caching(
        self,
        samples: list[BenchmarkSample],
    ) -> int:
        """
        Stage A: Run perception pipeline and cache results.

        Args:
            samples: List of benchmark samples

        Returns:
            Number of videos cached
        """
        cached_count = 0
        skipped_count = 0

        # Get unique video paths
        video_paths = set()
        for sample in samples:
            video_path = sample.video_path
            if video_path:
                video_paths.add(video_path)

        logger.info(f"Stage A: Caching perception for {len(video_paths)} unique videos")

        for i, video_path in enumerate(video_paths):
            # Check if already cached
            if self.cache.has_cache(video_path):
                skipped_count += 1
                continue

            # Run perception (placeholder)
            features = self._run_perception_pipeline(video_path)

            if features:
                self.cache.save(video_path, features)
                cached_count += 1

            if (i + 1) % 20 == 0:
                logger.info(
                    f"  Progress: {i + 1}/{len(video_paths)} "
                    f"(cached: {cached_count}, skipped: {skipped_count})"
                )

        logger.info(f"Stage A complete: cached {cached_count}, skipped {skipped_count}")
        return cached_count

    def _run_perception_pipeline(self, video_path: str) -> Optional[CachedFeatures]:
        """
        Run perception pipeline for a video.

        Placeholder implementation - actual would call full perception stack.

        Args:
            video_path: Path to video file

        Returns:
            CachedFeatures or None on error
        """
        features = CachedFeatures(
            video_hash="",
            video_path=video_path,
            video_duration_sec=60.0,  # Placeholder
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
        retrieval_time = 0.0

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

        # Run FULL PIPELINE inference
        predicted, perception_time, generation_time = self._run_inference(sample, config, cached_features)
        ground_truth = sample.ground_truth

        # Evaluate correctness
        correct = self._check_correctness(predicted, ground_truth, sample)

        # Compute metrics
        num_frames = config.frame_sampling.max_frames
        input_tokens = 2048  # Larger context for VideoGameQA
        output_tokens = len(predicted.split()) * 2

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
    ) -> tuple[str, float, float]:
        """
        Run FULL PIPELINE inference on a sample.

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
        """
        Check if prediction is correct.

        Handles different answer formats:
        - Binary (yes/no)
        - MCQ (A/B/C/D)
        - Free text (exact match for now)

        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            sample: Original sample

        Returns:
            True if correct
        """
        pred_norm = predicted.lower().strip()
        gt_norm = str(ground_truth).lower().strip()

        # Direct match
        if pred_norm == gt_norm:
            return True

        # MCQ handling - extract letter
        if len(pred_norm) > 0 and pred_norm[0] in "abcd":
            pred_letter = pred_norm[0]
            if len(gt_norm) > 0 and gt_norm[0] in "abcd":
                return pred_letter == gt_norm[0]
            if gt_norm in ["a", "b", "c", "d"]:
                return pred_letter == gt_norm

        # Binary handling
        yes_variants = {"yes", "true", "1", "correct"}
        no_variants = {"no", "false", "0", "incorrect"}

        if pred_norm in yes_variants and gt_norm in yes_variants:
            return True
        if pred_norm in no_variants and gt_norm in no_variants:
            return True

        return False

    def run_evaluation(
        self,
        task: Optional[str],
        config_name: str,
        max_samples: Optional[int] = None,
        data_root: str = "data",
    ) -> dict:
        """
        Run evaluation for a specific task and config.

        Args:
            task: Task name or None for all tasks
            config_name: Model config name
            max_samples: Maximum samples to evaluate
            data_root: Root directory for benchmark data

        Returns:
            Evaluation results dict
        """
        # Get loader
        loader = VideoGameQALoader(data_root=data_root, tasks=[task] if task else None)

        # Get config
        if config_name not in self.configs:
            raise ValueError(f"Unknown config: {config_name}")
        config = self.configs[config_name]()

        # Load samples
        samples = list(loader)
        if max_samples:
            samples = samples[:max_samples]

        task_str = task or "all_tasks"
        logger.info(f"Evaluating VideoGameQA-Bench ({task_str}) with {config_name}")
        logger.info(f"  Samples: {len(samples)}")

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

            if (i + 1) % 25 == 0:
                logger.info(f"  Progress: {i + 1}/{len(samples)}")

        return self.tracker.compute_aggregates()

    def run_all_tasks(
        self,
        config_name: str,
        max_samples_per_task: Optional[int] = None,
        data_root: str = "data",
    ) -> dict:
        """
        Run evaluation for all tasks with a specific config.

        Args:
            config_name: Model config name
            max_samples_per_task: Max samples per task
            data_root: Root directory for benchmark data

        Returns:
            Combined evaluation results
        """
        for task in self.available_tasks:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Task: {task}")
            logger.info(f"{'=' * 60}")

            # Create task-specific loader
            loader = VideoGameQALoader(data_root=data_root, tasks=[task])
            samples = list(loader)

            if max_samples_per_task:
                samples = samples[:max_samples_per_task]

            # Get config
            config = self.configs[config_name]()

            # Run perception caching for this task's samples
            self.run_perception_caching(samples)

            # Evaluate
            for sample in samples:
                cached_features = None
                if sample.video_path:
                    cached_features = self.cache.load(sample.video_path)
                self.evaluate_sample(sample, config, cached_features)

        return self.tracker.compute_aggregates()

    def run_all_configs(
        self,
        task: Optional[str] = None,
        max_samples: Optional[int] = None,
        data_root: str = "data",
    ) -> dict:
        """
        Run evaluation for all configs on a task.

        Args:
            task: Specific task or None for all
            max_samples: Max samples per config
            data_root: Root directory for benchmark data

        Returns:
            Combined evaluation results
        """
        for config_name in self.configs:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Config: {config_name}")
            logger.info(f"{'=' * 60}")

            self.run_evaluation(
                task=task,
                config_name=config_name,
                max_samples=max_samples,
                data_root=data_root,
            )

        return self.tracker.compute_aggregates()

    def run_full_phase2(
        self,
        max_samples_per_task: Optional[int] = None,
        data_root: str = "data",
    ) -> dict:
        """
        Run complete Phase 2 evaluation.

        Evaluates all model configs on all VideoGameQA-Bench tasks.

        Args:
            max_samples_per_task: Maximum samples per task per config
            data_root: Root directory for benchmark data

        Returns:
            Complete evaluation results
        """
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2 EVALUATION: VideoGameQA-Bench")
        logger.info("=" * 80)

        for config_name in self.configs:
            logger.info(f"\n\n{'#' * 70}")
            logger.info(f"# MODEL: {config_name}")
            logger.info(f"{'#' * 70}")

            self.run_all_tasks(
                config_name=config_name,
                max_samples_per_task=max_samples_per_task,
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
        description="Phase 2 Evaluation: VideoGameQA-Bench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available tasks: {", ".join(VIDEOGAMEQA_TASKS.keys())}

Examples:
    # Run specific task with specific config
    python run_phase2.py --task visual_unit_test --config gvp_light

    # Run all tasks with one config
    python run_phase2.py --all-tasks --config gvp_full

    # Run one task with all configs
    python run_phase2.py --task needle_haystack --all-configs

    # Run complete Phase 2
    python run_phase2.py --full --max-samples-per-task 50
        """,
    )

    # Task selection
    parser.add_argument(
        "--task",
        type=str,
        choices=list(VIDEOGAMEQA_TASKS.keys()),
        help="Specific task to evaluate",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Run all available tasks",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run complete Phase 2 (all tasks, all configs)",
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
        help="Maximum samples to evaluate (for single task/config)",
    )
    parser.add_argument(
        "--max-samples-per-task",
        type=int,
        default=None,
        help="Maximum samples per task (for --all-tasks or --full)",
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
        default="results/phase2",
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
    if not args.full and not args.task and not args.all_tasks:
        parser.error("Either --task, --all-tasks, or --full is required")

    if (args.task or args.all_tasks) and not args.config and not args.all_configs and not args.full:
        parser.error("Either --config or --all-configs is required")

    # Create evaluator with full pipeline
    evaluator = Phase2Evaluator(
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
        evaluator.run_full_phase2(
            max_samples_per_task=args.max_samples_per_task,
            data_root=args.data_root,
        )
    elif args.all_configs:
        evaluator.run_all_configs(
            task=args.task if args.task else None,
            max_samples=args.max_samples,
            data_root=args.data_root,
        )
    elif args.all_tasks:
        evaluator.run_all_tasks(
            config_name=args.config,
            max_samples_per_task=args.max_samples_per_task or args.max_samples,
            data_root=args.data_root,
        )
    else:
        evaluator.run_evaluation(
            task=args.task,
            config_name=args.config,
            max_samples=args.max_samples,
            data_root=args.data_root,
        )


if __name__ == "__main__":
    main()
