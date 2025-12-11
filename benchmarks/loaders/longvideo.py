"""
Long Video Benchmark Loaders.

Provides loaders for long-video understanding benchmarks:
- LongVideoBench: Web videos with subtitles, up to ~1 hour
- MLVU: Multi-task long video understanding (3-120 minutes)
- LVBench: Extreme long video benchmark

These are used for Phase 3 stress testing to verify that the
timeline/KB approach scales to generic long-video tasks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from .base import (
    AnswerFormat,
    BenchmarkConfig,
    BenchmarkInfo,
    BenchmarkLoader,
    BenchmarkSample,
    TaskType,
)

logger = logging.getLogger(__name__)


class LongVideoBenchLoader(BenchmarkLoader):
    """
    Loader for LongVideoBench dataset.

    LongVideoBench contains 3,763 web videos with subtitles (up to ~1 hour)
    and 6,678 human-annotated MCQ questions across 17 categories.

    Paper: https://arxiv.org/abs/2407.15754
    """

    @property
    def name(self) -> str:
        return "LongVideoBench"

    @property
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="LongVideoBench",
            version="1.0",
            description="Long video understanding benchmark (up to 1 hour)",
            total_samples=6678,
            tasks=[
                "temporal_reasoning",
                "summarization",
                "event_localization",
                "counting",
                "causality",
            ],
            categories=[
                "film", "tv_series", "documentary", "sports",
                "news", "vlogs", "gaming", "education",
            ],
            avg_video_duration_sec=1800.0,  # ~30 minutes average
            citation="LongVideoBench: A Benchmark for Long-form Video Understanding",
            url="https://longvideobench.github.io/",
        )

    def _load_annotations(self) -> list[dict]:
        """Load LongVideoBench annotations."""
        annotations = []
        data_dir = Path(self.config.data_dir)

        # Try standard annotation files
        possible_files = [
            "annotations.json",
            "longvideobench.json",
            "test.json",
            "val.json",
        ]

        annotation_file = None
        if self.config.annotation_file:
            annotation_file = data_dir / self.config.annotation_file
        else:
            for fname in possible_files:
                candidate = data_dir / fname
                if candidate.exists():
                    annotation_file = candidate
                    break

        if annotation_file and annotation_file.exists():
            with open(annotation_file, "r") as f:
                data = json.load(f)

            if isinstance(data, list):
                annotations = data
            elif isinstance(data, dict):
                if "samples" in data:
                    annotations = data["samples"]
                elif "questions" in data:
                    annotations = data["questions"]
                else:
                    # May be keyed by video ID
                    for vid_id, vid_data in data.items():
                        if isinstance(vid_data, list):
                            for q in vid_data:
                                q["video_id"] = vid_id
                                annotations.append(q)
                        elif isinstance(vid_data, dict):
                            vid_data["video_id"] = vid_id
                            annotations.append(vid_data)

        logger.info(f"Loaded {len(annotations)} annotations from LongVideoBench")
        return annotations

    def _convert_sample(self, raw: dict, idx: int) -> BenchmarkSample:
        """Convert LongVideoBench annotation to BenchmarkSample."""
        sample_id = raw.get("id", raw.get("question_id", f"lvb_{idx:05d}"))

        # Video path
        video_path = raw.get("video_path", raw.get("video"))
        video_id = raw.get("video_id")
        if not video_path and video_id:
            video_path = f"videos/{video_id}.mp4"

        if video_path:
            full_path = Path(self.config.data_dir) / video_path
            video_path = str(full_path) if full_path.exists() else video_path

        # Question and options
        question = raw.get("question", raw.get("query", ""))
        options = raw.get("options", raw.get("choices", []))
        answer = raw.get("answer", raw.get("correct_answer", ""))

        # Task type inference
        task_str = raw.get("task", raw.get("type", ""))
        if "temporal" in task_str.lower() or "when" in question.lower():
            task_type = TaskType.TEMPORAL_REASONING
        elif "count" in task_str.lower() or "how many" in question.lower():
            task_type = TaskType.VISUAL_QA
        else:
            task_type = TaskType.VISUAL_QA

        return BenchmarkSample(
            sample_id=sample_id,
            benchmark_name=self.name,
            video_path=video_path,
            video_url=raw.get("video_url", raw.get("url")),
            question=question,
            options=options,
            ground_truth=str(answer),
            task_type=task_type,
            answer_format=AnswerFormat.MCQ if options else AnswerFormat.FREE_TEXT,
            category=raw.get("category", raw.get("domain", "general")),
            video_duration_sec=raw.get("duration", raw.get("video_duration")),
            start_time_sec=raw.get("start_time"),
            end_time_sec=raw.get("end_time"),
            metadata={
                "video_id": video_id,
                "has_subtitles": raw.get("has_subtitles", False),
                "language": raw.get("language", "en"),
            },
        )


class MLVULoader(BenchmarkLoader):
    """
    Loader for MLVU (Multi-task Long Video Understanding) dataset.

    MLVU contains 1,730 videos with 3,102 QA pairs, ranging from
    3 minutes to 2 hours. It includes 9 different tasks covering
    various aspects of long video understanding.

    Paper: https://arxiv.org/abs/2406.04264
    """

    @property
    def name(self) -> str:
        return "MLVU"

    @property
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="MLVU",
            version="1.0",
            description="Multi-task Long Video Understanding (3-120 minutes)",
            total_samples=3102,
            tasks=[
                "topic_reasoning",
                "anomaly_recognition",
                "video_summarization",
                "needle_qa",
                "ego_reasoning",
                "plot_qa",
                "action_order",
                "action_count",
                "subPlot_qa",
            ],
            categories=["movies", "surveillance", "egocentric", "games", "vlogs"],
            avg_video_duration_sec=2400.0,  # ~40 minutes average
            citation="MLVU: A Comprehensive Benchmark for Multi-task Long Video Understanding",
            url="https://github.com/JUNJIE99/MLVU",
        )

    def _load_annotations(self) -> list[dict]:
        """Load MLVU annotations."""
        annotations = []
        data_dir = Path(self.config.data_dir)

        # MLVU may have task-specific annotation files
        possible_files = [
            "annotations.json",
            "mlvu.json",
            "test.json",
            "val.json",
        ]

        # Also check for task-specific files
        task_files = list(data_dir.glob("*_test.json")) + list(data_dir.glob("*_val.json"))

        annotation_file = None
        if self.config.annotation_file:
            annotation_file = data_dir / self.config.annotation_file
        else:
            for fname in possible_files:
                candidate = data_dir / fname
                if candidate.exists():
                    annotation_file = candidate
                    break

        # Load from main annotation file
        if annotation_file and annotation_file.exists():
            with open(annotation_file, "r") as f:
                data = json.load(f)

            if isinstance(data, list):
                annotations = data
            elif isinstance(data, dict):
                if "samples" in data:
                    annotations = data["samples"]
                else:
                    for task_name, task_samples in data.items():
                        if isinstance(task_samples, list):
                            for sample in task_samples:
                                sample["task"] = task_name
                                annotations.append(sample)

        # Load from task-specific files
        if not annotations and task_files:
            for task_file in task_files:
                task_name = task_file.stem.replace("_test", "").replace("_val", "")
                try:
                    with open(task_file, "r") as f:
                        task_data = json.load(f)

                    task_samples = task_data if isinstance(task_data, list) else task_data.get("samples", [])
                    for sample in task_samples:
                        sample["task"] = task_name
                        annotations.append(sample)
                except Exception as e:
                    logger.warning(f"Failed to load {task_file}: {e}")

        logger.info(f"Loaded {len(annotations)} annotations from MLVU")
        return annotations

    def _convert_sample(self, raw: dict, idx: int) -> BenchmarkSample:
        """Convert MLVU annotation to BenchmarkSample."""
        sample_id = raw.get("id", raw.get("question_id", f"mlvu_{idx:05d}"))

        # Video path
        video_path = raw.get("video_path", raw.get("video"))
        if video_path:
            full_path = Path(self.config.data_dir) / video_path
            video_path = str(full_path) if full_path.exists() else video_path

        # Question and options
        question = raw.get("question", raw.get("query", ""))
        options = raw.get("options", raw.get("choices", []))
        answer = raw.get("answer", raw.get("correct_answer", ""))

        # Task type mapping
        task_str = raw.get("task", "").lower()
        task_mapping = {
            "needle": TaskType.NEEDLE_IN_HAYSTACK,
            "anomaly": TaskType.GLITCH_DETECTION,
            "action_order": TaskType.TEMPORAL_REASONING,
            "action_count": TaskType.VISUAL_QA,
            "plot": TaskType.TEMPORAL_REASONING,
            "topic": TaskType.VISUAL_QA,
            "ego": TaskType.VISUAL_QA,
        }

        task_type = TaskType.VISUAL_QA
        for key, ttype in task_mapping.items():
            if key in task_str:
                task_type = ttype
                break

        return BenchmarkSample(
            sample_id=sample_id,
            benchmark_name=self.name,
            video_path=video_path,
            question=question,
            options=options,
            ground_truth=str(answer),
            task_type=task_type,
            answer_format=AnswerFormat.MCQ if options else AnswerFormat.FREE_TEXT,
            category=raw.get("category", raw.get("genre", task_str)),
            subcategory=task_str,
            video_duration_sec=raw.get("duration", raw.get("video_duration")),
            start_time_sec=raw.get("start_time"),
            end_time_sec=raw.get("end_time"),
            metadata={
                "task": task_str,
                "video_genre": raw.get("genre"),
            },
        )

    def get_task_samples(self, task: str) -> list[BenchmarkSample]:
        """Get samples for a specific MLVU task."""
        if not self._loaded:
            self.load()

        task_lower = task.lower()
        return [
            s for s in self._samples
            if task_lower in s.metadata.get("task", "").lower()
            or task_lower in s.subcategory.lower()
        ]


def create_longvideobench_loader(
    data_dir: str,
    max_samples: Optional[int] = None,
    max_duration_sec: Optional[float] = None,
    categories: Optional[list[str]] = None,
) -> LongVideoBenchLoader:
    """
    Factory function to create a LongVideoBench loader.

    Args:
        data_dir: Path to LongVideoBench data directory
        max_samples: Maximum samples to load
        max_duration_sec: Filter videos by maximum duration
        categories: Filter by video categories

    Returns:
        Configured LongVideoBenchLoader
    """
    config = BenchmarkConfig(
        data_dir=data_dir,
        max_samples=max_samples,
        max_duration_sec=max_duration_sec,
        categories=categories,
    )
    return LongVideoBenchLoader(config)


def create_mlvu_loader(
    data_dir: str,
    max_samples: Optional[int] = None,
    tasks: Optional[list[str]] = None,
    max_duration_sec: Optional[float] = None,
) -> MLVULoader:
    """
    Factory function to create an MLVU loader.

    Args:
        data_dir: Path to MLVU data directory
        max_samples: Maximum samples to load
        tasks: Filter by MLVU tasks (needle_qa, action_order, etc.)
        max_duration_sec: Filter videos by maximum duration

    Returns:
        Configured MLVULoader
    """
    config = BenchmarkConfig(
        data_dir=data_dir,
        max_samples=max_samples,
        tasks=tasks,
        max_duration_sec=max_duration_sec,
    )
    return MLVULoader(config)
