"""
VideoGameQA-Bench Data Loader.

VideoGameQA-Bench (NeurIPS 2025) is a benchmark designed to test VLMs on
real game QA workflows including visual unit tests, visual regression,
needle-in-a-haystack, glitch detection, and bug report generation.

Website: https://asgaardlab.github.io/videogameqa-bench/
Paper: (NeurIPS 2025 Datasets Track)

Tasks:
1. Visual Unit Test - Verify specific game states
2. Visual Regression - Detect visual differences between versions
3. Video Needle-in-Haystack - Find specific events in long videos
4. Video Glitch Detection - Identify glitches in gameplay
5. Bug Report Generation - Generate structured bug reports

Data format (expected):
    videogameqa/
    ├── annotations/
    │   ├── visual_unit_test.json
    │   ├── visual_regression.json
    │   ├── needle_haystack.json
    │   ├── glitch_detection.json
    │   └── bug_report.json
    └── videos/
        └── ...
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


# VideoGameQA-Bench task mapping
TASK_MAPPING = {
    "visual_unit_test": TaskType.VISUAL_QA,
    "visual_regression": TaskType.VISUAL_QA,
    "needle_haystack": TaskType.NEEDLE_IN_HAYSTACK,
    "needle_in_haystack": TaskType.NEEDLE_IN_HAYSTACK,
    "video_needle": TaskType.NEEDLE_IN_HAYSTACK,
    "glitch_detection": TaskType.GLITCH_DETECTION,
    "video_glitch": TaskType.GLITCH_DETECTION,
    "bug_report": TaskType.BUG_REPORT,
    "bug_generation": TaskType.BUG_REPORT,
    "temporal_qa": TaskType.TEMPORAL_REASONING,
    "action_qa": TaskType.ACTION_RECOGNITION,
}


class VideoGameQALoader(BenchmarkLoader):
    """
    Loader for VideoGameQA-Bench dataset.

    Handles loading multiple task types from the benchmark and
    converting them to the standardized BenchmarkSample format.
    """

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self._task_files: dict[str, Path] = {}

    @property
    def name(self) -> str:
        return "VideoGameQA-Bench"

    @property
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="VideoGameQA-Bench",
            version="1.0",
            description="Game QA benchmark with 9 tasks and ~4786 QA pairs (NeurIPS 2025)",
            total_samples=4786,
            tasks=[
                "visual_unit_test",
                "visual_regression",
                "needle_haystack",
                "glitch_detection",
                "bug_report",
            ],
            categories=["gameplay", "ui", "physics", "rendering", "audio"],
            avg_video_duration_sec=60.0,
            citation="VideoGameQA-Bench: Evaluating Vision-Language Models on Game QA (NeurIPS 2025)",
            url="https://asgaardlab.github.io/videogameqa-bench/",
        )

    def _discover_task_files(self) -> dict[str, Path]:
        """Discover annotation files for each task."""
        task_files = {}
        data_dir = Path(self.config.data_dir)

        # Check annotations subdirectory
        anno_dir = data_dir / "annotations"
        if not anno_dir.exists():
            anno_dir = data_dir

        # Look for task-specific files
        for json_file in anno_dir.glob("*.json"):
            fname = json_file.stem.lower()
            for task_key in TASK_MAPPING.keys():
                if task_key in fname:
                    task_files[task_key] = json_file
                    break

        # Also check for combined annotation file
        combined_files = ["all.json", "annotations.json", "videogameqa.json"]
        for fname in combined_files:
            candidate = anno_dir / fname
            if candidate.exists():
                task_files["combined"] = candidate
                break

        return task_files

    def _load_annotations(self) -> list[dict]:
        """Load annotations from VideoGameQA-Bench format."""
        annotations = []
        data_dir = Path(self.config.data_dir)

        self._task_files = self._discover_task_files()

        if not self._task_files:
            logger.warning(f"No annotation files found in {data_dir}")
            return []

        # Check for combined file first
        if "combined" in self._task_files:
            with open(self._task_files["combined"], "r") as f:
                data = json.load(f)

            if isinstance(data, list):
                annotations = data
            elif isinstance(data, dict):
                # May be organized by task
                for task_key, task_data in data.items():
                    if isinstance(task_data, list):
                        for item in task_data:
                            item["task"] = task_key
                            annotations.append(item)

            logger.info(f"Loaded {len(annotations)} from combined file")
            return annotations

        # Load from separate task files
        for task_key, file_path in self._task_files.items():
            if task_key == "combined":
                continue

            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                task_annotations = data if isinstance(data, list) else data.get("samples", [])
                for item in task_annotations:
                    item["task"] = task_key
                    annotations.append(item)

                logger.info(f"Loaded {len(task_annotations)} from {task_key}")
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        # Filter by configured tasks
        if self.config.tasks:
            task_set = set(self.config.tasks)
            annotations = [a for a in annotations if a.get("task") in task_set]

        return annotations

    def _convert_sample(self, raw: dict, idx: int) -> BenchmarkSample:
        """Convert VideoGameQA-Bench annotation to BenchmarkSample."""
        sample_id = raw.get("id", raw.get("sample_id", f"vgqa_{idx:05d}"))

        # Resolve video path
        video_path = raw.get("video_path", raw.get("video"))
        if video_path:
            full_path = Path(self.config.data_dir) / video_path
            video_path = str(full_path) if full_path.exists() else video_path

        # Handle image-only samples
        image_path = raw.get("image_path", raw.get("image"))
        if image_path:
            full_path = Path(self.config.data_dir) / image_path
            image_path = str(full_path) if full_path.exists() else image_path

        # Get question and answer
        question = raw.get("question", raw.get("query", ""))
        options = raw.get("options", raw.get("choices", []))
        answer = raw.get("answer", raw.get("ground_truth", ""))

        # Map task to TaskType
        task_str = raw.get("task", "visual_qa")
        task_type = TASK_MAPPING.get(task_str, TaskType.VISUAL_QA)

        # Determine answer format
        if task_type == TaskType.BUG_REPORT:
            answer_format = AnswerFormat.FREE_TEXT
        elif options:
            answer_format = AnswerFormat.MCQ
        elif answer.lower() in ("yes", "no", "true", "false"):
            answer_format = AnswerFormat.BINARY
        else:
            answer_format = AnswerFormat.FREE_TEXT

        # Extract temporal info for needle-in-haystack
        start_time = raw.get("start_time", raw.get("target_start"))
        end_time = raw.get("end_time", raw.get("target_end"))

        return BenchmarkSample(
            sample_id=sample_id,
            benchmark_name=self.name,
            video_path=video_path,
            image_path=image_path,
            question=question,
            options=options,
            ground_truth=str(answer),
            task_type=task_type,
            answer_format=answer_format,
            category=raw.get("category", task_str),
            subcategory=raw.get("subcategory", ""),
            game_name=raw.get("game", raw.get("game_name")),
            description=raw.get("description"),
            video_duration_sec=raw.get("duration", raw.get("video_duration")),
            start_time_sec=start_time,
            end_time_sec=end_time,
            metadata={
                "task": task_str,
                "difficulty": raw.get("difficulty"),
                "expected_output": raw.get("expected_output"),
            },
        )

    def get_task_samples(self, task: str) -> list[BenchmarkSample]:
        """Get samples for a specific task."""
        if not self._loaded:
            self.load()

        task_lower = task.lower()
        return [
            s for s in self._samples
            if s.metadata.get("task", "").lower() == task_lower
            or s.category.lower() == task_lower
        ]

    def get_needle_haystack_samples(self) -> list[BenchmarkSample]:
        """Get needle-in-haystack samples (for long-range retrieval testing)."""
        return self.get_by_task(TaskType.NEEDLE_IN_HAYSTACK)

    def get_glitch_samples(self) -> list[BenchmarkSample]:
        """Get glitch detection samples."""
        return self.get_by_task(TaskType.GLITCH_DETECTION)

    def get_bug_report_samples(self) -> list[BenchmarkSample]:
        """Get bug report generation samples."""
        return self.get_by_task(TaskType.BUG_REPORT)


def create_videogameqa_loader(
    data_dir: str,
    max_samples: Optional[int] = None,
    tasks: Optional[list[str]] = None,
) -> VideoGameQALoader:
    """
    Factory function to create a VideoGameQA-Bench loader.

    Args:
        data_dir: Path to VideoGameQA-Bench data directory
        max_samples: Maximum samples to load
        tasks: Filter by task types (needle_haystack, glitch_detection, bug_report, etc.)

    Returns:
        Configured VideoGameQALoader

    Example:
        # Load only needle-in-haystack and glitch detection tasks
        loader = create_videogameqa_loader(
            data_dir="data/videogameqa",
            tasks=["needle_haystack", "glitch_detection"],
            max_samples=500,
        )
    """
    config = BenchmarkConfig(
        data_dir=data_dir,
        max_samples=max_samples,
        tasks=tasks,
    )
    return VideoGameQALoader(config)
