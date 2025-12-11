"""
Base classes for benchmark data loading.

Provides a unified interface for loading samples from various benchmarks,
enabling consistent evaluation across different data formats.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of benchmark tasks."""

    GLITCH_DETECTION = "glitch_detection"
    PHYSICS_VIOLATION = "physics_violation"
    VISUAL_QA = "visual_qa"
    NEEDLE_IN_HAYSTACK = "needle_in_haystack"
    BUG_REPORT = "bug_report"
    TEMPORAL_REASONING = "temporal_reasoning"
    ACTION_RECOGNITION = "action_recognition"
    OBJECT_TRACKING = "object_tracking"


class AnswerFormat(Enum):
    """Expected answer format."""

    BINARY = "binary"  # Yes/No
    MCQ = "mcq"  # Multiple choice A/B/C/D
    FREE_TEXT = "free_text"  # Open-ended
    CLASSIFICATION = "classification"  # Category label


@dataclass
class BenchmarkSample:
    """
    A single benchmark sample with standardized fields.

    This provides a unified interface regardless of the source benchmark.
    """

    # Identity
    sample_id: str
    benchmark_name: str

    # Video/Image data
    video_path: Optional[str] = None
    image_path: Optional[str] = None  # For image-only samples
    video_url: Optional[str] = None  # For downloading

    # Question
    question: str = ""
    options: list[str] = field(default_factory=list)  # For MCQ
    ground_truth: str = ""  # Correct answer

    # Task metadata
    task_type: TaskType = TaskType.VISUAL_QA
    answer_format: AnswerFormat = AnswerFormat.FREE_TEXT
    category: str = "general"
    subcategory: str = ""

    # Video metadata
    video_duration_sec: Optional[float] = None
    start_time_sec: Optional[float] = None  # For temporal localization
    end_time_sec: Optional[float] = None

    # Additional context
    game_name: Optional[str] = None
    description: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sample_id": self.sample_id,
            "benchmark_name": self.benchmark_name,
            "video_path": self.video_path,
            "image_path": self.image_path,
            "question": self.question,
            "options": self.options,
            "ground_truth": self.ground_truth,
            "task_type": self.task_type.value,
            "answer_format": self.answer_format.value,
            "category": self.category,
            "game_name": self.game_name,
            "video_duration_sec": self.video_duration_sec,
        }

    def get_prompt(self, include_options: bool = True) -> str:
        """Generate a prompt for the LLM."""
        prompt = self.question

        if include_options and self.options:
            prompt += "\n\nOptions:"
            for i, opt in enumerate(self.options):
                letter = chr(ord('A') + i)
                prompt += f"\n{letter}) {opt}"
            prompt += "\n\nAnswer with the letter of the correct option."

        return prompt


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark loading."""

    # Data paths
    data_dir: str = ""
    video_dir: Optional[str] = None
    annotation_file: Optional[str] = None

    # Subset selection
    max_samples: Optional[int] = None
    tasks: Optional[list[str]] = None  # Filter by task type
    categories: Optional[list[str]] = None  # Filter by category
    min_duration_sec: Optional[float] = None
    max_duration_sec: Optional[float] = None

    # Sampling
    random_seed: int = 42
    shuffle: bool = False

    def __post_init__(self):
        if self.data_dir:
            self.data_dir = str(Path(self.data_dir).expanduser())


@dataclass
class BenchmarkInfo:
    """Information about a benchmark."""

    name: str
    version: str
    description: str
    total_samples: int
    tasks: list[str]
    categories: list[str]
    avg_video_duration_sec: Optional[float] = None
    citation: Optional[str] = None
    url: Optional[str] = None


class BenchmarkLoader(ABC):
    """
    Abstract base class for benchmark data loaders.

    Implementations should handle the specific format of each benchmark
    and convert samples to the standardized BenchmarkSample format.
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize the loader.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self._samples: list[BenchmarkSample] = []
        self._loaded = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Benchmark name."""
        pass

    @property
    @abstractmethod
    def info(self) -> BenchmarkInfo:
        """Get benchmark information."""
        pass

    @abstractmethod
    def _load_annotations(self) -> list[dict]:
        """Load raw annotations from benchmark files."""
        pass

    @abstractmethod
    def _convert_sample(self, raw: dict, idx: int) -> BenchmarkSample:
        """Convert a raw annotation to BenchmarkSample."""
        pass

    def load(self) -> list[BenchmarkSample]:
        """
        Load all samples from the benchmark.

        Returns:
            List of BenchmarkSample objects
        """
        if self._loaded:
            return self._samples

        logger.info(f"Loading {self.name} benchmark from {self.config.data_dir}")

        # Load raw annotations
        raw_annotations = self._load_annotations()
        logger.info(f"Found {len(raw_annotations)} raw annotations")

        # Convert to samples
        samples = []
        for idx, raw in enumerate(raw_annotations):
            try:
                sample = self._convert_sample(raw, idx)
                samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to convert sample {idx}: {e}")

        # Apply filters
        samples = self._apply_filters(samples)

        # Apply max_samples limit
        if self.config.max_samples and len(samples) > self.config.max_samples:
            if self.config.shuffle:
                import random
                random.seed(self.config.random_seed)
                random.shuffle(samples)
            samples = samples[:self.config.max_samples]

        self._samples = samples
        self._loaded = True

        logger.info(f"Loaded {len(self._samples)} samples from {self.name}")
        return self._samples

    def _apply_filters(self, samples: list[BenchmarkSample]) -> list[BenchmarkSample]:
        """Apply configured filters to samples."""
        filtered = samples

        # Filter by tasks
        if self.config.tasks:
            task_set = set(self.config.tasks)
            filtered = [s for s in filtered if s.task_type.value in task_set]

        # Filter by categories
        if self.config.categories:
            cat_set = set(self.config.categories)
            filtered = [s for s in filtered if s.category in cat_set]

        # Filter by duration
        if self.config.min_duration_sec:
            filtered = [
                s for s in filtered
                if s.video_duration_sec is None or s.video_duration_sec >= self.config.min_duration_sec
            ]
        if self.config.max_duration_sec:
            filtered = [
                s for s in filtered
                if s.video_duration_sec is None or s.video_duration_sec <= self.config.max_duration_sec
            ]

        return filtered

    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._samples)

    def __iter__(self) -> Iterator[BenchmarkSample]:
        if not self._loaded:
            self.load()
        return iter(self._samples)

    def __getitem__(self, idx: int) -> BenchmarkSample:
        if not self._loaded:
            self.load()
        return self._samples[idx]

    def get_by_task(self, task_type: TaskType) -> list[BenchmarkSample]:
        """Get all samples of a specific task type."""
        if not self._loaded:
            self.load()
        return [s for s in self._samples if s.task_type == task_type]

    def get_by_category(self, category: str) -> list[BenchmarkSample]:
        """Get all samples of a specific category."""
        if not self._loaded:
            self.load()
        return [s for s in self._samples if s.category == category]

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the loaded samples."""
        if not self._loaded:
            self.load()

        task_counts = {}
        category_counts = {}
        durations = []

        for sample in self._samples:
            task = sample.task_type.value
            task_counts[task] = task_counts.get(task, 0) + 1

            category_counts[sample.category] = category_counts.get(sample.category, 0) + 1

            if sample.video_duration_sec:
                durations.append(sample.video_duration_sec)

        return {
            "total_samples": len(self._samples),
            "tasks": task_counts,
            "categories": category_counts,
            "avg_duration_sec": sum(durations) / len(durations) if durations else None,
            "min_duration_sec": min(durations) if durations else None,
            "max_duration_sec": max(durations) if durations else None,
        }

    def export_sample_list(self, path: str | Path) -> None:
        """Export sample list to JSON for inspection."""
        if not self._loaded:
            self.load()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        export = {
            "benchmark": self.name,
            "total_samples": len(self._samples),
            "samples": [s.to_dict() for s in self._samples],
        }

        with open(path, "w") as f:
            json.dump(export, f, indent=2)

        logger.info(f"Exported sample list to {path}")
