"""
PhysGame Data Loader.

PhysGame is a benchmark for physical commonsense violations in gameplay videos.
It contains 880 videos with glitches and MCQ questions about what physics
rules were violated.

Paper: https://arxiv.org/abs/2406.18716
Website: https://physgame.github.io/

Physics domains:
- Mechanics (gravity, collision, friction)
- Kinematics (velocity, acceleration, trajectory)
- Optics (lighting, shadows, reflections)
- Materials (deformation, destruction, fluids)

Data format (expected):
    physgame/
    ├── annotations.json
    └── videos/
        ├── mechanics/
        │   ├── gravity_001.mp4
        │   └── ...
        └── ...

Annotation format:
    {
        "id": "mechanics_gravity_001",
        "video_path": "videos/mechanics/gravity_001.mp4",
        "domain": "mechanics",
        "category": "gravity",
        "question": "What physical law is violated in this video?",
        "options": ["Gravity", "Conservation of momentum", "Friction", "Inertia"],
        "answer": "A"
    }
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


# PhysGame physics domains and categories
PHYSICS_DOMAINS = {
    "mechanics": ["gravity", "collision", "friction", "inertia", "momentum"],
    "kinematics": ["velocity", "acceleration", "trajectory", "rotation"],
    "optics": ["lighting", "shadows", "reflections", "transparency"],
    "materials": ["deformation", "destruction", "fluids", "particles"],
}


class PhysGameLoader(BenchmarkLoader):
    """
    Loader for PhysGame dataset.

    Handles loading and converting PhysGame annotations for
    physical commonsense violation detection.
    """

    @property
    def name(self) -> str:
        return "PhysGame"

    @property
    def info(self) -> BenchmarkInfo:
        all_categories = []
        for cats in PHYSICS_DOMAINS.values():
            all_categories.extend(cats)

        return BenchmarkInfo(
            name="PhysGame",
            version="1.0",
            description="Physical commonsense violations in gameplay videos",
            total_samples=880,
            tasks=["physics_violation"],
            categories=all_categories,
            avg_video_duration_sec=8.0,  # Typically short clips
            citation="PhysGame: Uncovering Physical Commonsense Violations in Gameplay Videos",
            url="https://physgame.github.io/",
        )

    def _load_annotations(self) -> list[dict]:
        """Load annotations from PhysGame format."""
        annotations = []
        data_dir = Path(self.config.data_dir)

        # Try annotation files
        possible_files = [
            "annotations.json",
            "physgame.json",
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
                elif "data" in data:
                    annotations = data["data"]
                else:
                    annotations = list(data.values())

            logger.info(f"Loaded {len(annotations)} annotations from {annotation_file}")
        else:
            # Scan video directories by domain
            logger.warning(f"No annotation file found, scanning directories")
            annotations = self._scan_by_domain()

        return annotations

    def _scan_by_domain(self) -> list[dict]:
        """Scan video directories organized by physics domain."""
        annotations = []
        data_dir = Path(self.config.data_dir)
        video_dir = data_dir / (self.config.video_dir or "videos")

        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

        for domain, categories in PHYSICS_DOMAINS.items():
            domain_dir = video_dir / domain
            if domain_dir.exists():
                for video_path in domain_dir.glob("**/*"):
                    if video_path.suffix.lower() in video_extensions:
                        # Try to infer category from filename
                        category = "unknown"
                        for cat in categories:
                            if cat in video_path.stem.lower():
                                category = cat
                                break

                        annotations.append({
                            "id": video_path.stem,
                            "video_path": str(video_path.relative_to(data_dir)),
                            "domain": domain,
                            "category": category,
                        })

        # Also scan flat structure
        if not annotations:
            for video_path in video_dir.glob("**/*"):
                if video_path.suffix.lower() in video_extensions:
                    annotations.append({
                        "id": video_path.stem,
                        "video_path": str(video_path.relative_to(data_dir)),
                        "domain": "unknown",
                        "category": "unknown",
                    })

        return annotations

    def _convert_sample(self, raw: dict, idx: int) -> BenchmarkSample:
        """Convert PhysGame annotation to BenchmarkSample."""
        sample_id = raw.get("id", f"physgame_{idx:04d}")

        # Resolve video path
        video_path = raw.get("video_path", raw.get("video"))
        if video_path:
            full_path = Path(self.config.data_dir) / video_path
            video_path = str(full_path) if full_path.exists() else video_path

        # Build question
        question = raw.get(
            "question",
            "What physical law or commonsense rule is violated in this gameplay video?"
        )

        # Get options and answer
        options = raw.get("options", [])
        answer = raw.get("answer", "")

        # If no options provided, create default physics options
        if not options:
            domain = raw.get("domain", "unknown")
            if domain in PHYSICS_DOMAINS:
                # Create options from domain categories
                options = [cat.replace("_", " ").title() for cat in PHYSICS_DOMAINS[domain]]
                options.append("No violation")
            else:
                options = [
                    "Gravity violation",
                    "Collision physics error",
                    "Velocity/momentum error",
                    "Visual rendering error",
                    "No violation",
                ]

        # Determine answer format
        if len(options) > 0:
            answer_format = AnswerFormat.MCQ
        else:
            answer_format = AnswerFormat.FREE_TEXT

        # Category is the specific physics concept
        category = raw.get("category", raw.get("physics_category", "unknown"))
        domain = raw.get("domain", "unknown")

        return BenchmarkSample(
            sample_id=sample_id,
            benchmark_name=self.name,
            video_path=video_path,
            question=question,
            options=options,
            ground_truth=str(answer),
            task_type=TaskType.PHYSICS_VIOLATION,
            answer_format=answer_format,
            category=category,
            subcategory=domain,
            game_name=raw.get("game", raw.get("game_name")),
            description=raw.get("description"),
            video_duration_sec=raw.get("duration", raw.get("duration_sec")),
            metadata={
                "domain": domain,
                "violation_type": raw.get("violation_type"),
            },
        )

    def get_by_domain(self, domain: str) -> list[BenchmarkSample]:
        """Get all samples from a specific physics domain."""
        if not self._loaded:
            self.load()
        return [
            s for s in self._samples
            if s.metadata.get("domain") == domain or s.subcategory == domain
        ]


def create_physgame_loader(
    data_dir: str,
    max_samples: Optional[int] = None,
    domains: Optional[list[str]] = None,
) -> PhysGameLoader:
    """
    Factory function to create a PhysGame loader.

    Args:
        data_dir: Path to PhysGame data directory
        max_samples: Maximum samples to load
        domains: Filter by physics domains (mechanics, kinematics, optics, materials)

    Returns:
        Configured PhysGameLoader
    """
    config = BenchmarkConfig(
        data_dir=data_dir,
        max_samples=max_samples,
    )
    loader = PhysGameLoader(config)

    # Apply domain filter after loading if specified
    if domains:
        loader.load()
        loader._samples = [
            s for s in loader._samples
            if s.metadata.get("domain") in domains or s.subcategory in domains
        ]

    return loader
