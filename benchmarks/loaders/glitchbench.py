"""
GlitchBench Data Loader.

GlitchBench (CVPR 2024) is a benchmark for evaluating LMMs on glitch detection
in gameplay videos. It contains 593 glitches curated from r/GamePhysics across
205 games.

Paper: https://arxiv.org/abs/2312.05291
Dataset: https://huggingface.co/datasets/glitchbench/GlitchBench

Data format (expected):
    glitchbench/
    ├── annotations.json
    └── videos/
        ├── sample_001.mp4
        └── ...

Annotation format:
    {
        "id": "sample_001",
        "video_path": "videos/sample_001.mp4",
        "game": "GTA V",
        "glitch_type": "physics",
        "description": "Car floating in air",
        "question": "Is there a glitch in this gameplay?",
        "answer": "yes"
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .base import (
    AnswerFormat,
    BenchmarkConfig,
    BenchmarkInfo,
    BenchmarkLoader,
    BenchmarkSample,
    TaskType,
)

logger = logging.getLogger(__name__)


class GlitchBenchLoader(BenchmarkLoader):
    """
    Loader for GlitchBench dataset.

    Handles loading and converting GlitchBench annotations to
    the standardized BenchmarkSample format.
    """

    @property
    def name(self) -> str:
        return "GlitchBench"

    @property
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="GlitchBench",
            version="1.0",
            description="Benchmark for glitch detection in gameplay videos (CVPR 2024)",
            total_samples=593,
            tasks=["glitch_detection", "glitch_classification"],
            categories=[
                "physics", "rendering", "animation", "collision",
                "texture", "ai_behavior", "clipping", "other"
            ],
            citation="GlitchBench: Can Large Multimodal Models Detect Video Game Glitches? (CVPR 2024)",
            url="https://huggingface.co/datasets/glitchbench/GlitchBench",
        )

    def _load_annotations(self) -> list[dict]:
        """Load annotations from GlitchBench format (JSON or parquet)."""
        annotations = []
        data_dir = Path(self.config.data_dir)

        # First, try loading from parquet files (HuggingFace dataset format)
        parquet_dir = data_dir / "data"
        if parquet_dir.exists():
            parquet_files = list(parquet_dir.glob("*.parquet"))
            if parquet_files:
                try:
                    import pyarrow.parquet as pq
                    import tempfile
                    import os
                    
                    # Create a temp directory for extracted images
                    images_dir = data_dir / "images"
                    images_dir.mkdir(exist_ok=True)
                    
                    for pq_file in parquet_files:
                        table = pq.read_table(pq_file)
                        df_dict = {col: table[col].to_pylist() for col in table.column_names}
                        
                        for i in range(len(table)):
                            sample_id = df_dict.get("id", [None])[i] or f"sample_{i}"
                            
                            # Extract image from bytes and save to disk
                            image_data = df_dict.get("image", [None])[i]
                            image_path = None
                            if image_data:
                                if isinstance(image_data, dict) and "bytes" in image_data:
                                    img_bytes = image_data["bytes"]
                                    img_path = images_dir / f"{sample_id}.png"
                                    if not img_path.exists():
                                        with open(img_path, "wb") as f:
                                            f.write(img_bytes)
                                    image_path = str(img_path.relative_to(data_dir))
                                elif isinstance(image_data, bytes):
                                    img_path = images_dir / f"{sample_id}.png"
                                    if not img_path.exists():
                                        with open(img_path, "wb") as f:
                                            f.write(image_data)
                                    image_path = str(img_path.relative_to(data_dir))
                            
                            annotations.append({
                                "id": sample_id,
                                "image_path": image_path,
                                "game": df_dict.get("game", ["unknown"])[i] or "unknown",
                                "glitch_type": df_dict.get("glitch-type", ["unknown"])[i] or "unknown",
                                "description": df_dict.get("description", [""])[i] or "",
                                "source": df_dict.get("source", [""])[i] or "",
                                "answer": "yes",  # All GlitchBench samples are confirmed glitches
                            })
                    
                    logger.info(f"Loaded {len(annotations)} samples from parquet files")
                    return annotations
                    
                except ImportError:
                    logger.warning("pyarrow not installed, cannot read parquet files")
                except Exception as e:
                    logger.warning(f"Failed to load parquet files: {e}")

        # Try JSON annotation files
        possible_files = [
            "annotations.json",
            "glitchbench.json",
            "data.json",
            "metadata.json",
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

            # Handle different JSON structures
            if isinstance(data, list):
                annotations = data
            elif isinstance(data, dict):
                if "samples" in data:
                    annotations = data["samples"]
                elif "data" in data:
                    annotations = data["data"]
                elif "annotations" in data:
                    annotations = data["annotations"]
                else:
                    # Assume dict is keyed by sample ID
                    annotations = [{"id": k, **v} for k, v in data.items()]

            logger.info(f"Loaded {len(annotations)} annotations from {annotation_file}")
        else:
            # Fall back to scanning video directory
            logger.warning(f"No annotation file found in {data_dir}, scanning for videos")
            annotations = self._scan_videos()

        return annotations

    def _scan_videos(self) -> list[dict]:
        """Scan video directory to create basic annotations."""
        annotations = []
        data_dir = Path(self.config.data_dir)
        video_dir = data_dir / (self.config.video_dir or "videos")

        if not video_dir.exists():
            video_dir = data_dir

        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

        for video_path in video_dir.glob("**/*"):
            if video_path.suffix.lower() in video_extensions:
                annotations.append({
                    "id": video_path.stem,
                    "video_path": str(video_path.relative_to(data_dir)),
                    "game": "unknown",
                    "glitch_type": "unknown",
                })

        return annotations

    def _convert_sample(self, raw: dict, idx: int) -> BenchmarkSample:
        """Convert GlitchBench annotation to BenchmarkSample."""
        sample_id = raw.get("id", f"glitchbench_{idx:04d}")

        # Resolve video path
        video_path = raw.get("video_path", raw.get("video", raw.get("file")))
        if video_path:
            full_path = Path(self.config.data_dir) / video_path
            video_path = str(full_path) if full_path.exists() else video_path

        # Handle image-only samples
        image_path = raw.get("image_path", raw.get("image"))
        if image_path:
            full_path = Path(self.config.data_dir) / image_path
            image_path = str(full_path) if full_path.exists() else image_path

        # Build question
        question = raw.get("question", "Is there a glitch in this gameplay? Describe what you see.")

        # Determine if MCQ or binary
        options = raw.get("options", [])
        answer = raw.get("answer", raw.get("label", ""))

        if not options:
            # Default to binary yes/no
            options = ["Yes, there is a glitch", "No, this is normal gameplay"]
            answer_format = AnswerFormat.BINARY
        else:
            answer_format = AnswerFormat.MCQ

        # Map glitch type to category
        glitch_type = raw.get("glitch_type", raw.get("category", "unknown"))

        return BenchmarkSample(
            sample_id=sample_id,
            benchmark_name=self.name,
            video_path=video_path,
            image_path=image_path,
            question=question,
            options=options,
            ground_truth=str(answer),
            task_type=TaskType.GLITCH_DETECTION,
            answer_format=answer_format,
            category=glitch_type,
            game_name=raw.get("game", raw.get("game_name")),
            description=raw.get("description", raw.get("caption")),
            video_duration_sec=raw.get("duration", raw.get("duration_sec")),
            metadata={
                "source": raw.get("source", "r/GamePhysics"),
                "original_id": raw.get("original_id"),
            },
        )


def create_glitchbench_loader(
    data_dir: str,
    max_samples: Optional[int] = None,
    categories: Optional[list[str]] = None,
) -> GlitchBenchLoader:
    """
    Factory function to create a GlitchBench loader.

    Args:
        data_dir: Path to GlitchBench data directory
        max_samples: Maximum samples to load
        categories: Filter by glitch categories

    Returns:
        Configured GlitchBenchLoader
    """
    config = BenchmarkConfig(
        data_dir=data_dir,
        max_samples=max_samples,
        categories=categories,
    )
    return GlitchBenchLoader(config)
