"""
Perception Caching Infrastructure.

Provides two-stage caching for evaluation:
1. Stage A: Compute and cache perception features (embeddings, timeline, KB)
2. Stage B: Load cached features for fast QA evaluation

This enables running multiple model configurations (Baseline-Plain, GVP-Light,
GVP-Full) without recomputing expensive perception features.

Cache structure:
    cache_dir/
    ├── {video_hash}/
    │   ├── metadata.json      # Video info, cache version
    │   ├── frames.npz         # Sampled frame indices
    │   ├── siglip.npz         # SigLIP embeddings
    │   ├── videomae.npz       # VideoMAE temporal features
    │   ├── sam_entities.json  # SAM3 entity detections
    │   ├── ocr.json           # OCR text extractions
    │   ├── audio.json         # Audio events + ASR
    │   ├── timeline.json      # Timeline index export
    │   └── kb.json            # Knowledge base export
    └── index.json             # Cache index for quick lookup
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Cache version for compatibility checking
CACHE_VERSION = "1.0.0"


@dataclass
class CacheConfig:
    """Configuration for perception cache."""

    cache_dir: str = "data/cache/perception"
    max_cache_size_gb: float = 50.0
    compression: bool = True

    # What to cache
    cache_frames: bool = True
    cache_siglip: bool = True
    cache_videomae: bool = True
    cache_sam: bool = True
    cache_ocr: bool = True
    cache_audio: bool = True
    cache_timeline: bool = True
    cache_kb: bool = True

    # Frame sampling
    target_fps: float = 4.0  # Frames per second for caching
    max_frames: int = 512  # Maximum frames to cache per video


@dataclass
class CachedFeatures:
    """Container for cached perception features."""

    video_hash: str
    video_path: str
    video_duration_sec: float

    # Frame data
    frame_indices: Optional[np.ndarray] = None
    frame_timestamps: Optional[np.ndarray] = None

    # Embeddings
    siglip_embeddings: Optional[np.ndarray] = None  # (N, 1152)
    videomae_embeddings: Optional[np.ndarray] = None  # (N, 768)

    # Structured data
    sam_entities: list[dict] = field(default_factory=list)
    ocr_detections: list[dict] = field(default_factory=list)
    audio_events: list[dict] = field(default_factory=list)
    asr_segments: list[dict] = field(default_factory=list)

    # High-level structures
    timeline: Optional[dict] = None
    knowledge_base: Optional[dict] = None

    # Metadata
    cache_version: str = CACHE_VERSION
    cached_at: str = ""
    perception_config: dict = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        """Get metadata dictionary."""
        return {
            "video_hash": self.video_hash,
            "video_path": self.video_path,
            "video_duration_sec": self.video_duration_sec,
            "cache_version": self.cache_version,
            "cached_at": self.cached_at,
            "perception_config": self.perception_config,
            "has_siglip": self.siglip_embeddings is not None,
            "has_videomae": self.videomae_embeddings is not None,
            "has_sam": len(self.sam_entities) > 0,
            "has_ocr": len(self.ocr_detections) > 0,
            "has_audio": len(self.audio_events) > 0,
            "has_timeline": self.timeline is not None,
            "has_kb": self.knowledge_base is not None,
            "num_frames": len(self.frame_indices) if self.frame_indices is not None else 0,
        }


class PerceptionCache:
    """
    Two-stage perception caching for efficient evaluation.

    Stage A (caching): Run perception pipeline and save results
    Stage B (loading): Load cached features for QA evaluation

    Example:
        cache = PerceptionCache(cache_dir="data/cache")

        # Stage A: Cache perception for a video
        if not cache.has_cache(video_path):
            features = run_perception_pipeline(video_path)
            cache.save(video_path, features)

        # Stage B: Load cached features for evaluation
        features = cache.load(video_path)
        answer = run_qa_model(features, question)
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize the perception cache.

        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load or create index
        self._index_path = self.cache_dir / "index.json"
        self._index: dict[str, dict] = {}
        self._load_index()

        logger.info(f"PerceptionCache initialized at {self.cache_dir}")
        logger.info(f"  Cached videos: {len(self._index)}")

    def _load_index(self) -> None:
        """Load cache index from disk."""
        if self._index_path.exists():
            try:
                with open(self._index_path, "r") as f:
                    self._index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                self._index = {}

    def _save_index(self) -> None:
        """Save cache index to disk."""
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    @staticmethod
    def compute_video_hash(video_path: str) -> str:
        """
        Compute a hash for a video file.

        Uses file path + modification time + size for fast hashing.
        """
        path = Path(video_path)
        if not path.exists():
            # Use path string as fallback
            return hashlib.md5(video_path.encode()).hexdigest()[:16]

        stat = path.stat()
        hash_input = f"{video_path}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

    def _get_cache_path(self, video_hash: str) -> Path:
        """Get the cache directory path for a video."""
        return self.cache_dir / video_hash

    def has_cache(self, video_path: str) -> bool:
        """Check if a video has cached features."""
        video_hash = self.compute_video_hash(video_path)
        return video_hash in self._index

    def get_cache_info(self, video_path: str) -> Optional[dict]:
        """Get cache metadata for a video."""
        video_hash = self.compute_video_hash(video_path)
        return self._index.get(video_hash)

    def save(
        self,
        video_path: str,
        features: CachedFeatures,
    ) -> str:
        """
        Save perception features to cache.

        Args:
            video_path: Path to source video
            features: CachedFeatures object to save

        Returns:
            Cache directory path
        """
        video_hash = self.compute_video_hash(video_path)
        cache_path = self._get_cache_path(video_hash)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Update features metadata
        features.video_hash = video_hash
        features.video_path = video_path
        features.cache_version = CACHE_VERSION
        features.cached_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Save metadata
        with open(cache_path / "metadata.json", "w") as f:
            json.dump(features.to_metadata(), f, indent=2)

        # Save frame data
        if self.config.cache_frames and features.frame_indices is not None:
            np.savez_compressed(
                cache_path / "frames.npz",
                indices=features.frame_indices,
                timestamps=features.frame_timestamps,
            )

        # Save embeddings
        if self.config.cache_siglip and features.siglip_embeddings is not None:
            np.savez_compressed(
                cache_path / "siglip.npz",
                embeddings=features.siglip_embeddings,
            )

        if self.config.cache_videomae and features.videomae_embeddings is not None:
            np.savez_compressed(
                cache_path / "videomae.npz",
                embeddings=features.videomae_embeddings,
            )

        # Save structured data
        if self.config.cache_sam and features.sam_entities:
            with open(cache_path / "sam_entities.json", "w") as f:
                json.dump(features.sam_entities, f)

        if self.config.cache_ocr and features.ocr_detections:
            with open(cache_path / "ocr.json", "w") as f:
                json.dump(features.ocr_detections, f)

        if self.config.cache_audio and (features.audio_events or features.asr_segments):
            with open(cache_path / "audio.json", "w") as f:
                json.dump({
                    "events": features.audio_events,
                    "asr": features.asr_segments,
                }, f)

        # Save high-level structures
        if self.config.cache_timeline and features.timeline:
            with open(cache_path / "timeline.json", "w") as f:
                json.dump(features.timeline, f)

        if self.config.cache_kb and features.knowledge_base:
            with open(cache_path / "kb.json", "w") as f:
                json.dump(features.knowledge_base, f)

        # Update index
        self._index[video_hash] = features.to_metadata()
        self._save_index()

        logger.info(f"Cached features for {video_path} -> {cache_path}")
        return str(cache_path)

    def load(self, video_path: str) -> Optional[CachedFeatures]:
        """
        Load cached features for a video.

        Args:
            video_path: Path to source video

        Returns:
            CachedFeatures or None if not cached
        """
        video_hash = self.compute_video_hash(video_path)

        if video_hash not in self._index:
            return None

        cache_path = self._get_cache_path(video_hash)
        if not cache_path.exists():
            # Index out of sync
            del self._index[video_hash]
            self._save_index()
            return None

        # Load metadata
        metadata_path = cache_path / "metadata.json"
        if not metadata_path.exists():
            return None

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Check version
        if metadata.get("cache_version") != CACHE_VERSION:
            logger.warning(
                f"Cache version mismatch for {video_path}: "
                f"{metadata.get('cache_version')} != {CACHE_VERSION}"
            )

        features = CachedFeatures(
            video_hash=video_hash,
            video_path=metadata.get("video_path", video_path),
            video_duration_sec=metadata.get("video_duration_sec", 0.0),
            cache_version=metadata.get("cache_version", CACHE_VERSION),
            cached_at=metadata.get("cached_at", ""),
            perception_config=metadata.get("perception_config", {}),
        )

        # Load frame data
        frames_path = cache_path / "frames.npz"
        if frames_path.exists():
            data = np.load(frames_path)
            features.frame_indices = data.get("indices")
            features.frame_timestamps = data.get("timestamps")

        # Load embeddings
        siglip_path = cache_path / "siglip.npz"
        if siglip_path.exists():
            data = np.load(siglip_path)
            features.siglip_embeddings = data.get("embeddings")

        videomae_path = cache_path / "videomae.npz"
        if videomae_path.exists():
            data = np.load(videomae_path)
            features.videomae_embeddings = data.get("embeddings")

        # Load structured data
        sam_path = cache_path / "sam_entities.json"
        if sam_path.exists():
            with open(sam_path, "r") as f:
                features.sam_entities = json.load(f)

        ocr_path = cache_path / "ocr.json"
        if ocr_path.exists():
            with open(ocr_path, "r") as f:
                features.ocr_detections = json.load(f)

        audio_path = cache_path / "audio.json"
        if audio_path.exists():
            with open(audio_path, "r") as f:
                audio_data = json.load(f)
                features.audio_events = audio_data.get("events", [])
                features.asr_segments = audio_data.get("asr", [])

        # Load high-level structures
        timeline_path = cache_path / "timeline.json"
        if timeline_path.exists():
            with open(timeline_path, "r") as f:
                features.timeline = json.load(f)

        kb_path = cache_path / "kb.json"
        if kb_path.exists():
            with open(kb_path, "r") as f:
                features.knowledge_base = json.load(f)

        logger.debug(f"Loaded cached features for {video_path}")
        return features

    def invalidate(self, video_path: str) -> bool:
        """
        Invalidate (delete) cache for a video.

        Args:
            video_path: Path to source video

        Returns:
            True if cache was deleted
        """
        video_hash = self.compute_video_hash(video_path)

        if video_hash not in self._index:
            return False

        cache_path = self._get_cache_path(video_hash)

        # Delete cache directory
        import shutil
        if cache_path.exists():
            shutil.rmtree(cache_path)

        # Update index
        del self._index[video_hash]
        self._save_index()

        logger.info(f"Invalidated cache for {video_path}")
        return True

    def get_statistics(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_size = 0
        for video_hash in self._index:
            cache_path = self._get_cache_path(video_hash)
            if cache_path.exists():
                for file_path in cache_path.glob("**/*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size

        return {
            "num_cached_videos": len(self._index),
            "total_size_bytes": total_size,
            "total_size_gb": total_size / (1024**3),
            "cache_version": CACHE_VERSION,
            "cache_dir": str(self.cache_dir),
        }

    def list_cached_videos(self) -> list[dict]:
        """List all cached videos with metadata."""
        return [
            {"video_hash": k, **v}
            for k, v in self._index.items()
        ]


def create_perception_cache(
    cache_dir: str = "data/cache/perception",
    max_size_gb: float = 50.0,
) -> PerceptionCache:
    """
    Factory function to create a perception cache.

    Args:
        cache_dir: Directory for cache storage
        max_size_gb: Maximum cache size in GB

    Returns:
        Configured PerceptionCache
    """
    config = CacheConfig(
        cache_dir=cache_dir,
        max_cache_size_gb=max_size_gb,
    )
    return PerceptionCache(config)
