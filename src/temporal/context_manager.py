"""
Temporal Context Manager Module.

This module implements hierarchical temporal context management for long-horizon
video understanding. It maintains a multi-level summary structure:

Level 0 (Fine): Individual events and observations (1-5 seconds)
Level 1 (Clip): Summarized clip segments (10-30 seconds)
Level 2 (Scene): High-level scene summaries (1-5 minutes)
Level 3 (Session): Global session context (5+ minutes)

Key features:
- Automatic hierarchical compression as context grows
- Efficient retrieval at any temporal resolution
- Memory-bounded context with configurable limits
- LLM-friendly export format
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional
import json

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class ContextLevel(Enum):
    """Hierarchical context levels."""

    FINE = 0  # Raw observations (1-5 sec)
    CLIP = 1  # Clip summaries (10-30 sec)
    SCENE = 2  # Scene summaries (1-5 min)
    SESSION = 3  # Session summary (5+ min)


@dataclass
class ContextEntry:
    """A single entry in the temporal context."""

    start_time: float
    end_time: float
    level: ContextLevel
    content: str  # Text description
    embedding: Optional[Any] = None  # Optional embedding vector
    entity_ids: list[str] = field(default_factory=list)
    event_count: int = 1
    importance: float = 1.0  # Priority score for compression decisions
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get entry duration in seconds."""
        return self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "level": self.level.name,
            "content": self.content,
            "entity_ids": self.entity_ids,
            "event_count": self.event_count,
            "importance": self.importance,
        }

    def format_for_prompt(self, include_time: bool = True) -> str:
        """Format entry for LLM prompt."""
        if include_time:
            return f"[{self.start_time:.1f}s-{self.end_time:.1f}s] {self.content}"
        return self.content


@dataclass
class ContextManagerConfig:
    """Configuration for TemporalContextManager."""

    # Level thresholds (when to compress)
    fine_to_clip_threshold: int = 10  # Compress after N fine entries
    clip_to_scene_threshold: int = 6  # Compress after N clip entries
    scene_to_session_threshold: int = 4  # Compress after N scene entries

    # Time-based compression triggers
    fine_duration_sec: float = 5.0  # Max fine entry duration
    clip_duration_sec: float = 30.0  # Target clip summary duration
    scene_duration_sec: float = 300.0  # Target scene summary duration (5 min)

    # Memory limits (max entries per level)
    max_fine_entries: int = 50
    max_clip_entries: int = 20
    max_scene_entries: int = 10
    max_session_entries: int = 5

    # Output limits
    max_context_tokens: int = 2000  # Approx token limit for LLM context
    max_context_chars: int = 8000  # Character limit approximation

    # Importance scoring
    high_importance_keywords: list[str] = field(default_factory=lambda: [
        "boss", "death", "killed", "damage", "health", "critical",
        "spawn", "phase", "attack", "defeat", "victory", "fail"
    ])


class TemporalContextManager:
    """
    Manages hierarchical temporal context for long-horizon video understanding.

    The manager maintains multiple levels of temporal summaries, automatically
    compressing fine-grained observations into higher-level summaries as the
    context grows. This enables efficient reasoning over extended timelines
    while staying within token limits.

    Example:
        >>> manager = TemporalContextManager()
        >>>
        >>> # Add fine-grained observations
        >>> manager.add_observation(
        ...     start_time=0.0,
        ...     end_time=2.0,
        ...     content="Player enters dark hallway",
        ...     entity_ids=["player_001"]
        ... )
        >>> manager.add_observation(
        ...     start_time=2.0,
        ...     end_time=5.0,
        ...     content="Skeleton enemy spawns ahead",
        ...     entity_ids=["enemy_001"]
        ... )
        >>>
        >>> # Get context for LLM
        >>> context = manager.get_context_for_prompt()
        >>> print(context)
    """

    def __init__(
        self,
        config: Optional[ContextManagerConfig] = None,
        summarizer: Optional[Callable[[list[str]], str]] = None,
    ):
        """
        Initialize the Temporal Context Manager.

        Args:
            config: Configuration options. Uses defaults if not provided.
            summarizer: Optional function to summarize multiple entries.
                       If not provided, uses simple concatenation.
        """
        self.config = config or ContextManagerConfig()
        self.summarizer = summarizer or self._default_summarizer

        # Hierarchical storage
        self._fine: deque[ContextEntry] = deque(maxlen=self.config.max_fine_entries)
        self._clips: deque[ContextEntry] = deque(maxlen=self.config.max_clip_entries)
        self._scenes: deque[ContextEntry] = deque(maxlen=self.config.max_scene_entries)
        self._session: deque[ContextEntry] = deque(maxlen=self.config.max_session_entries)

        # Tracking
        self._current_time: float = 0.0
        self._total_observations: int = 0
        self._total_compressions: int = 0

        logger.info("TemporalContextManager initialized")

    def add_observation(
        self,
        start_time: float,
        end_time: float,
        content: str,
        entity_ids: Optional[list[str]] = None,
        importance: Optional[float] = None,
        embedding: Optional[Any] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Add a fine-grained observation to the context.

        Args:
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            content: Text description of the observation
            entity_ids: List of involved entity IDs
            importance: Optional importance score (auto-computed if None)
            embedding: Optional embedding vector
            metadata: Additional metadata
        """
        # Compute importance if not provided
        if importance is None:
            importance = self._compute_importance(content)

        entry = ContextEntry(
            start_time=start_time,
            end_time=end_time,
            level=ContextLevel.FINE,
            content=content,
            embedding=embedding,
            entity_ids=entity_ids or [],
            event_count=1,
            importance=importance,
            metadata=metadata or {},
        )

        self._fine.append(entry)
        self._current_time = max(self._current_time, end_time)
        self._total_observations += 1

        # Check if compression is needed
        self._maybe_compress()

    def add_from_timeline_event(
        self,
        event: Any,  # TimelineEvent from timeline_indexer
    ) -> None:
        """
        Add observation from a TimelineEvent.

        Args:
            event: TimelineEvent object from timeline_indexer
        """
        end_time = event.timestamp + (event.duration or 1.0)

        self.add_observation(
            start_time=event.timestamp,
            end_time=end_time,
            content=event.description,
            entity_ids=[event.entity_id] if event.entity_id else [],
            importance=self._priority_to_importance(event.priority),
            metadata={"modality": event.modality.value if hasattr(event.modality, 'value') else str(event.modality)},
        )

    def _priority_to_importance(self, priority: Any) -> float:
        """Convert event priority to importance score."""
        priority_map = {
            0: 1.0,  # CRITICAL
            1: 0.8,  # HIGH
            2: 0.5,  # MEDIUM
            3: 0.3,  # LOW
            4: 0.1,  # DEBUG
        }
        if hasattr(priority, 'value'):
            return priority_map.get(priority.value, 0.5)
        return priority_map.get(priority, 0.5)

    def _compute_importance(self, content: str) -> float:
        """Compute importance score based on content keywords."""
        content_lower = content.lower()
        score = 0.5  # Base importance

        for keyword in self.config.high_importance_keywords:
            if keyword in content_lower:
                score += 0.15

        return min(1.0, score)

    def _default_summarizer(self, contents: list[str]) -> str:
        """Default summarization: combine contents with semicolons."""
        # Deduplicate similar content
        seen = set()
        unique = []
        for c in contents:
            key = c[:50].lower()  # Use first 50 chars as key
            if key not in seen:
                seen.add(key)
                unique.append(c)

        if len(unique) <= 3:
            return "; ".join(unique)
        else:
            # Keep most important items
            return f"{unique[0]}; ...[{len(unique)-2} events]...; {unique[-1]}"

    def _maybe_compress(self) -> None:
        """Check if compression is needed at any level and perform it."""
        # Fine -> Clip compression
        if len(self._fine) >= self.config.fine_to_clip_threshold:
            self._compress_fine_to_clip()

        # Clip -> Scene compression
        if len(self._clips) >= self.config.clip_to_scene_threshold:
            self._compress_clip_to_scene()

        # Scene -> Session compression
        if len(self._scenes) >= self.config.scene_to_session_threshold:
            self._compress_scene_to_session()

    def _compress_fine_to_clip(self) -> None:
        """Compress fine entries into a clip summary."""
        if len(self._fine) < 2:
            return

        # Take oldest entries for compression
        entries_to_compress = []
        num_to_take = min(
            self.config.fine_to_clip_threshold,
            len(self._fine) - 2  # Keep at least 2 recent entries
        )

        for _ in range(num_to_take):
            if self._fine:
                entries_to_compress.append(self._fine.popleft())

        if not entries_to_compress:
            return

        # Create clip summary
        clip = self._merge_entries(entries_to_compress, ContextLevel.CLIP)
        self._clips.append(clip)
        self._total_compressions += 1

        logger.debug(
            f"Compressed {len(entries_to_compress)} fine entries into clip "
            f"[{clip.start_time:.1f}s-{clip.end_time:.1f}s]"
        )

    def _compress_clip_to_scene(self) -> None:
        """Compress clip entries into a scene summary."""
        if len(self._clips) < 2:
            return

        entries_to_compress = []
        num_to_take = min(
            self.config.clip_to_scene_threshold,
            len(self._clips) - 1
        )

        for _ in range(num_to_take):
            if self._clips:
                entries_to_compress.append(self._clips.popleft())

        if not entries_to_compress:
            return

        scene = self._merge_entries(entries_to_compress, ContextLevel.SCENE)
        self._scenes.append(scene)
        self._total_compressions += 1

        logger.debug(
            f"Compressed {len(entries_to_compress)} clips into scene "
            f"[{scene.start_time:.1f}s-{scene.end_time:.1f}s]"
        )

    def _compress_scene_to_session(self) -> None:
        """Compress scene entries into session summary."""
        if len(self._scenes) < 2:
            return

        entries_to_compress = []
        num_to_take = min(
            self.config.scene_to_session_threshold,
            len(self._scenes) - 1
        )

        for _ in range(num_to_take):
            if self._scenes:
                entries_to_compress.append(self._scenes.popleft())

        if not entries_to_compress:
            return

        session = self._merge_entries(entries_to_compress, ContextLevel.SESSION)
        self._session.append(session)
        self._total_compressions += 1

        logger.debug(
            f"Compressed {len(entries_to_compress)} scenes into session "
            f"[{session.start_time:.1f}s-{session.end_time:.1f}s]"
        )

    def _merge_entries(
        self,
        entries: list[ContextEntry],
        target_level: ContextLevel,
    ) -> ContextEntry:
        """Merge multiple entries into a single higher-level entry."""
        if not entries:
            raise ValueError("Cannot merge empty entries list")

        # Compute time bounds
        start_time = min(e.start_time for e in entries)
        end_time = max(e.end_time for e in entries)

        # Collect all entity IDs
        all_entities = []
        for e in entries:
            all_entities.extend(e.entity_ids)
        unique_entities = list(dict.fromkeys(all_entities))  # Preserve order

        # Summarize content
        contents = [e.content for e in entries]
        summary = self.summarizer(contents)

        # Compute aggregate importance (weighted by event count)
        total_events = sum(e.event_count for e in entries)
        weighted_importance = sum(
            e.importance * e.event_count for e in entries
        ) / total_events if total_events > 0 else 0.5

        return ContextEntry(
            start_time=start_time,
            end_time=end_time,
            level=target_level,
            content=summary,
            entity_ids=unique_entities[:10],  # Limit entity list
            event_count=total_events,
            importance=weighted_importance,
            metadata={"merged_from": len(entries)},
        )

    def get_context_for_prompt(
        self,
        max_chars: Optional[int] = None,
        include_levels: Optional[list[ContextLevel]] = None,
        time_range: Optional[tuple[float, float]] = None,
    ) -> str:
        """
        Get formatted context string for LLM prompting.

        Args:
            max_chars: Maximum character limit. Uses config default if None.
            include_levels: Which context levels to include. All if None.
            time_range: Optional (start, end) time filter.

        Returns:
            Formatted context string ready for LLM prompt
        """
        max_chars = max_chars or self.config.max_context_chars

        # Collect entries from requested levels
        all_entries = []

        if include_levels is None or ContextLevel.SESSION in include_levels:
            all_entries.extend(self._session)
        if include_levels is None or ContextLevel.SCENE in include_levels:
            all_entries.extend(self._scenes)
        if include_levels is None or ContextLevel.CLIP in include_levels:
            all_entries.extend(self._clips)
        if include_levels is None or ContextLevel.FINE in include_levels:
            all_entries.extend(self._fine)

        # Filter by time range if specified
        if time_range:
            start_t, end_t = time_range
            all_entries = [
                e for e in all_entries
                if e.end_time >= start_t and e.start_time <= end_t
            ]

        # Sort by time
        all_entries.sort(key=lambda e: e.start_time)

        # Build context string within character limit
        lines = ["## Temporal Context"]
        current_level = None
        chars_used = len(lines[0])

        for entry in all_entries:
            # Add level header if changed
            if entry.level != current_level:
                header = f"\n### {entry.level.name.title()} Events"
                if chars_used + len(header) > max_chars:
                    break
                lines.append(header)
                chars_used += len(header)
                current_level = entry.level

            # Format and add entry
            line = f"- {entry.format_for_prompt()}"
            if chars_used + len(line) > max_chars:
                lines.append("- [... context truncated ...]")
                break
            lines.append(line)
            chars_used += len(line) + 1

        return "\n".join(lines)

    def get_context_at_time(
        self,
        timestamp: float,
        window_sec: float = 30.0,
    ) -> str:
        """
        Get context around a specific timestamp.

        Args:
            timestamp: Target timestamp in seconds
            window_sec: Window size around timestamp

        Returns:
            Formatted context string
        """
        return self.get_context_for_prompt(
            time_range=(timestamp - window_sec, timestamp + window_sec)
        )

    def get_recent_context(
        self,
        duration_sec: float = 60.0,
    ) -> str:
        """
        Get recent context from the last N seconds.

        Args:
            duration_sec: How far back to look

        Returns:
            Formatted context string
        """
        start_time = max(0, self._current_time - duration_sec)
        return self.get_context_for_prompt(
            time_range=(start_time, self._current_time)
        )

    def get_entity_context(
        self,
        entity_id: str,
    ) -> str:
        """
        Get all context entries involving a specific entity.

        Args:
            entity_id: Entity ID to filter by

        Returns:
            Formatted context string
        """
        all_entries = list(self._fine) + list(self._clips) + list(self._scenes) + list(self._session)
        filtered = [e for e in all_entries if entity_id in e.entity_ids]
        filtered.sort(key=lambda e: e.start_time)

        if not filtered:
            return f"No context found for entity: {entity_id}"

        lines = [f"## Context for Entity: {entity_id}"]
        for entry in filtered:
            lines.append(f"- {entry.format_for_prompt()}")

        return "\n".join(lines)

    def get_statistics(self) -> dict[str, Any]:
        """Get context manager statistics."""
        return {
            "current_time": self._current_time,
            "total_observations": self._total_observations,
            "total_compressions": self._total_compressions,
            "fine_entries": len(self._fine),
            "clip_entries": len(self._clips),
            "scene_entries": len(self._scenes),
            "session_entries": len(self._session),
            "total_entries": (
                len(self._fine) + len(self._clips) +
                len(self._scenes) + len(self._session)
            ),
        }

    def export_to_json(self, path: Optional[str] = None) -> dict:
        """
        Export context to JSON format.

        Args:
            path: Optional file path to save JSON

        Returns:
            Dictionary representation
        """
        export = {
            "statistics": self.get_statistics(),
            "fine": [e.to_dict() for e in self._fine],
            "clips": [e.to_dict() for e in self._clips],
            "scenes": [e.to_dict() for e in self._scenes],
            "session": [e.to_dict() for e in self._session],
        }

        if path:
            from pathlib import Path as PathLib
            p = PathLib(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump(export, f, indent=2)
            logger.info(f"Context exported to {path}")

        return export

    def clear(self) -> None:
        """Clear all context."""
        self._fine.clear()
        self._clips.clear()
        self._scenes.clear()
        self._session.clear()
        self._current_time = 0.0
        self._total_observations = 0
        self._total_compressions = 0
        logger.info("Context cleared")


def create_context_manager(
    max_fine_entries: int = 50,
    max_clip_entries: int = 20,
    summarizer: Optional[Callable[[list[str]], str]] = None,
) -> TemporalContextManager:
    """
    Factory function to create a configured TemporalContextManager.

    Args:
        max_fine_entries: Maximum fine-grained entries to keep
        max_clip_entries: Maximum clip entries to keep
        summarizer: Optional custom summarization function

    Returns:
        Configured TemporalContextManager instance
    """
    config = ContextManagerConfig(
        max_fine_entries=max_fine_entries,
        max_clip_entries=max_clip_entries,
    )
    return TemporalContextManager(config=config, summarizer=summarizer)
