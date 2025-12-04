"""
Timeline Indexer Module.

This module aligns all modality outputs (visual, audio, OCR) on a unified
timeline to create external memory for the LLM. Key capabilities:
1. Multi-modal event alignment on common timeline
2. Structured time-tagged event representation
3. Retrieval-ready format for context selection
4. Deduplication and event merging

References:
- [A: 33] Timeline alignment and structured representation
- [A: 7] Time-tagged event format
- [A: 18, A: 34] Retrieval strategy for context limits
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Callable

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Types of modalities that contribute to timeline."""

    VISUAL = "visual"  # SAM/SigLIP entity observations
    TEMPORAL = "temporal"  # HiCo compressed tokens
    AUDIO = "audio"  # Qwen2-Audio events
    SPEECH = "speech"  # Transcribed speech
    OCR = "ocr"  # Extracted text
    SYSTEM = "system"  # System-generated events


class EventPriority(Enum):
    """Priority levels for event filtering/ranking."""

    CRITICAL = 0  # Must include (e.g., boss phase change)
    HIGH = 1  # Important game events
    MEDIUM = 2  # Normal observations
    LOW = 3  # Background/ambient
    DEBUG = 4  # Debugging only


@dataclass
class TimelineEvent:
    """
    A single event on the unified timeline.
    
    Represents any modality observation with time alignment.
    """

    timestamp: float  # Event time in seconds
    modality: ModalityType
    description: str  # Natural language description
    priority: EventPriority = EventPriority.MEDIUM
    duration: float = 0.0  # Event duration (0 for instant)
    entity_id: Optional[str] = None  # Associated entity from SAM
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

    @property
    def end_time(self) -> float:
        return self.timestamp + self.duration

    def format_compact(self) -> str:
        """Format as compact timeline entry."""
        time_str = f"[{self._format_time(self.timestamp)}]"

        if self.modality == ModalityType.SPEECH:
            return f'{time_str} "{self.description}"'
        elif self.modality == ModalityType.AUDIO:
            return f"{time_str} (Audio: {self.description})"
        elif self.modality == ModalityType.OCR:
            return f'{time_str} (Text: "{self.description}")'
        else:
            return f"{time_str} {self.description}"

    def format_verbose(self) -> str:
        """Format with full details."""
        time_str = self._format_time(self.timestamp)
        entity_str = f" [{self.entity_id}]" if self.entity_id else ""
        return f"[{time_str}]{entity_str} ({self.modality.value}) {self.description}"

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as MM:SS."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"


@dataclass
class TimelineSegment:
    """A segment of the timeline covering a time range."""

    start_time: float
    end_time: float
    events: list[TimelineEvent] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def event_count(self) -> int:
        return len(self.events)

    def get_summary(self) -> str:
        """Get a natural language summary of this segment."""
        if not self.events:
            return f"[{TimelineEvent._format_time(self.start_time)}-{TimelineEvent._format_time(self.end_time)}] No events"

        event_strs = [e.format_compact() for e in sorted(self.events, key=lambda e: e.timestamp)]
        return "\n".join(event_strs)


@dataclass
class TimelineConfig:
    """Configuration for Timeline Indexer."""

    # Event merging
    merge_window_sec: float = 0.5  # Merge events within this window
    dedupe_threshold: float = 0.9  # Text similarity for deduplication

    # Retrieval settings
    default_context_window: float = 30.0  # Default retrieval window (seconds)
    max_events_per_query: int = 50  # Maximum events to return

    # Formatting
    compact_format: bool = True  # Use compact or verbose format
    include_low_priority: bool = False  # Include LOW priority events


class EventMerger:
    """Merges and deduplicates timeline events."""

    def __init__(self, config: TimelineConfig):
        self.config = config

    def merge_nearby_events(
        self,
        events: list[TimelineEvent],
    ) -> list[TimelineEvent]:
        """
        Merge events that are temporally close and semantically similar.
        
        Args:
            events: List of events to merge
            
        Returns:
            Merged event list
        """
        if not events:
            return []

        # Sort by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        merged = []
        current_group: list[TimelineEvent] = [sorted_events[0]]

        for event in sorted_events[1:]:
            last_event = current_group[-1]

            # Check if should merge
            time_diff = event.timestamp - last_event.timestamp
            same_modality = event.modality == last_event.modality
            same_entity = event.entity_id == last_event.entity_id

            if time_diff <= self.config.merge_window_sec and same_modality and same_entity:
                current_group.append(event)
            else:
                # Finalize current group
                merged.append(self._merge_group(current_group))
                current_group = [event]

        # Finalize last group
        if current_group:
            merged.append(self._merge_group(current_group))

        return merged

    def _merge_group(self, group: list[TimelineEvent]) -> TimelineEvent:
        """Merge a group of similar events into one."""
        if len(group) == 1:
            return group[0]

        # Use earliest timestamp
        timestamp = min(e.timestamp for e in group)

        # Combine descriptions
        unique_descs = list(dict.fromkeys(e.description for e in group))
        description = "; ".join(unique_descs[:3])  # Limit combined length

        # Use highest priority
        priority = min((e.priority for e in group), key=lambda p: p.value)

        # Calculate duration spanning all events
        end_time = max(e.end_time for e in group)
        duration = end_time - timestamp

        return TimelineEvent(
            timestamp=timestamp,
            modality=group[0].modality,
            description=description,
            priority=priority,
            duration=duration,
            entity_id=group[0].entity_id,
            confidence=max(e.confidence for e in group),
        )

    def deduplicate(
        self,
        events: list[TimelineEvent],
    ) -> list[TimelineEvent]:
        """Remove duplicate events based on content similarity."""
        if not events:
            return []

        unique = []
        seen_descs: set[str] = set()

        for event in events:
            # Simple deduplication by exact match
            key = f"{event.modality.value}:{event.description.lower()[:50]}"
            if key not in seen_descs:
                unique.append(event)
                seen_descs.add(key)

        return unique


class TimelineIndexer:
    """
    Main interface for multi-modal timeline alignment.
    
    Aligns outputs from all perception modules onto a unified
    timeline to create structured external memory for the LLM.
    
    Example:
        >>> indexer = TimelineIndexer()
        >>> 
        >>> # Add events from different modalities
        >>> indexer.add_event(
        ...     timestamp=83.0,
        ...     modality=ModalityType.VISUAL,
        ...     description="Boss: HP drops to 50%",
        ...     entity_id="boss_dragon_001"
        ... )
        >>> indexer.add_event(
        ...     timestamp=84.0,
        ...     modality=ModalityType.AUDIO,
        ...     description="roar"
        ... )
        >>> 
        >>> # Build transcript
        >>> transcript = indexer.build_structured_transcript()
        >>> # "[01:23] Boss: HP drops to 50%\\n[01:24] (Audio: roar)"
        >>> 
        >>> # Query specific time range
        >>> events = indexer.query_range(80.0, 90.0)
    """

    def __init__(
        self,
        config: Optional[TimelineConfig] = None,
    ):
        """
        Initialize the Timeline Indexer.

        Args:
            config: Timeline configuration. Uses defaults if not provided.
        """
        self.config = config or TimelineConfig()
        self.merger = EventMerger(self.config)

        # Event storage
        self._events: list[TimelineEvent] = []
        self._events_by_modality: dict[ModalityType, list[TimelineEvent]] = {
            m: [] for m in ModalityType
        }
        self._events_by_entity: dict[str, list[TimelineEvent]] = {}

        # Timeline bounds
        self._min_time: float = float("inf")
        self._max_time: float = float("-inf")

        logger.info("TimelineIndexer initialized")

    def add_event(
        self,
        timestamp: float,
        modality: ModalityType,
        description: str,
        priority: EventPriority = EventPriority.MEDIUM,
        duration: float = 0.0,
        entity_id: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[dict] = None,
    ) -> TimelineEvent:
        """
        Add a single event to the timeline.
        
        Args:
            timestamp: Event time in seconds
            modality: Source modality type
            description: Natural language description
            priority: Event priority level
            duration: Event duration (0 for instant)
            entity_id: Associated entity ID
            confidence: Confidence score
            metadata: Additional metadata
            
        Returns:
            The created TimelineEvent
        """
        event = TimelineEvent(
            timestamp=timestamp,
            modality=modality,
            description=description,
            priority=priority,
            duration=duration,
            entity_id=entity_id,
            confidence=confidence,
            metadata=metadata or {},
        )

        self._events.append(event)
        self._events_by_modality[modality].append(event)

        if entity_id:
            if entity_id not in self._events_by_entity:
                self._events_by_entity[entity_id] = []
            self._events_by_entity[entity_id].append(event)

        # Update bounds
        self._min_time = min(self._min_time, timestamp)
        self._max_time = max(self._max_time, event.end_time)

        return event

    def add_events_batch(
        self,
        events: list[dict],
    ) -> list[TimelineEvent]:
        """
        Add multiple events from dictionaries.
        
        Args:
            events: List of event dictionaries with keys matching add_event params
            
        Returns:
            List of created TimelineEvent objects
        """
        created = []
        for event_dict in events:
            event = self.add_event(**event_dict)
            created.append(event)
        return created

    def add_from_hico_tokens(
        self,
        tokens: list,  # TemporalToken objects
    ) -> None:
        """Add events from HiCo temporal tokens."""
        for token in tokens:
            self.add_event(
                timestamp=token.start_time,
                modality=ModalityType.TEMPORAL,
                description=f"Temporal context ({token.source_frame_count} frames)",
                duration=token.end_time - token.start_time,
                metadata={"compression_level": token.compression_level.value},
            )

    def add_from_sam_entities(
        self,
        entities: list,  # TrackedEntity objects
        frame_idx: int,
        timestamp: float,
    ) -> None:
        """Add events from SAM entity detections."""
        for entity in entities:
            if frame_idx in entity.frame_masks:
                mask = entity.frame_masks[frame_idx]
                self.add_event(
                    timestamp=timestamp,
                    modality=ModalityType.VISUAL,
                    description=f"{entity.concept_label} detected",
                    entity_id=entity.entity_id,
                    confidence=mask.confidence,
                    metadata={"bbox": mask.bbox.to_xyxy()},
                )

    def add_from_ocr_frame(
        self,
        ocr_frame,  # OCRFrame object
    ) -> None:
        """Add events from OCR detections."""
        for detection in ocr_frame.detections:
            self.add_event(
                timestamp=ocr_frame.timestamp,
                modality=ModalityType.OCR,
                description=detection.text,
                confidence=detection.confidence,
                priority=EventPriority.MEDIUM if detection.category in ["damage", "health"] else EventPriority.LOW,
                metadata={"category": detection.category, "bbox": detection.bbox},
            )

    def add_from_audio_result(
        self,
        audio_result,  # AudioAnalysisResult object
    ) -> None:
        """Add events from audio analysis."""
        for seg in audio_result.transcriptions:
            self.add_event(
                timestamp=seg.start_time,
                modality=ModalityType.SPEECH,
                description=seg.text,
                duration=seg.duration,
                confidence=seg.confidence,
            )

        for event in audio_result.events:
            self.add_event(
                timestamp=event.start_time,
                modality=ModalityType.AUDIO,
                description=event.description,
                duration=event.duration,
                confidence=event.confidence,
            )

    def build_structured_transcript(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        modalities: Optional[list[ModalityType]] = None,
        max_events: Optional[int] = None,
    ) -> str:
        """
        Build a structured text transcript of the timeline.
        
        This is the primary output format for LLM consumption.
        
        Args:
            start_time: Start of range (default: beginning)
            end_time: End of range (default: end)
            modalities: Filter by modalities (default: all)
            max_events: Maximum events to include
            
        Returns:
            Formatted timeline transcript
        """
        events = self._get_filtered_events(start_time, end_time, modalities)

        # Apply max_events limit
        max_events = max_events or self.config.max_events_per_query
        if len(events) > max_events:
            # Prioritize higher priority events
            events.sort(key=lambda e: (e.priority.value, e.timestamp))
            events = events[:max_events]
            events.sort(key=lambda e: e.timestamp)

        if self.config.compact_format:
            lines = [e.format_compact() for e in events]
        else:
            lines = [e.format_verbose() for e in events]

        return "\n".join(lines)

    def query_range(
        self,
        start_time: float,
        end_time: float,
        modalities: Optional[list[ModalityType]] = None,
    ) -> list[TimelineEvent]:
        """
        Query events within a time range.
        
        Args:
            start_time: Range start
            end_time: Range end
            modalities: Optional modality filter
            
        Returns:
            List of events in range
        """
        return self._get_filtered_events(start_time, end_time, modalities)

    def query_around_timestamp(
        self,
        timestamp: float,
        window: Optional[float] = None,
    ) -> list[TimelineEvent]:
        """
        Query events around a specific timestamp.
        
        Args:
            timestamp: Center timestamp
            window: Total window size (default from config)
            
        Returns:
            List of events in window
        """
        window = window or self.config.default_context_window
        half_window = window / 2
        return self.query_range(timestamp - half_window, timestamp + half_window)

    def query_by_entity(
        self,
        entity_id: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> list[TimelineEvent]:
        """
        Query all events for a specific entity.
        
        Args:
            entity_id: Entity ID from SAM
            start_time: Optional start filter
            end_time: Optional end filter
            
        Returns:
            List of events for entity
        """
        events = self._events_by_entity.get(entity_id, [])

        if start_time is not None:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time is not None:
            events = [e for e in events if e.timestamp <= end_time]

        return sorted(events, key=lambda e: e.timestamp)

    def query_by_modality(
        self,
        modality: ModalityType,
    ) -> list[TimelineEvent]:
        """Get all events of a specific modality."""
        return sorted(
            self._events_by_modality[modality],
            key=lambda e: e.timestamp,
        )

    def get_segments(
        self,
        segment_duration: float = 10.0,
    ) -> list[TimelineSegment]:
        """
        Divide timeline into fixed-duration segments.
        
        Args:
            segment_duration: Duration of each segment
            
        Returns:
            List of TimelineSegment objects
        """
        if not self._events:
            return []

        segments = []
        current_start = self._min_time

        while current_start < self._max_time:
            current_end = current_start + segment_duration
            segment_events = [
                e for e in self._events
                if current_start <= e.timestamp < current_end
            ]

            segments.append(
                TimelineSegment(
                    start_time=current_start,
                    end_time=current_end,
                    events=sorted(segment_events, key=lambda e: e.timestamp),
                )
            )

            current_start = current_end

        return segments

    def _get_filtered_events(
        self,
        start_time: Optional[float],
        end_time: Optional[float],
        modalities: Optional[list[ModalityType]],
    ) -> list[TimelineEvent]:
        """Get events with filters applied."""
        events = self._events.copy()

        # Time filter
        if start_time is not None:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time is not None:
            events = [e for e in events if e.timestamp <= end_time]

        # Modality filter
        if modalities:
            events = [e for e in events if e.modality in modalities]

        # Priority filter
        if not self.config.include_low_priority:
            events = [e for e in events if e.priority != EventPriority.LOW]

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)

        return events

    def merge_and_dedupe(self) -> int:
        """
        Merge and deduplicate all events in timeline.
        
        Returns:
            Number of events removed
        """
        original_count = len(self._events)

        # Merge nearby events
        self._events = self.merger.merge_nearby_events(self._events)

        # Deduplicate
        self._events = self.merger.deduplicate(self._events)

        # Rebuild indices
        self._rebuild_indices()

        removed = original_count - len(self._events)
        logger.info(f"Merged/deduped: {original_count} -> {len(self._events)} events")
        return removed

    def _rebuild_indices(self) -> None:
        """Rebuild secondary indices after modification."""
        self._events_by_modality = {m: [] for m in ModalityType}
        self._events_by_entity = {}

        for event in self._events:
            self._events_by_modality[event.modality].append(event)
            if event.entity_id:
                if event.entity_id not in self._events_by_entity:
                    self._events_by_entity[event.entity_id] = []
                self._events_by_entity[event.entity_id].append(event)

    def get_statistics(self) -> dict:
        """Get timeline statistics."""
        modality_counts = {
            m.value: len(events)
            for m, events in self._events_by_modality.items()
        }

        return {
            "total_events": len(self._events),
            "unique_entities": len(self._events_by_entity),
            "time_range": (self._min_time, self._max_time),
            "duration": self._max_time - self._min_time if self._events else 0,
            "events_by_modality": modality_counts,
        }

    def clear(self) -> None:
        """Clear all events from timeline."""
        self._events.clear()
        self._events_by_modality = {m: [] for m in ModalityType}
        self._events_by_entity.clear()
        self._min_time = float("inf")
        self._max_time = float("-inf")
        logger.info("Timeline cleared")


def create_timeline_indexer(
    merge_window: float = 0.5,
    max_events_per_query: int = 50,
) -> TimelineIndexer:
    """
    Factory function to create a Timeline Indexer.

    Args:
        merge_window: Time window for event merging
        max_events_per_query: Maximum events per retrieval

    Returns:
        Configured TimelineIndexer instance
    """
    config = TimelineConfig(
        merge_window_sec=merge_window,
        max_events_per_query=max_events_per_query,
    )
    return TimelineIndexer(config=config)