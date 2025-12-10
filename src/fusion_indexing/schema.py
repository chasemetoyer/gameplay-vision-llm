"""
Frozen Schema Definitions for Knowledge Base and Timeline.

This module defines the stable, versioned schema for the entity-centric
knowledge base and multimodal timeline. The schema is designed for:

1. JSON serialization and export
2. Cross-run consistency
3. LLM prompt formatting
4. External tool integration

Schema Version: 1.0.0

Entity Schema:
- entity_id: Unique identifier (from SAM tracking)
- concept_label: Semantic label (e.g., "dragon", "player")
- category: High-level category (PLAYER, ENEMY, NPC, etc.)
- state_history: Time-series of position/bbox/attributes
- relationships: Edges to other entities

Timeline Schema:
- events: Timestamped multimodal events
- segments: Grouped event windows with summaries
- modality_indices: Fast lookup by modality type
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Schema version for compatibility checking
SCHEMA_VERSION = "1.0.0"


class SchemaVersion(Enum):
    """Supported schema versions."""
    V1_0_0 = "1.0.0"


# ============================================================================
# Entity Schema
# ============================================================================


class EntityCategorySchema(str, Enum):
    """Standardized entity categories."""

    PLAYER = "player"
    ENEMY = "enemy"
    BOSS = "boss"
    NPC = "npc"
    ITEM = "item"
    WEAPON = "weapon"
    PROJECTILE = "projectile"
    UI_ELEMENT = "ui_element"
    ENVIRONMENT = "environment"
    EFFECT = "effect"
    UNKNOWN = "unknown"


class RelationTypeSchema(str, Enum):
    """Standardized relationship types."""

    # Spatial
    NEAR = "near"
    CONTAINS = "contains"
    ABOVE = "above"
    BELOW = "below"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"

    # Interaction
    ATTACKS = "attacks"
    DAMAGES = "damages"
    HEALS = "heals"
    COLLIDES_WITH = "collides_with"
    FOLLOWS = "follows"
    TARGETS = "targets"
    PICKS_UP = "picks_up"
    USES = "uses"

    # State
    TRANSFORMS_INTO = "transforms_into"
    SPAWNS = "spawns"
    DESTROYS = "destroys"
    TRIGGERS = "triggers"


@dataclass
class EntityStateSchema:
    """
    Schema for a single entity state observation.

    Captures the entity's state at a specific point in time.
    """

    timestamp: float  # Seconds from video start
    position_x: Optional[float] = None  # Center X coordinate
    position_y: Optional[float] = None  # Center Y coordinate
    bbox_x1: Optional[float] = None  # Bounding box left
    bbox_y1: Optional[float] = None  # Bounding box top
    bbox_x2: Optional[float] = None  # Bounding box right
    bbox_y2: Optional[float] = None  # Bounding box bottom
    visible: bool = True
    confidence: float = 1.0
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        result = {
            "timestamp": self.timestamp,
            "visible": self.visible,
            "confidence": self.confidence,
        }
        if self.position_x is not None:
            result["position"] = {
                "x": self.position_x,
                "y": self.position_y,
            }
        if self.bbox_x1 is not None:
            result["bbox"] = {
                "x1": self.bbox_x1,
                "y1": self.bbox_y1,
                "x2": self.bbox_x2,
                "y2": self.bbox_y2,
            }
        if self.attributes:
            result["attributes"] = self.attributes
        return result


@dataclass
class RelationshipSchema:
    """
    Schema for a relationship between two entities.

    Captures causal or spatial relationships with temporal scope.
    """

    source_id: str
    target_id: str
    relation_type: str  # RelationTypeSchema value
    start_time: float
    end_time: Optional[float] = None  # None = ongoing
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        result = {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "start_time": self.start_time,
            "confidence": self.confidence,
            "is_active": self.end_time is None,
        }
        if self.end_time is not None:
            result["end_time"] = self.end_time
            result["duration"] = self.end_time - self.start_time
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class EntitySchema:
    """
    Schema for a tracked entity in the knowledge base.

    Represents a persistent entity across multiple frames with
    full state history and relationship tracking.
    """

    entity_id: str
    concept_label: str
    category: str  # EntityCategorySchema value
    first_seen: float
    last_seen: float
    is_active: bool = True
    state_history: list[EntityStateSchema] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "entity_id": self.entity_id,
            "concept_label": self.concept_label,
            "category": self.category,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "is_active": self.is_active,
            "duration": self.last_seen - self.first_seen,
            "state_count": len(self.state_history),
            "state_history": [s.to_dict() for s in self.state_history],
            "attributes": self.attributes,
        }


# ============================================================================
# Timeline Event Schema
# ============================================================================


class ModalityTypeSchema(str, Enum):
    """Standardized modality types for timeline events."""

    VISUAL = "visual"  # SAM/SigLIP detections
    TEMPORAL = "temporal"  # VideoMAE/HiCo events
    AUDIO = "audio"  # Sound effects, music
    SPEECH = "speech"  # Whisper transcriptions
    OCR = "ocr"  # On-screen text
    SYSTEM = "system"  # Game state changes


class EventPrioritySchema(str, Enum):
    """Event priority levels."""

    CRITICAL = "critical"  # Boss spawns, deaths, major events
    HIGH = "high"  # Combat, significant actions
    MEDIUM = "medium"  # Item pickups, dialogue
    LOW = "low"  # Ambient events
    DEBUG = "debug"  # Internal events


@dataclass
class TimelineEventSchema:
    """
    Schema for a single timeline event.

    Captures a timestamped event from any modality.
    """

    timestamp: float
    modality: str  # ModalityTypeSchema value
    description: str
    priority: str = "medium"  # EventPrioritySchema value
    duration: Optional[float] = None
    entity_id: Optional[str] = None  # Associated entity
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        result = {
            "timestamp": self.timestamp,
            "modality": self.modality,
            "description": self.description,
            "priority": self.priority,
            "confidence": self.confidence,
        }
        if self.duration is not None:
            result["duration"] = self.duration
            result["end_time"] = self.timestamp + self.duration
        if self.entity_id:
            result["entity_id"] = self.entity_id
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class TimelineSegmentSchema:
    """
    Schema for a segment of the timeline.

    Groups events into logical segments with summaries.
    """

    start_time: float
    end_time: float
    events: list[TimelineEventSchema] = field(default_factory=list)
    summary: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.end_time - self.start_time,
            "event_count": len(self.events),
            "events": [e.to_dict() for e in self.events],
            "summary": self.summary,
        }


# ============================================================================
# Full Export Schema
# ============================================================================


@dataclass
class KnowledgeBaseExport:
    """
    Complete knowledge base export schema.

    Contains all entities, relationships, and metadata for
    JSON serialization.
    """

    schema_version: str = SCHEMA_VERSION
    export_timestamp: str = ""
    video_source: Optional[str] = None
    video_duration: Optional[float] = None
    entities: list[EntitySchema] = field(default_factory=list)
    relationships: list[RelationshipSchema] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.export_timestamp:
            self.export_timestamp = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "schema_version": self.schema_version,
            "export_timestamp": self.export_timestamp,
            "video_source": self.video_source,
            "video_duration": self.video_duration,
            "entity_count": len(self.entities),
            "relationship_count": len(self.relationships),
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "statistics": self.statistics,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())
        logger.info(f"Knowledge base exported to {path}")


@dataclass
class TimelineExport:
    """
    Complete timeline export schema.

    Contains all events, segments, and metadata for
    JSON serialization.
    """

    schema_version: str = SCHEMA_VERSION
    export_timestamp: str = ""
    video_source: Optional[str] = None
    video_duration: Optional[float] = None
    events: list[TimelineEventSchema] = field(default_factory=list)
    segments: list[TimelineSegmentSchema] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.export_timestamp:
            self.export_timestamp = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "schema_version": self.schema_version,
            "export_timestamp": self.export_timestamp,
            "video_source": self.video_source,
            "video_duration": self.video_duration,
            "event_count": len(self.events),
            "segment_count": len(self.segments),
            "events": [e.to_dict() for e in self.events],
            "segments": [s.to_dict() for s in self.segments],
            "statistics": self.statistics,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())
        logger.info(f"Timeline exported to {path}")


@dataclass
class FullSessionExport:
    """
    Complete session export combining KB and timeline.

    This is the recommended export format for analysis runs.
    """

    schema_version: str = SCHEMA_VERSION
    export_timestamp: str = ""
    video_source: Optional[str] = None
    video_duration: Optional[float] = None
    knowledge_base: KnowledgeBaseExport = field(default_factory=KnowledgeBaseExport)
    timeline: TimelineExport = field(default_factory=TimelineExport)
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.export_timestamp:
            self.export_timestamp = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "schema_version": self.schema_version,
            "export_timestamp": self.export_timestamp,
            "video_source": self.video_source,
            "video_duration": self.video_duration,
            "knowledge_base": self.knowledge_base.to_dict(),
            "timeline": self.timeline.to_dict(),
            "config": self.config,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())
        logger.info(f"Full session exported to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "FullSessionExport":
        """Load from JSON file."""
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)

        # Version check
        if data.get("schema_version") != SCHEMA_VERSION:
            logger.warning(
                f"Schema version mismatch: file={data.get('schema_version')}, "
                f"current={SCHEMA_VERSION}"
            )

        # Reconstruct objects (simplified - full implementation would recursively rebuild)
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            export_timestamp=data.get("export_timestamp", ""),
            video_source=data.get("video_source"),
            video_duration=data.get("video_duration"),
            config=data.get("config", {}),
        )


# ============================================================================
# Conversion Utilities
# ============================================================================


def convert_kb_to_schema(kb: Any) -> KnowledgeBaseExport:
    """
    Convert a KnowledgeBaseBuilder instance to export schema.

    Args:
        kb: KnowledgeBaseBuilder instance

    Returns:
        KnowledgeBaseExport ready for JSON serialization
    """
    entities = []
    relationships = []

    # Convert entities
    for entity_id, entity in kb._entities.items():
        state_history = []
        for state in entity.state_history:
            state_schema = EntityStateSchema(
                timestamp=state.timestamp,
                position_x=state.position[0] if state.position else None,
                position_y=state.position[1] if state.position else None,
                bbox_x1=state.bbox[0] if state.bbox else None,
                bbox_y1=state.bbox[1] if state.bbox else None,
                bbox_x2=state.bbox[2] if state.bbox else None,
                bbox_y2=state.bbox[3] if state.bbox else None,
                visible=state.visible,
                attributes=state.attributes,
            )
            state_history.append(state_schema)

        entity_schema = EntitySchema(
            entity_id=entity.entity_id,
            concept_label=entity.concept_label,
            category=entity.category.value,
            first_seen=entity.first_seen,
            last_seen=entity.last_seen,
            is_active=entity.is_active,
            state_history=state_history,
            attributes=entity.attributes,
        )
        entities.append(entity_schema)

    # Convert relationships
    for edge in kb._relationships:
        rel_schema = RelationshipSchema(
            source_id=edge.source_id,
            target_id=edge.target_id,
            relation_type=edge.relation_type.value,
            start_time=edge.start_time,
            end_time=edge.end_time,
            confidence=edge.confidence,
            metadata=edge.metadata,
        )
        relationships.append(rel_schema)

    return KnowledgeBaseExport(
        entities=entities,
        relationships=relationships,
        statistics=kb.get_statistics(),
    )


def convert_timeline_to_schema(timeline: Any) -> TimelineExport:
    """
    Convert a TimelineIndexer instance to export schema.

    Args:
        timeline: TimelineIndexer instance

    Returns:
        TimelineExport ready for JSON serialization
    """
    events = []

    # Convert events
    for event in timeline._events:
        event_schema = TimelineEventSchema(
            timestamp=event.timestamp,
            modality=event.modality.value,
            description=event.description,
            priority=event.priority.value,
            duration=event.duration,
            entity_id=event.entity_id,
            confidence=event.confidence,
            metadata=event.metadata,
        )
        events.append(event_schema)

    return TimelineExport(
        events=events,
        statistics=timeline.get_statistics(),
    )


# ============================================================================
# Schema Documentation
# ============================================================================


def get_schema_documentation() -> str:
    """
    Get human-readable schema documentation.

    Returns:
        Formatted documentation string
    """
    doc = f"""
================================================================================
GAMEPLAY VISION LLM - Schema Documentation
Version: {SCHEMA_VERSION}
================================================================================

ENTITY CATEGORIES:
{chr(10).join(f'  - {c.value}: {c.name}' for c in EntityCategorySchema)}

RELATIONSHIP TYPES:
{chr(10).join(f'  - {r.value}: {r.name}' for r in RelationTypeSchema)}

MODALITY TYPES:
{chr(10).join(f'  - {m.value}: {m.name}' for m in ModalityTypeSchema)}

EVENT PRIORITIES:
{chr(10).join(f'  - {p.value}: {p.name}' for p in EventPrioritySchema)}

--------------------------------------------------------------------------------
ENTITY STATE SCHEMA
--------------------------------------------------------------------------------
Fields:
  - timestamp (float): Seconds from video start
  - position_x (float, optional): Center X coordinate
  - position_y (float, optional): Center Y coordinate
  - bbox_x1, bbox_y1, bbox_x2, bbox_y2 (float, optional): Bounding box
  - visible (bool): Whether entity is visible
  - confidence (float): Detection confidence [0-1]
  - attributes (dict): Additional key-value attributes

--------------------------------------------------------------------------------
TIMELINE EVENT SCHEMA
--------------------------------------------------------------------------------
Fields:
  - timestamp (float): Event start time in seconds
  - modality (str): One of {[m.value for m in ModalityTypeSchema]}
  - description (str): Human-readable event description
  - priority (str): One of {[p.value for p in EventPrioritySchema]}
  - duration (float, optional): Event duration in seconds
  - entity_id (str, optional): Associated entity ID
  - confidence (float): Event confidence [0-1]
  - metadata (dict): Additional key-value metadata

--------------------------------------------------------------------------------
EXPORT FORMAT
--------------------------------------------------------------------------------
Full session exports include:
  - knowledge_base: All entities and relationships
  - timeline: All events and segments
  - config: System configuration used
  - video_source: Source video path/URL
  - video_duration: Total video duration

Example export command:
  session.save("outputs/session_export.json")

================================================================================
"""
    return doc


if __name__ == "__main__":
    print(get_schema_documentation())
