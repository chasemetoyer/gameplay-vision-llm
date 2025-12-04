"""
Knowledge Base Builder Module.

This module builds an entity-centric knowledge graph from SAM 3 tracking
data and associated events. Key capabilities:
1. Entity registration with persistent IDs
2. Relationship encoding (attacks, collisions, interactions)
3. State change tracking over time
4. Graph/table export for LLM prompting

References:
- [A: 16] Entity-centric knowledge base
- [A: 25, A: 22] Explicit relationships for causal reasoning
- [A: 29] Structured input for LLM
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RelationType(Enum):
    """Types of relationships between entities."""

    # Spatial relationships
    NEAR = "near"  # Entities in proximity
    CONTAINS = "contains"  # One entity inside another
    ABOVE = "above"
    BELOW = "below"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"

    # Interaction relationships
    ATTACKS = "attacks"
    HEALS = "heals"
    COLLIDES_WITH = "collides_with"
    FOLLOWS = "follows"
    TARGETS = "targets"

    # State relationships
    TRANSFORMS_INTO = "transforms_into"
    SPAWNS = "spawns"
    DESTROYS = "destroys"


class EntityCategory(Enum):
    """High-level categories for entities."""

    PLAYER = "player"
    ENEMY = "enemy"
    NPC = "npc"
    ITEM = "item"
    PROJECTILE = "projectile"
    UI_ELEMENT = "ui_element"
    ENVIRONMENT = "environment"
    EFFECT = "effect"
    UNKNOWN = "unknown"


@dataclass
class EntityState:
    """A snapshot of an entity's state at a point in time."""

    timestamp: float
    position: Optional[tuple[float, float]] = None  # Center (x, y)
    bbox: Optional[tuple[float, float, float, float]] = None  # x1, y1, x2, y2
    visible: bool = True
    attributes: dict = field(default_factory=dict)

    def distance_to(self, other: "EntityState") -> Optional[float]:
        """Calculate distance to another entity state."""
        if self.position is None or other.position is None:
            return None
        dx = self.position[0] - other.position[0]
        dy = self.position[1] - other.position[1]
        return (dx**2 + dy**2) ** 0.5


@dataclass
class EntityNode:
    """
    A node in the knowledge graph representing an entity.
    
    Tracks entity identity, category, and state history.
    """

    entity_id: str
    concept_label: str
    category: EntityCategory = EntityCategory.UNKNOWN
    first_seen: float = 0.0
    last_seen: float = 0.0
    is_active: bool = True

    # State history
    state_history: list[EntityState] = field(default_factory=list)

    # Semantic attributes
    attributes: dict = field(default_factory=dict)

    def add_state(self, state: EntityState) -> None:
        """Add a new state observation."""
        self.state_history.append(state)
        self.last_seen = max(self.last_seen, state.timestamp)

    def get_state_at(self, timestamp: float) -> Optional[EntityState]:
        """Get the closest state to a timestamp."""
        if not self.state_history:
            return None

        closest = min(
            self.state_history,
            key=lambda s: abs(s.timestamp - timestamp),
        )
        return closest

    def get_latest_state(self) -> Optional[EntityState]:
        """Get the most recent state."""
        if not self.state_history:
            return None
        return max(self.state_history, key=lambda s: s.timestamp)

    def get_attribute_changes(self, attr_name: str) -> list[tuple[float, Any, Any]]:
        """
        Get history of changes to a specific attribute.
        
        Returns:
            List of (timestamp, old_value, new_value) tuples
        """
        changes = []
        prev_value = None

        for state in sorted(self.state_history, key=lambda s: s.timestamp):
            curr_value = state.attributes.get(attr_name)
            if curr_value != prev_value:
                changes.append((state.timestamp, prev_value, curr_value))
                prev_value = curr_value

        return changes


@dataclass
class RelationshipEdge:
    """
    An edge in the knowledge graph representing a relationship.
    
    Captures relationships between entities with temporal scope.
    """

    source_id: str
    target_id: str
    relation_type: RelationType
    start_time: float
    end_time: Optional[float] = None  # None = ongoing
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.end_time is None

    @property
    def duration(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    def format_description(self) -> str:
        """Format as natural language description."""
        return f"{self.source_id} {self.relation_type.value} {self.target_id}"


@dataclass
class KnowledgeBaseConfig:
    """Configuration for Knowledge Base Builder."""

    # Entity tracking
    proximity_threshold: float = 50.0  # Pixels for NEAR relationship
    collision_iou_threshold: float = 0.3  # IoU for collision detection

    # Relationship inference
    infer_spatial_relations: bool = True
    infer_interactions: bool = True
    relation_timeout: float = 5.0  # Seconds before ending inactive relations

    # Export settings
    max_history_per_entity: int = 100  # State history limit
    include_inactive_entities: bool = False


class RelationshipInferrer:
    """Infers relationships between entities based on observations."""

    def __init__(self, config: KnowledgeBaseConfig):
        self.config = config

    def infer_spatial_relations(
        self,
        entity1: EntityNode,
        entity2: EntityNode,
        timestamp: float,
    ) -> list[RelationType]:
        """Infer spatial relationships between two entities."""
        relations = []

        state1 = entity1.get_state_at(timestamp)
        state2 = entity2.get_state_at(timestamp)

        if state1 is None or state2 is None:
            return relations

        # Check proximity
        distance = state1.distance_to(state2)
        if distance is not None and distance < self.config.proximity_threshold:
            relations.append(RelationType.NEAR)

        # Check relative positions
        if state1.position and state2.position:
            dx = state2.position[0] - state1.position[0]
            dy = state2.position[1] - state1.position[1]

            if abs(dx) > abs(dy):
                if dx > 0:
                    relations.append(RelationType.LEFT_OF)
                else:
                    relations.append(RelationType.RIGHT_OF)
            else:
                if dy > 0:
                    relations.append(RelationType.ABOVE)
                else:
                    relations.append(RelationType.BELOW)

        # Check containment
        if state1.bbox and state2.bbox:
            if self._is_contained(state2.bbox, state1.bbox):
                relations.append(RelationType.CONTAINS)

        return relations

    def infer_collision(
        self,
        entity1: EntityNode,
        entity2: EntityNode,
        timestamp: float,
    ) -> bool:
        """Check if two entities are colliding."""
        state1 = entity1.get_state_at(timestamp)
        state2 = entity2.get_state_at(timestamp)

        if state1 is None or state2 is None:
            return False
        if state1.bbox is None or state2.bbox is None:
            return False

        iou = self._compute_iou(state1.bbox, state2.bbox)
        return iou >= self.config.collision_iou_threshold

    def _is_contained(
        self,
        inner: tuple[float, float, float, float],
        outer: tuple[float, float, float, float],
    ) -> bool:
        """Check if inner bbox is contained in outer."""
        return (
            inner[0] >= outer[0]
            and inner[1] >= outer[1]
            and inner[2] <= outer[2]
            and inner[3] <= outer[3]
        )

    def _compute_iou(
        self,
        box1: tuple[float, float, float, float],
        box2: tuple[float, float, float, float],
    ) -> float:
        """Compute Intersection over Union."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


class KnowledgeBaseBuilder:
    """
    Main interface for building entity-centric knowledge graphs.
    
    Creates structured representations of entities and their
    relationships that can be used for LLM causal reasoning.
    
    Example:
        >>> kb = KnowledgeBaseBuilder()
        >>> 
        >>> # Register entities from SAM tracking
        >>> for entity in sam_entities:
        ...     kb.register_entity(
        ...         entity_id=entity.entity_id,
        ...         concept_label=entity.concept_label,
        ...         category=EntityCategory.ENEMY
        ...     )
        >>> 
        >>> # Add relationships
        >>> kb.add_relationship(
        ...     source_id="player_001",
        ...     target_id="boss_dragon_001",
        ...     relation_type=RelationType.ATTACKS,
        ...     timestamp=83.0
        ... )
        >>> 
        >>> # Query entity history
        >>> history = kb.query_entity_history("boss_dragon_001")
        >>> 
        >>> # Export for LLM
        >>> prompt_text = kb.export_for_llm()
    """

    def __init__(
        self,
        config: Optional[KnowledgeBaseConfig] = None,
    ):
        """
        Initialize the Knowledge Base Builder.

        Args:
            config: Configuration. Uses defaults if not provided.
        """
        self.config = config or KnowledgeBaseConfig()
        self.inferrer = RelationshipInferrer(self.config)

        # Graph storage
        self._entities: dict[str, EntityNode] = {}
        self._relationships: list[RelationshipEdge] = []
        self._relationships_by_source: dict[str, list[RelationshipEdge]] = {}
        self._relationships_by_target: dict[str, list[RelationshipEdge]] = {}

        logger.info("KnowledgeBaseBuilder initialized")

    def register_entity(
        self,
        entity_id: str,
        concept_label: str,
        category: EntityCategory = EntityCategory.UNKNOWN,
        timestamp: float = 0.0,
        initial_state: Optional[EntityState] = None,
        attributes: Optional[dict] = None,
    ) -> EntityNode:
        """
        Register a new entity in the knowledge base.
        
        Args:
            entity_id: Unique entity identifier (from SAM)
            concept_label: Semantic label
            category: Entity category
            timestamp: First seen timestamp
            initial_state: Initial state observation
            attributes: Initial attributes
            
        Returns:
            The created or updated EntityNode
        """
        if entity_id in self._entities:
            # Update existing entity
            entity = self._entities[entity_id]
            entity.last_seen = max(entity.last_seen, timestamp)
            if initial_state:
                entity.add_state(initial_state)
            return entity

        entity = EntityNode(
            entity_id=entity_id,
            concept_label=concept_label,
            category=category,
            first_seen=timestamp,
            last_seen=timestamp,
            attributes=attributes or {},
        )

        if initial_state:
            entity.add_state(initial_state)

        self._entities[entity_id] = entity
        self._relationships_by_source[entity_id] = []
        self._relationships_by_target[entity_id] = []

        logger.debug(f"Registered entity: {entity_id} ({concept_label})")
        return entity

    def update_entity_state(
        self,
        entity_id: str,
        timestamp: float,
        position: Optional[tuple[float, float]] = None,
        bbox: Optional[tuple[float, float, float, float]] = None,
        visible: bool = True,
        attributes: Optional[dict] = None,
    ) -> None:
        """
        Update an entity's state at a timestamp.
        
        Args:
            entity_id: Entity to update
            timestamp: Observation timestamp
            position: Center position (x, y)
            bbox: Bounding box (x1, y1, x2, y2)
            visible: Whether entity is visible
            attributes: State attributes
        """
        entity = self._entities.get(entity_id)
        if entity is None:
            logger.warning(f"Entity not found: {entity_id}")
            return

        state = EntityState(
            timestamp=timestamp,
            position=position,
            bbox=bbox,
            visible=visible,
            attributes=attributes or {},
        )

        entity.add_state(state)

        # Trim history if needed
        if len(entity.state_history) > self.config.max_history_per_entity:
            entity.state_history = entity.state_history[-self.config.max_history_per_entity :]

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        timestamp: float,
        end_time: Optional[float] = None,
        confidence: float = 1.0,
        metadata: Optional[dict] = None,
    ) -> RelationshipEdge:
        """
        Add a relationship between two entities.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relation_type: Type of relationship
            timestamp: Start time of relationship
            end_time: End time (None = ongoing)
            confidence: Confidence score
            metadata: Additional metadata
            
        Returns:
            The created RelationshipEdge
        """
        edge = RelationshipEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            start_time=timestamp,
            end_time=end_time,
            confidence=confidence,
            metadata=metadata or {},
        )

        self._relationships.append(edge)

        if source_id in self._relationships_by_source:
            self._relationships_by_source[source_id].append(edge)
        if target_id in self._relationships_by_target:
            self._relationships_by_target[target_id].append(edge)

        logger.debug(f"Added relationship: {edge.format_description()}")
        return edge

    def end_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        end_time: float,
    ) -> bool:
        """
        End an active relationship.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relation_type: Type of relationship to end
            end_time: End timestamp
            
        Returns:
            True if relationship was found and ended
        """
        for edge in self._relationships_by_source.get(source_id, []):
            if (
                edge.target_id == target_id
                and edge.relation_type == relation_type
                and edge.is_active
            ):
                edge.end_time = end_time
                return True
        return False

    def infer_relationships_at(
        self,
        timestamp: float,
    ) -> list[RelationshipEdge]:
        """
        Infer relationships between all entities at a timestamp.
        
        Args:
            timestamp: Time to analyze
            
        Returns:
            List of inferred relationships
        """
        inferred = []
        entities = list(self._entities.values())

        for i, entity1 in enumerate(entities):
            for entity2 in entities[i + 1 :]:
                # Skip if either is inactive
                if not entity1.is_active or not entity2.is_active:
                    continue

                # Infer spatial relations
                if self.config.infer_spatial_relations:
                    relations = self.inferrer.infer_spatial_relations(
                        entity1, entity2, timestamp
                    )
                    for rel_type in relations:
                        edge = self.add_relationship(
                            source_id=entity1.entity_id,
                            target_id=entity2.entity_id,
                            relation_type=rel_type,
                            timestamp=timestamp,
                            confidence=0.8,
                        )
                        inferred.append(edge)

                # Check for collisions
                if self.config.infer_interactions:
                    if self.inferrer.infer_collision(entity1, entity2, timestamp):
                        edge = self.add_relationship(
                            source_id=entity1.entity_id,
                            target_id=entity2.entity_id,
                            relation_type=RelationType.COLLIDES_WITH,
                            timestamp=timestamp,
                            confidence=0.9,
                        )
                        inferred.append(edge)

        return inferred

    def get_entity(self, entity_id: str) -> Optional[EntityNode]:
        """Get an entity by ID."""
        return self._entities.get(entity_id)

    def get_entities_by_category(
        self,
        category: EntityCategory,
    ) -> list[EntityNode]:
        """Get all entities of a category."""
        return [e for e in self._entities.values() if e.category == category]

    def query_entity_history(
        self,
        entity_id: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> list[EntityState]:
        """
        Query the state history of an entity.
        
        Args:
            entity_id: Entity to query
            start_time: Optional start filter
            end_time: Optional end filter
            
        Returns:
            List of EntityState objects
        """
        entity = self._entities.get(entity_id)
        if entity is None:
            return []

        states = entity.state_history

        if start_time is not None:
            states = [s for s in states if s.timestamp >= start_time]
        if end_time is not None:
            states = [s for s in states if s.timestamp <= end_time]

        return sorted(states, key=lambda s: s.timestamp)

    def get_relationships_for_entity(
        self,
        entity_id: str,
        as_source: bool = True,
        as_target: bool = True,
    ) -> list[RelationshipEdge]:
        """Get all relationships involving an entity."""
        edges = []

        if as_source:
            edges.extend(self._relationships_by_source.get(entity_id, []))
        if as_target:
            edges.extend(self._relationships_by_target.get(entity_id, []))

        return edges

    def get_active_relationships(
        self,
        timestamp: Optional[float] = None,
    ) -> list[RelationshipEdge]:
        """Get all active relationships at a timestamp."""
        return [
            e for e in self._relationships
            if e.is_active or (timestamp and e.end_time and e.end_time > timestamp)
        ]

    def get_entity_relationship_summary(
        self,
        entity_id: str,
    ) -> str:
        """Get a natural language summary of entity relationships."""
        entity = self._entities.get(entity_id)
        if entity is None:
            return f"Entity {entity_id} not found."

        lines = [f"Entity: {entity.entity_id} ({entity.concept_label})"]
        lines.append(f"Category: {entity.category.value}")
        lines.append(f"Active: {entity.is_active}")
        lines.append(f"Seen: {entity.first_seen:.1f}s - {entity.last_seen:.1f}s")

        # Get relationships
        outgoing = self._relationships_by_source.get(entity_id, [])
        incoming = self._relationships_by_target.get(entity_id, [])

        if outgoing:
            lines.append("Outgoing relationships:")
            for edge in outgoing[:5]:  # Limit
                lines.append(f"  - {edge.format_description()}")

        if incoming:
            lines.append("Incoming relationships:")
            for edge in incoming[:5]:
                lines.append(f"  - {edge.format_description()}")

        return "\n".join(lines)

    def export_as_table(self) -> list[dict]:
        """
        Export knowledge base as a table format.
        
        Returns:
            List of entity dictionaries
        """
        rows = []

        for entity in self._entities.values():
            if not self.config.include_inactive_entities and not entity.is_active:
                continue

            latest = entity.get_latest_state()
            outgoing = len(self._relationships_by_source.get(entity.entity_id, []))
            incoming = len(self._relationships_by_target.get(entity.entity_id, []))

            rows.append({
                "entity_id": entity.entity_id,
                "label": entity.concept_label,
                "category": entity.category.value,
                "first_seen": entity.first_seen,
                "last_seen": entity.last_seen,
                "is_active": entity.is_active,
                "position": latest.position if latest else None,
                "outgoing_relations": outgoing,
                "incoming_relations": incoming,
                "attributes": entity.attributes,
            })

        return rows

    def export_as_graph(self) -> dict:
        """
        Export as graph structure for visualization or processing.
        
        Returns:
            Dictionary with 'nodes' and 'edges' lists
        """
        nodes = []
        edges = []

        for entity in self._entities.values():
            if not self.config.include_inactive_entities and not entity.is_active:
                continue

            nodes.append({
                "id": entity.entity_id,
                "label": entity.concept_label,
                "category": entity.category.value,
            })

        for edge in self._relationships:
            edges.append({
                "source": edge.source_id,
                "target": edge.target_id,
                "type": edge.relation_type.value,
                "start_time": edge.start_time,
                "end_time": edge.end_time,
            })

        return {"nodes": nodes, "edges": edges}

    def export_for_llm(
        self,
        max_entities: int = 20,
        max_relationships: int = 30,
    ) -> str:
        """
        Export knowledge base as text for LLM prompting.
        
        Creates a structured text representation that can be
        included in LLM context for reasoning.
        
        Args:
            max_entities: Maximum entities to include
            max_relationships: Maximum relationships to include
            
        Returns:
            Formatted text for LLM prompt
        """
        lines = ["## Entity Knowledge Base", ""]

        # Sort entities by last seen (most recent first)
        entities = sorted(
            self._entities.values(),
            key=lambda e: e.last_seen,
            reverse=True,
        )

        # Filter inactive if configured
        if not self.config.include_inactive_entities:
            entities = [e for e in entities if e.is_active]

        entities = entities[:max_entities]

        lines.append("### Entities")
        for entity in entities:
            latest = entity.get_latest_state()
            pos_str = ""
            if latest and latest.position:
                pos_str = f" at ({latest.position[0]:.0f}, {latest.position[1]:.0f})"

            lines.append(
                f"- **{entity.entity_id}**: {entity.concept_label} "
                f"({entity.category.value}){pos_str}"
            )

        lines.append("")
        lines.append("### Relationships")

        # Get recent relationships
        recent_rels = sorted(
            self._relationships,
            key=lambda e: e.start_time,
            reverse=True,
        )[:max_relationships]

        for edge in recent_rels:
            time_str = f"[{edge.start_time:.1f}s]"
            status = "ongoing" if edge.is_active else f"ended {edge.end_time:.1f}s"
            lines.append(
                f"- {time_str} {edge.source_id} {edge.relation_type.value} "
                f"{edge.target_id} ({status})"
            )

        return "\n".join(lines)

    def get_statistics(self) -> dict:
        """Get knowledge base statistics."""
        active_entities = sum(1 for e in self._entities.values() if e.is_active)
        active_relations = sum(1 for r in self._relationships if r.is_active)

        category_counts = {}
        for entity in self._entities.values():
            cat = entity.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

        relation_counts = {}
        for rel in self._relationships:
            rtype = rel.relation_type.value
            relation_counts[rtype] = relation_counts.get(rtype, 0) + 1

        return {
            "total_entities": len(self._entities),
            "active_entities": active_entities,
            "total_relationships": len(self._relationships),
            "active_relationships": active_relations,
            "entities_by_category": category_counts,
            "relationships_by_type": relation_counts,
        }

    def clear(self) -> None:
        """Clear all data from knowledge base."""
        self._entities.clear()
        self._relationships.clear()
        self._relationships_by_source.clear()
        self._relationships_by_target.clear()
        logger.info("Knowledge base cleared")


def create_knowledge_base(
    proximity_threshold: float = 50.0,
    infer_relations: bool = True,
) -> KnowledgeBaseBuilder:
    """
    Factory function to create a Knowledge Base Builder.

    Args:
        proximity_threshold: Pixel threshold for NEAR relationships
        infer_relations: Enable automatic relationship inference

    Returns:
        Configured KnowledgeBaseBuilder instance
    """
    config = KnowledgeBaseConfig(
        proximity_threshold=proximity_threshold,
        infer_spatial_relations=infer_relations,
        infer_interactions=infer_relations,
    )
    return KnowledgeBaseBuilder(config=config)
