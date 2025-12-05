import unittest
import logging
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from fusion_indexing.timeline_indexer import (
    TimelineIndexer, TimelineConfig, ModalityType, EventPriority
)
from fusion_indexing.knowledge_base_builder import (
    KnowledgeBaseBuilder, KnowledgeBaseConfig, EntityCategory
)
# We mock the input objects since we don't want to rely on real models here
from dataclasses import dataclass

@dataclass
class MockTrackedEntity:
    entity_id: str
    concept_label: str
    frame_masks: dict
    
@dataclass
class MockMask:
    confidence: float
    bbox: object
    
@dataclass
class MockBbox:
    def to_xyxy(self): return [10, 10, 50, 50]
    def center(self): return (30, 30)
    def area(self): return 1600

class TestFusionIntegration(unittest.TestCase):
    """
    Integration tests for Fusion & Indexing Phase.
    Verifies that TimelineIndexer and KnowledgeBaseBuilder correctly 
    consume and structure data from perception modules.
    """
    
    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.timeline_indexer = TimelineIndexer(TimelineConfig())
        self.kb_builder = KnowledgeBaseBuilder(KnowledgeBaseConfig())

    def test_timeline_alignment(self):
        """Test alignment of multi-modal events on timeline."""
        print("\n[Test] Timeline Alignment")
        
        # 1. Add "Visual" Event (SAM)
        self.timeline_indexer.add_event(
            timestamp=1.5,
            modality=ModalityType.VISUAL,
            description="Player detected",
            entity_id="player_01",
            confidence=0.95
        )
        
        # 2. Add "Speech" Event (Qwen-Audio/ASR)
        self.timeline_indexer.add_event(
            timestamp=1.6,
            modality=ModalityType.SPEECH,
            description="Let's do this!",
            confidence=0.98
        )
        
        # 3. Add "OCR" Event (PaddleOCR)
        self.timeline_indexer.add_event(
            timestamp=1.55,
            modality=ModalityType.OCR,
            description="HP: 100/100",
            metadata={"category": "health"}
        )
        
        # Verify chronological order
        events = self.timeline_indexer.query_range(0, 5)
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0].modality, ModalityType.VISUAL) # 1.5
        self.assertEqual(events[1].modality, ModalityType.OCR)    # 1.55
        self.assertEqual(events[2].modality, ModalityType.SPEECH) # 1.6
        
        # Verify transcript generation
        transcript = self.timeline_indexer.build_structured_transcript()
        print(f"Generated Transcript:\n{transcript}")
        self.assertIn("[00:01] Player detected", transcript)
        self.assertIn("HP: 100/100", transcript)

    def test_kb_entity_tracking(self):
        """Test entity state tracking in Knowledge Base."""
        print("\n[Test] KB Entity Tracking")
        
        # Simulate SAM tracking a boss across frames
        entity_id = "boss_01"
        concept = "Boss"
        
        # Frame 1: Boss appears
        mask1 = MockMask(0.9, MockBbox())
        entity1 = MockTrackedEntity(entity_id, concept, {0: mask1})
        
        self.kb_builder.update_from_tracking([entity1], frame_idx=0, timestamp=10.0)
        
        # Verify node creation
        node = self.kb_builder.get_entity(entity_id)
        self.assertIsNotNone(node)
        self.assertEqual(node.concept_label, concept)
        self.assertEqual(len(node.state_history), 1)
        
        # Frame 2: Boss moves (simulate with update)
        entity1.frame_masks[10] = mask1 # Add mask for frame 10
        self.kb_builder.update_from_tracking([entity1], frame_idx=10, timestamp=11.0)
        
        # Verify history update
        self.assertEqual(len(node.state_history), 2)
        print(f"Entity History: {len(node.state_history)} states")

    def test_timeline_kb_query(self):
        """Test retrieving interaction history."""
        print("\n[Test] Timeline + KB Query")
        
        # Register entity in KB
        self.kb_builder.register_entity("player_1", "Player", EntityCategory.PLAYER)
        
        # Add events to timeline linked to entity
        self.timeline_indexer.add_event(
            timestamp=5.0,
            modality=ModalityType.VISUAL,
            description="Player casts spell",
            entity_id="player_1"
        )
        self.timeline_indexer.add_event(
            timestamp=5.5,
            modality=ModalityType.AUDIO,
            description="Explosion sound",
            entity_id="player_1" # Inferred link
        )
        
        # Query timeline by entity
        player_events = self.timeline_indexer.query_by_entity("player_1")
        self.assertEqual(len(player_events), 2)
        self.assertEqual(player_events[0].description, "Player casts spell")

if __name__ == "__main__":
    unittest.main()
