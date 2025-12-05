"""
Unit tests for Qwen3-VL Reasoning Core.

Tests the retrieval logic and prompt construction without loading the full model.
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from agent_core.qwen_reasoning_core import (
    ReasoningCoreConfig,
    TimelineRetriever,
    VisualInputProcessor,
    QwenVLCore,
)


class MockTimelineEvent:
    """Mock event for testing."""

    def __init__(self, timestamp: float, description: str):
        self.timestamp = timestamp
        self.description = description

    def format_compact(self) -> str:
        mins = int(self.timestamp // 60)
        secs = int(self.timestamp % 60)
        return f"[{mins:02d}:{secs:02d}] {self.description}"


class TestTimelineRetriever(unittest.TestCase):
    """Tests for TimelineRetriever."""

    def setUp(self):
        self.config = ReasoningCoreConfig()
        self.retriever = TimelineRetriever(self.config)

    def test_parse_timestamp_mm_ss(self):
        """Test parsing MM:SS format."""
        ts = self.retriever.parse_timestamp("What happened at 5:30?")
        self.assertEqual(ts, 330.0)  # 5*60 + 30

    def test_parse_timestamp_hh_mm_ss(self):
        """Test parsing HH:MM:SS format."""
        ts = self.retriever.parse_timestamp("Show me 1:05:30")
        self.assertEqual(ts, 3930.0)  # 1*3600 + 5*60 + 30

    def test_parse_timestamp_seconds(self):
        """Test parsing 'at X seconds' format."""
        ts = self.retriever.parse_timestamp("What's at 45 seconds?")
        self.assertEqual(ts, 45.0)

    def test_parse_timestamp_none(self):
        """Test query with no timestamp."""
        ts = self.retriever.parse_timestamp("Why did the player die?")
        self.assertIsNone(ts)

    def test_retrieve_by_timestamp(self):
        """Test time-based retrieval."""
        # Setup mock events
        events = [
            MockTimelineEvent(10.0, "Player spawns"),
            MockTimelineEvent(30.0, "Enemy appears"),
            MockTimelineEvent(60.0, "Battle starts"),
            MockTimelineEvent(90.0, "Player dies"),
        ]
        self.retriever._timeline_events = events

        # Query around 30s with 20s window
        results = self.retriever.retrieve_by_timestamp(30.0, window=20.0)

        # Should get events at 10, 30 (not 60, 90)
        self.assertEqual(len(results), 2)
        descriptions = [e.description for e in results]
        self.assertIn("Player spawns", descriptions)
        self.assertIn("Enemy appears", descriptions)


class TestVisualInputProcessor(unittest.TestCase):
    """Tests for VisualInputProcessor."""

    def setUp(self):
        self.config = ReasoningCoreConfig()
        self.processor = VisualInputProcessor(self.config)

    def test_process_region_tokens(self):
        """Test region token formatting."""
        regions = [
            {"label": "player", "bbox": [10, 20, 100, 200], "confidence": 0.95},
            {"label": "enemy", "bbox": [200, 50, 300, 150], "confidence": 0.88},
        ]

        result = self.processor.process_region_tokens(regions)

        self.assertIn("Detected regions in frame:", result)
        self.assertIn("player", result)
        self.assertIn("enemy", result)
        self.assertIn("0.95", result)

    def test_process_region_tokens_empty(self):
        """Test empty region list."""
        result = self.processor.process_region_tokens([])
        self.assertEqual(result, "")


class TestQwenVLCore(unittest.TestCase):
    """Tests for QwenVLCore (without model loading)."""

    def setUp(self):
        self.config = ReasoningCoreConfig()
        self.core = QwenVLCore(self.config)

    def test_format_timeline_context(self):
        """Test timeline context formatting."""
        events = [
            MockTimelineEvent(60.0, "Player enters dungeon"),
            MockTimelineEvent(30.0, "Game starts"),  # Out of order
            MockTimelineEvent(90.0, "Boss appears"),
        ]

        result = self.core.format_timeline_context(events)

        # Should be sorted by timestamp
        lines = result.split("\n")
        self.assertTrue(lines[0].startswith("[00:30]"))
        self.assertTrue(lines[1].startswith("[01:00]"))
        self.assertTrue(lines[2].startswith("[01:30]"))

    def test_build_prompt_structure(self):
        """Test prompt message structure."""
        messages = self.core.build_prompt(
            query="What happened?",
            timeline_context="[00:30] Game started",
            current_frame=None,
        )

        # Should have system and user messages
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")

        # User content should have timeline and query
        user_content = messages[1]["content"]
        text_parts = [c for c in user_content if c.get("type") == "text"]
        full_text = " ".join(c["text"] for c in text_parts)

        self.assertIn("Timeline Context", full_text)
        self.assertIn("Game started", full_text)
        self.assertIn("What happened?", full_text)


class TestHybridRetrieval(unittest.TestCase):
    """Tests for hybrid retrieval logic."""

    def setUp(self):
        self.config = ReasoningCoreConfig()
        self.retriever = TimelineRetriever(self.config)

        # Setup mock events
        self.events = [
            MockTimelineEvent(30.0, "Player health drops to 50%"),
            MockTimelineEvent(60.0, "Enemy attacks player"),
            MockTimelineEvent(90.0, "Player uses healing potion"),
            MockTimelineEvent(120.0, "Player defeats enemy"),
        ]
        self.retriever._timeline_events = self.events

    def test_hybrid_with_timestamp(self):
        """Test hybrid retrieval with timestamp in query."""
        results = self.retriever.hybrid_retrieve("What happened at 1:00?")

        # Should use time-based retrieval
        # Window is 30s, so 60±30 = 30-90
        self.assertTrue(len(results) >= 2)

    def test_hybrid_without_timestamp(self):
        """Test hybrid retrieval without timestamp."""
        # Without embedder, returns empty for semantic
        results = self.retriever.hybrid_retrieve("Why did the player get hurt?")

        # Without sentence-transformers installed, returns empty
        # This is expected behavior - semantic disabled
        self.assertIsInstance(results, list)


# =============================================================================
# Section 9 Component Tests
# =============================================================================

class TestSpecialTokens(unittest.TestCase):
    """Tests for SpecialTokens class."""
    
    def test_format_timestamp(self):
        """Test timestamp token formatting."""
        from agent_core.qwen_reasoning_core import SpecialTokens
        
        self.assertEqual(SpecialTokens.format_timestamp(0), "<ts=00:00>")
        self.assertEqual(SpecialTokens.format_timestamp(65), "<ts=01:05>")
        self.assertEqual(SpecialTokens.format_timestamp(3661), "<ts=61:01>")
    
    def test_parse_timestamp(self):
        """Test timestamp token parsing."""
        from agent_core.qwen_reasoning_core import SpecialTokens
        
        self.assertEqual(SpecialTokens.parse_timestamp("<ts=00:00>"), 0)
        self.assertEqual(SpecialTokens.parse_timestamp("<ts=01:05>"), 65)
        self.assertIsNone(SpecialTokens.parse_timestamp("invalid"))


class TestTriggerDetector(unittest.TestCase):
    """Tests for TriggerDetector."""
    
    def setUp(self):
        self.config = ReasoningCoreConfig()
        self.config.trigger_concepts = ["player", "enemy", "boss"]
        self.config.trigger_confidence_threshold = 0.8
        
        from agent_core.qwen_reasoning_core import TriggerDetector
        self.detector = TriggerDetector(self.config)
    
    def test_visual_trigger_fires(self):
        """Test that visual trigger fires on matching concept."""
        detections = [
            {"label": "boss_monster", "confidence": 0.95},
            {"label": "tree", "confidence": 0.9},
        ]
        
        trigger = self.detector.check_visual_trigger(detections, 10.0)
        
        self.assertIsNotNone(trigger)
        self.assertEqual(trigger.trigger_type, "concept")
        self.assertEqual(trigger.source, "SAM3")
        self.assertEqual(trigger.details["concept"], "boss")
    
    def test_visual_trigger_low_confidence(self):
        """Test that low confidence doesn't trigger."""
        detections = [
            {"label": "boss", "confidence": 0.5},  # Below threshold
        ]
        
        trigger = self.detector.check_visual_trigger(detections, 10.0)
        self.assertIsNone(trigger)
    
    def test_audio_trigger_fires(self):
        """Test audio trigger detection."""
        audio_events = [
            {"event": "explosion_sound", "confidence": 0.9},
        ]
        self.config.audio_trigger_events = ["explosion"]
        
        trigger = self.detector.check_audio_trigger(audio_events, 10.0)
        
        self.assertIsNotNone(trigger)
        self.assertEqual(trigger.trigger_type, "audio")


class TestTemporalContextManager(unittest.TestCase):
    """Tests for TemporalContextManager."""
    
    def setUp(self):
        self.config = ReasoningCoreConfig()
        self.config.temporal_window_minutes = 1.0  # 60 seconds for testing
        
        from agent_core.qwen_reasoning_core import TemporalContextManager
        self.manager = TemporalContextManager(self.config)
    
    def test_add_and_get_context(self):
        """Test adding and retrieving context."""
        import torch
        
        # Add embeddings
        self.manager.add_context(10.0, torch.randn(128))
        self.manager.add_context(20.0, torch.randn(128))
        
        timestamps, embeddings = self.manager.get_context()
        
        self.assertEqual(len(timestamps), 2)
        self.assertEqual(embeddings.shape[0], 2)
    
    def test_window_pruning(self):
        """Test that old context is pruned."""
        import torch
        
        # Add embeddings
        self.manager.add_context(0.0, torch.randn(128))
        self.manager.add_context(30.0, torch.randn(128))
        self.manager.add_context(70.0, torch.randn(128))  # 70s window = 10-70
        
        timestamps, _ = self.manager.get_context()
        
        # Only embeddings within 60s window from latest should remain
        self.assertEqual(len(timestamps), 2)  # 30.0 and 70.0


class TestPerceptionReasoningLoop(unittest.TestCase):
    """Tests for PerceptionReasoningLoop."""
    
    def setUp(self):
        from agent_core.qwen_reasoning_core import PerceptionReasoningLoop
        self.loop = PerceptionReasoningLoop()
    
    def test_start_stop(self):
        """Test loop start/stop."""
        self.assertFalse(self.loop._is_running)
        
        self.loop.start()
        self.assertTrue(self.loop._is_running)
        
        self.loop.stop()
        self.assertFalse(self.loop._is_running)
    
    def test_set_query(self):
        """Test setting pending query."""
        self.loop.set_query("What happened?")
        self.assertEqual(self.loop._pending_query, "What happened?")
    
    def test_get_status(self):
        """Test status reporting."""
        self.loop.start()
        self.loop.set_query("Test query")
        
        status = self.loop.get_status()
        
        self.assertTrue(status["is_running"])
        self.assertEqual(status["pending_query"], "Test query")
    
    def test_process_frame_not_running(self):
        """Test that processing returns None when not running."""
        result = self.loop.process_frame(timestamp=10.0)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
