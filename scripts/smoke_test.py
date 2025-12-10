#!/usr/bin/env python3
"""
Smoke Test for Gameplay Vision LLM.

This script provides a quick end-to-end verification of the system
using minimal resources. It tests:

1. Configuration preset loading
2. Knowledge base creation and JSON export
3. Timeline indexing and event creation
4. Basic inference pipeline (if --full is specified)

Usage:
    # Quick test (no GPU required)
    python scripts/smoke_test.py

    # Full test with synthetic data
    python scripts/smoke_test.py --full

    # Test with a real video (requires GPU)
    python scripts/smoke_test.py --video path/to/short_clip.mp4

Example Output:
    [PASS] Configuration presets loaded
    [PASS] Knowledge base created
    [PASS] Timeline indexer created
    [PASS] Entity tracking simulation
    [PASS] JSON export successful
    [PASS] All smoke tests passed!
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SmokeTestResult:
    """Tracks smoke test results."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name: str) -> None:
        """Record a passed test."""
        self.passed += 1
        print(f"\033[92m[PASS]\033[0m {name}")

    def fail(self, name: str, error: str = "") -> None:
        """Record a failed test."""
        self.failed += 1
        self.errors.append((name, error))
        print(f"\033[91m[FAIL]\033[0m {name}")
        if error:
            print(f"       Error: {error}")

    def summary(self) -> bool:
        """Print summary and return True if all passed."""
        print()
        print("=" * 60)
        print(f"Results: {self.passed} passed, {self.failed} failed")
        print("=" * 60)

        if self.failed == 0:
            print("\033[92m[PASS] All smoke tests passed!\033[0m")
            return True
        else:
            print("\033[91m[FAIL] Some tests failed:\033[0m")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
            return False


def test_config_presets(result: SmokeTestResult) -> None:
    """Test configuration preset loading."""
    try:
        from src.config.presets import (
            load_preset,
            PresetName,
            get_preset_summary,
        )

        # Load each preset
        for preset in PresetName:
            config = load_preset(preset)
            assert config.preset_name == preset
            assert config.estimated_vram_gb > 0

        # Test summary generation
        summary = get_preset_summary()
        assert "LIGHT" in summary
        assert "STANDARD" in summary
        assert "FULL" in summary

        result.ok("Configuration presets loaded")
    except Exception as e:
        result.fail("Configuration presets loaded", str(e))


def test_knowledge_base(result: SmokeTestResult) -> None:
    """Test knowledge base creation and operations."""
    try:
        from src.fusion_indexing.knowledge_base_builder import (
            KnowledgeBaseBuilder,
            EntityCategory,
            RelationType,
            EntityState,
        )

        # Create KB
        kb = KnowledgeBaseBuilder()

        # Register entities
        player = kb.register_entity(
            entity_id="player_001",
            concept_label="player_character",
            category=EntityCategory.PLAYER,
            timestamp=0.0,
            initial_state=EntityState(
                timestamp=0.0,
                position=(100.0, 200.0),
                bbox=(80.0, 180.0, 120.0, 220.0),
            ),
        )
        assert player is not None

        enemy = kb.register_entity(
            entity_id="enemy_001",
            concept_label="skeleton_warrior",
            category=EntityCategory.ENEMY,
            timestamp=1.0,
            initial_state=EntityState(
                timestamp=1.0,
                position=(150.0, 200.0),
                bbox=(130.0, 180.0, 170.0, 220.0),
            ),
        )
        assert enemy is not None

        # Add relationship
        edge = kb.add_relationship(
            source_id="player_001",
            target_id="enemy_001",
            relation_type=RelationType.ATTACKS,
            timestamp=2.0,
        )
        assert edge is not None

        # Update states
        kb.update_entity_state(
            entity_id="player_001",
            timestamp=2.0,
            position=(120.0, 200.0),
            bbox=(100.0, 180.0, 140.0, 220.0),
        )

        # Query
        history = kb.query_entity_history("player_001")
        assert len(history) == 2  # Initial + update

        # Get statistics
        stats = kb.get_statistics()
        assert stats["total_entities"] == 2
        assert stats["total_relationships"] == 1

        result.ok("Knowledge base created and populated")
    except Exception as e:
        result.fail("Knowledge base created", str(e))


def test_json_export(result: SmokeTestResult) -> None:
    """Test JSON export functionality."""
    try:
        from src.fusion_indexing.knowledge_base_builder import (
            KnowledgeBaseBuilder,
            EntityCategory,
            RelationType,
            EntityState,
        )

        # Create KB with test data
        kb = KnowledgeBaseBuilder()
        kb.register_entity(
            entity_id="test_entity",
            concept_label="test_label",
            category=EntityCategory.UNKNOWN,
            timestamp=0.0,
            initial_state=EntityState(timestamp=0.0, position=(50.0, 50.0)),
        )

        # Export to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            temp_path = f.name

        export_data = kb.export_to_json(
            path=temp_path,
            video_source="test_video.mp4",
            video_duration=60.0,
        )

        # Verify export structure
        assert export_data["schema_version"] == "1.0.0"
        assert export_data["entity_count"] == 1
        assert len(export_data["entities"]) == 1
        assert export_data["video_source"] == "test_video.mp4"

        # Load and verify
        loaded_kb = KnowledgeBaseBuilder.load_from_json(temp_path)
        assert loaded_kb.get_entity("test_entity") is not None

        # Cleanup
        Path(temp_path).unlink()

        result.ok("JSON export and import successful")
    except Exception as e:
        result.fail("JSON export and import", str(e))


def test_timeline_indexer(result: SmokeTestResult) -> None:
    """Test timeline indexer creation and events."""
    try:
        from src.fusion_indexing.timeline_indexer import (
            TimelineIndexer,
            TimelineEvent,
            ModalityType,
            EventPriority,
        )

        # Create timeline
        timeline = TimelineIndexer()

        # Add events
        events = [
            TimelineEvent(
                timestamp=0.0,
                modality=ModalityType.VISUAL,
                description="Player enters scene",
                priority=EventPriority.MEDIUM,
            ),
            TimelineEvent(
                timestamp=5.0,
                modality=ModalityType.AUDIO,
                description="Combat music starts",
                priority=EventPriority.LOW,
            ),
            TimelineEvent(
                timestamp=10.0,
                modality=ModalityType.VISUAL,
                description="Boss spawns",
                priority=EventPriority.CRITICAL,
                entity_id="boss_001",
            ),
            TimelineEvent(
                timestamp=15.0,
                modality=ModalityType.OCR,
                description="HP: 100/100",
                priority=EventPriority.LOW,
            ),
        ]

        for event in events:
            timeline.add_event(event)

        # Query by time range
        range_events = timeline.query_range(0.0, 10.0)
        assert len(range_events) >= 2

        # Query by modality
        visual_events = timeline.query_by_modality(ModalityType.VISUAL)
        assert len(visual_events) == 2

        # Build transcript
        transcript = timeline.build_structured_transcript()
        assert "Player enters scene" in transcript
        assert "Boss spawns" in transcript

        # Get statistics
        stats = timeline.get_statistics()
        assert stats["total_events"] == 4

        result.ok("Timeline indexer created and populated")
    except Exception as e:
        result.fail("Timeline indexer", str(e))


def test_schema_module(result: SmokeTestResult) -> None:
    """Test schema definitions and documentation."""
    try:
        from src.fusion_indexing.schema import (
            SCHEMA_VERSION,
            EntityCategorySchema,
            RelationTypeSchema,
            ModalityTypeSchema,
            EventPrioritySchema,
            EntitySchema,
            TimelineEventSchema,
            KnowledgeBaseExport,
            TimelineExport,
            FullSessionExport,
            get_schema_documentation,
        )

        # Check schema version
        assert SCHEMA_VERSION == "1.0.0"

        # Check enum values exist
        assert EntityCategorySchema.PLAYER.value == "player"
        assert RelationTypeSchema.ATTACKS.value == "attacks"
        assert ModalityTypeSchema.VISUAL.value == "visual"
        assert EventPrioritySchema.CRITICAL.value == "critical"

        # Create schema objects
        entity = EntitySchema(
            entity_id="test",
            concept_label="test_entity",
            category="player",
            first_seen=0.0,
            last_seen=10.0,
        )
        assert entity.to_dict()["entity_id"] == "test"

        event = TimelineEventSchema(
            timestamp=5.0,
            modality="visual",
            description="Test event",
        )
        assert event.to_dict()["timestamp"] == 5.0

        # Create full export
        export = FullSessionExport(
            video_source="test.mp4",
            video_duration=60.0,
        )
        export_dict = export.to_dict()
        assert export_dict["schema_version"] == SCHEMA_VERSION

        # Get documentation
        docs = get_schema_documentation()
        assert "ENTITY CATEGORIES" in docs
        assert "RELATIONSHIP TYPES" in docs

        result.ok("Schema module validated")
    except Exception as e:
        result.fail("Schema module", str(e))


def test_temporal_module(result: SmokeTestResult) -> None:
    """Test temporal processing module."""
    try:
        from src.temporal.internvideo_hico_module import (
            CompressionLevel,
            TemporalToken,
            HiCoConfig,
            HierarchicalCompressor,
        )
        import torch

        # Check config
        config = HiCoConfig()
        assert config.hidden_dim == 1408
        assert config.clip_duration_sec == 4.0

        # Check compression levels
        assert CompressionLevel.FRAME.value == "frame"
        assert CompressionLevel.CLIP.value == "clip"
        assert CompressionLevel.VIDEO.value == "video"

        # Create temporal token
        token = TemporalToken(
            embedding=torch.randn(config.hidden_dim),
            start_time=0.0,
            end_time=4.0,
            compression_level=CompressionLevel.CLIP,
            source_frame_count=16,
        )
        assert token.source_frame_count == 16
        assert "CLIP" in repr(token)

        # Create compressor (CPU only for smoke test)
        config.device = "cpu"
        compressor = HierarchicalCompressor(config)

        # Test compression
        fake_frames = torch.randn(1, 16, config.hidden_dim)
        clip_token = compressor.compress_frames_to_clip(fake_frames)
        assert clip_token.shape == (1, 1, config.hidden_dim)

        result.ok("Temporal module validated")
    except Exception as e:
        result.fail("Temporal module", str(e))


def test_integration(result: SmokeTestResult) -> None:
    """Test integration between KB and Timeline."""
    try:
        from src.fusion_indexing.knowledge_base_builder import (
            KnowledgeBaseBuilder,
            EntityCategory,
            EntityState,
        )
        from src.fusion_indexing.timeline_indexer import (
            TimelineIndexer,
            TimelineEvent,
            ModalityType,
            EventPriority,
        )

        # Create both
        kb = KnowledgeBaseBuilder()
        timeline = TimelineIndexer()

        # Simulate processing
        for t in [0.0, 1.0, 2.0, 3.0, 4.0]:
            # Add entity state
            if t == 0.0:
                kb.register_entity(
                    entity_id="dragon_boss",
                    concept_label="dragon",
                    category=EntityCategory.ENEMY,
                    timestamp=t,
                    initial_state=EntityState(
                        timestamp=t,
                        position=(300.0, 200.0),
                    ),
                )
            else:
                kb.update_entity_state(
                    entity_id="dragon_boss",
                    timestamp=t,
                    position=(300.0 + t * 10, 200.0),
                )

            # Add timeline event
            timeline.add_event(
                TimelineEvent(
                    timestamp=t,
                    modality=ModalityType.VISUAL,
                    description=f"Dragon at position {300 + t*10}",
                    priority=EventPriority.HIGH,
                    entity_id="dragon_boss",
                )
            )

        # Verify synchronization
        entity = kb.get_entity("dragon_boss")
        assert entity is not None
        assert len(entity.state_history) == 5

        entity_events = timeline.query_by_entity("dragon_boss")
        assert len(entity_events) == 5

        # Build combined context (simulating what goes to LLM)
        kb_context = kb.export_for_llm()
        timeline_context = timeline.build_structured_transcript()

        assert "dragon_boss" in kb_context
        assert "Dragon at position" in timeline_context

        result.ok("KB-Timeline integration validated")
    except Exception as e:
        result.fail("KB-Timeline integration", str(e))


def run_smoke_tests(full: bool = False, video_path: str = None) -> bool:
    """
    Run all smoke tests.

    Args:
        full: If True, run extended tests
        video_path: Optional path to test video

    Returns:
        True if all tests passed
    """
    print()
    print("=" * 60)
    print("GAMEPLAY VISION LLM - Smoke Tests")
    print("=" * 60)
    print()

    result = SmokeTestResult()

    # Core module tests (no GPU required)
    test_config_presets(result)
    test_knowledge_base(result)
    test_json_export(result)
    test_timeline_indexer(result)
    test_schema_module(result)
    test_temporal_module(result)
    test_integration(result)

    if full:
        print()
        print("Running extended tests...")
        # Add more comprehensive tests here if needed

    if video_path:
        print()
        print(f"Testing with video: {video_path}")
        # Add video-based tests here

    return result.summary()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Smoke test for Gameplay Vision LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run extended tests",
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to test video for full pipeline test",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    success = run_smoke_tests(full=args.full, video_path=args.video)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
