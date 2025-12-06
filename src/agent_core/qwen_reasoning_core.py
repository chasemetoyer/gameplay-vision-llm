"""
Qwen3-VL Reasoning Core Module.

This module implements the final orchestration layer for the Perception-Reasoning
Loop, integrating Qwen3-VL-8B-Instruct as the core reasoning LLM.

Key capabilities:
1. Hybrid retrieval (time-based + semantic) from timeline index
2. Dynamic resolution visual input processing
3. Full perception-reasoning loop orchestration

References:
- [A: 17] Intermediate representation consumption
- [A: 18] Segmentation + retrieval strategy
- [A: 34] Context window management
- [B: 67] Naive Dynamic Resolution
- [B: 82] Perception-Reasoning Loop
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch

if TYPE_CHECKING:
    from PIL import Image
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ReasoningCoreConfig:
    """Configuration for the Qwen3-VL Reasoning Core."""

    # Model settings
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    use_flash_attention: bool = True

    # Context budget (tokens)
    max_timeline_tokens: int = 5000
    max_visual_tokens: int = 2000
    max_total_tokens: int = 8000

    # Retrieval settings
    retrieval_window_sec: float = 30.0  # ± window for time-based retrieval
    semantic_top_k: int = 10  # Top-k for semantic retrieval
    embedding_model: str = "all-MiniLM-L6-v2"  # Sentence transformer model

    # Visual processing (Dynamic Resolution)
    min_pixels: int = 256 * 32 * 32  # ~262K pixels minimum
    max_pixels: int = 512 * 32 * 32  # ~524K pixels maximum

    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20

    # =========================================================================
    # Section 9: Projector Dimensions
    # =========================================================================
    siglip_dim: int = 1152          # SigLIP 2 So400m embedding dimension
    
    # Video embedding dimensions (choose one based on what's working)
    internvideo_dim: int = 1408     # InternVideo2-1B (legacy, may not work with transformers 5.x)
    videomae_dim: int = 768         # VideoMAE (transformers 5.x compatible)
    
    # Audio embedding dimensions (choose one based on what's working)
    audiomae_dim: int = 1024        # AudioMAE embedding dimension (legacy)
    wav2vec2_dim: int = 1024        # Wav2Vec2-Large (transformers 5.x compatible, same dim)
    
    llm_hidden_dim: int = 4096      # Qwen3-VL-8B hidden dimension

    # =========================================================================
    # Section 9: Temporal Context (HiCo)
    # =========================================================================
    temporal_window_minutes: float = 5.0  # Rolling context window
    max_hico_tokens: int = 2048           # Max tokens for HiCo context

    # =========================================================================
    # Section 9: Trigger Detection
    # =========================================================================
    trigger_confidence_threshold: float = 0.8
    trigger_concepts: list = field(default_factory=lambda: [
        "player", "enemy", "health_bar", "boss", "death"
    ])
    audio_trigger_events: list = field(default_factory=lambda: [
        "speech", "explosion", "alert", "damage"
    ])

    # System prompt
    system_prompt: str = """You are an expert gameplay analyst. You have access to:
1. A timeline of events extracted from the video (OCR, speech, visual detections)
2. The current video frame for visual grounding
3. An entity knowledge base tracking game objects

Analyze the provided context and answer the user's question accurately.
When referencing events, cite their timestamps. Be concise but thorough."""


# =============================================================================
# Special Tokens for Multimodal Interleaving
# =============================================================================

class SpecialTokens:
    """
    Special token definitions for multimodal content interleaving.
    
    Token format per research doc Section 2 (Data Schema):
    - <ts=MM:SS> : Timestamp markers
    - <v>        : Video embedding placeholder
    - <r>        : Region embedding placeholder (SAM3+SigLIP2)
    - <a>        : Audio embedding placeholder
    """
    
    TIMESTAMP_PREFIX = "<ts="
    TIMESTAMP_SUFFIX = ">"
    VIDEO_TOKEN = "<v>"
    REGION_TOKEN = "<r>"
    AUDIO_TOKEN = "<a>"
    
    # Embedding placeholder markers
    VIDEO_EMB = "<v_emb>"
    REGION_EMB = "<r_emb>"
    AUDIO_EMB = "<a_emb>"
    
    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Format timestamp as <ts=MM:SS> token."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"<ts={mins:02d}:{secs:02d}>"
    
    @staticmethod
    def parse_timestamp(token: str) -> Optional[float]:
        """Parse <ts=MM:SS> token back to seconds."""
        import re
        match = re.match(r"<ts=(\d+):(\d+)>", token)
        if match:
            mins, secs = map(int, match.groups())
            return mins * 60 + secs
        return None
    
    @classmethod
    def format_interleaved_context(
        cls,
        events: list,
        include_embeddings: bool = False,
    ) -> str:
        """
        Format events with special tokens for multimodal interleaving.
        
        Args:
            events: List of timeline events with timestamp and modality info
            include_embeddings: Whether to include embedding placeholders
            
        Returns:
            Formatted string with special tokens
        """
        lines = []
        
        for event in events:
            ts_token = cls.format_timestamp(event.timestamp)
            
            # Determine modality token
            modality = getattr(event, "modality", "text")
            if modality == "visual":
                mod_token = cls.REGION_EMB if include_embeddings else cls.REGION_TOKEN
            elif modality == "audio":
                mod_token = cls.AUDIO_EMB if include_embeddings else cls.AUDIO_TOKEN
            elif modality == "video":
                mod_token = cls.VIDEO_EMB if include_embeddings else cls.VIDEO_TOKEN
            else:
                mod_token = ""
            
            # Format: <ts=MM:SS> [<mod>] description
            if mod_token:
                lines.append(f"{ts_token} {mod_token} {event.description}")
            else:
                lines.append(f"{ts_token} {event.description}")
        
        return "\n".join(lines)


# =============================================================================
# MultiModal Projector (Section 3, Phase 2)
# =============================================================================

class MultiModalProjector(torch.nn.Module):
    """
    Projects encoder embeddings to LLM hidden dimension.
    
    Bridges the dimension mismatch between perception encoders and Qwen.
    Per research doc: "You need to bridge the dimension mismatch between 
    your encoders and Qwen2.5 (Hidden size 3584)."
    
    Architecture:
        Linear(encoder_dim, llm_dim) -> GELU -> Linear(llm_dim, llm_dim)
    """
    
    def __init__(self, encoder_dim: int, llm_dim: int = 4096):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(encoder_dim, llm_dim),
            torch.nn.GELU(),
            torch.nn.Linear(llm_dim, llm_dim),
        )
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings to LLM space.
        
        Args:
            x: Tensor of shape (batch, seq_len, encoder_dim) or (batch, encoder_dim)
            
        Returns:
            Projected tensor of shape (batch, seq_len, llm_dim) or (batch, llm_dim)
        """
        return self.net(x)


class ProjectorBank:
    """
    Collection of projectors for all modalities.
    
    Manages separate projectors for:
    - SigLIP2 (region embeddings): 1152 -> 3584
    - VideoMAE (video embeddings): 768 -> 3584 (transformers 5.x compatible)
    - InternVideo2 (video embeddings): 1408 -> 3584 (legacy)
    - Wav2Vec2/AudioMAE (audio embeddings): 1024 -> 3584
    """
    
    def __init__(self, config: ReasoningCoreConfig):
        self.config = config
        self.device = config.device
        
        # Initialize projectors
        self.siglip_proj = MultiModalProjector(
            config.siglip_dim, config.llm_hidden_dim
        )
        
        # VideoMAE projector (768-dim, transformers 5.x compatible)
        self.videomae_proj = MultiModalProjector(
            config.videomae_dim, config.llm_hidden_dim
        )
        
        # Legacy InternVideo projector (1408-dim, may not work with transformers 5.x)
        self.video_proj = MultiModalProjector(
            config.internvideo_dim, config.llm_hidden_dim
        )
        
        # Audio projector (1024-dim, works with both Wav2Vec2 and AudioMAE)
        self.audio_proj = MultiModalProjector(
            config.audiomae_dim, config.llm_hidden_dim
        )
        
        self._initialized = False
    
    def to(self, device: str) -> "ProjectorBank":
        """Move all projectors to device."""
        self.siglip_proj = self.siglip_proj.to(device)
        self.videomae_proj = self.videomae_proj.to(device)
        self.video_proj = self.video_proj.to(device)
        self.audio_proj = self.audio_proj.to(device)
        self.device = device
        return self
    
    def project_region(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project SigLIP2 region embeddings (1152 -> 3584)."""
        return self.siglip_proj(embeddings.to(self.device))
    
    def project_videomae(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project VideoMAE embeddings (768 -> 3584)."""
        return self.videomae_proj(embeddings.to(self.device))
    
    def project_video(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project InternVideo2 HiCo embeddings (1408 -> 3584, legacy)."""
        return self.video_proj(embeddings.to(self.device))
    
    def project_audio(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project Wav2Vec2/AudioMAE embeddings (1024 -> 3584)."""
        return self.audio_proj(embeddings.to(self.device))
    
    def load_weights(self, path: str) -> None:
        """Load pre-trained projector weights."""
        state_dict = torch.load(path, map_location=self.device, weights_only=False)
        
        # Load the 3 trained modalities
        if "siglip" in state_dict:
            self.siglip_proj.load_state_dict(state_dict["siglip"])
        if "videomae" in state_dict:
            self.videomae_proj.load_state_dict(state_dict["videomae"])
        if "audio" in state_dict:
            self.audio_proj.load_state_dict(state_dict["audio"])
        
        # Legacy: also try to load video_proj if present
        if "video" in state_dict:
            self.video_proj.load_state_dict(state_dict["video"])
        
        self._initialized = True
        logger.info(f"Loaded projector weights from {path}")
    
    def save_weights(self, path: str) -> None:
        """Save projector weights."""
        state_dict = {
            "siglip": self.siglip_proj.state_dict(),
            "videomae": self.videomae_proj.state_dict(),
            "audio": self.audio_proj.state_dict(),
            # Also save legacy video_proj for backward compatibility
            "video": self.video_proj.state_dict(),
        }
        torch.save(state_dict, path)
        logger.info(f"Saved projector weights to {path}")


# =============================================================================
# Trigger Detector (Section 9, Step 4)
# =============================================================================

@dataclass
class TriggerEvent:
    """Represents a trigger activation event."""
    timestamp: float
    trigger_type: str  # "concept", "audio", "threshold"
    source: str        # e.g., "SAM3", "Qwen2-Audio"
    confidence: float
    details: dict = field(default_factory=dict)


class TriggerDetector:
    """
    Monitors perception outputs for activation events.
    
    Per Section 9: "A 'Trigger' (e.g., SAM 3 detects 'box', Qwen2-Audio 
    detects 'doorbell') activates the Reasoning Core."
    """
    
    def __init__(self, config: ReasoningCoreConfig):
        self.config = config
        self.pending_triggers: list[TriggerEvent] = []
        self._last_trigger_time: float = 0.0
        self._cooldown_sec: float = 1.0  # Minimum time between triggers
    
    def check_visual_trigger(
        self,
        detections: list[dict],
        timestamp: float,
    ) -> Optional[TriggerEvent]:
        """
        Check SAM3 detections for trigger concepts.
        
        Args:
            detections: List of SAM3 detection dicts with 'label', 'confidence'
            timestamp: Current video timestamp
            
        Returns:
            TriggerEvent if triggered, None otherwise
        """
        for det in detections:
            label = det.get("label", "").lower()
            confidence = det.get("confidence", 0.0)
            
            # Check if label matches any trigger concept
            for concept in self.config.trigger_concepts:
                if concept.lower() in label:
                    if confidence >= self.config.trigger_confidence_threshold:
                        return TriggerEvent(
                            timestamp=timestamp,
                            trigger_type="concept",
                            source="SAM3",
                            confidence=confidence,
                            details={"label": label, "concept": concept},
                        )
        
        return None
    
    def check_audio_trigger(
        self,
        audio_events: list[dict],
        timestamp: float,
    ) -> Optional[TriggerEvent]:
        """
        Check audio events for triggers.
        
        Args:
            audio_events: List of audio event dicts with 'event', 'confidence'
            timestamp: Current timestamp
            
        Returns:
            TriggerEvent if triggered, None otherwise
        """
        for event in audio_events:
            event_type = event.get("event", "").lower()
            confidence = event.get("confidence", 0.0)
            
            for trigger_event in self.config.audio_trigger_events:
                if trigger_event.lower() in event_type:
                    if confidence >= self.config.trigger_confidence_threshold:
                        return TriggerEvent(
                            timestamp=timestamp,
                            trigger_type="audio",
                            source="Qwen2-Audio",
                            confidence=confidence,
                            details={"event": event_type},
                        )
        
        return None
    
    def process_frame(
        self,
        timestamp: float,
        visual_detections: Optional[list[dict]] = None,
        audio_events: Optional[list[dict]] = None,
    ) -> Optional[TriggerEvent]:
        """
        Process a frame and check all trigger sources.
        
        Args:
            timestamp: Current timestamp
            visual_detections: SAM3 detections
            audio_events: Audio analysis events
            
        Returns:
            TriggerEvent if any trigger fires, None otherwise
        """
        # Check cooldown
        if timestamp - self._last_trigger_time < self._cooldown_sec:
            return None
        
        # Check visual triggers
        if visual_detections:
            trigger = self.check_visual_trigger(visual_detections, timestamp)
            if trigger:
                self._last_trigger_time = timestamp
                self.pending_triggers.append(trigger)
                logger.info(f"Visual trigger: {trigger.details}")
                return trigger
        
        # Check audio triggers
        if audio_events:
            trigger = self.check_audio_trigger(audio_events, timestamp)
            if trigger:
                self._last_trigger_time = timestamp
                self.pending_triggers.append(trigger)
                logger.info(f"Audio trigger: {trigger.details}")
                return trigger
        
        return None
    
    def get_pending_triggers(self) -> list[TriggerEvent]:
        """Get and clear pending triggers."""
        triggers = self.pending_triggers.copy()
        self.pending_triggers.clear()
        return triggers


# =============================================================================
# Temporal Context Manager (Section 9, Step 3)
# =============================================================================

class TemporalContextManager:
    """
    Manages rolling compressed temporal context from InternVideo2.5 HiCo.
    
    Per Section 9: "InternVideo2.5 (HiCo) maintains a rolling compressed 
    context of the last 5-10 minutes of footage."
    """
    
    def __init__(self, config: ReasoningCoreConfig):
        self.config = config
        self.window_seconds = config.temporal_window_minutes * 60
        
        # FIFO buffer of compressed embeddings
        self._embeddings: list[tuple[float, torch.Tensor]] = []  # (timestamp, embedding)
        self._max_tokens = config.max_hico_tokens
    
    def add_context(
        self,
        timestamp: float,
        hico_embedding: torch.Tensor,
    ) -> None:
        """
        Add a compressed HiCo embedding to the context buffer.
        
        Args:
            timestamp: Timestamp for this embedding
            hico_embedding: Compressed embedding from InternVideo2.5
        """
        self._embeddings.append((timestamp, hico_embedding))
        
        # Prune old embeddings outside the window
        cutoff = timestamp - self.window_seconds
        self._embeddings = [
            (ts, emb) for ts, emb in self._embeddings
            if ts >= cutoff
        ]
        
        # Also limit by token count
        while len(self._embeddings) > self._max_tokens:
            self._embeddings.pop(0)
    
    def get_context(
        self,
        current_timestamp: Optional[float] = None,
    ) -> tuple[list[float], Optional[torch.Tensor]]:
        """
        Get the current temporal context.
        
        Args:
            current_timestamp: Optional timestamp to filter by
            
        Returns:
            Tuple of (timestamps, stacked embeddings tensor)
        """
        if not self._embeddings:
            return [], None
        
        timestamps = [ts for ts, _ in self._embeddings]
        embeddings = [emb for _, emb in self._embeddings]
        
        # Stack into single tensor
        stacked = torch.stack(embeddings, dim=0)
        
        return timestamps, stacked
    
    def get_context_summary(self) -> str:
        """Get a text summary of the temporal context."""
        if not self._embeddings:
            return "No temporal context available."
        
        oldest_ts = self._embeddings[0][0]
        newest_ts = self._embeddings[-1][0]
        duration = newest_ts - oldest_ts
        
        return (
            f"Temporal context: {len(self._embeddings)} frames, "
            f"{duration:.1f}s duration "
            f"({oldest_ts:.1f}s to {newest_ts:.1f}s)"
        )
    
    def clear(self) -> None:
        """Clear all context."""
        self._embeddings.clear()


# =============================================================================
# Timeline Retriever (Hybrid: Time-based + Semantic)
# =============================================================================

class TimelineRetriever:
    """
    Hybrid retrieval system for timeline events.
    
    Implements both time-based and semantic retrieval strategies:
    - Time-based: For queries with explicit timestamps (e.g., "at 5:30")
    - Semantic: For general queries using embedding similarity
    """

    # Regex patterns for timestamp parsing
    TIMESTAMP_PATTERNS = [
        r"(\d{1,2}):(\d{2}):(\d{2})",  # HH:MM:SS
        r"(\d{1,2}):(\d{2})",  # MM:SS
        r"at\s+(\d+(?:\.\d+)?)\s*(?:sec|seconds?|s)?",  # "at 30 seconds"
        r"around\s+(\d+(?:\.\d+)?)\s*(?:sec|seconds?|s)?",  # "around 30s"
    ]

    def __init__(self, config: ReasoningCoreConfig):
        self.config = config
        self._embedder = None
        self._timeline_embeddings = None
        self._timeline_events = []

    def _load_embedder(self) -> None:
        """Lazy load the sentence transformer for semantic search."""
        if self._embedder is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedder: {self.config.embedding_model}")
            self._embedder = SentenceTransformer(self.config.embedding_model)
            logger.info("Embedder loaded successfully")
        except ImportError:
            logger.warning("sentence-transformers not installed.")
            logger.warning("Semantic retrieval disabled. Install with:")
            logger.warning("  pip install sentence-transformers")
            self._embedder = None
        except Exception as e:
            logger.warning(f"Failed to load embedder: {e}")
            self._embedder = None

    def index_timeline(self, timeline_indexer) -> None:
        """
        Index timeline events for semantic retrieval.
        
        Args:
            timeline_indexer: TimelineIndexer instance with events
        """
        self._load_embedder()

        # Get all events from timeline
        events = timeline_indexer._events if hasattr(timeline_indexer, "_events") else []
        self._timeline_events = events

        if not events or self._embedder is None:
            logger.info(f"Indexed {len(events)} events (semantic disabled)")
            return

        # Create embeddings for all event descriptions
        descriptions = [e.description for e in events]
        self._timeline_embeddings = self._embedder.encode(
            descriptions,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        logger.info(f"Indexed {len(events)} events with embeddings")

    def parse_timestamp(self, query: str) -> Optional[float]:
        """
        Extract timestamp from query if present.
        
        Args:
            query: User query string
            
        Returns:
            Timestamp in seconds, or None if not found
        """
        query_lower = query.lower()

        for pattern in self.TIMESTAMP_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                groups = match.groups()
                if len(groups) == 3:  # HH:MM:SS
                    h, m, s = map(int, groups)
                    return h * 3600 + m * 60 + s
                elif len(groups) == 2:  # MM:SS
                    m, s = map(int, groups)
                    return m * 60 + s
                elif len(groups) == 1:  # Just seconds
                    return float(groups[0])

        return None

    def retrieve_by_timestamp(
        self,
        timestamp: float,
        window: Optional[float] = None,
        timeline_indexer=None,
    ) -> list:
        """
        Retrieve events around a specific timestamp.
        
        Args:
            timestamp: Center timestamp in seconds
            window: Time window (± seconds). Uses config default if None.
            timeline_indexer: Optional timeline to query from
            
        Returns:
            List of TimelineEvent objects
        """
        window = window or self.config.retrieval_window_sec

        if timeline_indexer is not None:
            return timeline_indexer.query_around_timestamp(timestamp, window * 2)

        # Fallback to cached events
        start = timestamp - window
        end = timestamp + window
        return [
            e for e in self._timeline_events
            if start <= e.timestamp <= end
        ]

    def retrieve_by_semantic(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list:
        """
        Retrieve events by semantic similarity to query.
        
        Args:
            query: User query string
            top_k: Number of results to return
            
        Returns:
            List of TimelineEvent objects sorted by relevance
        """
        self._load_embedder()

        if self._embedder is None or self._timeline_embeddings is None:
            logger.warning("Semantic retrieval unavailable, returning empty")
            return []

        top_k = top_k or self.config.semantic_top_k

        # Encode query
        query_embedding = self._embedder.encode(
            query,
            convert_to_tensor=True,
        )

        # Compute similarities
        from sentence_transformers import util
        similarities = util.cos_sim(query_embedding, self._timeline_embeddings)[0]

        # Get top-k indices
        top_indices = similarities.argsort(descending=True)[:top_k]

        return [self._timeline_events[i] for i in top_indices.cpu().numpy()]

    def hybrid_retrieve(
        self,
        query: str,
        timeline_indexer=None,
    ) -> list:
        """
        Hybrid retrieval: time-based if timestamp found, else semantic.
        
        Args:
            query: User query string
            timeline_indexer: Optional TimelineIndexer instance
            
        Returns:
            List of relevant TimelineEvent objects
        """
        # Try to extract timestamp from query
        timestamp = self.parse_timestamp(query)

        if timestamp is not None:
            logger.info(f"Time-based retrieval at {timestamp:.1f}s")
            events = self.retrieve_by_timestamp(
                timestamp,
                timeline_indexer=timeline_indexer,
            )
            # Also add semantic results for context
            if self._embedder is not None:
                semantic_events = self.retrieve_by_semantic(query, top_k=5)
                # Merge, avoiding duplicates
                seen_ids = {id(e) for e in events}
                for e in semantic_events:
                    if id(e) not in seen_ids:
                        events.append(e)
            return events
        else:
            logger.info("Semantic retrieval (no timestamp found)")
            return self.retrieve_by_semantic(query)


# =============================================================================
# Visual Input Processor (Dynamic Resolution)
# =============================================================================

class VisualInputProcessor:
    """
    Processes visual inputs for Qwen3-VL with dynamic resolution.
    
    Implements Naive Dynamic Resolution to allocate tokens based on
    information density, ensuring high-detail regions are preserved.
    """

    def __init__(self, config: ReasoningCoreConfig):
        self.config = config

    def process_frame(
        self,
        image: "Image.Image",
        detail_level: str = "auto",
    ) -> dict:
        """
        Process a frame for Qwen3-VL input.
        
        Args:
            image: PIL Image of the current frame
            detail_level: "low", "high", or "auto"
            
        Returns:
            Message content dict for Qwen3-VL
        """
        if detail_level == "low":
            min_pix = 128 * 32 * 32
            max_pix = 256 * 32 * 32
        elif detail_level == "high":
            min_pix = 512 * 32 * 32
            max_pix = 1024 * 32 * 32
        else:  # auto
            min_pix = self.config.min_pixels
            max_pix = self.config.max_pixels

        return {
            "type": "image",
            "image": image,
            "min_pixels": min_pix,
            "max_pixels": max_pix,
        }

    def process_region_tokens(
        self,
        regions: list[dict],
    ) -> str:
        """
        Format region tokens from SAM/SigLIP for text context.
        
        Args:
            regions: List of region dicts with 'label', 'bbox', 'embedding'
            
        Returns:
            Formatted text describing detected regions
        """
        if not regions:
            return ""

        lines = ["Detected regions in frame:"]
        for i, region in enumerate(regions):
            label = region.get("label", f"region_{i}")
            bbox = region.get("bbox", [])
            confidence = region.get("confidence", 0.0)

            if bbox:
                lines.append(
                    f"  - {label}: bbox={bbox}, confidence={confidence:.2f}"
                )
            else:
                lines.append(f"  - {label}: confidence={confidence:.2f}")

        return "\n".join(lines)


# =============================================================================
# Qwen3-VL Reasoning Core
# =============================================================================

class QwenVLCore:
    """
    Main orchestrator for the Perception-Reasoning Loop.
    
    Integrates:
    - TimelineRetriever for context retrieval
    - VisualInputProcessor for frame handling
    - Qwen3-VL for reasoning and response generation
    
    Example:
        >>> core = QwenVLCore()
        >>> response = core.reason(
        ...     query="What happened at 5:30?",
        ...     current_frame=frame_image,
        ...     timeline_indexer=timeline,
        ... )
    """

    def __init__(
        self,
        config: Optional[ReasoningCoreConfig] = None,
        lora_path: Optional[str] = None,
    ):
        self.config = config or ReasoningCoreConfig()
        self.lora_path = lora_path
        self.retriever = TimelineRetriever(self.config)
        self.visual_processor = VisualInputProcessor(self.config)

        self._model = None
        self._processor = None

        logger.info("QwenVLCore initialized")

    def _load_model(self) -> None:
        """Lazy load Qwen3-VL model and processor."""
        if self._model is not None:
            return

        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

            logger.info(f"Loading Qwen3-VL: {self.config.model_name}")

            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                self.config.model_name,
            )

            # Load model - try with flash attention first, then without
            model_kwargs = {
                "torch_dtype": self.config.dtype,
                "device_map": "auto",
            }
            
            # Try flash attention if enabled
            if self.config.use_flash_attention:
                try:
                    self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                        self.config.model_name,
                        attn_implementation="flash_attention_2",
                        **model_kwargs,
                    )
                    logger.info("Loaded with flash_attention_2")
                except Exception as fa_error:
                    logger.warning(f"Flash attention not available: {fa_error}")
                    logger.info("Falling back to standard attention...")
                    self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                        self.config.model_name,
                        **model_kwargs,
                    )
            else:
                self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.config.model_name,
                    **model_kwargs,
                )
            
            # Apply LoRA adapter if provided
            if self.lora_path and os.path.exists(self.lora_path):
                try:
                    from peft import PeftModel
                    
                    logger.info(f"Loading LoRA adapter from {self.lora_path}...")
                    self._model = PeftModel.from_pretrained(
                        self._model,
                        self.lora_path,
                    )
                    logger.info("LoRA adapter applied successfully")
                except ImportError:
                    logger.warning("PEFT not installed. LoRA adapter not applied.")
                    logger.warning("Install with: pip install peft")
                except Exception as e:
                    logger.warning(f"Failed to load LoRA adapter: {e}")
            
            self._model.eval()
            logger.info("Qwen3-VL loaded successfully")

        except ImportError as e:
            logger.error(f"ImportError loading Qwen3-VL: {e}")
            logger.warning("Qwen3VLForConditionalGeneration not available.")
            logger.warning("Please update transformers: pip install -U transformers")
            self._model = "placeholder"
            self._processor = None
        except Exception as e:
            logger.error(f"Failed to load Qwen3-VL: {e}")
            import traceback
            traceback.print_exc()
            self._model = "placeholder"
            self._processor = None

    def index_timeline(self, timeline_indexer) -> None:
        """
        Index timeline for semantic retrieval.
        
        Args:
            timeline_indexer: TimelineIndexer with events to index
        """
        self.retriever.index_timeline(timeline_indexer)

    def build_prompt(
        self,
        query: str,
        timeline_context: str,
        current_frame: Optional["Image.Image"] = None,
        region_tokens: Optional[str] = None,
        knowledge_base_context: Optional[str] = None,
    ) -> list[dict]:
        """
        Construct the message list for Qwen3-VL.
        
        Args:
            query: User question
            timeline_context: Retrieved timeline events formatted as text
            current_frame: Optional current video frame
            region_tokens: Optional region descriptions from SAM/SigLIP
            knowledge_base_context: Optional KB export
            
        Returns:
            List of message dicts for apply_chat_template
        """
        messages = []

        # System message
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": self.config.system_prompt}],
        })

        # Build user content
        user_content = []

        # Add current frame if provided
        if current_frame is not None:
            user_content.append(
                self.visual_processor.process_frame(current_frame)
            )

        # Build context text
        context_parts = []

        if timeline_context:
            context_parts.append("## Timeline Context\n" + timeline_context)

        if region_tokens:
            context_parts.append("## Visual Regions\n" + region_tokens)

        if knowledge_base_context:
            context_parts.append("## Entity Knowledge Base\n" + knowledge_base_context)

        # Add context as text
        if context_parts:
            context_text = "\n\n".join(context_parts)
            user_content.append({"type": "text", "text": context_text})

        # Add user query
        user_content.append({
            "type": "text",
            "text": f"\n## Question\n{query}",
        })

        messages.append({
            "role": "user",
            "content": user_content,
        })

        return messages

    def format_timeline_context(
        self,
        events: list,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Format retrieved events as timeline context text.
        
        Args:
            events: List of TimelineEvent objects
            max_tokens: Maximum token budget (approximate by chars)
            
        Returns:
            Formatted timeline string
        """
        if not events:
            return "No relevant events found in timeline."

        max_tokens = max_tokens or self.config.max_timeline_tokens

        # Sort by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        lines = []
        char_count = 0
        char_limit = max_tokens * 4  # Rough estimate: 4 chars per token

        for event in sorted_events:
            if hasattr(event, "format_compact"):
                line = event.format_compact()
            else:
                # Fallback formatting
                mins = int(event.timestamp // 60)
                secs = int(event.timestamp % 60)
                line = f"[{mins:02d}:{secs:02d}] {event.description}"

            if char_count + len(line) > char_limit:
                lines.append("... (more events truncated)")
                break

            lines.append(line)
            char_count += len(line)

        return "\n".join(lines)

    def reason(
        self,
        query: str,
        current_frame: Optional["Image.Image"] = None,
        timeline_indexer=None,
        knowledge_base=None,
        region_detections: Optional[list[dict]] = None,
    ) -> str:
        """
        Execute the full Perception-Reasoning Loop.
        
        This is the main entry point for the agent.
        
        Args:
            query: User question
            current_frame: Optional current video frame for visual grounding
            timeline_indexer: TimelineIndexer with indexed events
            knowledge_base: Optional KnowledgeBaseBuilder for entity context
            region_detections: Optional SAM/SigLIP region data
            
        Returns:
            Model response string
        """
        self._load_model()

        if self._processor is None:
            return "[Error: Qwen3-VL model not loaded]"

        # Step 1: Retrieve relevant context
        logger.info(f"Processing query: {query[:50]}...")

        if timeline_indexer is not None:
            self.retriever.index_timeline(timeline_indexer)

        events = self.retriever.hybrid_retrieve(query, timeline_indexer)
        timeline_context = self.format_timeline_context(events)
        logger.info(f"Retrieved {len(events)} events")

        # Step 2: Process visual input
        region_tokens = None
        if region_detections:
            region_tokens = self.visual_processor.process_region_tokens(
                region_detections
            )

        # Step 3: Get knowledge base context
        kb_context = None
        if knowledge_base is not None:
            kb_context = knowledge_base.export_for_llm(
                max_entities=15,
                max_relationships=20,
            )

        # Step 4: Build prompt
        messages = self.build_prompt(
            query=query,
            timeline_context=timeline_context,
            current_frame=current_frame,
            region_tokens=region_tokens,
            knowledge_base_context=kb_context,
        )

        # Step 5: Prepare inputs
        try:
            # Check if we have images
            has_image = current_frame is not None

            if has_image:
                # Use qwen-vl-utils for image processing if available
                try:
                    from qwen_vl_utils import process_vision_info

                    text = self._processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    images, videos, _ = process_vision_info(
                        messages,
                        image_patch_size=16,
                        return_video_kwargs=True,
                    )
                    inputs = self._processor(
                        text=text,
                        images=images,
                        videos=videos,
                        do_resize=False,
                        return_tensors="pt",
                    )
                except ImportError:
                    # Fallback: use processor directly
                    inputs = self._processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
            else:
                # Text-only mode
                inputs = self._processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )

            inputs = inputs.to(self._model.device)

        except Exception as e:
            logger.error(f"Failed to prepare inputs: {e}")
            return f"[Error preparing inputs: {e}]"

        # Step 6: Generate response
        try:
            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    do_sample=True,
                )

            # Trim input tokens from output
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            response = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            return response.strip()

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"[Error during generation: {e}]"


# =============================================================================
# Perception-Reasoning Loop (Section 9)
# =============================================================================

class PerceptionReasoningLoop:
    """
    Main orchestration class for the full Perception-Reasoning Loop.
    
    Per Section 9 Architecture:
    1. Input Ingestion: Video ring buffer, audio to Qwen2-Audio
    2. Primary Scan: SAM3 detector, PaddleOCR on ROIs
    3. Temporal Context: InternVideo2.5 HiCo rolling context
    4. Trigger Event: SAM3/Audio activates Reasoning Core
    5. Reasoning & Response: Qwen3-VL generates response
    
    Example:
        >>> loop = PerceptionReasoningLoop()
        >>> loop.start()
        >>> 
        >>> # Process frames
        >>> for frame, audio, timestamp in video_stream:
        ...     response = loop.process_frame(
        ...         frame=frame,
        ...         audio_chunk=audio,
        ...         timestamp=timestamp,
        ...     )
        ...     if response:
        ...         print(response)
    """
    
    def __init__(
        self,
        config: Optional[ReasoningCoreConfig] = None,
        timeline_indexer=None,
        knowledge_base=None,
        projector_weights_path: Optional[str] = None,
        lora_path: Optional[str] = None,
    ):
        self.config = config or ReasoningCoreConfig()
        
        # Store paths
        self.projector_weights_path = projector_weights_path
        self.lora_path = lora_path
        
        # Core components - pass lora_path to QwenVLCore for LoRA loading
        self.reasoning_core = QwenVLCore(self.config, lora_path=lora_path)
        self.trigger_detector = TriggerDetector(self.config)
        self.temporal_context = TemporalContextManager(self.config)
        
        # Initialize projectors and load weights if available
        self.projectors = ProjectorBank(self.config)
        self.projectors.to(self.config.device)
        if projector_weights_path and os.path.exists(projector_weights_path):
            self.projectors.load_weights(projector_weights_path)
            logger.info(f"Loaded projector weights from {projector_weights_path}")
        else:
            logger.info("Projectors initialized (no pre-trained weights loaded)")
        
        # External references
        self.timeline_indexer = timeline_indexer
        self.knowledge_base = knowledge_base
        
        # State
        self._is_running = False
        self._current_timestamp = 0.0
        self._pending_query: Optional[str] = None
        
        logger.info("PerceptionReasoningLoop initialized")
    
    def start(self) -> None:
        """Start the perception-reasoning loop."""
        self._is_running = True
        logger.info("Perception-Reasoning Loop started")
    
    def stop(self) -> None:
        """Stop the perception-reasoning loop."""
        self._is_running = False
        self.temporal_context.clear()
        logger.info("Perception-Reasoning Loop stopped")
    
    def set_query(self, query: str) -> None:
        """
        Set a pending user query to be answered on next trigger.
        
        Args:
            query: User's question
        """
        self._pending_query = query
        logger.info(f"Query set: {query[:50]}...")
    
    def add_hico_context(
        self,
        timestamp: float,
        hico_embedding: torch.Tensor,
    ) -> None:
        """
        Add InternVideo2.5 HiCo compressed context.
        
        Args:
            timestamp: Frame timestamp
            hico_embedding: Compressed embedding from HiCo
        """
        self.temporal_context.add_context(timestamp, hico_embedding)
    
    def process_frame(
        self,
        frame: Optional["Image.Image"] = None,
        timestamp: float = 0.0,
        visual_detections: Optional[list[dict]] = None,
        audio_events: Optional[list[dict]] = None,
        ocr_results: Optional[list[dict]] = None,
        region_embeddings: Optional[torch.Tensor] = None,
        videomae_embeddings: Optional[torch.Tensor] = None,
        audio_embeddings: Optional[torch.Tensor] = None,
        force_reason: bool = False,
    ) -> Optional[str]:
        """
        Process a single frame through the perception-reasoning loop.
        
        Per Section 9 flow:
        1. Check for trigger events (SAM3, Audio)
        2. If triggered (or forced), activate reasoning core
        3. Return response if generated
        
        Args:
            frame: Current video frame (PIL Image)
            timestamp: Frame timestamp in seconds
            visual_detections: SAM3 detection results
            audio_events: Qwen2-Audio event detections
            ocr_results: PaddleOCR text extractions
            region_embeddings: Pre-computed SigLIP2 embeddings (N, 1152)
            videomae_embeddings: Pre-computed VideoMAE embeddings (N, 768)
            audio_embeddings: Pre-computed Wav2Vec2 embeddings (N, 1024)
            force_reason: Force reasoning even without trigger
            
        Returns:
            Response string if reasoning was triggered, None otherwise
        """
        if not self._is_running:
            return None
        
        self._current_timestamp = timestamp
        
        # Step 4: Check for trigger events
        trigger = self.trigger_detector.process_frame(
            timestamp=timestamp,
            visual_detections=visual_detections,
            audio_events=audio_events,
        )
        
        # Only reason if triggered or forced
        if trigger is None and not force_reason:
            return None
        
        # Step 5: Activate Reasoning Core
        logger.info(f"Reasoning activated at {timestamp:.1f}s")
        
        # Determine query
        query = self._pending_query
        if query is None:
            # Generate automatic query based on trigger
            if trigger:
                query = self._generate_trigger_query(trigger)
            else:
                query = "Describe what's happening in this frame."
        
        # Prepare region detections with OCR
        all_detections = visual_detections or []
        if ocr_results:
            for ocr in ocr_results:
                all_detections.append({
                    "label": f"text: {ocr.get('text', '')}",
                    "confidence": ocr.get("confidence", 0.0),
                    "bbox": ocr.get("bbox", []),
                })
        
        # Get temporal context summary
        temporal_summary = self.temporal_context.get_context_summary()
        
        # Project multimodal embeddings if provided
        projected = self.project_embeddings(
            siglip_embeddings=region_embeddings,
            videomae_embeddings=videomae_embeddings,
            audio_embeddings=audio_embeddings,
        )
        multimodal_context = self.get_multimodal_context(projected)
        
        # Build full context string
        context_parts = []
        if temporal_summary:
            context_parts.append(f"[Temporal Context: {temporal_summary}]")
        if multimodal_context:
            context_parts.append(multimodal_context)
        
        full_context = "\n".join(context_parts)
        full_query = f"{query}\n\n{full_context}" if full_context else query
        
        # Execute reasoning
        response = self.reasoning_core.reason(
            query=full_query,
            current_frame=frame,
            timeline_indexer=self.timeline_indexer,
            knowledge_base=self.knowledge_base,
            region_detections=all_detections,
        )
        
        # Clear pending query after answering
        self._pending_query = None
        
        return response
    
    def _generate_trigger_query(self, trigger: TriggerEvent) -> str:
        """Generate an automatic query based on trigger event."""
        if trigger.trigger_type == "concept":
            concept = trigger.details.get("concept", "object")
            return f"A {concept} was just detected. Describe what's happening with it."
        elif trigger.trigger_type == "audio":
            event = trigger.details.get("event", "sound")
            return f"An audio event '{event}' was detected. What's happening?"
        else:
            return "Something triggered. Describe the current situation."
    
    def reason_now(
        self,
        query: str,
        frame: Optional["Image.Image"] = None,
    ) -> str:
        """
        Immediately run reasoning with a specific query.
        
        Args:
            query: User question
            frame: Optional current frame
            
        Returns:
            Response string
        """
        return self.reasoning_core.reason(
            query=query,
            current_frame=frame,
            timeline_indexer=self.timeline_indexer,
            knowledge_base=self.knowledge_base,
        )
    
    def project_embeddings(
        self,
        siglip_embeddings: Optional[torch.Tensor] = None,
        videomae_embeddings: Optional[torch.Tensor] = None,
        audio_embeddings: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Project raw encoder embeddings into LLM space using trained projectors.
        
        This is the key inference step that transforms perception embeddings
        into multimodal tokens that the LLM can understand.
        
        Args:
            siglip_embeddings: SigLIP2 region embeddings (N, 1152)
            videomae_embeddings: VideoMAE temporal embeddings (N, 768)
            audio_embeddings: Wav2Vec2 audio embeddings (N, 1024)
            
        Returns:
            Dict of projected embeddings, each (N, 4096)
        """
        projected = {}
        
        with torch.no_grad():
            if siglip_embeddings is not None:
                siglip_embeddings = siglip_embeddings.to(self.config.device).float()
                projected["siglip"] = self.projectors.project_region(siglip_embeddings)
                
            if videomae_embeddings is not None:
                videomae_embeddings = videomae_embeddings.to(self.config.device).float()
                projected["videomae"] = self.projectors.project_videomae(videomae_embeddings)
                
            if audio_embeddings is not None:
                audio_embeddings = audio_embeddings.to(self.config.device).float()
                projected["audio"] = self.projectors.project_audio(audio_embeddings)
        
        return projected
    
    def get_multimodal_context(
        self,
        projected_embeddings: dict[str, torch.Tensor],
    ) -> str:
        """
        Format projected embeddings as context string for the LLM.
        
        This creates textual placeholders that reference the projected embeddings,
        which can be injected into the prompt alongside the actual embeddings.
        
        Args:
            projected_embeddings: Dict from project_embeddings()
            
        Returns:
            Context string describing available multimodal inputs
        """
        parts = []
        
        if "siglip" in projected_embeddings:
            num_regions = projected_embeddings["siglip"].shape[0]
            parts.append(f"[{num_regions} visual region embeddings available]")
            
        if "videomae" in projected_embeddings:
            num_temporal = projected_embeddings["videomae"].shape[0]
            parts.append(f"[{num_temporal} temporal video embeddings available]")
            
        if "audio" in projected_embeddings:
            num_audio = projected_embeddings["audio"].shape[0]
            parts.append(f"[{num_audio} audio embeddings available]")
        
        if parts:
            return "[Multimodal Context: " + ", ".join(parts) + "]"
        return ""
    
    def get_status(self) -> dict:
        """Get current loop status."""
        return {
            "is_running": self._is_running,
            "current_timestamp": self._current_timestamp,
            "pending_query": self._pending_query,
            "temporal_context": self.temporal_context.get_context_summary(),
            "pending_triggers": len(self.trigger_detector.pending_triggers),
            "projector_weights_loaded": self.projector_weights_path is not None,
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_reasoning_core(
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    use_flash_attention: bool = True,
    max_timeline_tokens: int = 5000,
) -> QwenVLCore:
    """
    Factory function to create a configured QwenVLCore.
    
    Args:
        model_name: HuggingFace model identifier
        use_flash_attention: Enable flash attention 2
        max_timeline_tokens: Token budget for timeline context
        
    Returns:
        Configured QwenVLCore instance
    """
    config = ReasoningCoreConfig(
        model_name=model_name,
        use_flash_attention=use_flash_attention,
        max_timeline_tokens=max_timeline_tokens,
    )
    return QwenVLCore(config=config)


def create_perception_loop(
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    timeline_indexer=None,
    knowledge_base=None,
    trigger_concepts: Optional[list[str]] = None,
    projector_weights_path: Optional[str] = None,
    lora_path: Optional[str] = None,
) -> PerceptionReasoningLoop:
    """
    Factory function to create a full Perception-Reasoning Loop.
    
    Args:
        model_name: HuggingFace model identifier
        timeline_indexer: TimelineIndexer instance
        knowledge_base: KnowledgeBaseBuilder instance
        trigger_concepts: List of concepts that trigger reasoning
        projector_weights_path: Path to trained projector weights (.pt)
        lora_path: Path to LoRA adapter directory
        
    Returns:
        Configured PerceptionReasoningLoop instance
    """
    config = ReasoningCoreConfig(model_name=model_name)
    
    if trigger_concepts:
        config.trigger_concepts = trigger_concepts
    
    return PerceptionReasoningLoop(
        config=config,
        timeline_indexer=timeline_indexer,
        knowledge_base=knowledge_base,
        projector_weights_path=projector_weights_path,
        lora_path=lora_path,
    )
