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

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch

if TYPE_CHECKING:
    from PIL import Image
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# =============================================================================
# Conversation History (Multi-turn Support)
# =============================================================================

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float  # Unix timestamp when this turn was added
    metadata: dict = field(default_factory=dict)  # Optional metadata (e.g., video timestamp, confidence)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationTurn":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


class ConversationHistory:
    """
    Manages multi-turn conversation history for contextual Q&A.

    Features:
    - Maintains rolling window of conversation turns
    - Supports context summarization for long conversations
    - Enables follow-up questions like "What happened next?"
    - Tracks video timestamps mentioned in conversation

    Example:
        >>> history = ConversationHistory(max_turns=20)
        >>> history.add_user_message("What happened at 5:30?")
        >>> history.add_assistant_message("The player defeated the boss.")
        >>> history.add_user_message("What happened next?")
        >>> context = history.get_context_for_prompt()
    """

    def __init__(
        self,
        max_turns: int = 20,
        max_tokens_estimate: int = 4000,
        summarize_after: int = 15,
    ):
        """
        Initialize conversation history.

        Args:
            max_turns: Maximum number of turns to keep in full detail
            max_tokens_estimate: Approximate token budget for history
            summarize_after: Start summarizing older turns after this many
        """
        self.max_turns = max_turns
        self.max_tokens_estimate = max_tokens_estimate
        self.summarize_after = summarize_after

        self._turns: list[ConversationTurn] = []
        self._summary: Optional[str] = None  # Summary of older conversation
        self._video_timestamps_mentioned: list[float] = []  # Track timestamps discussed
        self._session_start: float = time.time()

    def add_user_message(
        self,
        content: str,
        video_timestamp: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Add a user message to the conversation.

        Args:
            content: The user's message/question
            video_timestamp: Optional video timestamp being discussed
            metadata: Optional additional metadata
        """
        meta = metadata or {}
        if video_timestamp is not None:
            meta["video_timestamp"] = video_timestamp
            self._video_timestamps_mentioned.append(video_timestamp)

        turn = ConversationTurn(
            role="user",
            content=content,
            timestamp=time.time(),
            metadata=meta,
        )
        self._turns.append(turn)
        self._maybe_truncate()

    def add_assistant_message(
        self,
        content: str,
        confidence: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Add an assistant response to the conversation.

        Args:
            content: The assistant's response
            confidence: Optional confidence score (0-1)
            metadata: Optional additional metadata
        """
        meta = metadata or {}
        if confidence is not None:
            meta["confidence"] = confidence

        turn = ConversationTurn(
            role="assistant",
            content=content,
            timestamp=time.time(),
            metadata=meta,
        )
        self._turns.append(turn)
        self._maybe_truncate()

    def _maybe_truncate(self) -> None:
        """Truncate history if it exceeds limits."""
        if len(self._turns) > self.max_turns:
            # Keep the most recent turns, summarize older ones
            overflow = len(self._turns) - self.max_turns
            old_turns = self._turns[:overflow]
            self._turns = self._turns[overflow:]

            # Create summary of old turns
            old_summary = self._summarize_turns(old_turns)
            if self._summary:
                self._summary = f"{self._summary}\n{old_summary}"
            else:
                self._summary = old_summary

    def _summarize_turns(self, turns: list[ConversationTurn]) -> str:
        """Create a brief summary of conversation turns."""
        if not turns:
            return ""

        summaries = []
        for turn in turns:
            role = "User" if turn.role == "user" else "Assistant"
            # Truncate long content
            content = turn.content[:100] + "..." if len(turn.content) > 100 else turn.content
            summaries.append(f"{role}: {content}")

        return "[Earlier conversation summary: " + " | ".join(summaries) + "]"

    def get_context_for_prompt(
        self,
        include_summary: bool = True,
        max_recent_turns: Optional[int] = None,
    ) -> str:
        """
        Get conversation history formatted for inclusion in prompt.

        Args:
            include_summary: Whether to include summary of older turns
            max_recent_turns: Limit to N most recent turns (None = all)

        Returns:
            Formatted conversation history string
        """
        parts = []

        # Include summary of older conversation if available
        if include_summary and self._summary:
            parts.append(self._summary)

        # Get recent turns
        turns = self._turns
        if max_recent_turns is not None:
            turns = turns[-max_recent_turns:]

        # Format turns
        for turn in turns:
            role_label = "User" if turn.role == "user" else "Assistant"
            parts.append(f"{role_label}: {turn.content}")

        return "\n".join(parts)

    def get_messages_for_chat(self) -> list[dict]:
        """
        Get conversation history as chat messages for apply_chat_template.

        Returns:
            List of message dicts with 'role' and 'content'
        """
        messages = []
        for turn in self._turns:
            messages.append({
                "role": turn.role,
                "content": [{"type": "text", "text": turn.content}],
            })
        return messages

    def get_last_user_query(self) -> Optional[str]:
        """Get the most recent user query."""
        for turn in reversed(self._turns):
            if turn.role == "user":
                return turn.content
        return None

    def get_last_assistant_response(self) -> Optional[str]:
        """Get the most recent assistant response."""
        for turn in reversed(self._turns):
            if turn.role == "assistant":
                return turn.content
        return None

    def get_mentioned_timestamps(self) -> list[float]:
        """Get all video timestamps mentioned in conversation."""
        return self._video_timestamps_mentioned.copy()

    def get_last_mentioned_timestamp(self) -> Optional[float]:
        """Get the most recently mentioned video timestamp."""
        if self._video_timestamps_mentioned:
            return self._video_timestamps_mentioned[-1]
        return None

    def clear(self) -> None:
        """Clear all conversation history."""
        self._turns.clear()
        self._summary = None
        self._video_timestamps_mentioned.clear()
        self._session_start = time.time()

    def get_turn_count(self) -> int:
        """Get the number of turns in current history."""
        return len(self._turns)

    def is_follow_up_query(self, query: str) -> bool:
        """
        Detect if query is a follow-up that needs conversation context.

        Args:
            query: The user's query

        Returns:
            True if this appears to be a follow-up question
        """
        follow_up_patterns = [
            r'\bwhat happened next\b',
            r'\bwhat about\b',
            r'\band then\b',
            r'\bafter that\b',
            r'\bbefore that\b',
            r'\bwhy did (he|she|they|it|the player)\b',
            r'\bwhat did (he|she|they|it|the player) do\b',
            r'\bcan you explain more\b',
            r'\btell me more\b',
            r'\bwhat else\b',
            r'\bhow did that happen\b',
            r'\bwhy\?$',
            r'^why\b',
            r'^how\b',
            r'^what\b.*\bthat\b',
            r'\bthe same\b',
            r'\bit\b.*\?$',  # Questions ending with "it?"
        ]

        query_lower = query.lower().strip()
        for pattern in follow_up_patterns:
            if re.search(pattern, query_lower):
                return True

        return False

    def to_dict(self) -> dict:
        """Serialize conversation history to dictionary."""
        return {
            "turns": [t.to_dict() for t in self._turns],
            "summary": self._summary,
            "video_timestamps_mentioned": self._video_timestamps_mentioned,
            "session_start": self._session_start,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationHistory":
        """Deserialize conversation history from dictionary."""
        history = cls()
        history._turns = [ConversationTurn.from_dict(t) for t in data.get("turns", [])]
        history._summary = data.get("summary")
        history._video_timestamps_mentioned = data.get("video_timestamps_mentioned", [])
        history._session_start = data.get("session_start", time.time())
        return history

    def save(self, path: str) -> None:
        """Save conversation history to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved conversation history to {path}")

    @classmethod
    def load(cls, path: str) -> "ConversationHistory":
        """Load conversation history from file."""
        with open(path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded conversation history from {path}")
        return cls.from_dict(data)


# =============================================================================
# Feature Cache (Avoid Reprocessing)
# =============================================================================

class FeatureCache:
    """
    Caches extracted features to avoid reprocessing the same video.

    Features:
    - Caches SigLIP, VideoMAE, Wav2Vec2 embeddings
    - Caches SAM detections, OCR results, speech transcriptions
    - Uses content hash to detect if video has changed
    - Supports disk persistence for cross-session caching

    Example:
        >>> cache = FeatureCache(cache_dir="data/cache")
        >>>
        >>> # Check if features exist
        >>> if cache.has_features(video_path):
        ...     features = cache.load_features(video_path)
        ... else:
        ...     features = extract_features(video_path)
        ...     cache.save_features(video_path, features)
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        max_cache_size_gb: float = 10.0,
    ):
        """
        Initialize feature cache.

        Args:
            cache_dir: Directory for cached features
            max_cache_size_gb: Maximum cache size in GB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size_bytes = int(max_cache_size_gb * 1024 * 1024 * 1024)

        # In-memory cache for current session
        self._memory_cache: dict[str, dict] = {}

        logger.info(f"FeatureCache initialized: {cache_dir}")

    def _get_video_hash(self, video_path: str) -> str:
        """
        Generate a hash for the video file.

        Uses file size + first/last 1MB for fast hashing.
        """
        path = Path(video_path)
        if not path.exists():
            return ""

        file_size = path.stat().st_size

        # Read first and last 1MB for hashing
        chunk_size = min(1024 * 1024, file_size)

        hasher = hashlib.sha256()
        hasher.update(str(file_size).encode())

        with open(path, "rb") as f:
            hasher.update(f.read(chunk_size))
            if file_size > chunk_size * 2:
                f.seek(-chunk_size, 2)
                hasher.update(f.read(chunk_size))

        return hasher.hexdigest()[:16]

    def _get_cache_path(self, video_path: str) -> Path:
        """Get cache file path for a video."""
        video_hash = self._get_video_hash(video_path)
        video_name = Path(video_path).stem
        return self.cache_dir / f"{video_name}_{video_hash}.pt"

    def has_features(self, video_path: str) -> bool:
        """Check if features are cached for this video."""
        # Check memory cache first
        cache_key = self._get_video_hash(video_path)
        if cache_key in self._memory_cache:
            return True

        # Check disk cache
        cache_path = self._get_cache_path(video_path)
        return cache_path.exists()

    def load_features(self, video_path: str) -> Optional[dict]:
        """
        Load cached features for a video.

        Returns:
            Dictionary of features or None if not cached
        """
        cache_key = self._get_video_hash(video_path)

        # Check memory cache
        if cache_key in self._memory_cache:
            logger.info(f"Loaded features from memory cache")
            return self._memory_cache[cache_key]

        # Check disk cache
        cache_path = self._get_cache_path(video_path)
        if cache_path.exists():
            try:
                features = torch.load(cache_path, map_location="cpu", weights_only=False)
                self._memory_cache[cache_key] = features
                logger.info(f"Loaded features from disk cache: {cache_path}")
                return features
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                cache_path.unlink(missing_ok=True)

        return None

    def save_features(self, video_path: str, features: dict) -> None:
        """
        Save features to cache.

        Args:
            video_path: Path to the video file
            features: Dictionary of extracted features
        """
        cache_key = self._get_video_hash(video_path)
        cache_path = self._get_cache_path(video_path)

        # Save to memory
        self._memory_cache[cache_key] = features

        # Save to disk
        try:
            # Ensure we don't exceed cache size
            self._cleanup_old_cache()

            torch.save(features, cache_path)
            logger.info(f"Saved features to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _cleanup_old_cache(self) -> None:
        """Remove old cache files if exceeding size limit."""
        cache_files = list(self.cache_dir.glob("*.pt"))
        if not cache_files:
            return

        # Calculate total size
        total_size = sum(f.stat().st_size for f in cache_files)

        if total_size > self.max_cache_size_bytes:
            # Sort by modification time (oldest first)
            cache_files.sort(key=lambda f: f.stat().st_mtime)

            # Remove oldest until under limit
            while total_size > self.max_cache_size_bytes * 0.8 and cache_files:
                oldest = cache_files.pop(0)
                total_size -= oldest.stat().st_size
                oldest.unlink()
                logger.info(f"Removed old cache: {oldest}")

    def clear(self) -> None:
        """Clear all cached features."""
        self._memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.pt"):
            cache_file.unlink()
        logger.info("Cleared all feature caches")

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.pt"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "num_cached_videos": len(cache_files),
            "total_size_mb": total_size / (1024 * 1024),
            "memory_cache_entries": len(self._memory_cache),
        }


# =============================================================================
# Response Confidence Scorer
# =============================================================================

class ConfidenceScorer:
    """
    Estimates confidence scores for model responses.

    Factors considered:
    - Amount of relevant context available
    - Specificity of the question
    - Presence of timestamp citations in response
    - Token probability statistics (if available)

    Example:
        >>> scorer = ConfidenceScorer()
        >>> confidence = scorer.score_response(
        ...     query="What happened at 5:30?",
        ...     response="The player defeated the boss at [05:30].",
        ...     context_events=events,
        ... )
        >>> print(f"Confidence: {confidence:.2f}")
    """

    def __init__(self):
        self.min_events_for_high_confidence = 3
        self.timestamp_citation_pattern = re.compile(r'\[?\d{1,2}:\d{2}\]?')

    def score_response(
        self,
        query: str,
        response: str,
        context_events: Optional[list] = None,
        token_probs: Optional[list[float]] = None,
    ) -> float:
        """
        Score the confidence of a response.

        Args:
            query: The user's question
            response: The model's response
            context_events: Timeline events used for context
            token_probs: Optional token probabilities from generation

        Returns:
            Confidence score between 0.0 and 1.0
        """
        scores = []

        # Factor 1: Context availability (0.0 - 0.3)
        if context_events:
            num_events = len(context_events)
            if num_events >= self.min_events_for_high_confidence:
                scores.append(0.3)
            else:
                scores.append(0.1 * num_events)
        else:
            scores.append(0.0)

        # Factor 2: Response contains timestamp citations (0.0 - 0.25)
        citations = self.timestamp_citation_pattern.findall(response)
        if citations:
            scores.append(min(0.25, 0.05 * len(citations)))
        else:
            scores.append(0.0)

        # Factor 3: Response length and structure (0.0 - 0.25)
        if len(response) > 50:
            # Check for reasoning structure
            has_reasoning = "**Reasoning:**" in response or "because" in response.lower()
            has_answer = "**Answer:**" in response

            if has_reasoning and has_answer:
                scores.append(0.25)
            elif has_reasoning or has_answer:
                scores.append(0.15)
            else:
                scores.append(0.1)
        else:
            scores.append(0.05)

        # Factor 4: Query specificity match (0.0 - 0.2)
        query_has_timestamp = bool(re.search(r'\d{1,2}:\d{2}', query))
        response_has_timestamp = bool(citations)

        if query_has_timestamp and response_has_timestamp:
            scores.append(0.2)
        elif not query_has_timestamp:
            scores.append(0.15)  # General questions get baseline
        else:
            scores.append(0.05)  # Asked about time but no citation

        # Combine scores
        total_confidence = sum(scores)

        # Apply token probability adjustment if available
        if token_probs:
            avg_prob = sum(token_probs) / len(token_probs)
            total_confidence *= (0.5 + 0.5 * avg_prob)

        return min(1.0, max(0.0, total_confidence))


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

    # System prompt with tool calling support
    system_prompt: str = """You are an expert gameplay analyst. You have access to:
1. A complete timeline of events extracted from the entire video (OCR text, speech transcription, visual detections)
2. Representative video frames for visual grounding
3. An entity knowledge base tracking game objects across the video

IMPORTANT: You are analyzing the FULL VIDEO, not just a single frame. Use the timeline context to understand what happened throughout the entire video.

## Available Tools

You have access to the following tool for looking up external game information:

**search_web(query: str)** - Search the web for game-related information including:
- Boss strategies and weaknesses
- Game mechanics and lore
- Character abilities and stats
- Item locations and effects

To use a tool, output a tool call in this EXACT format (on its own line):
<tool_call>search_web("your search query here")</tool_call>

When to use tools:
- When asked about game-specific knowledge not visible in the video (e.g., "what is this boss weak to?")
- When the user asks about strategies, lore, or mechanics
- When you need additional context to provide a complete answer

After receiving tool results, incorporate them into your final answer.

## Response Format

For the FIRST question about a video, provide full reasoning:
**Reasoning:**
[Brief analysis of key events - keep this concise, max 3-4 sentences]

**Answer:**
[Your answer with timestamp citations]

For FOLLOW-UP questions:
- DO NOT repeat context you've already provided
- DO NOT re-describe the video or boss fight
- Focus ONLY on answering the new question directly
- Keep your answer brief and specific

If you need to cite timestamps, use format [MM:SS]."""


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
# Tool Call Parser (Autonomous Tool Calling)
# =============================================================================

@dataclass
class ToolCall:
    """Represents a parsed tool call from model output."""
    tool_name: str
    arguments: str
    raw_match: str


class ToolCallParser:
    """
    Parses and executes tool calls from model output.
    
    Enables ChatGPT-like autonomous tool calling where the model
    can decide to search for information when needed.
    
    Example model output:
        I need more information about this boss.
        <tool_call>search_web("Tyronoe the Ferryman weakness")</tool_call>
    """
    
    # Pattern to match tool calls: <tool_call>tool_name("args")</tool_call>
    TOOL_CALL_PATTERN = re.compile(
        r'<tool_call>\s*(\w+)\s*\(\s*["\'](.+?)["\']\s*\)\s*</tool_call>',
        re.IGNORECASE | re.DOTALL
    )
    
    def __init__(self, knowledge_searcher=None):
        """
        Initialize tool call parser.
        
        Args:
            knowledge_searcher: GameKnowledgeSearch instance for web lookups
        """
        self._knowledge_searcher = knowledge_searcher
        self._available_tools = {"search_web"}
    
    def parse_tool_calls(self, text: str) -> list[ToolCall]:
        """
        Parse tool calls from model output.
        
        Args:
            text: Model output text
            
        Returns:
            List of ToolCall objects found in the text
        """
        tool_calls = []
        
        for match in self.TOOL_CALL_PATTERN.finditer(text):
            tool_name = match.group(1).lower()
            arguments = match.group(2)
            
            if tool_name in self._available_tools:
                tool_calls.append(ToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                    raw_match=match.group(0),
                ))
        
        return tool_calls
    
    def has_tool_calls(self, text: str) -> bool:
        """Check if text contains any tool calls."""
        return bool(self.TOOL_CALL_PATTERN.search(text))
    
    def execute_tool(self, tool_call: ToolCall) -> str:
        """
        Execute a single tool call.
        
        Args:
            tool_call: The tool call to execute
            
        Returns:
            Tool execution result as formatted string
        """
        logger.info(f"Executing tool: {tool_call.tool_name}({tool_call.arguments})")
        
        if tool_call.tool_name == "search_web":
            return self._execute_search(tool_call.arguments)
        else:
            return f"[Unknown tool: {tool_call.tool_name}]"
    
    def _execute_search(self, query: str) -> str:
        """Execute web search and format results."""
        if not self._knowledge_searcher:
            # Fallback to direct DuckDuckGo search
            try:
                from duckduckgo_search import DDGS
                
                results = []
                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=5):
                        title = r.get("title", "")
                        body = r.get("body", "")[:200]
                        results.append(f"- **{title}**: {body}")
                
                if results:
                    return "## Web Search Results\n\n" + "\n".join(results)
                else:
                    return "[No search results found]"
                    
            except Exception as e:
                logger.warning(f"Direct search failed: {e}")
                return f"[Search failed: {e}]"
        else:
            # Use the knowledge searcher
            return self._knowledge_searcher.execute_tool_call(query, "general")
    
    def execute_all_tools(self, text: str) -> tuple[str, list[str]]:
        """
        Parse and execute all tool calls in text.
        
        Args:
            text: Model output with potential tool calls
            
        Returns:
            Tuple of (text_with_tools_removed, list_of_tool_results)
        """
        tool_calls = self.parse_tool_calls(text)
        
        if not tool_calls:
            return text, []
        
        results = []
        cleaned_text = text
        
        for tc in tool_calls:
            # Execute the tool
            result = self.execute_tool(tc)
            results.append(result)
            
            # Remove the tool call from text
            cleaned_text = cleaned_text.replace(tc.raw_match, "")
        
        return cleaned_text.strip(), results
    
    def format_tool_results_for_prompt(self, results: list[str]) -> str:
        """Format tool results for inclusion in follow-up prompt."""
        if not results:
            return ""
        
        formatted = "\n\n## Tool Results\n\n"
        for i, result in enumerate(results, 1):
            if len(results) > 1:
                formatted += f"### Result {i}\n{result}\n\n"
            else:
                formatted += result + "\n"
        
        formatted += "\nNow use these results to provide a complete answer to the user's question.\n"
        return formatted


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
    - ConversationHistory for multi-turn support
    - ConfidenceScorer for response confidence estimation

    Example:
        >>> core = QwenVLCore()
        >>> response, confidence = core.reason(
        ...     query="What happened at 5:30?",
        ...     current_frame=frame_image,
        ...     timeline_indexer=timeline,
        ... )
        >>> # Follow-up question uses conversation context
        >>> response, confidence = core.reason(
        ...     query="What happened next?",
        ...     current_frame=frame_image,
        ...     timeline_indexer=timeline,
        ... )
    """

    def __init__(
        self,
        config: Optional[ReasoningCoreConfig] = None,
        lora_path: Optional[str] = None,
        conversation_history: Optional[ConversationHistory] = None,
        enable_web_search: bool = True,
        game_name: Optional[str] = None,
    ):
        self.config = config or ReasoningCoreConfig()
        self.lora_path = lora_path
        self.retriever = TimelineRetriever(self.config)
        self.visual_processor = VisualInputProcessor(self.config)

        # Multi-turn conversation support
        self.conversation_history = conversation_history or ConversationHistory()

        # Confidence scoring
        self.confidence_scorer = ConfidenceScorer()

        # Web search for game knowledge
        self.enable_web_search = enable_web_search
        self._knowledge_searcher = None
        self._game_detector = None
        
        # Tool call parser for autonomous tool calling (must be before _init_search_tools)
        self._tool_parser: Optional[ToolCallParser] = None
        
        if enable_web_search:
            self._init_search_tools(game_name)

        self._model = None
        self._processor = None

        # Track last retrieved events for confidence scoring
        self._last_retrieved_events: list = []

        # Track search results for context
        self._last_search_results: Optional[str] = None

        logger.info("QwenVLCore initialized with multi-turn conversation and web search support")

    def _init_search_tools(self, game_name: Optional[str] = None) -> None:
        """Initialize web search tools and tool call parser."""
        try:
            from agent_core.game_knowledge_search import (
                GameKnowledgeSearcher,
                GameDetector,
            )

            self._knowledge_searcher = GameKnowledgeSearcher()
            self._game_detector = GameDetector()
            
            # Initialize tool call parser with knowledge searcher
            self._tool_parser = ToolCallParser(knowledge_searcher=self._knowledge_searcher)

            if game_name:
                self._knowledge_searcher.set_game_context(game_name)
                logger.info(f"Game context set: {game_name}")

        except ImportError as e:
            logger.warning(f"Could not initialize search tools: {e}")
            self.enable_web_search = False
            # Still create tool parser without knowledge searcher (uses direct DuckDuckGo)
            self._tool_parser = ToolCallParser()

    def set_game_context(self, game_name: str, genre: Optional[str] = None) -> None:
        """Set the current game being analyzed for better search results."""
        if self._knowledge_searcher:
            self._knowledge_searcher.set_game_context(game_name, genre)
            logger.info(f"Game context updated: {game_name}")

    def detect_game_from_content(
        self,
        ocr_results: Optional[list[dict]] = None,
        speech_results: Optional[list[dict]] = None,
    ) -> Optional[str]:
        """
        Auto-detect the game from video content.

        Args:
            ocr_results: OCR text detections from video
            speech_results: Speech transcriptions from video

        Returns:
            Detected game name or None
        """
        if not self._game_detector:
            return None

        detected = None

        if ocr_results:
            context = self._game_detector.detect_from_ocr(ocr_results)
            if context:
                detected = context.game_name

        if not detected and speech_results:
            context = self._game_detector.detect_from_speech(speech_results)
            if context:
                detected = context.game_name

        if detected and self._knowledge_searcher:
            self._knowledge_searcher.set_game_context(detected)

        return detected

    def search_game_knowledge(
        self,
        query: str,
        search_type: str = "general",
    ) -> str:
        """
        Search for game-related information.

        Args:
            query: What to search for
            search_type: Type of search - "general", "wiki", "guide", "lore"

        Returns:
            Formatted search results
        """
        if not self._knowledge_searcher:
            return "Web search is not available."

        return self._knowledge_searcher.execute_tool_call(query, search_type)

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
        include_conversation_history: bool = True,
        web_search_results: Optional[str] = None,
    ) -> list[dict]:
        """
        Construct the message list for Qwen3-VL.

        Args:
            query: User question
            timeline_context: Retrieved timeline events formatted as text
            current_frame: Optional current video frame
            region_tokens: Optional region descriptions from SAM/SigLIP
            knowledge_base_context: Optional KB export
            include_conversation_history: Whether to include prior conversation
            web_search_results: Optional web search results for game knowledge

        Returns:
            List of message dicts for apply_chat_template
        """
        messages = []

        # System message with conversation and search awareness
        system_text = self.config.system_prompt

        if include_conversation_history and self.conversation_history.get_turn_count() > 0:
            system_text += "\n\nYou have access to the conversation history. Use it to understand follow-up questions and maintain context."

        # Add game context if available
        if self._knowledge_searcher and self._knowledge_searcher.game_context:
            game_ctx = self._knowledge_searcher.game_context
            if game_ctx.game_name:
                system_text += f"\n\nCurrent game: **{game_ctx.game_name}**"
                if game_ctx.game_genre:
                    system_text += f" ({game_ctx.game_genre})"

        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        })

        # Add conversation history as prior messages
        if include_conversation_history:
            history_messages = self.conversation_history.get_messages_for_chat()
            messages.extend(history_messages)

        # Build user content for current query
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

        # Add web search results if available
        if web_search_results:
            context_parts.append(web_search_results)

        # Add conversation history summary if it's a follow-up question
        if include_conversation_history and self.conversation_history.is_follow_up_query(query):
            conv_context = self.conversation_history.get_context_for_prompt(max_recent_turns=4)
            if conv_context:
                context_parts.append("## Recent Conversation\n" + conv_context)

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
        video_timestamp: Optional[float] = None,
        track_conversation: bool = True,
        return_confidence: bool = False,
    ) -> Union[str, tuple[str, float]]:
        """
        Execute the full Perception-Reasoning Loop with multi-turn support.

        This is the main entry point for the agent. Automatically tracks
        conversation history for follow-up questions.

        Args:
            query: User question
            current_frame: Optional current video frame for visual grounding
            timeline_indexer: TimelineIndexer with indexed events
            knowledge_base: Optional KnowledgeBaseBuilder for entity context
            region_detections: Optional SAM/SigLIP region data
            video_timestamp: Optional timestamp of current frame
            track_conversation: Whether to track this exchange in history
            return_confidence: Whether to return confidence score

        Returns:
            If return_confidence is False: Model response string
            If return_confidence is True: Tuple of (response, confidence_score)
        """
        self._load_model()

        if self._processor is None:
            error_msg = "[Error: Qwen3-VL model not loaded]"
            return (error_msg, 0.0) if return_confidence else error_msg

        # Step 1: Handle follow-up questions
        is_follow_up = self.conversation_history.is_follow_up_query(query)
        if is_follow_up:
            logger.info("Detected follow-up question, using conversation context")
            # For follow-ups like "what happened next?", use last mentioned timestamp
            if video_timestamp is None:
                video_timestamp = self.conversation_history.get_last_mentioned_timestamp()

        # Step 2: Retrieve relevant context
        logger.info(f"Processing query: {query[:50]}...")

        if timeline_indexer is not None:
            self.retriever.index_timeline(timeline_indexer)

        events = self.retriever.hybrid_retrieve(query, timeline_indexer)
        self._last_retrieved_events = events  # Store for confidence scoring
        timeline_context = self.format_timeline_context(events)
        logger.info(f"Retrieved {len(events)} events")

        # Step 3: Process visual input
        region_tokens = None
        if region_detections:
            region_tokens = self.visual_processor.process_region_tokens(
                region_detections
            )

        # Step 4: Get knowledge base context
        kb_context = None
        if knowledge_base is not None:
            kb_context = knowledge_base.export_for_llm(
                max_entities=15,
                max_relationships=20,
            )

        # Step 5: Build prompt (with conversation history)
        messages = self.build_prompt(
            query=query,
            timeline_context=timeline_context,
            current_frame=current_frame,
            region_tokens=region_tokens,
            knowledge_base_context=kb_context,
            include_conversation_history=track_conversation,
        )

        # Step 6: Prepare inputs
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
            error_msg = f"[Error preparing inputs: {e}]"
            return (error_msg, 0.0) if return_confidence else error_msg

        # Step 7: Generate response
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

            response = response.strip()

            # Step 8: Track conversation
            if track_conversation:
                self.conversation_history.add_user_message(
                    query,
                    video_timestamp=video_timestamp,
                )

            # Step 9: Calculate confidence score
            confidence = self.confidence_scorer.score_response(
                query=query,
                response=response,
                context_events=self._last_retrieved_events,
            )

            # Step 10: Track assistant response with confidence
            if track_conversation:
                self.conversation_history.add_assistant_message(
                    response,
                    confidence=confidence,
                )

            logger.info(f"Response generated (confidence: {confidence:.2f})")

            if return_confidence:
                return response, confidence
            return response

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"[Error during generation: {e}]"

    def clear_conversation(self) -> None:
        """Clear conversation history to start a fresh session."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")

    def save_conversation(self, path: str) -> None:
        """Save conversation history to file."""
        self.conversation_history.save(path)

    def load_conversation(self, path: str) -> None:
        """Load conversation history from file."""
        self.conversation_history = ConversationHistory.load(path)

    def get_conversation_summary(self) -> dict:
        """Get a summary of the current conversation state."""
        return {
            "turn_count": self.conversation_history.get_turn_count(),
            "timestamps_mentioned": self.conversation_history.get_mentioned_timestamps(),
            "last_query": self.conversation_history.get_last_user_query(),
            "last_response": self.conversation_history.get_last_assistant_response()[:100] + "..."
                if self.conversation_history.get_last_assistant_response() else None,
        }

    def reason_streaming(
        self,
        query: str,
        current_frame: Optional["Image.Image"] = None,
        timeline_indexer=None,
        knowledge_base=None,
        region_detections: Optional[list[dict]] = None,
        video_timestamp: Optional[float] = None,
        track_conversation: bool = True,
    ):
        """
        Execute reasoning with streaming output (yields tokens as generated).

        Same as reason() but yields each token as it's generated for
        real-time display like ChatGPT. Also tracks conversation history.

        Args:
            query: User question
            current_frame: Optional current video frame
            timeline_indexer: TimelineIndexer with indexed events
            knowledge_base: Optional KnowledgeBaseBuilder
            region_detections: Optional SAM/SigLIP region data
            video_timestamp: Optional timestamp of current frame
            track_conversation: Whether to track this exchange in history

        Yields:
            str: Each new token/chunk of text as it's generated
        """
        self._load_model()

        if self._processor is None:
            yield "[Error: Qwen3-VL model not loaded]"
            return

        # Handle follow-up questions
        is_follow_up = self.conversation_history.is_follow_up_query(query)
        if is_follow_up:
            logger.info("Detected follow-up question, using conversation context")
            if video_timestamp is None:
                video_timestamp = self.conversation_history.get_last_mentioned_timestamp()

        # Prepare inputs
        logger.info(f"Processing query (streaming): {query[:50]}...")

        if timeline_indexer is not None:
            self.retriever.index_timeline(timeline_indexer)

        events = self.retriever.hybrid_retrieve(query, timeline_indexer)
        self._last_retrieved_events = events
        timeline_context = self.format_timeline_context(events)

        region_tokens = None
        if region_detections:
            region_tokens = self.visual_processor.process_region_tokens(
                region_detections
            )

        kb_context = None
        if knowledge_base is not None:
            kb_context = knowledge_base.export_for_llm(
                max_entities=15,
                max_relationships=20,
            )

        messages = self.build_prompt(
            query=query,
            timeline_context=timeline_context,
            current_frame=current_frame,
            region_tokens=region_tokens,
            knowledge_base_context=kb_context,
            include_conversation_history=track_conversation,
        )

        # Track user message before streaming
        if track_conversation:
            self.conversation_history.add_user_message(
                query,
                video_timestamp=video_timestamp,
            )

        # Prepare inputs
        try:
            has_image = current_frame is not None

            if has_image:
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
                    inputs = self._processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
            else:
                inputs = self._processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )

            inputs = inputs.to(self._model.device)

        except Exception as e:
            yield f"[Error preparing inputs: {e}]"
            return

        # Generate with streaming
        try:
            from transformers import TextIteratorStreamer
            from threading import Thread

            # Create streamer
            streamer = TextIteratorStreamer(
                self._processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            # Generation kwargs
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=True,
                streamer=streamer,
            )

            # Run generation in a thread
            thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
            thread.start()

            # Collect full response while yielding tokens
            full_response = []
            for token in streamer:
                full_response.append(token)
                yield token

            thread.join()
            
            response_text = "".join(full_response).strip()
            
            # Check for tool calls in response
            if self._tool_parser and self._tool_parser.has_tool_calls(response_text):
                logger.info("Tool call detected in response, executing...")
                yield "\n\n🔍 *Searching for additional information...*\n\n"
                
                # Execute tool calls
                cleaned_text, tool_results = self._tool_parser.execute_all_tools(response_text)
                
                # Format tool results
                tool_context = self._tool_parser.format_tool_results_for_prompt(tool_results)
                
                # Show search results to user
                for result in tool_results:
                    yield f"{result}\n\n"
                
                yield "---\n\n📝 *Generating complete answer with search results...*\n\n"
                
                # Build follow-up prompt with tool results
                followup_messages = self.build_prompt(
                    query=f"{query}\n\n{tool_context}",
                    timeline_context=timeline_context,
                    current_frame=current_frame,
                    region_tokens=region_tokens,
                    knowledge_base_context=kb_context,
                    include_conversation_history=False,  # Already included context
                )
                
                # Generate follow-up response with tool results
                try:
                    if has_image:
                        try:
                            from qwen_vl_utils import process_vision_info
                            followup_text = self._processor.apply_chat_template(
                                followup_messages,
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                            images, videos, _ = process_vision_info(
                                followup_messages,
                                image_patch_size=16,
                                return_video_kwargs=True,
                            )
                            followup_inputs = self._processor(
                                text=followup_text,
                                images=images,
                                videos=videos,
                                do_resize=False,
                                return_tensors="pt",
                            )
                        except ImportError:
                            followup_inputs = self._processor.apply_chat_template(
                                followup_messages,
                                tokenize=True,
                                add_generation_prompt=True,
                                return_dict=True,
                                return_tensors="pt",
                            )
                    else:
                        followup_inputs = self._processor.apply_chat_template(
                            followup_messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_dict=True,
                            return_tensors="pt",
                        )
                    
                    followup_inputs = followup_inputs.to(self._model.device)
                    
                    # Create new streamer for follow-up
                    followup_streamer = TextIteratorStreamer(
                        self._processor.tokenizer,
                        skip_prompt=True,
                        skip_special_tokens=True,
                    )
                    
                    followup_kwargs = dict(
                        **followup_inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        top_k=self.config.top_k,
                        do_sample=True,
                        streamer=followup_streamer,
                    )
                    
                    followup_thread = Thread(target=self._model.generate, kwargs=followup_kwargs)
                    followup_thread.start()
                    
                    followup_response = []
                    for token in followup_streamer:
                        followup_response.append(token)
                        yield token
                    
                    followup_thread.join()
                    
                    # Use the follow-up response as final
                    response_text = "".join(followup_response).strip()
                    
                except Exception as e:
                    logger.warning(f"Follow-up generation failed: {e}")
                    # Fall back to original response without tool call tags
                    response_text = cleaned_text

            # Track assistant response after streaming completes
            if track_conversation:
                confidence = self.confidence_scorer.score_response(
                    query=query,
                    response=response_text,
                    context_events=self._last_retrieved_events,
                )
                self.conversation_history.add_assistant_message(
                    response_text,
                    confidence=confidence,
                )
                logger.info(f"Streaming response tracked (confidence: {confidence:.2f})")

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"[Error during generation: {e}]"


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
