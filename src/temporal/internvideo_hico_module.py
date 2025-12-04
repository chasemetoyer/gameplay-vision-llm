"""
InternVideo2.5 Hierarchical Token Compression (HiCo) Module.

This module implements the temporal context processing for long videos using
InternVideo2.5 with Hierarchical Token Compression. HiCo enables efficient
processing of videos spanning minutes to hours by:
1. Segmenting into clips
2. Extracting frame-level features
3. Compressing to clip-level tokens
4. Further compressing to video-level representations

References:
- [B: 61, B: 44] InternVideo2.5 architecture
- [B: 63, B: 64] HiCo mechanism for temporal compression
- [A: 31] Two-stage compression for rolling context
- [B: 83] Rolling compressed context for agent reasoning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class CompressionLevel(Enum):
    """Levels of hierarchical token compression."""

    FRAME = "frame"  # Raw frame-level tokens
    CLIP = "clip"  # Compressed clip-level tokens
    VIDEO = "video"  # Highly compressed video-level tokens


@dataclass
class TemporalToken:
    """Represents a compressed temporal token with metadata."""

    embedding: torch.Tensor  # Shape: (hidden_dim,)
    start_time: float  # Start timestamp in seconds
    end_time: float  # End timestamp in seconds
    compression_level: CompressionLevel
    source_frame_count: int  # Number of original frames represented
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"TemporalToken(level={self.compression_level.value}, "
            f"time=[{self.start_time:.2f}s-{self.end_time:.2f}s], "
            f"frames={self.source_frame_count})"
        )


@dataclass
class HiCoConfig:
    """Configuration for Hierarchical Token Compression."""

    # Model settings
    model_name: str = "OpenGVLab/InternVL_2_5_HiCo_R16"
    hidden_dim: int = 1408
    device: str = "cuda"
    dtype: torch.dtype = torch.float16

    # Temporal segmentation
    clip_duration_sec: float = 4.0  # Duration of each clip segment
    frames_per_clip: int = 16  # Frames to sample per clip
    clip_overlap_sec: float = 0.5  # Overlap between consecutive clips

    # Compression settings
    frame_to_clip_ratio: int = 4  # Compress 4 frame tokens -> 1 clip token
    clip_to_video_ratio: int = 8  # Compress 8 clip tokens -> 1 video token
    max_context_tokens: int = 256  # Maximum tokens in rolling context

    # Memory management
    cache_compressed_tokens: bool = True
    max_cached_clips: int = 100


class FrameEncoder(nn.Module):
    """
    Encodes individual video frames into dense feature representations.
    
    Uses InternVideo2.5 HiCo's vision encoder backbone to extract
    spatiotemporal features from raw frames. Compatible with RunPod GPU deployment.
    """

    def __init__(self, config: HiCoConfig):
        super().__init__()
        self.config = config
        self._model = None
        self._tokenizer = None

    def _load_encoder(self) -> None:
        """Lazy load the InternVideo encoder with AutoTokenizer."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading InternVideo encoder: {self.config.model_name}")
            
            # Load tokenizer for text processing
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )
            
            # Load model with trust_remote_code for custom InternVL code
            self._model = AutoModel.from_pretrained(
                self.config.model_name,
                torch_dtype=self.config.dtype,
                trust_remote_code=True,
            ).to(self.config.device)
            self._model.eval()
            
            logger.info("InternVideo encoder loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load InternVideo model: {e}")
            logger.warning("Using placeholder encoder for development")
            self._model = self._create_placeholder_encoder()
            self._tokenizer = None

    def _create_placeholder_encoder(self) -> nn.Module:
        """Create a placeholder encoder for development/testing."""

        class PlaceholderEncoder(nn.Module):
            def __init__(self, hidden_dim: int):
                super().__init__()
                self.proj = nn.Linear(3 * 224 * 224, hidden_dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                batch_size, num_frames = x.shape[:2]
                x = x.view(batch_size * num_frames, -1)
                return self.proj(x).view(batch_size, num_frames, -1)

        return PlaceholderEncoder(self.config.hidden_dim).to(self.config.device)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of video frames.

        Args:
            frames: Tensor of shape (batch, num_frames, C, H, W)

        Returns:
            Frame-level features of shape (batch, num_frames, hidden_dim)
        """
        self._load_encoder()

        with torch.no_grad():
            # Normalize frames to expected range
            if frames.max() > 1.0:
                frames = frames / 255.0

            features = self._model(frames.to(self.config.device))

            # Handle different output formats
            if isinstance(features, dict):
                features = features.get("last_hidden_state", features.get("pooler_output"))
            elif hasattr(features, "last_hidden_state"):
                features = features.last_hidden_state

            return features


class HierarchicalCompressor(nn.Module):
    """
    Implements the two-stage Hierarchical Token Compression (HiCo).
    
    Stage 1: Frame-level tokens -> Clip-level tokens
        Groups consecutive frame tokens and applies learned compression
        
    Stage 2: Clip-level tokens -> Video-level tokens
        Further compresses for long-range temporal reasoning
    """

    def __init__(self, config: HiCoConfig):
        super().__init__()
        self.config = config

        # Stage 1: Frame -> Clip compression
        self.frame_to_clip_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        self.clip_query = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
        self.clip_norm = nn.LayerNorm(config.hidden_dim)

        # Stage 2: Clip -> Video compression
        self.clip_to_video_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        self.video_query = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
        self.video_norm = nn.LayerNorm(config.hidden_dim)

        # Temporal position encoding
        self.temporal_pe = nn.Embedding(1024, config.hidden_dim)

    def compress_frames_to_clip(
        self,
        frame_tokens: torch.Tensor,
        num_output_tokens: int = 1,
    ) -> torch.Tensor:
        """
        Compress frame-level tokens to clip-level representation.

        Args:
            frame_tokens: Shape (batch, num_frames, hidden_dim)
            num_output_tokens: Number of clip tokens to produce

        Returns:
            Clip tokens of shape (batch, num_output_tokens, hidden_dim)
        """
        batch_size = frame_tokens.shape[0]

        # Add temporal position encoding
        positions = torch.arange(frame_tokens.shape[1], device=frame_tokens.device)
        frame_tokens = frame_tokens + self.temporal_pe(positions)

        # Expand query for batch
        query = self.clip_query.expand(batch_size, num_output_tokens, -1)

        # Cross-attention: query attends to all frame tokens
        compressed, _ = self.frame_to_clip_attn(
            query=query,
            key=frame_tokens,
            value=frame_tokens,
        )

        return self.clip_norm(compressed + query)

    def compress_clips_to_video(
        self,
        clip_tokens: torch.Tensor,
        num_output_tokens: int = 1,
    ) -> torch.Tensor:
        """
        Compress clip-level tokens to video-level representation.

        Args:
            clip_tokens: Shape (batch, num_clips, hidden_dim)
            num_output_tokens: Number of video tokens to produce

        Returns:
            Video tokens of shape (batch, num_output_tokens, hidden_dim)
        """
        batch_size = clip_tokens.shape[0]

        # Expand query for batch
        query = self.video_query.expand(batch_size, num_output_tokens, -1)

        # Cross-attention: query attends to all clip tokens
        compressed, _ = self.clip_to_video_attn(
            query=query,
            key=clip_tokens,
            value=clip_tokens,
        )

        return self.video_norm(compressed + query)


class InternVideoHiCoModule:
    """
    Main interface for processing long videos with InternVideo2.5 and HiCo.
    
    This module handles:
    1. Video segmentation into manageable clips
    2. Frame-level feature extraction via InternVideo2.5
    3. Hierarchical compression (HiCo) for context efficiency
    4. Rolling context management for streaming/long videos
    
    Example:
        >>> module = InternVideoHiCoModule()
        >>> 
        >>> # Process a long video
        >>> video_frames = load_video("gameplay.mp4")  # (T, H, W, C)
        >>> temporal_tokens = module.segment_and_compress(
        ...     video_frames,
        ...     fps=30.0,
        ...     target_level=CompressionLevel.CLIP
        ... )
        >>> 
        >>> # Get rolling context for LLM
        >>> context = module.get_rolling_context(max_tokens=128)
    """

    def __init__(
        self,
        config: Optional[HiCoConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the InternVideo HiCo module.

        Args:
            config: HiCo configuration. Uses defaults if not provided.
            device: Override device from config.
        """
        self.config = config or HiCoConfig()
        if device:
            self.config.device = device

        # Initialize components
        self.frame_encoder = FrameEncoder(self.config)
        self.compressor = HierarchicalCompressor(self.config)
        self.compressor.to(self.config.device)

        # Token cache for rolling context
        self._token_cache: list[TemporalToken] = []
        self._processed_duration: float = 0.0

        logger.info(
            f"InternVideoHiCoModule initialized with device={self.config.device}"
        )

    def segment_video(
        self,
        frames: NDArray[np.uint8],
        fps: float,
    ) -> list[tuple[NDArray[np.uint8], float, float]]:
        """
        Segment a video into overlapping clips.

        Args:
            frames: Video frames array of shape (T, H, W, C)
            fps: Frames per second of the source video

        Returns:
            List of (clip_frames, start_time, end_time) tuples
        """
        total_frames = len(frames)
        total_duration = total_frames / fps

        clip_frame_count = int(self.config.clip_duration_sec * fps)
        overlap_frames = int(self.config.clip_overlap_sec * fps)
        stride = clip_frame_count - overlap_frames

        clips = []
        start_frame = 0

        while start_frame < total_frames:
            end_frame = min(start_frame + clip_frame_count, total_frames)
            clip_frames = frames[start_frame:end_frame]

            start_time = start_frame / fps
            end_time = end_frame / fps

            clips.append((clip_frames, start_time, end_time))

            start_frame += stride

            # Ensure we don't create tiny trailing clips
            if total_frames - start_frame < clip_frame_count // 2:
                break

        logger.debug(
            f"Segmented {total_duration:.1f}s video into {len(clips)} clips"
        )
        return clips

    def _sample_clip_frames(
        self,
        clip_frames: NDArray[np.uint8],
    ) -> torch.Tensor:
        """
        Uniformly sample frames from a clip for encoding.

        Args:
            clip_frames: Clip frames of shape (T, H, W, C)

        Returns:
            Sampled frames tensor of shape (1, num_samples, C, H, W)
        """
        num_frames = len(clip_frames)
        num_samples = min(num_frames, self.config.frames_per_clip)

        # Uniform sampling indices
        indices = np.linspace(0, num_frames - 1, num_samples, dtype=int)
        sampled = clip_frames[indices]

        # Convert to tensor: (T, H, W, C) -> (1, T, C, H, W)
        tensor = torch.from_numpy(sampled).permute(0, 3, 1, 2).unsqueeze(0)
        return tensor.to(self.config.dtype)

    def segment_and_compress(
        self,
        frames: NDArray[np.uint8],
        fps: float,
        target_level: CompressionLevel = CompressionLevel.CLIP,
    ) -> list[TemporalToken]:
        """
        Segment a video and compress to the target representation level.
        
        This is the primary method for processing long videos. It:
        1. Segments the video into clips
        2. Extracts frame-level features for each clip
        3. Applies HiCo compression based on target_level

        Args:
            frames: Video frames array of shape (T, H, W, C)
            fps: Frames per second of the source video
            target_level: Desired compression level (FRAME, CLIP, or VIDEO)

        Returns:
            List of TemporalToken objects at the target compression level
        """
        clips = self.segment_video(frames, fps)
        tokens = []

        # Process each clip
        clip_tokens_list = []
        for clip_frames, start_time, end_time in clips:
            # Sample and encode frames
            sampled = self._sample_clip_frames(clip_frames)
            frame_features = self.frame_encoder(sampled)  # (1, T, D)

            if target_level == CompressionLevel.FRAME:
                # Return individual frame tokens
                for i in range(frame_features.shape[1]):
                    frame_time = start_time + (end_time - start_time) * i / frame_features.shape[1]
                    tokens.append(
                        TemporalToken(
                            embedding=frame_features[0, i].cpu(),
                            start_time=frame_time,
                            end_time=frame_time + (end_time - start_time) / frame_features.shape[1],
                            compression_level=CompressionLevel.FRAME,
                            source_frame_count=1,
                        )
                    )
            else:
                # Compress frames to clip token
                clip_token = self.compressor.compress_frames_to_clip(frame_features)
                clip_tokens_list.append((clip_token, start_time, end_time, len(clip_frames)))

        # Handle clip-level output
        if target_level == CompressionLevel.CLIP and clip_tokens_list:
            for clip_token, start_time, end_time, frame_count in clip_tokens_list:
                tokens.append(
                    TemporalToken(
                        embedding=clip_token[0, 0].cpu(),
                        start_time=start_time,
                        end_time=end_time,
                        compression_level=CompressionLevel.CLIP,
                        source_frame_count=frame_count,
                    )
                )

        # Handle video-level compression
        if target_level == CompressionLevel.VIDEO and clip_tokens_list:
            # Stack all clip tokens
            all_clip_tokens = torch.cat(
                [ct for ct, _, _, _ in clip_tokens_list], dim=1
            )

            # Compute number of output tokens based on ratio
            num_output = max(
                1, len(clip_tokens_list) // self.config.clip_to_video_ratio
            )

            video_tokens = self.compressor.compress_clips_to_video(
                all_clip_tokens, num_output_tokens=num_output
            )

            total_start = clip_tokens_list[0][1]
            total_end = clip_tokens_list[-1][2]
            total_frames = sum(fc for _, _, _, fc in clip_tokens_list)

            for i in range(num_output):
                tokens.append(
                    TemporalToken(
                        embedding=video_tokens[0, i].cpu(),
                        start_time=total_start,
                        end_time=total_end,
                        compression_level=CompressionLevel.VIDEO,
                        source_frame_count=total_frames,
                    )
                )

        # Cache tokens for rolling context
        if self.config.cache_compressed_tokens:
            self._token_cache.extend(tokens)
            self._processed_duration = tokens[-1].end_time if tokens else 0.0

            # Trim cache if too large
            while len(self._token_cache) > self.config.max_cached_clips:
                self._token_cache.pop(0)

        logger.info(
            f"Compressed {len(frames)} frames into {len(tokens)} "
            f"{target_level.value}-level tokens"
        )

        return tokens

    def get_rolling_context(
        self,
        max_tokens: Optional[int] = None,
        time_range: Optional[tuple[float, float]] = None,
    ) -> torch.Tensor:
        """
        Get the rolling compressed context for the LLM.
        
        This method retrieves the compressed temporal context that can be
        fed directly to the LLM for video understanding tasks.

        Args:
            max_tokens: Maximum number of tokens to return.
                       Uses config default if not specified.
            time_range: Optional (start, end) time range to filter tokens.

        Returns:
            Stacked token embeddings of shape (num_tokens, hidden_dim)
        """
        max_tokens = max_tokens or self.config.max_context_tokens
        tokens = self._token_cache

        # Filter by time range if specified
        if time_range:
            start_t, end_t = time_range
            tokens = [
                t for t in tokens
                if t.start_time >= start_t and t.end_time <= end_t
            ]

        # Limit to max tokens (keep most recent)
        if len(tokens) > max_tokens:
            tokens = tokens[-max_tokens:]

        if not tokens:
            return torch.zeros(0, self.config.hidden_dim)

        embeddings = torch.stack([t.embedding for t in tokens])
        return embeddings

    def get_context_summary(self) -> dict:
        """Get a summary of the current temporal context state."""
        if not self._token_cache:
            return {
                "num_tokens": 0,
                "duration_covered": 0.0,
                "compression_levels": [],
            }

        levels = [t.compression_level.value for t in self._token_cache]
        return {
            "num_tokens": len(self._token_cache),
            "duration_covered": self._processed_duration,
            "time_range": (
                self._token_cache[0].start_time,
                self._token_cache[-1].end_time,
            ),
            "compression_levels": list(set(levels)),
            "total_source_frames": sum(
                t.source_frame_count for t in self._token_cache
            ),
        }

    def clear_cache(self) -> None:
        """Clear the token cache."""
        self._token_cache.clear()
        self._processed_duration = 0.0
        logger.info("Token cache cleared")


def create_hico_module(
    model_name: str = "OpenGVLab/InternVL_2_5_HiCo_R16",
    device: str = "cuda",
    clip_duration: float = 4.0,
    max_context_tokens: int = 256,
) -> InternVideoHiCoModule:
    """
    Factory function to create a configured HiCo module.

    Args:
        model_name: HuggingFace model identifier
        device: Compute device ('cuda' or 'cpu')
        clip_duration: Duration of each clip segment in seconds
        max_context_tokens: Maximum tokens to keep in rolling context

    Returns:
        Configured InternVideoHiCoModule instance
    """
    config = HiCoConfig(
        model_name=model_name,
        device=device,
        clip_duration_sec=clip_duration,
        max_context_tokens=max_context_tokens,
    )
    return InternVideoHiCoModule(config=config)
