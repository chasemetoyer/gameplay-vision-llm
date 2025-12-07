"""
MLX-based Qwen3-VL Reasoning Core for Apple Silicon.

This module provides an alternative backend using Apple's MLX framework
for efficient inference on M1/M2/M3/M4 Macs with unified memory.

Requires:
    pip install mlx mlx-vlm

For Mac M4 Pro with 24GB, uses 4-bit quantized models:
    - Qwen3-VL-8B: ~4-5GB (4-bit)
    - Total pipeline: ~13-14GB
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Generator
from threading import Thread

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class MLXConfig:
    """Configuration for MLX-based inference."""
    
    # Model settings - use Qwen3-VL for consistency with main branch
    # Available options:
    #   - "mlx-community/Qwen3-VL-8B-Instruct-4bit" (if available)
    #   - "lmstudio-community/Qwen3-VL-8B-Instruct-4bit-MLX"
    #   - Fallback: "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
    model_name: str = "mlx-community/Qwen3-VL-8B-Instruct-4bit"
    
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    # System prompt
    system_prompt: str = """You are an expert gameplay analyst. You have access to:
1. A timeline of events extracted from the video (OCR, speech, visual detections)
2. The current video frame for visual grounding
3. An entity knowledge base tracking game objects

IMPORTANT: Before answering, show your reasoning step-by-step:
1. First, identify the relevant events and timestamps from the context
2. Then, explain how these events connect to the question
3. Finally, provide your answer with specific timestamp citations

Format your response as:
**Reasoning:**
[Your step-by-step analysis here]

**Answer:**
[Your final answer with timestamp citations]"""


class MLXQwenVLCore:
    """
    MLX-based Qwen Vision-Language Model for Apple Silicon.
    
    Uses mlx-vlm for efficient inference on M1/M2/M3/M4 Macs.
    
    Example:
        >>> core = MLXQwenVLCore()
        >>> response = core.reason(
        ...     query="What happened at 5:30?",
        ...     current_frame=frame_image,
        ...     timeline_context="[05:28] Player attacks boss...",
        ... )
    """
    
    def __init__(self, config: Optional[MLXConfig] = None):
        self.config = config or MLXConfig()
        self._model = None
        self._processor = None
        self._loaded = False
        
        logger.info(f"MLXQwenVLCore initialized with model: {self.config.model_name}")
    
    def _load_model(self) -> None:
        """Lazy load the MLX model and processor."""
        if self._loaded:
            return
        
        try:
            from mlx_vlm import load, generate
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config
            
            logger.info(f"Loading MLX model: {self.config.model_name}")
            
            self._model, self._processor = load(self.config.model_name)
            self._generate = generate
            self._apply_chat_template = apply_chat_template
            self._load_config = load_config
            
            self._loaded = True
            logger.info("MLX model loaded successfully!")
            
        except ImportError as e:
            logger.error(f"mlx-vlm not installed: {e}")
            logger.error("Install with: pip install mlx-vlm")
            raise
        except Exception as e:
            logger.error(f"Failed to load MLX model: {e}")
            raise
    
    def reason(
        self,
        query: str,
        current_frame: Optional["Image.Image"] = None,
        timeline_context: str = "",
        region_tokens: str = "",
        knowledge_base_context: str = "",
    ) -> str:
        """
        Execute reasoning with the MLX model.
        
        Args:
            query: User question
            current_frame: Optional PIL Image for visual context
            timeline_context: Formatted timeline events
            region_tokens: Visual region descriptions
            knowledge_base_context: Entity knowledge base export
            
        Returns:
            Model response string
        """
        self._load_model()
        
        # Build context
        context_parts = []
        if timeline_context:
            context_parts.append(f"## Timeline Context\n{timeline_context}")
        if region_tokens:
            context_parts.append(f"## Visual Regions\n{region_tokens}")
        if knowledge_base_context:
            context_parts.append(f"## Entity Knowledge Base\n{knowledge_base_context}")
        
        context = "\n\n".join(context_parts) if context_parts else ""
        
        # Build prompt
        if context:
            user_message = f"{context}\n\n## Question\n{query}"
        else:
            user_message = query
        
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_message},
        ]
        
        # Handle image if provided
        if current_frame is not None:
            # mlx-vlm expects image in the message content
            messages[-1]["images"] = [current_frame]
        
        try:
            # Apply chat template
            prompt = self._apply_chat_template(
                self._processor,
                self._load_config(self.config.model_name),
                prompt=user_message,
                images=[current_frame] if current_frame else None,
            )
            
            # Generate response
            output = self._generate(
                self._model,
                self._processor,
                prompt,
                image=current_frame,
                max_tokens=self.config.max_new_tokens,
                temp=self.config.temperature,
                top_p=self.config.top_p,
                verbose=False,
            )
            
            return output.strip()
            
        except Exception as e:
            logger.error(f"MLX generation failed: {e}")
            return f"[Error during generation: {e}]"
    
    def reason_streaming(
        self,
        query: str,
        current_frame: Optional["Image.Image"] = None,
        timeline_context: str = "",
        region_tokens: str = "",
        knowledge_base_context: str = "",
    ) -> Generator[str, None, None]:
        """
        Execute reasoning with streaming output.
        
        Yields tokens as they're generated for real-time display.
        """
        self._load_model()
        
        # Build context (same as reason method)
        context_parts = []
        if timeline_context:
            context_parts.append(f"## Timeline Context\n{timeline_context}")
        if region_tokens:
            context_parts.append(f"## Visual Regions\n{region_tokens}")
        if knowledge_base_context:
            context_parts.append(f"## Entity Knowledge Base\n{knowledge_base_context}")
        
        context = "\n\n".join(context_parts) if context_parts else ""
        
        if context:
            user_message = f"{context}\n\n## Question\n{query}"
        else:
            user_message = query
        
        try:
            from mlx_vlm import stream_generate
            
            # Apply chat template
            prompt = self._apply_chat_template(
                self._processor,
                self._load_config(self.config.model_name),
                prompt=user_message,
                images=[current_frame] if current_frame else None,
            )
            
            # Stream generate
            for token in stream_generate(
                self._model,
                self._processor,
                prompt,
                image=current_frame,
                max_tokens=self.config.max_new_tokens,
                temp=self.config.temperature,
            ):
                yield token
                
        except ImportError:
            # Fallback to non-streaming if stream_generate not available
            logger.warning("stream_generate not available, falling back to non-streaming")
            response = self.reason(
                query=query,
                current_frame=current_frame,
                timeline_context=timeline_context,
                region_tokens=region_tokens,
                knowledge_base_context=knowledge_base_context,
            )
            yield response
            
        except Exception as e:
            logger.error(f"MLX streaming generation failed: {e}")
            yield f"[Error during generation: {e}]"


# Factory function for easy creation
def create_mlx_reasoning_core(
    model_name: str = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> MLXQwenVLCore:
    """
    Create an MLX-based reasoning core optimized for Apple Silicon.
    
    Args:
        model_name: HuggingFace model ID (should be MLX-quantized)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Configured MLXQwenVLCore instance
    """
    config = MLXConfig(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return MLXQwenVLCore(config)
