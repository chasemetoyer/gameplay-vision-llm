"""
Full Pipeline Model Inference Integration for Benchmarks.

Provides a unified interface to run the complete Gameplay Vision LLM pipeline
for benchmark evaluation:
- Frame extraction at configurable FPS
- SAM3 visual detection
- SigLIP semantic encoding
- VideoMAE temporal encoding
- Wav2Vec audio encoding
- OCR text extraction
- Whisper speech transcription
- Timeline indexing
- Knowledge base construction
- QwenVL reasoning with LoRA/projectors

This uses the same process_video function from realtime_inference.py
to ensure benchmarks test the full model, not just the LLM.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Any

import torch
from PIL import Image

# Add project root and scripts to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Import the full pipeline from realtime_inference
from realtime_inference import (
    process_video,
    answer_query,
    extract_frames,
    build_timeline_index,
)

from src.config.presets import load_preset, get_preset_summary
from benchmarks.loaders.base import BenchmarkSample

if TYPE_CHECKING:
    from benchmarks.model_configs import EvalModelConfig
    from benchmarks.perception_cache import CachedFeatures
    from src.agent_core.qwen_reasoning_core import PerceptionReasoningLoop

logger = logging.getLogger(__name__)


class FullPipelineRunner:
    """
    Runs the FULL perception + reasoning pipeline for benchmark evaluation.
    
    This is the same pipeline as realtime_inference.py:
    1. Frame extraction
    2. SAM3 detection
    3. SigLIP embeddings
    4. VideoMAE temporal (if enabled)
    5. Wav2Vec audio (if enabled)
    6. OCR extraction
    7. Whisper transcription
    8. Timeline building
    9. Knowledge base construction
    10. LLM reasoning with projectors + LoRA
    """
    
    def __init__(
        self,
        preset: str = "standard",
        projector_weights: str = "outputs/projector_weights.pt",
        lora_path: str = "outputs/lora_adapter",
        device: str = "cuda",
        use_cache: bool = True,
        cache_dir: str = "data/cache",
    ):
        """
        Initialize the full pipeline runner.
        
        Args:
            preset: Configuration preset ("light", "standard", "full")
            projector_weights: Path to trained projector weights
            lora_path: Path to LoRA adapter directory
            device: Device to run on
            use_cache: Whether to cache features between runs
            cache_dir: Directory for feature cache
        """
        self.preset = preset
        self.device = device
        self.projector_weights = projector_weights
        self.lora_path = lora_path
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        
        # Load preset configuration
        self.preset_config = load_preset(preset)
        logger.info(f"Loaded preset '{preset}' (~{self.preset_config.estimated_vram_gb:.0f}GB VRAM)")
        
        # Track processed videos and their loops
        self._video_loops: dict[str, "PerceptionReasoningLoop"] = {}
        
        # Stats
        self.total_perception_time = 0.0
        self.total_inference_time = 0.0
        self.total_samples = 0
        
        logger.info("FullPipelineRunner initialized")
        logger.info(f"  Preset: {preset}")
        logger.info(f"  Projector: {projector_weights}")
        logger.info(f"  LoRA: {lora_path}")
        logger.info(f"  Cache: {use_cache}")
    
    def process_video(self, video_path: str) -> "PerceptionReasoningLoop":
        """
        Process a video through the full perception pipeline.
        
        This runs:
        - Frame extraction
        - SAM3 detection
        - SigLIP embeddings
        - VideoMAE temporal (if preset enables)
        - Wav2Vec audio (if preset enables)
        - OCR extraction (using preset's backend)
        - Whisper transcription (using preset's model)
        - Timeline building
        - Knowledge base construction
        
        Results are cached if use_cache=True.
        
        Args:
            video_path: Path to video file
            
        Returns:
            PerceptionReasoningLoop ready for Q&A
        """
        # Check if already processed
        if video_path in self._video_loops:
            logger.info(f"Using cached pipeline for: {Path(video_path).name}")
            return self._video_loops[video_path]
        
        logger.info(f"Processing video through full pipeline: {Path(video_path).name}")
        t0 = time.time()
        
        # Extract FPS from preset (inference config, not perception)
        fps = self.preset_config.inference.fps
        
        # Run the full pipeline (same as realtime_inference.py)
        loop = process_video(
            video_path=video_path,
            device=self.device,
            fps=fps,
            use_sam=True,
            projector_weights=self.projector_weights,
            lora_path=self.lora_path,
            use_cache=self.use_cache,
            cache_dir=self.cache_dir,
            preset_config=self.preset_config,
        )
        
        processing_time = time.time() - t0
        self.total_perception_time += processing_time
        
        logger.info(f"Full pipeline completed in {processing_time:.1f}s")
        
        # Cache the loop for reuse
        self._video_loops[video_path] = loop
        
        return loop
    
    def run_inference(
        self,
        sample: BenchmarkSample,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ) -> tuple[str, float, float]:
        """
        Run full pipeline inference on a benchmark sample.
        
        Args:
            sample: Benchmark sample with question and video/image
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Tuple of (predicted_answer, perception_time_sec, inference_time_sec)
        """
        # Get video path
        video_path = sample.video_path
        if not video_path:
            # For image-only samples, we can't run full video pipeline
            logger.warning(f"Sample {sample.sample_id} has no video, using simplified inference")
            return self._run_image_only_inference(sample)
        
        if not os.path.exists(video_path):
            logger.error(f"Video not found: {video_path}")
            return "", 0.0, 0.0
        
        # Process video through full pipeline (cached if already done)
        perception_start = time.time()
        loop = self.process_video(video_path)
        perception_time = time.time() - perception_start
        
        # Get the question
        question = sample.question or sample.get_formatted_prompt()
        
        # Add MCQ options to question if present
        if sample.options:
            options_text = "\n".join([
                f"  ({chr(65+i)}) {opt}" 
                for i, opt in enumerate(sample.options)
            ])
            question = f"{question}\n\nOptions:\n{options_text}\n\nAnswer with the letter only (A, B, C, etc.)."
        
        # Get timestamp if specified
        timestamp = sample.start_time_sec
        
        # Run inference
        inference_start = time.time()
        try:
            response = answer_query(loop, question, timestamp=timestamp)
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            response = ""
        inference_time = time.time() - inference_start
        
        # Update stats
        self.total_inference_time += inference_time
        self.total_samples += 1
        
        # Parse answer from response
        answer = self._parse_answer(response, sample)
        
        return answer, perception_time, inference_time
    
    def _run_image_only_inference(
        self,
        sample: BenchmarkSample,
    ) -> tuple[str, float, float]:
        """Run simplified inference for image-only samples."""
        # For images, we use the simpler approach
        from benchmarks.model_inference import get_model_runner
        
        runner = get_model_runner(
            projector_weights_path=self.projector_weights,
            lora_path=self.lora_path,
            device=self.device,
        )
        
        answer, inference_time = runner.run_inference(sample)
        return answer, 0.0, inference_time
    
    def _parse_answer(self, response: str, sample: BenchmarkSample) -> str:
        """Parse the answer from model response."""
        if not response:
            return ""
        
        response = response.strip()
        
        # If MCQ, try to extract the letter answer
        if sample.options:
            response_upper = response.upper()
            import re
            
            # Pattern: single letter A-D at start
            match = re.match(r'^([A-D])\b', response_upper)
            if match:
                return match.group(1)
            
            # Pattern: (A) or [A]
            match = re.search(r'[\(\[]([A-D])[\)\]]', response_upper)
            if match:
                return match.group(1)
            
            # Pattern: Answer: A
            match = re.search(r'(?:answer|option)[:\s]+([A-D])\b', response_upper)
            if match:
                return match.group(1)
            
            # Check if response contains the option text
            for i, opt in enumerate(sample.options):
                if opt.lower() in response.lower():
                    return chr(65 + i)
        
        # For binary questions
        response_lower = response.lower()
        if "yes" in response_lower or "true" in response_lower:
            return "yes"
        if "no" in response_lower or "false" in response_lower:
            return "no"
        
        # Return first line
        first_line = response.split('\n')[0]
        return first_line[:200]
    
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return {
            "total_samples": self.total_samples,
            "total_perception_time_sec": self.total_perception_time,
            "total_inference_time_sec": self.total_inference_time,
            "avg_perception_time_sec": self.total_perception_time / max(1, len(self._video_loops)),
            "avg_inference_time_sec": self.total_inference_time / max(1, self.total_samples),
        }
    
    def clear_cache(self) -> None:
        """Clear cached video loops to free memory."""
        self._video_loops.clear()
        logger.info("Cleared video loop cache")


# Global runner instance
_full_pipeline_runner: Optional[FullPipelineRunner] = None


def get_full_pipeline_runner(
    preset: str = "standard",
    projector_weights: str = "outputs/projector_weights.pt",
    lora_path: str = "outputs/lora_adapter",
    device: str = "cuda",
    force_reload: bool = False,
) -> FullPipelineRunner:
    """
    Get the global full pipeline runner instance.
    
    Args:
        preset: Configuration preset
        projector_weights: Path to projector weights
        lora_path: Path to LoRA adapter
        device: Device to run on
        force_reload: Force runner reload
        
    Returns:
        FullPipelineRunner instance
    """
    global _full_pipeline_runner
    
    if _full_pipeline_runner is None or force_reload:
        _full_pipeline_runner = FullPipelineRunner(
            preset=preset,
            projector_weights=projector_weights,
            lora_path=lora_path,
            device=device,
        )
    
    return _full_pipeline_runner


def run_full_pipeline_inference(
    sample: BenchmarkSample,
    preset: str = "standard",
) -> tuple[str, float, float]:
    """
    Convenience function to run full pipeline inference.
    
    Args:
        sample: Benchmark sample
        preset: Configuration preset
        
    Returns:
        Tuple of (predicted_answer, perception_time, inference_time)
    """
    runner = get_full_pipeline_runner(preset=preset)
    return runner.run_inference(sample)


# =============================================================================
# Backward compatibility - keep simplified runner for quick tests
# =============================================================================

class BenchmarkModelRunner:
    """
    Simplified model runner (single-frame inference only).
    
    Use FullPipelineRunner for proper benchmarking with the full pipeline.
    This is kept for backward compatibility and quick tests.
    """
    
    def __init__(
        self,
        projector_weights_path: str = "outputs/projector_weights.pt",
        lora_path: str = "outputs/lora_adapter",
        device: str = "cuda",
        use_flash_attn: bool = True,
        load_8bit: bool = False,
    ):
        self.device = device
        self.projector_weights_path = projector_weights_path
        self.lora_path = lora_path
        self._qwen_core = None
        self.total_samples = 0
        self.total_inference_time = 0.0
        logger.info("BenchmarkModelRunner initialized (simplified single-frame mode)")
    
    def load_model(self) -> None:
        if self._qwen_core is not None:
            return
        
        from src.agent_core.qwen_reasoning_core import QwenVLCore, ReasoningCoreConfig
        
        logger.info("Loading Qwen3-VL model...")
        config = ReasoningCoreConfig(
            device=self.device,
            max_new_tokens=256,  # Ensure complete responses
            temperature=0.3,    # Lower temperature for more deterministic benchmark answers
        )
        self._qwen_core = QwenVLCore(config)
        
        # Note: QwenVLCore._load_model() is called automatically when reason() is invoked
        # No need to call it explicitly here
        
        logger.info("Model initialized (will load on first inference)")
    
    def run_inference(self, sample: BenchmarkSample, timeline_context: Optional[str] = None) -> tuple[str, float]:
        self.load_model()
        
        t0 = time.time()
        question = sample.question or sample.get_formatted_prompt()
        
        # Load single frame
        current_frame = None
        if sample.image_path and os.path.exists(sample.image_path):
            current_frame = Image.open(sample.image_path).convert("RGB")
        elif sample.video_path and os.path.exists(sample.video_path):
            import cv2
            cap = cv2.VideoCapture(sample.video_path)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2)
                ret, frame = cap.read()
                if ret:
                    current_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
        
        # Run SAM3 + SigLIP perception on the frame
        perception_context = ""
        region_detections = []
        if current_frame is not None:
            perception_context, region_detections = self._run_perception(current_frame)
        
        # Build enhanced question with perception context
        enhanced_question = question
        if perception_context:
            enhanced_question = f"{question}\n\n[Visual Analysis]\n{perception_context}"
        
        # Add options
        if sample.options:
            options_text = "\n".join([f"  ({chr(65+i)}) {opt}" for i, opt in enumerate(sample.options)])
            enhanced_question = f"{enhanced_question}\n\nOptions:\n{options_text}\n\nAnswer with the letter only."
        
        try:
            response = self._qwen_core.reason(
                query=enhanced_question,
                current_frame=current_frame,
                region_detections=region_detections if region_detections else None,
                track_conversation=False,  # Don't track for benchmark runs
                return_confidence=False,
            )
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            response = ""
        
        inference_time = time.time() - t0
        self.total_samples += 1
        self.total_inference_time += inference_time
        
        # Log FULL response for debugging
        logger.info(f"=" * 60)
        logger.info(f"FULL MODEL RESPONSE ({len(response)} chars):")
        logger.info(response)
        logger.info(f"=" * 60)
        
        # Parse answer
        answer = self._parse_answer(response, sample)
        return answer, inference_time
    
    def _run_perception(self, frame: Image.Image) -> tuple[str, list[dict]]:
        """Run SAM3 perception on a single frame for entity detection."""
        import numpy as np
        
        region_detections = []
        descriptions = []
        
        # Run SAM3 for entity detection
        try:
            from perception.sam_concept_segmenter import SAMConceptSegmenter, SAMConfig
            
            if not hasattr(self, '_sam_segmenter') or self._sam_segmenter is None:
                logger.info("Loading SAM3 for image analysis...")
                config = SAMConfig(device=self.device)
                self._sam_segmenter = SAMConceptSegmenter(config)
            
            frame_np = np.array(frame)
            concepts = ["game character", "player", "vehicle", "object", "anomaly", "glitch"]
            
            entities = self._sam_segmenter.segment_with_prompts(
                frame=frame_np,
                frame_idx=0,
                concept_prompts=concepts,
            )
            
            for entity in entities:
                label = entity.label if hasattr(entity, 'label') else "entity"
                conf = entity.confidence if hasattr(entity, 'confidence') else 0.5
                region_detections.append({
                    "label": label,
                    "confidence": conf,
                })
                descriptions.append(f"- Detected: {label} (conf: {conf:.2f})")
            
            if entities:
                logger.info(f"SAM3 detected {len(entities)} entities in image")
            else:
                descriptions.append("- No specific game entities detected")
            
        except Exception as e:
            logger.warning(f"SAM3 perception failed: {e}")
            descriptions.append("- Visual analysis unavailable")
        
        perception_context = "\n".join(descriptions) if descriptions else ""
        return perception_context, region_detections
    
    def _parse_answer(self, response: str, sample: BenchmarkSample) -> str:
        if not response:
            return ""
        response = response.strip()
        response_lower = response.lower()
        
        # For MCQ, check for letter answers first
        if sample.options:
            import re
            match = re.match(r'^([A-D])\b', response.upper())
            if match:
                logger.info(f"  → Matched MCQ letter: {match.group(1)}")
                return match.group(1)
        
        # Helper to check if pattern is negated (preceded by "no", "not", "isn't", etc.)
        def is_negated(text: str, pattern: str) -> bool:
            import re
            # Find all occurrences of pattern and check if any are preceded by negation
            negation_words = r'\b(no|not|isn\'t|aren\'t|wasn\'t|weren\'t|don\'t|doesn\'t|didn\'t|without|lacks?|absence)\b'
            # Check for negation within 30 chars before the pattern
            for match in re.finditer(re.escape(pattern), text):
                start = max(0, match.start() - 30)
                prefix = text[start:match.start()]
                if re.search(negation_words, prefix):
                    return True
            return False
        
        # Check for explicit denial FIRST (these are clear "no glitch" statements)
        no_glitch_patterns = [
            "no glitch", "no visible glitch", "is not a glitch", "isn't a glitch",
            "normal gameplay", "this is normal", "nothing unusual",
            "no clear indication", "no indication of a glitch", "not a glitch",
            "no specific glitch", "no obvious glitch", "appears normal",
            "no anomalies", "looks normal", "nothing abnormal",
            "no visual evidence", "no signs of", "no physics error", "no clipping",
        ]
        for pattern in no_glitch_patterns:
            if pattern in response_lower:
                logger.info(f"  → Matched NO pattern: '{pattern}'")
                return "no"
        
        # Check for explicit glitch affirmation (only if not negated)
        glitch_yes_patterns = [
            "there is a glitch", "yes, there is", "is a glitch", "glitch detected", 
            "appears to be a glitch", "clearly a glitch", "this is a glitch",
            "visible glitch", "glitch in this", "glitch can be seen", "glitch is present",
            "physics error", "clipping error", "texture error", "animation bug",
        ]
        for pattern in glitch_yes_patterns:
            if pattern in response_lower:
                # Check if this pattern is negated
                if is_negated(response_lower, pattern):
                    logger.info(f"  → Pattern '{pattern}' found but NEGATED, treating as NO")
                    return "no"
                logger.info(f"  → Matched YES pattern: '{pattern}' (not negated)")
                return "yes"
        
        # Fall back to simple yes/no if at start
        if response_lower.startswith("yes"):
            logger.info(f"  → Matched: starts with 'yes'")
            return "yes"
        if response_lower.startswith("no"):
            logger.info(f"  → Matched: starts with 'no'")
            return "no"
        
        # Look for Answer: section in structured response
        import re
        answer_match = re.search(r'\*\*answer[:\s]*\*\*\s*([^\n]+)', response_lower)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            logger.info(f"  → Found Answer section: '{answer_text}'")
            if any(kw in answer_text for kw in ["yes", "glitch", "bug"]):
                return "yes"
            if any(kw in answer_text for kw in ["no", "normal"]):
                return "no"
        
        # No pattern matched
        logger.info(f"  → No pattern matched, returning first line")
        return response.split('\n')[0][:200]
    
    def get_stats(self) -> dict:
        return {
            "total_samples": self.total_samples,
            "total_time_sec": self.total_inference_time,
            "avg_time_per_sample_sec": self.total_inference_time / max(1, self.total_samples),
        }


_model_runner: Optional[BenchmarkModelRunner] = None


def get_model_runner(
    projector_weights_path: str = "outputs/projector_weights.pt",
    lora_path: str = "outputs/lora_adapter",
    device: str = "cuda",
    force_reload: bool = False,
) -> BenchmarkModelRunner:
    global _model_runner
    if _model_runner is None or force_reload:
        _model_runner = BenchmarkModelRunner(
            projector_weights_path=projector_weights_path,
            lora_path=lora_path,
            device=device,
        )
    return _model_runner
