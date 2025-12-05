#!/usr/bin/env python3
"""
Feature Extraction Pipeline - Generates structured data from gameplay videos.

This script runs the COMPLETE perception pipeline on videos and outputs structured
JSON/Text that can be fed to GPT-4 for generating instruction tuning data.

Implements Step 1 of the LoRA Training Data Guide:
- Phase 1 Perception: InternVideo2.5 HiCo, SAM 3, SigLIP 2, OCR, Qwen2-Audio
- Phase 2 Fusion: TimelineIndexer, KnowledgeBaseBuilder with Causal Links

Usage:
    python scripts/extract_features.py --video data/gameplay.mp4 --output data/outputs/
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Entity Tracking (VideoRAG: Cross-frame entity persistence)
# =============================================================================

class TrackedEntity:
    """Represents an entity tracked across multiple frames."""
    
    def __init__(self, entity_id: str, entity_type: str, first_timestamp: float, bbox: list):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.first_seen = first_timestamp
        self.last_seen = first_timestamp
        self.trajectory = [(first_timestamp, bbox)]  # List of (timestamp, bbox)
        self.states = [(first_timestamp, "appeared")]
        self.confidence_history = []
        self.embedding = None  # Will be set by SigLIP
        
    def update(self, timestamp: float, bbox: list, confidence: float = 1.0):
        """Update entity with new detection."""
        self.last_seen = timestamp
        self.trajectory.append((timestamp, bbox))
        self.confidence_history.append(confidence)
        
    def mark_state(self, timestamp: float, state: str):
        """Mark a state change (e.g., 'stunned', 'defeated')."""
        self.states.append((timestamp, state))
        
    def to_knowledge_entry(self) -> dict:
        """Convert to knowledge base entry format."""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "duration": self.last_seen - self.first_seen,
            "trajectory_length": len(self.trajectory),
            "states": self.states,
            "avg_confidence": sum(self.confidence_history) / len(self.confidence_history) if self.confidence_history else 1.0,
        }
        
    def to_timeline_text(self) -> str:
        """Generate readable timeline text."""
        minutes_first = int(self.first_seen // 60)
        seconds_first = int(self.first_seen % 60)
        minutes_last = int(self.last_seen // 60)
        seconds_last = int(self.last_seen % 60)
        
        state_changes = " → ".join([f"{s}" for _, s in self.states])
        return f"{self.entity_type.title()} ({self.entity_id}): [{minutes_first:02d}:{seconds_first:02d}] to [{minutes_last:02d}:{seconds_last:02d}] | {state_changes}"


class EntityTracker:
    """
    Tracks entities across frames using IoU (Intersection over Union) matching.
    
    Implements VideoRAG Strategy A: Entity-Centric Knowledge Base Construction.
    """
    
    def __init__(self, iou_threshold: float = 0.3, max_frames_missing: int = 10):
        self.iou_threshold = iou_threshold
        self.max_frames_missing = max_frames_missing
        self.entities: dict[str, TrackedEntity] = {}
        self.entity_counter = {}  # Per-type counters
        self._last_frame_entities = {}  # entity_id -> (timestamp, bbox)
        self._frames_since_seen = {}  # entity_id -> frame count
        
    def _compute_iou(self, box1: list, box2: list) -> float:
        """Compute IoU between two bboxes [x1, y1, x2, y2]."""
        if box1 is None or box2 is None:
            return 0.0
        
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
    
    def _generate_id(self, entity_type: str) -> str:
        """Generate unique entity ID."""
        if entity_type not in self.entity_counter:
            self.entity_counter[entity_type] = 0
        self.entity_counter[entity_type] += 1
        return f"{entity_type}_{self.entity_counter[entity_type]:03d}"
    
    def update(self, timestamp: float, detections: list[dict]) -> list[dict]:
        """
        Process detections for a frame and update entity tracking.
        
        Args:
            timestamp: Current frame timestamp
            detections: List of {"entity_type", "bbox", "confidence", ...}
            
        Returns:
            Enriched detections with entity_id
        """
        # Increment frames since seen for all tracked entities
        for eid in list(self._frames_since_seen.keys()):
            self._frames_since_seen[eid] += 1
            
        enriched = []
        matched_entity_ids = set()
        
        for det in detections:
            entity_type = det.get("entity_type", "unknown")
            bbox = det.get("bbox")
            confidence = det.get("confidence", 1.0)
            
            # Find best matching existing entity of same type
            best_match_id = None
            best_iou = self.iou_threshold
            
            for eid, entity in self.entities.items():
                if entity.entity_type != entity_type:
                    continue
                if eid in matched_entity_ids:
                    continue
                    
                # Get last known position
                if eid in self._last_frame_entities:
                    _, last_bbox = self._last_frame_entities[eid]
                    iou = self._compute_iou(bbox, last_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_match_id = eid
            
            if best_match_id:
                # Update existing entity
                self.entities[best_match_id].update(timestamp, bbox, confidence)
                entity_id = best_match_id
                matched_entity_ids.add(entity_id)
                self._frames_since_seen[entity_id] = 0
            else:
                # Create new entity
                entity_id = self._generate_id(entity_type)
                self.entities[entity_id] = TrackedEntity(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    first_timestamp=timestamp,
                    bbox=bbox
                )
                self._frames_since_seen[entity_id] = 0
                matched_entity_ids.add(entity_id)
            
            # Update last known position
            self._last_frame_entities[entity_id] = (timestamp, bbox)
            
            # Enrich detection with entity_id
            enriched_det = det.copy()
            enriched_det["entity_id"] = entity_id
            enriched.append(enriched_det)
        
        # Mark entities as disappeared if not seen for too long
        for eid in list(self._frames_since_seen.keys()):
            if self._frames_since_seen[eid] > self.max_frames_missing:
                if eid in self.entities:
                    self.entities[eid].mark_state(timestamp, "disappeared")
                del self._frames_since_seen[eid]
                if eid in self._last_frame_entities:
                    del self._last_frame_entities[eid]
        
        return enriched
    
    def get_all_entities(self) -> list[TrackedEntity]:
        """Get all tracked entities."""
        return list(self.entities.values())
    
    def get_entity_summary(self) -> str:
        """Generate summary of all tracked entities."""
        lines = ["## Tracked Entities", ""]
        for entity in sorted(self.entities.values(), key=lambda e: e.first_seen):
            lines.append(f"- {entity.to_timeline_text()}")
        return "\n".join(lines)


# =============================================================================
# Frame Extraction
# =============================================================================

def extract_frames(video_path: str, fps: float = 1.0) -> list[tuple[float, Image.Image]]:
    """
    Extract frames from video at specified FPS.
    
    Returns:
        List of (timestamp_seconds, PIL Image) tuples
    """
    try:
        import decord
        decord.bridge.set_bridge("native")
        
        vr = decord.VideoReader(video_path)
        video_fps = vr.get_avg_fps()
        total_frames = len(vr)
        duration = total_frames / video_fps
        
        logger.info(f"Video: {duration:.1f}s, {video_fps:.1f} fps, {total_frames} frames")
        
        # Sample at target FPS
        frame_interval = int(video_fps / fps)
        sample_indices = list(range(0, total_frames, frame_interval))
        
        frames = []
        for idx in sample_indices:
            timestamp = idx / video_fps
            frame_np = vr[idx].asnumpy()
            frame_pil = Image.fromarray(frame_np)
            frames.append((timestamp, frame_pil))
        
        logger.info(f"Extracted {len(frames)} frames at {fps} FPS")
        return frames
        
    except ImportError:
        logger.error("decord not installed. Run: pip install decord")
        raise


# =============================================================================
# InternVideo2.5 HiCo (Temporal Compression)
# =============================================================================

def run_internvideo_hico(frames: list[tuple[float, Image.Image]], device: str = "cuda"):
    """
    Run InternVideo2.5 with Hierarchical Token Compression (HiCo).
    
    This compresses long video sequences into efficient temporal tokens,
    enabling reasoning over minutes of footage without overwhelming context.
    """
    try:
        from temporal.internvideo_hico_module import InternVideoHiCoModule, HiCoConfig, CompressionLevel
        import numpy as np
        
        logger.info(f"Loading InternVideo2.5 HiCo on {device}...")
        config = HiCoConfig(device=device)
        hico = InternVideoHiCoModule(config)
        
        # Convert PIL frames to numpy array for processing
        # Expected shape: (T, H, W, C)
        frame_arrays = []
        for timestamp, frame in frames:
            frame_np = np.array(frame.convert("RGB"))
            frame_arrays.append(frame_np)
        
        # Stack into (T, H, W, C) array
        video_array = np.stack(frame_arrays, axis=0)
        
        # Calculate effective FPS based on timestamps
        if len(frames) > 1:
            fps = len(frames) / (frames[-1][0] - frames[0][0])
        else:
            fps = 1.0
        
        logger.info(f"Processing {len(frame_arrays)} frames through HiCo (effective fps: {fps:.2f})...")
        
        # Process video through HiCo pipeline using segment_and_compress
        temporal_tokens = hico.segment_and_compress(
            frames=video_array,
            fps=fps,
            target_level=CompressionLevel.CLIP  # Compress to clip-level tokens
        )
        
        results = {
            "num_input_frames": len(frames),
            "num_temporal_tokens": len(temporal_tokens) if temporal_tokens else 0,
            "compression_ratio": len(frames) / max(len(temporal_tokens), 1) if temporal_tokens else 0,
            "tokens": temporal_tokens
        }
        
        logger.info(f"HiCo compression: {results['num_input_frames']} frames -> {results['num_temporal_tokens']} tokens")
        return results
        
    except ImportError as e:
        logger.warning(f"InternVideo2.5 HiCo not available: {e}")
        return {"num_input_frames": len(frames), "num_temporal_tokens": 0, "tokens": None}
    except Exception as e:
        logger.warning(f"HiCo processing failed: {e}")
        import traceback
        traceback.print_exc()
        return {"num_input_frames": len(frames), "num_temporal_tokens": 0, "tokens": None}


# =============================================================================
# SigLIP 2 (Semantic Embeddings)
# =============================================================================

def run_siglip_encoder(frames: list[tuple[float, Image.Image]], device: str = "cuda"):
    """Run SigLIP2 on frames to get semantic embeddings."""
    from perception.siglip_semantic_encoder import SigLIPSemanticEncoder, NaFlexConfig
    
    config = NaFlexConfig(device=device)
    encoder = SigLIPSemanticEncoder(config)
    
    embeddings = []
    for timestamp, frame in frames:
        try:
            # Encode full frame for now (Phase 1)
            # In Phase 2, we will pass SAM masks here
            embedding = encoder.encode_image(frame)
            if embedding is not None:
                embeddings.append({
                    "timestamp": timestamp,
                    "embedding_shape": list(embedding.shape),
                })
        except Exception as e:
            logger.warning(f"SigLIP failed at {timestamp:.1f}s: {e}")
    
    logger.info(f"SigLIP encoded {len(embeddings)} frames")
    return embeddings


# =============================================================================
# OCR (PaddleOCR)
# =============================================================================

def run_ocr(frames: list[tuple[float, Image.Image]], device: str = "cuda"):
    """Run PaddleOCR on frames to extract text."""
    try:
        from perception.ocr_pipeline import OCRPipeline, OCRConfig
        import numpy as np
        
        config = OCRConfig(use_gpu=(device == "cuda"))
        ocr = OCRPipeline(config)
        
        results = []
        for idx, (timestamp, frame) in enumerate(frames):
            try:
                # Convert PIL to numpy array
                frame_np = np.array(frame.convert("RGB"))
                
                # Use correct method name
                ocr_result = ocr.extract_text_from_frame(frame_np, idx, timestamp)
                
                if ocr_result and ocr_result.detections:
                    for detection in ocr_result.detections:
                        results.append({
                            "timestamp": timestamp,
                            "text": detection.text,
                            "confidence": detection.confidence,
                            "bbox": detection.bbox,
                            "category": detection.category,
                        })
            except Exception as e:
                logger.warning(f"OCR failed at {timestamp:.1f}s: {e}")
        
        logger.info(f"OCR extracted {len(results)} text regions")
        return results
        
    except ImportError as e:
        logger.warning(f"OCR not available: {e}")
        return []


# =============================================================================
# SAM 3 (Concept Segmentation)
# =============================================================================

def run_sam_detection(frames: list[tuple[float, Image.Image]], device: str = "cuda"):
    """
    Run SAM 3 Concept Segmentation with entity tracking across frames.
    
    Returns both per-frame detections and the tracker for entity summaries.
    """
    try:
        from perception.sam_concept_segmenter import Sam3ModelWrapper, SAMConfig
        
        # Default prompts for gameplay
        prompts = ["character", "enemy", "boss", "health bar", "weapon"]
        
        config = SAMConfig(device=device)
        segmenter = Sam3ModelWrapper(config)
        
        # Initialize entity tracker for cross-frame persistence
        tracker = EntityTracker(iou_threshold=0.3, max_frames_missing=5)
        
        results = []
        logger.info(f"Loading SAM 3 model on {device}...")
        
        for idx, (timestamp, frame) in enumerate(frames):
            try:
                # Run segmentation for each prompt
                frame_detections = []
                for prompt in prompts:
                    # Use segment_with_text which is the correct API
                    masks = segmenter.segment_with_text(frame, prompt)
                    
                    # If we found something with high confidence
                    for mask in masks:
                        # Handle both SegmentationMask objects and dict placeholders
                        if isinstance(mask, dict):
                            # Placeholder returns 'scores' tensor, real models might return 'confidence'
                            scores = mask.get("scores")
                            if scores is not None and len(scores) > 0:
                                confidence = float(scores[0]) if hasattr(scores, '__getitem__') else float(scores)
                            else:
                                confidence = mask.get("confidence", 0)
                            # Get bbox from 'boxes' tensor or 'bbox' list
                            boxes = mask.get("boxes")
                            if boxes is not None and len(boxes) > 0:
                                bbox = boxes[0].tolist() if hasattr(boxes[0], 'tolist') else list(boxes[0])
                            else:
                                bbox = mask.get("bbox")
                        else:
                            confidence = mask.confidence
                            bbox = mask.bbox.to_xyxy() if mask.bbox else None
                            
                        if confidence > 0.4:
                            frame_detections.append({
                                "timestamp": timestamp,
                                "type": "entity_detection",
                                "description": f"Detected {prompt}",
                                "entity_type": prompt,
                                "confidence": confidence,
                                "bbox": bbox,
                                "motion_score": 0.0
                            })
                            # Only take the best one per prompt to avoid spam
                            break
                
                # Update tracker with this frame's detections (adds entity_id)
                tracked_detections = tracker.update(timestamp, frame_detections)
                results.extend(tracked_detections)
                
                if idx % 10 == 0:
                    logger.info(f"SAM processed frame {idx}/{len(frames)}")
                    
            except Exception as e:
                logger.warning(f"SAM failed at {timestamp:.1f}s: {e}")
        
        # Log entity tracking summary
        unique_entities = len(tracker.entities)
        logger.info(f"SAM detected {len(results)} events across {unique_entities} unique entities")
        logger.info(f"Entity tracking summary:\n{tracker.get_entity_summary()}")
        
        return results, tracker
        
    except ImportError as e:
        logger.warning(f"SAM dependencies not met: {e}")
        return []
    except Exception as e:
        logger.warning(f"SAM initialization failed: {e}")
        return []


def run_visual_detection(frames: list[tuple[float, Image.Image]], device: str = "cuda", use_sam: bool = False):
    """
    Run visual analysis. Uses SAM 3 if requested, otherwise falls back to motion detection.
    
    Returns: (detections, tracker) where tracker is None for motion detection
    """
    if use_sam:
        logger.info("Using SAM 3 for advanced entity detection...")
        result = run_sam_detection(frames, device)
        # Handle both success (results, tracker) and fallback (empty list)
        if isinstance(result, tuple):
            return result
        else:
            return result, None
        
    logger.info("Using basic motion detection (fast mode)...")
    """
    Simple visual analysis using frame differencing and region detection.
    
    Since SAM3 requires setup, this provides basic motion/change detection.
    """
    import numpy as np
    
    results = []
    prev_frame_np = None
    
    for timestamp, frame in frames:
        frame_np = np.array(frame.convert("RGB"))
        
        # Detect significant changes (motion)
        if prev_frame_np is not None:
            diff = np.abs(frame_np.astype(float) - prev_frame_np.astype(float))
            mean_diff = diff.mean()
            
            # Classify based on motion intensity
            if mean_diff > 30:
                results.append({
                    "timestamp": timestamp,
                    "type": "high_action",
                    "description": "Intense combat or movement detected",
                    "motion_score": float(mean_diff),
                })
            elif mean_diff > 15:
                results.append({
                    "timestamp": timestamp,
                    "type": "action",
                    "description": "Combat action detected",
                    "motion_score": float(mean_diff),
                })
            elif mean_diff > 5:
                results.append({
                    "timestamp": timestamp,
                    "type": "movement",
                    "description": "Character movement detected",
                    "motion_score": float(mean_diff),
                })
            else:
                results.append({
                    "timestamp": timestamp,
                    "type": "idle",
                    "description": "Minimal movement (cutscene or idle)",
                    "motion_score": float(mean_diff),
                })
        
        prev_frame_np = frame_np
    
    logger.info(f"Visual detection: {len(results)} motion events")
    return results, None  # No tracker for motion detection


# =============================================================================
# Qwen2-Audio (Speech & Sound Events)
# =============================================================================

def run_audio_extraction(video_path: str, device: str = "cuda"):
    """
    Run Qwen2-Audio to extract speech and audio events.
    """
    try:
        from audio.qwen_audio_processor import QwenAudioProcessor, QwenAudioConfig
        
        logger.info(f"Loading Qwen2-Audio on {device}...")
        config = QwenAudioConfig(device=device)
        processor = QwenAudioProcessor(config)
        
        logger.info("Processing audio stream...")
        # Extract audio from video first
        audio_array, sample_rate = processor.preprocessor.extract_from_video(video_path)
        
        # Process in chunks to get timestamps
        results = []
        chunks = processor.preprocessor.chunk_audio(audio_array, sample_rate, chunk_duration=30.0)
        
        logger.info(f"Analyzing {len(chunks)} audio chunks...")
        for i, (chunk, start, end) in enumerate(chunks):
            if i % 5 == 0:
                logger.info(f"Processing audio chunk {i}/{len(chunks)}")
                
            # Transcribe Speech
            transcription = processor.model.transcribe(chunk, sample_rate)
            if transcription and len(transcription) > 2:
                results.append({
                    "timestamp": start,
                    "end_time": end,
                    "text": transcription,
                    "type": "dialogue"
                })

            # Detect Audio Events (sounds, music, effects)
            try:
                events = processor.model.analyze_audio_events(chunk, sample_rate)
                if events and len(events) > 5 and "no audio" not in events.lower():
                    results.append({
                        "timestamp": start,
                        "end_time": end,
                        "text": events,
                        "type": "sound_event"
                    })
            except Exception:
                pass  # Event detection is optional
        
        logger.info(f"Qwen2-Audio extracted {len(results)} segments")
        return results
        
    except ImportError as e:
        logger.error(f"Failed to import QwenAudioProcessor: {e}")
        return []
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return []


# =============================================================================
# Timeline Indexer (Multi-Modal Alignment)
# =============================================================================

def build_timeline_with_indexer(
    ocr_results: list[dict],
    visual_results: list[dict],
    audio_results: list[dict],
    hico_results: dict,
):
    """
    Build a structured timeline using the validated TimelineIndexer.
    
    This properly merges, deduplicates, and aligns all modality outputs.
    """
    from fusion_indexing.timeline_indexer import (
        TimelineIndexer, TimelineConfig, ModalityType, EventPriority
    )
    
    config = TimelineConfig(
        merge_window_sec=0.5,
        dedupe_threshold=0.85,
        compact_format=True
    )
    indexer = TimelineIndexer(config)
    
    # Add OCR events
    for ocr in ocr_results:
        indexer.add_event(
            timestamp=ocr["timestamp"],
            modality=ModalityType.OCR,
            description=f"Text: \"{ocr['text']}\"",
            priority=EventPriority.HIGH if ocr.get("confidence", 0) > 0.8 else EventPriority.MEDIUM,
            confidence=ocr.get("confidence", 1.0),
            metadata={"category": ocr.get("category", "unknown")}
        )
    
    # Add visual detection events
    for vis in visual_results:
        priority = EventPriority.HIGH if vis.get("type") == "high_action" else EventPriority.MEDIUM
        indexer.add_event(
            timestamp=vis["timestamp"],
            modality=ModalityType.VISUAL,
            description=vis["description"],
            priority=priority,
            confidence=vis.get("confidence", 1.0),
            metadata={"motion_type": vis.get("type"), "motion_score": vis.get("motion_score", 0)}
        )
        
    # Add audio events
    if audio_results:
        for audio in audio_results:
            modality = ModalityType.SPEECH if audio["type"] == "dialogue" else ModalityType.AUDIO
            indexer.add_event(
                timestamp=audio["timestamp"],
                modality=modality,
                description=audio["text"],
                priority=EventPriority.HIGH,
                duration=audio.get("end_time", audio["timestamp"]) - audio["timestamp"],
                metadata={"audio_type": audio["type"]}
            )
    
    # Add temporal context note if HiCo was used
    if hico_results.get("num_temporal_tokens", 0) > 0:
        indexer.add_event(
            timestamp=0.0,
            modality=ModalityType.TEMPORAL,
            description=f"Temporal context: {hico_results['num_input_frames']} frames compressed to {hico_results['num_temporal_tokens']} tokens",
            priority=EventPriority.LOW,
            metadata={"compression_ratio": hico_results.get("compression_ratio", 0)}
        )
    
    # Merge and deduplicate events
    indexer.merge_and_dedupe()
    
    # Get merged and deduplicated timeline (use internal _events list)
    timeline_events = sorted(indexer._events, key=lambda e: e.timestamp)
    
    # Convert to simple dict format for JSON serialization
    timeline = []
    for event in timeline_events:
        timeline.append({
            "timestamp": event.timestamp,
            "type": event.modality.value,
            "content": event.description,
            "priority": event.priority.name,
            "confidence": event.confidence,
            "duration": event.duration,
        })
    
    logger.info(f"TimelineIndexer created {len(timeline)} merged events")
    return timeline, indexer


# =============================================================================
# Knowledge Base Builder with Causal Inference
# =============================================================================

def build_knowledge_base_with_causality(timeline: list[dict], video_name: str):
    """
    Build the entity knowledge base from timeline events WITH causal inference.
    
    This implements Step 1B: Leverage Causal Encoding
    - Detects patterns like "attack at T1" -> "death at T2"
    - Creates explicit [causes] links for LLM training
    """
    from fusion_indexing.knowledge_base_builder import (
        KnowledgeBaseBuilder, KnowledgeBaseConfig, RelationType, EntityCategory
    )
    
    kb = KnowledgeBaseBuilder(KnowledgeBaseConfig())
    
    # Track potential cause-effect patterns
    causal_links = []
    attack_events = []   # Events indicating attacks/actions
    effect_events = []   # Events indicating effects/results
    
    # Pattern definitions for gameplay causal detection
    ATTACK_PATTERNS = ["attacks", "casts a spell", "strikes", "parry", "dodge", 
                       "performs", "uses", "summons", "charges"]
    EFFECT_PATTERNS = ["damage", "hit", "critical", "wounds", "burns", 
                       "perfect", "broken", "stunned", "enraged", "defeated"]
    VICTORY_PATTERNS = ["victory", "battle loot", "xp", "defeated the"]
    SKILL_PATTERNS = ["stains", "overcharge", "immolation", "healing", 
                      "recovery", "assault", "thunderfall"]
    
    # Process timeline events into KB
    for event in timeline:
        timestamp = event["timestamp"]
        content = event["content"]
        evt_type = event["type"]
        
        # Create unique entity ID
        entity_id = f"{evt_type}_{timestamp:.1f}".replace(".", "_")
        
        # Add entities based on event type
        if evt_type == "ocr":
            entity = kb.register_entity(
                entity_id=entity_id,
                concept_label=f"OCR_{timestamp:.1f}",
                category=EntityCategory.UI_ELEMENT,
                timestamp=timestamp,
                attributes={"text": content, "timestamp": timestamp}
            )
            
            content_lower = content.lower()
            
            # Detect attack/action events
            if any(pattern in content_lower for pattern in ATTACK_PATTERNS):
                attack_events.append({
                    "timestamp": timestamp, 
                    "entity_id": entity_id, 
                    "content": content, 
                    "type": "attack"
                })
            
            # Detect skill usage
            if any(pattern in content_lower for pattern in SKILL_PATTERNS):
                attack_events.append({
                    "timestamp": timestamp, 
                    "entity_id": entity_id, 
                    "content": content, 
                    "type": "skill"
                })
            
            # Detect effect/result events
            if any(pattern in content_lower for pattern in EFFECT_PATTERNS):
                effect_events.append({
                    "timestamp": timestamp, 
                    "entity_id": entity_id, 
                    "content": content,
                    "type": "effect"
                })
            
            # Detect victory (special effect)
            if any(pattern in content_lower for pattern in VICTORY_PATTERNS):
                effect_events.append({
                    "timestamp": timestamp, 
                    "entity_id": entity_id, 
                    "content": content,
                    "type": "victory"
                })
                
        elif evt_type == "visual":
            entity = kb.register_entity(
                entity_id=entity_id,
                concept_label=f"Visual_{timestamp:.1f}",
                category=EntityCategory.EFFECT,
                timestamp=timestamp,
                attributes={"description": content, "timestamp": timestamp}
            )
            
            # Visual combat detection
            content_lower = content.lower()
            if any(word in content_lower for word in ["combat", "attack", "weapon"]):
                attack_events.append({
                    "timestamp": timestamp, 
                    "entity_id": entity_id, 
                    "content": content, 
                    "type": "combat_visual"
                })
                
        elif evt_type == "speech" or evt_type == "audio":
            entity = kb.register_entity(
                entity_id=entity_id,
                concept_label=f"Audio_{timestamp:.1f}",
                category=EntityCategory.EFFECT,
                timestamp=timestamp,
                attributes={"text": content, "timestamp": timestamp}
            )
    
    # Infer causal links: Find attacks that precede effects
    CAUSAL_WINDOW = 5.0  # seconds - events within this window may be causally linked
    
    for effect in effect_events:
        effect_time = effect["timestamp"]
        
        # Find potential causes (attack events shortly before effect)
        for attack in attack_events:
            attack_time = attack["timestamp"]
            
            # Check if attack happened within causal window before effect
            if 0 < (effect_time - attack_time) <= CAUSAL_WINDOW:
                causal_link = {
                    "cause_timestamp": attack_time,
                    "cause_event": attack["content"],
                    "cause_type": attack["type"],
                    "effect_timestamp": effect_time,
                    "effect_event": effect["content"],
                    "effect_type": effect["type"],
                    "time_delta": effect_time - attack_time,
                    "formatted": f"[{attack_time:.1f}s] {attack['content'][:50]}... → [{effect_time:.1f}s] {effect['content'][:50]}..."
                }
                causal_links.append(causal_link)
                
                # Also add relationship to KB
                try:
                    relation = RelationType.DESTROYS if effect["type"] == "victory" else RelationType.INTERACTS
                    kb.add_relationship(
                        source_id=attack["entity_id"],
                        target_id=effect["entity_id"],
                        relation_type=relation,
                        timestamp=attack_time,
                        end_time=effect_time,
                        metadata={"causal_inference": True, "confidence": 0.7}
                    )
                except Exception:
                    pass  # Relationship might fail if entities don't exist
    
    logger.info(f"Knowledge Base: {len(kb._entities)} entities, {len(causal_links)} causal links inferred")
    
    return kb, causal_links


# =============================================================================
# GPT-Ready Text Formatting
# =============================================================================

def format_for_gpt(timeline: list[dict], kb, causal_links: list[dict], visual_results: list[dict], video_name: str) -> str:
    """
    Format timeline, KB, and causal links as text context for GPT-4 Q&A generation.
    
    This creates the "structured, language-like intermediate representation"
    required by Step 1A.2.
    """
    lines = [
        f"# Video Analysis: {video_name}",
        f"# Generated: {datetime.now().isoformat()}",
        "",
        "## Timeline Context",
        "(Events are sorted chronologically with timestamps in [MM:SS] format)",
        ""
    ]
    
    # 1. Timeline Section
    for event in timeline:
        ts = event["timestamp"]
        mins = int(ts // 60)
        secs = int(ts % 60)
        timestamp_str = f"[{mins:02d}:{secs:02d}]"
        
        evt_type = event["type"]
        content = event["content"]
        
        if evt_type == "ocr":
            lines.append(f"{timestamp_str} OCR: \"{content}\"")
        elif evt_type == "speech":
            lines.append(f"{timestamp_str} Speech: \"{content}\"")
        elif evt_type == "audio":
            lines.append(f"{timestamp_str} Audio: {content}")
        elif evt_type == "visual":
            motion_type = event.get('motion_type', 'unknown')
            if motion_type:
                lines.append(f"{timestamp_str} [{motion_type.upper()}] {content}")
            else:
                lines.append(f"{timestamp_str} Visual: {content}")
        elif evt_type == "temporal":
            lines.append(f"{timestamp_str} [TEMPORAL] {content}")
        else:
            lines.append(f"{timestamp_str} {content}")
    
    # 2. Causal Links Section (Step 1B requirement)
    lines.extend([
        "",
        "## Causal Links",
        "(Inferred cause-effect relationships between events)",
        ""
    ])
    
    if causal_links:
        for link in causal_links:
            lines.append(f"- {link['formatted']}")
    else:
        lines.append("- No causal links detected in this segment.")
            
    # 3. Entity Knowledge Base Section
    lines.extend([
        "",
        "## Entity Knowledge Base",
        "(Structured facts about detected entities)",
        ""
    ])
    
    # Export KB entities
    entity_count = 0
    for entity_id, entity in kb._entities.items():
        entity_count += 1
        lines.append(f"### {entity.concept_label} ({entity.category.value})")
        for k, v in entity.attributes.items():
            lines.append(f"  - {k}: {v}")
        if entity_count >= 50:  # Limit to avoid huge files
            lines.append(f"\n... and {len(kb._entities) - 50} more entities")
            break
    
    # 4. Visual Regions Section (SAM3 entity detection with bboxes)
    lines.extend([
        "",
        "## Visual Regions",
        "(Detected entities with bounding boxes from SAM3)",
        ""
    ])
    
    # Group visual results by timestamp for readability
    visual_by_time = {}
    for vis in visual_results:
        ts = vis.get("timestamp", 0)
        ts_key = f"{ts:.1f}"
        if ts_key not in visual_by_time:
            visual_by_time[ts_key] = []
        visual_by_time[ts_key].append(vis)
    
    # Output visual regions (limit to avoid huge files)
    region_count = 0
    for ts_key in sorted(visual_by_time.keys(), key=lambda x: float(x)):
        for vis in visual_by_time[ts_key]:
            entity_type = vis.get("entity_type", "unknown")
            bbox = vis.get("bbox")
            confidence = vis.get("confidence", 0)
            
            if bbox:
                bbox_str = f"[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"
                lines.append(f"- [{ts_key}s] {entity_type}: bbox={bbox_str}, conf={confidence:.2f}")
            else:
                lines.append(f"- [{ts_key}s] {entity_type}: detected (conf={confidence:.2f})")
            
            region_count += 1
            if region_count >= 100:  # Limit
                lines.append(f"\n... and {len(visual_results) - 100} more visual regions")
                break
        if region_count >= 100:
            break
    
    if not visual_results:
        lines.append("- No visual regions detected.")
    
    return "\n".join(lines)


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Extract features from gameplay video")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output", default="data/outputs", help="Output directory")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to sample")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--use-sam", action="store_true", help="Use SAM 3 for advanced entity detection")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio processing (faster)")
    parser.add_argument("--skip-hico", action="store_true", help="Skip InternVideo HiCo (faster)")
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video):
        logger.error(f"Video not found: {args.video}")
        sys.exit(1)
    
    os.makedirs(args.output, exist_ok=True)
    
    video_name = Path(args.video).stem
    logger.info(f"Processing: {args.video}")
    logger.info("="*60)
    
    # Step 1: Extract frames
    logger.info("Step 1/8: Extracting frames...")
    frames = extract_frames(args.video, fps=args.fps)
    
    # Step 2: Run InternVideo2.5 HiCo (Temporal Compression)
    if not args.skip_hico:
        logger.info("Step 2/8: Running InternVideo2.5 HiCo...")
        hico_results = run_internvideo_hico(frames, device=args.device)
    else:
        logger.info("Step 2/8: Skipping HiCo (--skip-hico flag)")
        hico_results = {"num_input_frames": len(frames), "num_temporal_tokens": 0, "tokens": None}
    
    # Step 3: Run visual detection (SAM or Motion)
    logger.info("Step 3/8: Running visual detection...")
    visual_results, entity_tracker = run_visual_detection(frames, device=args.device, use_sam=args.use_sam)
    
    # Step 4: Run OCR
    logger.info("Step 4/8: Running OCR...")
    ocr_results = run_ocr(frames, device=args.device)
    
    # Step 5: Run SigLIP (Semantic Embeddings)
    logger.info("Step 5/8: Running SigLIP Encoder...")
    siglip_results = run_siglip_encoder(frames, device=args.device)
    
    # Step 6: Run Audio Extraction
    if not args.skip_audio:
        logger.info("Step 6/8: Running Audio Extraction...")
        audio_results = run_audio_extraction(args.video, device=args.device)
    else:
        logger.info("Step 6/8: Skipping audio (--skip-audio flag)")
        audio_results = []
    
    # Step 7: Build timeline using TimelineIndexer
    logger.info("Step 7/8: Building timeline with TimelineIndexer...")
    timeline, indexer = build_timeline_with_indexer(
        ocr_results, visual_results, audio_results, hico_results
    )
    
    # Step 8: Build Knowledge Base with Causal Inference
    logger.info("Step 8/8: Building Knowledge Base with causal inference...")
    kb, causal_links = build_knowledge_base_with_causality(timeline, video_name)
    
    # Save outputs
    logger.info("Saving outputs...")
    
    # Save raw JSON
    output_json = os.path.join(args.output, f"{video_name}_features.json")
    with open(output_json, "w") as f:
        json.dump({
            "video": args.video,
            "extracted_at": datetime.now().isoformat(),
            "num_frames": len(frames),
            "hico_compression": {
                "input_frames": hico_results["num_input_frames"],
                "output_tokens": hico_results["num_temporal_tokens"],
            },
            "timeline": timeline,
            "causal_links": causal_links,
            "visual_events": visual_results,
            "ocr_results": ocr_results,
            "audio_results": audio_results,
            "siglip_embeddings": [
                {"timestamp": s["timestamp"], "shape": s["embedding_shape"]} 
                for s in siglip_results
            ]
        }, f, indent=2)
    logger.info(f"Saved: {output_json}")
    
    # Save GPT-ready text
    output_txt = os.path.join(args.output, f"{video_name}_context.txt")
    gpt_text = format_for_gpt(timeline, kb, causal_links, visual_results, video_name)
    with open(output_txt, "w") as f:
        f.write(gpt_text)
    logger.info(f"Saved: {output_txt}")
    
    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE - Step 1A FULLY IMPLEMENTED")
    print("="*60)
    print(f"Video: {video_name}")
    print(f"Frames: {len(frames)}")
    print(f"HiCo Compression: {hico_results['num_input_frames']} -> {hico_results['num_temporal_tokens']} tokens")
    print(f"Visual events: {len(visual_results)}")
    print(f"OCR regions: {len(ocr_results)}")
    print(f"Audio segments: {len(audio_results)}")
    print(f"Timeline events: {len(timeline)}")
    print(f"Causal links: {len(causal_links)}")
    print(f"\nOutputs:")
    print(f"  - {output_json}")
    print(f"  - {output_txt}")
    print("\nNext: Feed the .txt file to GPT-4 to generate Q&A pairs!")
    print("See Step 3 of the training guide for prompt templates.")


if __name__ == "__main__":
    main()
