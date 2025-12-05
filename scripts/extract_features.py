#!/usr/bin/env python3
"""
Feature Extraction Pipeline - Generates structured data from gameplay videos.

This script runs the perception pipeline on videos and outputs structured
JSON that can be fed to GPT-4 for generating instruction tuning data.

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


def run_siglip_encoder(frames: list[tuple[float, Image.Image]], device: str = "cuda"):
    """Run SigLIP2 on frames to get semantic embeddings."""
    from perception.siglip_semantic_encoder import SigLIPSemanticEncoder, NaFlexConfig
    
    config = NaFlexConfig(device=device)
    encoder = SigLIPSemanticEncoder(config)
    
    embeddings = []
    for timestamp, frame in frames:
        try:
            result = encoder.encode_regions([frame])
            if result:
                embeddings.append({
                    "timestamp": timestamp,
                    "embedding_shape": list(result[0].embedding.shape),
                })
        except Exception as e:
            logger.warning(f"SigLIP failed at {timestamp:.1f}s: {e}")
    
    logger.info(f"SigLIP encoded {len(embeddings)} frames")
    return embeddings


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


def run_sam_detection(frames: list[tuple[float, Image.Image]], device: str = "cuda"):
    """
    Run SAM 3 Concept Segmentation to detect specific game entities.
    """
    try:
        from perception.sam_concept_segmenter import SamConceptSegmenter, SAMConfig
        
        # Default prompts for gameplay
        prompts = ["character", "enemy", "boss", "health bar", "weapon"]
        
        config = SAMConfig(device=device)
        segmenter = SamConceptSegmenter(config)
        
        results = []
        logger.info(f"Loading SAM 3 model on {device}...")
        
        for idx, (timestamp, frame) in enumerate(frames):
            try:
                # Run segmentation for each prompt
                frame_events = []
                for prompt in prompts:
                    masks = segmenter.segment_frame(frame, idx, prompt)
                    
                    # If we found something with high confidence
                    for mask in masks:
                        if mask.confidence > 0.4:
                            frame_events.append({
                                "timestamp": timestamp,
                                "type": "entity_detection",
                                "description": f"Detected {prompt}",
                                "entity_type": prompt,
                                "confidence": mask.confidence,
                                "bbox": mask.bbox.to_xyxy() if mask.bbox else None,
                                "motion_score": 0.0 # Placeholder
                            })
                            # Only take the best one per prompt to avoid spam
                            break
                
                results.extend(frame_events)
                
                if idx % 10 == 0:
                    logger.info(f"SAM processed frame {idx}/{len(frames)}")
                    
            except Exception as e:
                logger.warning(f"SAM failed at {timestamp:.1f}s: {e}")
                
        logger.info(f"SAM detected {len(results)} entity events")
        return results
        
    except ImportError as e:
        logger.warning(f"SAM dependencies not met: {e}")
        return []
    except Exception as e:
        logger.warning(f"SAM initialization failed: {e}")
        return []


def run_visual_detection(frames: list[tuple[float, Image.Image]], device: str = "cuda", use_sam: bool = False):
    """
    Run visual analysis. Uses SAM 3 if requested, otherwise falls back to motion detection.
    """
    if use_sam:
        logger.info("Using SAM 3 for advanced entity detection...")
        return run_sam_detection(frames, device)
        
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
    return results


def build_timeline(
    frames: list[tuple[float, Image.Image]],
    ocr_results: list[dict],
    visual_results: list[dict],
) -> list[dict]:
    """
    Build a structured timeline from perception outputs.
    
    This creates the intermediate representation for LLM training.
    """
    timeline = []
    
    # Add OCR events
    for ocr in ocr_results:
        timeline.append({
            "timestamp": ocr["timestamp"],
            "type": "ocr",
            "content": ocr["text"],
            "confidence": ocr["confidence"],
        })
    
    # Add visual detection events (motion analysis)
    for vis in visual_results:
        timeline.append({
            "timestamp": vis["timestamp"],
            "type": "visual",
            "content": vis["description"],
            "motion_type": vis["type"],
            "motion_score": vis.get("motion_score", 0),
        })
    
    # Sort by timestamp
    timeline.sort(key=lambda x: x["timestamp"])
    
    return timeline


def build_knowledge_base(timeline: list[dict], video_name: str):
    """Build the entity knowledge base from timeline events."""
    from fusion_indexing.knowledge_base_builder import KnowledgeBaseBuilder, KnowledgeBaseConfig
    
    kb = KnowledgeBaseBuilder(KnowledgeBaseConfig())
    
    # Process timeline events into KB
    for event in timeline:
        timestamp = event["timestamp"]
        content = event["content"]
        evt_type = event["type"]
        
        # Add entities based on event type
        if evt_type == "ocr":
            # Link OCR text to video entity
            kb.add_entity(
                name=f"OCR_Event_{timestamp:.1f}",
                entity_type="text_event",
                attributes={"text": content, "timestamp": timestamp}
            )
        elif evt_type == "visual":
            # Link visual event to video entity
            kb.add_entity(
                name=f"Visual_Event_{timestamp:.1f}",
                entity_type="visual_event",
                attributes={"description": content, "timestamp": timestamp}
            )
            
    return kb


def format_for_gpt(timeline: list[dict], kb, video_name: str) -> str:
    """
    Format timeline and KB as text context for GPT-4 Q&A generation.
    """
    lines = [
        f"# Video Analysis: {video_name}",
        f"# Generated: {datetime.now().isoformat()}",
        "",
        "## Timeline Context",
        ""
    ]
    
    # 1. Timeline Section
    for event in timeline:
        ts = event["timestamp"]
        mins = int(ts // 60)
        secs = int(ts % 60)
        timestamp_str = f"[{mins:02d}:{secs:02d}]"
        
        if event["type"] == "ocr":
            lines.append(f"{timestamp_str} OCR: \"{event['content']}\" (conf: {event.get('confidence', 0):.2f})")
        elif event["type"] == "audio":
            lines.append(f"{timestamp_str} Audio: {event['content']}")
        elif event["type"] == "visual":
            motion_type = event.get('motion_type', 'unknown')
            lines.append(f"{timestamp_str} [{motion_type.upper()}] {event['content']}")
        else:
            lines.append(f"{timestamp_str} {event['content']}")
            
    # 2. Entity Knowledge Base Section
    lines.extend([
        "",
        "## Entity Knowledge Base",
        "(Structured facts and potential causal links)",
        ""
    ])
    
    # Export KB to text format
    # Since KB export might be complex, we'll do a simple dump of entities for now
    # In a real scenario, you'd use kb.export_to_text() or similar
    for entity_id, entity in kb.entities.items():
        lines.append(f"- Entity: {entity.name} ({entity.entity_type})")
        for k, v in entity.attributes.items():
            lines.append(f"  - {k}: {v}")
            
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Extract features from gameplay video")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output", default="data/outputs", help="Output directory")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to sample")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--use-sam", action="store_true", help="Use SAM 3 for advanced entity detection")
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video):
        logger.error(f"Video not found: {args.video}")
        sys.exit(1)
    
    os.makedirs(args.output, exist_ok=True)
    
    video_name = Path(args.video).stem
    logger.info(f"Processing: {args.video}")
    
    # Step 1: Extract frames
    logger.info("Step 1: Extracting frames...")
    frames = extract_frames(args.video, fps=args.fps)
    
    # Step 2: Run visual detection (SAM or Motion)
    logger.info("Step 2: Running visual detection...")
    visual_results = run_visual_detection(frames, device=args.device, use_sam=args.use_sam)
    
    # Step 3: Run OCR
    logger.info("Step 3: Running OCR...")
    ocr_results = run_ocr(frames, device=args.device)
    
    # Step 4: Run SigLIP (Semantic Embeddings)
    logger.info("Step 4: Running SigLIP Encoder...")
    siglip_results = run_siglip_encoder(frames, device=args.device)
    
    # Step 5: Build timeline
    logger.info("Step 5: Building timeline...")
    timeline = build_timeline(frames, ocr_results, visual_results, siglip_results)
    
    # Step 6: Build Knowledge Base
    logger.info("Step 6: Building Knowledge Base...")
    kb = build_knowledge_base(timeline, video_name)
    
    # Step 7: Save outputs
    logger.info("Step 7: Saving outputs...")
    
    # Save raw JSON
    output_json = os.path.join(args.output, f"{video_name}_features.json")
    with open(output_json, "w") as f:
        json.dump({
            "video": args.video,
            "extracted_at": datetime.now().isoformat(),
            "num_frames": len(frames),
            "timeline": timeline,
            "visual_events": visual_results,
            "ocr_results": ocr_results,
            "siglip_embeddings": [
                {"timestamp": s["timestamp"], "shape": s["embedding_shape"]} 
                for s in siglip_results
            ]
        }, f, indent=2)
    logger.info(f"Saved: {output_json}")
    
    # Save GPT-ready text
    output_txt = os.path.join(args.output, f"{video_name}_context.txt")
    gpt_text = format_for_gpt(timeline, kb, video_name)
    with open(output_txt, "w") as f:
        f.write(gpt_text)
    logger.info(f"Saved: {output_txt}")
    
    # Print summary
    print("\n" + "="*50)
    print(f"EXTRACTION COMPLETE")
    print("="*50)
    print(f"Video: {video_name}")
    print(f"Frames: {len(frames)}")
    print(f"OCR regions: {len(ocr_results)}")
    print(f"Timeline events: {len(timeline)}")
    print(f"\nOutputs:")
    print(f"  - {output_json}")
    print(f"  - {output_txt}")
    print("\nNext: Feed the .txt file to GPT-4 to generate Q&A pairs!")


if __name__ == "__main__":
    main()
