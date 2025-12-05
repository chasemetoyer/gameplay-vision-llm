#!/usr/bin/env python3
"""
End-to-End Inference Pipeline for Gameplay Video LLM.

This script implements the complete agentic reasoning system:
1. Perception: SAM3, SigLIP, OCR (InternVideo/Audio when available)
2. Fusion: TimelineIndexer + KnowledgeBaseBuilder
3. Retrieval: TimelineRetriever with hybrid time/semantic search
4. Reasoning: Qwen-VL with ProjectorBank alignment

Usage:
    python scripts/inference.py --video path/to/video.mp4 --query "What happens at 02:30?"
    python scripts/inference.py --video path/to/video.mp4 --interactive
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Video Frame Extraction
# =============================================================================

def extract_frames(video_path: str, fps: float = 0.5) -> list[tuple[float, Image.Image]]:
    """Extract frames from video at specified FPS."""
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    
    logger.info(f"Video: {duration:.1f}s, {video_fps:.0f} fps, {total_frames} frames")
    
    # Calculate frame interval
    frame_interval = int(video_fps / fps) if fps < video_fps else 1
    
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append((timestamp, pil_image))
        
        frame_idx += 1
    
    cap.release()
    logger.info(f"Extracted {len(frames)} frames at {fps} FPS")
    return frames


# =============================================================================
# Perception Pipeline
# =============================================================================

def run_perception_pipeline(
    frames: list[tuple[float, Image.Image]],
    video_path: str,
    device: str = "cuda",
    use_sam: bool = True,
) -> dict:
    """
    Run the complete perception pipeline on video frames.
    
    Returns dict with: ocr_results, visual_results, siglip_embeddings, audio_results
    """
    results = {
        "ocr_results": [],
        "visual_results": [],
        "siglip_embeddings": [],
        "audio_results": [],
        "hico_results": {"num_input_frames": len(frames), "num_temporal_tokens": len(frames) // 2}
    }
    
    # 1. OCR (PaddleOCR)
    logger.info("Running OCR...")
    try:
        from perception.ocr_pipeline import OCRPipeline, OCRConfig
        import numpy as np
        
        config = OCRConfig(use_gpu=(device == "cuda"))
        ocr = OCRPipeline(config)
        
        for timestamp, frame in frames:
            try:
                frame_np = np.array(frame)
                text_regions = ocr.extract_text(frame_np)
                for region in text_regions:
                    results["ocr_results"].append({
                        "timestamp": timestamp,
                        "text": region.text,
                        "bbox": region.bbox,
                        "confidence": region.confidence,
                    })
            except Exception as e:
                logger.debug(f"OCR failed at {timestamp:.1f}s: {e}")
        
        logger.info(f"OCR: {len(results['ocr_results'])} text regions")
    except Exception as e:
        logger.warning(f"OCR pipeline failed: {e}")
    
    # 2. Visual Detection (SAM3 or Motion)
    logger.info("Running visual detection...")
    if use_sam:
        try:
            from perception.sam_concept_segmenter import Sam3ModelWrapper, SAMConfig
            
            prompts = ["character", "enemy", "boss", "health bar", "weapon"]
            config = SAMConfig(device=device)
            segmenter = Sam3ModelWrapper(config)
            
            for idx, (timestamp, frame) in enumerate(frames):
                try:
                    for prompt in prompts:
                        masks = segmenter.segment_with_text(frame, prompt)
                        for mask in masks:
                            if isinstance(mask, dict):
                                scores = mask.get("scores")
                                confidence = float(scores[0]) if scores is not None and len(scores) > 0 else 0.5
                                boxes = mask.get("boxes")
                                bbox = boxes[0].tolist() if boxes is not None and len(boxes) > 0 else None
                            else:
                                confidence = mask.confidence
                                bbox = mask.bbox.to_xyxy() if mask.bbox else None
                            
                            if confidence > 0.4:
                                results["visual_results"].append({
                                    "timestamp": timestamp,
                                    "entity_type": prompt,
                                    "confidence": confidence,
                                    "bbox": bbox,
                                })
                                break  # One per prompt
                    
                    if idx % 20 == 0:
                        logger.info(f"SAM3 processed frame {idx}/{len(frames)}")
                except Exception as e:
                    logger.debug(f"SAM3 failed at {timestamp:.1f}s: {e}")
            
            logger.info(f"SAM3: {len(results['visual_results'])} entities")
        except Exception as e:
            logger.warning(f"SAM3 failed: {e}, using motion detection")
            use_sam = False
    
    if not use_sam:
        # Fallback to motion detection
        import numpy as np
        prev_frame = None
        for timestamp, frame in frames:
            frame_np = np.array(frame.convert("L"))
            if prev_frame is not None:
                diff = np.abs(frame_np.astype(float) - prev_frame.astype(float))
                motion = diff.mean()
                if motion > 15:
                    results["visual_results"].append({
                        "timestamp": timestamp,
                        "entity_type": "motion",
                        "description": "Movement detected",
                        "motion_score": float(motion),
                    })
            prev_frame = frame_np
    
    # 3. SigLIP Embeddings
    logger.info("Running SigLIP encoder...")
    try:
        from perception.siglip_semantic_encoder import SigLIPSemanticEncoder, NaFlexConfig
        
        config = NaFlexConfig(device=device)
        encoder = SigLIPSemanticEncoder(config)
        
        for timestamp, frame in frames:
            try:
                embedding = encoder.encode_image(frame)
                if embedding is not None:
                    results["siglip_embeddings"].append({
                        "timestamp": timestamp,
                        "embedding": embedding.cpu(),
                        "shape": list(embedding.shape),
                    })
            except Exception as e:
                logger.debug(f"SigLIP failed at {timestamp:.1f}s: {e}")
        
        logger.info(f"SigLIP: {len(results['siglip_embeddings'])} embeddings")
    except Exception as e:
        logger.warning(f"SigLIP failed: {e}")
    
    return results


# =============================================================================
# Fusion & Indexing
# =============================================================================

def build_timeline_and_kb(
    perception_results: dict,
    video_name: str,
) -> tuple:
    """Build timeline index and knowledge base from perception outputs."""
    from fusion_indexing.timeline_indexer import TimelineIndexer, TimelineConfig
    from fusion_indexing.knowledge_base_builder import KnowledgeBaseBuilder, KnowledgeBaseConfig, EntityCategory
    
    # Initialize components
    timeline_config = TimelineConfig()
    indexer = TimelineIndexer(timeline_config)
    kb = KnowledgeBaseBuilder(KnowledgeBaseConfig())
    
    # Add OCR events
    for ocr in perception_results["ocr_results"]:
        indexer.add_event(
            timestamp=ocr["timestamp"],
            event_type="ocr",
            content=f"Text: \"{ocr['text']}\"",
            confidence=ocr.get("confidence", 1.0),
        )
    
    # Add visual events
    for vis in perception_results["visual_results"]:
        description = vis.get("description", f"Detected {vis.get('entity_type', 'object')}")
        indexer.add_event(
            timestamp=vis["timestamp"],
            event_type="visual",
            content=description,
            confidence=vis.get("confidence", 1.0),
        )
    
    # Add audio events
    for audio in perception_results.get("audio_results", []):
        indexer.add_event(
            timestamp=audio["timestamp"],
            event_type="audio" if audio.get("type") == "sound_event" else "speech",
            content=audio.get("text", ""),
            confidence=1.0,
        )
    
    # Build timeline
    timeline = indexer.get_chronological_events()
    
    # Build knowledge base from timeline
    for event in timeline:
        entity_id = f"{event['type']}_{event['timestamp']:.1f}".replace(".", "_")
        category = EntityCategory.UI_ELEMENT if event["type"] == "ocr" else EntityCategory.EFFECT
        kb.register_entity(
            entity_id=entity_id,
            concept_label=f"{event['type'].upper()}_{event['timestamp']:.1f}",
            category=category,
            timestamp=event["timestamp"],
            attributes={"content": event["content"], "type": event["type"]}
        )
    
    logger.info(f"Timeline: {len(timeline)} events, KB: {len(kb._entities)} entities")
    return indexer, kb, timeline


# =============================================================================
# Retrieval
# =============================================================================

def retrieve_context(
    query: str,
    timeline_indexer,
    kb,
    max_tokens: int = 4000,
) -> str:
    """Retrieve relevant context for a query."""
    from agent_core.qwen_reasoning_core import TimelineRetriever, ReasoningCoreConfig
    
    config = ReasoningCoreConfig()
    retriever = TimelineRetriever(config)
    
    # Index timeline
    retriever.index_timeline(timeline_indexer)
    
    # Retrieve relevant events
    events = retriever.hybrid_retrieve(query, timeline_indexer)
    
    # Format as context text
    context_parts = []
    for event in events[:50]:  # Limit to avoid token overflow
        timestamp = event.timestamp if hasattr(event, 'timestamp') else event.get('timestamp', 0)
        content = event.content if hasattr(event, 'content') else event.get('content', '')
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        context_parts.append(f"[{minutes:02d}:{seconds:02d}] {content}")
    
    context = "\n".join(context_parts)
    logger.info(f"Retrieved {len(events)} events for query")
    return context


# =============================================================================
# Reasoning with Qwen-VL
# =============================================================================

def reason_with_qwen(
    query: str,
    context: str,
    device: str = "cuda",
) -> str:
    """Use Qwen-VL to answer the query based on context."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info("Loading Qwen model for reasoning...")
        
        # Try to load Qwen model
        model_name = "Qwen/Qwen2.5-7B-Instruct"  # Text-only for now
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Format prompt
        system_prompt = """You are a gameplay video analyst. You have access to a timeline of events extracted from a gameplay video. Answer questions based on the provided context.

When referencing events, cite their timestamps in [MM:SS] format.
Be specific and accurate based on the data provided."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""## Video Timeline Context:
{context}

## Question:
{query}

## Answer:"""}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()
        
    except Exception as e:
        logger.warning(f"Qwen reasoning failed: {e}")
        return f"[Reasoning unavailable: {e}]\n\nContext summary: {len(context.split(chr(10)))} events found."


# =============================================================================
# Main Pipeline
# =============================================================================

def run_inference(
    video_path: str,
    query: str,
    device: str = "cuda",
    fps: float = 0.5,
    use_sam: bool = True,
    cache_dir: Optional[str] = None,
) -> str:
    """
    Complete end-to-end inference pipeline.
    
    Args:
        video_path: Path to video file
        query: User question about the video
        device: cuda or cpu
        fps: Frames per second to sample
        use_sam: Use SAM3 for visual detection
        cache_dir: Directory to cache extraction results
        
    Returns:
        Answer string
    """
    video_name = Path(video_path).stem
    cache_file = None
    
    # Check for cached results
    if cache_dir:
        cache_file = Path(cache_dir) / f"{video_name}_cache.pt"
        if cache_file.exists():
            logger.info(f"Loading cached results from {cache_file}")
            cached = torch.load(cache_file)
            perception_results = cached["perception"]
            indexer = cached["indexer"]
            kb = cached["kb"]
            timeline = cached["timeline"]
        else:
            perception_results = None
    else:
        perception_results = None
    
    # Run pipeline if not cached
    if perception_results is None:
        # 1. Extract frames
        logger.info("=" * 60)
        logger.info("Step 1: Extracting frames...")
        frames = extract_frames(video_path, fps)
        
        # 2. Run perception
        logger.info("=" * 60)
        logger.info("Step 2: Running perception pipeline...")
        perception_results = run_perception_pipeline(
            frames, video_path, device, use_sam
        )
        
        # 3. Build timeline and KB
        logger.info("=" * 60)
        logger.info("Step 3: Building timeline and knowledge base...")
        indexer, kb, timeline = build_timeline_and_kb(perception_results, video_name)
        
        # Cache results
        if cache_file:
            os.makedirs(cache_file.parent, exist_ok=True)
            torch.save({
                "perception": perception_results,
                "indexer": indexer,
                "kb": kb,
                "timeline": timeline,
            }, cache_file)
            logger.info(f"Cached results to {cache_file}")
    
    # 4. Retrieve context
    logger.info("=" * 60)
    logger.info(f"Step 4: Retrieving context for query: '{query}'")
    context = retrieve_context(query, indexer, kb)
    
    # 5. Reason with Qwen
    logger.info("=" * 60)
    logger.info("Step 5: Reasoning with Qwen...")
    answer = reason_with_qwen(query, context, device)
    
    return answer


def interactive_mode(video_path: str, device: str, fps: float, use_sam: bool):
    """Interactive Q&A mode."""
    video_name = Path(video_path).stem
    
    print(f"\n{'='*60}")
    print(f"Interactive Mode: {video_name}")
    print("Type 'quit' or 'exit' to stop")
    print(f"{'='*60}\n")
    
    # Pre-process video
    cache_dir = "data/cache"
    
    while True:
        try:
            query = input("\n🎮 Your question: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not query:
                continue
            
            print("\n🔍 Processing...")
            answer = run_inference(
                video_path, query, device, fps, use_sam, cache_dir
            )
            print(f"\n📝 Answer:\n{answer}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="End-to-end gameplay video inference")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--query", help="Question to ask about the video")
    parser.add_argument("--interactive", action="store_true", help="Interactive Q&A mode")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--fps", type=float, default=0.5, help="Frames per second to sample")
    parser.add_argument("--use-sam", action="store_true", help="Use SAM3 for detection")
    parser.add_argument("--cache-dir", default="data/cache", help="Cache directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)
    
    if args.interactive:
        interactive_mode(args.video, args.device, args.fps, args.use_sam)
    elif args.query:
        answer = run_inference(
            args.video, args.query, args.device, args.fps, 
            args.use_sam, args.cache_dir
        )
        print(f"\n📝 Answer:\n{answer}")
    else:
        print("Error: Provide --query or use --interactive mode")
        sys.exit(1)


if __name__ == "__main__":
    main()
