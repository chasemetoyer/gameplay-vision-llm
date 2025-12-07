#!/usr/bin/env python3
"""
Real-Time Video Inference for Mac (Apple Silicon) - FULL PIPELINE.

Optimized for M1/M2/M3/M4 Macs using:
- MLX for Qwen3-VL (4-bit quantized, ~5GB)
- MPS (Metal) for SAM3, SigLIP, VideoMAE (~7GB)
- CPU for OCR
- MPS/CPU for Whisper

Total memory: ~16GB (fits in 24GB M4 Pro)

Usage:
    python scripts/realtime_inference_mac.py --video gameplay.mp4 --interactive
    python scripts/realtime_inference_mac.py --video gameplay.mp4 --interactive --use-sam
"""

import argparse
import logging
import os
import sys
import re
import tempfile
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress noisy logs
for name in ["httpx", "httpcore", "huggingface_hub", "transformers", "urllib3", "filelock"]:
    logging.getLogger(name).setLevel(logging.WARNING)

# Suppress tqdm progress bars from huggingface_hub during inference
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


def get_device():
    """Get the best available device for Mac."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract_frames(video_path: str, fps: float = 0.5) -> list[tuple[float, Image.Image]]:
    """Extract frames from video at specified FPS."""
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    
    logger.info(f"Video: {duration:.1f}s, {video_fps:.0f} fps, {total_frames} frames")
    
    frame_interval = max(1, int(video_fps / fps))
    frames = []
    frame_idx = 0
    
    pbar = tqdm(total=total_frames // frame_interval, desc="Extracting frames")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frames.append((timestamp, frame_pil))
            pbar.update(1)
        
        frame_idx += 1
    
    pbar.close()
    cap.release()
    
    logger.info(f"Extracted {len(frames)} frames at {fps} FPS")
    return frames


def run_sam3_detection_mac(frames: list[tuple[float, Image.Image]], device: str = "mps"):
    """
    Run SAM3 for visual entity detection on Mac with MPS.
    """
    print("=" * 50)
    print("🎯 SAM3 ENTITY DETECTION (Mac MPS)")
    print("=" * 50)
    
    try:
        from perception.sam_concept_segmenter import SAMConceptSegmenter, SAMConfig
        
        print(f"   [1/3] Loading SAM3 on {device}...")
        config = SAMConfig(device=device)
        segmenter = SAMConceptSegmenter(config)
        
        # Gameplay concepts
        concepts = ["player", "enemy", "boss", "health bar", "weapon", "character", "object"]
        
        all_detections = []
        
        print(f"   [2/3] Processing {len(frames)} frames...")
        for timestamp, frame in tqdm(frames, desc="SAM3 detection"):
            frame_np = np.array(frame)
            frame_idx = int(timestamp * 10)
            
            try:
                entities = segmenter.segment_with_prompts(
                    frame=frame_np,
                    frame_idx=frame_idx,
                    concept_prompts=concepts,
                )
                
                detections = []
                for entity in entities:
                    detections.append({
                        "label": entity.concept_label,
                        "bbox": entity.bbox,
                        "confidence": entity.confidence,
                    })
                
                all_detections.append({
                    "timestamp": timestamp,
                    "detections": detections,
                })
                
            except Exception as e:
                logger.debug(f"SAM3 error at {timestamp:.1f}s: {e}")
                all_detections.append({"timestamp": timestamp, "detections": []})
        
        total_det = sum(len(d["detections"]) for d in all_detections)
        print(f"   [3/3] ✅ Detected {total_det} entities across {len(frames)} frames")
        
        return all_detections
        
    except Exception as e:
        print(f"   ❌ SAM3 failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def extract_siglip_embeddings_mac(
    frames: list[tuple[float, Image.Image]], 
    sam_results: list[dict] = None,
    device: str = "mps"
):
    """
    Extract SigLIP embeddings on Mac with MPS.
    """
    print("=" * 50)
    print("🖼️ SIGLIP EMBEDDINGS (Mac MPS)")
    print("=" * 50)
    
    try:
        from perception.siglip_semantic_encoder import SigLIPSemanticEncoder, NaFlexConfig
        
        print(f"   [1/3] Loading SigLIP on {device}...")
        config = NaFlexConfig(device=device)
        encoder = SigLIPSemanticEncoder(config)
        
        embeddings = []
        
        print(f"   [2/3] Processing {len(frames)} frames...")
        
        if sam_results:
            # Extract from SAM regions
            for frame_data, sam_data in zip(frames, sam_results):
                timestamp, frame = frame_data
                
                for det in sam_data.get("detections", [])[:5]:  # Limit per frame
                    try:
                        bbox = det.get("bbox")
                        if bbox:
                            x1, y1, x2, y2 = [int(c) for c in bbox]
                            region = frame.crop((x1, y1, x2, y2))
                            
                            emb = encoder.encode_image(region)
                            embeddings.append({
                                "timestamp": timestamp,
                                "embedding": emb,
                                "label": det.get("label", "region"),
                            })
                    except Exception as e:
                        logger.debug(f"SigLIP region error: {e}")
        else:
            # Full frame embeddings
            for timestamp, frame in tqdm(frames, desc="SigLIP"):
                try:
                    emb = encoder.encode_image(frame)
                    embeddings.append({
                        "timestamp": timestamp,
                        "embedding": emb,
                        "label": "frame",
                    })
                except Exception as e:
                    logger.debug(f"SigLIP error at {timestamp:.1f}s: {e}")
        
        print(f"   [3/3] ✅ Generated {len(embeddings)} embeddings")
        return embeddings
        
    except Exception as e:
        print(f"   ❌ SigLIP failed: {e}")
        return []


def extract_videomae_embeddings_mac(frames: list[tuple[float, Image.Image]], device: str = "mps"):
    """
    Extract VideoMAE temporal embeddings on Mac with MPS.
    """
    print("=" * 50)
    print("🎬 VIDEOMAE EMBEDDINGS (Mac MPS)")
    print("=" * 50)
    
    try:
        from transformers import VideoMAEModel, VideoMAEImageProcessor
        
        print(f"   [1/3] Loading VideoMAE on {device}...")
        processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device).eval()
        
        embeddings = []
        window_size = 16  # VideoMAE window
        
        print(f"   [2/3] Processing {len(frames)} frames in windows of {window_size}...")
        
        for i in tqdm(range(0, len(frames), window_size // 2), desc="VideoMAE"):
            window_frames = frames[i:i + window_size]
            if len(window_frames) < 4:
                continue
            
            try:
                timestamp = window_frames[len(window_frames) // 2][0]
                pil_frames = [f[1] for f in window_frames]
                
                inputs = processor(pil_frames, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
                
                embeddings.append({
                    "timestamp": timestamp,
                    "embedding": embedding.cpu(),
                })
                
            except Exception as e:
                logger.debug(f"VideoMAE error at window {i}: {e}")
        
        print(f"   [3/3] ✅ Generated {len(embeddings)} temporal embeddings")
        return embeddings
        
    except Exception as e:
        print(f"   ❌ VideoMAE failed: {e}")
        return []


def run_ocr_extraction(frames: list[tuple[float, Image.Image]]):
    """Run OCR on frames - CPU for Mac."""
    print("=" * 50)
    print("📝 OCR TEXT EXTRACTION (CPU)")
    print("=" * 50)
    
    try:
        from perception.ocr_pipeline import OCRPipeline, OCRConfig
        
        print("   [1/3] Loading PaddleOCR...")
        config = OCRConfig(use_gpu=False)
        ocr = OCRPipeline(config)
        
        results = []
        print(f"   [2/3] Processing {len(frames)} frames...")
        
        for idx, (timestamp, frame) in enumerate(tqdm(frames, desc="OCR")):
            try:
                frame_np = np.array(frame.convert("RGB"))
                ocr_result = ocr.extract_text_from_frame(frame_np, idx, timestamp)
                if ocr_result and ocr_result.detections:
                    for det in ocr_result.detections:
                        results.append({
                            "timestamp": timestamp,
                            "text": det.text,
                            "confidence": det.confidence,
                            "type": "ocr",
                        })
            except Exception as e:
                logger.debug(f"OCR error at {timestamp:.1f}s: {e}")
        
        print(f"   [3/3] ✅ Extracted {len(results)} text regions")
        return results
        
    except ImportError:
        print("   ⚠️ PaddleOCR not available, skipping OCR")
        return []


def run_speech_transcription_mac(video_path: str, device: str = "mps"):
    """
    Extract speech transcription using Whisper on Mac.
    """
    print("=" * 50)
    print("🎤 WHISPER SPEECH TRANSCRIPTION (Mac)")
    print("=" * 50)
    
    try:
        import whisper
        import subprocess
        
        print("   [1/4] Extracting audio from video...")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
               "-ar", "16000", "-ac", "1", tmp_path]
        result = subprocess.run(cmd, capture_output=True)
        
        if result.returncode != 0:
            print("   ⚠️ FFmpeg failed")
            return []
        
        # Use base model for Mac (good balance of speed/quality)
        print("   [2/4] Loading Whisper (base model)...")
        model = whisper.load_model("base")
        
        print("   [3/4] Transcribing audio...")
        result = model.transcribe(tmp_path, language="en", verbose=False)
        
        os.unlink(tmp_path)
        
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "timestamp": seg["start"],
                "end_time": seg["end"],
                "text": seg["text"].strip(),
                "type": "speech",
            })
        
        print(f"   [4/4] ✅ Transcribed {len(segments)} speech segments")
        return segments
        
    except Exception as e:
        print(f"   ❌ Speech transcription failed: {e}")
        return []


def build_timeline_index(ocr_results: list, speech_results: list, sam_results: list = None):
    """Build timeline from all modalities."""
    from fusion_indexing.timeline_indexer import (
        TimelineIndexer, ModalityType, EventPriority
    )
    
    print("=" * 50)
    print("📋 BUILDING TIMELINE INDEX")
    print("=" * 50)
    
    indexer = TimelineIndexer()
    
    for ocr in ocr_results:
        indexer.add_event(
            timestamp=ocr["timestamp"],
            modality=ModalityType.OCR,
            description=ocr["text"],
            confidence=ocr.get("confidence", 0.8),
            priority=EventPriority.MEDIUM,
        )
    
    for speech in speech_results:
        indexer.add_event(
            timestamp=speech["timestamp"],
            modality=ModalityType.SPEECH,
            description=speech["text"],
            duration=speech.get("end_time", speech["timestamp"]) - speech["timestamp"],
            priority=EventPriority.HIGH,
        )
    
    if sam_results:
        for sam in sam_results:
            for det in sam.get("detections", []):
                indexer.add_event(
                    timestamp=sam["timestamp"],
                    modality=ModalityType.VISUAL,
                    description=f"Detected: {det['label']}",
                    confidence=det.get("confidence", 0.7),
                    priority=EventPriority.MEDIUM,
                )
    
    indexer.merge_and_dedupe()
    
    stats = indexer.get_statistics()
    print(f"   ✅ Timeline: {stats['total_events']} events indexed")
    
    return indexer


def process_video_mac(
    video_path: str,
    fps: float = 0.5,
    use_sam: bool = False,
):
    """
    Process video for Mac - FULL PIPELINE with MPS acceleration.
    
    Args:
        video_path: Path to video file
        fps: Frames per second to sample
        use_sam: Enable SAM3 detection (adds ~4GB memory)
    """
    from agent_core.mlx_reasoning_core import MLXQwenVLCore, MLXConfig
    
    device = get_device()
    print("\n" + "=" * 60)
    print("🍎 PROCESSING VIDEO (Mac Full Pipeline)")
    print("=" * 60)
    print(f"   Device: {device.upper()}")
    print(f"   SAM3: {'Enabled' if use_sam else 'Disabled'}")
    print("=" * 60)
    
    # 1. Extract frames
    print("\n[Step 1/7] Extracting frames...")
    frames = extract_frames(video_path, fps=fps)
    
    # 2. SAM3 detection (optional)
    sam_results = None
    if use_sam:
        print("\n[Step 2/7] Running SAM3 detection...")
        sam_results = run_sam3_detection_mac(frames, device)
    else:
        print("\n[Step 2/7] Skipping SAM3 (use --use-sam to enable)")
    
    # 3. SigLIP embeddings
    print("\n[Step 3/7] Extracting SigLIP embeddings...")
    siglip_embs = extract_siglip_embeddings_mac(frames, sam_results, device)
    
    # 4. VideoMAE embeddings
    print("\n[Step 4/7] Extracting VideoMAE embeddings...")
    videomae_embs = extract_videomae_embeddings_mac(frames, device)
    
    # 5. OCR
    print("\n[Step 5/7] Extracting text (OCR)...")
    ocr_results = run_ocr_extraction(frames)
    
    # 6. Speech transcription
    print("\n[Step 6/7] Transcribing speech...")
    speech_results = run_speech_transcription_mac(video_path, device)
    
    # 7. Build timeline
    print("\n[Step 7/7] Building timeline index...")
    timeline_indexer = build_timeline_index(ocr_results, speech_results, sam_results)
    
    # Initialize MLX reasoning core (loaded lazily)
    print("\n[Loading] Initializing MLX Qwen3-VL (4-bit)...")
    config = MLXConfig()
    reasoning_core = MLXQwenVLCore(config)
    
    # Store data for interactive mode
    context = {
        "frames": frames,
        "sam_results": sam_results,
        "siglip_embs": siglip_embs,
        "videomae_embs": videomae_embs,
        "ocr_results": ocr_results,
        "speech_results": speech_results,
        "timeline_indexer": timeline_indexer,
        "reasoning_core": reasoning_core,
    }
    
    print("\n" + "=" * 60)
    print("✅ VIDEO PROCESSING COMPLETE (Mac Full Pipeline)")
    print("=" * 60)
    print(f"   📹 Frames extracted: {len(frames)}")
    if sam_results:
        total_det = sum(len(d["detections"]) for d in sam_results)
        print(f"   🎯 SAM3 detections: {total_det}")
    print(f"   🖼️  SigLIP embeddings: {len(siglip_embs)}")
    print(f"   🎬 VideoMAE embeddings: {len(videomae_embs)}")
    print(f"   📝 OCR text regions: {len(ocr_results)}")
    print(f"   🎤 Speech segments: {len(speech_results)}")
    print(f"   📋 Timeline events: {timeline_indexer.get_statistics()['total_events']}")
    print(f"   🧠 LLM: MLX Qwen3-VL (4-bit)")
    print("=" * 60)
    
    return context


def format_timeline_context(timeline_indexer, query: str = "") -> str:
    """Format timeline events as context text."""
    events = timeline_indexer._events if hasattr(timeline_indexer, "_events") else []
    
    if not events:
        return "No events in timeline."
    
    sorted_events = sorted(events, key=lambda e: e.timestamp)
    
    lines = []
    for event in sorted_events[:50]:
        mins = int(event.timestamp // 60)
        secs = int(event.timestamp % 60)
        lines.append(f"[{mins:02d}:{secs:02d}] {event.description}")
    
    return "\n".join(lines)


def interactive_mode_mac(context: dict):
    """Interactive Q&A session with streaming output (Mac)."""
    print("\n" + "=" * 60)
    print("🎮 INTERACTIVE GAMEPLAY ANALYSIS (Mac Full Pipeline)")
    print("=" * 60)
    print("Commands:")
    print("  @<MM:SS> <question>  - Ask about specific timestamp")
    print("  <question>           - Ask about whole video")
    print("  quit                 - Exit")
    print("=" * 60 + "\n")
    
    reasoning_core = context["reasoning_core"]
    timeline_indexer = context["timeline_indexer"]
    frames = context["frames"]
    
    while True:
        try:
            user_input = input("🎮 Your question: ").strip()
            
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Parse timestamp if provided
            timestamp = None
            query = user_input
            
            ts_match = re.match(r'@(\d+):(\d+)\s+(.*)', user_input)
            if ts_match:
                mins, secs, query = ts_match.groups()
                timestamp = int(mins) * 60 + int(secs)
                print(f"📍 Focusing on timestamp: {mins}:{secs}")
            
            print("\n🔍 Analyzing...\n")
            print("📝 Response:")
            
            # Get frame for context
            frame = None
            if timestamp is not None and frames:
                closest_frame = min(frames, key=lambda f: abs(f[0] - timestamp))
                frame = closest_frame[1]
            elif frames:
                frame = frames[len(frames) // 2][1]
            
            # Get timeline context
            timeline_context = format_timeline_context(timeline_indexer, query)
            
            # Stream response
            for token in reasoning_core.reason_streaming(
                query=query,
                current_frame=frame,
                timeline_context=timeline_context,
            ):
                print(token, end="", flush=True)
            
            print("\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Gameplay video inference (Mac Full Pipeline)")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--interactive", action="store_true", help="Interactive Q&A mode")
    parser.add_argument("--query", help="Single question to ask")
    parser.add_argument("--fps", type=float, default=0.5, help="Frames per second to sample")
    parser.add_argument("--use-sam", action="store_true", help="Enable SAM3 detection (+4GB memory)")
    
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video):
        print(f"❌ Error: Video not found: {args.video}")
        sys.exit(1)
    
    # Process video
    context = process_video_mac(args.video, fps=args.fps, use_sam=args.use_sam)
    
    if args.interactive:
        interactive_mode_mac(context)
    elif args.query:
        timeline_context = format_timeline_context(context["timeline_indexer"])
        frame = context["frames"][len(context["frames"]) // 2][1] if context["frames"] else None
        
        print("\n🔍 Analyzing...\n")
        response = context["reasoning_core"].reason(
            query=args.query,
            current_frame=frame,
            timeline_context=timeline_context,
        )
        print(f"📝 Response:\n{response}")
    else:
        print("❌ Error: Provide --query or use --interactive mode")
        sys.exit(1)


if __name__ == "__main__":
    main()
