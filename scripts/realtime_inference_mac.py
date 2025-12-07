#!/usr/bin/env python3
"""
Real-Time Video Inference for Mac (Apple Silicon).

Optimized for M1/M2/M3/M4 Macs using MLX for quantized inference.
Uses ~13-14GB memory with 4-bit quantized models.

Usage:
    python scripts/realtime_inference_mac.py --video gameplay.mp4 --interactive
"""

import argparse
import logging
import os
import sys
import re
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress noisy logs
for name in ["httpx", "httpcore", "huggingface_hub", "transformers", "urllib3"]:
    logging.getLogger(name).setLevel(logging.WARNING)


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


def run_ocr_extraction(frames: list[tuple[float, Image.Image]]):
    """Run OCR on frames - simplified for Mac."""
    logger.info("Running OCR extraction...")
    
    try:
        from perception.ocr_pipeline import OCRPipeline, OCRConfig
        import numpy as np
        
        # Force CPU for OCR on Mac
        config = OCRConfig(use_gpu=False)
        ocr = OCRPipeline(config)
        
        results = []
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
        
        logger.info(f"OCR: {len(results)} text regions extracted")
        return results
        
    except ImportError:
        logger.warning("OCR pipeline not available, skipping OCR")
        return []


def run_speech_transcription_mac(video_path: str):
    """
    Extract speech transcription using Whisper (CPU/MPS for Mac).
    Uses smaller model for efficiency.
    """
    print("=" * 50)
    print("🎤 WHISPER SPEECH TRANSCRIPTION (Mac)")
    print("=" * 50)
    
    try:
        import whisper
        import subprocess
        import tempfile
        
        print("   [1/4] Extracting audio from video...")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
               "-ar", "16000", "-ac", "1", tmp_path]
        result = subprocess.run(cmd, capture_output=True)
        
        if result.returncode != 0:
            print(f"   ⚠️ FFmpeg failed")
            return []
        
        # Use smaller model for Mac
        print("   [2/4] Loading Whisper model (base - Mac optimized)...")
        model = whisper.load_model("base")  # Smaller model for Mac
        
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


def build_timeline_index(ocr_results: list, speech_results: list):
    """Build timeline from OCR and speech."""
    from fusion_indexing.timeline_indexer import (
        TimelineIndexer, ModalityType, EventPriority
    )
    
    logger.info("Building timeline index...")
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
    
    indexer.merge_and_dedupe()
    
    stats = indexer.get_statistics()
    logger.info(f"Timeline: {stats['total_events']} events indexed")
    
    return indexer


def process_video_mac(
    video_path: str,
    fps: float = 0.5,
):
    """
    Process video for Mac - optimized pipeline without heavy GPU models.
    
    Skips SAM3, SigLIP, VideoMAE to save memory.
    Uses OCR + Speech + MLX Qwen for reasoning.
    """
    from agent_core.mlx_reasoning_core import MLXQwenVLCore, MLXConfig
    
    logger.info("=" * 60)
    logger.info("PROCESSING VIDEO (Mac Optimized)")
    logger.info("=" * 60)
    
    # 1. Extract frames
    logger.info("\n1. Extracting frames...")
    frames = extract_frames(video_path, fps=fps)
    
    # 2. Run OCR (CPU)
    logger.info("\n2. Extracting text (OCR)...")
    ocr_results = run_ocr_extraction(frames)
    
    # 3. Run speech transcription
    logger.info("\n3. Transcribing speech (Whisper)...")
    speech_results = run_speech_transcription_mac(video_path)
    
    # 4. Build timeline
    logger.info("\n4. Building timeline index...")
    timeline_indexer = build_timeline_index(ocr_results, speech_results)
    
    # 5. Initialize MLX reasoning core
    logger.info("\n5. Initializing MLX Reasoning Core...")
    config = MLXConfig()
    reasoning_core = MLXQwenVLCore(config)
    
    # Store data for interactive mode
    context = {
        "frames": frames,
        "ocr_results": ocr_results,
        "speech_results": speech_results,
        "timeline_indexer": timeline_indexer,
        "reasoning_core": reasoning_core,
    }
    
    print("\n" + "=" * 60)
    print("✅ VIDEO PROCESSING COMPLETE (Mac)")
    print("=" * 60)
    print(f"   📹 Frames extracted: {len(frames)}")
    print(f"   📝 OCR text regions: {len(ocr_results)}")
    print(f"   🎤 Speech segments: {len(speech_results)}")
    print(f"   📋 Timeline events: {timeline_indexer.get_statistics()['total_events']}")
    print(f"   🧠 Using: MLX Qwen (4-bit quantized)")
    print("=" * 60)
    
    return context


def format_timeline_context(timeline_indexer, query: str = "") -> str:
    """Format timeline events as context text."""
    events = timeline_indexer._events if hasattr(timeline_indexer, "_events") else []
    
    if not events:
        return "No events in timeline."
    
    sorted_events = sorted(events, key=lambda e: e.timestamp)
    
    lines = []
    for event in sorted_events[:50]:  # Limit to 50 events
        mins = int(event.timestamp // 60)
        secs = int(event.timestamp % 60)
        lines.append(f"[{mins:02d}:{secs:02d}] {event.description}")
    
    return "\n".join(lines)


def interactive_mode_mac(context: dict):
    """Interactive Q&A session with streaming output (Mac)."""
    print("\n" + "=" * 60)
    print("🎮 INTERACTIVE GAMEPLAY ANALYSIS (Mac MLX)")
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
    parser = argparse.ArgumentParser(description="Gameplay video inference (Mac MLX)")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--interactive", action="store_true", help="Interactive Q&A mode")
    parser.add_argument("--query", help="Single question to ask")
    parser.add_argument("--fps", type=float, default=0.5, help="Frames per second to sample")
    
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video):
        print(f"❌ Error: Video not found: {args.video}")
        sys.exit(1)
    
    # Process video
    context = process_video_mac(args.video, fps=args.fps)
    
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
