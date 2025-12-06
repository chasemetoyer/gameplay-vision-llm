#!/usr/bin/env python3
"""
Real-Time Video Inference Script.

Supports YouTube links and local MP4 files for full multimodal gameplay analysis.
Uses trained projectors to convert perception embeddings to LLM-space tokens.

Usage:
    # YouTube video
    python scripts/realtime_inference.py --video "https://youtube.com/watch?v=..." --interactive
    
    # Local MP4
    python scripts/realtime_inference.py --video gameplay.mp4 --query "What happened when the player died?"
"""

import argparse
import logging
import os
import sys
import tempfile
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


# =============================================================================
# Video Download & Loading
# =============================================================================

def is_youtube_url(url: str) -> bool:
    """Check if URL is a YouTube link."""
    youtube_patterns = [
        r'(https?://)?(www\.)?youtube\.com/watch\?v=',
        r'(https?://)?(www\.)?youtu\.be/',
        r'(https?://)?(www\.)?youtube\.com/shorts/',
    ]
    return any(re.match(pattern, url) for pattern in youtube_patterns)


def download_youtube(url: str, output_dir: str = "data/videos") -> str:
    """
    Download YouTube video using yt-dlp.
    
    Returns path to downloaded video.
    """
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp not installed. Run: pip install yt-dlp")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure yt-dlp
    ydl_opts = {
        'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'merge_output_format': 'mp4',
        'quiet': False,
        'no_warnings': True,
    }
    
    logger.info(f"Downloading: {url}")
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_title = info.get('title', 'video')
        # Clean filename
        safe_title = re.sub(r'[^\w\s-]', '', video_title).strip()[:50]
        video_path = os.path.join(output_dir, f"{safe_title}.mp4")
        
        # Find actual downloaded file
        for f in os.listdir(output_dir):
            if f.endswith('.mp4') and safe_title[:20].lower() in f.lower():
                video_path = os.path.join(output_dir, f)
                break
    
    logger.info(f"Downloaded to: {video_path}")
    return video_path


def extract_frames(video_path: str, fps: float = 0.5) -> list[tuple[float, Image.Image]]:
    """Extract frames from video at specified FPS."""
    try:
        import decord
        decord.bridge.set_bridge("native")
        
        vr = decord.VideoReader(video_path)
        video_fps = vr.get_avg_fps()
        total_frames = len(vr)
        duration = total_frames / video_fps
        
        logger.info(f"Video: {duration:.1f}s, {video_fps:.1f} fps, {total_frames} frames")
        
        # Sample at target FPS
        frame_interval = max(1, int(video_fps / fps))
        sample_indices = list(range(0, total_frames, frame_interval))
        
        frames = []
        for idx in tqdm(sample_indices, desc="Extracting frames"):
            timestamp = idx / video_fps
            frame_np = vr[idx].asnumpy()
            frame_pil = Image.fromarray(frame_np)
            frames.append((timestamp, frame_pil))
        
        logger.info(f"Extracted {len(frames)} frames at {fps} FPS")
        return frames
        
    except ImportError:
        # Fallback to cv2
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        logger.info(f"Video: {duration:.1f}s, {video_fps:.0f} fps")
        
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
                pil_image = Image.fromarray(frame_rgb)
                frames.append((timestamp, pil_image))
                pbar.update(1)
            
            frame_idx += 1
        
        pbar.close()
        cap.release()
        logger.info(f"Extracted {len(frames)} frames")
        return frames


# =============================================================================
# Visual Detection (SAM3)
# =============================================================================

def run_sam3_detection(frames: list[tuple[float, Image.Image]], device: str = "cuda"):
    """
    Run SAM3 for visual entity detection on frames.
    
    Returns list of detections per frame with bounding boxes and masks.
    """
    try:
        from perception.sam_concept_segmenter import SAMConceptSegmenter, SAMConfig
        import numpy as np
        
        logger.info("Loading SAM3 for entity detection...")
        config = SAMConfig(device=device)
        segmenter = SAMConceptSegmenter(config)
        
        # Concepts to detect in gameplay
        concepts = ["player", "enemy", "boss", "health bar", "weapon", "character", "object"]
        
        all_detections = []
        
        for timestamp, frame in tqdm(frames, desc="SAM3 detection"):
            frame_np = np.array(frame)
            frame_idx = int(timestamp * 10)  # Approximate frame index
            
            try:
                entities = segmenter.segment_with_prompts(
                    frame=frame_np,
                    frame_idx=frame_idx,
                    concept_prompts=concepts,
                )
                
                detections = []
                for entity in entities:
                    detections.append({
                        "timestamp": timestamp,
                        "label": entity.label if hasattr(entity, 'label') else "entity",
                        "confidence": entity.confidence if hasattr(entity, 'confidence') else 0.8,
                        "bbox": entity.bbox if hasattr(entity, 'bbox') else None,
                        "mask": entity.frame_masks.get(frame_idx) if hasattr(entity, 'frame_masks') else None,
                    })
                
                all_detections.append({
                    "timestamp": timestamp,
                    "frame": frame,
                    "detections": detections,
                })
                
            except Exception as e:
                logger.debug(f"SAM3 detection failed at {timestamp:.1f}s: {e}")
                all_detections.append({
                    "timestamp": timestamp,
                    "frame": frame,
                    "detections": [],
                })
        
        total_detections = sum(len(d["detections"]) for d in all_detections)
        logger.info(f"SAM3: {total_detections} detections across {len(frames)} frames")
        return all_detections
        
    except ImportError as e:
        logger.warning(f"SAM3 not available: {e}")
        logger.info("Falling back to full-frame processing...")
        return [{"timestamp": t, "frame": f, "detections": []} for t, f in frames]
    except Exception as e:
        logger.warning(f"SAM3 detection failed: {e}")
        return [{"timestamp": t, "frame": f, "detections": []} for t, f in frames]


# =============================================================================
# Embedding Extraction
# =============================================================================

def extract_siglip_embeddings(
    frames: list[tuple[float, Image.Image]], 
    sam_results: list[dict] = None,
    device: str = "cuda"
):
    """
    Extract SigLIP embeddings from frames.
    
    If SAM3 results are provided, extracts embeddings from detected regions.
    Otherwise, processes full frames.
    """
    try:
        from perception.siglip_semantic_encoder import SigLIPSemanticEncoder, NaFlexConfig
        import numpy as np
        
        logger.info("Loading SigLIP encoder...")
        config = NaFlexConfig(device=device)
        encoder = SigLIPSemanticEncoder(config)
        
        embeddings = []
        
        # Use SAM3 regions if available
        if sam_results and any(d["detections"] for d in sam_results):
            logger.info("Encoding SAM3-detected regions with SigLIP...")
            for sam_frame in tqdm(sam_results, desc="SigLIP (regions)"):
                timestamp = sam_frame["timestamp"]
                frame = sam_frame["frame"]
                frame_np = np.array(frame)
                
                if sam_frame["detections"]:
                    # Encode each detected region
                    for det in sam_frame["detections"]:
                        try:
                            if det.get("mask") and hasattr(det["mask"], 'mask'):
                                # Use masked region
                                mask = det["mask"].mask
                                masked = frame_np.copy()
                                masked[~mask] = 0
                                region_img = Image.fromarray(masked)
                            elif det.get("bbox"):
                                # Crop to bbox
                                x1, y1, x2, y2 = det["bbox"]
                                region_img = frame.crop((x1, y1, x2, y2))
                            else:
                                region_img = frame
                            
                            emb = encoder.encode_image(region_img)
                            if emb is not None:
                                embeddings.append({
                                    "timestamp": timestamp,
                                    "label": det.get("label", "region"),
                                    "embedding": emb.cpu() if isinstance(emb, torch.Tensor) else torch.tensor(emb),
                                })
                        except Exception as e:
                            logger.debug(f"Region encoding failed: {e}")
                else:
                    # No detections, encode full frame
                    emb = encoder.encode_image(frame)
                    if emb is not None:
                        embeddings.append({
                            "timestamp": timestamp,
                            "label": "full_frame",
                            "embedding": emb.cpu() if isinstance(emb, torch.Tensor) else torch.tensor(emb),
                        })
        else:
            # Full-frame encoding (fallback)
            logger.info("Encoding full frames with SigLIP...")
            for timestamp, frame in tqdm(frames, desc="SigLIP (frames)"):
                try:
                    emb = encoder.encode_image(frame)
                    if emb is not None:
                        embeddings.append({
                            "timestamp": timestamp,
                            "label": "full_frame",
                            "embedding": emb.cpu() if isinstance(emb, torch.Tensor) else torch.tensor(emb),
                        })
                except Exception as e:
                    logger.warning(f"SigLIP failed at {timestamp:.1f}s: {e}")
        
        if embeddings:
            logger.info(f"SigLIP: {len(embeddings)} embeddings ({embeddings[0]['embedding'].shape[-1]}-dim)")
        else:
            logger.info("SigLIP: 0 embeddings")
        return embeddings
        
    except Exception as e:
        logger.warning(f"SigLIP extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def extract_videomae_embeddings(frames: list[tuple[float, Image.Image]], device: str = "cuda"):
    """Extract VideoMAE temporal embeddings from frame chunks."""
    try:
        from transformers import VideoMAEModel, AutoImageProcessor
        import numpy as np
        
        logger.info("Loading VideoMAE...")
        processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device).eval()
        
        embeddings = []
        chunk_size = 16  # VideoMAE expects 16 frames
        
        for i in range(0, len(frames) - chunk_size + 1, chunk_size // 2):
            chunk_frames = frames[i:i + chunk_size]
            if len(chunk_frames) < chunk_size:
                continue
            
            # Get middle timestamp
            mid_timestamp = chunk_frames[chunk_size // 2][0]
            
            # Prepare frames
            frame_list = [np.array(f[1]) for f in chunk_frames]
            inputs = processor(frame_list, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Pool over sequence dimension
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
            
            embeddings.append({
                "timestamp": mid_timestamp,
                "embedding": embedding.cpu(),
            })
        
        logger.info(f"VideoMAE: {len(embeddings)} embeddings (768-dim)")
        return embeddings
        
    except Exception as e:
        logger.warning(f"VideoMAE extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def extract_wav2vec_embeddings(video_path: str, device: str = "cuda"):
    """Extract Wav2Vec2 audio embeddings."""
    try:
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
        import subprocess
        import numpy as np
        
        logger.info("Loading Wav2Vec2-Large...")
        extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large").to(device).eval()
        
        # Extract audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
               "-ar", "16000", "-ac", "1", tmp_path]
        subprocess.run(cmd, capture_output=True)
        
        import wave
        with wave.open(tmp_path, 'rb') as wf:
            sr = wf.getframerate()
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)
            audio = audio / 32768.0
        
        os.unlink(tmp_path)
        
        # Process in chunks
        chunk_duration = 10  # seconds
        chunk_samples = chunk_duration * sr
        embeddings = []
        
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples]
            if len(chunk) < sr:  # Skip very short chunks
                continue
            
            timestamp = i / sr
            inputs = extractor(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
            
            embeddings.append({
                "timestamp": timestamp,
                "embedding": embedding.cpu(),
            })
        
        logger.info(f"Wav2Vec2: {len(embeddings)} embeddings (1024-dim)")
        return embeddings
        
    except Exception as e:
        logger.warning(f"Wav2Vec2 extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return []


# =============================================================================
# OCR Extraction
# =============================================================================

def run_ocr_extraction(frames: list[tuple[float, Image.Image]], device: str = "cuda"):
    """
    Run OCR on frames to extract on-screen text (subtitles, UI, damage numbers).
    
    Uses PaddleOCR in a subprocess to avoid CUDA conflicts.
    """
    import subprocess
    import json as json_module
    from pathlib import Path
    import numpy as np
    
    logger.info("Running OCR extraction...")
    
    try:
        # Try to use OCR pipeline directly first
        from perception.ocr_pipeline import OCRPipeline, OCRConfig
        
        config = OCRConfig(use_gpu=(device == "cuda"))
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
    except Exception as e:
        logger.warning(f"OCR extraction failed: {e}")
        return []


def run_speech_transcription(video_path: str, device: str = "cuda"):
    """
    Extract speech transcription from video audio using Whisper.
    
    Returns list of speech segments with timestamps.
    """
    try:
        import whisper
        import numpy as np
        import subprocess
        
        logger.info("Running speech transcription with Whisper...")
        
        # Extract audio from video
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
               "-ar", "16000", "-ac", "1", tmp_path]
        subprocess.run(cmd, capture_output=True)
        
        # Load Whisper model
        model = whisper.load_model("base", device=device)
        
        # Transcribe
        result = model.transcribe(tmp_path, 
                                  language="en",
                                  word_timestamps=True,
                                  verbose=False)
        
        os.unlink(tmp_path)
        
        # Extract segments with timestamps
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "timestamp": seg["start"],
                "end_time": seg["end"],
                "text": seg["text"].strip(),
                "type": "speech",
            })
        
        logger.info(f"Whisper: {len(segments)} speech segments transcribed")
        return segments
        
    except ImportError:
        logger.warning("Whisper not installed. Run: pip install openai-whisper")
        return []
    except Exception as e:
        logger.warning(f"Speech transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def build_timeline_index(
    ocr_results: list,
    speech_results: list,
    sam_results: list = None,
):
    """
    Build a TimelineIndexer with OCR, speech, and visual detection events.
    """
    from fusion_indexing.timeline_indexer import (
        TimelineIndexer, 
        ModalityType, 
        EventPriority
    )
    
    logger.info("Building timeline index...")
    indexer = TimelineIndexer()
    
    # Add OCR events
    for ocr in ocr_results:
        indexer.add_event(
            timestamp=ocr["timestamp"],
            modality=ModalityType.OCR,
            description=ocr["text"],
            confidence=ocr.get("confidence", 0.8),
            priority=EventPriority.MEDIUM,
        )
    
    # Add speech events
    for speech in speech_results:
        indexer.add_event(
            timestamp=speech["timestamp"],
            modality=ModalityType.SPEECH,
            description=speech["text"],
            duration=speech.get("end_time", speech["timestamp"]) - speech["timestamp"],
            priority=EventPriority.HIGH,
        )
    
    # Add visual detection events from SAM3
    if sam_results:
        for sam_frame in sam_results:
            for det in sam_frame.get("detections", []):
                indexer.add_event(
                    timestamp=sam_frame["timestamp"],
                    modality=ModalityType.VISUAL,
                    description=f"{det.get('label', 'entity')} detected",
                    confidence=det.get("confidence", 0.8),
                    priority=EventPriority.LOW,
                )
    
    # Merge and deduplicate
    indexer.merge_and_dedupe()
    
    stats = indexer.get_statistics()
    logger.info(f"Timeline: {stats['total_events']} events indexed")
    logger.info(f"   OCR: {stats['events_by_modality'].get('ocr', 0)}")
    logger.info(f"   Speech: {stats['events_by_modality'].get('speech', 0)}")
    logger.info(f"   Visual: {stats['events_by_modality'].get('visual', 0)}")
    
    return indexer


# =============================================================================
# Main Processing
# =============================================================================

def process_video(
    video_path: str,
    device: str = "cuda",
    fps: float = 0.5,
    use_sam: bool = True,
    projector_weights: str = "outputs/projector_weights.pt",
    lora_path: str = "outputs/lora_adapter",
):
    """
    Process video and return initialized PerceptionReasoningLoop.
    """
    from agent_core.qwen_reasoning_core import (
        PerceptionReasoningLoop,
        ReasoningCoreConfig,
    )
    
    logger.info("=" * 60)
    logger.info("PROCESSING VIDEO")
    logger.info("=" * 60)
    
    # 1. Extract frames
    logger.info("\n1. Extracting frames...")
    frames = extract_frames(video_path, fps=fps)
    
    # 2. Run SAM3 detection (if enabled)
    sam_results = None
    if use_sam:
        logger.info("\n2a. Running SAM3 visual detection...")
        sam_results = run_sam3_detection(frames, device)
    
    # 3. Extract embeddings
    logger.info("\n2b. Extracting multimodal embeddings...")
    siglip_embs = extract_siglip_embeddings(frames, sam_results=sam_results, device=device)
    videomae_embs = extract_videomae_embeddings(frames, device)
    wav2vec_embs = extract_wav2vec_embeddings(video_path, device)
    
    # 4. Run OCR extraction
    logger.info("\n3. Extracting text (OCR)...")
    ocr_results = run_ocr_extraction(frames, device)
    
    # 5. Run speech transcription
    logger.info("\n4. Transcribing speech (Whisper)...")
    speech_results = run_speech_transcription(video_path, device)
    
    # 6. Build timeline index
    logger.info("\n5. Building timeline index...")
    timeline_indexer = build_timeline_index(ocr_results, speech_results, sam_results)
    
    # 7. Initialize loop with projectors and timeline
    logger.info("\n6. Initializing PerceptionReasoningLoop...")
    config = ReasoningCoreConfig(device=device)
    
    loop = PerceptionReasoningLoop(
        config=config,
        timeline_indexer=timeline_indexer,
        projector_weights_path=projector_weights if os.path.exists(projector_weights) else None,
        lora_path=lora_path if os.path.exists(lora_path) else None,
    )
    
    # Index the timeline for semantic retrieval
    loop.reasoning_core.index_timeline(timeline_indexer)
    
    # Store embeddings and detections for later use
    loop._cached_embeddings = {
        "siglip": siglip_embs,
        "videomae": videomae_embs,
        "wav2vec": wav2vec_embs,
        "frames": frames,
        "sam_results": sam_results,
        "ocr_results": ocr_results,
        "speech_results": speech_results,
        "timeline_indexer": timeline_indexer,
    }
    
    logger.info(f"\n✅ Video processed!")
    logger.info(f"   Frames: {len(frames)}")
    if sam_results:
        total_det = sum(len(d["detections"]) for d in sam_results)
        logger.info(f"   SAM3 detections: {total_det}")
    logger.info(f"   SigLIP embeddings: {len(siglip_embs)}")
    logger.info(f"   VideoMAE embeddings: {len(videomae_embs)}")
    logger.info(f"   Wav2Vec2 embeddings: {len(wav2vec_embs)}")
    logger.info(f"   OCR text regions: {len(ocr_results)}")
    logger.info(f"   Speech segments: {len(speech_results)}")
    logger.info(f"   Timeline events: {timeline_indexer.get_statistics()['total_events']}")
    
    return loop


def answer_query(loop, query: str, timestamp: Optional[float] = None) -> str:
    """Answer a query using the processed video context."""
    
    # Get embeddings near timestamp (or all if not specified)
    cached = loop._cached_embeddings
    
    # Get relevant embeddings
    if timestamp is not None:
        # Filter embeddings near timestamp
        window = 30.0  # ±30 seconds
        siglip_near = [e for e in cached["siglip"] if abs(e["timestamp"] - timestamp) < window]
        videomae_near = [e for e in cached["videomae"] if abs(e["timestamp"] - timestamp) < window]
        wav2vec_near = [e for e in cached["wav2vec"] if abs(e["timestamp"] - timestamp) < window]
    else:
        # Use all embeddings (sample if too many)
        siglip_near = cached["siglip"][:20]
        videomae_near = cached["videomae"][:10]
        wav2vec_near = cached["wav2vec"][:10]
    
    # Stack embeddings
    siglip_tensor = torch.stack([e["embedding"] for e in siglip_near]) if siglip_near else None
    videomae_tensor = torch.stack([e["embedding"] for e in videomae_near]) if videomae_near else None
    wav2vec_tensor = torch.stack([e["embedding"] for e in wav2vec_near]) if wav2vec_near else None
    
    # Get a frame for visual context
    frame = None
    if timestamp is not None and cached["frames"]:
        closest_frame = min(cached["frames"], key=lambda f: abs(f[0] - timestamp))
        frame = closest_frame[1]
    elif cached["frames"]:
        frame = cached["frames"][len(cached["frames"]) // 2][1]
    
    # Start loop and process
    loop.start()
    loop.set_query(query)
    
    response = loop.process_frame(
        frame=frame,
        timestamp=timestamp or 0.0,
        region_embeddings=siglip_tensor,
        videomae_embeddings=videomae_tensor,
        audio_embeddings=wav2vec_tensor,
        force_reason=True,
    )
    
    return response or "[No response generated]"


def interactive_mode(loop):
    """Interactive Q&A session."""
    print("\n" + "=" * 60)
    print("🎮 INTERACTIVE GAMEPLAY ANALYSIS")
    print("=" * 60)
    print("Commands:")
    print("  @<MM:SS> <question>  - Ask about specific timestamp")
    print("  <question>           - Ask about whole video")
    print("  quit                 - Exit")
    print("=" * 60 + "\n")
    
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
            response = answer_query(loop, query, timestamp)
            
            print(f"📝 Response:\n{response}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
    
    loop.stop()


def main():
    parser = argparse.ArgumentParser(description="Real-time gameplay video inference")
    parser.add_argument("--video", required=True, help="YouTube URL or path to video file")
    parser.add_argument("--query", help="Single question to ask")
    parser.add_argument("--interactive", action="store_true", help="Interactive Q&A mode")
    parser.add_argument("--timestamp", type=str, help="Timestamp to focus on (MM:SS format)")
    parser.add_argument("--fps", type=float, default=0.5, help="Frames per second to sample")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--use-sam", action="store_true", help="Use SAM3 for entity detection")
    parser.add_argument("--projector-weights", default="outputs/projector_weights.pt", 
                        help="Path to projector weights")
    parser.add_argument("--lora-path", default="outputs/lora_adapter",
                        help="Path to LoRA adapter")
    parser.add_argument("--output-dir", default="data/videos",
                        help="Directory for downloaded videos")
    
    args = parser.parse_args()
    
    # Determine video source
    video_path = args.video
    
    if is_youtube_url(args.video):
        print(f"\n📺 Detected YouTube URL")
        video_path = download_youtube(args.video, args.output_dir)
    elif not os.path.exists(args.video):
        print(f"❌ Error: Video not found: {args.video}")
        sys.exit(1)
    
    # Process video
    loop = process_video(
        video_path=video_path,
        device=args.device,
        fps=args.fps,
        use_sam=args.use_sam,
        projector_weights=args.projector_weights,
        lora_path=args.lora_path,
    )
    
    # Handle mode
    if args.interactive:
        interactive_mode(loop)
    elif args.query:
        # Parse timestamp if provided
        timestamp = None
        if args.timestamp:
            ts_match = re.match(r'(\d+):(\d+)', args.timestamp)
            if ts_match:
                mins, secs = ts_match.groups()
                timestamp = int(mins) * 60 + int(secs)
        
        print("\n🔍 Analyzing...\n")
        response = answer_query(loop, args.query, timestamp)
        print(f"📝 Response:\n{response}")
        loop.stop()
    else:
        print("❌ Error: Provide --query or use --interactive mode")
        sys.exit(1)


if __name__ == "__main__":
    main()
