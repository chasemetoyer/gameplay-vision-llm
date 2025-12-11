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

# Suppress noisy HTTP and HuggingFace logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("ppocr").setLevel(logging.WARNING)


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
    
    # Configure yt-dlp - force H.264 codec (avoid AV1 which OpenCV can't decode)
    ydl_opts = {
        'format': 'bestvideo[height<=720][vcodec^=avc1]+bestaudio[acodec^=mp4a]/bestvideo[height<=720][vcodec^=avc]+bestaudio/best[height<=720]',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'merge_output_format': 'mp4',
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
        'postprocessor_args': ['-c:v', 'libx264', '-crf', '23', '-c:a', 'aac'],
        'quiet': False,
        'no_warnings': True,
    }
    
    logger.info(f"Downloading: {url}")
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get('id', 'video')
        video_title = info.get('title', 'video')
        
        # Find the actual downloaded file (may have unicode chars)
        downloaded_file = None
        for f in os.listdir(output_dir):
            if f.endswith('.mp4'):
                # Check if this is a recent file
                downloaded_file = os.path.join(output_dir, f)
                break
        
        # Create a clean ASCII filename
        safe_title = re.sub(r'[^\w\s-]', '', video_title)
        safe_title = re.sub(r'\s+', '_', safe_title).strip()[:40]
        clean_path = os.path.join(output_dir, f"{safe_title}_{video_id}.mp4")
        
        # Rename if needed to avoid unicode issues with decord
        if downloaded_file and downloaded_file != clean_path:
            import shutil
            shutil.move(downloaded_file, clean_path)
            video_path = clean_path
        else:
            video_path = downloaded_file or clean_path
    
    logger.info(f"Downloaded to: {video_path}")
    return video_path


def extract_frames(video_path: str, fps: float = 0.5) -> list[tuple[float, Image.Image]]:
    """Extract frames from video at specified FPS."""
    import cv2
    
    # Use cv2 (more robust with various formats)
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


# =============================================================================
# Visual Detection (SAM3)
# =============================================================================

def run_sam3_detection(
    frames: list[tuple[float, Image.Image]], 
    device: str = "cuda",
    concepts: list[str] = None,
):
    """
    Run SAM3 for visual entity detection on frames.
    
    Args:
        frames: List of (timestamp, PIL.Image) tuples
        device: Device to run on
        concepts: List of concept prompts for detection. If None, uses default gameplay concepts.
                  For benchmarks, extract concepts from question/choices for better relevance.
    
    Returns list of detections per frame with bounding boxes and masks.
    """
    try:
        from perception.sam_concept_segmenter import SAMConceptSegmenter, SAMConfig
        import numpy as np
        
        logger.info("Loading SAM3 for entity detection...")
        config = SAMConfig(device=device)
        segmenter = SAMConceptSegmenter(config)
        
        # Use provided concepts or default gameplay concepts
        if concepts is None:
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

def run_ocr_extraction(frames: list[tuple[float, Image.Image]], device: str = "cuda", backend: str = "paddleocr"):
    """
    Run OCR on frames to extract on-screen text (subtitles, UI, damage numbers).
    
    Args:
        frames: List of (timestamp, PIL.Image) tuples
        device: Device to run on ("cuda" or "cpu")
        backend: OCR backend to use ("paddleocr" or "tesseract")
    """
    import subprocess
    import json as json_module
    from pathlib import Path
    import numpy as np
    
    logger.info(f"Running OCR extraction (backend={backend})...")
    
    results = []
    
    if backend == "tesseract":
        # Use Tesseract OCR (lighter, no GPU needed)
        try:
            import pytesseract
            
            for idx, (timestamp, frame) in enumerate(tqdm(frames, desc="OCR (tesseract)")):
                try:
                    frame_rgb = frame.convert("RGB")
                    text = pytesseract.image_to_string(frame_rgb).strip()
                    if text:
                        results.append({
                            "timestamp": timestamp,
                            "text": text,
                            "confidence": 0.8,  # Tesseract default
                            "type": "ocr",
                        })
                except Exception as e:
                    logger.debug(f"OCR error at {timestamp:.1f}s: {e}")
            
            logger.info(f"OCR (tesseract): {len(results)} text regions extracted")
            return results
            
        except ImportError:
            logger.warning("pytesseract not installed, falling back to PaddleOCR")
            backend = "paddleocr"
    
    # PaddleOCR backend
    try:
        from perception.ocr_pipeline import OCRPipeline, OCRConfig
        
        config = OCRConfig(use_gpu=(device == "cuda"))
        ocr = OCRPipeline(config)
        
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
        
        logger.info(f"OCR (paddleocr): {len(results)} text regions extracted")
        return results
        
    except ImportError:
        logger.warning("OCR pipeline not available, skipping OCR")
        return []
    except Exception as e:
        logger.warning(f"OCR extraction failed: {e}")
        return []


def run_speech_transcription(video_path: str, device: str = "cuda", model_size: str = "base"):
    """
    Extract speech transcription from video audio using Whisper.
    
    Args:
        video_path: Path to video file
        device: Device to run on ("cuda" or "cpu")
        model_size: Whisper model size ("tiny", "base", "small", "medium", "large", "large-v3")
    
    Returns list of speech segments with timestamps.
    """
    print("=" * 50)
    print(f"🎤 WHISPER SPEECH TRANSCRIPTION ({model_size})")
    print("=" * 50)
    
    try:
        import whisper
        import numpy as np
        import subprocess
        
        print(f"   [1/4] Extracting audio from video...")
        
        # Extract audio from video
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
               "-ar", "16000", "-ac", "1", tmp_path]
        result = subprocess.run(cmd, capture_output=True)
        
        if result.returncode != 0:
            print(f"   ⚠️ FFmpeg failed: {result.stderr.decode()[:200]}")
            return []
        
        print(f"   [2/4] Loading Whisper model ({model_size})...")
        
        # Load Whisper model
        model = whisper.load_model(model_size, device=device)
        
        print(f"   [3/4] Transcribing audio...")
        
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
        
        print(f"   [4/4] ✅ Transcribed {len(segments)} speech segments")
        
        # Show first few segments
        if segments:
            print("   Sample transcriptions:")
            for seg in segments[:3]:
                ts = f"{int(seg['timestamp']//60):02d}:{int(seg['timestamp']%60):02d}"
                print(f"      [{ts}] {seg['text'][:60]}...")
        
        return segments
        
    except ImportError as e:
        print(f"   ❌ Whisper not installed: {e}")
        return []
    except Exception as e:
        print(f"   ❌ Speech transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def build_timeline_index(
    ocr_results: list,
    speech_results: list,
    sam_results: list = None,
    min_ocr_confidence: float = 0.7,  # Filter low-confidence OCR
):
    """
    Build a TimelineIndexer with OCR, speech, and visual detection events.
    
    Args:
        ocr_results: OCR detection results
        speech_results: Speech transcription results
        sam_results: SAM3 detection results
        min_ocr_confidence: Minimum confidence to include OCR text (filters garbage)
    """
    from fusion_indexing.timeline_indexer import (
        TimelineIndexer, 
        ModalityType, 
        EventPriority
    )
    
    logger.info("Building timeline index...")
    indexer = TimelineIndexer()
    
    # Track OCR text to deduplicate similar strings
    seen_ocr_texts = set()
    ocr_filtered_count = 0
    ocr_deduped_count = 0
    
    # Add OCR events with confidence filtering and deduplication
    for ocr in ocr_results:
        conf = ocr.get("confidence", 0.8)
        text = ocr["text"].strip()
        
        # Skip low-confidence OCR (garbage text)
        if conf < min_ocr_confidence:
            ocr_filtered_count += 1
            continue
        
        # Skip very short text (likely noise)
        if len(text) < 3:
            ocr_filtered_count += 1
            continue
        
        # Simple deduplication: skip if we've seen very similar text
        # Normalize: lowercase, remove extra spaces
        normalized = " ".join(text.lower().split())
        if normalized in seen_ocr_texts:
            ocr_deduped_count += 1
            continue
        seen_ocr_texts.add(normalized)
        
        indexer.add_event(
            timestamp=ocr["timestamp"],
            modality=ModalityType.OCR,
            description=text,
            confidence=conf,
            priority=EventPriority.MEDIUM,
        )
    
    if ocr_filtered_count > 0:
        logger.info(f"Filtered {ocr_filtered_count} low-confidence OCR texts (conf < {min_ocr_confidence})")
    if ocr_deduped_count > 0:
        logger.info(f"Deduplicated {ocr_deduped_count} repetitive OCR texts")
    
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
    use_cache: bool = True,
    cache_dir: str = "data/cache",
    preset_config=None,  # SystemConfig from presets module
):
    """
    Process video and return initialized PerceptionReasoningLoop.

    Supports feature caching to avoid reprocessing the same video.
    
    Args:
        preset_config: Optional SystemConfig from presets module. If provided,
                      settings like use_videomae, ocr_backend, whisper_model
                      are taken from the preset.
    """
    import time
    from agent_core.qwen_reasoning_core import (
        PerceptionReasoningLoop,
        ReasoningCoreConfig,
        FeatureCache,
    )

    # Timing dictionary to track each stage
    timings = {}
    total_start = time.time()
    
    # Extract settings from preset config if provided
    use_videomae = True
    use_wav2vec = True
    ocr_backend = "paddleocr"
    whisper_model = "base"
    sam3_fps = None  # None means use all frames for SAM3
    
    if preset_config is not None:
        use_videomae = preset_config.perception.use_videomae
        use_wav2vec = preset_config.audio.use_wav2vec
        ocr_backend = preset_config.perception.ocr_backend
        sam3_fps = getattr(preset_config.perception, 'sam3_fps', None)
        # Extract model name without prefix for whisper
        whisper_full = preset_config.audio.whisper_model
        if "/" in whisper_full:
            whisper_model = whisper_full.split("/")[-1].replace("whisper-", "")
        else:
            whisper_model = whisper_full.replace("whisper-", "")
        logger.info(f"Using preset settings: VideoMAE={use_videomae}, Wav2Vec={use_wav2vec}, OCR={ocr_backend}, Whisper={whisper_model}, SAM3_FPS={sam3_fps}")

    logger.info("=" * 60)
    logger.info("PROCESSING VIDEO")
    logger.info("=" * 60)

    # Initialize feature cache
    feature_cache = FeatureCache(cache_dir=cache_dir) if use_cache else None

    # Check if we have cached features
    cached_features = None
    from_cache = False
    if feature_cache and feature_cache.has_features(video_path):
        print("\nFound cached features! Loading from cache...")
        cache_start = time.time()
        cached_features = feature_cache.load_features(video_path)
        timings["cache_load"] = time.time() - cache_start

    if cached_features:
        # Use cached features
        from_cache = True
        frames = cached_features.get("frames", [])
        sam_results = cached_features.get("sam_results")
        siglip_embs = cached_features.get("siglip", [])
        videomae_embs = cached_features.get("videomae", [])
        wav2vec_embs = cached_features.get("wav2vec", [])
        ocr_results = cached_features.get("ocr_results", [])
        speech_results = cached_features.get("speech_results", [])

        print(f"Loaded {len(frames)} frames and {len(siglip_embs)} embeddings from cache")

    else:
        # Extract features from scratch
        # 1. Extract frames
        logger.info("\n1. Extracting frames...")
        t0 = time.time()
        frames = extract_frames(video_path, fps=fps)
        timings["frame_extraction"] = time.time() - t0
        print(f"   ⏱️  Frame extraction: {timings['frame_extraction']:.1f}s")

        # 2. Run SAM3 detection (if enabled)
        sam_results = None
        if use_sam:
            logger.info("\n2a. Running SAM3 visual detection...")
            
            # Subsample frames for SAM3 if sam3_fps is set and lower than extraction fps
            sam_frames = frames
            if sam3_fps is not None and sam3_fps > 0 and sam3_fps < fps:
                # Calculate subsampling step based on fps ratio
                step = int(fps / sam3_fps)
                sam_frames = frames[::step]
                logger.info(f"SAM3 subsampling: {len(frames)} -> {len(sam_frames)} frames (step={step}, sam3_fps={sam3_fps})")
            
            t0 = time.time()
            sam_results = run_sam3_detection(sam_frames, device)
            timings["sam3"] = time.time() - t0
            print(f"   ⏱️  SAM3 detection: {timings['sam3']:.1f}s ({len(sam_frames)} frames)")
        else:
            timings["sam3"] = 0.0
            print("   ⏱️  SAM3: Skipped (disabled)")

        # 3. Extract embeddings
        logger.info("\n2b. Extracting multimodal embeddings...")
        
        t0 = time.time()
        siglip_embs = extract_siglip_embeddings(frames, sam_results=sam_results, device=device)
        timings["siglip"] = time.time() - t0
        print(f"   ⏱️  SigLIP encoding: {timings['siglip']:.1f}s")
        
        # VideoMAE (conditional based on preset)
        if use_videomae:
            t0 = time.time()
            videomae_embs = extract_videomae_embeddings(frames, device)
            timings["videomae"] = time.time() - t0
            print(f"   ⏱️  VideoMAE encoding: {timings['videomae']:.1f}s")
        else:
            videomae_embs = []
            timings["videomae"] = 0.0
            print("   ⏱️  VideoMAE: Skipped (disabled by preset)")
        
        # Wav2Vec2 (conditional based on preset)
        if use_wav2vec:
            t0 = time.time()
            wav2vec_embs = extract_wav2vec_embeddings(video_path, device)
            timings["wav2vec"] = time.time() - t0
            print(f"   ⏱️  Wav2Vec2 encoding: {timings['wav2vec']:.1f}s")
        else:
            wav2vec_embs = []
            timings["wav2vec"] = 0.0
            print("   ⏱️  Wav2Vec2: Skipped (disabled by preset)")

        # 4. Run OCR extraction
        logger.info("\n3. Extracting text (OCR)...")
        t0 = time.time()
        ocr_results = run_ocr_extraction(frames, device, backend=ocr_backend)
        timings["ocr"] = time.time() - t0
        print(f"   ⏱️  OCR extraction ({ocr_backend}): {timings['ocr']:.1f}s")

        # 5. Run speech transcription
        logger.info("\n4. Transcribing speech (Whisper)...")
        t0 = time.time()
        speech_results = run_speech_transcription(video_path, device, model_size=whisper_model)
        timings["whisper"] = time.time() - t0
        print(f"   ⏱️  Whisper transcription ({whisper_model}): {timings['whisper']:.1f}s")

        # Cache features for future use (excluding full frames to save space)
        if feature_cache:
            t0 = time.time()
            # Store frame timestamps and thumbnails instead of full frames
            frame_metadata = [(ts, f.size) for ts, f in frames]
            cache_data = {
                "frames": frames,  # Full frames for now, could optimize later
                "frame_metadata": frame_metadata,
                "sam_results": sam_results,
                "siglip": siglip_embs,
                "videomae": videomae_embs,
                "wav2vec": wav2vec_embs,
                "ocr_results": ocr_results,
                "speech_results": speech_results,
            }
            feature_cache.save_features(video_path, cache_data)
            timings["cache_save"] = time.time() - t0
            print(f"   ⏱️  Cache save: {timings['cache_save']:.1f}s")
            print("Features cached for future use")

    # 6. Build timeline index
    logger.info("\n5. Building timeline index...")
    t0 = time.time()
    timeline_indexer = build_timeline_index(ocr_results, speech_results, sam_results)
    timings["timeline"] = time.time() - t0

    # 7. Initialize loop with projectors and timeline
    logger.info("\n6. Initializing PerceptionReasoningLoop...")
    t0 = time.time()
    config = ReasoningCoreConfig(device=device)

    loop = PerceptionReasoningLoop(
        config=config,
        timeline_indexer=timeline_indexer,
        projector_weights_path=projector_weights if os.path.exists(projector_weights) else None,
        lora_path=lora_path if os.path.exists(lora_path) else None,
    )

    # Index the timeline for semantic retrieval
    loop.reasoning_core.index_timeline(timeline_indexer)
    timings["model_init"] = time.time() - t0

    # Eagerly load Qwen3-VL model so it's ready for inference
    logger.info("\n7. Loading Qwen3-VL reasoning model...")
    t0 = time.time()
    loop.reasoning_core._load_model()  # Force eager loading
    timings["qwen_load"] = time.time() - t0
    print(f"   ⏱️  Qwen3-VL load: {timings['qwen_load']:.1f}s")

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

    # Calculate total time
    total_time = time.time() - total_start
    timings["total"] = total_time

    # Print summary with explicit print (not logger)
    print("\n" + "=" * 60)
    print("VIDEO PROCESSING COMPLETE")
    print("=" * 60)
    print(f"   Frames extracted: {len(frames)}")
    if sam_results:
        total_det = sum(len(d["detections"]) for d in sam_results)
        print(f"   SAM3 detections: {total_det}")
    print(f"   SigLIP embeddings: {len(siglip_embs)}")
    print(f"   VideoMAE embeddings: {len(videomae_embs)}")
    print(f"   Wav2Vec2 embeddings: {len(wav2vec_embs)}")
    print(f"   OCR text regions: {len(ocr_results)}")
    print(f"   Speech segments: {len(speech_results)}")
    print(f"   Timeline events: {timeline_indexer.get_statistics()['total_events']}")
    if feature_cache:
        stats = feature_cache.get_cache_stats()
        print(f"   Cache: {stats['num_cached_videos']} videos, {stats['total_size_mb']:.1f} MB")
    
    # Print timing summary
    print()
    print("-" * 60)
    print("⏱️  TIMING BREAKDOWN")
    print("-" * 60)
    
    if from_cache:
        print(f"   {'Cache load:':<25} {timings.get('cache_load', 0):.1f}s")
    else:
        print(f"   {'Frame extraction:':<25} {timings.get('frame_extraction', 0):.1f}s")
        if use_sam:
            print(f"   {'SAM3 detection:':<25} {timings.get('sam3', 0):.1f}s")
        else:
            print(f"   {'SAM3 detection:':<25} (skipped)")
        print(f"   {'SigLIP encoding:':<25} {timings.get('siglip', 0):.1f}s")
        print(f"   {'VideoMAE encoding:':<25} {timings.get('videomae', 0):.1f}s")
        print(f"   {'Wav2Vec2 encoding:':<25} {timings.get('wav2vec', 0):.1f}s")
        print(f"   {'OCR extraction:':<25} {timings.get('ocr', 0):.1f}s")
        print(f"   {'Whisper transcription:':<25} {timings.get('whisper', 0):.1f}s")
        if 'cache_save' in timings:
            print(f"   {'Cache save:':<25} {timings.get('cache_save', 0):.1f}s")
    
    print(f"   {'Timeline building:':<25} {timings.get('timeline', 0):.1f}s")
    print(f"   {'Model initialization:':<25} {timings.get('model_init', 0):.1f}s")
    print(f"   {'Qwen3-VL load:':<25} {timings.get('qwen_load', 0):.1f}s")
    print("-" * 60)
    print(f"   {'TOTAL TIME:':<25} {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 60)

    # Store timings in loop for later access
    loop._processing_timings = timings

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


def answer_query_streaming(loop, query: str, timestamp: Optional[float] = None):
    """
    Answer a query with streaming output - yields tokens as they're generated.
    Supports multi-turn conversation with follow-up questions.
    """
    import sys

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

    # Start loop and use streaming
    loop.start()
    loop.set_query(query)

    # Use streaming generation with conversation tracking
    for token in loop.reasoning_core.reason_streaming(
        query=query,
        current_frame=frame,
        timeline_indexer=loop.timeline_indexer,
        knowledge_base=loop.knowledge_base,
        video_timestamp=timestamp,
        track_conversation=True,
    ):
        yield token


def interactive_mode(loop):
    """Interactive Q&A session with streaming output and multi-turn conversation."""
    import sys

    print("\n" + "=" * 60)
    print("INTERACTIVE GAMEPLAY ANALYSIS")
    print("=" * 60)
    print("Commands:")
    print("  @<MM:SS> <question>  - Ask about specific timestamp")
    print("  <question>           - Ask about whole video")
    print("  /clear               - Clear conversation history")
    print("  /history             - Show conversation summary")
    print("  /save <path>         - Save conversation to file")
    print("  /load <path>         - Load conversation from file")
    print("  /game <name>         - Set game context (e.g., /game Elden Ring)")
    print("  /search <query>      - Search web for game info")
    print("  /wiki <topic>        - Search game wiki for topic")
    print("  /boss <name>         - Look up boss strategy")
    print("  quit                 - Exit")
    print("=" * 60)
    print("TIP: You can ask follow-up questions like 'What happened next?'")
    print("=" * 60 + "\n")

    # Try to auto-detect game from video content
    try:
        cached = loop._cached_embeddings
        detected_game = loop.reasoning_core.detect_game_from_content(
            ocr_results=cached.get("ocr_results", []),
            speech_results=cached.get("speech_results", []),
        )
        if detected_game:
            print(f"Auto-detected game: {detected_game}\n")
    except Exception:
        pass

    while True:
        try:
            # Show conversation context indicator
            turn_count = loop.reasoning_core.conversation_history.get_turn_count()
            if turn_count > 0:
                prompt = f"[{turn_count} turns] Your question: "
            else:
                prompt = "Your question: "

            user_input = input(prompt).strip()

            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if not user_input:
                continue

            # Handle special commands
            if user_input.startswith("/"):
                if user_input == "/clear":
                    loop.reasoning_core.clear_conversation()
                    print("Conversation history cleared.\n")
                    continue
                elif user_input == "/history":
                    summary = loop.reasoning_core.get_conversation_summary()
                    print("\n--- Conversation Summary ---")
                    print(f"Turns: {summary['turn_count']}")
                    if summary['timestamps_mentioned']:
                        ts_list = [f"{int(t//60)}:{int(t%60):02d}" for t in summary['timestamps_mentioned']]
                        print(f"Timestamps discussed: {', '.join(ts_list)}")
                    if summary['last_query']:
                        print(f"Last query: {summary['last_query'][:50]}...")
                    print("----------------------------\n")
                    continue
                elif user_input.startswith("/save "):
                    path = user_input[6:].strip()
                    try:
                        loop.reasoning_core.save_conversation(path)
                        print(f"Conversation saved to: {path}\n")
                    except Exception as e:
                        print(f"Failed to save: {e}\n")
                    continue
                elif user_input.startswith("/load "):
                    path = user_input[6:].strip()
                    try:
                        loop.reasoning_core.load_conversation(path)
                        print(f"Conversation loaded from: {path}\n")
                    except Exception as e:
                        print(f"Failed to load: {e}\n")
                    continue
                elif user_input.startswith("/game "):
                    game_name = user_input[6:].strip()
                    try:
                        loop.reasoning_core.set_game_context(game_name)
                        print(f"Game context set to: {game_name}\n")
                    except Exception as e:
                        print(f"Failed to set game: {e}\n")
                    continue
                elif user_input.startswith("/search "):
                    search_query = user_input[8:].strip()
                    print(f"\nSearching for: {search_query}...")
                    try:
                        results = loop.reasoning_core.search_game_knowledge(search_query, "general")
                        print(results)
                    except Exception as e:
                        print(f"Search failed: {e}")
                    print()
                    continue
                elif user_input.startswith("/wiki "):
                    topic = user_input[6:].strip()
                    print(f"\nSearching wiki for: {topic}...")
                    try:
                        results = loop.reasoning_core.search_game_knowledge(topic, "wiki")
                        print(results)
                    except Exception as e:
                        print(f"Search failed: {e}")
                    print()
                    continue
                elif user_input.startswith("/boss "):
                    boss_name = user_input[6:].strip()
                    print(f"\nLooking up boss strategy: {boss_name}...")
                    try:
                        results = loop.reasoning_core.search_game_knowledge(
                            f"{boss_name} boss fight guide strategy",
                            "guide"
                        )
                        print(results)
                    except Exception as e:
                        print(f"Search failed: {e}")
                    print()
                    continue
                elif user_input.startswith("/lore "):
                    topic = user_input[6:].strip()
                    print(f"\nSearching lore for: {topic}...")
                    try:
                        results = loop.reasoning_core.search_game_knowledge(topic, "lore")
                        print(results)
                    except Exception as e:
                        print(f"Search failed: {e}")
                    print()
                    continue
                else:
                    print(f"Unknown command: {user_input}\n")
                    continue

            # Parse timestamp if provided
            timestamp = None
            query = user_input

            ts_match = re.match(r'@(\d+):(\d+)\s+(.*)', user_input)
            if ts_match:
                mins, secs, query = ts_match.groups()
                timestamp = int(mins) * 60 + int(secs)
                print(f"Focusing on timestamp: {mins}:{secs}")

            # Check if this is a follow-up question
            if loop.reasoning_core.conversation_history.is_follow_up_query(query):
                print("(Follow-up question detected)")

            print("\nAnalyzing...\n")
            print("Response:")

            # Stream tokens as they're generated (like ChatGPT typing)
            for token in answer_query_streaming(loop, query, timestamp):
                print(token, end="", flush=True)

            # Show confidence after response
            last_turn = loop.reasoning_core.conversation_history._turns[-1] if loop.reasoning_core.conversation_history._turns else None
            if last_turn and "confidence" in last_turn.metadata:
                confidence = last_turn.metadata["confidence"]
                confidence_bar = "=" * int(confidence * 10)
                print(f"\n\n[Confidence: {confidence:.0%} |{confidence_bar:<10}|]")
            else:
                print()

            print()  # Extra newline for readability

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break

    loop.stop()


def main():
    parser = argparse.ArgumentParser(description="Real-time gameplay video inference")
    parser.add_argument("--video", help="YouTube URL or path to video file")
    parser.add_argument("--query", help="Single question to ask")
    parser.add_argument("--interactive", action="store_true", help="Interactive Q&A mode")
    parser.add_argument("--timestamp", type=str, help="Timestamp to focus on (MM:SS format)")
    parser.add_argument("--preset", type=str, choices=["light", "standard", "full"],
                        help="Configuration preset (light: ~20GB VRAM, standard: ~28GB, full: ~45GB)")
    parser.add_argument("--list-presets", action="store_true",
                        help="Show available presets and exit")
    parser.add_argument("--fps", type=float, default=None, help="Frames per second to sample (default: from preset or 0.5)")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--use-sam", action="store_true", help="Use SAM3 for entity detection (overrides preset)")
    parser.add_argument("--projector-weights", default="outputs/projector_weights.pt",
                        help="Path to projector weights")
    parser.add_argument("--lora-path", default="outputs/lora_adapter",
                        help="Path to LoRA adapter")
    parser.add_argument("--output-dir", default="data/videos",
                        help="Directory for downloaded videos")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable feature caching (reprocess every time)")
    parser.add_argument("--cache-dir", default="data/cache",
                        help="Directory for cached features")
    parser.add_argument("--game", type=str, default=None,
                        help="Game name for web search context (e.g., 'Elden Ring')")
    parser.add_argument("--no-search", action="store_true",
                        help="Disable web search capability")

    args = parser.parse_args()

    # Handle --list-presets
    if args.list_presets:
        try:
            from config.presets import print_preset_summary
            print_preset_summary()
        except ImportError:
            print("Error: Could not import preset module")
            print("\nAvailable presets:")
            print("  light    - ~20GB VRAM (RTX 3090/4090)")
            print("  standard - ~28GB VRAM (A100 40GB)")
            print("  full     - ~45GB VRAM (A100 80GB/H100)")
        sys.exit(0)

    # Require --video if not using --list-presets
    if not args.video:
        parser.error("--video is required (or use --list-presets)")

    # Apply preset configuration if specified
    preset_config = None
    if args.preset:
        try:
            from config.presets import load_preset
            preset_config = load_preset(args.preset)
            logger.info(f"Using preset '{args.preset}' (~{preset_config.estimated_vram_gb:.0f}GB VRAM)")

            # Apply preset defaults if not overridden by CLI
            if args.fps is None:
                args.fps = preset_config.inference.fps
            if not args.use_sam:  # Only override if --use-sam not explicitly set
                args.use_sam = preset_config.perception.use_sam
        except ImportError as e:
            logger.warning(f"Could not load preset config: {e}")

    # Set default fps if still not set
    if args.fps is None:
        args.fps = 0.5

    # Determine video source
    video_path = args.video

    if is_youtube_url(args.video):
        print(f"\nDetected YouTube URL")
        video_path = download_youtube(args.video, args.output_dir)
    elif not os.path.exists(args.video):
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    # Process video
    loop = process_video(
        video_path=video_path,
        device=args.device,
        fps=args.fps,
        use_sam=args.use_sam,
        projector_weights=args.projector_weights,
        lora_path=args.lora_path,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        preset_config=preset_config,  # Pass preset settings for VideoMAE, OCR, Whisper
    )

    # Set game context if provided
    if args.game:
        loop.reasoning_core.set_game_context(args.game)
        print(f"Game context set: {args.game}")

    # Disable search if requested
    if args.no_search:
        loop.reasoning_core.enable_web_search = False

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

        print("\nAnalyzing...\n")
        response = answer_query(loop, args.query, timestamp)
        print(f"Response:\n{response}")
        loop.stop()
    else:
        print("Error: Provide --query or use --interactive mode")
        sys.exit(1)


if __name__ == "__main__":
    main()
