#!/usr/bin/env python3
"""
Quick test of full pipeline: SAM3 → OCR on just 2-3 frames.
Run this to verify everything works before running full extraction.
"""

import sys
import time
import logging
from pathlib import Path
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_full_pipeline(video_path: str, num_frames: int = 3):
    """Test full pipeline on limited frames."""
    
    print(f"\n{'='*60}")
    print("FULL PIPELINE TEST (SAM3 → OCR)")
    print(f"Video: {video_path}")
    print(f"Testing on: {num_frames} frames")
    print(f"{'='*60}\n")
    
    # Step 1: Extract a few frames
    print("Step 1: Extracting frames...")
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path)
    video_fps = vr.get_avg_fps()
    total_frames = len(vr)
    
    # Sample evenly across video
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        timestamp = idx / video_fps
        frame_np = vr[int(idx)].asnumpy()
        frame_pil = Image.fromarray(frame_np)
        frames.append((timestamp, frame_pil))
        print(f"  Extracted frame at {timestamp:.1f}s")
    
    # Step 2: Run SAM3
    print("\nStep 2: Running SAM3...")
    start = time.time()
    
    from transformers import Sam3Model, Sam3Processor
    import torch
    
    model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.float32)
    model = model.cuda().eval()
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    print(f"  SAM3 loaded in {time.time() - start:.1f}s")
    
    prompts = ["character", "enemy"]
    sam_results = []
    
    for i, (timestamp, frame) in enumerate(frames):
        start = time.time()
        for prompt in prompts:
            inputs = processor(images=frame, text=prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model(**inputs)
            results = processor.post_process_instance_segmentation(
                outputs, threshold=0.5, mask_threshold=0.5, target_sizes=[frame.size[::-1]]
            )
            for r in results:
                masks = r.get("masks", [])
                if len(masks) > 0:
                    sam_results.append({"timestamp": timestamp, "prompt": prompt, "masks": len(masks)})
        print(f"  Frame {i+1}/{len(frames)} [{timestamp:.1f}s] processed in {time.time() - start:.1f}s")
    
    print(f"  ✅ SAM3 found {len(sam_results)} detections")
    
    # Step 3: Run OCR (on CPU to avoid conflict)
    print("\nStep 3: Running OCR (CPU mode - disabling Paddle CUDA)...")
    start = time.time()
    
    # CRITICAL: Disable CUDA for PaddlePaddle BEFORE import to avoid CUDNN conflict
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide all GPUs from Paddle
    os.environ["FLAGS_use_cuda"] = "0"
    
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from perception.ocr_pipeline import OCRPipeline, OCRConfig
    
    config = OCRConfig(use_gpu=False)  # CPU to avoid CUDNN conflict
    ocr = OCRPipeline(config)
    
    ocr_results = []
    for i, (timestamp, frame) in enumerate(frames):
        frame_np = np.array(frame.convert("RGB"))
        result = ocr.extract_text_from_frame(frame_np, i, timestamp)
        if result and result.detections:
            ocr_results.extend(result.detections)
        print(f"  Frame {i+1}/{len(frames)} [{timestamp:.1f}s] - {len(result.detections) if result else 0} text regions")
    
    print(f"  ✅ OCR found {len(ocr_results)} text regions in {time.time() - start:.1f}s")
    
    # Summary
    print(f"\n{'='*60}")
    print("✅ FULL PIPELINE TEST PASSED!")
    print(f"   SAM3 detections: {len(sam_results)}")
    print(f"   OCR text regions: {len(ocr_results)}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/data/raw_videos/clair.mp4"
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    test_full_pipeline(video_path, num_frames)
