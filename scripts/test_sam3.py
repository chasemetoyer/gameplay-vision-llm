#!/usr/bin/env python3
"""Quick test for SAM3 on a single image."""

import sys
import time
from PIL import Image

def test_sam3(image_path: str, prompt: str = "character"):
    print(f"Testing SAM3 on: {image_path}")
    print(f"Prompt: '{prompt}'")
    
    # Load image
    print("Loading image...")
    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}")
    
    # Load SAM3
    print("\nLoading SAM3 model...")
    start = time.time()
    
    from transformers import Sam3Model, Sam3Processor
    
    model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype="float32")
    model = model.cuda().eval()
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    
    print(f"Model loaded in {time.time() - start:.1f}s")
    
    # Run inference
    print(f"\nRunning inference with prompt '{prompt}'...")
    start = time.time()
    
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
    print(f"Inputs prepared in {time.time() - start:.2f}s")
    
    print("Running model.forward()...")
    start = time.time()
    
    import torch
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"Inference completed in {time.time() - start:.2f}s")
    
    # Post-process
    print("\nPost-processing...")
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=[image.size[::-1]],
    )
    
    print(f"\nResults: {len(results)} detections")
    for i, r in enumerate(results):
        masks = r.get("masks", [])
        scores = r.get("scores", [])
        print(f"  Result {i}: {len(masks)} masks, scores: {scores}")
    
    print("\n✅ SAM3 test completed successfully!")

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/data/raw_videos/test.png"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "character"
    test_sam3(image_path, prompt)
