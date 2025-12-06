#!/usr/bin/env python3
"""Upload trained adapters to Hugging Face Hub."""

import os
from huggingface_hub import HfApi, login, create_repo

# ============================================
# CONFIGURATION - Edit these values
# ============================================
HF_USERNAME = "cjm249"  # Replace with your HuggingFace username
REPO_NAME = "gameplay-vision-llm-adapters"
# ============================================

def main():
    # Login (will open browser or prompt for token)
    print("Logging in to Hugging Face...")
    login()
    
    # Create repository
    repo_id = f"{HF_USERNAME}/{REPO_NAME}"
    print(f"\nCreating repository: {repo_id}")
    
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"Repository created/exists: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload files
    api = HfApi()
    outputs_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"\nUploading files from {outputs_dir}...")
    
    # Upload all files in outputs/
    api.upload_folder(
        folder_path=outputs_dir,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["*.pyc", "__pycache__", "upload_to_hf.py"],
    )
    
    print(f"\n✅ Upload complete!")
    print(f"   View at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
