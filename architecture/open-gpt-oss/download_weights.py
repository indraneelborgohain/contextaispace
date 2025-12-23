#!/usr/bin/env python3
"""
download_weights.py - Download GPT-OSS model weights from Hugging Face

This script downloads the gpt-oss model weights and saves them locally.
Supports downloading the 20B or smaller checkpoint versions.
"""
import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Error: huggingface_hub not installed.")
    print("Please install it with: pip install huggingface_hub")
    sys.exit(1)


def download_gpt_oss_weights(
    model_name: str = "omunaman/Open_Source_GPT_OSS_20B",
    output_dir: str = "weights",
    cache_dir: str = None,
    token: str = None
):
    """
    Download GPT-OSS model weights from Hugging Face Hub.
    
    Args:
        model_name: Hugging Face model repository name
        output_dir: Local directory to save the weights
        cache_dir: Cache directory for huggingface_hub
        token: Hugging Face API token (if needed for private models)
    """
    print(f"Downloading GPT-OSS weights from: {model_name}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Download the entire repository
        local_dir = snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
            cache_dir=cache_dir,
            token=token,
            resume_download=True,
            local_dir_use_symlinks=False,
        )
        
        print("\n" + "=" * 70)
        print(f"✅ Successfully downloaded weights to: {local_dir}")
        print("\nDownloaded files:")
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                filepath = os.path.join(root, file)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                rel_path = os.path.relpath(filepath, local_dir)
                print(f"  - {rel_path} ({size_mb:.2f} MB)")
        
        return local_dir
        
    except Exception as e:
        print(f"\n❌ Error downloading weights: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify the model name is correct")
        print("3. If it's a private model, provide a valid HF token with --token")
        print("4. Try: huggingface-cli login")
        sys.exit(1)


def download_specific_files(
    model_name: str,
    files: list,
    output_dir: str = "weights",
    token: str = None
):
    """
    Download specific files from the model repository.
    
    Args:
        model_name: Hugging Face model repository name
        files: List of filenames to download
        output_dir: Local directory to save the files
        token: Hugging Face API token
    """
    print(f"Downloading specific files from: {model_name}")
    print(f"Files: {files}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in files:
        try:
            print(f"\nDownloading {filename}...")
            local_path = hf_hub_download(
                repo_id=model_name,
                filename=filename,
                local_dir=output_dir,
                token=token,
                resume_download=True,
                local_dir_use_symlinks=False,
            )
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"✅ Downloaded: {filename} ({size_mb:.2f} MB)")
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download GPT-OSS model weights")
    parser.add_argument(
        "--model_name",
        type=str,
        default="omunaman/Open_Source_GPT_OSS_20B",
        help="Hugging Face model repository name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="weights",
        help="Directory to save downloaded weights"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for huggingface_hub"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (for private models)"
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        default=None,
        help="Download only specific files (e.g., config.json model.safetensors)"
    )
    
    args = parser.parse_args()
    
    if not HF_AVAILABLE:
        print("Error: huggingface_hub is required but not installed")
        print("Install with: pip install huggingface_hub")
        sys.exit(1)
    
    if args.files:
        # Download specific files only
        download_specific_files(
            model_name=args.model_name,
            files=args.files,
            output_dir=args.output_dir,
            token=args.token
        )
    else:
        # Download entire model
        download_gpt_oss_weights(
            model_name=args.model_name,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            token=args.token
        )


if __name__ == "__main__":
    main()
