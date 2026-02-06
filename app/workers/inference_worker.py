#!/usr/bin/env python
"""
GPT-OSS Inference Worker - Runs on SLURM compute nodes
This script is executed by SLURM to perform actual model inference
"""
import os
import sys
import json
import argparse
import traceback
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def run_inference(job_data: dict) -> dict:
    """
    Run GPT-OSS inference using the generateResults function.
    
    Args:
        job_data: Dictionary containing job_id, prompt, and checkpoint_path
        
    Returns:
        Dictionary with result or error
    """
    job_id = job_data["job_id"]
    prompt = job_data["prompt"]
    
    print(f"[{job_id}] Starting GPT-OSS inference")
    print(f"[{job_id}] Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"[{job_id}] Prompt: {prompt}")
    
    # Check CUDA availability
    import torch
    print(f"[{job_id}] CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"[{job_id}] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[{job_id}] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    try:
        start_time = datetime.now()
        
        # Import and run the generateResults function from inference.py
        from inference import generateResults
        
        print(f"[{job_id}] Running inference...")
        result = generateResults(prompt)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"[{job_id}] Inference completed in {duration:.2f} seconds")
        if result:
            print(f"[{job_id}] Generated {len(result)} characters")
        
        return {
            "success": True,
            "result": result,
            "duration_seconds": duration,
            "completed_at": end_time.isoformat()
        }
        
    except Exception as e:
        error_msg = f"Inference failed: {str(e)}"
        print(f"[{job_id}] ERROR: {error_msg}")
        traceback.print_exc()
        
        return {
            "success": False,
            "error": error_msg,
            "traceback": traceback.format_exc()
        }


def main():
    parser = argparse.ArgumentParser(description="GPT-OSS Inference Worker")
    parser.add_argument("--job-id", required=True, help="Unique job identifier")
    parser.add_argument("--job-file", required=True, help="Path to job JSON file")
    parser.add_argument("--result-file", required=True, help="Path to result JSON file")
    
    args = parser.parse_args()
    
    print(f"=" * 60)
    print(f"GPT-OSS Inference Worker")
    print(f"Job ID: {args.job_id}")
    print(f"Job File: {args.job_file}")
    print(f"Result File: {args.result_file}")
    print(f"=" * 60)
    
    # Load job data
    try:
        with open(args.job_file, 'r') as f:
            job_data = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load job file: {e}")
        result = {"success": False, "error": f"Failed to load job file: {e}"}
        with open(args.result_file, 'w') as f:
            json.dump(result, f, indent=2)
        sys.exit(1)
    
    # Run inference
    result = run_inference(job_data)
    
    # Save result
    try:
        with open(args.result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to: {args.result_file}")
    except Exception as e:
        print(f"ERROR: Failed to save result file: {e}")
        sys.exit(1)
    
    print(f"=" * 60)
    print("Worker finished")
    print(f"=" * 60)
    
    # Exit with appropriate code
    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
