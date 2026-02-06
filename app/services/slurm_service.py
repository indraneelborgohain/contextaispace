"""SLURM job submission service for GPT-OSS inference"""
import os
import sys
import json
import re
import subprocess
from pathlib import Path
from string import Template
from typing import Optional, Tuple, List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    JOBS_DIR, LOGS_DIR, RESULTS_DIR, PROJECT_DIR,
    SLURM_ACCOUNT, SLURM_TIME_LIMIT, MIN_GPUS_REQUIRED,
    MODEL_CHECKPOINT_PATH, VENV_PATH
)

# Partitions to exclude
EXCLUDED_PARTITIONS = [
    "cooper-lake", "defq-cpu", "defq-gpu-large", "defq-scavenger",
    "defq-gpu-small", "b200-7gpu-180gb", "b200-8gpu-180gb", "a4500-8gpu-20gb"
]

# SLURM template for GPT-OSS inference jobs
GPTOSS_INFERENCE_TEMPLATE = """#!/bin/bash

#SBATCH --partition=$partition
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=gptoss_$job_id
#SBATCH --error=$log_dir/$job_id.err
#SBATCH --output=$log_dir/$job_id.out
#SBATCH --time=$time_limit
#SBATCH --account=$account

# Environment setup
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# Load required modules
module load gcc/12.1.0
module load cuda11.2/toolkit/11.2.0
module load cudnn/8.2.0-cuda-11.3

echo "Starting GPT-OSS inference job: $job_id"
echo "CUDA devices: $$CUDA_VISIBLE_DEVICES"
hostname
date

# Activate virtual environment if exists
if [ -f "$venv_path/bin/activate" ]; then
    source $venv_path/bin/activate
fi

# Run the inference worker
cd $project_dir
python -u $worker_script --job-id $job_id --job-file $job_file --result-file $result_file

echo "Job completed"
date
"""


def get_used_gpus() -> Dict[str, int]:
    """Get number of GPUs currently used per partition"""
    try:
        cmd = ["squeue", "-o", "%i %P %u %T %C %b %G"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        used_gpus_per_node = {}
        for line in result.stdout.strip().split('\n'):
            if "RUNNING" in line and 'N/A' not in line:
                infos = line.split()
                nodename = infos[1]
                try:
                    used_gpus = int(infos[5].split(':')[-1])
                except (IndexError, ValueError):
                    continue
                
                if nodename not in used_gpus_per_node:
                    used_gpus_per_node[nodename] = used_gpus
                else:
                    used_gpus_per_node[nodename] += used_gpus
        
        return used_gpus_per_node
    except Exception:
        return {}


def get_available_gpu_partitions(min_gpus: int = 1) -> List[str]:
    """
    Get available SLURM partitions that meet the minimum GPU requirement.
    
    Args:
        min_gpus: Minimum number of GPUs required
        
    Returns:
        List of partition names
    """
    try:
        cmd = ["sinfo", "-o", "%P %G %a"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        available_partitions = []
        mapofusedgpus = get_used_gpus()
        
        for line in result.stdout.strip().split('\n'):
            if not line or 'gpu' not in line.lower():
                continue
            if 'mix' in line:
                continue
            
            parts = line.split()
            if len(parts) < 3:
                continue
                
            partition, gpus_str, avail = parts[0], parts[1], parts[2]
            partition = partition.replace("*", "")
            
            if partition in EXCLUDED_PARTITIONS:
                continue
            
            # Parse GPU count
            try:
                gpus_per_node = int(gpus_str.split(":")[2].split('(')[0])
            except (IndexError, ValueError):
                continue
            
            # Get total nodes
            try:
                cmd_p = ["scontrol", "show", "partition", partition]
                result_p = subprocess.run(cmd_p, capture_output=True, text=True, check=True)
                match = re.search(r'TotalNodes=(\d+)', result_p.stdout)
                no_nodes = int(match.group(1)) if match else 1
            except Exception:
                no_nodes = 1
            
            gpu_count = min(no_nodes * gpus_per_node, 100)
            
            if avail.lower() == 'up' and 'cpu' not in partition.lower():
                used = mapofusedgpus.get(partition, 0)
                if (gpu_count - used) >= min_gpus:
                    available_partitions.append(partition)
        
        return available_partitions
        
    except subprocess.CalledProcessError as e:
        print(f"Error running sinfo command: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []


def get_best_partition() -> Optional[str]:
    """
    Get the best available GPU partition.
    Returns the first available partition that meets GPU requirements.
    """
    available_partitions = get_available_gpu_partitions(min_gpus=MIN_GPUS_REQUIRED)
    
    if not available_partitions:
        print("Warning: No GPU partitions available!")
        return None
    
    # Prefer certain partitions (order by preference)
    preferred_order = ['a100-8gpu-80gb', 'h100-8gpu-80gb', 'a100-8gpu-40gb', 'v100-8gpu-16gb']
    
    for preferred in preferred_order:
        if preferred in available_partitions:
            return preferred
    
    return available_partitions[0]


def submit_inference_job(job_id: str, prompt: str) -> Tuple[bool, str, Optional[str]]:
    """
    Submit a SLURM job for GPT-OSS inference.
    
    Args:
        job_id: Unique job identifier
        prompt: The input prompt for the model
        
    Returns:
        Tuple of (success, message, slurm_job_id)
    """
    # Get available partition
    partition = get_best_partition()
    if not partition:
        return False, "No GPU partitions available", None
    
    # Prepare job data file
    job_data = {
        "job_id": job_id,
        "prompt": prompt,
        "checkpoint_path": MODEL_CHECKPOINT_PATH
    }
    
    job_file = JOBS_DIR / f"{job_id}.json"
    result_file = RESULTS_DIR / f"{job_id}.json"
    sbatch_file = JOBS_DIR / f"{job_id}.sbatch"
    
    # Write job data
    with open(job_file, 'w') as f:
        json.dump(job_data, f, indent=2)
    
    # Prepare sbatch script
    worker_script = Path(__file__).parent.parent / "workers" / "inference_worker.py"
    
    template = Template(GPTOSS_INFERENCE_TEMPLATE)
    script_content = template.substitute(
        partition=partition,
        job_id=job_id,
        log_dir=str(LOGS_DIR),
        time_limit=SLURM_TIME_LIMIT,
        account=SLURM_ACCOUNT,
        venv_path=VENV_PATH,
        project_dir=str(PROJECT_DIR),
        worker_script=str(worker_script),
        job_file=str(job_file),
        result_file=str(result_file)
    )
    
    # Write sbatch script
    with open(sbatch_file, 'w') as f:
        f.write(script_content)
    
    # Submit job
    try:
        result = subprocess.run(
            ["sbatch", str(sbatch_file)],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse SLURM job ID from output
        # Expected format: "Submitted batch job 12345"
        output = result.stdout.strip()
        slurm_job_id = output.split()[-1]
        
        return True, f"Job submitted to partition {partition}", slurm_job_id
        
    except subprocess.CalledProcessError as e:
        error_msg = f"SLURM submission failed: {e.stderr}"
        return False, error_msg, None
    except Exception as e:
        return False, f"Unexpected error: {str(e)}", None


def cancel_slurm_job(slurm_job_id: str) -> Tuple[bool, str]:
    """Cancel a SLURM job"""
    try:
        subprocess.run(
            ["scancel", slurm_job_id],
            capture_output=True,
            text=True,
            check=True
        )
        return True, f"Job {slurm_job_id} cancelled"
    except subprocess.CalledProcessError as e:
        return False, f"Failed to cancel job: {e.stderr}"


def get_slurm_job_status(slurm_job_id: str) -> Optional[str]:
    """Get the status of a SLURM job"""
    try:
        result = subprocess.run(
            ["squeue", "-j", slurm_job_id, "-o", "%T", "--noheader"],
            capture_output=True,
            text=True
        )
        status = result.stdout.strip()
        if status:
            return status.lower()  # e.g., 'running', 'pending', 'completing'
        return None  # Job not in queue (completed or failed)
    except Exception:
        return None


def check_job_result(job_id: str) -> Tuple[bool, Optional[dict]]:
    """Check if a job result file exists and read it"""
    result_file = RESULTS_DIR / f"{job_id}.json"
    
    if result_file.exists():
        try:
            with open(result_file, 'r') as f:
                return True, json.load(f)
        except Exception as e:
            return False, {"error": f"Failed to read result: {str(e)}"}
    
    return False, None
