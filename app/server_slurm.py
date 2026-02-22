#!/usr/bin/env python3
"""
Flask GPT-OSS Inference Server
Runs on login node, submits SLURM jobs for GPU inference on demand.
"""
import os
import sys
import json
import subprocess
import time
import uuid
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

RESULTS_DIR = Path("/tmp/gpt_oss_results")
RESULTS_DIR.mkdir(exist_ok=True)

# Path to your inference script
PROJECT_ROOT = Path(__file__).parent.parent


def run_inference_slurm(prompt, timeout=300):
    """Submit inference as a SLURM job and wait for result."""
    job_id = str(uuid.uuid4())[:8]
    input_file = RESULTS_DIR / f"{job_id}_input.json"
    output_file = RESULTS_DIR / f"{job_id}_output.json"

    # Save prompt
    input_file.write_text(json.dumps({"prompt": prompt}))

    # Inline SLURM script
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=infer-{job_id}
#SBATCH --partition=mr_neuro,tp_models,b200-scavenger,defq-scavenger,defq-scavenger,v100-16gpu-32gb,defq-scavenger,a100-8gpu-40gb,prostateai,defq-scavenger,deformreg,rapidplan,end2end,a100-8gpu-40gb,a100-4gpu-40gb,embolism,defq-scavenger,vascular,a100-8gpu-80gb,a100-8gpu-80gb,motionmgmt,defq-scavenger,h100-8gpu-80gb,defq-scavenger,v100-8gpu-16gb,defq-scavenger,vl_models,defq-scavenger,b200-scavenger,dosecomp,v100-16gpu-32gb,operational_twin,diffusion_models,mr_neuro
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --exclude=node016,node131,node015,node182
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --account=rctcu02376
#SBATCH --output={RESULTS_DIR}/{job_id}_slurm.log

python -c "
import sys, json
sys.path.insert(0, '{PROJECT_ROOT}')
from inference import create_models, generateResults

data = json.loads(open('{input_file}').read())
gen, sys_gen = create_models()
result = generateResults(data['prompt'], generator=gen, system_gen=sys_gen)
json.dump({{'response': result}}, open('{output_file}', 'w'))
"
"""
    script_file = RESULTS_DIR / f"{job_id}.sh"
    script_file.write_text(slurm_script)

    # Submit
    result = subprocess.run(["sbatch", str(script_file)], capture_output=True, text=True)
    if result.returncode != 0:
        return None, f"SLURM submit failed: {result.stderr}"

    # Poll for result
    start = time.time()
    while time.time() - start < timeout:
        if output_file.exists():
            response = json.loads(output_file.read_text())
            # Cleanup
            for f in RESULTS_DIR.glob(f"{job_id}*"):
                f.unlink()
            return response["response"], None
        time.sleep(2)

    # Cleanup on timeout
    for f in RESULTS_DIR.glob(f"{job_id}*"):
        f.unlink()
    return None, "Timeout waiting for inference result"


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "timestamp": int(time.time())})


@app.route('/api/inference', methods=['POST'])
def inference():
    try:
        data = request.get_json()
        if not data or not data.get("prompt"):
            return jsonify({"error": "Prompt is required"}), 400

        response, error = run_inference_slurm(data["prompt"])

        if error:
            return jsonify({"error": error}), 500

        return jsonify({
            "prompt": data["prompt"],
            "response": response,
            "timestamp": int(time.time())
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)