#!/usr/bin/env python3
"""
Flask GPT-OSS Server with SLURM Backend

A REST API server that accepts prompts, submits inference jobs to SLURM,
and returns results asynchronously. Run this on a login node - the actual
inference happens on GPU compute nodes via SLURM.
"""
import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from functools import wraps
import time

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from config import HOST, PORT, DEBUG, RESULTS_DIR
from models.job import (
    Job, JobStatus, create_job, get_job, update_job,
    get_all_jobs, delete_job
)
from services.slurm_service import (
    submit_inference_job, cancel_slurm_job,
    get_slurm_job_status, check_job_result,
    get_best_partition, get_available_gpu_partitions
)

app = Flask(__name__, static_folder='static')
CORS(app)

# Optional: Simple API key authentication
API_KEY = os.getenv("GPTOSS_API_KEY")


def require_api_key(f):
    """Decorator for optional API key authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if API_KEY:
            provided_key = request.headers.get("X-API-Key")
            if provided_key != API_KEY:
                return jsonify({"error": "Invalid or missing API key"}), 401
        return f(*args, **kwargs)
    return decorated


# ============================================================================
# Static Files & Health Endpoints
# ============================================================================

@app.route('/')
def index():
    """Serve the main chat interface"""
    return send_from_directory('static', 'index.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "mode": "slurm",
        "timestamp": int(time.time())
    })


@app.route('/api/v1/info', methods=['GET'])
def info():
    """Server info endpoint"""
    return jsonify({
        "service": "GPT-OSS SLURM Server",
        "version": "1.0.0",
        "mode": "asynchronous",
        "endpoints": {
            "submit_chat": "POST /api/chat",
            "submit": "POST /api/v1/inference",
            "status": "GET /api/v1/inference/<job_id>",
            "cancel": "DELETE /api/v1/inference/<job_id>",
            "jobs": "GET /api/v1/jobs",
            "partitions": "GET /api/v1/partitions"
        }
    })


@app.route('/api/v1/partitions', methods=['GET'])
@require_api_key
def get_partitions():
    """Get available GPU partitions"""
    min_gpus = request.args.get("min_gpus", 1, type=int)
    partitions = get_available_gpu_partitions(min_gpus=min_gpus)
    best_partition = get_best_partition()

    return jsonify({
        "available_partitions": partitions,
        "recommended_partition": best_partition,
        "min_gpus_requested": min_gpus
    })


# ============================================================================
# Chat Endpoint (compatible with existing UI)
# ============================================================================

@app.route('/api/chat', methods=['POST'])
@require_api_key
def chat():
    """
    Chat endpoint that submits a job to SLURM.
    
    This endpoint is async - it returns immediately with a job_id,
    and you need to poll /api/v1/inference/<job_id> for the result.
    
    Expected JSON:
    {
        "message": "User's question"
    }
    
    Returns:
    {
        "job_id": "uuid",
        "status": "submitted",
        "message": "Job submitted to SLURM. Poll /api/v1/inference/<job_id> for result."
    }
    """
    try:
        data = request.get_json()
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Create job
        job = create_job(prompt=user_message)

        # Submit to SLURM
        success, message, slurm_job_id = submit_inference_job(
            job_id=job.id,
            prompt=user_message
        )

        if success:
            update_job(
                job.id,
                status=JobStatus.SUBMITTED,
                slurm_job_id=slurm_job_id
            )

            return jsonify({
                "job_id": job.id,
                "status": "submitted",
                "slurm_job_id": slurm_job_id,
                "message": message,
                "poll_url": f"/api/v1/inference/{job.id}",
                "timestamp": int(time.time())
            }), 202
        else:
            update_job(job.id, status=JobStatus.FAILED, error=message)
            return jsonify({
                "job_id": job.id,
                "status": "failed",
                "error": message
            }), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Inference Endpoints
# ============================================================================

@app.route('/api/v1/inference', methods=['POST'])
@require_api_key
def submit_inference():
    """
    Submit a new inference job.
    
    Request body:
    {
        "prompt": "Your prompt here"
    }
    
    Returns:
    {
        "job_id": "uuid",
        "status": "submitted",
        "message": "Job submitted successfully"
    }
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "Request body is required"}), 400

    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    # Create job
    job = create_job(prompt=prompt)

    # Submit to SLURM
    success, message, slurm_job_id = submit_inference_job(
        job_id=job.id,
        prompt=prompt
    )

    if success:
        update_job(
            job.id,
            status=JobStatus.SUBMITTED,
            slurm_job_id=slurm_job_id
        )

        return jsonify({
            "job_id": job.id,
            "status": "submitted",
            "slurm_job_id": slurm_job_id,
            "message": message
        }), 202
    else:
        update_job(job.id, status=JobStatus.FAILED, error=message)

        return jsonify({
            "job_id": job.id,
            "status": "failed",
            "error": message
        }), 500


@app.route('/api/v1/inference/<job_id>', methods=['GET'])
@require_api_key
def get_inference_status(job_id: str):
    """
    Get the status and result of an inference job.
    
    Returns:
    {
        "job_id": "uuid",
        "status": "completed|running|queued|submitted|failed",
        "response": "Generated text...",  # if completed
        "error": "Error message..."  # if failed
    }
    """
    job = get_job(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    # Check and update status if job is still in progress
    if job.status in [JobStatus.SUBMITTED, JobStatus.RUNNING]:
        # Check SLURM status
        if job.slurm_job_id:
            slurm_status = get_slurm_job_status(job.slurm_job_id)

            if slurm_status == "running" and job.status != JobStatus.RUNNING:
                update_job(job.id, status=JobStatus.RUNNING)
                job = get_job(job_id)
            elif slurm_status is None:
                # Job finished in SLURM, check for result
                has_result, result_data = check_job_result(job_id)

                if has_result and result_data:
                    if result_data.get("success"):
                        update_job(
                            job.id,
                            status=JobStatus.COMPLETED,
                            result=result_data.get("result")
                        )
                    else:
                        update_job(
                            job.id,
                            status=JobStatus.FAILED,
                            error=result_data.get("error", "Unknown error")
                        )
                    job = get_job(job_id)

    # Build response
    response = {
        "job_id": job.id,
        "status": job.status.value,
        "prompt": job.prompt[:100] + "..." if len(job.prompt) > 100 else job.prompt,
        "created_at": job.created_at,
        "updated_at": job.updated_at
    }

    if job.slurm_job_id:
        response["slurm_job_id"] = job.slurm_job_id

    if job.status == JobStatus.COMPLETED and job.result:
        response["response"] = job.result

    if job.status == JobStatus.FAILED and job.error:
        response["error"] = job.error

    return jsonify(response)


@app.route('/api/v1/inference/<job_id>', methods=['DELETE'])
@require_api_key
def cancel_inference(job_id: str):
    """Cancel a running or queued inference job"""
    job = get_job(job_id)

    if not job:
        return jsonify({"error": "Job not found"}), 404

    if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
        return jsonify({
            "error": f"Cannot cancel job with status: {job.status.value}"
        }), 400

    # Cancel SLURM job if it exists
    if job.slurm_job_id:
        success, message = cancel_slurm_job(job.slurm_job_id)
        if not success:
            return jsonify({"error": message}), 500

    update_job(job.id, status=JobStatus.CANCELLED)

    return jsonify({
        "job_id": job.id,
        "status": "cancelled",
        "message": "Job cancelled successfully"
    })


@app.route('/api/v1/jobs', methods=['GET'])
@require_api_key
def list_jobs():
    """List all jobs"""
    limit = request.args.get("limit", 50, type=int)
    jobs = get_all_jobs(limit=limit)

    return jsonify({
        "jobs": [job.to_dict() for job in jobs],
        "count": len(jobs)
    })


@app.route('/api/clear', methods=['POST'])
def clear_conversation():
    """Clear conversation history (placeholder for now)"""
    return jsonify({
        'status': 'cleared',
        'message': 'Conversation cleared successfully'
    })


if __name__ == '__main__':
    print("=" * 60)
    print("Starting GPT-OSS SLURM Server")
    print("=" * 60)
    print("\nThis server runs on the login node and submits")
    print("inference jobs to GPU compute nodes via SLURM.")
    print("\n" + "=" * 60)
    print(f"Server: http://localhost:{PORT}")
    print(f"Chat UI: http://localhost:{PORT}")
    print(f"API Endpoint: http://localhost:{PORT}/api/chat")
    print(f"Inference API: http://localhost:{PORT}/api/v1/inference")
    print(f"Health Check: http://localhost:{PORT}/health")
    print("=" * 60)
    print("\nPress CTRL+C to stop the server\n")

    app.run(host=HOST, port=PORT, debug=DEBUG)
