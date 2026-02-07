# GPT-OSS 20B — RunPod Serverless Deployment

Deploy the GPT-OSS 20B model as a RunPod serverless endpoint.

## Prerequisites

- Docker installed locally (for building the image)
- A [RunPod](https://runpod.io) account
- A [Docker Hub](https://hub.docker.com) (or other registry) account
- GPU with at least 48 GB VRAM (A6000 / A100 recommended)

## Project Files

| File | Purpose |
|---|---|
| `Dockerfile` | Container image for RunPod serverless workers |
| `rp_handler.py` | RunPod serverless handler — loads model, processes requests |

## 1. Build & Push the Docker Image

```bash
# Build
docker build -t <your-dockerhub-user>/gptoss-runpod:latest .

# Push
docker push <your-dockerhub-user>/gptoss-runpod:latest
```

### Including Model Weights in the Image (optional)

To bake model weights directly into the image and avoid download times on cold start:

```bash
# Place weights in the expected directory first
mkdir -p model/gpt-oss-20b/original/
# Copy your safetensor shards into the directory above

# Then build — the COPY instruction will include them
docker build -t <your-dockerhub-user>/gptoss-runpod:latest .
```

Alternatively, set the `HF_REPO_ID` environment variable to download weights from HuggingFace at container startup (slower cold starts but smaller image).

## 2. Create a RunPod Serverless Endpoint

1. Go to [RunPod Console → Serverless](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Configure:
   - **Container Image**: `<your-dockerhub-user>/gptoss-runpod:latest`
   - **GPU Type**: 48GB+ VRAM (A6000, A100-40GB, A100-80GB)
   - **Min Workers**: `0` (scale to zero when idle)
   - **Max Workers**: set based on expected traffic
   - **Idle Timeout**: `60` seconds (keep warm between requests)
   - **Container Disk**: `50 GB` (more if weights are downloaded at runtime)
4. Add environment variables (if downloading weights at runtime):
   - `HF_REPO_ID` = your HuggingFace repo (e.g. `your-org/gptoss-20b`)
   - `HF_TOKEN` = your HuggingFace access token (if repo is private)
5. Click **Create**

## 3. Calling the Endpoint

### Synchronous (runsync)

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "What is the capital of France?",
      "max_tokens": 100,
      "temperature": 0.8
    }
  }'
```

### Asynchronous (run + status)

```bash
# Submit job
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/run" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Explain quantum entanglement",
      "max_tokens": 200,
      "temperature": 0.7
    }
  }'

# Check status (use the job ID from the response above)
curl "https://api.runpod.ai/v2/<ENDPOINT_ID>/status/<JOB_ID>" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>"
```

### Python Client

```python
import runpod

runpod.api_key = "your_runpod_api_key"
endpoint = runpod.Endpoint("ENDPOINT_ID")

result = endpoint.run_sync({
    "input": {
        "prompt": "What is the capital of France?",
        "max_tokens": 100,
        "temperature": 0.8,
    }
})

print(result["response"])
```

## 4. Request & Response Format

### Request Input

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | *required* | The user's input text |
| `max_tokens` | int | `100` | Max tokens to generate (1–4096) |
| `temperature` | float | `0.8` | Sampling temperature (0.01–2.0) |

### Response Output

```json
{
  "response": "Paris is the capital of France.",
  "system_message": "Be informative and concise...",
  "generation_time": 2.314
}
```

## 5. Local Testing

Test the handler locally before deploying:

```bash
# With Docker (requires NVIDIA GPU + nvidia-docker)
docker run --gpus all \
  -v /path/to/weights:/app/model/gpt-oss-20b/original \
  -p 8000:8000 \
  <your-dockerhub-user>/gptoss-runpod:latest

# Send a test request
curl -X POST http://localhost:8000/runsync \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "Hello, world!"}}'
```

## 6. Troubleshooting

| Issue | Solution |
|---|---|
| OOM (Out of Memory) | Use a GPU with more VRAM (A100-80GB recommended) |
| Slow cold starts | Bake weights into the Docker image instead of downloading at runtime |
| `TokenGenerator` errors | Verify checkpoint path and that all shard files are present |
| HuggingFace download fails | Check `HF_TOKEN` is set and has access to the repo |
