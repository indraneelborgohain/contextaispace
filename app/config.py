"""Configuration for the GPT-OSS Flask Server with SLURM Backend"""
import os
from pathlib import Path

# Server configuration
HOST = os.getenv("GPTOSS_SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("GPTOSS_SERVER_PORT", 5000))
DEBUG = os.getenv("GPTOSS_SERVER_DEBUG", "false").lower() == "true"

# Paths
BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
JOBS_DIR = BASE_DIR / "jobs"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
JOBS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# SLURM configuration
SLURM_ACCOUNT = os.getenv("SLURM_ACCOUNT", "pct_cav")
SLURM_TIME_LIMIT = os.getenv("SLURM_TIME_LIMIT", "01:00:00")
MIN_GPUS_REQUIRED = int(os.getenv("MIN_GPUS_REQUIRED", 1))

# Model configuration
MODEL_CHECKPOINT_PATH = os.getenv(
    "MODEL_CHECKPOINT_PATH", 
    str(PROJECT_DIR / "model/gpt-oss-120b/original/")
)

# Virtual environment path for SLURM jobs
VENV_PATH = os.getenv("VENV_PATH", "/pct_cav/Data/CMR/contextaispace/venv")

# Job timeout (seconds) - how long to wait before marking job as stale
JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT", 3600))

# SQLite database path for job tracking
DATABASE_PATH = BASE_DIR / "jobs.db"
