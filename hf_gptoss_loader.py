"""
Utilities for downloading GPT-OSS 20B weights from Hugging Face
and loading them into the local GPT-OSS model implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from huggingface_hub import snapshot_download

from architecture.gptoss import ModelConfig, Transformer
from architecture.model_loader import load_from_huggingface


def download_hf_weights(
    repo_id: str,
    local_dir: str | Path = "models/gptoss-20b",
    revision: Optional[str] = None,
    token: Optional[str] = None,
    allow_patterns: Optional[Iterable[str]] = None,
    ignore_patterns: Optional[Iterable[str]] = None,
) -> Path:
    """
    Download a Hugging Face repo snapshot to a local directory.

    Args:
        repo_id: Hugging Face repo id (e.g., "org/model-name").
        local_dir: Local directory to store the snapshot.
        revision: Optional revision/branch/tag/commit.
        token: Optional Hugging Face token (or set HF_TOKEN env var).
        allow_patterns: Optional allowlist patterns for downloading files.
        ignore_patterns: Optional ignore patterns.

    Returns:
        Path to the downloaded snapshot directory.
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        revision=revision,
        token=token,
        allow_patterns=list(allow_patterns) if allow_patterns else None,
        ignore_patterns=list(ignore_patterns) if ignore_patterns else None,
    )
    return Path(snapshot_path)


def load_gptoss_from_hf(
    repo_id: str,
    local_dir: str | Path = "models/gptoss-20b",
    device: str = "cuda:0",
    config: Optional[ModelConfig] = None,
    strict: bool = False,
    safetensors_filename: str = "model.safetensors",
    revision: Optional[str] = None,
    token: Optional[str] = None,
    allow_patterns: Optional[Iterable[str]] = None,
    ignore_patterns: Optional[Iterable[str]] = None,
) -> Transformer:
    """
    Download the repo snapshot (if needed) and load weights into the GPT-OSS model.

    Args:
        repo_id: Hugging Face repo id (e.g., "org/model-name").
        local_dir: Local directory to store the snapshot.
        device: Device to load model on (e.g., "cuda:0" or "cpu").
        config: Optional ModelConfig. If None, loads from config.json in repo.
        strict: Whether to strictly enforce state_dict keys match.
        safetensors_filename: Name of the safetensors file in the repo.
        revision: Optional revision/branch/tag/commit.
        token: Optional Hugging Face token (or set HF_TOKEN env var).
        allow_patterns: Optional allowlist patterns for downloading files.
        ignore_patterns: Optional ignore patterns.

    Returns:
        Loaded GPT-OSS Transformer model.
    """
    snapshot_path = download_hf_weights(
        repo_id=repo_id,
        local_dir=local_dir,
        revision=revision,
        token=token,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )

    return load_from_huggingface(
        model_path=str(snapshot_path),
        config=config,
        device=device,
        strict=strict,
        safetensors_filename=safetensors_filename,
    )
