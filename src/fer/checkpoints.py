"""Read and write the shared, metadata-checked model checkpoint format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .data import LABELS, PREPROCESSING
from .models import MODEL_IDS, make_model, parameter_count


def save_checkpoint(
    path: str | Path,
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> Path:
    """Save the existing FT11 state_dict plus metadata format atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(".tmp")
    torch.save({"state_dict": state_dict, "metadata": metadata}, temporary_path)
    temporary_path.replace(destination)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_manifest_sha256: str | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load an FT11 checkpoint without loading optimizer or executable objects."""
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Checkpoint must contain a state_dict and metadata.")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata is missing or invalid.")
    if metadata.get("model_id") not in MODEL_IDS:
        raise ValueError("Checkpoint model_id is not one of the FT11 baselines.")
    if metadata.get("label_order") != list(LABELS):
        raise ValueError("Checkpoint label order does not match the shared pipeline.")
    if metadata.get("preprocessing") != PREPROCESSING:
        raise ValueError("Checkpoint preprocessing does not match the shared pipeline.")
    if expected_manifest_sha256 and metadata.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("Checkpoint and manifest fingerprints do not match.")

    model = make_model(metadata["model_id"], num_classes=len(LABELS))
    model.load_state_dict(payload["state_dict"], strict=True)
    if metadata.get("parameters") not in (None, parameter_count(model)):
        raise ValueError("Checkpoint parameter count does not match its model.")
    model.to(device)
    model.eval()
    return model, metadata
