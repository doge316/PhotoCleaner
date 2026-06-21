"""Device selection utilities for PhotoCleaner.

Provides a single source of truth for picking CPU vs CUDA across the pipeline.
"""

from __future__ import annotations

import os

import torch


def get_device() -> torch.device:
    """Return the device the pipeline should use.

    Honors the ``PHOTOCLEANER_FORCE_CPU=1`` environment variable so users can
    explicitly fall back to CPU (useful for debugging or low-VRAM machines).
    Otherwise prefers CUDA when available.
    """
    if os.environ.get("PHOTOCLEANER_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
