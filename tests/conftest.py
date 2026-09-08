from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "data" / "samples"
MODEL_DIR = REPO_ROOT / "data" / "models"


@pytest.fixture
def synthetic_image() -> np.ndarray:
    """A deterministic 64x64 grayscale image with a mix of smooth, textured,
    and edge-heavy regions - enough for fast unit tests without needing the
    real dataset."""
    rng = np.random.default_rng(42)
    img = np.zeros((64, 64), dtype=np.uint8)
    img[:32, :32] = 180  # smooth flat region
    img[:32, 32:] = (rng.integers(0, 256, (32, 32))).astype(np.uint8)  # noisy/textured
    xs, ys = np.meshgrid(np.arange(32), np.arange(32))
    img[32:, :32] = ((xs + ys) % 2 * 255).astype(np.uint8)  # checkerboard = strong edges
    img[32:, 32:] = np.clip(128 + 40 * np.sin(np.arange(32) / 2.0), 0, 255).astype(np.uint8)
    return img


@pytest.fixture
def real_sample_paths() -> list[Path]:
    paths = sorted(SAMPLES_DIR.glob("*.pgm"))
    if not paths:
        pytest.skip(f"no sample images in {SAMPLES_DIR}; run scripts/fetch_dataset.py first")
    return paths


@pytest.fixture
def real_sample_image_path(real_sample_paths: list[Path]) -> Path:
    return real_sample_paths[0]


@pytest.fixture
def trained_model():
    from aide.model import AIDEModel

    if not (MODEL_DIR / "weights.json").exists():
        pytest.skip(f"no trained model at {MODEL_DIR}; run 'aide train' first")
    return AIDEModel.load(MODEL_DIR)
