"""Block partitioning and per-block feature extraction for AIDE.

Implements report section 2.1 steps 2-3: the image is split into
non-overlapping 8x8 or 16x16 blocks (size chosen adaptively from
resolution), and four normalised signals are computed per block:
texture (LBP + GLCM energy), Shannon entropy, pixel variance, and edge
strength (Sobel + Canny). These feed the AIDE scoring policy in
``aide.scoring``.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

FEATURE_NAMES = ("texture", "entropy", "variance", "edge_strength")


@dataclass(frozen=True)
class Block:
    """A single non-overlapping image block and its pixel bounds."""

    index: int
    row: int
    col: int
    size: int
    y0: int
    x0: int
    y1: int
    x1: int

    def slice(self):
        return slice(self.y0, self.y1), slice(self.x0, self.x1)


def choose_block_size(height: int, width: int, small: int = 8, large: int = 16,
                       resolution_threshold: int = 512) -> int:
    """Higher-resolution images use larger blocks to cut computation while
    keeping capacity, per report section 2.1 step 2."""
    return large if max(height, width) >= resolution_threshold else small


def partition_blocks(height: int, width: int, block_size: int) -> list[Block]:
    """Partition an HxW image into non-overlapping block_size x block_size
    blocks in row-major order. Trailing rows/columns that don't fill a full
    block are dropped (documented limitation for this prototype)."""
    blocks: list[Block] = []
    idx = 0
    n_rows = height // block_size
    n_cols = width // block_size
    for r in range(n_rows):
        for c in range(n_cols):
            y0, x0 = r * block_size, c * block_size
            blocks.append(Block(idx, r, c, block_size, y0, x0, y0 + block_size, x0 + block_size))
            idx += 1
    return blocks


def _normalized_shannon_entropy(values: np.ndarray, n_bins: int) -> float:
    hist, _ = np.histogram(values, bins=n_bins, range=(0, n_bins - 1 if n_bins <= 256 else 255))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist[hist > 0] / total
    entropy = -np.sum(p * np.log2(p))
    max_entropy = np.log2(n_bins)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def block_texture(gray_block: np.ndarray) -> float:
    """LBP histogram entropy + inverse GLCM energy, averaged into [0, 1].
    High texture (busy/complex regions) -> value near 1."""
    lbp = local_binary_pattern(gray_block, P=8, R=1, method="uniform")
    n_bins = 10  # P + 2 for uniform LBP with P=8
    lbp_entropy = _normalized_shannon_entropy(lbp.ravel(), n_bins)

    glcm = graycomatrix(gray_block, distances=[1], angles=[0], levels=256,
                         symmetric=True, normed=True)
    energy = float(graycoprops(glcm, "energy")[0, 0])
    energy = min(max(energy, 0.0), 1.0)

    return float(np.clip(0.5 * lbp_entropy + 0.5 * (1.0 - energy), 0.0, 1.0))


def block_entropy(gray_block: np.ndarray) -> float:
    """Shannon entropy H(X) of the pixel intensity distribution, normalised
    by log2(256)."""
    return _normalized_shannon_entropy(gray_block.ravel(), 256)


def block_variance(gray_block: np.ndarray) -> float:
    """Pixel standard deviation normalised by the theoretical max (127.5)."""
    std = float(gray_block.astype(np.float64).std())
    return float(np.clip(std / 127.5, 0.0, 1.0))


def block_edge_strength(gray_block: np.ndarray) -> float:
    """Sobel gradient magnitude + Canny edge density, averaged into [0, 1]."""
    gx = cv2.Sobel(gray_block, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_block, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    sobel_score = float(np.clip(mag.mean() / (4 * 255.0), 0.0, 1.0))

    edges = cv2.Canny(gray_block, 100, 200)
    canny_density = float((edges > 0).mean())

    return float(np.clip(0.5 * sobel_score + 0.5 * canny_density, 0.0, 1.0))


def extract_block_features(gray_block: np.ndarray) -> np.ndarray:
    """Return the 4-element feature vector [texture, entropy, variance,
    edge_strength] for one grayscale block. block_id is tracked separately
    (see report's v(b) definition) since it is a positional index, not a
    generalisable signal, and is therefore kept out of the ML input."""
    return np.array([
        block_texture(gray_block),
        block_entropy(gray_block),
        block_variance(gray_block),
        block_edge_strength(gray_block),
    ], dtype=np.float64)


def extract_image_features(gray: np.ndarray, block_size: int | None = None
                            ) -> tuple[list[Block], np.ndarray]:
    """Partition a grayscale image and extract the feature matrix (N, 4)
    over all blocks, in the same order as the returned Block list."""
    if gray.ndim != 2:
        raise ValueError("extract_image_features expects a single-channel (grayscale) image")
    h, w = gray.shape
    if block_size is None:
        block_size = choose_block_size(h, w)
    blocks = partition_blocks(h, w, block_size)
    features = np.zeros((len(blocks), 4), dtype=np.float64)
    for b in blocks:
        ys, xs = b.slice()
        features[b.index] = extract_block_features(gray[ys, xs])
    return blocks, features
