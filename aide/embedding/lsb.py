"""LSB substitution embedding - report 2.1 step 7(a).

Applied to smooth, low-texture blocks: highest capacity (bpp * num_pixels
bits) at lowest detectability risk in flat regions, since the low-order
bits of a flat region are already close to random-looking noise.
"""
from __future__ import annotations

import numpy as np

MAX_BPP = 4


def capacity_bits(block_shape: tuple[int, int], bpp: int) -> int:
    h, w = block_shape
    return h * w * bpp


def embed(block: np.ndarray, bpp: int, bits: np.ndarray) -> np.ndarray:
    flat = block.flatten().astype(np.uint8)
    n_pixels = flat.size
    needed = n_pixels * bpp
    if len(bits) != needed:
        raise ValueError(f"expected {needed} bits, got {len(bits)}")

    bits_2d = np.asarray(bits, dtype=np.uint8).reshape(n_pixels, bpp)
    weights = (1 << np.arange(bpp - 1, -1, -1)).astype(np.uint8)
    values = (bits_2d * weights).sum(axis=1).astype(np.uint8)

    mask = np.uint8((0xFF << bpp) & 0xFF)
    out = (flat & mask) | values
    return out.reshape(block.shape).astype(np.uint8)


def extract(block: np.ndarray, bpp: int, num_bits: int | None = None) -> np.ndarray:
    flat = block.flatten().astype(np.uint8)
    n_pixels = flat.size
    total_bits = n_pixels * bpp
    if num_bits is None:
        num_bits = total_bits

    values = flat & np.uint8((1 << bpp) - 1)
    bits_2d = np.zeros((n_pixels, bpp), dtype=np.uint8)
    for k in range(bpp):
        bits_2d[:, k] = (values >> (bpp - 1 - k)) & 1
    return bits_2d.reshape(-1)[:num_bits]
