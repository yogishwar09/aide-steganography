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


def embed(block: np.ndarray, bpp: int, bits: np.ndarray, delta: float | None = None) -> np.ndarray:
    """`delta` is accepted for interface parity with DCT/DWT and ignored."""
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


# LSB has no analog margin: a substituted bit is either intact or flipped,
# and every attack measured flips ~half of them. Reported as a constant,
# deliberately low reliability so reliability-guided decoding treats LSB
# bits as the first candidates for erasure.
LSB_SOFT_MAGNITUDE = 1.0


def extract_soft(block: np.ndarray, bpp: int, num_bits: int | None = None) -> np.ndarray:
    bits = extract(block, bpp, num_bits).astype(np.float64)
    return (bits * 2.0 - 1.0) * LSB_SOFT_MAGNITUDE
