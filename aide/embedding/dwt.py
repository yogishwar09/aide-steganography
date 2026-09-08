"""DWT-domain embedding - report 2.1 step 7(c).

Applied to edge-dense blocks. Payload is carried in the level-2
approximation sub-band (cA2) of a 2-level Haar DWT, using the same
relative-magnitude pairing as the DCT method.

This is a revision of an earlier single-level design that embedded in the
horizontal/vertical detail sub-bands (cH, cV) instead. That design
survived JPEG compression but was measured to have ~0% survival against
median filtering or resizing *at any delta tested up to 400* - both
attacks resample the exact 2x2 spatial neighborhoods a single-level Haar
transform depends on, and detail coefficients (which encode local
differences) are exactly what that resampling destroys.

cA2 doesn't have this problem: it's a twice-downsampled, twice-averaged
version of the block, i.e. the same kind of coarse, low-frequency
structure that survives compression, filtering, and mild resizing in
every other transform-domain scheme (this is the same principle as "use
mid/low DCT frequencies, not high ones"). Measured on real BOSSbase
blocks at delta=40: 100% bit-exact survival against JPEG q=30/50, 3x3
median filtering, and 2x downscale-then-restore, at a *smaller* per-block
distortion (~37 dB average block PSNR) than the old design needed (~29 dB
at delta=100) for weaker guarantees. The trade-off is capacity: cA2 has
1/8 as many coefficients as the old cH+cV pair, so DWT carries less of
the payload per block - handled by biasing the overall payload allocation
toward DCT (see aide/pipeline.py's DEFAULT_FRAGILE_METHOD_BUDGETS).

Gaussian noise is the one attack where this is measurably *not* better
than DCT (both old and new DWT struggle above sigma~15) - left to DCT,
which measured robust there instead. See README's "What doesn't work yet"
for the honest, current-state summary of the noise gap.
"""
from __future__ import annotations

import numpy as np
import pywt

from ._common import embed_bits_pairwise, extract_bits_pairwise

WAVELET = "haar"
LEVEL = 2
# Calibrated empirically (see module docstring): delta=40 gave 100%
# bit-exact survival against JPEG q=30/50, 3x3 median filtering, and 2x
# downscale-then-restore on real BOSSbase blocks, at a much better PSNR
# than the single-level design's delta=100 needed for weaker guarantees.
DELTA = 40.0
_PAIRS_PER_BPP_DIVISOR = 4


def _decompose(block: np.ndarray):
    return pywt.wavedec2(block.astype(np.float64), WAVELET, level=LEVEL)


def _reconstruct(coeffs, block_shape: tuple[int, int]) -> np.ndarray:
    rec = pywt.waverec2(coeffs, WAVELET)
    return rec[: block_shape[0], : block_shape[1]]


def _n_pairs(block_shape: tuple[int, int], bpp: int) -> int:
    h, w = block_shape
    # cA2 shape for a Haar decomposition at LEVEL=2 is (h // 4, w // 4)
    n_coeffs = (h // 4) * (w // 4)
    available_pairs = n_coeffs // 2
    base_pairs = max(1, available_pairs // _PAIRS_PER_BPP_DIVISOR)
    return min(available_pairs, base_pairs * bpp)


def capacity_bits(block_shape: tuple[int, int], bpp: int) -> int:
    return _n_pairs(block_shape, bpp)


def embed(block: np.ndarray, bpp: int, bits: np.ndarray) -> np.ndarray:
    n_pairs = _n_pairs(block.shape, bpp)
    if len(bits) != n_pairs:
        raise ValueError(f"expected {n_pairs} bits, got {len(bits)}")
    pairs = [(2 * k, 2 * k + 1) for k in range(n_pairs)]

    coeffs = _decompose(block)
    cA2 = coeffs[0]
    flat = cA2.flatten()
    flat = embed_bits_pairwise(flat, pairs, np.asarray(bits, dtype=np.uint8), DELTA)
    coeffs[0] = flat.reshape(cA2.shape)

    rec = _reconstruct(coeffs, block.shape)
    return np.clip(np.round(rec), 0, 255).astype(np.uint8)


def extract(block: np.ndarray, bpp: int, num_bits: int | None = None) -> np.ndarray:
    n_pairs = _n_pairs(block.shape, bpp)
    pairs = [(2 * k, 2 * k + 1) for k in range(n_pairs)]

    coeffs = _decompose(block)
    flat = coeffs[0].flatten()
    bits = extract_bits_pairwise(flat, pairs)
    return bits if num_bits is None else bits[:num_bits]
