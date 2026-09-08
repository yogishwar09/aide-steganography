"""DCT-domain embedding - report 2.1 step 7(b).

Applied to textured, mid-frequency-rich blocks. Mid-frequency DCT
coefficients (skipping the DC/low-frequency energy and the noisiest high
frequencies) are modified in pairs using relative-magnitude QIM, which
survives JPEG-style requantization better than raw coefficient LSB
substitution - matching the report's claim of inherent JPEG resilience.
"""
from __future__ import annotations

import numpy as np
from scipy.fftpack import dct, idct

from ._common import embed_bits_pairwise, extract_bits_pairwise, zigzag_indices


# Calibrated empirically against cv2 JPEG quality=50 re-compression of the
# whole stego image (not just the isolated block): deltas much below this
# get rounded away by JPEG's own DCT+quantization pass on the same
# coefficients, silently destroying the embedded relation even though the
# block itself looks fine in isolation. See aide/train.py's docstring for
# the general "measure, don't assume" approach this project takes.
DELTA = 40.0
_PAIRS_PER_BPP_DIVISOR = 8  # controls how many coefficient pairs one bpp "unit" buys


def _dct2(block: np.ndarray) -> np.ndarray:
    return dct(dct(block.astype(np.float64), axis=0, norm="ortho"), axis=1, norm="ortho")


def _idct2(coeffs: np.ndarray) -> np.ndarray:
    return idct(idct(coeffs, axis=0, norm="ortho"), axis=1, norm="ortho")


def _select_pairs(block_shape: tuple[int, int], bpp: int) -> list[tuple[int, int]]:
    h, w = block_shape
    assert h == w, "DCT embedding assumes square blocks"
    n = h * w
    zz = zigzag_indices(h)
    skip_low = max(1, n // 16)   # drop DC + very low frequencies (visible if disturbed)
    skip_high = max(1, n // 8)   # drop the noisiest highest frequencies
    mid = zz[skip_low: n - skip_high]

    base_pairs = max(1, len(mid) // (2 * _PAIRS_PER_BPP_DIVISOR))
    n_pairs = min(len(mid) // 2, base_pairs * bpp)
    return [(mid[2 * k], mid[2 * k + 1]) for k in range(n_pairs)]


def capacity_bits(block_shape: tuple[int, int], bpp: int) -> int:
    return len(_select_pairs(block_shape, bpp))


def embed(block: np.ndarray, bpp: int, bits: np.ndarray) -> np.ndarray:
    pairs = _select_pairs(block.shape, bpp)
    if len(bits) != len(pairs):
        raise ValueError(f"expected {len(pairs)} bits, got {len(bits)}")
    coeffs = _dct2(block)
    flat = coeffs.flatten()
    flat = embed_bits_pairwise(flat, pairs, np.asarray(bits, dtype=np.uint8), DELTA)
    coeffs = flat.reshape(block.shape)
    out = _idct2(coeffs)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def extract(block: np.ndarray, bpp: int, num_bits: int | None = None) -> np.ndarray:
    pairs = _select_pairs(block.shape, bpp)
    coeffs = _dct2(block)
    flat = coeffs.flatten()
    bits = extract_bits_pairwise(flat, pairs)
    return bits if num_bits is None else bits[:num_bits]
