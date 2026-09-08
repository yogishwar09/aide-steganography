"""DWT-domain embedding - report 2.1 step 7(c).

Applied to edge-dense blocks. Payload is carried in the horizontal/vertical
high-frequency sub-bands (cH, cV) of a single-level Haar DWT using the same
relative-magnitude pairing as the DCT method; the approximation sub-band
(cA, which holds most of the block's visible structure) and the diagonal
sub-band (cD, usually the noisiest/least reliable) are left untouched,
giving superior robustness against geometric attacks per the report.
"""
from __future__ import annotations

import numpy as np
import pywt

from ._common import embed_bits_pairwise, extract_bits_pairwise

WAVELET = "haar"
# See aide/embedding/dct.py's DELTA comment: calibrated against cv2 JPEG
# quality=50 re-compression of the whole stego image. DWT sub-band
# coefficients need a larger separation than DCT's to survive the same
# attack, especially at 16x16 block sizes (larger images).
DELTA = 100.0
_PAIRS_PER_BPP_DIVISOR = 16


def _subbands(block: np.ndarray):
    cA, (cH, cV, cD) = pywt.dwt2(block.astype(np.float64), WAVELET)
    return cA, cH, cV, cD


def _select_pairs(block_shape: tuple[int, int], bpp: int) -> tuple[list[tuple[int, int]], int]:
    h, w = block_shape
    sub_h, sub_w = h // 2, w // 2
    n_sub = sub_h * sub_w
    total = 2 * n_sub  # cH and cV concatenated

    base_pairs = max(1, total // (2 * _PAIRS_PER_BPP_DIVISOR))
    n_pairs = min(total // 2, base_pairs * bpp)
    pairs = [(2 * k, 2 * k + 1) for k in range(n_pairs)]
    return pairs, n_sub


def capacity_bits(block_shape: tuple[int, int], bpp: int) -> int:
    pairs, _ = _select_pairs(block_shape, bpp)
    return len(pairs)


def embed(block: np.ndarray, bpp: int, bits: np.ndarray) -> np.ndarray:
    pairs, n_sub = _select_pairs(block.shape, bpp)
    if len(bits) != len(pairs):
        raise ValueError(f"expected {len(pairs)} bits, got {len(bits)}")

    cA, cH, cV, cD = _subbands(block)
    combined = np.concatenate([cH.flatten(), cV.flatten()])
    combined = embed_bits_pairwise(combined, pairs, np.asarray(bits, dtype=np.uint8), DELTA)
    new_cH = combined[:n_sub].reshape(cH.shape)
    new_cV = combined[n_sub:].reshape(cV.shape)

    rec = pywt.idwt2((cA, (new_cH, new_cV, cD)), WAVELET)
    rec = rec[: block.shape[0], : block.shape[1]]
    return np.clip(np.round(rec), 0, 255).astype(np.uint8)


def extract(block: np.ndarray, bpp: int, num_bits: int | None = None) -> np.ndarray:
    pairs, _n_sub = _select_pairs(block.shape, bpp)
    cA, cH, cV, cD = _subbands(block)
    combined = np.concatenate([cH.flatten(), cV.flatten()])
    bits = extract_bits_pairwise(combined, pairs)
    return bits if num_bits is None else bits[:num_bits]
