"""Shared helpers for the transform-domain embedding methods (DCT, DWT).

Both use the same coefficient-pair relative-magnitude technique: to embed a
1 bit, force coeff[i] > coeff[j] by at least `delta`; to embed a 0 bit,
force coeff[i] < coeff[j] by at least `delta`. This is a standard
QIM-style, more-robust-than-raw-LSB approach for transform coefficients -
small enough shifts to stay imperceptible after the inverse transform, but
resilient to the mild requantization/rounding that JPEG compression and
similar attacks introduce.
"""
from __future__ import annotations

import numpy as np


def embed_bits_pairwise(coeffs: np.ndarray, pairs: list[tuple[int, int]],
                         bits: np.ndarray, delta: float) -> np.ndarray:
    coeffs = coeffs.copy()
    for (i, j), bit in zip(pairs, bits):
        a, b = coeffs[i], coeffs[j]
        mid = (a + b) / 2.0
        if bit:
            if a <= b + delta:
                coeffs[i] = mid + delta / 2
                coeffs[j] = mid - delta / 2
        else:
            if a >= b - delta:
                coeffs[i] = mid - delta / 2
                coeffs[j] = mid + delta / 2
    return coeffs


def extract_bits_pairwise(coeffs: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    bits = np.zeros(len(pairs), dtype=np.uint8)
    for k, (i, j) in enumerate(pairs):
        bits[k] = 1 if coeffs[i] > coeffs[j] else 0
    return bits


def zigzag_indices(n: int) -> list[int]:
    """Flat (row-major) indices of an n x n grid visited in standard JPEG
    zigzag (diagonal) order, low frequency first."""
    diagonals: list[list[tuple[int, int]]] = [[] for _ in range(2 * n - 1)]
    for i in range(n):
        for j in range(n):
            s = i + j
            if s % 2 == 0:
                diagonals[s].insert(0, (i, j))
            else:
                diagonals[s].append((i, j))
    order = [ij for diag in diagonals for ij in diag]
    return [r * n + c for r, c in order]
