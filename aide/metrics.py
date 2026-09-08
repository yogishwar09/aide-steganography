"""Evaluation metrics (report section 6.1): PSNR, SSIM, BER, recovery rate,
execution time, and memory footprint."""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import psutil
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(original: np.ndarray, other: np.ndarray) -> float:
    if np.array_equal(original, other):
        return float("inf")
    return float(peak_signal_noise_ratio(original, other, data_range=255))


def ssim(original: np.ndarray, other: np.ndarray) -> float:
    return float(structural_similarity(original, other, data_range=255))


def bit_error_rate(original: bytes, recovered: bytes) -> float:
    """Fraction of mismatched bits, comparing over the longer message's
    length (missing/extra bytes at the end count as fully wrong bits)."""
    n = max(len(original), len(recovered))
    if n == 0:
        return 0.0
    a = np.frombuffer(original.ljust(n, b"\x00"), dtype=np.uint8)
    b = np.frombuffer(recovered.ljust(n, b"\x00"), dtype=np.uint8)
    xor = np.bitwise_xor(a, b)
    mismatched_bits = int(np.unpackbits(xor).sum())
    return mismatched_bits / (n * 8)


def recovery_success(original: bytes, recovered: bytes | None) -> bool:
    return recovered is not None and recovered == original


@dataclass
class Timing:
    elapsed_seconds: float
    rss_delta_mb: float


@contextmanager
def measure():
    """Usage: with measure() as t: ...  then t.elapsed_seconds / t.rss_delta_mb"""
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    t0 = time.perf_counter()
    result = Timing(0.0, 0.0)
    try:
        yield result
    finally:
        result.elapsed_seconds = time.perf_counter() - t0
        result.rss_delta_mb = (process.memory_info().rss - rss_before) / (1024 * 1024)
