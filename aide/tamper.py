"""Tamper localization (report 2.3, recovery sub-pipeline step 1).

Two signals combine into a per-block confidence score and a binary erasure
mask used by ``aide.ecc``'s erasure-aware Reed-Solomon decoding:

1. Exact per-block SHA-256 hash verification (primary, strict) - any single
   changed bit fails this, so on its own it can't distinguish "mild JPEG
   requantization nudged every block a little" from "this block was
   spliced out and replaced."
2. A block-level histogram chi-square goodness-of-fit test against a small
   reference summary carried in the sidecar (secondary/confirmatory,
   graded) - this is what tells those two cases apart: a genuinely
   replaced/cropped block's histogram looks nothing like the reference,
   while mild global degradation only nudges it slightly.

Only blocks that fail *both* checks become ECC erasures; blocks that fail
only the strict hash check are left to Reed-Solomon's blind error
correction, which is the right tool for "every codeword has a few
scattered bit errors" as opposed to "this run of bytes is simply gone."

Note: this histogram test is a reference-vs-received goodness-of-fit check
used to confirm localized tampering. It is not the classical "pairs of
values" chi-square *steganalysis* attack from the literature (which
detects the mere presence of LSB steganography) - same statistic, a
different purpose.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

N_HIST_BINS = 8
# Calibrated against moderate diffuse noise (report O4: Gaussian sigma<=20)
# vs. full-block splicing on real BOSSbase blocks: at this threshold most
# diffuse-noise blocks are left to Reed-Solomon's blind error correction
# (avoids exhausting the 32-erasure/codeword budget on merely-noisy
# blocks), while genuinely replaced/spliced blocks are still reliably
# caught. See aide/tamper.py's module docstring; separation is not perfect
# at high noise levels (sigma>~25) or very subtle splices - a documented
# limitation of this simple per-block statistic.
DEFAULT_CHI_SQUARE_THRESHOLD = 800.0


def block_hash(block: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(block).tobytes()).hexdigest()


def block_histogram(block: np.ndarray, n_bins: int = N_HIST_BINS) -> list[float]:
    """Laplace-smoothed (add-1) normalised histogram: without smoothing, a
    bin with zero expected count makes the chi-square statistic blow up to
    an uninformative extreme from a single stray pixel landing in it."""
    hist, _ = np.histogram(block, bins=n_bins, range=(0, 255))
    hist = hist.astype(np.float64) + 1.0
    return (hist / hist.sum()).tolist()


@dataclass
class BlockReference:
    block_index: int
    sha256: str
    histogram: list[float]


def build_references(gray: np.ndarray, blocks) -> list[BlockReference]:
    """Computed at embed time on the final *stego* image (post-embedding)
    so recovery-time verification compares against what the receiver
    should actually see if nothing was tampered with in transit."""
    refs = []
    for b in blocks:
        ys, xs = b.slice()
        block = gray[ys, xs]
        refs.append(BlockReference(b.index, block_hash(block), block_histogram(block)))
    return refs


def _chi_square(observed_freq: np.ndarray, expected_freq: np.ndarray, n_pixels: int) -> float:
    observed = observed_freq * n_pixels
    expected = expected_freq * n_pixels
    eps = 1e-6
    return float(np.sum((observed - expected) ** 2 / (expected + eps)))


@dataclass
class TamperReport:
    hash_mismatch: np.ndarray  # bool per block: strict hash check failed
    chi_square: np.ndarray     # float per block: histogram deviation statistic
    confidence: np.ndarray     # float in [0, 1] per block: 1 = fully trusted
    erasure_mask: np.ndarray   # bool per block: treat as ECC erasure


def analyze(gray: np.ndarray, blocks, references: list[BlockReference],
            chi_square_threshold: float = DEFAULT_CHI_SQUARE_THRESHOLD) -> TamperReport:
    n = len(blocks)
    hash_mismatch = np.zeros(n, dtype=bool)
    chi_sq = np.zeros(n, dtype=np.float64)
    confidence = np.ones(n, dtype=np.float64)

    for ref in references:
        b = blocks[ref.block_index]
        ys, xs = b.slice()
        block = gray[ys, xs]
        matches = block_hash(block) == ref.sha256
        hash_mismatch[b.index] = not matches
        if matches:
            continue
        observed = np.array(block_histogram(block), dtype=np.float64)
        expected = np.array(ref.histogram, dtype=np.float64)
        stat = _chi_square(observed, expected, block.size)
        chi_sq[b.index] = stat
        confidence[b.index] = float(np.clip(1.0 - stat / (2 * chi_square_threshold), 0.0, 1.0))

    erasure_mask = hash_mismatch & (chi_sq >= chi_square_threshold)
    return TamperReport(hash_mismatch=hash_mismatch, chi_square=chi_sq,
                         confidence=confidence, erasure_mask=erasure_mask)
