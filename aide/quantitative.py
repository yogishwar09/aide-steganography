"""Quantitative LSB steganalysis: estimate the embedding *rate* (fraction
of pixels carrying message bits), not just a yes/no score.

Implements the two classical structural detectors the README previously
listed as missing, vectorised over the whole image:

* RS analysis - Fridrich, Goljan & Du, "Reliable detection of LSB
  steganography in color and grayscale images", ACM MM&Sec 2001 (the
  report's "Detectability (RS analysis)" metric).
* Sample Pairs Analysis (SPA) - Dumitrescu, Wu & Wang, "Detection of LSB
  steganography via sample pair analysis", IEEE TSP 2003.

Both return an estimated embedding rate p in [0, 1] (p ~ 0 for a clean
image). Because no reference implementation was available, correctness is
established empirically instead: `tests/test_quantitative.py` embeds known
LSB-replacement rates into real BOSSbase covers and checks the estimates
track the ground truth, and that clean covers estimate near zero.
"""
from __future__ import annotations

import numpy as np

# RS analysis discrimination mask over groups of 4 horizontally adjacent pixels.
RS_MASK = np.array([0, 1, 1, 0], dtype=np.int16)


def _flip_pos(x: np.ndarray) -> np.ndarray:
    """F1: 0<->1, 2<->3, ... (LSB flip)."""
    return x ^ 1


def _flip_neg(x: np.ndarray) -> np.ndarray:
    """F-1: -1<->0, 1<->2, 3<->4, ... i.e. F-1(x) = F1(x + 1) - 1."""
    return ((x + 1) ^ 1) - 1


def _discrimination(groups: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(groups, axis=1)).sum(axis=1)


def _rs_fractions(img: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Fractions of Regular and Singular groups under flipping mask `mask`
    (entries in {-1, 0, 1})."""
    h, w = img.shape
    n = len(mask)
    g = img[:, : (w // n) * n].astype(np.int16).reshape(-1, n)
    f0 = _discrimination(g)
    flipped = g.copy()
    for i, m in enumerate(mask):
        if m == 1:
            flipped[:, i] = _flip_pos(g[:, i])
        elif m == -1:
            flipped[:, i] = _flip_neg(g[:, i])
    f1 = _discrimination(flipped)
    total = len(g)
    return float((f1 > f0).sum() / total), float((f1 < f0).sum() / total)


def rs_analysis(gray: np.ndarray) -> dict:
    """Estimated LSB-replacement embedding rate via RS analysis.

    R_M, S_M (mask M) and R_-M, S_-M (mask -M) are measured on the image
    and on the image with every LSB flipped; in a clean image R_M ~ R_-M
    and S_M ~ S_-M, and LSB embedding pulls them apart along known curves.
    The rate follows from the smaller root z of
        2(d1 + d0) z^2 + (d_0 - d_1 - d1 - 3 d0) z + d0 - d_0 = 0,
    with d0 = R_M - S_M (image), d1 = R_M - S_M (LSB-flipped image),
    d_0 / d_1 the same for -M, and p = z / (z - 1/2).
    """
    img = gray.astype(np.int16)
    flipped = _flip_pos(img)
    r_m, s_m = _rs_fractions(img, RS_MASK)
    r_nm, s_nm = _rs_fractions(img, -RS_MASK)
    r_m1, s_m1 = _rs_fractions(flipped, RS_MASK)
    r_nm1, s_nm1 = _rs_fractions(flipped, -RS_MASK)

    d0, d1 = r_m - s_m, r_m1 - s_m1
    dn0, dn1 = r_nm - s_nm, r_nm1 - s_nm1
    a = 2.0 * (d1 + d0)
    b = dn0 - dn1 - d1 - 3.0 * d0
    c = d0 - dn0
    p = _smaller_root_rate(a, b, c, lambda z: z / (z - 0.5))
    return {"rate": p, "R_M": r_m, "S_M": s_m, "R_-M": r_nm, "S_-M": s_nm}


def _smaller_root_rate(a: float, b: float, c: float, to_rate) -> float:
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return 0.0
        roots = [-c / b]
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            disc = 0.0  # numerically tangent: take the double root
        sq = np.sqrt(disc)
        roots = [(-b + sq) / (2 * a), (-b - sq) / (2 * a)]
    z = min(roots, key=abs)
    try:
        p = float(to_rate(z))
    except ZeroDivisionError:
        return 1.0
    return float(np.clip(p, 0.0, 1.0)) if np.isfinite(p) else 1.0


def spa(gray: np.ndarray) -> dict:
    """Estimated LSB-replacement embedding rate via Sample Pairs Analysis.

    Over all horizontally and vertically adjacent pixel pairs (u, v):
        X = #{v even, u < v} + #{v odd, u > v}
        Y = #{v even, u > v} + #{v odd, u < v}
        K = #{floor(u/2) == floor(v/2)}  (pairs differing at most in the LSB)
    Natural images satisfy |X| ~ |Y|; LSB replacement at rate p breaks it
    predictably, giving 2K b^2 + 2(2X - N) b + (Y - X) = 0 with p = 2b
    (smaller root), N the number of pairs.
    """
    img = gray.astype(np.int32)
    us = np.concatenate([img[:, :-1].ravel(), img[:-1, :].ravel()])
    vs = np.concatenate([img[:, 1:].ravel(), img[1:, :].ravel()])
    v_even = (vs & 1) == 0
    x = int(np.sum((v_even & (us < vs)) | (~v_even & (us > vs))))
    y = int(np.sum((v_even & (us > vs)) | (~v_even & (us < vs))))
    k = int(np.sum((us >> 1) == (vs >> 1)))
    n = len(us)
    if k == 0:
        return {"rate": 0.0, "X": x, "Y": y, "K": k}
    rate = _smaller_root_rate(2.0 * k, 2.0 * (2 * x - n), float(y - x), lambda b: 2.0 * b)
    return {"rate": rate, "X": x, "Y": y, "K": k}
