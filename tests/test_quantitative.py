"""Ground-truth validation of RS analysis and Sample Pairs Analysis
(aide/quantitative.py): with no reference implementation to diff against,
correctness is established by embedding *known* LSB-replacement rates into
real covers and checking the estimators recover them."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from aide import steganalysis
from aide.quantitative import rs_analysis, spa


def _random_lsb(gray: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    flat = gray.ravel().copy()
    n = int(rate * flat.size)
    idx = rng.choice(flat.size, n, replace=False)
    flat[idx] = (flat[idx] & 0xFE) | rng.integers(0, 2, n, dtype=np.uint8)
    return flat.reshape(gray.shape)


@pytest.fixture
def covers(real_sample_paths):
    return [np.array(Image.open(p).convert("L")) for p in real_sample_paths[:12]]


@pytest.mark.parametrize("rate", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("estimator", [rs_analysis, spa])
def test_estimators_track_true_embedding_rate(covers, rate, estimator):
    """Measured on 20 BOSSbase covers: mean estimate within ~0.03 of the
    true rate for both estimators across 0.25-0.75."""
    rng = np.random.default_rng(0)
    est = [estimator(_random_lsb(g, rate, rng))["rate"] for g in covers]
    assert abs(float(np.median(est)) - rate) < 0.06


@pytest.mark.parametrize("estimator", [rs_analysis, spa])
def test_clean_covers_estimate_near_zero(covers, estimator):
    est = [estimator(g)["rate"] for g in covers]
    assert float(np.median(est)) < 0.05


def test_spa_handles_full_rate_embedding(covers):
    """RS analysis is known to be unstable as p -> 1 (its quadratic
    degenerates); SPA is not, which is one reason SPA is the primary
    structural signal in aide.steganalysis."""
    rng = np.random.default_rng(1)
    est = [spa(_random_lsb(g, 1.0, rng))["rate"] for g in covers]
    assert float(np.median(est)) > 0.9


def test_scattered_low_rate_lsb_is_detected_where_chi_square_alone_misses(covers):
    """The capability chi-square lacks: payload scattered at random over
    10% of pixels (not sequential). Calibrated SPA threshold catches it."""
    rng = np.random.default_rng(2)
    flagged = sum(steganalysis.analyze(_random_lsb(g, 0.10, rng)).verdict != "clean" for g in covers)
    assert flagged >= len(covers) - 1


def test_estimated_payload_reported_when_flagged(covers):
    rng = np.random.default_rng(3)
    rep = steganalysis.analyze(_random_lsb(covers[0], 0.5, rng))
    assert rep.verdict == "likely_steganographic"
    expected = 0.5 * covers[0].size / 8
    assert 0.7 * expected < rep.estimated_payload_bytes < 1.3 * expected
