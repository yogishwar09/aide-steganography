"""Attack-recovery integration tests.

These report *measured* behavior on a real BOSSbase cover, using a longer
(~470-byte) message that spans several Reed-Solomon codewords - the
default of aide/cli.py's `evaluate` command. Short messages are split into
shortened codewords automatically (aide/ecc.py's `plan`), so interleaving
protects them too; see tests/test_profiles_and_blind.py for those, and
`aide benchmark` for the multi-image recovery rates quoted in the README.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aide import attacks, pipeline

MULTI_CODEWORD_MESSAGE = (
    b"CLASSIFIED SITREP: Convoy Alpha departs staging area 0600Z, route Bravo-7. "
    b"Escort confirmed, rendezvous at checkpoint Delta. Air support standby channel 4. "
) * 3
SECRET = b"defence-shared-secret-2026"


@pytest.fixture
def high_capacity_sample(real_sample_paths: list[Path]) -> Path:
    """BOSSbase image '1022.pgm' has enough DCT/DWT capacity for the
    multi-codeword message above under the default fragile-method budgets
    (measured: 9352 bits of DCT+DWT capacity, the most of the first 30
    fetched samples - DCT's capacity was tightened considerably by the
    robust-frequency cap in aide/embedding/dct.py, so this matters more
    than it used to). Falls back to the first available sample if it
    isn't present, in which case tests may skip on AideCapacityError
    rather than fail outright."""
    for p in real_sample_paths:
        if p.name == "1022.pgm":
            return p
    return real_sample_paths[0]


@pytest.fixture
def embedded(high_capacity_sample, trained_model):
    try:
        return pipeline.embed(high_capacity_sample, MULTI_CODEWORD_MESSAGE, SECRET, trained_model)
    except pipeline.AideCapacityError:
        pytest.skip("sample image lacks enough robust capacity for the multi-codeword test message")


@pytest.mark.parametrize("quality", [50, 70])
def test_compression_recovers_message(embedded, quality):
    """Matches report O4's explicit target: JPEG Q=50."""
    attacked = attacks.compression(embedded.stego, quality=quality)
    rec = pipeline.recover(attacked, embedded.sidecar, SECRET)
    assert rec.success and rec.plaintext == MULTI_CODEWORD_MESSAGE


def test_splicing_recovers_via_tamper_localized_erasures(embedded):
    """The core interleaving + tamper-localization design goal: a
    localized region replacement should be recoverable because damaged
    blocks are (a) spread thinly across many codewords and (b) flagged as
    ECC erasures rather than left to blind correction.

    fraction=0.02 (not the report's 5-25% range) because at 3 codewords'
    worth of message, 5% splicing lands right at the ~32-erasure-per-
    codeword budget - close enough that whether it succeeds depends on
    the exact random region drawn (measured: succeeds for some seeds,
    fails for others). 2% leaves comfortable margin and passes reliably
    across seeds; see test_heavy_cropping_may_exceed_erasure_budget_but_is_detected
    below for the "detection still works even when correction can't keep
    up" case, and README for the longer discussion of this trade-off.
    """
    attacked = attacks.splicing(embedded.stego, fraction=0.02, seed=1)
    rec = pipeline.recover(attacked, embedded.sidecar, SECRET)
    assert rec.erasure_count > 0, "tamper localization should have flagged the spliced blocks"
    assert rec.success and rec.plaintext == MULTI_CODEWORD_MESSAGE


def test_mild_noise_recovers(embedded):
    attacked = attacks.gaussian_noise(embedded.stego, sigma=10, seed=1)
    rec = pipeline.recover(attacked, embedded.sidecar, SECRET)
    assert rec.success and rec.plaintext == MULTI_CODEWORD_MESSAGE


def test_median_filter_recovers(embedded):
    """Formerly an xfail sitting right at RS's t=16 boundary. Fixed by (a)
    routing payload away from LSB/DWT, which caused ~40% of median-filter
    bit errors on ~2% of the bits, (b) the profile's preferred bpp cap
    (fewer, lower-frequency DCT pairs), and (c) RS(255,191)-ratio parity -
    see aide/profiles.py. Measured 25/25 BOSSbase covers."""
    attacked = attacks.median_filter(embedded.stego, ksize=3)
    rec = pipeline.recover(attacked, embedded.sidecar, SECRET)
    assert rec.success and rec.plaintext == MULTI_CODEWORD_MESSAGE


SHORT_MESSAGE = MULTI_CODEWORD_MESSAGE[:156]


def test_heavy_scaling_recovers_with_default_profile(high_capacity_sample, trained_model):
    """0.5x down-and-up rescaling - "the most damaging" attack per report
    section 2.3, and a hard failure in AIDE 0.1. The profile's preferred
    bpp cap keeps payload in the low mid-frequency DCT pairs that survive
    resampling (measured 24/25 BOSSbase covers at the balanced profile with
    a 156-byte message). Capacity-dependent: a message too long for the
    cap forces it to relax (see pipeline.embed), and 0.5x survival at the
    balanced profile is then lost - use "robust" for long messages (below)."""
    res = pipeline.embed(high_capacity_sample, SHORT_MESSAGE, SECRET, trained_model)
    attacked = attacks.scaling(res.stego, factor=0.5)
    rec = pipeline.recover(attacked, res.sidecar, SECRET)
    assert rec.success and rec.plaintext == SHORT_MESSAGE


def test_heavy_scaling_recovers_long_message_with_robust_profile(embedded_robust):
    attacked = attacks.scaling(embedded_robust.stego, factor=0.5)
    rec = pipeline.recover(attacked, embedded_robust.sidecar, SECRET)
    assert rec.success and rec.plaintext == MULTI_CODEWORD_MESSAGE


@pytest.fixture
def embedded_robust(high_capacity_sample, trained_model):
    try:
        return pipeline.embed(high_capacity_sample, MULTI_CODEWORD_MESSAGE, SECRET, trained_model,
                              profile="robust")
    except pipeline.AideCapacityError:
        pytest.skip("sample image lacks capacity for the robust profile")


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_robust_profile_meets_o4_gaussian_noise_sigma_20(embedded_robust, seed):
    """Report O4's explicit noise target, end-to-end. Unreachable in AIDE
    0.1 at any single setting; met by the robust profile's larger DCT delta
    + RS parity + reliability-guided (GMD) decoding."""
    attacked = attacks.gaussian_noise(embedded_robust.stego, sigma=20, seed=seed)
    rec = pipeline.recover(attacked, embedded_robust.sidecar, SECRET)
    assert rec.success and rec.plaintext == MULTI_CODEWORD_MESSAGE


def test_robust_profile_survives_salt_and_pepper(embedded_robust):
    attacked = attacks.salt_and_pepper(embedded_robust.stego, amount=0.01, seed=1)
    rec = pipeline.recover(attacked, embedded_robust.sidecar, SECRET)
    assert rec.success and rec.plaintext == MULTI_CODEWORD_MESSAGE


def test_mild_scaling_recovers(embedded):
    """factor=0.75 is a much milder resize than the report's own
    'most damaging' 0.5x case above, and recovers reliably."""
    attacked = attacks.scaling(embedded.stego, factor=0.75)
    rec = pipeline.recover(attacked, embedded.sidecar, SECRET)
    assert rec.success and rec.plaintext == MULTI_CODEWORD_MESSAGE


def test_heavy_cropping_may_exceed_erasure_budget_but_is_detected(embedded):
    """A 15% crop typically damages more blocks than a single message's
    worth of RS codewords can correct even with perfect erasure marking
    (measured: often 150+ damaged blocks vs. a 32-byte-per-codeword
    erasure budget) - but tamper localization must still *detect* it."""
    attacked = attacks.cropping(embedded.stego, fraction=0.15, seed=1)
    rec = pipeline.recover(attacked, embedded.sidecar, SECRET)
    assert rec.erasure_count > 0  # detection works even when correction can't keep up
