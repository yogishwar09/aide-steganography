"""The trained steganalyser: feature extraction is well-formed, and a
detector trained by `aide train-detector` (if present) separates AIDE stego
from clean covers and guards against cover-source mismatch."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from aide import mlsteg, pipeline
from tests.conftest import MODEL_DIR


def test_feature_vector_shape_and_finiteness(real_sample_image_path):
    gray = np.array(Image.open(real_sample_image_path).convert("L"))
    f = mlsteg.extract_features(gray)
    assert f.ndim == 1 and f.size == 98 + len(mlsteg.GAP_BINS) - 1 + 3
    assert np.all(np.isfinite(f))


def test_spam_rows_are_conditional_distributions(real_sample_image_path):
    gray = np.array(Image.open(real_sample_image_path).convert("L"))
    spam = mlsteg.spam_features(gray).reshape(2, 7, 7)
    sums = spam.sum(axis=2)
    assert np.all((np.abs(sums - 1) < 1e-9) | (sums == 0))


def test_dct_gap_features_see_qim_embedding(real_sample_image_path, trained_model):
    """QIM forces |c_i - c_j| >= delta, so the smallest-difference bin must
    shrink after AIDE embedding - the fingerprint the detector learns."""
    gray = np.array(Image.open(real_sample_image_path).convert("L"))
    res = pipeline.embed(real_sample_image_path, b"x" * 200, b"k", trained_model, profile="robust")
    assert mlsteg.dct_gap_features(res.stego)[0] < mlsteg.dct_gap_features(gray)[0]


@pytest.fixture
def detector():
    if not (MODEL_DIR / mlsteg.DETECTOR_FILE).exists():
        pytest.skip("no trained detector; run `aide train-detector`")
    return mlsteg.Steganalyzer.load(MODEL_DIR)


def test_trained_detector_flags_aide_and_passes_clean(detector, real_sample_paths, trained_model):
    flagged_stego = flagged_clean = n = 0
    for p in real_sample_paths[-12:]:
        gray = np.array(Image.open(p).convert("L"))
        try:
            stego = pipeline.embed(p, b"y" * 150, b"k", trained_model).stego
        except pipeline.AideCapacityError:
            continue
        n += 1
        flagged_clean += detector.analyze(gray).verdict != "clean"
        flagged_stego += detector.analyze(stego).verdict != "clean"
    assert n >= 6
    assert flagged_stego >= n - 1 and flagged_clean <= 1


def test_out_of_distribution_input_is_capped(detector):
    """Cover-source mismatch guard: a synthetic pattern unlike any training
    photograph must never get the strongest ML verdict."""
    y, x = np.mgrid[0:512, 0:512]
    rng = np.random.default_rng(0)
    synth = np.clip(128 + 50 * np.sin(x / 9) * np.cos(y / 13) + 40 * (rng.random((512, 512)) - 0.5),
                    0, 255).astype(np.uint8)
    rep = detector.analyze(synth)
    assert not rep.in_distribution and rep.verdict != "likely_steganographic"
