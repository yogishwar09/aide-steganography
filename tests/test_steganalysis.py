"""Validates aide.steganalysis against ground truth: clean images (no
false-positive stampede), naive full-image LSB embedding at known rates
(does detection actually work), and AIDE's own adaptive stego output
(does AIDE evade this detector, as the report itself claims it should).

These thresholds and rates are the ones actually measured while building
the module - see aide/steganalysis.py's module and analyze() docstrings
for the full narrative, including a false-positive regression that was
caught and reverted here (an earlier design change amplified 1/40 clean
images being falsely flagged into 16/20).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from aide import pipeline, steganalysis


def _naive_lsb_embed(gray: np.ndarray, rate: float, seed: int = 0) -> np.ndarray:
    """Simulate an unsophisticated attacker: sequential LSB replacement
    with a random (i.e. encrypted-payload-like) bitstream over the first
    `rate` fraction of pixels, in raster order - the classical target of
    the chi-square pairs-of-values attack."""
    rng = np.random.default_rng(seed)
    stego = gray.copy()
    flat = stego.reshape(-1)
    n_pixels = int(flat.size * rate)
    bits = rng.integers(0, 2, size=n_pixels).astype(np.uint8)
    flat[:n_pixels] = (flat[:n_pixels] & 0xFE) | bits
    return stego


def test_clean_images_rarely_flagged(real_sample_paths: list[Path]):
    """A detector that cries wolf on most clean images is worse than no
    detector for a security use case. Measured false-positive rate on 40
    real BOSSbase images: 1/40 (2.5%) - consistent with the chi-square
    test's own significance level, not a systemic bias."""
    from PIL import Image

    false_positives = 0
    n = min(40, len(real_sample_paths))
    for p in real_sample_paths[:n]:
        gray = np.array(Image.open(p).convert("L"))
        report = steganalysis.analyze(gray)
        if report.verdict != "clean":
            false_positives += 1

    assert false_positives / n <= 0.10, (
        f"false positive rate {false_positives}/{n} exceeds the measured-acceptable bound"
    )


def test_full_image_naive_lsb_embedding_is_reliably_detected(real_sample_image_path):
    from PIL import Image

    gray = np.array(Image.open(real_sample_image_path).convert("L"))
    stego = _naive_lsb_embed(gray, rate=1.0)
    report = steganalysis.analyze(stego)

    assert report.chi_square_p > 0.9
    assert report.verdict == "likely_steganographic"


def test_detection_rate_increases_with_embedding_rate(real_sample_paths: list[Path]):
    """Detection power should scale with how much of the image carries
    payload - low rates are hard (matches the classical test's known
    limitations), full-image embedding is reliably caught."""
    from PIL import Image

    paths = real_sample_paths[:20]

    def detection_rate(rate: float) -> float:
        flagged = 0
        for p in paths:
            gray = np.array(Image.open(p).convert("L"))
            stego = _naive_lsb_embed(gray, rate=rate)
            if steganalysis.analyze(stego).verdict != "clean":
                flagged += 1
        return flagged / len(paths)

    low_rate_detection = detection_rate(0.1)
    full_rate_detection = detection_rate(1.0)

    assert full_rate_detection >= 0.9, "full-image LSB embedding should be reliably caught"
    assert full_rate_detection > low_rate_detection, (
        "detection should improve as more of the image is affected"
    )


def test_aide_own_stego_images_evade_this_classical_detector(real_sample_paths, trained_model):
    """The practical payoff: does AIDE's design actually deliver on the
    report's own claim of negative detectability? Measured: yes, for this
    classical chi-square detector - AIDE caps LSB at <1% of payload
    (too little to shift the whole-image pair-equalization statistic) and
    carries the rest in DCT/DWT coefficients, which this spatial-LSB-only
    detector cannot see at all. This does NOT mean AIDE is undetectable in
    general (see aide/steganalysis.py's docstring on SPA/RS analysis,
    not implemented here, and DCT/DWT-domain steganalysis, not attempted
    at all) - only that it evades *this specific, classical* tool, which
    is exactly the kind of measured, scoped claim this project tries to
    make rather than a blanket one.
    """
    secret = b"defence-shared-secret-2026"
    message = (
        b"CLASSIFIED SITREP: Convoy Alpha departs staging area 0600Z, route Bravo-7. "
        b"Escort confirmed, rendezvous at checkpoint Delta. Air support standby channel 4. "
    ) * 3

    flagged = 0
    tested = 0
    for p in real_sample_paths[:15]:
        try:
            result = pipeline.embed(p, message, secret, trained_model)
        except pipeline.AideCapacityError:
            continue
        report = steganalysis.analyze(result.stego)
        tested += 1
        if report.verdict != "clean":
            flagged += 1

    assert tested >= 5, "not enough images had capacity for the test message"
    assert flagged == 0, (
        f"{flagged}/{tested} AIDE stego images were flagged - AIDE's own LSB/DWT/DCT "
        f"budget split should keep it under this detector's radar"
    )


def test_spatial_suspicion_grid_shape_and_range(real_sample_image_path):
    from PIL import Image

    gray = np.array(Image.open(real_sample_image_path).convert("L"))
    grid = steganalysis.spatial_suspicion_grid(gray, grid_size=4)

    assert grid.shape == (4, 4)
    assert np.all((grid >= 0.0) & (grid <= 1.0))


def test_spatial_suspicion_grid_lights_up_the_embedded_region(real_sample_image_path):
    """A genuinely embedded region should measurably raise its own grid
    cells relative to their own before/after baseline. Does NOT assert
    those cells end up higher than every other cell in the image - see
    aide/steganalysis.py's module docstring: some clean regions
    legitimately produce high chi-square values too (a real, documented
    multiple-testing limitation), so only a same-region before/after
    comparison is a sound thing to assert here."""
    from PIL import Image

    gray = np.array(Image.open(real_sample_image_path).convert("L"))
    clean_grid = steganalysis.spatial_suspicion_grid(gray, grid_size=4)

    rng = np.random.default_rng(0)
    stego = gray.copy()
    h, w = gray.shape
    for r in range(0, h // 2):
        row = stego[r, :w // 2]
        bits = rng.integers(0, 2, size=row.size).astype(np.uint8)
        stego[r, :w // 2] = (row & 0xFE) | bits
    stego_grid = steganalysis.spatial_suspicion_grid(stego, grid_size=4)

    # top-left quadrant cells (rows 0-1, cols 0-1 of a 4x4 grid) should rise
    top_left_delta = (stego_grid[:2, :2] - clean_grid[:2, :2]).mean()
    assert top_left_delta > 0.3, "embedding a region should raise its own grid cells' suspicion"


def test_spatial_suspicion_grid_overlay_renders(real_sample_image_path):
    from PIL import Image

    from aide import visualize

    gray = np.array(Image.open(real_sample_image_path).convert("L"))
    report = steganalysis.analyze(gray)
    overlay = visualize.suspicion_heatmap_overlay(gray, np.array(report.spatial_grid))

    assert overlay.size == (gray.shape[1], gray.shape[0])
    assert overlay.mode == "RGB"
