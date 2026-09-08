"""Blind steganalysis: detecting hidden data WITHOUT the sidecar, secret,
or knowledge of embedding method - the defence/intelligence use case from
the report (section 7: "Border & port security... scan intercepted
digital media", "Cyber defence command centres... monitoring for
embedded covert data exfiltration"). This is the detection/counter side
of the same coin as `aide.pipeline`'s embedding: given an arbitrary
intercepted image, is someone smuggling data in it?

This is a genuinely different, harder problem than `aide.tamper`'s
tamper localization, which compares a received image against a *known
reference* (the sidecar built at embed time). Here there is no reference
- only the image itself and general statistical knowledge of what
"natural" photographic content looks like versus what embedding an
encrypted (i.e. high-entropy, near-random) payload does to it.

Techniques implemented, in order of how much statistical power they have:

1. Chi-square "pairs of values" attack (Westfeld & Pfitzmann, 2001) - the
   classical LSB-steganography detector, and the one the report itself
   names as a target metric ("Detectability (RS analysis): Negative").
   LSB replacement with a near-random payload pushes the frequency counts
   of value pairs (2k, 2k+1) toward equality; natural images usually do
   NOT have such pairs equalized. A large sample (the whole image, or a
   sizeable window/row-band of it) is needed for the test to have power -
   this does not work on tiny (e.g. 16x16 AIDE-block-sized) windows, so
   the spatial "where in the image" profile here uses row-bands, not the
   embedding grid.
2. LSB/bit-plane-1 correlation - a natural image's least-significant bit
   plane is weak but non-random noise correlated with the image's finer
   structure (sensor noise, smooth gradients); overwriting it with an
   encrypted payload's near-uniform-random bits drives that correlation
   toward zero. A weaker, supplementary signal, useful as a second
   opinion when chi-square's sample-size requirements aren't met.
3. LSB bit-balance - the crudest, weakest signal (deviation of the LSB
   plane's 1-fraction from 0.5) - many natural images are already close
   to balanced, so this alone is not reliable; included only as a third,
   low-weight vote in the combined score, matching how the literature
   treats it.

Not implemented: Sample Pairs Analysis and RS (Regular-Singular) rate
estimation, two more sophisticated blind detectors mentioned in the
literature (and RS analysis by name in the report). Both were considered
but deliberately left out rather than shipped with an unverified,
possibly-subtly-wrong implementation - this project's whole approach is
to measure and validate rather than assume, and getting either of these
right from memory without a reference implementation to check against
was assessed as too high a correctness risk for a detection tool. Real
future work, not a placeholder.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2 as _chi2_dist

MIN_EXPECTED_COUNT = 4  # standard practice: skip histogram bins too sparse to be meaningful


def chi_square_pov(gray: np.ndarray) -> dict:
    """Chi-square pairs-of-values statistic and its p-value over the given
    pixel window. p close to 1 => the value-pair histogram fits the
    "embedded" (equalized-pairs) hypothesis well = suspicious. p close to
    0 => fits a typical natural, non-equalized histogram = not suspicious
    by this test. Returns dof=0 (p=0.0) if the sample is too small/sparse
    for the test to be meaningful."""
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    chi2_stat = 0.0
    dof = 0
    for k in range(128):
        h0, h1 = float(hist[2 * k]), float(hist[2 * k + 1])
        expected = (h0 + h1) / 2.0
        if expected < MIN_EXPECTED_COUNT:
            continue
        chi2_stat += (h0 - expected) ** 2 / expected
        dof += 1
    if dof < 2:
        return {"p": 0.0, "chi2": 0.0, "dof": dof}
    p = float(1.0 - _chi2_dist.cdf(chi2_stat, dof - 1))
    return {"p": p, "chi2": float(chi2_stat), "dof": dof}


def lsb_correlation(gray: np.ndarray) -> float:
    """Correlation between the LSB plane (bit 0) and the next bit plane
    (bit 1). Natural images: weak positive correlation. Overwritten LSBs
    (random-looking payload): correlation collapses toward 0."""
    bit0 = (gray & 1).astype(np.float64).ravel()
    bit1 = ((gray >> 1) & 1).astype(np.float64).ravel()
    if bit0.std() == 0 or bit1.std() == 0:
        return 0.0
    return float(np.corrcoef(bit0, bit1)[0, 1])


def lsb_bit_balance(gray: np.ndarray) -> float:
    """abs(fraction of 1-bits in the LSB plane - 0.5). Smaller = more
    suspicious (closer to a coin-flip-random payload) but this alone is a
    weak signal - many natural images are already fairly balanced."""
    ones_fraction = float((gray & 1).mean())
    return abs(ones_fraction - 0.5)


@dataclass
class SteganalysisReport:
    chi_square_p: float          # whole-image chi-square p-value (primary signal)
    lsb_correlation: float       # bit0/bit1 correlation (secondary signal)
    lsb_bit_balance: float       # deviation from 0.5 (tertiary/weak signal)
    row_band_profile: list[float]  # chi-square p per horizontal row-band, for a spatial plot
    suspicion_score: float       # combined [0, 1], higher = more likely to carry hidden data
    verdict: str                 # "clean", "suspicious", or "likely_steganographic"


# Verdict thresholds and per-signal weights are calibrated in
# tests/test_steganalysis.py against known clean images and known,
# actually-embedded (naive sequential LSB) images - not asserted from
# theory alone.
SUSPICIOUS_THRESHOLD = 0.35
LIKELY_THRESHOLD = 0.65


def analyze(gray: np.ndarray, n_row_bands: int = 8) -> SteganalysisReport:
    """Full blind analysis of a single grayscale plane (use
    `aide.pipeline.load_image(path).y` for a color image - see that
    module's docstring on why this pipeline is luma-plane-only)."""
    chi = chi_square_pov(gray)
    corr = lsb_correlation(gray)
    balance = lsb_bit_balance(gray)

    h = gray.shape[0]
    band_size = max(1, h // n_row_bands)
    row_band_profile = []
    for i in range(0, h, band_size):
        band = gray[i:i + band_size]
        if band.size < 256 * MIN_EXPECTED_COUNT:  # too few pixels for a meaningful test
            row_band_profile.append(0.0)
            continue
        row_band_profile.append(chi_square_pov(band)["p"])

    # Combined score: the whole-image chi-square is the primary,
    # statistically well-powered signal. `row_band_profile` is exposed for
    # DISPLAY/diagnostic purposes only (a "where might it be" plot for an
    # analyst once the whole-image test is already positive) and
    # deliberately does NOT feed the verdict: an early version of this
    # function took max(row_band_profile) as a signal to catch partial-
    # image embedding, since a whole-image test can genuinely dilute a
    # partially-embedded region's signal (measured: a real 50%-of-image
    # LSB embed showed global p<0.01 while its embedded row-bands showed
    # p>0.9) - but taking the max of 8 per-band tests without a multiple-
    # testing correction is itself statistically unsound and was measured
    # to push 16/20 genuinely CLEAN images into a false "likely
    # steganographic" verdict (each band is individually low-powered, and
    # max-of-8 amplifies its false-positive rate roughly 8x). A tool that
    # cries wolf on most clean images is worse than not having it for a
    # security use case, so this was reverted rather than shipped -
    # detecting partial/localized embedding reliably is real, unresolved
    # future work (a proper multiple-testing correction, e.g. Bonferroni
    # or Benjamini-Hochberg, applied to the row-band profile, is the
    # obvious next step - not implemented here). See
    # tests/test_steganalysis.py for both the false-positive-rate
    # validation and the resulting partial-embedding-detection gap this
    # leaves, tested and documented rather than hidden.
    #
    # Correlation/balance are corroborating, lower-weight signals -
    # measured to have real dataset-dependent variance among genuinely
    # clean images (some real photos naturally sit near zero LSB/bit1
    # correlation), so they are weighted low enough that neither can push
    # a clean image over the "suspicious" line on its own.
    corr_signal = max(0.0, 1.0 - abs(corr) / 0.15)  # normalized: corr>=0.15 -> 0 suspicion
    balance_signal = max(0.0, 1.0 - balance / 0.05)  # normalized: balance>=0.05 -> 0 suspicion
    suspicion = 0.8 * chi["p"] + 0.13 * corr_signal + 0.07 * balance_signal

    if suspicion >= LIKELY_THRESHOLD:
        verdict = "likely_steganographic"
    elif suspicion >= SUSPICIOUS_THRESHOLD:
        verdict = "suspicious"
    else:
        verdict = "clean"

    return SteganalysisReport(
        chi_square_p=chi["p"], lsb_correlation=corr, lsb_bit_balance=balance,
        row_band_profile=row_band_profile, suspicion_score=float(suspicion), verdict=verdict,
    )
