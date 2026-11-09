"""Trained (machine-learning) steganalysis - the only kind of detector that
can see transform-domain embedding such as AIDE's own.

The classical detectors in aide.steganalysis / aide.quantitative model
what LSB replacement does to pixel statistics; AIDE's payload lives in DCT
coefficient-pair *relations*, which those tests are blind to by design.
Modern steganalysis instead learns the difference between cover and stego
feature distributions from examples. This module does that with two
compact feature families:

* SPAM (Pevny, Bas & Fridrich, "Steganalysis by subtractive pixel
  adjacency matrix", IEEE TIFS 2010), first-order, T=3: Markov transition
  probabilities of neighbouring-pixel differences, averaged into a
  straight (horizontal/vertical) and a diagonal group - 98 dims. Generic:
  any embedding that perturbs local pixel dependencies moves it.
* DCT pair-gap histograms (targeted at coefficient-pair QIM, i.e. AIDE's
  DCT method): for every block on the AIDE grid, the distribution of
  |c_i - c_j| over the zigzag mid-band pairs AIDE can use. QIM embedding
  forces |c_i - c_j| >= delta, so stego images show a deficit of small
  differences that natural images don't have - 32 dims + 3 summary ratios.

A gradient-boosted tree classifier is trained on clean vs AIDE-stego
covers (`aide train-detector`), using held-out covers for the reported
accuracy/AUC. That number is the honest answer to "is AIDE detectable?" -
by a detector that knows what to look for, it should be expected to be,
and the report's "negative detectability" claim only holds against the
classical blind tests (see README).
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from scipy.fft import dctn

from .embedding._common import zigzag_indices
from .features import choose_block_size

SPAM_T = 3
GAP_BINS = np.concatenate([np.arange(0, 64, 4), np.arange(64, 192, 16), [256, 1e9]])
DETECTOR_FILE = "steganalyzer.joblib"


def _spam_direction(diff_a: np.ndarray, diff_b: np.ndarray) -> np.ndarray:
    """P(D_next = b | D_cur = a) for clipped differences, as a (2T+1)^2 vector."""
    k = 2 * SPAM_T + 1
    a = np.clip(diff_a, -SPAM_T, SPAM_T).ravel() + SPAM_T
    b = np.clip(diff_b, -SPAM_T, SPAM_T).ravel() + SPAM_T
    joint = np.bincount(a * k + b, minlength=k * k).reshape(k, k).astype(np.float64)
    rows = joint.sum(axis=1, keepdims=True)
    return np.divide(joint, rows, out=np.zeros_like(joint), where=rows > 0).ravel()


def spam_features(gray: np.ndarray) -> np.ndarray:
    x = gray.astype(np.int32)
    straight, diag = [], []
    for img in (x, x[:, ::-1], x.T, x.T[:, ::-1]):  # ->, <-, down, up
        d = img[:, :-1] - img[:, 1:]
        straight.append(_spam_direction(d[:, :-1], d[:, 1:]))
    for img in (x, x[::-1], x[:, ::-1], x[::-1, ::-1]):  # 4 diagonals
        d = img[:-1, :-1] - img[1:, 1:]
        diag.append(_spam_direction(d[:-1, :-1], d[1:, 1:]))
    return np.concatenate([np.mean(straight, axis=0), np.mean(diag, axis=0)])


def _block_dct(gray: np.ndarray, bs: int) -> np.ndarray:
    h, w = gray.shape
    nh, nw = h // bs, w // bs
    blocks = gray[: nh * bs, : nw * bs].astype(np.float64).reshape(nh, bs, nw, bs).swapaxes(1, 2)
    return dctn(blocks.reshape(-1, bs, bs), axes=(1, 2), norm="ortho").reshape(-1, bs * bs)


def dct_gap_features(gray: np.ndarray) -> np.ndarray:
    bs = choose_block_size(*gray.shape)
    n = bs * bs
    zz = np.array(zigzag_indices(bs))
    mid = zz[max(1, n // 16): n - max(1, n // 8)]
    n_pairs = min(len(mid) // 2, max(1, n // 16))
    i_idx, j_idx = mid[0:2 * n_pairs:2], mid[1:2 * n_pairs:2]
    coeffs = _block_dct(gray, bs)
    d = np.abs(coeffs[:, i_idx] - coeffs[:, j_idx]).ravel()
    hist, _ = np.histogram(d, bins=GAP_BINS)
    hist = hist / max(1, d.size)
    small = (d < 8).mean()
    ratios = [small / max(1e-9, (d < 32).mean()), small / max(1e-9, (d < 64).mean()),
              float(np.median(d)) / 64.0]
    return np.concatenate([hist, ratios])


def extract_features(gray: np.ndarray) -> np.ndarray:
    return np.concatenate([spam_features(gray), dct_gap_features(gray)])


@dataclass
class DetectorReport:
    probability: float  # P(stego) from the trained classifier
    verdict: str        # "clean" | "suspicious" | "likely_steganographic"
    in_distribution: bool = True  # False -> cover-source mismatch, verdict capped
    note: str | None = None


class Steganalyzer:
    """Trained detector plus a cover-source-mismatch guard.

    A classifier trained on one cover source (here: BOSSbase photographs)
    is unreliable on images unlike anything it saw - the well-known
    "cover-source mismatch" problem in steganalysis. Measured: a synthetic
    sine-plus-noise test pattern with nothing embedded scored P(stego) =
    0.99. So an IsolationForest fitted on the training features flags
    out-of-distribution inputs, and for those the ML verdict is capped at
    "suspicious" with an explanatory note instead of asserting "likely
    steganographic" on evidence the model can't vouch for."""

    def __init__(self, clf, threshold: float = 0.5, meta: dict | None = None,
                 ood=None, ood_threshold: float | None = None):
        self.clf = clf
        self.threshold = threshold
        self.meta = meta or {}
        self.ood = ood
        self.ood_threshold = ood_threshold

    def predict_proba(self, gray: np.ndarray) -> float:
        return float(self.clf.predict_proba(extract_features(gray)[None, :])[0, 1])

    def analyze(self, gray: np.ndarray) -> DetectorReport:
        feats = extract_features(gray)[None, :]
        p = float(self.clf.predict_proba(feats)[0, 1])
        verdict = ("likely_steganographic" if p >= max(0.85, self.threshold)
                   else "suspicious" if p >= self.threshold else "clean")
        in_dist, note = True, None
        if self.ood is not None and self.ood_threshold is not None:
            in_dist = bool(self.ood.score_samples(feats)[0] >= self.ood_threshold)
            if not in_dist and verdict == "likely_steganographic":
                verdict = "suspicious"
                note = ("image is unlike the detector's training covers (cover-source mismatch); "
                        "ML verdict capped at 'suspicious'")
        return DetectorReport(probability=p, verdict=verdict, in_distribution=in_dist, note=note)

    def save(self, directory: Path) -> Path:
        path = Path(directory) / DETECTOR_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self.clf, "threshold": self.threshold, "meta": self.meta,
                     "ood": self.ood, "ood_threshold": self.ood_threshold}, path)
        return path

    @classmethod
    def load(cls, directory: Path) -> Steganalyzer:
        blob = joblib.load(Path(directory) / DETECTOR_FILE)
        return cls(blob["clf"], blob["threshold"], blob.get("meta"),
                   blob.get("ood"), blob.get("ood_threshold"))


# --- training ------------------------------------------------------------

_W_MODEL = None


def _init(model_dir: str) -> None:
    global _W_MODEL
    from .model import AIDEModel
    try:
        _W_MODEL = AIDEModel.load(Path(model_dir))
    except FileNotFoundError:
        _W_MODEL = AIDEModel.default()


def _sample_job(job) -> list[tuple[str, str, np.ndarray]]:
    """Features for one cover: the clean version plus one AIDE stego per
    variant. Returns (variant, image, features) rows."""
    from . import blind, pipeline
    path, variants, seed = job
    rng = np.random.default_rng(seed)
    gray = np.array(Image.open(path).convert("L"))
    rows = [("clean", Path(path).name, extract_features(gray))]
    for variant in variants:
        mode, profile = variant.split(":")
        msg = os.urandom(int(rng.integers(60, 260)))
        try:
            if mode == "blind":
                stego = blind.embed(Path(path), msg, b"detector-training", _W_MODEL, profile=profile).stego
            else:
                stego = pipeline.embed(Path(path), msg, b"detector-training", _W_MODEL, profile=profile).stego
        except pipeline.AideCapacityError:
            continue
        rows.append((variant, Path(path).name, extract_features(stego)))
    return rows


DEFAULT_VARIANTS = ("sidecar:quality", "sidecar:balanced", "sidecar:robust",
                    "blind:balanced", "blind:robust")


def train_detector(image_paths: list[Path], out_dir: Path, model_dir: Path = Path("data/models"),
                   variants=DEFAULT_VARIANTS, test_fraction: float = 0.3, seed: int = 0,
                   workers: int | None = None) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
    from sklearn.metrics import roc_auc_score

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    paths = list(image_paths)
    rng.shuffle(paths)
    n_test = max(1, int(len(paths) * test_fraction))
    split = {p.name: ("test" if i < n_test else "train") for i, p in enumerate(paths)}

    jobs = [(str(p), list(variants), seed + i) for i, p in enumerate(paths)]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    rows: list = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(str(model_dir),)) as ex:
        for r in ex.map(_sample_job, jobs):
            rows.extend(r)

    X = np.array([r[2] for r in rows])
    y = np.array([0 if r[0] == "clean" else 1 for r in rows])
    var = np.array([r[0] for r in rows])
    is_test = np.array([split[r[1]] == "test" for r in rows])

    # Class-balance the training set: one clean row per cover vs several
    # stego variants would otherwise bias the classifier toward "stego".
    w = np.where(y == 0, (y == 1)[~is_test].sum() / max(1, (y == 0)[~is_test].sum()), 1.0)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=seed)
    clf.fit(X[~is_test], y[~is_test], sample_weight=w[~is_test])

    # Cover-source-mismatch guard: anything scoring below the 1st
    # percentile of the training covers' own IsolationForest scores is
    # treated as out-of-distribution (~1% of in-distribution covers too,
    # which only ever *softens* a verdict, never raises one).
    ood = IsolationForest(n_estimators=200, random_state=seed).fit(X[~is_test])
    ood_threshold = float(np.quantile(ood.score_samples(X[~is_test]), 0.01))
    test_in_dist = float((ood.score_samples(X[is_test]) >= ood_threshold).mean())

    p_test = clf.predict_proba(X[is_test])[:, 1]
    y_test, v_test = y[is_test], var[is_test]
    # Operating threshold: 5% false-positive rate on held-out clean covers.
    clean_scores = p_test[y_test == 0]
    threshold = float(np.quantile(clean_scores, 0.95)) if len(clean_scores) else 0.5
    threshold = max(threshold, 0.5)
    per_variant = {}
    for v in variants:
        m = v_test == v
        if m.any():
            per_variant[v] = {
                "detected": float((p_test[m] >= threshold).mean()),
                "auc_vs_clean": float(roc_auc_score(
                    np.r_[np.zeros(len(clean_scores)), np.ones(m.sum())],
                    np.r_[clean_scores, p_test[m]])),
                "n": int(m.sum()),
            }
    meta = {
        "train_covers": len(paths) - n_test, "test_covers": n_test,
        "variants": list(variants), "feature_dims": int(X.shape[1]),
        "threshold_at_5pct_fpr": threshold,
        "test_false_positive_rate": float((clean_scores >= threshold).mean()) if len(clean_scores) else None,
        "test_auc_overall": float(roc_auc_score(y_test, p_test)) if len(set(y_test)) > 1 else None,
        "per_variant": per_variant,
        "ood_test_in_distribution_rate": test_in_dist,
        "training_seconds": round(time.perf_counter() - t0, 1),
    }
    Steganalyzer(clf, threshold, meta, ood, ood_threshold).save(out_dir)
    (Path(out_dir) / "steganalyzer_report.json").write_text(json.dumps(meta, indent=2))
    return meta
