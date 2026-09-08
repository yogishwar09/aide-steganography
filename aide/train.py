"""Train the AIDE decision engine (report O1 / phase 3).

Two things are learned, from real image blocks, not assumed:

1. WHICH METHOD: the report's own rule (texture < T_low -> LSB; edge >
   E_high -> DWT; else DCT) is used to generate the ground-truth method
   label, with T_low/E_high set from the 33rd/67th percentile of texture
   and edge strength observed across the sample images (so classes are
   roughly balanced instead of hand-picked). A DecisionTreeClassifier is
   then trained to reproduce and smooth that rule from all four features
   (including entropy/variance, which the hard rule ignores) - this is the
   "AI refines the score" step.
2. HOW MUCH (bpp) and CONFIDENCE (skip): for each block, under its assigned
   method, we actually embed random bits at bpp=4,3,2,1 and measure the
   resulting block PSNR, keeping the highest bpp that stays at/above a
   target PSNR. If even bpp=1 is unsafe, the block is labelled SKIP. This
   is empirical/measurement-grounded rather than circular.

The trained classifier's normalised feature_importances_ become the
explainable (alpha, beta, gamma, delta) weights in aide.scoring.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .embedding import METHODS
from .features import extract_image_features
from .model import AIDEModel, AIDEThresholds
from .scoring import ScoringWeights, compute_scores

DEFAULT_TARGET_PSNR = 42.0
BPP_CHOICES = (4, 3, 2, 1)
IMAGE_EXTS = (".pgm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# DCT/DWT need a much larger per-coefficient delta than LSB needs per-pixel
# to survive JPEG-style requantization (see aide/embedding/dct.py and
# dwt.py's DELTA comments, calibrated empirically against cv2 JPEG q=50).
# That inherently means lower PER-BLOCK PSNR at the handful of blocks they
# touch - but each only modifies a handful of coefficients, so the
# contribution to *whole-image* PSNR (the report's actual >42dB target,
# measured in aide.metrics.psnr over the full stego image) stays small.
# Gating DCT/DWT block selection at the same 42dB bar as LSB - which
# modifies every pixel in its blocks and whose local artifacts are far
# more perceptible - would empirically SKIP nearly all DCT/DWT blocks
# (measured: bpp=1 medians land around 30/29 dB) and collapse the "which
# method" decision back down to "LSB or nothing", defeating the point.
METHOD_PSNR_RELAXATION = {"LSB": 0.0, "DCT": -14.0, "DWT": -4.0}
# DWT's relaxation is much smaller than DCT's because the current
# implementation (2-level Haar, embedding in the coarse cA2 sub-band - see
# aide/embedding/dwt.py) measured a bpp=1 median block PSNR of ~40 dB,
# close to the 42 dB bar already; the older single-level detail-sub-band
# design needed the same -15 dB relaxation DCT still uses.


def _block_psnr(original: np.ndarray, stego: np.ndarray) -> float:
    mse = float(np.mean((original.astype(np.float64) - stego.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * float(np.log10((255.0 ** 2) / mse))


def _max_safe_bpp(block: np.ndarray, method: str, rng: np.random.Generator,
                   target_psnr: float) -> int | None:
    mod = METHODS[method]
    target_psnr = target_psnr + METHOD_PSNR_RELAXATION.get(method, 0.0)
    for bpp in BPP_CHOICES:
        cap = mod.capacity_bits(block.shape, bpp)
        if cap == 0:
            continue
        bits = rng.integers(0, 2, size=cap).astype(np.uint8)
        stego = mod.embed(block, bpp, bits)
        if _block_psnr(block, stego) >= target_psnr:
            return bpp
    return None


def list_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def build_training_set(image_paths: list[Path], target_psnr: float, seed: int = 0,
                        max_blocks: int | None = None):
    rng = np.random.default_rng(seed)

    all_blocks_info = []  # (gray, block_slice, feature_vector)
    feats_list = []
    for path in image_paths:
        gray = np.array(Image.open(path).convert("L"))
        blocks, feats = extract_image_features(gray)
        for b, feat in zip(blocks, feats):
            all_blocks_info.append((gray, b.slice()))
            feats_list.append(feat)
            if max_blocks and len(feats_list) >= max_blocks:
                break
        if max_blocks and len(feats_list) >= max_blocks:
            break

    X = np.array(feats_list)
    texture, edge = X[:, 0], X[:, 3]
    t_low = float(np.percentile(texture, 33))
    e_high = float(np.percentile(edge, 67))

    method_labels = np.where(texture < t_low, "LSB", np.where(edge > e_high, "DWT", "DCT"))
    method_labels = method_labels.astype("<U4")  # room for "SKIP" (numpy would else truncate it)

    bpp_labels = np.zeros(len(X), dtype=int)
    for i, ((gray, sl), method) in enumerate(zip(all_blocks_info, method_labels)):
        block = gray[sl]
        bpp = _max_safe_bpp(block, str(method), rng, target_psnr)
        if bpp is None:
            method_labels[i] = "SKIP"
            bpp_labels[i] = 0
        else:
            bpp_labels[i] = bpp

    return X, method_labels, bpp_labels, t_low, e_high


def train(image_dir: Path, out_dir: Path, target_psnr: float = DEFAULT_TARGET_PSNR,
          max_images: int | None = None, max_blocks: int | None = 4000, seed: int = 0) -> dict:
    t0 = time.perf_counter()
    image_paths = list_images(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"no images found in {image_dir}")
    if max_images:
        image_paths = image_paths[:max_images]

    X, method_labels, bpp_labels, t_low, e_high = build_training_set(
        image_paths, target_psnr, seed, max_blocks
    )
    kept_mask = method_labels != "SKIP"

    strat = method_labels if len(set(method_labels)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, method_labels, test_size=0.2, random_state=seed, stratify=strat
    )
    method_clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=seed)
    method_clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, method_clf.predict(X_train))
    val_acc = accuracy_score(y_val, method_clf.predict(X_val))

    bpp_reg = None
    bpp_val_mae = None
    if kept_mask.sum() >= 10:
        Xb, yb = X[kept_mask], bpp_labels[kept_mask]
        Xb_train, Xb_val, yb_train, yb_val = train_test_split(Xb, yb, test_size=0.2, random_state=seed)
        bpp_reg = DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=seed)
        bpp_reg.fit(Xb_train, yb_train)
        bpp_val_mae = float(mean_absolute_error(yb_val, bpp_reg.predict(Xb_val)))

    importances = method_clf.feature_importances_.astype(float)
    total = importances.sum()
    importances = importances / total if total > 0 else np.array([0.25, 0.25, 0.25, 0.25])
    weights = ScoringWeights(*[float(v) for v in importances])

    scores = compute_scores(X, weights)
    if kept_mask.any() and (~kept_mask).any():
        skip_scores, keep_scores = scores[~kept_mask], scores[kept_mask]
        if skip_scores.max() < keep_scores.min():
            tau_min = float((skip_scores.max() + keep_scores.min()) / 2.0)
        else:
            tau_min = float(np.percentile(scores, (~kept_mask).mean() * 100))
    else:
        tau_min = float(np.percentile(scores, 5))

    thresholds = AIDEThresholds(tau_min=tau_min, t_low=t_low, e_high=e_high)
    meta = {
        "dataset_dir": str(image_dir),
        "num_images": len(image_paths),
        "num_blocks": int(len(X)),
        "target_psnr_db": target_psnr,
        "method_train_accuracy": float(train_acc),
        "method_val_accuracy": float(val_acc),
        "bpp_val_mae": bpp_val_mae,
        "label_distribution": {k: int((method_labels == k).sum()) for k in sorted(set(method_labels))},
        "training_seconds": time.perf_counter() - t0,
    }

    model = AIDEModel(weights=weights, thresholds=thresholds, method_clf=method_clf,
                       bpp_reg=bpp_reg, meta=meta)
    model.save(out_dir)
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("data/samples"))
    parser.add_argument("--out", type=Path, default=Path("data/models"))
    parser.add_argument("--target-psnr", type=float, default=DEFAULT_TARGET_PSNR)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-blocks", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    meta = train(args.images, args.out, args.target_psnr, args.max_images, args.max_blocks, args.seed)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
