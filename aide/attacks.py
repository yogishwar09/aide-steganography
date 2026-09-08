"""Attack simulations for testing recovery (report section 2.3).

Six attack categories: compression, noise injection, cropping, pixel
tampering/splicing, filtering, and geometric scaling. All operate on a
grayscale uint8 image and return an image of the *same* shape, so the
fixed block grid the sidecar metadata describes stays aligned - cropping
and splicing are therefore simulated as fixed-canvas region replacement
(the removed/replaced area keeps the surrounding geometry intact) rather
than a true crop-and-shrink; an actual "shrink then resize back" scenario
is exactly what the scaling attack below covers.
"""
from __future__ import annotations

import cv2
import numpy as np


def compression(gray: np.ndarray, quality: int = 50) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)


def gaussian_noise(gray: np.ndarray, sigma: float = 20.0, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, gray.shape)
    return np.clip(gray.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def salt_and_pepper(gray: np.ndarray, amount: float = 0.02, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = gray.copy()
    roll = rng.random(gray.shape)
    out[roll < amount / 2] = 0
    out[roll > 1 - amount / 2] = 255
    return out


def _replace_region(gray: np.ndarray, fraction: float, seed: int | None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = gray.shape
    side = max(1, min(int(np.sqrt(fraction * h * w)), min(h, w)))
    y0 = int(rng.integers(0, max(1, h - side + 1)))
    x0 = int(rng.integers(0, max(1, w - side + 1)))
    out = gray.copy()
    out[y0:y0 + side, x0:x0 + side] = rng.integers(0, 256, (side, side), dtype=np.uint8)
    return out


def cropping(gray: np.ndarray, fraction: float = 0.15, seed: int | None = None) -> np.ndarray:
    """5-25% random region removal, canvas-size preserved (see module docstring)."""
    return _replace_region(gray, fraction, seed)


def splicing(gray: np.ndarray, fraction: float = 0.05, seed: int | None = None) -> np.ndarray:
    """Smaller, more surgical region replacement than cropping - models
    targeted pixel replacement / document splicing."""
    return _replace_region(gray, fraction, seed)


def median_filter(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.medianBlur(gray, ksize)


def gaussian_filter(gray: np.ndarray, ksize: int = 3, sigma: float = 1.0) -> np.ndarray:
    return cv2.GaussianBlur(gray, (ksize, ksize), sigma)


def scaling(gray: np.ndarray, factor: float = 0.5) -> np.ndarray:
    """Resize down then back up (report: 'resize + revert + resampling')."""
    h, w = gray.shape
    small = cv2.resize(gray, (max(1, int(w * factor)), max(1, int(h * factor))),
                        interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


ATTACKS = {
    "compression": compression,
    "gaussian_noise": gaussian_noise,
    "salt_pepper": salt_and_pepper,
    "cropping": cropping,
    "splicing": splicing,
    "median_filter": median_filter,
    "gaussian_filter": gaussian_filter,
    "scaling": scaling,
}
