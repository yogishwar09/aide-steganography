"""RGB cover image support (report 2.1 step 1: "Accepts RGB or grayscale
images"). See aide/pipeline.py's module docstring for the Y-channel-only
design this is built on."""
from __future__ import annotations

import numpy as np

from aide import attacks, metrics, pipeline


def test_load_image_detects_rgb_vs_grayscale(rgb_sample_path, real_sample_image_path):
    rgb = pipeline.load_image(rgb_sample_path)
    assert rgb.mode == "RGB"
    assert rgb.y.ndim == 2
    assert rgb.chroma is not None
    cb, cr = rgb.chroma
    assert cb.shape == rgb.y.shape == cr.shape

    gray = pipeline.load_image(real_sample_image_path)
    assert gray.mode == "L"
    assert gray.chroma is None


def test_compose_image_roundtrips_shape_and_mode(rgb_sample_path):
    loaded = pipeline.load_image(rgb_sample_path)
    composed = pipeline.compose_image(loaded.y, loaded.mode, loaded.chroma)
    assert composed.mode == "RGB"
    assert np.array(composed).shape == (*loaded.y.shape, 3)


def test_rgb_embed_recover_roundtrip(rgb_sample_path, trained_model):
    message = b"Rendezvous at grid 47N 12E at 0300 hours."
    secret = b"defence-shared-secret-2026"

    result = pipeline.embed(rgb_sample_path, message, secret, trained_model)
    assert result.mode == "RGB"
    assert result.chroma is not None

    stego_img = result.compose()
    assert stego_img.mode == "RGB"

    rec = pipeline.recover(result.stego, result.sidecar, secret)
    assert rec.success
    assert rec.plaintext == message


def test_rgb_stego_visually_close_to_cover(rgb_sample_path, trained_model):
    from PIL import Image

    message = b"short covert payload"
    secret = b"defence-shared-secret-2026"

    result = pipeline.embed(rgb_sample_path, message, secret, trained_model)
    cover_arr = np.array(Image.open(rgb_sample_path).convert("RGB"))
    stego_arr = np.array(result.compose())

    assert cover_arr.shape == stego_arr.shape
    assert metrics.psnr(cover_arr, stego_arr) > 25.0
    assert metrics.ssim(cover_arr, stego_arr) > 0.85


def test_rgb_survives_save_reload_roundtrip(tmp_path, rgb_sample_path, trained_model):
    """The embedding must survive being written to disk as a real color
    PNG and read back, not just staying in memory as the Y-plane array."""
    message = b"disk roundtrip check"
    secret = b"defence-shared-secret-2026"

    result = pipeline.embed(rgb_sample_path, message, secret, trained_model)
    stego_path = tmp_path / "stego.png"
    result.compose().save(stego_path)

    reloaded = pipeline.load_image(stego_path)
    assert reloaded.mode == "RGB"
    rec = pipeline.recover(reloaded.y, result.sidecar, secret)
    assert rec.success
    assert rec.plaintext == message


def test_rgb_attack_recovery_preserves_color(tmp_path, rgb_sample_path, trained_model):
    """Attacks operate on the luma plane; the attacked image should still
    reconstitute as valid, correctly-shaped RGB, and recovery should still
    work through a mild attack."""
    message = b"color attack test"
    secret = b"defence-shared-secret-2026"

    result = pipeline.embed(rgb_sample_path, message, secret, trained_model)
    attacked_y = attacks.compression(result.stego, quality=70)
    attacked_path = tmp_path / "attacked.png"
    pipeline.save_image(attacked_y, result.mode, result.chroma, attacked_path)

    reattacked = pipeline.load_image(attacked_path)
    assert reattacked.mode == "RGB"
    assert np.array(reattacked.chroma).shape[1:] == result.stego.shape

    rec = pipeline.recover(reattacked.y, result.sidecar, secret)
    assert rec.success
    assert rec.plaintext == message
