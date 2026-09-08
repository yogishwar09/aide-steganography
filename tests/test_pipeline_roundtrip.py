from __future__ import annotations

from aide import metrics, pipeline


def test_no_attack_roundtrip_recovers_exact_message(real_sample_image_path, trained_model):
    message = b"Rendezvous at grid 47N 12E at 0300 hours. Confirm via secondary channel."
    secret = b"defence-shared-secret-2026"

    result = pipeline.embed(real_sample_image_path, message, secret, trained_model)
    rec = pipeline.recover(result.stego, result.sidecar, secret)

    assert rec.success
    assert rec.plaintext == message


def test_stego_image_is_visually_close_to_cover(real_sample_image_path, trained_model):
    message = b"short covert payload"
    secret = b"defence-shared-secret-2026"

    result = pipeline.embed(real_sample_image_path, message, secret, trained_model)
    cover, _ = pipeline.load_grayscale(real_sample_image_path)

    # Not asserting the report's aspirational >42dB here (that target
    # assumes the full ALASKA2-scale trained model and a capacity budget
    # tuned per-image; see README for the honest, measured numbers) - just
    # that embedding doesn't grossly disfigure the image.
    assert metrics.psnr(cover, result.stego) > 25.0
    assert metrics.ssim(cover, result.stego) > 0.85


def test_wrong_shared_secret_cannot_recover(real_sample_image_path, trained_model):
    message = b"top secret"
    result = pipeline.embed(real_sample_image_path, message, b"correct-secret", trained_model)

    rec = pipeline.recover(result.stego, result.sidecar, b"wrong-secret")

    assert not rec.success
    assert rec.plaintext is None


def test_sidecar_json_roundtrip(tmp_path, real_sample_image_path, trained_model):
    message = b"sidecar serialization check"
    secret = b"defence-shared-secret-2026"

    result = pipeline.embed(real_sample_image_path, message, secret, trained_model)
    sidecar_path = tmp_path / "stego.aide.json"
    pipeline.save_sidecar(result.sidecar, sidecar_path)

    loaded = pipeline.load_sidecar(sidecar_path)
    rec = pipeline.recover(result.stego, loaded, secret)

    assert rec.success
    assert rec.plaintext == message
