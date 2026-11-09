"""Robustness profiles, shortened/configurable Reed-Solomon, reliability-
guided (GMD) decoding, the scrypt KDF, aide-0.1 sidecar compatibility, and
blind (sidecar-free) mode."""
from __future__ import annotations

import json

import numpy as np
import pytest

from aide import attacks, blind, crypto, ecc, pipeline
from aide.profiles import LEGACY_PROFILE, PROFILES, get_profile

MESSAGE = (b"CLASSIFIED SITREP: Convoy Alpha departs staging area 0600Z, route Bravo-7. "
           b"Escort confirmed, rendezvous at checkpoint Delta.")
SECRET = b"defence-shared-secret-2026"


# --- ECC -----------------------------------------------------------------

def test_plan_splits_short_messages_into_multiple_codewords():
    params = ecc.plan(60, nsym=64, min_codewords=4)
    n_cw = -(-(60 + ecc.LENGTH_HEADER) // params.data_len)
    assert n_cw >= 4
    # same parity ratio as the full-length code (within rounding)
    assert params.nsym / params.data_len == pytest.approx(64 / 191, rel=0.25)


def test_plan_uses_full_codewords_for_long_messages():
    params = ecc.plan(5000, nsym=32, min_codewords=4)
    # data bytes are balanced across codewords (less zero padding), so n
    # can land a few bytes under 255 - but parity stays the full nsym.
    assert params.codeword_len > 245 and params.nsym == 32


@pytest.mark.parametrize("nsym,k", [(16, 40), (64, 100), (128, 127)])
def test_shortened_codes_roundtrip_and_correct_t_errors(nsym, k):
    params = ecc.ECCParams(nsym=nsym, data_len=k)
    data = bytes(range(256)) * 2
    enc = bytearray(ecc.encode(data, params))
    for c in range(len(enc) // params.codeword_len):
        for i in range(params.t):
            enc[c * params.codeword_len + 2 * i] ^= 0xA5
    assert ecc.decode(bytes(enc), params=params) == data


def test_gmd_decoding_corrects_beyond_t_when_reliability_is_informative():
    params = ecc.ECCParams(nsym=32, data_len=223)
    data = bytes(range(200))
    enc = bytearray(ecc.encode(data, params))
    pos = np.random.default_rng(0).choice(255, 24, replace=False)  # 24 > t=16
    for i in pos:
        enc[i] ^= 0x5A
    with pytest.raises(ecc.AideECCError):
        ecc.decode(bytes(enc), params=params)
    rel = np.ones(255)
    rel[pos[:20]] = 0.01
    stats = ecc.DecodeStats()
    assert ecc.decode_with_reliability(bytes(enc), rel, None, params, stats) == data
    assert stats.gmd_codewords == 1


def test_ecc_params_reject_invalid_codeword_length():
    with pytest.raises(ValueError):
        ecc.ECCParams(nsym=100, data_len=200)


# --- crypto --------------------------------------------------------------

def test_scrypt_and_sha256_kdfs_produce_different_keys():
    h = b"\x01" * 32
    assert crypto.derive_key(h, b"pw", crypto.KDF_SCRYPT) != crypto.derive_key(h, b"pw", crypto.KDF_SHA256)
    assert len(crypto.derive_key(h, b"pw", crypto.KDF_SCRYPT)) == 32


def test_decrypt_with_wrong_kdf_is_rejected():
    h = b"\x02" * 32
    enc = crypto.encrypt(b"msg", h, b"pw", kdf=crypto.KDF_SCRYPT)
    with pytest.raises(crypto.AideAuthenticationError):
        crypto.decrypt(enc, h, b"pw", kdf=crypto.KDF_SHA256)
    assert crypto.decrypt(enc, h, b"pw", kdf=crypto.KDF_SCRYPT) == b"msg"


# --- profiles & sidecar --------------------------------------------------

def test_get_profile_resolves_names_and_rejects_unknown():
    assert get_profile(None).name == "balanced"
    assert get_profile("robust") is PROFILES["robust"]
    with pytest.raises(ValueError):
        get_profile("nope")


def test_profiles_trade_quality_for_strength_monotonically():
    names = ["quality", "balanced", "robust", "fortress"]
    deltas = [PROFILES[n].dct_delta for n in names]
    parities = [PROFILES[n].nsym for n in names]
    assert deltas == sorted(deltas) and parities == sorted(parities)


@pytest.mark.parametrize("profile", ["quality", "balanced", "robust"])
def test_every_profile_roundtrips_and_records_itself_in_sidecar(real_sample_image_path, trained_model, profile):
    try:
        res = pipeline.embed(real_sample_image_path, MESSAGE, SECRET, trained_model, profile=profile)
    except pipeline.AideCapacityError:
        pytest.skip("cover lacks capacity")
    sc = pipeline.Sidecar.from_json(res.sidecar.to_json())
    assert sc.profile.name == profile and sc.kdf == crypto.KDF_SCRYPT
    rec = pipeline.recover(res.stego, sc, SECRET)
    assert rec.success and rec.plaintext == MESSAGE


def test_aide_0_1_sidecar_without_profile_fields_parses_with_legacy_parameters(real_sample_image_path, trained_model):
    """Files written by AIDE 0.1 have no profile/ecc/kdf keys; they must
    keep decoding with the exact fixed parameters they were made with."""
    res = pipeline.embed(real_sample_image_path, MESSAGE, SECRET, trained_model,
                         profile=LEGACY_PROFILE)
    # Re-create what 0.1 would have produced: RS(255,223), sha256 KDF.
    image_hash = res.sidecar.image_hash
    enc = crypto.encrypt(MESSAGE, image_hash, SECRET, kdf=crypto.KDF_SHA256)
    payload = json.loads(res.sidecar.to_json())
    for k in ("profile", "ecc", "kdf"):
        payload.pop(k)
    payload["version"] = "aide-0.1"
    legacy = pipeline.Sidecar.from_json(json.dumps(payload))
    assert legacy.profile == LEGACY_PROFILE and legacy.ecc_params == ecc.DEFAULT_PARAMS
    assert legacy.kdf == crypto.KDF_SHA256
    assert enc.nonce  # sanity: the legacy KDF is still available for 0.1 payloads


# --- blind mode ----------------------------------------------------------

@pytest.fixture
def blind_embedded(real_sample_image_path, trained_model):
    return blind.embed(real_sample_image_path, MESSAGE, SECRET, trained_model, profile="robust")


def test_blind_roundtrip_needs_only_image_and_secret(blind_embedded):
    rec = blind.recover(blind_embedded.stego, SECRET)
    assert rec.success and rec.plaintext == MESSAGE and rec.profile == "robust"


def test_blind_wrong_secret_finds_nothing(blind_embedded):
    rec = blind.recover(blind_embedded.stego, b"not-the-secret")
    assert not rec.success and rec.plaintext is None


def test_blind_mode_survives_jpeg_and_noise(blind_embedded):
    for attacked in (attacks.compression(blind_embedded.stego, quality=50),
                     attacks.gaussian_noise(blind_embedded.stego, sigma=20, seed=3)):
        rec = blind.recover(attacked, SECRET)
        assert rec.success and rec.plaintext == MESSAGE


def test_blind_mode_survives_color_png_roundtrip(tmp_path, rgb_sample_path, trained_model):
    res = blind.embed(rgb_sample_path, MESSAGE, SECRET, trained_model, profile="balanced")
    out = tmp_path / "stego.png"
    res.compose().save(out)
    rec = blind.recover(pipeline.load_image(out).y, SECRET)
    assert res.mode == "RGB" and rec.success and rec.plaintext == MESSAGE


def test_blind_capacity_error_on_tiny_cover(tmp_path, trained_model):
    from PIL import Image
    p = tmp_path / "tiny.png"
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (48, 48), dtype=np.uint8)).save(p)
    with pytest.raises(pipeline.AideCapacityError):
        blind.embed(p, MESSAGE * 10, SECRET, trained_model)


def test_real_aide_0_1_file_decodes():
    """Regression fixture produced by the actual AIDE 0.1 code (git history,
    before profiles/shortened RS/scrypt): its stego image + aide-0.1
    sidecar must keep decoding - including after a JPEG re-save."""
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "v0_1_stego.png"
    sidecar = pipeline.load_sidecar(Path(str(fixture) + ".json"))
    assert json.loads(Path(str(fixture) + ".json").read_text())["version"] == "aide-0.1"
    stego = pipeline.load_image(fixture).y
    for img in (stego, attacks.compression(stego, quality=70)):
        rec = pipeline.recover(img, sidecar, b"fixture-secret")
        assert rec.success and rec.plaintext == b"AIDE 0.1 compatibility fixture"
