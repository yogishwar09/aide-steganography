"""Smoke + correctness tests for webapp/app.py's HTTP API, using Flask's
test client (no live server needed). Exercises the exact same `aide`
package code paths as the CLI - this is the same embed/attack/recover/
detect logic, just reached over HTTP instead of argv.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

WEBAPP_DIR = Path(__file__).resolve().parent.parent / "webapp"
sys.path.insert(0, str(WEBAPP_DIR))


@pytest.fixture
def client(trained_model):  # trained_model fixture ensures data/models exists, else skips
    import app as webapp_app  # noqa: PLC0415 (import here so sys.path is set up first)

    webapp_app.app.config["TESTING"] = True
    with webapp_app.app.test_client() as c:
        yield c


def _png_bytes(gray: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(gray.astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def test_index_page_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"AIDE" in resp.data


def test_attack_types_lists_all_attacks(client):
    from aide import attacks

    resp = client.get("/api/attack-types")
    assert resp.status_code == 200
    assert set(resp.get_json()) == set(attacks.ATTACKS)


def test_embed_endpoint_requires_all_fields(client):
    resp = client.post("/api/embed", data={})
    assert resp.status_code == 400


def test_full_workflow_embed_attack_recover(client, real_sample_image_path):
    gray = np.array(Image.open(real_sample_image_path).convert("L"))
    cover_bytes = _png_bytes(gray)

    embed_resp = client.post("/api/embed", data={
        "cover": (io.BytesIO(cover_bytes), "cover.png"),
        "message": "Rendezvous at grid 47N 12E at 0300 hours.",
        "secret": "web-test-secret",
    }, content_type="multipart/form-data")
    assert embed_resp.status_code == 200
    embed_data = embed_resp.get_json()
    assert embed_data["mode"] == "L"
    assert embed_data["psnr_db"] > 20
    assert "stego_b64" in embed_data and "sidecar" in embed_data

    attack_resp = client.post("/api/attack", json={
        "stego_b64": embed_data["stego_b64"], "type": "compression", "params": {"quality": 50},
    })
    assert attack_resp.status_code == 200
    attacked_b64 = attack_resp.get_json()["attacked_b64"]

    recover_resp = client.post("/api/recover", json={
        "image_b64": attacked_b64, "sidecar": embed_data["sidecar"], "secret": "web-test-secret",
    })
    assert recover_resp.status_code == 200
    recover_data = recover_resp.get_json()
    assert recover_data["success"] is True
    assert recover_data["message"] == "Rendezvous at grid 47N 12E at 0300 hours."


def test_recover_with_wrong_secret_fails_gracefully(client, real_sample_image_path):
    gray = np.array(Image.open(real_sample_image_path).convert("L"))
    cover_bytes = _png_bytes(gray)

    embed_resp = client.post("/api/embed", data={
        "cover": (io.BytesIO(cover_bytes), "cover.png"),
        "message": "top secret",
        "secret": "correct-secret",
    }, content_type="multipart/form-data")
    embed_data = embed_resp.get_json()

    recover_resp = client.post("/api/recover", json={
        "image_b64": embed_data["stego_b64"], "sidecar": embed_data["sidecar"], "secret": "wrong-secret",
    })
    assert recover_resp.status_code == 200
    recover_data = recover_resp.get_json()
    assert recover_data["success"] is False


def test_detect_endpoint_on_clean_image(client, real_sample_image_path):
    with open(real_sample_image_path, "rb") as f:
        resp = client.post("/api/detect", data={"image": (f, "sample.pgm")}, content_type="multipart/form-data")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["verdict"] in ("clean", "suspicious", "likely_steganographic")
    assert 0.0 <= data["suspicion_score"] <= 1.0
    assert len(data["row_band_profile"]) > 0
    assert len(data["spatial_grid"]) > 0
    assert "image_b64" in data


def test_suspicion_heatmap_endpoint(client, real_sample_image_path):
    with open(real_sample_image_path, "rb") as f:
        detect_resp = client.post("/api/detect", data={"image": (f, "sample.pgm")}, content_type="multipart/form-data")
    detect_data = detect_resp.get_json()

    heatmap_resp = client.post("/api/visualize/suspicion-heatmap", json={
        "image_b64": detect_data["image_b64"], "spatial_grid": detect_data["spatial_grid"],
    })
    assert heatmap_resp.status_code == 200
    assert "overlay_b64" in heatmap_resp.get_json()


def test_detect_endpoint_catches_naive_lsb(client, real_sample_image_path):
    gray = np.array(Image.open(real_sample_image_path).convert("L"))
    rng = np.random.default_rng(0)
    flat = gray.reshape(-1)
    bits = rng.integers(0, 2, size=flat.size).astype(np.uint8)
    flat[:] = (flat & 0xFE) | bits

    resp = client.post("/api/detect", data={
        "image": (io.BytesIO(_png_bytes(gray)), "suspect.png"),
    }, content_type="multipart/form-data")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["verdict"] == "likely_steganographic"


def test_model_info_endpoint(client):
    resp = client.get("/api/model-info")
    assert resp.status_code == 200
    data = resp.get_json()
    assert set(data["weights"]) == {"alpha", "beta", "gamma", "delta"}
    assert set(data["thresholds"]) == {"tau_min", "t_low", "e_high"}
    assert "meta" in data


def test_visualization_endpoints(client, real_sample_image_path):
    gray = np.array(Image.open(real_sample_image_path).convert("L"))
    cover_bytes = _png_bytes(gray)

    embed_resp = client.post("/api/embed", data={
        "cover": (io.BytesIO(cover_bytes), "cover.png"),
        "message": "Rendezvous at grid 47N 12E at 0300 hours.",
        "secret": "viz-test-secret",
    }, content_type="multipart/form-data")
    embed_data = embed_resp.get_json()

    map_resp = client.post("/api/visualize/embedding-map", json={
        "image_b64": embed_data["stego_b64"], "sidecar": embed_data["sidecar"],
    })
    assert map_resp.status_code == 200
    assert "overlay_b64" in map_resp.get_json()

    diff_resp = client.post("/api/visualize/diff", json={
        "cover_b64": base64_of(cover_bytes), "stego_b64": embed_data["stego_b64"],
    })
    assert diff_resp.status_code == 200
    assert "heatmap_b64" in diff_resp.get_json()

    attack_resp = client.post("/api/attack", json={
        "stego_b64": embed_data["stego_b64"], "type": "splicing", "params": {"fraction": 0.08, "seed": 1},
    })
    attacked_b64 = attack_resp.get_json()["attacked_b64"]

    tamper_resp = client.post("/api/visualize/tamper", json={
        "image_b64": attacked_b64, "sidecar": embed_data["sidecar"],
    })
    assert tamper_resp.status_code == 200
    tamper_data = tamper_resp.get_json()
    assert "overlay_b64" in tamper_data
    assert tamper_data["total_blocks"] > 0


def base64_of(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode("ascii")
