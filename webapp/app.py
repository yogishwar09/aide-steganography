"""AIDE web demo: a local, clickable interface over the `aide` package for
the two defence/intelligence sides of this project -

  Operator mode  - embed/recover a secret message through a cover image
                   (secure covert communication, tested against attacks)
  Analyst mode   - blind steganalysis: is an arbitrary intercepted image
                   carrying hidden data? (aide.detect, no secret needed)

This is a thin HTTP wrapper: every real computation happens in the `aide`
package exactly as it does via the CLI - this file has no independent
logic of its own for embedding, recovery, or detection. Stateless by
design: the browser holds all intermediate state (stego image bytes,
sidecar JSON) as it moves between steps, POSTing back what each next
endpoint needs; the server touches disk only via short-lived temp files
(the `aide` functions take file paths, not in-memory buffers) and cleans
up after every request.

Run with:
    python webapp/app.py
then open http://127.0.0.1:5000
"""
from __future__ import annotations

import base64
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aide import attacks, metrics, pipeline, steganalysis, visualize  # noqa: E402
from aide.features import partition_blocks  # noqa: E402
from aide.model import AIDEModel  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "data" / "models"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB upload cap
# Always pick up template edits without a restart, independent of debug
# mode (Jinja2 caches compiled templates in memory otherwise, which is
# confusing when editing webapp/templates/index.html with the server
# already running - a real issue hit while building this).
app.config["TEMPLATES_AUTO_RELOAD"] = True

_model_cache: AIDEModel | None = None


def get_model() -> AIDEModel:
    global _model_cache
    if _model_cache is None:
        try:
            _model_cache = AIDEModel.load(MODEL_DIR)
        except FileNotFoundError:
            _model_cache = AIDEModel.default()
    return _model_cache


def _image_to_b64(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_to_temp_file(b64_data: str, suffix: str = ".png") -> Path:
    """Writes decoded image bytes to a temp file, since `aide.pipeline`'s
    functions load from a path (they need the raw file bytes for the
    image-bound key derivation - see aide/crypto.py)."""
    raw = base64.b64decode(b64_data)
    fd, path = tempfile.mkstemp(suffix=suffix)
    Path(path).write_bytes(raw)
    return Path(path)


def _upload_to_temp_file(file_storage) -> Path:
    suffix = Path(file_storage.filename or "upload.png").suffix or ".png"
    fd, path = tempfile.mkstemp(suffix=suffix)
    file_storage.save(path)
    return Path(path)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/embed", methods=["POST"])
def api_embed():
    cover_file = request.files.get("cover")
    message = request.form.get("message", "")
    secret = request.form.get("secret", "")
    if not cover_file or not message or not secret:
        return jsonify({"error": "cover, message, and secret are all required"}), 400

    tmp_path = _upload_to_temp_file(cover_file)
    try:
        model = get_model()
        try:
            result = pipeline.embed(tmp_path, message.encode(), secret.encode(), model)
        except pipeline.AideCapacityError as exc:
            return jsonify({"error": str(exc)}), 400

        cover_arr = np.array(Image.open(tmp_path).convert(result.mode))
        stego_img = result.compose()
        stego_arr = np.array(stego_img)

        from collections import Counter
        method_bits = Counter()
        for e in result.sidecar.embedding_map:
            method_bits[e.method] += e.bit_end - e.bit_start
        total_bits = sum(method_bits.values()) or 1

        return jsonify({
            "stego_b64": _image_to_b64(stego_img),
            "sidecar": result.sidecar.to_json(),
            "mode": result.mode,
            "message_bytes": len(message.encode()),
            "blocks_used": len(result.sidecar.embedding_map),
            "payload_bytes": total_bits // 8,
            "method_breakdown": {m: round(100 * b / total_bits, 1) for m, b in method_bits.items()},
            "psnr_db": metrics.psnr(cover_arr, stego_arr),
            "ssim": metrics.ssim(cover_arr, stego_arr),
        })
    finally:
        tmp_path.unlink(missing_ok=True)


@app.route("/api/attack", methods=["POST"])
def api_attack():
    payload = request.get_json(force=True)
    stego_b64 = payload.get("stego_b64")
    attack_type = payload.get("type")
    params = payload.get("params") or {}
    if not stego_b64 or attack_type not in attacks.ATTACKS:
        return jsonify({"error": "stego_b64 and a valid attack type are required"}), 400

    tmp_path = _b64_to_temp_file(stego_b64)
    try:
        loaded = pipeline.load_image(tmp_path)
        attacked_y = attacks.ATTACKS[attack_type](loaded.y, **params)
        attacked_img = pipeline.compose_image(attacked_y, loaded.mode, loaded.chroma)
        return jsonify({"attacked_b64": _image_to_b64(attacked_img)})
    except TypeError as exc:
        return jsonify({"error": f"bad attack parameters: {exc}"}), 400
    finally:
        tmp_path.unlink(missing_ok=True)


@app.route("/api/recover", methods=["POST"])
def api_recover():
    payload = request.get_json(force=True)
    image_b64 = payload.get("image_b64")
    sidecar_json = payload.get("sidecar")
    secret = payload.get("secret", "")
    if not image_b64 or not sidecar_json or not secret:
        return jsonify({"error": "image_b64, sidecar, and secret are all required"}), 400

    tmp_path = _b64_to_temp_file(image_b64)
    try:
        y = pipeline.load_image(tmp_path).y
        sidecar = pipeline.Sidecar.from_json(sidecar_json)
        result = pipeline.recover(y, sidecar, secret.encode())
        return jsonify({
            "success": result.success,
            "message": result.plaintext.decode(errors="replace") if result.plaintext else None,
            "error": result.error,
            "erasure_count": result.erasure_count,
            "total_bytes": result.total_bytes,
        })
    except (json.JSONDecodeError, KeyError) as exc:
        return jsonify({"error": f"invalid sidecar: {exc}"}), 400
    finally:
        tmp_path.unlink(missing_ok=True)


@app.route("/api/detect", methods=["POST"])
def api_detect():
    image_file = request.files.get("image")
    if not image_file:
        return jsonify({"error": "image is required"}), 400

    tmp_path = _upload_to_temp_file(image_file)
    try:
        gray = pipeline.load_image(tmp_path).y
        report = steganalysis.analyze(gray)
        return jsonify({
            "verdict": report.verdict,
            "suspicion_score": report.suspicion_score,
            "chi_square_p": report.chi_square_p,
            "lsb_correlation": report.lsb_correlation,
            "lsb_bit_balance": report.lsb_bit_balance,
            "row_band_profile": report.row_band_profile,
        })
    finally:
        tmp_path.unlink(missing_ok=True)


@app.route("/api/attack-types")
def api_attack_types():
    return jsonify(sorted(attacks.ATTACKS))


@app.route("/api/visualize/embedding-map", methods=["POST"])
def api_visualize_embedding_map():
    """Color every block the AIDE engine actually used by which method
    (LSB/DCT/DWT) embedded into it - the most direct visual evidence of
    the adaptive, per-block, multi-method policy. See aide/visualize.py."""
    payload = request.get_json(force=True)
    image_b64 = payload.get("image_b64")
    sidecar_json = payload.get("sidecar")
    if not image_b64 or not sidecar_json:
        return jsonify({"error": "image_b64 and sidecar are required"}), 400

    tmp_path = _b64_to_temp_file(image_b64)
    try:
        gray = pipeline.load_image(tmp_path).y
        sidecar = pipeline.Sidecar.from_json(sidecar_json)
        blocks = partition_blocks(sidecar.height, sidecar.width, sidecar.block_size)
        overlay = visualize.embedding_map_overlay(gray, blocks, sidecar.embedding_map)
        return jsonify({"overlay_b64": _image_to_b64(overlay)})
    finally:
        tmp_path.unlink(missing_ok=True)


@app.route("/api/visualize/diff", methods=["POST"])
def api_visualize_diff():
    """Real, amplified |cover - stego| heatmap - honest about the fact
    that pixels did change, rather than only citing a PSNR number."""
    payload = request.get_json(force=True)
    cover_b64 = payload.get("cover_b64")
    stego_b64 = payload.get("stego_b64")
    if not cover_b64 or not stego_b64:
        return jsonify({"error": "cover_b64 and stego_b64 are required"}), 400

    cover_path = _b64_to_temp_file(cover_b64)
    stego_path = _b64_to_temp_file(stego_b64)
    try:
        cover_gray = pipeline.load_image(cover_path).y
        stego_gray = pipeline.load_image(stego_path).y
        if cover_gray.shape != stego_gray.shape:
            return jsonify({"error": "cover and stego dimensions do not match"}), 400
        heat = visualize.diff_heatmap(cover_gray, stego_gray)
        return jsonify({"heatmap_b64": _image_to_b64(heat)})
    finally:
        cover_path.unlink(missing_ok=True)
        stego_path.unlink(missing_ok=True)


@app.route("/api/visualize/tamper", methods=["POST"])
def api_visualize_tamper():
    """Which blocks does the receiver actually trust after an attack?
    Red = erasure (fed to Reed-Solomon as a known-bad byte range), amber =
    hash-mismatch left to blind error correction, untouched = trusted."""
    payload = request.get_json(force=True)
    image_b64 = payload.get("image_b64")
    sidecar_json = payload.get("sidecar")
    if not image_b64 or not sidecar_json:
        return jsonify({"error": "image_b64 and sidecar are required"}), 400

    tmp_path = _b64_to_temp_file(image_b64)
    try:
        gray = pipeline.load_image(tmp_path).y
        sidecar = pipeline.Sidecar.from_json(sidecar_json)
        blocks, report = visualize.tamper_report_for(gray, sidecar)
        overlay = visualize.tamper_overlay(gray, blocks, report)
        return jsonify({
            "overlay_b64": _image_to_b64(overlay),
            "erasure_blocks": int(report.erasure_mask.sum()),
            "hash_mismatch_blocks": int(report.hash_mismatch.sum()),
            "total_blocks": len(blocks),
        })
    finally:
        tmp_path.unlink(missing_ok=True)


@app.route("/api/model-info")
def api_model_info():
    """The learned decision engine's parameters, for the Model Insights
    panel - the report's own 'explainable, auditable, not a black box'
    novelty claim, made concrete."""
    model = get_model()
    return jsonify({
        "weights": model.weights.as_dict(),
        "thresholds": {
            "tau_min": model.thresholds.tau_min,
            "t_low": model.thresholds.t_low,
            "e_high": model.thresholds.e_high,
        },
        "meta": model.meta,
        "has_trained_classifier": model.method_clf is not None,
    })


if __name__ == "__main__":
    import logging
    import os

    port = int(os.environ.get("PORT", 5050))
    debug = os.environ.get("AIDE_WEB_DEBUG", "0") == "1"

    if not debug:
        # Quiet, presentation-friendly output: no reloader (which otherwise
        # restarts the process and reprints the whole startup banner), no
        # debugger, and no Werkzeug per-request access log line for every
        # click. Set AIDE_WEB_DEBUG=1 if you're actively editing app.py and
        # want auto-reload + tracebacks back.
        logging.getLogger("werkzeug").setLevel(logging.ERROR)

    model_status = "found" if (MODEL_DIR / "weights.json").exists() else "MISSING - run: aide train"
    print(f"Model directory: {MODEL_DIR} ({model_status})")
    print(f"AIDE Command Console ready -> http://127.0.0.1:{port}  (Ctrl+C to stop)")
    app.run(debug=debug, use_reloader=debug, port=port)
