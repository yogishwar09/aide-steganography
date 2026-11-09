"""AIDE web console + REST API: a local, clickable interface over the `aide`
package for the two defence/intelligence sides of this project -

  Operator mode  - embed/recover a secret message through a cover image
                   (secure covert communication, tested against attacks),
                   in sidecar or blind (self-contained) mode, with a
                   choice of robustness profile
  Analyst mode   - blind steganalysis: is an arbitrary intercepted image
                   carrying hidden data? Single image or batch screening.

This is a thin HTTP wrapper: every real computation happens in the `aide`
package exactly as it does via the CLI. Stateless by design: the browser
holds all intermediate state (stego image bytes, sidecar JSON) as it moves
between steps; the server touches disk only via short-lived temp files and
cleans up after every request.

Every endpoint is served at both /api/<name> (used by the bundled UI) and
the versioned /api/v1/<name> (for integrations; contract in
/api/v1/openapi.json). Set AIDE_API_KEY to require an `X-API-Key` header
on API calls - e.g. when exposing the console beyond localhost.

Run with:
    python webapp/app.py            (development server)
    gunicorn -w 2 -b 0.0.0.0:5050 webapp.app:app   (production, see Dockerfile)
"""
from __future__ import annotations

import base64
import hmac
import io
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
from flask import Flask, g, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aide import __version__, attacks, blind, metrics, pipeline, steganalysis, visualize  # noqa: E402
from aide.features import partition_blocks  # noqa: E402
from aide.model import AIDEModel  # noqa: E402
from aide.profiles import DEFAULT_PROFILE, PROFILES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("AIDE_MODEL_DIR", REPO_ROOT / "data" / "models"))
BENCHMARK_JSON = Path(os.environ.get("AIDE_BENCHMARK_JSON", REPO_ROOT / "reports" / "benchmark" / "benchmark.json"))
DEMO_DIR = Path(__file__).resolve().parent / "static" / "demo"
DEMO_COVER_LABELS = {
    "astronaut": "Astronaut portrait (colour)",
    "coffee": "Coffee cup (colour)",
    "rocket": "Rocket launch (colour)",
    "bossbase_1022": "BOSSbase #1022 (grayscale)",
}
MAX_BATCH = 50

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB upload cap
# Always pick up template edits without a restart, independent of debug
# mode (Jinja2 caches compiled templates in memory otherwise).
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.json.sort_keys = False  # keep profiles in quality -> fortress order for the UI
log = logging.getLogger("aide.web")

_model_cache: AIDEModel | None = None
_detector_cache: object | None = None
_detector_loaded = False


def get_model() -> AIDEModel:
    global _model_cache
    if _model_cache is None:
        try:
            _model_cache = AIDEModel.load(MODEL_DIR)
        except FileNotFoundError:
            _model_cache = AIDEModel.default()
    return _model_cache


def get_detector():
    """The trained ML steganalyser, if `aide train-detector` has been run."""
    global _detector_cache, _detector_loaded
    if not _detector_loaded:
        _detector_loaded = True
        try:
            from aide.mlsteg import Steganalyzer
            _detector_cache = Steganalyzer.load(MODEL_DIR)
        except FileNotFoundError:
            _detector_cache = None
    return _detector_cache


# ---------------------------------------------------------------- plumbing

class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@app.errorhandler(ApiError)
def _api_error(exc: ApiError):
    return jsonify({"error": str(exc)}), exc.status


@app.errorhandler(413)
def _too_large(_exc):
    return jsonify({"error": f"upload exceeds {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB limit"}), 413


@app.errorhandler(UnidentifiedImageError)
def _bad_image(_exc):
    return jsonify({"error": "could not decode the uploaded file as an image"}), 400


@app.errorhandler(Exception)
def _unhandled(exc):
    from werkzeug.exceptions import HTTPException
    if isinstance(exc, HTTPException):
        return jsonify({"error": exc.description}), exc.code
    log.exception("unhandled error in %s (request %s)", request.path, getattr(g, "request_id", "?"))
    return jsonify({"error": "internal error", "request_id": getattr(g, "request_id", None)}), 500


@app.before_request
def _before():
    g.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    g.t0 = time.perf_counter()
    api_key = os.environ.get("AIDE_API_KEY")
    if api_key and request.path.startswith("/api/") and request.path != "/api/v1/openapi.json":
        supplied = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(supplied, api_key):
            return jsonify({"error": "missing or invalid X-API-Key"}), 401
    return None


@app.after_request
def _after(resp):
    resp.headers["X-Request-ID"] = getattr(g, "request_id", "")
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline' "
        "https://fonts.googleapis.com; font-src https://fonts.gstatic.com; script-src 'self' 'unsafe-inline'")
    if request.path.startswith("/api/"):
        resp.headers.setdefault("Cache-Control", "no-store")
        log.info("%s %s -> %s in %.0f ms [%s]", request.method, request.path, resp.status_code,
                 (time.perf_counter() - getattr(g, "t0", time.perf_counter())) * 1000, g.request_id)
    return resp


def api_route(rule: str, **options):
    """Register a handler at /api<rule> and the versioned /api/v1<rule>."""
    def deco(fn):
        app.add_url_rule("/api" + rule, fn.__name__, fn, **options)
        app.add_url_rule("/api/v1" + rule, fn.__name__ + "_v1", fn, **options)
        return fn
    return deco


def _image_to_b64(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_to_temp_file(b64_data: str, suffix: str = ".png") -> Path:
    """Writes decoded image bytes to a temp file, since `aide.pipeline`'s
    functions load from a path (they need the raw file bytes for the
    image-bound key derivation - see aide/crypto.py)."""
    try:
        raw = base64.b64decode(b64_data, validate=True)
    except (ValueError, TypeError) as exc:
        raise ApiError(f"invalid base64 image data: {exc}") from exc
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    Path(path).write_bytes(raw)
    return Path(path)


def _upload_to_temp_file(file_storage) -> Path:
    suffix = Path(file_storage.filename or "upload.png").suffix or ".png"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    file_storage.save(path)
    return Path(path)


def _json_body() -> dict:
    payload = request.get_json(force=True, silent=True)
    if not isinstance(payload, dict):
        raise ApiError("request body must be a JSON object")
    return payload


def _profile_arg(value: str | None) -> str:
    value = value or DEFAULT_PROFILE
    if value not in PROFILES:
        raise ApiError(f"unknown profile {value!r}; choose from {sorted(PROFILES)}")
    return value


def _mode_arg(value: str | None, allowed=("sidecar", "blind")) -> str:
    value = value or "sidecar"
    if value not in allowed:
        raise ApiError(f"mode must be one of {list(allowed)}")
    return value


# ---------------------------------------------------------------- pages

@app.route("/")
def index():
    return render_template("index.html", version=__version__)


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "version": __version__,
                    "model": (MODEL_DIR / "weights.json").exists(),
                    "detector": get_detector() is not None})


# ---------------------------------------------------------------- demo

@api_route("/demo/covers")
def api_demo_covers():
    """Bundled cover images for the console's one-click Live Demo, so a
    presentation never depends on finding a file on the laptop."""
    covers = []
    for path in sorted(DEMO_DIR.glob("*.png")):
        with Image.open(path) as im:
            w, h = im.size
            color = im.mode != "L"
        covers.append({"id": path.stem, "label": DEMO_COVER_LABELS.get(path.stem, path.stem),
                       "url": f"/static/demo/{path.name}", "width": w, "height": h, "color": color})
    return jsonify(covers)


# ---------------------------------------------------------------- operator

@api_route("/profiles")
def api_profiles():
    return jsonify({"default": DEFAULT_PROFILE,
                    "profiles": {n: p.to_dict() for n, p in PROFILES.items()}})


@api_route("/capacity", methods=["POST"])
def api_capacity():
    cover_file = request.files.get("cover")
    if not cover_file:
        raise ApiError("cover is required")
    tmp_path = _upload_to_temp_file(cover_file)
    try:
        h, w = pipeline.load_image(tmp_path).y.shape
        side = pipeline.capacity_report(tmp_path, get_model())
        return jsonify({"width": w, "height": h, "profiles": {
            n: {"sidecar": side[n]["max_message_bytes"],
                "sidecar_most_robust_layout": side[n]["max_bytes_at_preferred_bpp_cap"],
                "blind": blind.max_message_bytes(h, w, n)} for n in PROFILES}})
    finally:
        tmp_path.unlink(missing_ok=True)


@api_route("/embed", methods=["POST"])
def api_embed():
    cover_file = request.files.get("cover")
    message = request.form.get("message", "")
    secret = request.form.get("secret", "")
    if not cover_file or not message or not secret:
        raise ApiError("cover, message, and secret are all required")
    profile = _profile_arg(request.form.get("profile"))
    mode = _mode_arg(request.form.get("embed_mode"))

    tmp_path = _upload_to_temp_file(cover_file)
    try:
        model = get_model()
        try:
            if mode == "blind":
                result = blind.embed(tmp_path, message.encode(), secret.encode(), model, profile=profile)
            else:
                result = pipeline.embed(tmp_path, message.encode(), secret.encode(), model, profile=profile)
        except pipeline.AideCapacityError as exc:
            raise ApiError(str(exc)) from exc

        cover_arr = np.array(Image.open(tmp_path).convert(result.mode))
        stego_img = result.compose()
        stego_arr = np.array(stego_img)

        out = {
            "stego_b64": _image_to_b64(stego_img),
            "mode": result.mode,          # color mode ("L"/"RGB"), kept for API compatibility
            "embed_mode": mode,
            "profile": profile,
            "message_bytes": len(message.encode()),
            "psnr_db": metrics.psnr(cover_arr, stego_arr),
            "ssim": metrics.ssim(cover_arr, stego_arr),
        }
        if mode == "blind":
            out |= {"sidecar": None, "blocks_used": result.blocks_used,
                    "payload_bytes": (result.header_bits + result.payload_bits) // 8,
                    "capacity_used_pct": round(100 * (result.header_bits + result.payload_bits)
                                               / max(1, result.capacity_bits), 1),
                    "method_breakdown": {"DCT": 100.0}}
        else:
            method_bits = Counter()
            for e in result.sidecar.embedding_map:
                method_bits[e.method] += e.bit_end - e.bit_start
            total_bits = sum(method_bits.values()) or 1
            out |= {"sidecar": result.sidecar.to_json(),
                    "blocks_used": len(result.sidecar.embedding_map),
                    "payload_bytes": total_bits // 8,
                    "ecc": result.sidecar.ecc_params.to_dict(),
                    "method_breakdown": {m: round(100 * b / total_bits, 1) for m, b in method_bits.items()}}
        return jsonify(out)
    finally:
        tmp_path.unlink(missing_ok=True)


@api_route("/attack", methods=["POST"])
def api_attack():
    payload = _json_body()
    stego_b64 = payload.get("stego_b64")
    attack_type = payload.get("type")
    params = payload.get("params") or {}
    if not stego_b64 or attack_type not in attacks.ATTACKS:
        raise ApiError("stego_b64 and a valid attack type are required")
    if not isinstance(params, dict):
        raise ApiError("params must be a JSON object")

    tmp_path = _b64_to_temp_file(stego_b64)
    try:
        loaded = pipeline.load_image(tmp_path)
        try:
            attacked_y = attacks.ATTACKS[attack_type](loaded.y, **params)
        except (TypeError, ValueError) as exc:
            raise ApiError(f"bad attack parameters: {exc}") from exc
        attacked_img = pipeline.compose_image(attacked_y, loaded.mode, loaded.chroma)
        return jsonify({"attacked_b64": _image_to_b64(attacked_img)})
    finally:
        tmp_path.unlink(missing_ok=True)


@api_route("/recover", methods=["POST"])
def api_recover():
    payload = _json_body()
    image_b64 = payload.get("image_b64")
    sidecar_json = payload.get("sidecar")
    secret = payload.get("secret", "")
    mode = _mode_arg(payload.get("embed_mode") or ("sidecar" if sidecar_json else "blind"))
    if not image_b64 or not secret or (mode == "sidecar" and not sidecar_json):
        raise ApiError("image_b64 and secret are required (plus sidecar in sidecar mode)")

    tmp_path = _b64_to_temp_file(image_b64)
    try:
        y = pipeline.load_image(tmp_path).y
        if mode == "blind":
            result = blind.recover(y, secret.encode())
            return jsonify({
                "success": result.success, "embed_mode": "blind", "profile": result.profile,
                "message": result.plaintext.decode(errors="replace") if result.plaintext else None,
                "error": result.error, "corrected_bytes": result.corrected_bytes,
                "gmd_codewords": result.gmd_codewords, "erasure_count": 0, "total_bytes": None,
            })
        try:
            sidecar = pipeline.Sidecar.from_json(sidecar_json)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ApiError(f"invalid sidecar: {exc}") from exc
        result = pipeline.recover(y, sidecar, secret.encode())
        return jsonify({
            "success": result.success, "embed_mode": "sidecar", "profile": sidecar.profile.name,
            "message": result.plaintext.decode(errors="replace") if result.plaintext else None,
            "error": result.error,
            "erasure_count": result.erasure_count,
            "total_bytes": result.total_bytes,
            "corrected_bytes": result.corrected_bytes,
            "gmd_codewords": result.gmd_codewords,
        })
    finally:
        tmp_path.unlink(missing_ok=True)


@api_route("/attack-types")
def api_attack_types():
    return jsonify(sorted(attacks.ATTACKS))


@api_route("/scenarios")
def api_scenarios():
    from aide.benchmark import SCENARIOS
    return jsonify([{"name": s.name, "attack": s.attack, "params": s.params} for s in SCENARIOS])


@api_route("/sweep", methods=["POST"])
def api_sweep():
    """Run every benchmark attack scenario against one stego image and
    report which ones the message survives - the Attack Lab's one-click
    robustness profile of a single embedding."""
    from aide.benchmark import SCENARIOS
    payload = _json_body()
    image_b64 = payload.get("image_b64")
    sidecar_json = payload.get("sidecar")
    secret = payload.get("secret", "")
    mode = _mode_arg(payload.get("embed_mode") or ("sidecar" if sidecar_json else "blind"))
    if not image_b64 or not secret or (mode == "sidecar" and not sidecar_json):
        raise ApiError("image_b64 and secret are required (plus sidecar in sidecar mode)")
    sidecar = None
    if mode == "sidecar":
        try:
            sidecar = pipeline.Sidecar.from_json(sidecar_json)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ApiError(f"invalid sidecar: {exc}") from exc

    tmp_path = _b64_to_temp_file(image_b64)
    try:
        y = pipeline.load_image(tmp_path).y
    finally:
        tmp_path.unlink(missing_ok=True)

    def _recover(img):
        if mode == "blind":
            return blind.recover(img, secret.encode())
        return pipeline.recover(img, sidecar, secret.encode())

    reference = _recover(y)
    if not reference.success:
        raise ApiError(f"cannot recover the unattacked image with this secret: {reference.error}")
    rows = []
    for i, sc in enumerate(SCENARIOS):
        attacked = sc.apply(y, seed=11 + i)
        t0 = time.perf_counter()
        rec = _recover(attacked)
        ok = rec.success and rec.plaintext == reference.plaintext
        rows.append({
            "name": sc.name, "attack": sc.attack, "params": sc.params, "success": ok,
            "ber": metrics.bit_error_rate(reference.plaintext, rec.plaintext or b""),
            "corrected_bytes": rec.corrected_bytes, "ms": round((time.perf_counter() - t0) * 1000, 1),
            "psnr_vs_stego": None if sc.attack is None else round(metrics.psnr(y, attacked), 2),
            "error": None if ok else (rec.error or "plaintext mismatch"),
        })
    passed = sum(r["success"] for r in rows)
    return jsonify({"embed_mode": mode, "passed": passed, "total": len(rows), "results": rows})


# ---------------------------------------------------------------- analyst

def _detect_gray(gray: np.ndarray) -> dict:
    report = steganalysis.analyze(gray)
    out = {
        "verdict": report.verdict,
        "blind_verdict": report.verdict,
        "suspicion_score": report.suspicion_score,
        "chi_square_p": report.chi_square_p,
        "lsb_correlation": report.lsb_correlation,
        "lsb_bit_balance": report.lsb_bit_balance,
        "spa_rate": report.spa_rate,
        "rs_rate": report.rs_rate,
        "estimated_payload_bytes": report.estimated_payload_bytes,
        "row_band_profile": report.row_band_profile,
        "spatial_grid": report.spatial_grid,
        "ml": None,
    }
    detector = get_detector()
    if detector is not None:
        ml = detector.analyze(gray)
        out["ml"] = {"verdict": ml.verdict, "probability": ml.probability,
                     "in_distribution": ml.in_distribution, "note": ml.note}
        rank = {"clean": 0, "suspicious": 1, "likely_steganographic": 2}
        if rank[ml.verdict] > rank[out["verdict"]]:
            out["verdict"] = ml.verdict
    return out


@api_route("/detect", methods=["POST"])
def api_detect():
    image_file = request.files.get("image")
    if not image_file:
        raise ApiError("image is required")

    tmp_path = _upload_to_temp_file(image_file)
    try:
        gray = pipeline.load_image(tmp_path).y
        out = _detect_gray(gray)
        # echoed back so the client can request the heatmap overlay
        # (/api/visualize/suspicion-heatmap) without re-uploading
        out["image_b64"] = _image_to_b64(Image.fromarray(gray, mode="L"))
        return jsonify(out)
    finally:
        tmp_path.unlink(missing_ok=True)


@api_route("/detect/batch", methods=["POST"])
def api_detect_batch():
    """Screen many images in one request (border/port media screening).
    Returns one compact row per file, most suspicious first."""
    files = request.files.getlist("images")
    if not files:
        raise ApiError("one or more files under 'images' are required")
    if len(files) > MAX_BATCH:
        raise ApiError(f"at most {MAX_BATCH} images per batch")
    rows = []
    for f in files:
        tmp_path = _upload_to_temp_file(f)
        try:
            gray = pipeline.load_image(tmp_path).y
            r = _detect_gray(gray)
            rows.append({"name": f.filename, "verdict": r["verdict"],
                         "suspicion_score": r["suspicion_score"], "spa_rate": r["spa_rate"],
                         "rs_rate": r["rs_rate"], "chi_square_p": r["chi_square_p"],
                         "ml_probability": r["ml"]["probability"] if r["ml"] else None,
                         "ml_in_distribution": r["ml"]["in_distribution"] if r["ml"] else None,
                         "width": int(gray.shape[1]), "height": int(gray.shape[0])})
        except (UnidentifiedImageError, OSError) as exc:
            rows.append({"name": f.filename, "verdict": "error", "error": str(exc)})
        finally:
            tmp_path.unlink(missing_ok=True)
    rank = {"likely_steganographic": 3, "suspicious": 2, "clean": 1, "error": 0}
    rows.sort(key=lambda r: (-rank[r["verdict"]], -(r.get("ml_probability") or 0), -(r.get("suspicion_score") or 0)))
    summary = Counter(r["verdict"] for r in rows)
    return jsonify({"scanned": len(rows), "summary": dict(summary), "results": rows})


# ---------------------------------------------------------------- visualize

@api_route("/visualize/embedding-map", methods=["POST"])
def api_visualize_embedding_map():
    """Color every block the AIDE engine actually used by which method
    (LSB/DCT/DWT) embedded into it - the most direct visual evidence of
    the adaptive, per-block, multi-method policy. See aide/visualize.py."""
    payload = _json_body()
    image_b64 = payload.get("image_b64")
    sidecar_json = payload.get("sidecar")
    if not image_b64 or not sidecar_json:
        raise ApiError("image_b64 and sidecar are required")

    tmp_path = _b64_to_temp_file(image_b64)
    try:
        gray = pipeline.load_image(tmp_path).y
        sidecar = pipeline.Sidecar.from_json(sidecar_json)
        blocks = partition_blocks(sidecar.height, sidecar.width, sidecar.block_size)
        overlay = visualize.embedding_map_overlay(gray, blocks, sidecar.embedding_map)
        return jsonify({"overlay_b64": _image_to_b64(overlay)})
    finally:
        tmp_path.unlink(missing_ok=True)


@api_route("/visualize/diff", methods=["POST"])
def api_visualize_diff():
    """Real, amplified |cover - stego| heatmap - honest about the fact
    that pixels did change, rather than only citing a PSNR number."""
    payload = _json_body()
    cover_b64 = payload.get("cover_b64")
    stego_b64 = payload.get("stego_b64")
    if not cover_b64 or not stego_b64:
        raise ApiError("cover_b64 and stego_b64 are required")

    cover_path = _b64_to_temp_file(cover_b64)
    stego_path = _b64_to_temp_file(stego_b64)
    try:
        cover_gray = pipeline.load_image(cover_path).y
        stego_gray = pipeline.load_image(stego_path).y
        if cover_gray.shape != stego_gray.shape:
            raise ApiError("cover and stego dimensions do not match")
        heat = visualize.diff_heatmap(cover_gray, stego_gray)
        return jsonify({"heatmap_b64": _image_to_b64(heat)})
    finally:
        cover_path.unlink(missing_ok=True)
        stego_path.unlink(missing_ok=True)


@api_route("/visualize/tamper", methods=["POST"])
def api_visualize_tamper():
    """Which blocks does the receiver actually trust after an attack?
    Red = erasure (fed to Reed-Solomon as a known-bad byte range), amber =
    hash-mismatch left to blind error correction, untouched = trusted."""
    payload = _json_body()
    image_b64 = payload.get("image_b64")
    sidecar_json = payload.get("sidecar")
    if not image_b64 or not sidecar_json:
        raise ApiError("image_b64 and sidecar are required")

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


@api_route("/visualize/suspicion-heatmap", methods=["POST"])
def api_visualize_suspicion_heatmap():
    """Where in an arbitrary intercepted image does blind steganalysis
    look most suspicious? Diagnostic only - see aide/steganalysis.py."""
    payload = _json_body()
    image_b64 = payload.get("image_b64")
    grid = payload.get("spatial_grid")
    if not image_b64 or not grid:
        raise ApiError("image_b64 and spatial_grid are required")

    tmp_path = _b64_to_temp_file(image_b64)
    try:
        gray = pipeline.load_image(tmp_path).y
        overlay = visualize.suspicion_heatmap_overlay(gray, np.array(grid))
        return jsonify({"overlay_b64": _image_to_b64(overlay)})
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------- insights

@api_route("/model-info")
def api_model_info():
    """The learned decision engine's parameters, for the Model Insights
    panel - the report's own 'explainable, auditable, not a black box'
    novelty claim, made concrete."""
    model = get_model()
    detector = get_detector()
    return jsonify({
        "version": __version__,
        "weights": model.weights.as_dict(),
        "thresholds": {
            "tau_min": model.thresholds.tau_min,
            "t_low": model.thresholds.t_low,
            "e_high": model.thresholds.e_high,
        },
        "meta": model.meta,
        "has_trained_classifier": model.method_clf is not None,
        "detector": detector.meta if detector is not None else None,
    })


@api_route("/benchmark")
def api_benchmark():
    """Summary of the most recent `aide benchmark` run (not run per request
    - a full benchmark takes minutes)."""
    if not BENCHMARK_JSON.exists():
        return jsonify({"available": False,
                        "hint": "run `aide benchmark --out reports/benchmark` to generate one"})
    data = json.loads(BENCHMARK_JSON.read_text())
    return jsonify({"available": True, "config": data["config"], "summary": data["summary"]})


@app.route("/api/v1/openapi.json")
def openapi():
    def op(summary, body=None, multipart=None):
        o = {"summary": summary, "responses": {"200": {"description": "OK"},
                                               "400": {"description": "Invalid request"}}}
        if body:
            o["requestBody"] = {"content": {"application/json": {"schema": {
                "type": "object", "properties": {k: {"type": t} for k, t in body.items()}}}}}
        if multipart:
            o["requestBody"] = {"content": {"multipart/form-data": {"schema": {
                "type": "object", "properties": {k: ({"type": "string", "format": "binary"} if t == "file"
                                                     else {"type": t}) for k, t in multipart.items()}}}}}
        return o
    spec = {
        "openapi": "3.0.3",
        "info": {"title": "AIDE API", "version": __version__,
                 "description": "Adaptive steganography (embed/recover) and steganalysis (detect). "
                                "If the server sets AIDE_API_KEY, send it as X-API-Key."},
        "components": {"securitySchemes": {"apiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"}}},
        "paths": {
            "/api/v1/profiles": {"get": op("List robustness profiles")},
            "/api/v1/capacity": {"post": op("Max message bytes per profile/mode", multipart={"cover": "file"})},
            "/api/v1/embed": {"post": op("Embed a message", multipart={
                "cover": "file", "message": "string", "secret": "string",
                "profile": "string", "embed_mode": "string"})},
            "/api/v1/attack": {"post": op("Apply a simulated attack", body={
                "stego_b64": "string", "type": "string", "params": "object"})},
            "/api/v1/recover": {"post": op("Recover a message", body={
                "image_b64": "string", "secret": "string", "sidecar": "string", "embed_mode": "string"})},
            "/api/v1/detect": {"post": op("Steganalysis of one image", multipart={"image": "file"})},
            "/api/v1/detect/batch": {"post": op(f"Screen up to {MAX_BATCH} images", multipart={"images": "file"})},
            "/api/v1/attack-types": {"get": op("List attack simulations")},
            "/api/v1/scenarios": {"get": op("List benchmark attack scenarios")},
            "/api/v1/sweep": {"post": op("Run every attack scenario against one stego image", body={
                "image_b64": "string", "secret": "string", "sidecar": "string", "embed_mode": "string"})},
            "/api/v1/demo/covers": {"get": op("Bundled demo cover images")},
            "/api/v1/model-info": {"get": op("Decision-engine and detector parameters")},
            "/api/v1/benchmark": {"get": op("Latest benchmark summary")},
        },
        "security": [{"apiKey": []}],
    }
    return jsonify(spec)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    debug = os.environ.get("AIDE_WEB_DEBUG", "0") == "1"
    logging.basicConfig(level=logging.INFO if debug else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not debug:
        # Quiet, presentation-friendly output: no reloader, no debugger, and
        # no per-request access log. Set AIDE_WEB_DEBUG=1 for auto-reload,
        # tracebacks and request logging.
        logging.getLogger("werkzeug").setLevel(logging.ERROR)

    model_status = "found" if (MODEL_DIR / "weights.json").exists() else "MISSING - run: aide train"
    print(f"AIDE {__version__} - model directory: {MODEL_DIR} ({model_status})")
    print(f"AIDE Command Console ready -> http://127.0.0.1:{port}  (Ctrl+C to stop)")
    app.run(debug=debug, use_reloader=debug, port=port)
