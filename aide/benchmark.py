"""Reproducible, multi-image benchmark of the full AIDE pipeline.

Answers the report's evaluation questions (section 6.1, objectives O4/O6)
with measured numbers over many real cover images instead of one demo
image: for every (robustness profile x attack scenario), what fraction of
messages are recovered *bit-exactly*, at what BER, PSNR, SSIM and speed?

    aide benchmark --images data/samples --num-images 40 --out reports/bench

writes `benchmark.json` (every individual trial) and `benchmark.md` (the
summary table). Trials are independent, so they run in a process pool.

Cover images that lack enough capacity for the message under a profile
are reported separately as `capacity_skipped` rather than counted as
recovery failures - a capacity error is a clear up-front refusal, not a
silent corruption, and conflating the two would misstate robustness.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import attacks, metrics, pipeline
from .model import AIDEModel
from .profiles import PROFILES

DEFAULT_MESSAGE = (
    b"CLASSIFIED SITREP: Convoy Alpha departs staging area 0600Z, route Bravo-7. "
    b"Escort confirmed, rendezvous at checkpoint Delta. Air support standby channel 4. "
)
DEFAULT_SECRET = b"aide-benchmark-shared-secret"


@dataclass(frozen=True)
class Scenario:
    name: str
    attack: str | None
    params: dict = field(default_factory=dict)

    def apply(self, gray: np.ndarray, seed: int) -> np.ndarray:
        if self.attack is None:
            return gray
        kwargs = dict(self.params)
        if self.attack in attacks.SEEDED_ATTACKS:
            kwargs.setdefault("seed", seed)
        return attacks.ATTACKS[self.attack](gray, **kwargs)


# The report's six attack categories at graded strengths, plus a lossless
# baseline. `O4` marks the report's explicit robustness targets.
SCENARIOS: list[Scenario] = [
    Scenario("none", None),
    Scenario("jpeg_q90", "compression", {"quality": 90}),
    Scenario("jpeg_q70", "compression", {"quality": 70}),
    Scenario("jpeg_q50 (O4)", "compression", {"quality": 50}),
    Scenario("jpeg_q30", "compression", {"quality": 30}),
    Scenario("noise_s5", "gaussian_noise", {"sigma": 5}),
    Scenario("noise_s10", "gaussian_noise", {"sigma": 10}),
    Scenario("noise_s20 (O4)", "gaussian_noise", {"sigma": 20}),
    Scenario("salt_pepper_1%", "salt_pepper", {"amount": 0.01}),
    Scenario("median_3x3", "median_filter", {"ksize": 3}),
    Scenario("gaussian_blur", "gaussian_filter", {"ksize": 3, "sigma": 1.0}),
    Scenario("scale_0.75", "scaling", {"factor": 0.75}),
    Scenario("scale_0.5", "scaling", {"factor": 0.5}),
    Scenario("splice_2%", "splicing", {"fraction": 0.02}),
    Scenario("splice_5%", "splicing", {"fraction": 0.05}),
    Scenario("crop_10%", "cropping", {"fraction": 0.10}),
]
SCENARIOS_BY_NAME = {s.name: s for s in SCENARIOS}


@dataclass
class Trial:
    image: str
    profile: str
    scenario: str
    status: str                 # "recovered" | "failed" | "capacity_skipped"
    ber: float | None = None
    psnr_db: float | None = None
    ssim: float | None = None
    embed_s: float | None = None
    recover_s: float | None = None
    corrected_bytes: int | None = None
    gmd_codewords: int | None = None
    error: str | None = None


_WORKER_MODEL: AIDEModel | None = None


def _init_worker(model_dir: str | None) -> None:
    global _WORKER_MODEL
    try:
        _WORKER_MODEL = AIDEModel.load(Path(model_dir)) if model_dir else AIDEModel.default()
    except FileNotFoundError:
        _WORKER_MODEL = AIDEModel.default()


def _run_image(job: tuple) -> list[dict]:
    image_path, profile_name, scenario_names, message, secret, seed = job
    model = _WORKER_MODEL or AIDEModel.default()
    name = Path(image_path).name
    out: list[dict] = []
    t0 = time.perf_counter()
    try:
        res = pipeline.embed(Path(image_path), message, secret, model, profile=profile_name)
    except pipeline.AideCapacityError as exc:
        return [asdict(Trial(name, profile_name, sc, "capacity_skipped", error=str(exc)[:160]))
                for sc in scenario_names]
    embed_s = time.perf_counter() - t0
    cover = np.array(Image.open(image_path).convert(res.mode))
    stego_full = np.array(res.compose())
    psnr = metrics.psnr(cover, stego_full)
    ssim = metrics.ssim(cover, stego_full)

    for sc_name in scenario_names:
        sc = SCENARIOS_BY_NAME[sc_name]
        attacked = sc.apply(res.stego, seed)
        t1 = time.perf_counter()
        rec = pipeline.recover(attacked, res.sidecar, secret)
        rec_s = time.perf_counter() - t1
        ok = rec.success and rec.plaintext == message
        out.append(asdict(Trial(
            name, profile_name, sc_name, "recovered" if ok else "failed",
            ber=metrics.bit_error_rate(message, rec.plaintext or b""),
            psnr_db=psnr, ssim=ssim, embed_s=embed_s, recover_s=rec_s,
            corrected_bytes=rec.corrected_bytes, gmd_codewords=rec.gmd_codewords,
            error=None if ok else (rec.error or "plaintext mismatch")[:160],
        )))
    return out


def run(image_paths: list[Path], profiles: list[str] | None = None,
        scenarios: list[str] | None = None, message: bytes = DEFAULT_MESSAGE,
        secret: bytes = DEFAULT_SECRET, model_dir: Path | None = Path("data/models"),
        workers: int | None = None, seed: int = 7) -> dict:
    profiles = profiles or list(PROFILES)
    scenario_names = scenarios or [s.name for s in SCENARIOS]
    for sc in scenario_names:
        if sc not in SCENARIOS_BY_NAME:
            raise ValueError(f"unknown scenario {sc!r}; choose from {list(SCENARIOS_BY_NAME)}")
    jobs = [(str(p), prof, scenario_names, message, secret, seed + i)
            for prof in profiles for i, p in enumerate(image_paths)]

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.perf_counter()
    trials: list[dict] = []
    if workers == 1:
        _init_worker(str(model_dir) if model_dir else None)
        for j in jobs:
            trials.extend(_run_image(j))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(str(model_dir) if model_dir else None,)) as ex:
            for rows in ex.map(_run_image, jobs):
                trials.extend(rows)

    return {
        "config": {
            "num_images": len(image_paths), "profiles": profiles, "scenarios": scenario_names,
            "message_bytes": len(message), "seed": seed, "workers": workers,
            "wall_seconds": round(time.perf_counter() - t0, 1),
        },
        "summary": summarize(trials, profiles, scenario_names),
        "trials": trials,
    }


def _mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(statistics.fmean(xs)) if xs else None


def summarize(trials: list[dict], profiles: list[str], scenario_names: list[str]) -> dict:
    summary: dict = {}
    for prof in profiles:
        rows = [t for t in trials if t["profile"] == prof]
        embedded = [t for t in rows if t["status"] != "capacity_skipped"]
        per_sc = {}
        for sc in scenario_names:
            r = [t for t in embedded if t["scenario"] == sc]
            n_ok = sum(t["status"] == "recovered" for t in r)
            per_sc[sc] = {
                "trials": len(r), "recovered": n_ok,
                "recovery_rate": (n_ok / len(r)) if r else None,
                "mean_ber": _mean(t["ber"] for t in r),
                "mean_recover_s": _mean(t["recover_s"] for t in r),
            }
        images_embedded = {t["image"] for t in embedded}
        images_skipped = {t["image"] for t in rows if t["status"] == "capacity_skipped"}
        first = [t for t in embedded if t["scenario"] == scenario_names[0]]
        summary[prof] = {
            "images_embedded": len(images_embedded),
            "capacity_skipped": len(images_skipped),
            "mean_psnr_db": _mean(t["psnr_db"] for t in first),
            "mean_ssim": _mean(t["ssim"] for t in first),
            "mean_embed_s": _mean(t["embed_s"] for t in first),
            "scenarios": per_sc,
        }
    return summary


def to_markdown(result: dict) -> str:
    cfg, summary = result["config"], result["summary"]
    profiles = cfg["profiles"]
    lines = [
        "# AIDE benchmark",
        "",
        f"{cfg['num_images']} cover images x {len(profiles)} profiles x "
        f"{len(cfg['scenarios'])} scenarios, {cfg['message_bytes']}-byte message, "
        f"seed {cfg['seed']} ({cfg['wall_seconds']} s wall).",
        "",
        "Recovery rate = fraction of embedded covers whose message came back bit-exact "
        "(AES-GCM verified).",
        "",
        "| Scenario | " + " | ".join(profiles) + " |",
        "|---|" + "---:|" * len(profiles),
    ]
    for sc in cfg["scenarios"]:
        cells = []
        for prof in profiles:
            s = summary[prof]["scenarios"][sc]
            cells.append("-" if s["recovery_rate"] is None
                         else f"{100 * s['recovery_rate']:.0f}% ({s['recovered']}/{s['trials']})")
        lines.append(f"| {sc} | " + " | ".join(cells) + " |")

    def row(label, key, fmt):
        vals = [summary[p][key] for p in profiles]
        return f"| {label} | " + " | ".join("-" if v is None else fmt.format(v) for v in vals) + " |"

    lines += [
        row("**mean PSNR (dB)**", "mean_psnr_db", "{:.2f}"),
        row("**mean SSIM**", "mean_ssim", "{:.4f}"),
        row("**mean embed time (s)**", "mean_embed_s", "{:.3f}"),
        "| **covers with capacity** | " + " | ".join(
            f"{summary[p]['images_embedded']}/{summary[p]['images_embedded'] + summary[p]['capacity_skipped']}"
            for p in profiles) + " |",
        "",
    ]
    return "\n".join(lines)


def write_report(result: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jp, mp = out_dir / "benchmark.json", out_dir / "benchmark.md"
    jp.write_text(json.dumps(result, indent=2))
    mp.write_text(to_markdown(result))
    return jp, mp


def inference_speed(image_paths: list[Path], model: AIDEModel, model_dir: Path | None = None,
                    repeats: int = 3) -> dict:
    """Report objectives O1 (< 5 ms/block decision inference on CPU) and
    O7 (< 10 MB model footprint; end-to-end speed). Measured, single
    process, on this machine - rerun on the deployment target (e.g. a
    Raspberry Pi 4) for numbers that apply there."""
    import resource
    import sys

    from .features import extract_image_features

    feat_ms, decide_ms, n_blocks = [], [], 0
    for p in image_paths:
        gray = np.array(Image.open(p).convert("L"))
        for _ in range(repeats):
            t0 = time.perf_counter()
            blocks, feats = extract_image_features(gray)
            t1 = time.perf_counter()
            model.decide(feats)
            t2 = time.perf_counter()
            feat_ms.append((t1 - t0) * 1000 / len(blocks))
            decide_ms.append((t2 - t1) * 1000 / len(blocks))
        n_blocks += len(blocks)

    # Single-block latency: the report's literal "per block" inference cost
    # (no batching benefit).
    gray = np.array(Image.open(image_paths[0]).convert("L"))
    blocks, feats = extract_image_features(gray)
    t0 = time.perf_counter()
    for i in range(min(200, len(feats))):
        model.decide(feats[i:i + 1])
    single_ms = (time.perf_counter() - t0) * 1000 / min(200, len(feats))

    embed_s, recover_s = [], []
    for p in image_paths:
        t0 = time.perf_counter()
        try:
            res = pipeline.embed(p, DEFAULT_MESSAGE, DEFAULT_SECRET, model)
        except pipeline.AideCapacityError:
            continue
        t1 = time.perf_counter()
        pipeline.recover(res.stego, res.sidecar, DEFAULT_SECRET)
        embed_s.append(t1 - t0)
        recover_s.append(time.perf_counter() - t1)

    model_bytes = None
    if model_dir is not None:
        files = [Path(model_dir) / f for f in ("aide_model.joblib", "weights.json")]
        model_bytes = sum(f.stat().st_size for f in files if f.exists())
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024

    return {
        "images": len(image_paths),
        "blocks_measured": n_blocks,
        "feature_extraction_ms_per_block": _mean(feat_ms),
        "decision_ms_per_block_batched": _mean(decide_ms),
        "decision_ms_single_block": single_ms,
        "o1_target_ms_per_block": 5.0,
        "o1_met": (_mean(feat_ms) or 0) + single_ms < 5.0,
        "embed_s_per_image": _mean(embed_s),
        "recover_s_per_image": _mean(recover_s),
        "model_size_kb": round(model_bytes / 1024, 1) if model_bytes else None,
        "o7_target_model_mb": 10.0,
        "process_peak_rss_mb": round(peak_mb, 1),
        "classifier": model.meta.get("classifier", "tree"),
    }
