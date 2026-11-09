"""AIDE command-line interface.

Operator (secure communication):
    aide embed      embed a secret message (sidecar or blind mode, choice of profile)
    aide extract    recover it (auto-detects sidecar vs blind mode)
    aide capacity   how much a cover can carry under each profile/mode
    aide attack     apply a simulated attack
    aide evaluate   embed + every attack + measured PSNR/SSIM/BER/timing

Analyst (threat detection):
    aide detect     blind steganalysis of one image or a whole folder

Engineering:
    aide benchmark       multi-image recovery/quality benchmark, or inference speed
    aide train           train the AIDE decision engine (tree or tiny MLP)
    aide train-detector  train the ML steganalyser
    aide profiles        list robustness profiles
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__, attacks, blind, metrics, pipeline, steganalysis
from .model import AIDEModel
from .profiles import DEFAULT_PROFILE, PROFILES
from .train import IMAGE_EXTS
from .train import main as train_main

log = logging.getLogger("aide")

DEFAULT_EVAL_MESSAGE = (
    "CLASSIFIED SITREP: Convoy Alpha departs staging area 0600Z, route Bravo-7. "
    "Escort confirmed, rendezvous at checkpoint Delta. Air support standby channel 4. "
) * 3


def _load_model(model_dir: Path) -> AIDEModel:
    try:
        return AIDEModel.load(model_dir)
    except FileNotFoundError:
        log.warning("no trained model at %s, using formula-only default "
                    "(run 'aide train' first for the real decision engine)", model_dir)
        return AIDEModel.default()


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _read_secret(args) -> bytes:
    if getattr(args, "secret_file", None):
        return Path(args.secret_file).read_bytes().strip()
    if args.secret is None:
        raise SystemExit("error: --secret or --secret-file is required")
    return args.secret.encode()


def _quality(cover_path: Path, mode: str, composed: Image.Image) -> dict:
    # Compare the full original file against the full composed stego (RGB
    # recombined with untouched chroma, if applicable) - a fairer visual
    # quality report than comparing just the embedded Y/luma plane.
    cover_arr = np.array(Image.open(cover_path).convert(mode))
    stego_arr = np.array(composed)
    return {"psnr_db": metrics.psnr(cover_arr, stego_arr), "ssim": metrics.ssim(cover_arr, stego_arr)}


def cmd_embed(args: argparse.Namespace) -> int:
    model = _load_model(args.model)
    message = args.message.encode() if args.message else Path(args.message_file).read_bytes()
    secret = _read_secret(args)

    try:
        if args.mode == "blind":
            result = blind.embed(args.cover, message, secret, model, profile=args.profile)
        else:
            result = pipeline.embed(args.cover, message, secret, model, profile=args.profile)
    except pipeline.AideCapacityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    composed = result.compose()
    composed.save(args.out)
    if args.out.suffix.lower() in (".jpg", ".jpeg"):
        log.warning("saved as JPEG: the save itself is a lossy attack - fine for robust profiles, "
                    "but prefer PNG for lossless delivery")
    report = {"stego": str(args.out), "mode": args.mode, "profile": args.profile,
              "message_bytes": len(message), "color_mode": result.mode}
    if args.mode == "sidecar":
        sidecar_path = args.sidecar or Path(str(args.out) + ".aide.json")
        pipeline.save_sidecar(result.sidecar, sidecar_path)
        report |= {"sidecar": str(sidecar_path), "blocks_used": len(result.sidecar.embedding_map),
                   "ecc": result.sidecar.ecc_params.to_dict()}
    else:
        report |= {"blocks_used": result.blocks_used,
                   "bits_used": result.header_bits + result.payload_bits,
                   "capacity_bits": result.capacity_bits}
    report |= _quality(args.cover, result.mode, composed)
    _print(report)
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    stego = pipeline.load_image(args.stego).y
    secret = _read_secret(args)
    sidecar_path = args.sidecar or Path(str(args.stego) + ".aide.json")
    mode = args.mode
    if mode == "auto":
        mode = "sidecar" if Path(sidecar_path).exists() else "blind"

    if mode == "sidecar":
        result = pipeline.recover(stego, pipeline.load_sidecar(sidecar_path), secret)
        info = {"mode": "sidecar", "erasures": result.erasure_count, "total_bytes": result.total_bytes,
                "corrected_bytes": result.corrected_bytes, "gmd_codewords": result.gmd_codewords}
    else:
        result = blind.recover(stego, secret)
        info = {"mode": "blind", "profile": result.profile, "corrected_bytes": result.corrected_bytes,
                "gmd_codewords": result.gmd_codewords}

    if not result.success:
        _print({"success": False, "error": result.error} | info)
        return 1
    if args.out:
        Path(args.out).write_bytes(result.plaintext)
        _print({"success": True, "message_bytes": len(result.plaintext), "out": str(args.out)} | info)
    else:
        _print({"success": True, "message": result.plaintext.decode(errors="replace")} | info)
    return 0


def cmd_capacity(args: argparse.Namespace) -> int:
    """Largest message (bytes) each profile/mode can carry in this cover -
    computed exactly from one feature pass and the deterministic ECC plan."""
    model = _load_model(args.model)
    h, w = pipeline.load_image(args.cover).y.shape
    side = pipeline.capacity_report(args.cover, model)
    out = {"cover": str(args.cover), "size": [w, h], "profiles": {}}
    for name in PROFILES:
        out["profiles"][name] = {
            "sidecar_max_bytes": side[name]["max_message_bytes"],
            "sidecar_max_bytes_most_robust_layout": side[name]["max_bytes_at_preferred_bpp_cap"],
            "blind_max_bytes": blind.max_message_bytes(h, w, name),
            "description": PROFILES[name].description,
        }
    _print(out)
    return 0


def cmd_attack(args: argparse.Namespace) -> int:
    loaded = pipeline.load_image(args.stego)
    fn = attacks.ATTACKS[args.type]
    kwargs = json.loads(args.params) if args.params else {}
    attacked_y = fn(loaded.y, **kwargs)
    # Attacks operate on the luma plane only (see pipeline.py's module
    # docstring); recombine with the original chroma so a color stego
    # still comes back out as a valid, viewable color image.
    pipeline.save_image(attacked_y, loaded.mode, loaded.chroma, args.out)
    _print({"attacked": str(args.out), "type": args.type, "params": kwargs})
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    model = _load_model(args.model)
    message = args.message.encode()
    secret = _read_secret(args)

    with metrics.measure() as t:
        if args.mode == "blind":
            result = blind.embed(args.cover, message, secret, model, profile=args.profile)
        else:
            result = pipeline.embed(args.cover, message, secret, model, profile=args.profile)

    def _recover(img):
        if args.mode == "blind":
            return blind.recover(img, secret)
        return pipeline.recover(img, result.sidecar, secret)

    report = {"cover": str(args.cover), "mode": args.mode, "profile": args.profile,
              "message_bytes": len(message), "embed_time_s": t.elapsed_seconds,
              "color_mode": result.mode} | _quality(args.cover, result.mode, result.compose())
    report["attacks"] = {}
    rec = _recover(result.stego)
    report["attacks"]["none"] = {"success": rec.success, "exact_match": rec.success and rec.plaintext == message}
    for name, fn in attacks.ATTACKS.items():
        kwargs = {"seed": 1} if name in attacks.SEEDED_ATTACKS else {}
        attacked = fn(result.stego, **kwargs)
        with metrics.measure() as t2:
            rec = _recover(attacked)
        report["attacks"][name] = {
            "success": rec.success,
            "exact_match": rec.success and rec.plaintext == message,
            "ber": metrics.bit_error_rate(message, rec.plaintext) if rec.plaintext else 1.0,
            "corrected_bytes": rec.corrected_bytes,
            "recover_time_s": t2.elapsed_seconds,
            "error": rec.error,
        }
    _print(report)
    return 0


def _load_detector(model_dir: Path):
    from .mlsteg import DETECTOR_FILE, Steganalyzer
    if (Path(model_dir) / DETECTOR_FILE).exists():
        return Steganalyzer.load(model_dir)
    return None


VERDICT_RANK = {"clean": 0, "suspicious": 1, "likely_steganographic": 2}


def detect_image(path: Path, detector=None) -> dict:
    """Run every detector on one image; the overall verdict is the most
    severe of the blind (statistical) and trained (ML) verdicts."""
    gray = pipeline.load_image(path).y
    rep = steganalysis.analyze(gray)
    out = {
        "image": str(path),
        "verdict": rep.verdict,
        "blind": {
            "verdict": rep.verdict,
            "suspicion_score": rep.suspicion_score,
            "chi_square_p": rep.chi_square_p,
            "spa_rate": rep.spa_rate,
            "rs_rate": rep.rs_rate,
            "estimated_payload_bytes": rep.estimated_payload_bytes,
            "lsb_correlation": rep.lsb_correlation,
            "lsb_bit_balance": rep.lsb_bit_balance,
        },
    }
    if detector is not None:
        ml = detector.analyze(gray)
        out["ml"] = {"verdict": ml.verdict, "probability": ml.probability,
                     "in_distribution": ml.in_distribution,
                     "note": ml.note or "trained on AIDE-family DCT coefficient-pair embedding"}
        if VERDICT_RANK[ml.verdict] > VERDICT_RANK[out["verdict"]]:
            out["verdict"] = ml.verdict
    return out


def cmd_detect(args: argparse.Namespace) -> int:
    """Blind steganalysis: no sidecar, no secret, no assumption about how
    (or whether) the image was steganographically modified - the defence/
    intelligence "is this intercepted image smuggling hidden data" use
    case. Pass a directory to screen every image in it."""
    detector = None if args.no_ml else _load_detector(args.model)
    target = Path(args.image)
    if target.is_dir():
        pattern = "**/*" if args.recursive else "*"
        paths = sorted(p for p in target.glob(pattern) if p.suffix.lower() in IMAGE_EXTS)
        rows = []
        for p in paths:
            try:
                rows.append(detect_image(p, detector))
            except Exception as exc:  # one unreadable file must not stop a screening run
                rows.append({"image": str(p), "verdict": "error", "error": str(exc)})
        flagged = [r for r in rows if r["verdict"] not in ("clean", "error")]
        summary = {"scanned": len(rows), "flagged": len(flagged),
                   "by_verdict": {v: sum(r["verdict"] == v for r in rows)
                                  for v in ("clean", "suspicious", "likely_steganographic", "error")}}
        if args.csv:
            with open(args.csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["image", "verdict", "suspicion_score", "spa_rate", "rs_rate",
                            "chi_square_p", "ml_probability"])
                for r in rows:
                    b = r.get("blind", {})
                    w.writerow([r["image"], r["verdict"], b.get("suspicion_score"), b.get("spa_rate"),
                                b.get("rs_rate"), b.get("chi_square_p"), r.get("ml", {}).get("probability")])
            summary["csv"] = str(args.csv)
        _print({"summary": summary, "flagged": flagged if not args.all else rows})
        return 1 if flagged else 0

    out = detect_image(target, detector)
    _print(out)
    return 1 if out["verdict"] != "clean" else 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from . import benchmark
    from .train import list_images

    paths = list_images(args.images)[: args.num_images]
    if not paths:
        print(f"error: no images in {args.images}", file=sys.stderr)
        return 1
    if args.inference:
        _print(benchmark.inference_speed(paths, _load_model(args.model), args.model))
        return 0
    result = benchmark.run(paths, profiles=args.profiles, scenarios=args.scenarios,
                           model_dir=args.model, workers=args.workers, seed=args.seed)
    jp, mp = benchmark.write_report(result, args.out)
    print(benchmark.to_markdown(result))
    print(f"\nwrote {jp} and {mp}")
    return 0


def cmd_train_detector(args: argparse.Namespace) -> int:
    from .mlsteg import train_detector
    from .train import list_images

    paths = list_images(args.images)[: args.max_images]
    _print(train_detector(paths, args.out, args.model, workers=args.workers, seed=args.seed))
    return 0


def cmd_profiles(args: argparse.Namespace) -> int:
    _print({n: p.to_dict() | {"default": n == DEFAULT_PROFILE} for n, p in PROFILES.items()})
    return 0


def _add_secret(p: argparse.ArgumentParser, required_note: str = "") -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--secret", type=str, help="shared secret" + required_note)
    g.add_argument("--secret-file", type=Path,
                   help="read the shared secret from a file (keeps it out of shell history)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aide", description="AIDE steganography framework")
    p.add_argument("--version", action="version", version=f"aide {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging to stderr")
    sub = p.add_subparsers(dest="command", required=True)
    profiles = sorted(PROFILES)

    pe = sub.add_parser("embed", help="embed a secret message into a cover image")
    pe.add_argument("cover", type=Path)
    pe.add_argument("out", type=Path)
    g = pe.add_mutually_exclusive_group(required=True)
    g.add_argument("--message", type=str)
    g.add_argument("--message-file", type=Path)
    _add_secret(pe)
    pe.add_argument("--profile", choices=profiles, default=DEFAULT_PROFILE,
                    help="robustness profile (see 'aide profiles')")
    pe.add_argument("--mode", choices=("sidecar", "blind"), default="sidecar",
                    help="sidecar: JSON map travels separately (enables tamper localization); "
                         "blind: self-contained image, needs only the secret")
    pe.add_argument("--model", type=Path, default=Path("data/models"))
    pe.add_argument("--sidecar", type=Path, default=None)
    pe.set_defaults(func=cmd_embed)

    px = sub.add_parser("extract", help="recover a message from a (possibly attacked) stego image")
    px.add_argument("stego", type=Path)
    _add_secret(px)
    px.add_argument("--mode", choices=("auto", "sidecar", "blind"), default="auto",
                    help="auto: use the sidecar if one exists next to the image, else blind")
    px.add_argument("--sidecar", type=Path, default=None)
    px.add_argument("--out", type=Path, default=None, help="write recovered bytes here instead of printing as text")
    px.set_defaults(func=cmd_extract)

    pc = sub.add_parser("capacity", help="maximum message size per profile and mode for a cover")
    pc.add_argument("cover", type=Path)
    pc.add_argument("--model", type=Path, default=Path("data/models"))
    pc.set_defaults(func=cmd_capacity)

    pa = sub.add_parser("attack", help="apply a simulated attack to a stego image")
    pa.add_argument("stego", type=Path)
    pa.add_argument("out", type=Path)
    pa.add_argument("--type", choices=sorted(attacks.ATTACKS), required=True)
    pa.add_argument("--params", type=str, default=None, help='JSON kwargs, e.g. \'{"quality": 50}\'')
    pa.set_defaults(func=cmd_attack)

    pd = sub.add_parser("detect", help="blind steganalysis of an image or a folder (no secret/sidecar needed)")
    pd.add_argument("image", type=Path, help="image file, or a directory to screen")
    pd.add_argument("--recursive", action="store_true", help="with a directory: include subfolders")
    pd.add_argument("--csv", type=Path, default=None, help="with a directory: also write a CSV report")
    pd.add_argument("--all", action="store_true", help="with a directory: list every image, not just flagged")
    pd.add_argument("--no-ml", action="store_true", help="skip the trained ML detector")
    pd.add_argument("--model", type=Path, default=Path("data/models"))
    pd.set_defaults(func=cmd_detect)

    pv = sub.add_parser("evaluate", help="embed + run every attack + report PSNR/SSIM/BER/timing")
    pv.add_argument("cover", type=Path)
    pv.add_argument("--message", type=str, default=DEFAULT_EVAL_MESSAGE,
                    help="Defaults to a ~470-byte message spanning several RS codewords.")
    _add_secret(pv)
    pv.add_argument("--profile", choices=profiles, default=DEFAULT_PROFILE)
    pv.add_argument("--mode", choices=("sidecar", "blind"), default="sidecar")
    pv.add_argument("--model", type=Path, default=Path("data/models"))
    pv.set_defaults(func=cmd_evaluate)

    pb = sub.add_parser("benchmark", help="multi-image recovery/quality benchmark (or --inference speed)")
    pb.add_argument("--images", type=Path, default=Path("data/samples"))
    pb.add_argument("--num-images", type=int, default=30)
    pb.add_argument("--profiles", nargs="+", choices=profiles, default=None)
    pb.add_argument("--scenarios", nargs="+", default=None, help="subset of scenario names")
    pb.add_argument("--out", type=Path, default=Path("reports/benchmark"))
    pb.add_argument("--workers", type=int, default=None)
    pb.add_argument("--seed", type=int, default=7)
    pb.add_argument("--inference", action="store_true",
                    help="measure per-block inference latency / footprint (report O1, O7) instead")
    pb.add_argument("--model", type=Path, default=Path("data/models"))
    pb.set_defaults(func=cmd_benchmark)

    pt = sub.add_parser("train-detector", help="train the ML steganalyser on clean vs AIDE-stego covers")
    pt.add_argument("--images", type=Path, default=Path("data/samples"))
    pt.add_argument("--max-images", type=int, default=None)
    pt.add_argument("--out", type=Path, default=Path("data/models"))
    pt.add_argument("--model", type=Path, default=Path("data/models"))
    pt.add_argument("--workers", type=int, default=None)
    pt.add_argument("--seed", type=int, default=0)
    pt.set_defaults(func=cmd_train_detector)

    pp = sub.add_parser("profiles", help="list robustness profiles")
    pp.set_defaults(func=cmd_profiles)

    # Stub only, for discoverability in --help: "train" is intercepted in
    # main() below before argparse sees it, since aide.train has its own
    # independent argument parser.
    sub.add_parser("train", help="train the AIDE decision engine (see aide.train)")

    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "train":
        train_main(argv[1:])  # aide.train has its own argument parser
        return 0

    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if args.command in ("embed", "extract", "evaluate") and not (args.secret or args.secret_file):
        parser.error("--secret or --secret-file is required")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
