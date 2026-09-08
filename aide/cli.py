"""AIDE command-line interface: embed, extract, attack, evaluate, train."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import attacks, metrics, pipeline
from .model import AIDEModel
from .train import main as train_main


def _load_model(model_dir: Path) -> AIDEModel:
    try:
        return AIDEModel.load(model_dir)
    except FileNotFoundError:
        print(f"warning: no trained model at {model_dir}, using formula-only default "
              f"(run 'aide train' first for the real decision engine)", file=sys.stderr)
        return AIDEModel.default()


def cmd_embed(args: argparse.Namespace) -> int:
    model = _load_model(args.model)
    message = args.message.encode() if args.message else Path(args.message_file).read_bytes()
    secret = args.secret.encode()

    try:
        result = pipeline.embed(args.cover, message, secret, model)
    except pipeline.AideCapacityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    pipeline.save_grayscale(result.stego, args.out)
    sidecar_path = args.sidecar or Path(str(args.out) + ".aide.json")
    pipeline.save_sidecar(result.sidecar, sidecar_path)

    cover, _ = pipeline.load_grayscale(args.cover)
    print(json.dumps({
        "stego": str(args.out),
        "sidecar": str(sidecar_path),
        "message_bytes": len(message),
        "blocks_used": len(result.sidecar.embedding_map),
        "psnr_db": metrics.psnr(cover, result.stego),
        "ssim": metrics.ssim(cover, result.stego),
    }, indent=2))
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    stego, _ = pipeline.load_grayscale(args.stego)
    sidecar_path = args.sidecar or Path(str(args.stego) + ".aide.json")
    sidecar = pipeline.load_sidecar(sidecar_path)
    secret = args.secret.encode()

    result = pipeline.recover(stego, sidecar, secret)
    if not result.success:
        print(json.dumps({"success": False, "error": result.error,
                           "erasures": result.erasure_count, "total_bytes": result.total_bytes}, indent=2))
        return 1

    if args.out:
        Path(args.out).write_bytes(result.plaintext)
        print(json.dumps({"success": True, "message_bytes": len(result.plaintext), "out": str(args.out)}, indent=2))
    else:
        print(json.dumps({"success": True, "message": result.plaintext.decode(errors="replace")}, indent=2))
    return 0


def cmd_attack(args: argparse.Namespace) -> int:
    gray, _ = pipeline.load_grayscale(args.stego)
    fn = attacks.ATTACKS[args.type]
    kwargs = json.loads(args.params) if args.params else {}
    attacked = fn(gray, **kwargs)
    pipeline.save_grayscale(attacked, args.out)
    print(json.dumps({"attacked": str(args.out), "type": args.type, "params": kwargs}, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    model = _load_model(args.model)
    message = args.message.encode()
    secret = args.secret.encode()

    with metrics.measure() as t:
        result = pipeline.embed(args.cover, message, secret, model)
    cover, _ = pipeline.load_grayscale(args.cover)

    report = {
        "cover": str(args.cover),
        "message_bytes": len(message),
        "embed_time_s": t.elapsed_seconds,
        "psnr_db": metrics.psnr(cover, result.stego),
        "ssim": metrics.ssim(cover, result.stego),
        "attacks": {},
    }

    no_attack = pipeline.recover(result.stego, result.sidecar, secret)
    report["attacks"]["none"] = {
        "success": no_attack.success,
        "exact_match": no_attack.success and no_attack.plaintext == message,
    }

    for name, fn in attacks.ATTACKS.items():
        attacked = fn(result.stego)
        with metrics.measure() as t2:
            rec = pipeline.recover(attacked, result.sidecar, secret)
        ber = metrics.bit_error_rate(message, rec.plaintext) if rec.plaintext else 1.0
        report["attacks"][name] = {
            "success": rec.success,
            "exact_match": rec.success and rec.plaintext == message,
            "ber": ber,
            "erasures": f"{rec.erasure_count}/{rec.total_bytes}",
            "recover_time_s": t2.elapsed_seconds,
            "error": rec.error,
        }

    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aide", description="AIDE steganography framework")
    sub = p.add_subparsers(dest="command", required=True)

    pe = sub.add_parser("embed", help="embed a secret message into a cover image")
    pe.add_argument("cover", type=Path)
    pe.add_argument("out", type=Path)
    g = pe.add_mutually_exclusive_group(required=True)
    g.add_argument("--message", type=str)
    g.add_argument("--message-file", type=Path)
    pe.add_argument("--secret", type=str, required=True, help="shared secret for key derivation")
    pe.add_argument("--model", type=Path, default=Path("data/models"))
    pe.add_argument("--sidecar", type=Path, default=None)
    pe.set_defaults(func=cmd_embed)

    px = sub.add_parser("extract", help="recover a message from a (possibly attacked) stego image")
    px.add_argument("stego", type=Path)
    px.add_argument("--secret", type=str, required=True)
    px.add_argument("--sidecar", type=Path, default=None)
    px.add_argument("--out", type=Path, default=None, help="write recovered bytes here instead of printing as text")
    px.set_defaults(func=cmd_extract)

    pa = sub.add_parser("attack", help="apply a simulated attack to a stego image")
    pa.add_argument("stego", type=Path)
    pa.add_argument("out", type=Path)
    pa.add_argument("--type", choices=sorted(attacks.ATTACKS), required=True)
    pa.add_argument("--params", type=str, default=None, help='JSON kwargs, e.g. \'{"quality": 50}\'')
    pa.set_defaults(func=cmd_attack)

    pv = sub.add_parser("evaluate", help="embed + run every attack + report PSNR/SSIM/BER/timing")
    pv.add_argument("cover", type=Path)
    pv.add_argument(
        "--message", type=str,
        default=(
            "CLASSIFIED SITREP: Convoy Alpha departs staging area 0600Z, route Bravo-7. "
            "Escort confirmed, rendezvous at checkpoint Delta. Air support standby channel 4. "
        ) * 3,
        help="Defaults to a ~450-byte message spanning multiple Reed-Solomon codewords - "
             "a short (<~200 byte) message fits in a single codeword, which gets no benefit "
             "from block interleaving (see aide/pipeline.py's module docstring) and is "
             "correspondingly far more fragile to any attack.",
    )
    pv.add_argument("--secret", type=str, required=True)
    pv.add_argument("--model", type=Path, default=Path("data/models"))
    pv.set_defaults(func=cmd_evaluate)

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
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
