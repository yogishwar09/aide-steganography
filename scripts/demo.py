"""End-to-end AIDE walkthrough: embed a secret message, run every simulated
attack, attempt recovery, and print a measured (not aspirational) metrics
table.

Usage:
    python scripts/demo.py [--cover data/samples/1022.pgm] [--profile balanced]
                           [--mode sidecar|blind] [--model data/models]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aide import attacks, blind, metrics, pipeline  # noqa: E402
from aide.model import AIDEModel  # noqa: E402
from aide.profiles import DEFAULT_PROFILE, PROFILES  # noqa: E402

SECRET = b"defence-shared-secret-2026"
# A realistic short covert message. Since AIDE 0.2, short messages are
# split into several shortened RS codewords automatically (aide/ecc.py's
# `plan`), so interleaving protects them too.
MESSAGE = (
    b"CLASSIFIED SITREP: Convoy Alpha departs staging area 0600Z, route Bravo-7. "
    b"Escort confirmed, rendezvous at checkpoint Delta. Air support standby channel 4. "
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cover", type=Path, default=Path("data/samples/1022.pgm"))
    parser.add_argument("--model", type=Path, default=Path("data/models"))
    parser.add_argument("--profile", choices=sorted(PROFILES), default=DEFAULT_PROFILE)
    parser.add_argument("--mode", choices=("sidecar", "blind"), default="sidecar")
    args = parser.parse_args()
    print(f"Profile: {args.profile} - {PROFILES[args.profile].description}")
    print(f"Mode: {args.mode}")

    print(f"Cover image: {args.cover}")
    print(f"Message ({len(MESSAGE)} bytes): {MESSAGE[:70]!r}...")
    print()

    try:
        model = AIDEModel.load(args.model)
        print(f"Loaded trained AIDE model from {args.model} "
              f"(method val accuracy: {model.meta.get('method_val_accuracy', '?')})")
    except FileNotFoundError:
        print(f"No trained model at {args.model} - run 'aide train' first. "
              f"Falling back to the formula-only default model.")
        model = AIDEModel.default()
    print(f"  learned weights (alpha,beta,gamma,delta): {model.weights.as_dict()}")
    print(f"  thresholds: {model.thresholds}")
    print()

    with metrics.measure() as t_embed:
        try:
            if args.mode == "blind":
                result = blind.embed(args.cover, MESSAGE, SECRET, model, profile=args.profile)
            else:
                result = pipeline.embed(args.cover, MESSAGE, SECRET, model, profile=args.profile)
        except pipeline.AideCapacityError as exc:
            print(f"EMBEDDING FAILED: {exc}")
            print("Try a different --cover image with more texture/edge content, or a lighter --profile.")
            return

    def recover(img):
        if args.mode == "blind":
            return blind.recover(img, SECRET)
        return pipeline.recover(img, result.sidecar, SECRET)

    # Compare the full original file against the full composed stego (RGB
    # recombined with untouched chroma for a color cover) - see cli.py's
    # cmd_embed for why this is fairer than comparing just the Y plane.
    cover_arr = np.array(Image.open(args.cover).convert(result.mode))
    stego_arr = np.array(result.compose())
    psnr = metrics.psnr(cover_arr, stego_arr)
    ssim = metrics.ssim(cover_arr, stego_arr)

    print(f"Embedding complete in {t_embed.elapsed_seconds * 1000:.1f} ms "
          f"({t_embed.rss_delta_mb:.2f} MB RSS delta)")
    print(f"  cover mode: {result.mode}")
    if args.mode == "sidecar":
        from collections import Counter
        method_bits = Counter()
        for e in result.sidecar.embedding_map:
            method_bits[e.method] += e.bit_end - e.bit_start
        total_bits = sum(method_bits.values())
        p = result.sidecar.ecc_params
        print(f"  blocks used: {len(result.sidecar.embedding_map)}")
        print(f"  payload after AES-256-GCM + RS({p.codeword_len},{p.data_len}): {total_bits // 8} bytes "
              f"({total_bits // 8 // p.codeword_len} codewords, t={p.t} each)")
        for m, bits in sorted(method_bits.items()):
            print(f"    {m}: {bits} bits ({100 * bits / total_bits:.1f}%)")
    else:
        print(f"  blocks used: {result.blocks_used} (keyed layout; header {result.header_bits} bits, "
              f"payload {result.payload_bits} bits of {result.capacity_bits})")
    print(f"  stego PSNR: {psnr:.2f} dB   SSIM: {ssim:.4f}")
    print()

    print("--- No-attack recovery ---")
    rec = recover(result.stego)
    print(f"  success={rec.success}  exact_match={rec.success and rec.plaintext == MESSAGE}")
    print()

    print("--- Attack & recovery sweep ---")
    header = f"{'attack':<18} {'result':<8} {'BER':>8} {'corrected':>10} {'time_ms':>9}"
    print(header)
    print("-" * len(header))
    for name, fn in attacks.ATTACKS.items():
        attacked = fn(result.stego, **({"seed": 1} if name in attacks.SEEDED_ATTACKS else {}))
        with metrics.measure() as t_rec:
            rec = recover(attacked)
        ber = metrics.bit_error_rate(MESSAGE, rec.plaintext) if rec.plaintext else 1.0
        ok = rec.success and rec.plaintext == MESSAGE
        print(f"{name:<18} {'OK' if ok else 'FAIL':<8} {ber:>8.4f} "
              f"{rec.corrected_bytes:>10} {t_rec.elapsed_seconds * 1000:>9.1f}")

    print()
    print("Default attack strengths are the harshest in each category (e.g. noise")
    print("sigma=20, 0.5x scaling, 15% crop). See README.md 'Measured results' for the")
    print("multi-image benchmark per profile, or run `aide benchmark`.")


if __name__ == "__main__":
    main()
