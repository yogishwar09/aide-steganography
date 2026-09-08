"""End-to-end AIDE walkthrough: embed a secret message, run every simulated
attack, attempt recovery, and print a measured (not aspirational) metrics
table.

Usage:
    python scripts/demo.py [--cover data/samples/1.pgm] [--model data/models]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aide import attacks, metrics, pipeline  # noqa: E402
from aide.model import AIDEModel  # noqa: E402

SECRET = b"defence-shared-secret-2026"
# Multi-codeword by design: a short message fits in a single 255-byte RS
# codeword and gets zero benefit from block interleaving (interleaving only
# helps once damage can be spread *across* codewords) - see
# aide/pipeline.py's module docstring. This is what a realistic short
# covert message spanning a few codewords looks like.
MESSAGE = (
    b"CLASSIFIED SITREP: Convoy Alpha departs staging area 0600Z, route Bravo-7. "
    b"Escort confirmed, rendezvous at checkpoint Delta. Air support standby channel 4. "
) * 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cover", type=Path, default=Path("data/samples/1.pgm"))
    parser.add_argument("--model", type=Path, default=Path("data/models"))
    args = parser.parse_args()

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
            result = pipeline.embed(args.cover, MESSAGE, SECRET, model)
        except pipeline.AideCapacityError as exc:
            print(f"EMBEDDING FAILED: {exc}")
            print("Try a different --cover image with more texture/edge content.")
            return

    cover, _ = pipeline.load_grayscale(args.cover)
    psnr = metrics.psnr(cover, result.stego)
    ssim = metrics.ssim(cover, result.stego)

    from collections import Counter
    method_bits = Counter()
    for e in result.sidecar.embedding_map:
        method_bits[e.method] += e.bit_end - e.bit_start
    total_bits = sum(method_bits.values())

    print(f"Embedding complete in {t_embed.elapsed_seconds * 1000:.1f} ms "
          f"({t_embed.rss_delta_mb:.2f} MB RSS delta)")
    print(f"  blocks used: {len(result.sidecar.embedding_map)}")
    print(f"  payload after AES-256-GCM + RS(255,223): {total_bits // 8} bytes "
          f"({total_bits // 8 // 255} codewords)")
    for m, bits in sorted(method_bits.items()):
        print(f"    {m}: {bits} bits ({100 * bits / total_bits:.1f}%)")
    print(f"  stego PSNR: {psnr:.2f} dB   SSIM: {ssim:.4f}")
    print()

    print("--- No-attack recovery ---")
    rec = pipeline.recover(result.stego, result.sidecar, SECRET)
    print(f"  success={rec.success}  exact_match={rec.success and rec.plaintext == MESSAGE}")
    print()

    print("--- Attack & recovery sweep ---")
    header = f"{'attack':<18} {'result':<8} {'BER':>8} {'erasures':>12} {'time_ms':>9}"
    print(header)
    print("-" * len(header))
    for name, fn in attacks.ATTACKS.items():
        attacked = fn(result.stego)
        with metrics.measure() as t_rec:
            rec = pipeline.recover(attacked, result.sidecar, SECRET)
        ber = metrics.bit_error_rate(MESSAGE, rec.plaintext) if rec.plaintext else 1.0
        ok = rec.success and rec.plaintext == MESSAGE
        print(f"{name:<18} {'OK' if ok else 'FAIL':<8} {ber:>8.4f} "
              f"{f'{rec.erasure_count}/{rec.total_bytes}':>12} "
              f"{t_rec.elapsed_seconds * 1000:>9.1f}")

    print()
    print("See README.md 'What actually works' / 'What doesn't work yet' for the")
    print("honest, measured story behind these numbers - they are not the report's")
    print("aspirational targets, which assume full ALASKA2-scale training.")


if __name__ == "__main__":
    main()
