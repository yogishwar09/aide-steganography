<img src="webapp/static/logo-wordmark.svg" alt="AIDE — Adaptive Intelligent Dynamic Embedding" width="360">

A working implementation of the AIDE steganography framework described in
`AIDE_Technical_Report.pdf` (CD753PC Project, MGIT Hyderabad): CPU-only
adaptive image steganography that combines a learned block-level scoring
policy, AES-256-GCM encryption, reliability-guided Reed-Solomon error
correction with interleaving, transform-domain embedding, and tamper
localization + recovery. It also includes a detection toolkit (classical,
quantitative and machine-learning steganalysis).

It covers **both sides** of the report's defence and intelligence use case
(section 7):

- **Operator:** secure covert communication through images that survive
  compression, noise, filtering, rescaling and cropping on the way.
- **Analyst:** screen intercepted media for hidden data, with no secret
  and no sidecar.

Every number below is **measured** by code in this repository (`aide
benchmark`, the test suite) on real BOSSbase photographs. The README is
explicit about where the project still falls short. See [Known
limitations](#known-limitations-and-future-work) before citing a number.

## What's new in 0.2

| Area | AIDE 0.1 | AIDE 0.2 |
|---|---|---|
| Robustness | One fixed setting; Gaussian σ=20, salt & pepper and 0.5× scaling **failed outright** | **Four measured profiles**. `robust` meets report O4 end-to-end: σ=20 noise, JPEG Q30–50, salt & pepper, median filter and 10% crop all at **100%** |
| Error correction | Fixed RS(255,223); short messages got no interleaving benefit | Configurable parity, **shortened codewords** for short messages, and **GMD reliability-guided decoding** |
| Sidecar | Mandatory JSON side file | Optional: **blind mode** embeds a self-describing, keyed payload that needs only the image and the secret |
| Key derivation | Single SHA-256 | **scrypt** (memory-hard; resists offline brute force of weak passphrases). 0.1 sidecars still decode |
| Detection | Chi-square only (catches only sequential full-image LSB) | Adds **Sample Pairs Analysis + RS analysis** (validated against ground truth; estimate the payload size) and a **trained ML steganalyser** with a cover-source-mismatch guard |
| Decision engine | Decision tree | Tree **or tiny MLP** (report phase 3); 3× faster feature extraction (numerically identical) |
| Tooling | CLI + local demo UI | `capacity`, `benchmark`, `train-detector`, folder screening to CSV, **versioned REST API + OpenAPI**, API-key auth, Docker, CI, lint |
| Tests | 85 | **146** |

Same 30 BOSSbase covers and 156-byte message, AIDE 0.1 parameters vs the
0.2 default (`balanced`) profile:

| Attack | 0.1 | 0.2 `balanced` | 0.2 `robust` |
|---|---:|---:|---:|
| JPEG Q50 / Q30 | 100% / 100% | 100% / 100% | 100% / 100% |
| Gaussian noise σ=20 (O4) | 0% | 0% | **100%** |
| Salt & pepper 1% | 0% | 0% | **100%** |
| Median filter 3×3 | 69% | **100%** | **100%** |
| Rescale 0.5× | 0% | **92%** | **96%** |
| Splice 5% | 85% | **100%** | **100%** |
| Crop 10% | 50% | **88%** | **100%** |
| Mean PSNR / SSIM | 39.7 dB / 0.975 | 40.1 dB / 0.967 | 35.0 dB / 0.912 |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,web,data,report]"   # or: pip install -r requirements.txt && pip install -e .
```

Fetch real cover images: a subset of BOSSbase 1.01, pulled with HTTP range
requests. It falls back to scikit-image samples if the host is unreachable.

```bash
python scripts/fetch_dataset.py --count 300
```

Train the decision engine and the steganalyser:

```bash
aide train --images data/samples --out data/models                 # decision tree (default)
aide train --images data/samples --out data/models --classifier mlp # or the tiny MLP
aide train-detector --images data/samples --out data/models         # ML steganalyser (~1-3 min)
```

`make install data train detector test bench web` wraps these steps.

## Quick start (CLI)

```bash
# Operator
aide profiles                                   # list robustness profiles
aide capacity cover.png                         # exact max message size per profile and mode
aide embed cover.png stego.png --message "grid 47N 12E" --secret-file key.txt --profile robust
aide embed cover.png stego.png --message "grid 47N 12E" --secret "pw" --mode blind   # no sidecar
aide attack stego.png attacked.png --type gaussian_noise --params '{"sigma": 20}'
aide extract attacked.png --secret-file key.txt  # auto: uses <image>.aide.json if present, else blind
aide evaluate cover.png --secret "pw" --profile robust                               # every attack, one image

# Analyst
aide detect suspect.jpg                         # chi-square + SPA + RS + ML, one verdict
aide detect /media/intercepted --recursive --csv screening.csv   # batch screening

# Engineering
aide benchmark --num-images 30 --out reports/benchmark            # multi-image recovery benchmark
aide benchmark --inference --num-images 5                         # report O1/O7 latency and footprint
python scripts/demo.py --profile robust --mode blind              # narrated walkthrough
```

Use `--secret-file` to keep the secret out of shell history. Exit codes are
script-friendly: `extract` returns 1 on failure, and `detect` returns 1 if
anything is flagged.

## Robustness profiles

A profile is a named, measured operating point on the quality ↔
robustness ↔ capacity trade-off (`aide/profiles.py`). The profile used is
recorded in the sidecar or the blind header, so the receiver never has to
guess it.

| Profile | DCT δ | RS parity (per 255) | Mean PSNR | Mean SSIM | Built for |
|---|---:|---:|---:|---:|---|
| `quality` | 20 | 64 | **43.4 dB** | **0.984** | Lossless channels and JPEG ≥ Q50; meets the report's >42 dB / >0.98 targets |
| `balanced` *(default)* | 36 | 64 | 40.1 dB | 0.967 | JPEG ≥ Q30, noise σ ≤ 10, filtering, rescaling, splicing |
| `robust` | 64 | 104 | 35.0 dB | 0.912 | **Report O4**: σ=20 noise, salt & pepper, 10% crop |
| `fortress` | 80 | 176 | 30.5 dB | 0.803 | Hostile channels; visible cost; short messages |

Full measured matrix (`reports/benchmark/benchmark.md`, 30 BOSSbase
covers, 156-byte message, bit-exact AES-GCM-verified recovery):

| Scenario | quality | balanced | robust | fortress |
|---|---:|---:|---:|---:|
| none | 100% | 100% | 100% | 100% |
| JPEG Q90 / Q70 / Q50 | 100% | 100% | 100% | 100% |
| JPEG Q30 | 20% | 100% | 100% | 100% |
| Gaussian noise σ=5 | 100% | 100% | 100% | 100% |
| Gaussian noise σ=10 | 0% | 100% | 100% | 100% |
| Gaussian noise σ=20 (O4) | 0% | 0% | 100% | 100% |
| Salt & pepper 1% | 0% | 0% | 100% | 100% |
| Median 3×3 | 52% | 100% | 100% | 100% |
| Gaussian blur | 100% | 100% | 100% | 100% |
| Rescale 0.75× | 100% | 100% | 100% | 100% |
| Rescale 0.5× | 40% | 92% | 96% | 100% |
| Splice 2% / 5% | 100% | 100% | 100% | 100% |
| Crop 10% | 84% | 88% | 100% | 100% |
| Covers with capacity for the message | 25/30 | 25/30 | 24/30 | 20/30 |

Robustness depends on the message length relative to capacity. Each
profile prefers its most robust layout (few, low-frequency DCT pairs per
block) and relaxes it only when a long message would not fit otherwise. So
a message near a cover's capacity is less robust than these 156-byte runs.
`aide capacity` reports both limits.

## Modes: sidecar vs blind

| | Sidecar (default) | Blind |
|---|---|---|
| Receiver needs | Stego image + `<image>.aide.json` + secret | Stego image + secret |
| Block layout | AIDE-selected blocks, ranked by S(b) | Keyed pseudo-random permutation of all blocks (SHA-256 of the secret) |
| Adaptivity | Where / how much / which method per block | Per-block embedding *strength* from S(b); saturation-aware |
| Tamper map | Per-block SHA-256 + histogram → spatial map, RS erasures | None; damage is absorbed by reliability-guided decoding |
| Capacity (512×512, `balanced`) | ~870 B | ~650 B (fixed 960-bit header) |

**How blind mode works** (`aide/blind.py`). A 14-byte header holds the
magic, version, profile id, an 8-byte random salt and the ciphertext
length. It is RS-protected and stored three times with soft (margin-summed)
combining, followed by the RS-encoded, interleaved ciphertext.

Extraction only reads the *sign* of each coefficient-pair difference, so
the receiver needs no per-block map. A wrong secret finds no header at all
("no AIDE blind header found"), and without the secret an adversary cannot
even locate the payload blocks.

Measured on 16 covers: `robust` blind survives JPEG Q30, σ=20 noise, median
filtering, 5% splicing and 10% cropping at 15–16/16, and `fortress` blind
survives 0.5× rescaling at 16/16. PSNR is about 3–4 dB below sidecar mode
at the same profile, because of the header and because every block carries
payload.

## Web console and REST API

```bash
python webapp/app.py                        # development: http://127.0.0.1:5050
gunicorn -w 2 -b 0.0.0.0:5050 webapp.app:app  # production
docker build -t aide . && docker run -p 5050:5050 -v "$PWD/data:/app/data" aide
```

- **Operator:** choose a profile and mode and see live capacity for the
  chosen cover. Then embed, simulate an attack, inspect the tamper/erasure
  map (sidecar mode) and recover. The recovery status reports how many
  bytes ECC repaired and whether GMD decoding rescued any codewords.
  Embedding-map and difference-heatmap views are included.
- **Analyst:** run a single-image assessment across all three detector
  families (verdict, SPA/RS payload estimate, ML probability, suspicion
  heatmap), or batch-screen up to 50 images into a ranked table with CSV
  export.
- **Model Insights:** shows the learned S(b) weights, thresholds and label
  distribution, the **robustness benchmark matrix**, and the **ML
  steganalyser's held-out evaluation**.

Every endpoint is served at `/api/<name>` and at the versioned
`/api/v1/<name>`. The contract is in `/api/v1/openapi.json`, and
`/healthz` is the health check.

Set `AIDE_API_KEY` to require an `X-API-Key` header on API calls. Responses
carry `X-Request-ID`, a CSP and anti-framing headers, and every error is
returned as JSON (`{"error": ...}`). The server is stateless: the browser
holds the stego image and sidecar between steps, and temp files are
deleted after each request.

## Architecture

```
aide/
  features.py      block partitioning + texture (LBP+GLCM), entropy, variance, edge
                   (GLCM/entropy/variance vectorised across blocks, identical results)
  scoring.py       S(b) = alpha*T + beta*E + gamma*V + delta*D
  model.py         AIDEModel: learned weights/thresholds + tree or MLP classifier/regressor
  train.py         measurement-based training (see below)
  profiles.py      robustness profiles (deltas, RS parity, budgets, bpp cap)
  crypto.py        AES-256-GCM; scrypt (default) or SHA-256 KDF over image hash/salt + secret
  ecc.py           RS with configurable parity, shortened-codeword planning, interleaving,
                   GMD reliability-guided decoding
  embedding/       lsb.py, dct.py (mid-frequency pair QIM), dwt.py (2-level Haar cA2 pair QIM)
                   each with embed / extract / extract_soft (signed margins)
  pipeline.py      sidecar-mode embed()/recover(), capacity_report()
  blind.py         blind-mode embed()/recover(): keyed layout, self-describing header
  tamper.py        per-block SHA-256 + histogram chi-square -> tamper mask + confidence
  attacks.py       6 attack categories, 8 simulations
  metrics.py       PSNR, SSIM, BER, timing/memory
  steganalysis.py  blind detection: chi-square PoV + SPA/RS (verdict), LSB signals, heatmap grid
  quantitative.py  RS analysis and Sample Pairs Analysis embedding-rate estimators
  mlsteg.py        SPAM + DCT pair-gap features, gradient-boosted detector, OOD guard
  benchmark.py     parallel multi-image benchmark + inference-speed measurement
  visualize.py     embedding map, diff heatmap, tamper overlay, suspicion heatmap
  cli.py           `aide` command-line entry point
webapp/            Flask console + REST API (gunicorn-ready)
reports/benchmark/ the committed benchmark run quoted in this README
```

### The decision engine: trained on real measurements

The report says the scoring weights are "learned during training" but does
not say what the labels are. To avoid circularity:

1. **Method labels** come from the report's own rule (`texture < T_low →
   LSB`, `edge > E_high → DWT`, else DCT). The thresholds are the 33rd and
   67th percentiles over the training blocks, so the classes are roughly
   balanced. A decision tree (or MLP) learns to reproduce and smooth that
   rule from all four features. Its feature importances (permutation
   importances for the MLP) become the explainable (α, β, γ, δ).
2. **Payload (bpp) and SKIP labels** are measured. Random bits are actually
   embedded at bpp 4/3/2/1 and the block PSNR is measured. The label is the
   highest safe bpp, or SKIP if none is safe.

Trained on 334 BOSSbase images (342k blocks): **87.7%** validation accuracy
(tree). The MLP reaches 85.4% on a 20k-block subsample.

### How the robustness gains were obtained (all measured)

Each change was adopted only after `aide benchmark` showed it helped:

- **Route the payload away from LSB and DWT.** On ~2% of the payload bits
  they caused ~40% of the bit errors under median filtering (flip rates:
  LSB 29–54%, DWT 8–13%, DCT ~0.5%). Zeroing their budgets raised 10%-crop
  recovery from 52% to 88% and median-filter recovery from 80% to 88%, at
  *better* PSNR. The engine still decides where, how much, and which method
  per block. Blocks it assigns to LSB/DWT are simply left untouched (pass
  `fragile_method_budgets` to re-enable them).
- **Prefer a low-frequency layout.** Capping blocks at 8 DCT pairs (the
  lowest robust mid-band pairs) took balanced-profile 0.5× rescaling from
  0% to 92% and median filtering from 88% to 100%, at the same PSNR.
- **Shortened RS codewords.** A 156-byte message used to occupy one
  255-byte codeword, so interleaving had nothing to spread damage across.
  `ecc.plan` now splits short payloads into ≥4 shortened codewords at the
  same parity ratio. Shortened RS codes are standard practice (DVB, QR
  codes).
- **GMD decoding (Forney, 1966).** Every extracted bit carries a soft
  margin |cᵢ − cⱼ| / δ. Bytes that an attack pushed near the decision
  boundary are erased first, so RS can fix up to 2t errors instead of t
  when the low-reliability guesses are right. Candidates for every erasure
  count are compared by reliability-weighted distance. The original
  first-success rule was measured to *mis-correct* on short codewords (all
  13 failures in a 20-trial test were wrong-but-valid codewords).
  Minimum-distance selection reduced that (recovery went from 7/20 to
  9/20), and saturation awareness (next item) closed the rest.
  A wrong codeword can never produce wrong plaintext, because the AES-GCM
  tag rejects it.
- **Saturation awareness (blind mode).** Clipped (pure black/white) blocks
  lose margin to rounding. The sender boosts their strength and the
  receiver down-weights them. On a cover with 34% saturated pixels, σ=20
  recovery went from 9/20 to 20/20.

### Design notes carried over from 0.1

- **Sidecar.** Recovery in sidecar mode needs the per-block embedding map,
  reference hashes, RS parameters and the GCM nonce. These are "transmitted
  alongside" (report), implemented as `<stego>.aide.json`. Its version is
  `aide-0.2`; `aide-0.1` files still decode with their original parameters.
- **Color covers.** Embedding uses the Y (luma) plane; Cb/Cr pass through
  untouched. Attack simulation is luma-only, so chroma subsampling in real
  JPEG encoders is not modelled.
- **Bit-level capacity accounting.** DCT/DWT blocks carry a handful of bits
  each, so the payload is a flat bitstream rather than whole bytes per
  block.

## Detection: steganalysis toolkit

`aide detect` answers the analyst's question: given an image with no
secret, no sidecar and no knowledge of how it might have been modified,
does it carry hidden data? Three detector families feed one verdict:

1. **Chi-square pairs-of-values** (Westfeld & Pfitzmann 2001). Catches
   sequential, near-full-image LSB embedding.
2. **Sample Pairs Analysis** (Dumitrescu et al. 2003) as the primary signal
   and **RS analysis** (Fridrich et al. 2001, named in the report) as
   corroboration (`aide/quantitative.py`). These *estimate the embedding
   rate*, so the report includes an estimated payload in bytes. No
   reference implementation was available, so both are **validated against
   ground truth**: known random-LSB rates were embedded into real covers.

   | True rate | 0 | 0.10 | 0.25 | 0.50 | 0.75 | 1.0 |
   |---|---:|---:|---:|---:|---:|---:|
   | SPA mean estimate | 0.03 | 0.12 | 0.27 | 0.51 | 0.76 | 0.98 |
   | RS mean estimate | 0.04 | 0.14 | 0.28 | 0.53 | 0.76 | unstable (known RS degeneracy as p→1) |

   Thresholds are calibrated on 200 clean covers that are disjoint from the
   test set (SPA rate > 6.1% is the clean 97.5th percentile). Held-out
   results: **1/40 false positives**, and **randomly scattered** LSB at a
   10% rate is detected in **40/40** covers. Chi-square alone catches only
   sequential embedding.
3. **Trained ML steganalyser** (`aide/mlsteg.py`, `aide train-detector`).
   It uses 98-dim SPAM features (Pevný et al. 2010) plus 28 targeted DCT
   coefficient-pair gap features, with a gradient-boosted classifier
   trained on clean vs AIDE-stego covers. This is the only family that can
   see transform-domain embedding. The operating point is a 5% FPR on
   held-out clean covers.
   - **Cover-source-mismatch guard.** A classifier trained on BOSSbase
     photographs scored a synthetic, clean test pattern at P(stego) = 0.99.
     An IsolationForest fitted on the training features now flags
     out-of-distribution inputs. For those, the ML verdict is capped at
     "suspicious" and the output says why. 98.7% of held-out real covers
     count as in-distribution.

## Security analysis

**Confidentiality and integrity.** AES-256-GCM. The key is
scrypt(secret, image hash or salt) with N=2¹⁴, r=8, p=1, which adds
roughly 16 MB and tens of milliseconds per guess for an offline attacker.
The GCM tag is verified before anything is returned, so recovery yields the
exact message or an error, never a corrupted message. Blind mode derives
its nonce from a fresh random salt that is also mixed into the key, so
every (key, nonce) pair is single-use.

**Undetectability: an honest finding.** AIDE stego images **evade the
classical blind detectors**: chi-square, SPA and RS all score clean (0/15
flagged in `tests/test_steganalysis.py`). That is the report's "negative
detectability" claim, and it holds for those tools. **It does not hold
against a detector that knows the embedding family.** On 100 held-out
covers, the targeted ML steganalyser detects every AIDE variant (sidecar
quality, balanced and robust; blind balanced and robust) at **100%, with
AUC 0.998–1.000 and a 1% false-positive rate**.

The cause is structural. Coefficient-pair QIM forces |cᵢ − cⱼ| ≥ δ, which
leaves a deficit of small pair differences that natural images do not
have. The tool to measure it ships with the project, so this is stated as a
known property rather than hidden. See future work.

**Threat model.** Without the secret, an adversary cannot decrypt, forge a
message (GCM), or, in blind mode, locate the payload (keyed layout). An
adversary *can* destroy the payload: any attack beyond the chosen
profile's tolerance, or geometric attacks (rotation, true cropping) that
desynchronize the block grid. They can also detect it statistically, as
described above.

## Report objectives: status

| Objective | Target | Status (measured) |
|---|---|---|
| O1 lightweight model | < 5 ms/block CPU inference | **Met:** 0.10 ms/block feature extraction + 0.06 ms single-block decision (`aide benchmark --inference`); model 13 KB |
| O2 adaptive locations/payload | per-block map, unique per image | **Met:** score-ranked, gated, per-block bpp and method |
| O3 encryption | AES-256-GCM, image-bound key | **Met,** hardened with scrypt |
| O4 robustness | BER < 0.01, > 98% at JPEG Q50 and σ=20 | **Met by `robust`:** 24/24 at both (BER 0); `balanced` meets JPEG only |
| O5 tamper localization | per-block hash map + chi-square + RS analysis | **Met:** hash + histogram tamper map; chi-square, SPA and RS analysis |
| O6 recovery | > 98% under moderate attacks | **Met by `robust`:** 100% on 15 of 16 scenarios, 96% on 0.5× rescale |
| O7 vs CNN baselines | ≥ 4× faster, < 10 MB | **Partial:** 13 KB model, ~0.15 s embed / 0.04 s recover per 512² image on one CPU core; the SteganoGAN/HiDDeN comparison has not been run |
| Quality | PSNR > 42 dB, SSIM > 0.98 | **Met by `quality`** (43.4 dB / 0.984); stronger profiles trade quality for robustness |

## Known limitations and future work

What remains is listed plainly:

- **Detectable by targeted steganalysis** (see Security analysis). The fix
  is research-level: side-informed embedding with syndrome-trellis codes
  (STC) to minimise a distortion function, or keyed and dithered pair
  selection to remove the QIM gap. Not attempted.
- **Geometric desynchronization.** Rotation, true cropping (canvas
  change) and aspect-ratio changes misalign the block grid. Simulated
  cropping replaces regions in place. Fixing this needs a registration
  template or invariant-domain embedding.
- **`quality` is not noise-robust** (0% at σ ≥ 10), by design. Use
  `balanced` or stronger for noisy channels.
- **Capacity vs robustness.** Long messages near capacity lose the
  low-frequency layout preference and with it most 0.5×-rescale survival.
- **Training scale.** Trained on 334 BOSSbase images, not ALASKA2 (80k+).
  `aide train` and `aide train-detector` accept any folder, so scaling up
  is a data task. The ML detector is trained on one cover source and its
  OOD guard only softens verdicts.
- **CNN baselines (O7)** were not benchmarked, because this environment
  has no pretrained SteganoGAN/HiDDeN weights.
- **Raspberry Pi timings** are not measured. Run `aide benchmark
  --inference` on the target.
- **Color-aware attacks.** Chroma-subsampling JPEG effects are not
  modelled (luma-only pipeline).
- **The Docker image** builds from the included Dockerfile and its
  entrypoint is verified locally with gunicorn. The container build itself
  runs in CI.

## Testing and quality

```bash
pytest -q         # 146 tests: unit, ground-truth validation, end-to-end CLI/HTTP, attack recovery,
                  # and a real AIDE 0.1 file (tests/fixtures) that must keep decoding
ruff check .      # lint (configured in pyproject.toml)
aide benchmark    # the multi-image numbers above
```

CI (`.github/workflows/ci.yml`) runs lint, then tests on Python 3.10–3.12
with freshly fetched covers and freshly trained models, then a benchmark
smoke test and a Docker build with a container health check.

## Technology stack

OpenCV, scikit-learn, NumPy, SciPy, PyWavelets, Pillow, scikit-image,
pycryptodome (AES-GCM, scrypt), reedsolo, psutil, joblib, Flask + gunicorn,
pytest, ruff. CPU-only throughout; no GPU/CUDA dependency.
