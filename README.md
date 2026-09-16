<img src="webapp/static/logo-wordmark.svg" alt="AIDE — Adaptive Intelligent Dynamic Embedding" width="360">

A working prototype of the AIDE steganography pipeline described in
`AIDE_Technical_Report.pdf` (CD753PC Project Stage I, MGIT Hyderabad): a
CPU-only image steganography system combining a learned block-level
scoring policy, AES-256-GCM encryption, Reed-Solomon ECC with
interleaving, adaptive LSB/DCT/DWT embedding, and tamper localization +
recovery.

Built for the report's own defence and intelligence framing (section 7:
secure military/satellite/drone communication, border and port media
screening, cyber defence command centres) - which means covering **both
sides**: secure covert communication (`aide embed`/`extract`, for
authorized secure channels) *and* detecting someone else's covert
communication (`aide detect`, blind steganalysis with no secret or
sidecar - the "is this intercepted image smuggling hidden data" analyst
question). See [Detection: blind
steganalysis](#detection-blind-steganalysis-defence--intelligence) below.

This is a **real, tested implementation**, not a restatement of the
report. It's also **honest about where it falls short** of the report's
two-semester, ALASKA2-scale targets — see [What doesn't work
yet](#what-doesnt-work-yet) below before you cite any number from this
project in a review.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # installs the `aide` console script
```

Fetch training/demo images (pulls a subset of the real BOSSbase 1.01
dataset via HTTP range requests — no need to download the full 1.67 GB
archive; falls back to scikit-image's bundled sample photos if the host is
unreachable):

```bash
python scripts/fetch_dataset.py --count 300
```

Train the AIDE decision engine on the fetched images:

```bash
aide train --images data/samples --out data/models
```

Run the full walkthrough (embed → attack → recover, with a measured
metrics table):

```bash
python scripts/demo.py
```

Or launch the local web console (a clickable UI over the same code, with
both an Operator/secure-comms mode and an Analyst/threat-detection mode
— see [Web console](#web-console) below):

```bash
python webapp/app.py
```

then open the URL it prints (defaults to http://127.0.0.1:5050).

Run the test suite:

```bash
pytest tests/ -q
```

## CLI

```bash
aide embed  cover.png stego.png --message "secret text" --secret "shared-secret"
aide extract stego.png --secret "shared-secret"
aide attack stego.png attacked.png --type compression --params '{"quality": 50}'
aide extract attacked.png --secret "shared-secret" --sidecar stego.png.aide.json
aide evaluate cover.png --secret "shared-secret"   # embed + every attack + metrics
aide detect some_intercepted_image.jpg             # blind steganalysis, no secret/sidecar needed
aide train --images data/samples --out data/models
```

`embed` writes two files: the stego image, and a `<stego>.aide.json`
sidecar carrying the per-block embedding map, tamper-reference hashes, RS
parameters, and the AES-GCM nonce. See [Side-channel
metadata](#side-channel-metadata-the-sidecar-file) for why this exists.

## Web console

```bash
python webapp/app.py
```

A local, single-page web UI (`webapp/app.py` + `webapp/templates/index.html`)
over the exact same `aide` package the CLI uses - every button is a thin
HTTP wrapper around an `aide.pipeline`/`aide.steganalysis` call, no
independent logic of its own. Two modes, matching this project's two
defence/intelligence sides:

- **Operator — Secure Communication**: upload a cover image, type a
  message and shared secret, embed, optionally simulate an interception
  attack, and recover - all in the browser, with PSNR/SSIM/method-mix
  metrics and the same attack list `aide attack` exposes.
- **Analyst — Threat Detection**: upload any image and run blind
  steganalysis (`aide detect`) - verdict, suspicion score, per-signal
  breakdown, and the row-band diagnostic plot, with the same measured
  false-positive/detection-rate story documented above.
- **Model Insights**: the trained decision engine's actual learned
  parameters - training-set size, validation accuracy, the S(b) scoring
  weights as a bar chart, the confidence-gate/method thresholds, and the
  training label distribution. Makes the report's "explainable, not a
  black box" claim concrete instead of asserted - see `aide/model.py`.

### Visual analysis (`aide/visualize.py`)

Three views turn the decision engine's choices and the attack's actual
effect into images instead of only numbers, each rendered on demand from
the Operator tab:

- **Embedding map** - colors every block the engine actually used by
  which method (LSB/DCT/DWT) embedded into it. The most direct visual
  evidence of the report's central novelty: a learned, per-block,
  multi-method policy, not a fixed rule - the method assignment visibly
  tracks the cover image's texture and edges.
- **Difference heatmap** - the real, amplified `|cover - stego|` pixel
  difference. Honest about the fact that pixels did change (steganography
  isn't magic), rather than only citing a PSNR number.
- **Tamper/erasure map** (after an attack) - which blocks the receiver
  actually trusts: red for erasure (fed to Reed-Solomon as a known-bad
  byte range), amber for hash-mismatch-but-corrected-by-blind-ECC,
  untouched for trusted. The concrete payoff of `aide.tamper`.

Stateless by design: the Flask backend holds no session - the browser
keeps the stego image and sidecar (as base64/JSON) between steps and
POSTs back whatever the next endpoint needs, and the server only ever
touches disk via short-lived temp files (`aide.pipeline`'s functions need
a file path for the image-bound key derivation - see
`aide/crypto.py`) that are deleted after each request. Local only (Flask's
built-in dev server, `debug=True`) - not hardened for exposure to a
network; see the warning it prints on startup.

Tested the same way as everything else: `tests/test_webapp.py` drives
the Flask app via its test client (no live server needed) through the
full embed → attack → recover workflow, both detection cases, and all
three visualization endpoints; `tests/test_visualize.py` unit-tests the
overlay/heatmap generation directly.

## Architecture

```
aide/
  features.py     block partitioning (8x8/16x16, adaptive) + per-block
                   texture (LBP+GLCM), Shannon entropy, variance, edge
                   strength (Sobel+Canny)
  scoring.py       S(b) = alpha*T + beta*E + gamma*V + delta*D
  model.py         AIDEModel: learned weights/thresholds + trained
                   sklearn classifier/regressor
  train.py         empirical, measurement-based training (see below)
  crypto.py        AES-256-GCM, key = SHA256(image_hash + shared_secret)
  ecc.py           Reed-Solomon RS(255,223) + block interleaving, with
                   erasure support
  embedding/
    lsb.py         variable bits-per-pixel LSB substitution
    dct.py         mid-frequency DCT coefficient-pair QIM
    dwt.py         2-level Haar DWT, coefficient-pair QIM in the coarse
                   approximation sub-band
  tamper.py        per-block SHA-256 + histogram chi-square -> spatial
                   tamper mask + confidence
  attacks.py       6 attack categories, 8 concrete simulations
  metrics.py       PSNR, SSIM, BER, timing/memory
  pipeline.py      embed()/recover() orchestration + sidecar I/O
  steganalysis.py  blind detection (chi-square PoV, LSB correlation) -
                   no secret/sidecar needed; the counter-intelligence side
  cli.py           `aide` command-line entry point
```

### The AIDE decision engine: trained on real measurements, not assumptions

The report specifies a scoring formula with weights "learned during
training" but doesn't say what the training labels *are*. Two choices
were made here, deliberately avoiding circularity:

1. **Method selection** (LSB/DCT/DWT/skip): the report's own rule
   (`texture < T_low -> LSB`, `edge > E_high -> DWT`, else `DCT`) generates
   the ground-truth label, with `T_low`/`E_high` set from the 33rd/67th
   percentile of texture/edge observed across the training images (so
   classes come out roughly balanced rather than hand-tuned). A
   `DecisionTreeClassifier` is then trained to reproduce and smooth that
   rule from all four features — this is the "AI refines the score" step.
   Its normalized `feature_importances_` become the explainable
   `(alpha, beta, gamma, delta)` weights.
2. **Payload (bpp) and the confidence gate**: for each block, under its
   assigned method, we actually embed random bits at bpp=4,3,2,1 and
   measure the real resulting block PSNR, keeping the highest bpp that
   stays at/above a target. If even bpp=1 is unsafe, the block is labelled
   `SKIP`. A `DecisionTreeRegressor` learns to predict this from the
   features. **This is measured, not assumed.**

Run `aide train` yourself to see this in action — it prints train/val
accuracy and the learned weights.

### Side-channel metadata: the sidecar file

Recovery needs the per-block embedding map, reference hashes, RS
parameters, and the GCM nonce. The report says these are "transmitted
alongside" the stego image — this prototype implements that literally as
a JSON sidecar file, assumed to travel over a channel separate from (and
more reliable than) the possibly-attacked image itself. A fully
self-describing steganographic header (embedding this metadata *in* the
image too) is real future work, not built here — it doesn't change which
novelty is being demonstrated.

### Color images: Y-plane only

The whole pipeline (features, AIDE decisions, embedding, tamper
localization, attack simulation) operates on one 2D plane. For a color
cover, `pipeline.load_image` converts to YCbCr and that plane is Y
(luma); Cb/Cr are carried through untouched and recombined with
`compose_image`/`EmbedResult.compose()` on save. This is standard
practice (the eye is far less sensitive to chroma changes at high spatial
frequency), and keeps the embedding pipeline single-channel rather than
3x more complex. It does mean attack simulation isn't truly
"color-aware" - e.g. it doesn't model chroma subsampling the way a real
JPEG encoder's color-space handling would - which is a reasonable,
documented scope boundary rather than an attempt at that separate
problem.

### Why bit-level, not byte-level, capacity accounting

DCT/DWT blocks often carry only a handful of *bits* (a 16x16 DWT block at
bpp=1 can be as low as 4 bits). Early versions of this pipeline rounded
each block down to whole bytes, which was measured to waste the large
majority of DCT/DWT capacity and push the payload back onto
high-capacity-but-fragile LSB blocks. `pipeline.py` tracks a flat
bitstream instead, so every bit of every method's real capacity gets used.

## What actually works (measured)

Run `python scripts/demo.py` to reproduce these numbers yourself — they
vary by cover image and message length, but this is representative:

- **Exact recovery with no attack**: 100%, PSNR ~33-51 dB depending on how
  much of the image is needed for the message (short messages on
  higher-capacity images comfortably clear the report's 42 dB target;
  longer messages that need more DCT/DWT blocks for robustness trade some
  PSNR for it — a real, measured trade-off, not a bug).
- **JPEG compression, Q=30-70**: recovers reliably. This matches and
  exceeds report objective O4's explicit target (Q=50).
- **Localized splicing/tampering** (small-to-moderate region
  replacement): recovers reliably *when* the message spans enough RS
  codewords for interleaving to spread the damage thin — see the
  single-codeword caveat below. Tamper localization correctly
  distinguishes a genuinely replaced block (100% detection in testing)
  from mild diffuse noise (kept below the erasure budget so it doesn't
  cannibalize Reed-Solomon's blind-correction capacity).
- **Mild-to-moderate Gaussian noise** (sigma~10) and **Gaussian
  smoothing filters**: recover reliably.
- **Median filtering and mild (0.75x) geometric scaling**: DWT was
  revised mid-project to embed in the coarse 2-level wavelet
  approximation sub-band instead of the detail sub-bands, and DCT's
  coefficient-pair selection was capped to a frequency band measured
  robust to both attacks; LSB and DWT's payload shares were both capped
  small (`pipeline.DEFAULT_FRAGILE_METHOD_BUDGETS`) since even a small
  share of either was measured to tip an otherwise-passing recovery into
  failing. Recovers reliably now, matching median-filter and mild-scaling
  resilience the report doesn't even explicitly call out as difficult.
- **Cryptographic integrity**: wrong shared secret, or any tampering with
  the ciphertext/tag, is always correctly rejected (`AideAuthenticationError`).
- **Color (RGB) cover images**: converted to YCbCr and embedded in the Y
  (luma) plane only, with Cb/Cr chroma carried through untouched and
  recombined on save - full color fidelity outside the embedded plane,
  survives a save/reload round trip through a real color PNG and a
  compression attack applied to the color image. `aide embed`/`extract`/
  `attack` all auto-detect grayscale vs. color and handle both.

## Detection: blind steganalysis (defence & intelligence)

`aide detect <image>` answers a genuinely different question than
`extract`: given an image with **no known secret, no sidecar, and no
assumption about whether or how it was modified**, does it look like it's
carrying hidden data? This is the report's own border/port screening and
cyber-defence-command-centre use case (section 7) - a counter-
intelligence analyst's question, not a covert-communication operator's.

**How it works** (`aide/steganalysis.py`): the classical chi-square
pairs-of-values attack (Westfeld & Pfitzmann, 2001) as the primary,
statistically well-powered signal, corroborated by two weaker signals
(LSB/next-bit-plane correlation, LSB bit-balance). Combined into a
suspicion score and a `clean` / `suspicious` / `likely_steganographic`
verdict.

**Measured, not assumed** (`tests/test_steganalysis.py`):
- **False-positive rate on real, clean images: 1/40 (2.5%)** - consistent
  with the statistical test's own significance level, not a systemic
  bias. An earlier version of the scoring, which folded a per-row-band
  breakdown into the verdict to catch partially-embedded images, was
  caught by testing to inflate this to **16/20 false positives** (taking
  the max of 8 correlated statistical tests without a multiple-testing
  correction is itself unsound) and was reverted before shipping - a
  detector that cries wolf on most clean images is worse than none for a
  security tool. That row-band breakdown is still surfaced in the output
  as a diagnostic "where might it be" plot, explicitly labeled as not
  factoring into the verdict.
- **Naive, full-image sequential LSB embedding (the classic, still
  common, unsophisticated case): reliably detected** - 100% at full
  embedding rate, degrading honestly at lower rates (a known, documented
  limitation of the classical whole-image test, not hidden).
- **AIDE's own adaptive stego images evade this detector entirely
  (0/15 flagged)** - a real, verified answer to the report's own
  "Detectability (RS analysis): Negative" target claim, for this specific
  classical tool. AIDE's LSB payload share is capped under 1% (too little
  to shift the whole-image pair-equalization statistic this test looks
  for) and the rest lives in DCT/DWT coefficients this spatial-LSB-only
  detector cannot see at all. This is a scoped, honest claim, not a
  blanket one: it says nothing about detectability by more sophisticated
  tools (Sample Pairs Analysis, RS rate estimation, transform-domain
  steganalysis, or a trained ML classifier) - see "What doesn't work yet"
  below for why those weren't attempted.

## What doesn't work yet

Documented rather than papered over, per this project's own "measure,
don't assume" rule:

- **LSB is fragile to everything** (~45-55% bit-error rate under every
  attack tried, including a plain re-save). This is real and expected —
  LSB depends on exact pixel values. It's capped at a token 0.5% of the
  payload (`pipeline.DEFAULT_FRAGILE_METHOD_BUDGETS`) specifically so it
  doesn't dominate a codeword's error budget while staying visible in the
  embedding map; earlier versions of this pipeline let it reach >50% of
  payload bits and silently broke compression resilience as a result.
- **Median filtering is boundary-sensitive, not fully guaranteed.** After
  the fixes above, a representative median-filter recovery test now sits
  right at Reed-Solomon's correction threshold (t=16 errors per codeword)
  instead of far past it - it passes on most runs but can still fail
  depending on tiny floating-point rounding differences in scipy's DCT
  (`tests/test_attacks_recovery.py::test_median_filter_recovers` is
  marked `xfail(strict=False)` for exactly this reason: usually an
  unexpected pass, occasionally a real failure). A safety margin in the
  capacity allocator, not yet implemented, would close this gap for good.
- **Aggressive (0.5x) geometric scaling still fails outright** - and per
  the report's own text, this is expected: "Scaling attacks are the most
  damaging" (report section 2.3). Measured per-codeword error counts are
  well past the correction budget even with DCT carrying nearly the
  entire payload, unlike median filtering's boundary case above.
- **Gaussian noise at sigma=20 (report O4's explicit target) and salt &
  pepper noise both fail at the default DCT delta** - and this one *is* a
  quick, deliberate dial rather than an unsolved problem: measured
  per-codeword error counts at sigma=20 (~100-120 out of 255 bytes) drop
  to full recoverability once `DELTA` is raised from 40 to ~150, at a
  real, measured image-quality cost (block PSNR ~30dB -> ~20dB - see the
  table in `aide/embedding/dct.py`). This project ships the smaller
  default because 40 already fully covers JPEG compression and median
  filtering without paying that quality cost for a noise level nothing
  else in the pipeline (LSB, DWT) is tuned to survive either - raising it
  to actually hit O4's target end-to-end would need matching increases
  across all three methods, which is future work, not a one-line change.
- Heavier attacks still fail regardless of delta: >~2-3% cropping/
  splicing relative to how many RS codewords the message spans exceeds
  the erasure budget outright (detection still works - see
  `test_heavy_cropping_may_exceed_erasure_budget_but_is_detected`).
- **Single-codeword messages get zero interleaving benefit.** RS(255,223)
  always produces at least one full 255-byte codeword no matter how short
  the plaintext is, and interleaving only redistributes bytes *within*
  existing codewords — with one codeword, every byte already belongs to
  it regardless of where it's embedded spatially. Use a message long
  enough to span multiple codewords (roughly 200+ plaintext bytes) to see
  the interleaving defense actually do something; `aide evaluate` and
  `scripts/demo.py` default to one for this reason.
- **Trained on 334 real BOSSbase images (~342,000 blocks)**, not the
  report's 80,000+-image ALASKA2 corpus. Training is fast and scales
  roughly linearly (measured: 20s at 50,000 blocks -> 0.875 validation
  accuracy, 2.5 minutes at the full ~342k -> 0.877) - `aide train
  --images <dir> --max-images N --max-blocks N` accepts any local image
  folder, so fetching more BOSSbase images (or pointing at ALASKA2/your
  own images) is a data problem now, not a code change.
- **`aide detect`'s blind steganalysis only catches naive, spatial-LSB
  steganography.** Sample Pairs Analysis and RS (Regular-Singular) rate
  estimation - two more sophisticated blind detectors from the
  literature, RS analysis by name in the report - were deliberately not
  implemented rather than shipped from a possibly-subtly-wrong memory of
  their derivation (both involve non-trivial quadratic-equation fits this
  project had no reference implementation to validate against). DCT/DWT-
  coefficient-domain steganalysis (which could in principle detect AIDE's
  own transform-domain payload) wasn't attempted at all. See
  `aide/steganalysis.py`'s module docstring.

## Technology stack

Matches the report's section 8: OpenCV, scikit-learn, NumPy,
pycryptodome, reedsolo, scipy, PyWavelets, Pillow, scikit-image, pytest,
requests, psutil, joblib. CPU-only throughout; no GPU/CUDA dependency.
