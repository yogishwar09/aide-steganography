# Changelog

## Unreleased

### Deployment
- **Render Blueprint** (`render.yaml`): free-plan Docker deploy with one
  worker, `/healthz` health check, a generated `AIDE_API_KEY` and
  auto-deploy on push. gunicorn's worker count now comes from
  `WEB_CONCURRENCY` (default 2 in the image).
- Live at https://aide-l64i.onrender.com. On the free plan's fractional CPU
  a full Live Demo mission takes about 35 s (about 4.5 s on a laptop);
  single-threading the numeric libraries was measured and made no
  difference, so the limit is the CPU share itself.
- **Hugging Face Spaces deploy** (`scripts/deploy_space.py`, `make deploy`):
  creates a Docker Space, uploads only what the image needs, and stores a
  generated `AIDE_API_KEY` as a Space secret.
- The trained models in `data/models` are now versioned and baked into the
  Docker image, so a deployment runs the trained engine and ML detector
  instead of falling back to the formula engine.
- Runtime versions are pinned in `constraints.txt` (scikit-learn pickles are
  version-sensitive).
- `AIDE_API_KEY` now gates the versioned `/api/v1` endpoints only; the
  bundled console stays usable for anyone with the link.

### Web console
- **Live Demo page** (now the console's landing page): one click runs the
  whole mission on a bundled cover — encrypt, Reed-Solomon encode, adaptive
  embed, the 16-scenario attack gauntlet, recovery through a chained
  noise + JPEG channel, and analyst screening of cover vs stego — then shows
  a scoreboard (attacks survived, PSNR/SSIM, recovery, detector verdicts,
  end-to-end time). Profile and blind/sidecar mode are selectable, and every
  number is measured live.
- `GET /api/v1/demo/covers` serves the bundled demo covers
  (`webapp/static/demo/`); a test guarantees each one survives every
  scenario under `robust`/blind.
- Sidebar status pill reports live engine/detector state from `/healthz`.
- **Vision & Roadmap page**: field challenges and how AIDE answers them,
  the 0.1 → 0.2 engineering journey with measured before/after recovery
  rates, open challenges with their path forward, a three-phase roadmap,
  application domains, and a responsible-use note. The headline recovery
  figure is read live from the committed benchmark.
- The roadmap is now centred on **Major Project Phase 2** — a one-month,
  four-week plan (deploy & secure; real-world channels & colour; geometric
  resilience & bigger data; evaluate & document), each week with tasks, a
  deliverable and a measurable target, plus a today → end-of-Phase-2 targets
  table. Long-term scope follows as 3–6 month, 6–12 month and 1–2 year
  horizons, and open challenges are tagged with the Phase 2 week that
  addresses them.

## 0.2.0 — 2026-10-11

Everything below is measured on real BOSSbase covers (`aide benchmark`,
`reports/benchmark/benchmark.md`). See README for the full tables.

### Robustness
- **Robustness profiles** (`aide/profiles.py`): `quality`, `balanced`
  (default), `robust`, `fortress`. Each bundles the DCT delta, the RS parity,
  the method budgets and a preferred bpp cap, and is recorded in the sidecar
  or blind header. `robust` meets report O4 end-to-end (σ=20 noise and JPEG
  Q50, 24/24 covers).
- **Reliability-guided (GMD) Reed-Solomon decoding** (`ecc.decode_with_reliability`):
  soft per-bit margins from new `extract_soft` extractors, with
  minimum-distance candidate selection.
- **Shortened RS codewords** (`ecc.plan`): short messages now span ≥4
  codewords, so interleaving protects them. Parity is configurable.
- Payload is routed away from LSB and DWT (measured: ~40% of median-filter
  errors on ~2% of bits). The preferred low-frequency layout is relaxed
  only when a message would not fit.
- Result for the default profile vs 0.1 on the same 30 covers: median filter
  69→100%, 0.5× rescale 0→92%, 5% splice 85→100%, 10% crop 50→88%, at
  +0.4 dB PSNR.

### New capabilities
- **Blind mode** (`aide/blind.py`, `--mode blind`): self-contained stego
  that needs only the secret. Keyed block layout, compact RS-protected
  header with soft-combined repetition, saturation-aware strength and
  reliability.
- **Exact capacity** (`aide capacity`, `/api/capacity`) per profile and
  mode, from one feature pass.
- **Quantitative steganalysis** (`aide/quantitative.py`): Sample Pairs
  Analysis and RS analysis, validated against known embedding rates and
  calibrated to a ~2.5% false-positive rate. Now part of the `detect`
  verdict, with an estimated payload size.
- **Trained ML steganalyser** (`aide/mlsteg.py`, `aide train-detector`):
  SPAM + DCT pair-gap features, gradient boosting, and an IsolationForest
  cover-source-mismatch guard.
- **Tiny-MLP decision engine** option (`aide train --classifier mlp`).
- **Benchmark harness** (`aide benchmark`): parallel multi-image recovery
  matrix plus inference-speed / footprint measurement (report O1, O7).
- **Folder screening** (`aide detect <dir> --csv`) and batch screening in
  the web console.

### Security
- **scrypt key derivation** (N=2¹⁴, r=8, p=1) replaces single SHA-256 for
  new embeddings; the KDF is recorded in the sidecar.
- `--secret-file` keeps secrets out of shell history.
- Web: optional `AIDE_API_KEY` (constant-time check), CSP and anti-framing
  headers, request IDs, JSON errors for every failure path.
- Honest finding, documented: AIDE evades the classical blind detectors but
  **is detected ~100% by the targeted ML steganalyser** (AUC ≥ 0.998).

### Engineering
- Versioned REST API (`/api/v1/*`) with an OpenAPI spec, plus `/healthz`.
- gunicorn-ready app, `Dockerfile` (non-root, health check), `Makefile`,
  GitHub Actions CI (lint, tests on 3.10–3.12, benchmark smoke test, Docker
  build).
- `pyproject.toml` extras (`web`, `data`, `report`, `dev`) and ruff config;
  the code base is lint-clean.
- Feature extraction is 3× faster (vectorised GLCM energy, entropy and
  variance; numerically identical to the per-block definitions).
- 146 tests (was 85), including a real AIDE 0.1 stego + sidecar fixture that
  must keep decoding.

### Compatibility
- `aide-0.1` sidecars decode unchanged (legacy RS(255,223), SHA-256 KDF).
- The `/api/*` request and response fields used by 0.1 clients are kept;
  new fields are additive.

## 0.1.0

Initial prototype: AIDE decision engine, AES-256-GCM, RS(255,223) +
interleaving, LSB/DCT/DWT embedding, tamper localization, chi-square
steganalysis, CLI and local web console.
