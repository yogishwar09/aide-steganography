"""Publish the AIDE console + REST API to a Hugging Face Space (Docker SDK).

The Space builds the repository's Dockerfile, so it runs exactly what CI
builds and health-checks. Only the files the image needs are uploaded
(no tests, docs, datasets or local scratch images), with a Space-specific
README whose YAML header tells Hugging Face how to run the container.

One-time setup (by you, in a terminal):
    pip install -e ".[deploy]"
    hf auth login            # paste a *write* token from huggingface.co/settings/tokens

Then:
    python scripts/deploy_space.py                    # -> <your-username>/aide
    python scripts/deploy_space.py --space me/aide-demo

The versioned /api/v1 endpoints are gated by an AIDE_API_KEY Space secret.
On first deploy a random key is generated, stored as that secret and saved
to ./.space_api_key (gitignored); later deploys reuse it.
"""
from __future__ import annotations

import argparse
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KEY_FILE = REPO_ROOT / ".space_api_key"
GITHUB_URL = "https://github.com/yogishwar09/aide-steganography"

# Everything the Dockerfile COPYs, and nothing else.
SPACE_FILES = ["Dockerfile", "pyproject.toml", "constraints.txt", "aide", "webapp", "reports", "data/models"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")

SPACE_README = """---
title: AIDE Steganography Console
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 5050
pinned: false
short_description: Adaptive, encrypted, attack-robust image steganography
---

# AIDE — Adaptive Intelligent Dynamic Embedding

CPU-only adaptive image steganography: AES-256-GCM encryption, Reed-Solomon
error correction with soft-decision decoding, a learned per-block embedding
policy, tamper localization and a steganalysis toolkit.

Open the **Live Demo** page and click **Run full mission** to watch a
message get encrypted, embedded, pushed through 16 attacks, recovered and
screened — every number measured live.

Source, benchmarks and documentation: {github}

The versioned REST API (`/api/v1`, contract at `/api/v1/openapi.json`)
requires an `X-API-Key` header.
"""


def git_is_clean() -> bool:
    out = subprocess.run(["git", "status", "--porcelain", "--", *SPACE_FILES],
                         cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    return not out.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", help="Space id <user>/<name> (default: <your-username>/aide)")
    parser.add_argument("--private", action="store_true", help="create the Space as private")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="deploy even if the deployed files have uncommitted changes")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import LocalTokenNotFoundError
    except ImportError:
        print('huggingface_hub is not installed: pip install -e ".[deploy]"', file=sys.stderr)
        return 1

    if not args.allow_dirty and not git_is_clean():
        print("Deployed files have uncommitted changes; commit them first or pass --allow-dirty.",
              file=sys.stderr)
        return 1
    if not (REPO_ROOT / "data" / "models" / "weights.json").exists():
        print("data/models is missing - run `aide train` and `aide train-detector` first.", file=sys.stderr)
        return 1

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except LocalTokenNotFoundError:
        print("Not logged in to Hugging Face. Run `hf auth login` with a write token, then retry.",
              file=sys.stderr)
        return 1
    space = args.space or f"{user}/aide"

    api.create_repo(space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)

    if KEY_FILE.exists():
        api_key = KEY_FILE.read_text().strip()
    else:
        api_key = secrets.token_urlsafe(32)
        KEY_FILE.write_text(api_key + "\n")
        KEY_FILE.chmod(0o600)
    api.add_space_secret(space, "AIDE_API_KEY", api_key)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for name in SPACE_FILES:
            src, dst = REPO_ROOT / name, stage / name
            if src.is_dir():
                shutil.copytree(src, dst, ignore=IGNORE)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        (stage / "README.md").write_text(SPACE_README.format(github=GITHUB_URL), encoding="utf-8")
        api.upload_folder(folder_path=stage, repo_id=space, repo_type="space",
                          commit_message=f"Deploy AIDE from {commit}", delete_patterns="*")

    host = space.replace("/", "-").replace("_", "-").lower()
    print(f"Deployed {commit} to https://huggingface.co/spaces/{space}")
    print(f"App URL (live once the image builds, ~3-5 min): https://{host}.hf.space")
    print(f"API key for /api/v1 is stored in {KEY_FILE.name} (gitignored).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
