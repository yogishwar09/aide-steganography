"""Fetch a small real-image dataset for training/demoing AIDE.

Primary source: BOSSbase 1.01 (10,000 grayscale 512x512 PGM images), pulled
as a *subset* via HTTP range requests so we don't have to download the full
~1.67 GB archive. If the remote host is unreachable, falls back to the
photographic sample images bundled with scikit-image so the rest of the
pipeline always has something real to work with.

Usage:
    python scripts/fetch_dataset.py --count 300
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import requests

BOSSBASE_URL = "http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"
TIMEOUT = 30


class HTTPRangeFile(io.RawIOBase):
    """A minimal seekable/readable file-like object backed by HTTP Range
    requests, so zipfile.ZipFile can read a remote archive's central
    directory and individual members without downloading the whole file."""

    def __init__(self, url: str, session: requests.Session):
        self.url = url
        self.session = session
        self._pos = 0
        head = self.session.head(url, timeout=TIMEOUT, allow_redirects=True)
        head.raise_for_status()
        length = head.headers.get("Content-Length")
        if length is None or head.headers.get("Accept-Ranges", "").lower() != "bytes":
            raise RuntimeError("Remote server does not support ranged requests")
        self.length = int(length)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self.length + offset
        else:
            raise ValueError(f"invalid whence {whence}")
        return self._pos

    def tell(self) -> int:
        return self._pos

    def readinto(self, b: bytearray) -> int:
        want = len(b)
        if want == 0 or self._pos >= self.length:
            return 0
        end = min(self._pos + want, self.length) - 1
        resp = self.session.get(
            self.url, headers={"Range": f"bytes={self._pos}-{end}"}, timeout=TIMEOUT
        )
        resp.raise_for_status()
        data = resp.content
        n = len(data)
        b[:n] = data
        self._pos += n
        return n


def fetch_boss_subset(count: int, out_dir: Path) -> int:
    session = requests.Session()
    remote = HTTPRangeFile(BOSSBASE_URL, session)
    zf = zipfile.ZipFile(remote)
    names = sorted(n for n in zf.namelist() if n.lower().endswith((".pgm", ".png", ".jpg")))
    if not names:
        raise RuntimeError("No image members found in remote archive")
    chosen = names[:count]
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for name in chosen:
        data = zf.read(name)
        dest = out_dir / Path(name).name
        dest.write_bytes(data)
        written += 1
        if written % 25 == 0:
            print(f"  ...{written}/{len(chosen)} images fetched", file=sys.stderr)
    return written


def fetch_skimage_fallback(out_dir: Path) -> int:
    from skimage import data as skdata
    from PIL import Image
    import numpy as np

    names = [
        "astronaut", "camera", "coffee", "chelsea", "coins", "horse",
        "brick", "grass", "gravel", "rocket", "hubble_deep_field",
        "moon", "page", "text", "checkerboard",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for name in names:
        loader = getattr(skdata, name, None)
        if loader is None:
            continue
        try:
            arr = loader()
        except Exception:
            continue
        arr = np.asarray(arr)
        if arr.ndim == 2:
            img = Image.fromarray(arr).convert("L")
        elif arr.ndim == 3:
            img = Image.fromarray(arr[..., :3]).convert("RGB")
        else:
            continue
        img.save(out_dir / f"{name}.png")
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=300, help="number of BOSSbase images to fetch")
    parser.add_argument("--out", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    print(f"Attempting BOSSbase 1.01 subset fetch ({args.count} images) via HTTP range requests...")
    try:
        n = fetch_boss_subset(args.count, args.out)
        print(f"SOURCE=bossbase COUNT={n} DIR={args.out}")
        return
    except Exception as exc:  # noqa: BLE001
        print(f"BOSSbase fetch failed ({exc!r}); falling back to scikit-image sample images.", file=sys.stderr)

    n = fetch_skimage_fallback(args.out)
    print(f"SOURCE=skimage-fallback COUNT={n} DIR={args.out}")


if __name__ == "__main__":
    main()
