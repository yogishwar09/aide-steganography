"""Reed-Solomon resilience layer with block interleaving.

Report section 2.1 step 6: RS(255,223), t=16 byte-error-correction capacity
per 255-byte codeword (~14% redundancy). Bytes are interleaved across the
embedding order so that spatially localized damage (a crop, a splice) is
spread thinly across many codewords instead of destroying a few codewords
completely - each codeword only needs to lose <=t bytes (or, with known
erasure positions from tamper localization, <=2t bytes) to stay correctable.

Two generalisations of the report's fixed RS(255,223) code, both carried
in `ECCParams` and recorded in the sidecar/header so the receiver decodes
with exactly what the sender used:

1. Configurable parity (`nsym`). RS(255,223) is the default, but the
   robustness profiles in `aide.profiles` trade capacity for a larger
   correction budget (e.g. RS(255,159), t=48) - the difference between
   failing and passing report O4's Gaussian sigma=20 target, measured in
   `scripts/benchmark.py`.
2. Shortened codewords for short messages. A fixed 255-byte codeword
   means any message under ~219 bytes fits in ONE codeword, and
   interleaving can't spread damage across codewords that don't exist
   (this used to be a documented limitation). `plan()` instead splits the
   payload into at least `min_codewords` shortened RS(n, k) codewords with
   the *same parity ratio*, so the per-codeword correction fraction is
   unchanged while localized damage gets spread thin again. Shortened RS
   codes are standard practice (e.g. DVB, QR codes) and reedsolo supports
   them natively via `nsize`.
"""
from __future__ import annotations

import math
import struct
from dataclasses import asdict, dataclass

import numpy as np
from reedsolo import ReedSolomonError, RSCodec

NSYM = 32  # default parity bytes per codeword -> t=16 correctable errors, or 32 known erasures
DATA_LEN = 223  # default RS(255,223) message bytes per codeword
CODEWORD_LEN = DATA_LEN + NSYM  # 255
MAX_CODEWORD_LEN = 255  # GF(2^8) limit
LENGTH_HEADER = 4  # bytes, big-endian uint32 original payload length, RS-protected too


class AideECCError(Exception):
    """Raised when Reed-Solomon decoding fails beyond correction capacity."""


@dataclass(frozen=True)
class ECCParams:
    nsym: int = NSYM
    data_len: int = DATA_LEN

    def __post_init__(self):
        if self.nsym < 2 or self.data_len < 1 or self.codeword_len > MAX_CODEWORD_LEN:
            raise ValueError(f"invalid RS parameters: n={self.codeword_len}, k={self.data_len}")

    @property
    def codeword_len(self) -> int:
        return self.data_len + self.nsym

    @property
    def t(self) -> int:
        """Blind (unknown-position) byte errors correctable per codeword."""
        return self.nsym // 2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> ECCParams:
        return cls(**d) if d else cls()


DEFAULT_PARAMS = ECCParams()


def plan(payload_len: int, nsym: int = NSYM, min_codewords: int = 4) -> ECCParams:
    """Choose shortened-codeword parameters for a payload of `payload_len`
    bytes at the parity ratio of a full RS(255, 255-nsym) code, using at
    least `min_codewords` codewords when the payload is small enough that
    one full-length codeword would otherwise hold all of it."""
    full_k = MAX_CODEWORD_LEN - nsym
    ratio = nsym / full_k
    total = payload_len + LENGTH_HEADER
    m = max(min_codewords, math.ceil(total / full_k))
    k = max(1, math.ceil(total / m))
    k = min(k, full_k)
    parity = max(2, math.ceil(k * ratio))
    parity += parity % 2  # even nsym -> t = nsym/2 exactly
    parity = min(parity, MAX_CODEWORD_LEN - k)
    return ECCParams(nsym=parity, data_len=k)


def _codec(params: ECCParams) -> RSCodec:
    return RSCodec(params.nsym, nsize=params.codeword_len)


def _pad_to_codewords(data: bytes, params: ECCParams) -> bytes:
    payload = struct.pack(">I", len(data)) + data
    remainder = len(payload) % params.data_len
    if remainder:
        payload += bytes(params.data_len - remainder)
    return payload


def encode(data: bytes, params: ECCParams = DEFAULT_PARAMS) -> bytes:
    """RS-encode `data`. Output length is always an exact multiple of the
    codeword length, which is what makes the interleaving matrix below
    well defined."""
    padded = _pad_to_codewords(data, params)
    encoded = bytes(_codec(params).encode(padded))
    assert len(encoded) % params.codeword_len == 0
    return encoded


def decode(encoded: bytes, erase_positions: list[int] | None = None,
           params: ECCParams = DEFAULT_PARAMS) -> bytes:
    """Reverse `encode`, correcting up to t errors or 2t known erasures per
    codeword. `erase_positions` are byte indices into `encoded` (codeword
    order, i.e. after de-interleaving) that tamper localization flagged as
    unreliable."""
    try:
        rmes, _rmesecc, _errata = _codec(params).decode(bytearray(encoded), erase_pos=erase_positions)
    except ReedSolomonError as exc:
        raise AideECCError(f"Reed-Solomon decoding failed beyond correction capacity: {exc}") from exc
    rmes = bytes(rmes)
    if len(rmes) < LENGTH_HEADER:
        raise AideECCError("decoded payload shorter than length header")
    (orig_len,) = struct.unpack(">I", rmes[:LENGTH_HEADER])
    payload = rmes[LENGTH_HEADER:]
    if orig_len > len(payload):
        raise AideECCError("decoded length header inconsistent with recovered payload size")
    return payload[:orig_len]


def build_interleave_permutation(num_codewords: int, codeword_len: int = CODEWORD_LEN) -> np.ndarray:
    """perm such that interleaved[k] = encoded[perm[k]]. Reading the
    codeword matrix column-major (byte 0 of every codeword, then byte 1 of
    every codeword, ...) means bytes that end up spatially adjacent in the
    embedding order belong to *different* codewords."""
    idx = np.arange(num_codewords * codeword_len).reshape(num_codewords, codeword_len)
    return idx.T.reshape(-1)


def _perm_for(n: int, codeword_len: int) -> np.ndarray:
    if n % codeword_len != 0:
        raise ValueError("data must be a multiple of the codeword length")
    return build_interleave_permutation(n // codeword_len, codeword_len)


def interleave(encoded: bytes, codeword_len: int = CODEWORD_LEN) -> bytes:
    perm = _perm_for(len(encoded), codeword_len)
    arr = np.frombuffer(encoded, dtype=np.uint8)
    return arr[perm].tobytes()


def deinterleave(interleaved: bytes, codeword_len: int = CODEWORD_LEN) -> bytes:
    """Inverse of `interleave`, recovering original codeword-order bytes."""
    perm = _perm_for(len(interleaved), codeword_len)
    arr = np.frombuffer(interleaved, dtype=np.uint8)
    out = np.empty_like(arr)
    out[perm] = arr
    return out.tobytes()


def deinterleave_mask(interleaved_mask: np.ndarray, codeword_len: int = CODEWORD_LEN) -> np.ndarray:
    """Apply the same permutation as `deinterleave` to a boolean array that
    was recorded in interleaved (spatial embedding) order - e.g. a per-byte
    'unreliable' flag from tamper localization - so it lines up with
    codeword-order bytes for use as `erase_positions` in `decode`."""
    perm = _perm_for(len(interleaved_mask), codeword_len)
    out = np.empty_like(interleaved_mask)
    out[perm] = interleaved_mask
    return out


@dataclass
class DecodeStats:
    codewords: int = 0
    corrected_bytes: int = 0      # total byte positions repaired (errors + erasures)
    gmd_codewords: int = 0        # codewords that needed reliability-guided erasures
    failed_codewords: int = 0


def decode_with_reliability(encoded: bytes, reliability: np.ndarray | None = None,
                            erase_positions: list[int] | None = None,
                            params: ECCParams = DEFAULT_PARAMS,
                            stats: DecodeStats | None = None) -> bytes:
    """Generalized-minimum-distance (GMD) style decoding, one codeword at a
    time (Forney, 1966).

    A plain RS decoder treats every byte as equally trustworthy, so it can
    fix at most t = nsym/2 unknown errors. But the transform-domain
    extractors know *how confidently* each bit was read (the coefficient
    pair's margin - see `aide.embedding._common.pair_margins`): bytes that
    an attack pushed near the decision boundary are where errors cluster.
    Declaring the f least-reliable bytes as erasures lets RS correct any
    pattern with 2*errors + erasures <= nsym, so when the low-reliability
    guesses are right, up to 2t errors become correctable instead of t.

    Procedure per codeword: try a plain decode (plus any tamper-localized
    erasures); on failure, retry with the 2, 4, 6, ... least-reliable
    remaining bytes erased until it succeeds or the parity budget is used
    up. A wrong-but-valid codeword (mis-correction) is still caught later
    by the AES-GCM tag, so this cannot silently return a wrong message.

    `reliability` is per byte in codeword order (higher = more trusted);
    `erase_positions` are known-bad positions from tamper localization.
    """
    n = params.codeword_len
    if len(encoded) % n:
        raise ValueError("data must be a multiple of the codeword length")
    codec = _codec(params)
    known = np.zeros(len(encoded), dtype=bool)
    if erase_positions:
        known[np.asarray(erase_positions, dtype=np.int64)] = True
    rel = np.full(len(encoded), np.inf) if reliability is None else np.asarray(reliability, dtype=np.float64)
    stats = stats if stats is not None else DecodeStats()

    data_parts = []
    for c in range(len(encoded) // n):
        cw = bytearray(encoded[c * n:(c + 1) * n])
        base = list(np.where(known[c * n:(c + 1) * n])[0])
        if len(base) > params.nsym:
            # More flagged bytes than RS can erase: keep the least reliable.
            base = sorted(base, key=lambda i: rel[c * n + i])[:params.nsym]
        base_set = set(base)
        candidates = [i for i in np.argsort(rel[c * n:(c + 1) * n], kind="stable") if i not in base_set]
        # Evaluate every erasure count and keep the candidate codeword with
        # the smallest reliability-weighted distance to what was received.
        # Taking the *first* decode that succeeds was measured to mis-correct
        # often on short codewords (with f erasures, only nsym - f parity
        # symbols remain to detect a wrong solution): all 13 failures in a
        # 20-trial sigma=20 blind-mode test were wrong-but-valid codewords
        # that the GCM tag then rejected. Minimum-distance selection (what
        # makes GMD a maximum-likelihood-style decoder rather than a greedy
        # one) lifted that test from 7/20 to 9/20; blind mode's saturation
        # weighting closed the rest.
        received = np.frombuffer(bytes(cw), dtype=np.uint8)
        cw_rel = np.where(np.isfinite(rel[c * n:(c + 1) * n]), rel[c * n:(c + 1) * n], 1.0)
        decoded, best_cost, best_f, best_fixed = None, np.inf, 0, 0
        max_f = params.nsym - len(base) if reliability is not None else 0
        for f in range(0, max_f + 1, 2):
            erase = base + [int(i) for i in candidates[:f]]
            try:
                msg, full, errata = codec.decode(cw, erase_pos=erase or None)
            except ReedSolomonError:
                continue
            changed = np.frombuffer(bytes(full), dtype=np.uint8) != received
            cost = float(cw_rel[changed].sum())
            if cost < best_cost:
                decoded, best_cost, best_f, best_fixed = msg, cost, f, int(changed.sum())
        stats.codewords += 1
        if decoded is None:
            stats.failed_codewords += 1
            raise AideECCError(
                f"Reed-Solomon decoding failed beyond correction capacity in codeword {c} "
                f"(nsym={params.nsym}, {len(base)} tamper erasures)")
        stats.corrected_bytes += best_fixed
        if best_f:
            stats.gmd_codewords += 1
        data_parts.append(bytes(decoded))

    rmes = b"".join(data_parts)
    if len(rmes) < LENGTH_HEADER:
        raise AideECCError("decoded payload shorter than length header")
    (orig_len,) = struct.unpack(">I", rmes[:LENGTH_HEADER])
    payload = rmes[LENGTH_HEADER:]
    if orig_len > len(payload):
        raise AideECCError("decoded length header inconsistent with recovered payload size")
    return payload[:orig_len]
