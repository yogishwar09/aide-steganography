"""Reed-Solomon RS(255,223) resilience layer with block interleaving.

Report section 2.1 step 6: RS(255,223), t=16 byte-error-correction capacity
per 255-byte codeword (~14% redundancy). Bytes are interleaved across the
embedding order so that spatially localized damage (a crop, a splice) is
spread thinly across many codewords instead of destroying a few codewords
completely - each codeword only needs to lose <=16 bytes (or, with known
erasure positions from tamper localization, <=32 bytes) to stay correctable.
"""
from __future__ import annotations

import struct

import numpy as np
from reedsolo import RSCodec, ReedSolomonError

NSYM = 32  # parity bytes per codeword -> t=16 correctable errors, or 32 known erasures
DATA_LEN = 223  # RS(255,223) message bytes per codeword
CODEWORD_LEN = DATA_LEN + NSYM  # 255
LENGTH_HEADER = 4  # bytes, big-endian uint32 original payload length, RS-protected too


class AideECCError(Exception):
    """Raised when Reed-Solomon decoding fails beyond correction capacity."""


def _codec() -> RSCodec:
    return RSCodec(NSYM, nsize=CODEWORD_LEN)


def _pad_to_codewords(data: bytes) -> bytes:
    payload = struct.pack(">I", len(data)) + data
    remainder = len(payload) % DATA_LEN
    if remainder:
        payload += bytes(DATA_LEN - remainder)
    return payload


def encode(data: bytes) -> bytes:
    """RS(255,223)-encode `data`. Output length is always an exact multiple
    of 255 (one full codeword per 223 padded input bytes), which is what
    makes the interleaving matrix below well defined."""
    padded = _pad_to_codewords(data)
    encoded = bytes(_codec().encode(padded))
    assert len(encoded) % CODEWORD_LEN == 0
    return encoded


def decode(encoded: bytes, erase_positions: list[int] | None = None) -> bytes:
    """Reverse `encode`, correcting up to 16 errors or 32 known erasures per
    255-byte codeword. `erase_positions` are byte indices into `encoded`
    (codeword order, i.e. after de-interleaving) that tamper localization
    flagged as unreliable."""
    try:
        rmes, _rmesecc, _errata = _codec().decode(bytearray(encoded), erase_pos=erase_positions)
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


def build_interleave_permutation(num_codewords: int) -> np.ndarray:
    """perm such that interleaved[k] = encoded[perm[k]]. Reading the
    codeword matrix column-major (byte 0 of every codeword, then byte 1 of
    every codeword, ...) means bytes that end up spatially adjacent in the
    embedding order belong to *different* codewords."""
    idx = np.arange(num_codewords * CODEWORD_LEN).reshape(num_codewords, CODEWORD_LEN)
    return idx.T.reshape(-1)


def interleave(encoded: bytes) -> bytes:
    n = len(encoded)
    if n % CODEWORD_LEN != 0:
        raise ValueError("encoded data must be a multiple of the codeword length")
    perm = build_interleave_permutation(n // CODEWORD_LEN)
    arr = np.frombuffer(encoded, dtype=np.uint8)
    return arr[perm].tobytes()


def deinterleave(interleaved: bytes) -> bytes:
    """Inverse of `interleave`, recovering original codeword-order bytes."""
    n = len(interleaved)
    if n % CODEWORD_LEN != 0:
        raise ValueError("interleaved data must be a multiple of the codeword length")
    perm = build_interleave_permutation(n // CODEWORD_LEN)
    arr = np.frombuffer(interleaved, dtype=np.uint8)
    out = np.empty_like(arr)
    out[perm] = arr
    return out.tobytes()


def deinterleave_mask(interleaved_mask: np.ndarray) -> np.ndarray:
    """Apply the same permutation as `deinterleave` to a boolean array that
    was recorded in interleaved (spatial embedding) order - e.g. a per-byte
    'unreliable' flag from tamper localization - so it lines up with
    codeword-order bytes for use as `erase_positions` in `decode`."""
    n = len(interleaved_mask)
    if n % CODEWORD_LEN != 0:
        raise ValueError("mask length must be a multiple of the codeword length")
    perm = build_interleave_permutation(n // CODEWORD_LEN)
    out = np.empty_like(interleaved_mask)
    out[perm] = interleaved_mask
    return out
