"""Blind (sidecar-free) mode: a self-describing stego image that needs only
the shared secret to decode.

Sidecar mode (`aide.pipeline`) ships the per-block embedding map, tamper
references and crypto parameters in a separate JSON file - the report's
"transmitted alongside". That is strong (it enables hash-based tamper
localization) but it is a second artifact the receiver must obtain. Blind
mode removes it:

* WHERE - a keyed layout. Blocks are visited in a pseudo-random order
  derived from SHA-256("aide-layout" || secret), so the receiver
  re-derives exactly the same order from the secret and the image size,
  and an adversary without the secret cannot even locate the payload.
* HOW (adaptively) - every block carries DCT coefficient-pair bits at a
  fixed pair count, but the AIDE score sets each block's embedding
  *strength*: textured blocks (high S(b), better perceptual masking) get a
  larger delta, smooth blocks a smaller one. Extraction only reads the
  sign of each pair difference, so the receiver needs no per-block map.
* WHAT - a compact 14-byte header (magic, version+profile, 8-byte random
  salt, ciphertext length) is RS-protected and stored three times with
  soft (margin-summing) combining, followed by the RS-encoded, interleaved
  ciphertext+tag using the profile's parameters. The header is the
  dominant fixed cost of blind mode (it must survive the harshest profile
  before the receiver knows which profile was used), so it is kept small:
  the GCM nonce is derived from the salt rather than stored, which is safe
  because each fresh salt also yields a fresh, single-use key.

Pair count per block (BPP below): blind mode uses DCT's 8 lowest robust
mid-frequency pairs per 16x16 block rather than the full 16. Measured on
12 BOSSbase covers, this lifted 0.5x-rescale survival from 0/12 to 10/12
(balanced) and 12/12 (fortress) and improved median-filter survival, for
~0.4 dB PSNR - the higher-frequency pairs are the ones resampling attacks
destroy (the same effect aide/embedding/dct.py documents).

The random salt stands in for the report's image_hash in the key
derivation (key = SHA-256(salt || secret)): the receiver only ever sees
the stego image, never the pristine cover, so a cover hash could not be
recomputed anyway - see aide/crypto.py's design note.

Trade-off vs sidecar mode, stated plainly: without the sidecar's
reference hashes there is no hash-based tamper map; localized damage is
instead absorbed by reliability-guided RS decoding (damaged blocks read
back with small pair margins and are erased first - aide/ecc.py).
"""
from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import crypto, ecc
from .embedding import dct
from .features import choose_block_size, extract_image_features, partition_blocks
from .model import AIDEModel
from .pipeline import AideCapacityError, LoadedImage, compose_image, load_image
from .profiles import PROFILES, RobustnessProfile, get_profile

MAGIC = 0xAB
VERSION = 1
PROFILE_IDS = {"quality": 1, "balanced": 2, "robust": 3, "fortress": 4}
PROFILE_BY_ID = {v: k for k, v in PROFILE_IDS.items()}
SALT_LEN = 8
_HEADER_FMT = ">BB8sI"  # magic, version<<4 | profile id, salt, ciphertext length
HEADER_LEN = struct.calcsize(_HEADER_FMT)  # 14
# Header: 14 data bytes + 4-byte RS length prefix, 22 parity -> t=11 per 40-byte codeword.
HEADER_ECC = ecc.ECCParams(nsym=22, data_len=HEADER_LEN + ecc.LENGTH_HEADER)
HEADER_REPEATS = 3
BPP = 2  # 8 lowest mid-frequency pairs per 16x16 block - see module docstring
HEADER_DELTA_BOOST = 1.25
# Per-block strength multiplier from the AIDE score S(b) in [0, 1].
STRENGTH_MIN, STRENGTH_SPAN = 0.8, 0.6
SELF_CHECK_RETRIES = 2
SATURATION_BOOST = 1.0


class AideBlindDecodeError(Exception):
    """No decodable AIDE blind payload for this secret (wrong secret, not an
    AIDE image, or damaged beyond the header's correction capacity)."""


@dataclass
class BlindEmbedResult:
    stego: np.ndarray
    mode: str
    chroma: tuple[np.ndarray, np.ndarray] | None
    profile: RobustnessProfile
    block_size: int
    blocks_used: int
    header_bits: int
    payload_bits: int
    capacity_bits: int

    def compose(self):
        return compose_image(self.stego, self.mode, self.chroma)


@dataclass
class BlindRecoverResult:
    success: bool
    plaintext: bytes | None
    error: str | None
    profile: str | None = None
    corrected_bytes: int = 0
    gmd_codewords: int = 0


def _layout(n_blocks: int, secret: bytes, height: int, width: int) -> np.ndarray:
    seed = hashlib.sha256(b"aide-layout|" + secret + struct.pack(">II", height, width)).digest()
    rng = np.random.default_rng(np.frombuffer(seed, dtype=np.uint64))
    return rng.permutation(n_blocks)


def _header_bytes(profile_id: int, salt: bytes, ct_len: int) -> bytes:
    return struct.pack(_HEADER_FMT, MAGIC, (VERSION << 4) | profile_id, salt, ct_len)


def _nonce_from_salt(salt: bytes) -> bytes:
    return hashlib.sha256(b"aide-blind-nonce|" + salt).digest()[:crypto.NONCE_LEN]


def _bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def capacity_bits(height: int, width: int) -> int:
    bs = choose_block_size(height, width)
    return len(partition_blocks(height, width, bs)) * dct.capacity_bits((bs, bs), BPP)


def _payload_plan(ct_len: int, profile: RobustnessProfile) -> ecc.ECCParams:
    return ecc.plan(ct_len + crypto.TAG_LEN, nsym=profile.nsym, min_codewords=profile.min_codewords)


def embed(cover_path: Path, message: bytes, shared_secret: bytes, model: AIDEModel,
          profile: str | RobustnessProfile | None = None) -> BlindEmbedResult:
    prof = get_profile(profile)
    if prof.name not in PROFILE_IDS:
        raise ValueError(f"blind mode supports the built-in profiles {sorted(PROFILE_IDS)}")
    loaded: LoadedImage = load_image(cover_path)
    gray = loaded.y
    h, w = gray.shape

    blocks, features = extract_image_features(gray)
    bs = blocks[0].size if blocks else choose_block_size(h, w)
    scores = np.clip(model.score(features), 0.0, 1.0) if len(blocks) else np.zeros(0)
    per_block = dct.capacity_bits((bs, bs), BPP)

    salt = os.urandom(SALT_LEN)
    enc = crypto.encrypt(message, salt, shared_secret, kdf=crypto.KDF_SCRYPT,
                         nonce=_nonce_from_salt(salt))
    header = ecc.encode(_header_bytes(PROFILE_IDS[prof.name], salt, len(enc.ciphertext)), HEADER_ECC)
    header_bits = np.tile(_bits(header), HEADER_REPEATS)

    params = _payload_plan(len(enc.ciphertext), prof)
    payload = ecc.interleave(ecc.encode(enc.ciphertext + enc.tag, params), params.codeword_len)
    payload_bits = _bits(payload)

    stream = np.concatenate([header_bits, payload_bits])
    capacity = len(blocks) * per_block
    if len(stream) > capacity:
        raise AideCapacityError(
            f"blind mode needs {len(stream)} bits ({len(header_bits)} header + "
            f"{len(payload_bits)} payload at profile {prof.name!r}) but the cover offers "
            f"{capacity} ({len(blocks)} blocks x {per_block}). Use a larger cover, a shorter "
            f"message, or a lighter profile.")

    order = _layout(len(blocks), shared_secret, h, w)
    n_used = -(-len(stream) // per_block)
    padded = np.zeros(n_used * per_block, dtype=np.uint8)
    padded[:len(stream)] = stream
    # Unused tail bits of the last block get random fill, not zeros, so the
    # pair relations there look like everywhere else.
    padded[len(stream):] = np.frombuffer(os.urandom(len(padded) - len(stream)), dtype=np.uint8) & 1
    n_header_blocks = -(-len(header_bits) // per_block)

    stego = gray.copy()
    for k in range(n_used):
        b = blocks[order[k]]
        ys, xs = b.slice()
        chunk = padded[k * per_block:(k + 1) * per_block]
        strength = STRENGTH_MIN + STRENGTH_SPAN * float(scores[b.index])
        if k < n_header_blocks:
            strength = max(strength, HEADER_DELTA_BOOST)
        # Clipping to [0, 255] eats into the margin of partly saturated
        # blocks; pre-compensate (the receiver separately down-weights
        # them - see _saturation_weight).
        sat = float(((gray[ys, xs] <= 2) | (gray[ys, xs] >= 253)).mean())
        delta = prof.dct_delta * strength * (1.0 + SATURATION_BOOST * sat)
        for _ in range(SELF_CHECK_RETRIES + 1):
            out = dct.embed(gray[ys, xs], BPP, chunk, delta=delta)
            # Rounding/clipping to uint8 can collapse a pair in saturated
            # blocks; verify and strengthen rather than ship a known error.
            if np.array_equal(dct.extract(out, BPP), chunk):
                break
            delta *= 1.5
        stego[ys, xs] = out

    return BlindEmbedResult(
        stego=stego, mode=loaded.mode, chroma=loaded.chroma, profile=prof, block_size=bs,
        blocks_used=n_used, header_bits=len(header_bits), payload_bits=len(payload_bits),
        capacity_bits=capacity,
    )


def _saturation_weight(block: np.ndarray) -> float:
    """Reliability weight for a block from what the receiver can see.

    In heavily clipped (pure-black/white) blocks, rounding and clipping to
    [0, 255] eat into the embedded pair margins, so those bits are the
    first to flip under noise (measured: a BOSSbase cover with 34%
    saturated pixels had a 5.7% bit-error rate at sigma=20 vs 0.7-2% for
    typical covers). Scaling their soft values down makes reliability-
    guided RS decoding erase them first and makes header soft-combining
    trust the cleaner copies more - without any side information, since
    saturation is visible in the received image itself."""
    sat = float(((block <= 2) | (block >= 253)).mean())
    return float(np.clip(1.0 - 1.5 * sat, 0.05, 1.0))


def _soft_stream(gray: np.ndarray, shared_secret: bytes, n_bits: int, start_bit: int = 0):
    h, w = gray.shape
    bs = choose_block_size(h, w)
    blocks = partition_blocks(h, w, bs)
    per_block = dct.capacity_bits((bs, bs), BPP)
    order = _layout(len(blocks), shared_secret, h, w)
    first, last = start_bit // per_block, -(-(start_bit + n_bits) // per_block)
    if last > len(blocks):
        raise AideBlindDecodeError("image too small for the declared payload")
    soft = []
    for k in range(first, last):
        b = blocks[order[k]]
        ys, xs = b.slice()
        blk = gray[ys, xs]
        soft.append(dct.extract_soft(blk, BPP) * _saturation_weight(blk))
    flat = np.concatenate(soft) if soft else np.zeros(0)
    off = start_bit - first * per_block
    return flat[off:off + n_bits]


def _decode_bytes(soft: np.ndarray, params: ecc.ECCParams, scale: float,
                  stats: ecc.DecodeStats, interleaved: bool) -> bytes:
    bits = (soft > 0).astype(np.uint8)
    rel = (np.abs(soft) / scale).reshape(-1, 8).min(axis=1)
    data = np.packbits(bits).tobytes()
    if interleaved:
        data = ecc.deinterleave(data, params.codeword_len)
        rel = ecc.deinterleave_mask(rel, params.codeword_len)
    return ecc.decode_with_reliability(data, rel, None, params, stats)


def recover(stego: np.ndarray, shared_secret: bytes) -> BlindRecoverResult:
    stats = ecc.DecodeStats()
    try:
        n_hdr = HEADER_ECC.codeword_len * 8
        soft = _soft_stream(stego, shared_secret, n_hdr * HEADER_REPEATS)
        combined = soft.reshape(HEADER_REPEATS, n_hdr).sum(axis=0)
        # Reliability scale: no absolute delta known before the header is
        # read, so normalise by the median margin magnitude.
        scale = float(np.median(np.abs(combined))) or 1.0
        try:
            raw = _decode_bytes(combined, HEADER_ECC, scale, stats, interleaved=False)
        except ecc.AideECCError as exc:
            raise AideBlindDecodeError(
                "no AIDE blind header found for this secret (wrong secret, not a blind-mode "
                "AIDE image, or the header region is damaged beyond correction)") from exc
        if len(raw) != HEADER_LEN:
            raise AideBlindDecodeError("no AIDE blind header found for this secret")
        magic, ver_pid, salt, ct_len = struct.unpack(_HEADER_FMT, raw)
        version, pid = ver_pid >> 4, ver_pid & 0x0F
        if magic != MAGIC or version != VERSION or pid not in PROFILE_BY_ID:
            raise AideBlindDecodeError("no AIDE blind header found for this secret")
        prof = PROFILES[PROFILE_BY_ID[pid]]
        params = _payload_plan(ct_len, prof)
        n_cw = -(-(ct_len + crypto.TAG_LEN + ecc.LENGTH_HEADER) // params.data_len)
        n_payload_bits = n_cw * params.codeword_len * 8
        psoft = _soft_stream(stego, shared_secret, n_payload_bits, start_bit=n_hdr * HEADER_REPEATS)
        try:
            protected = _decode_bytes(psoft, params, prof.dct_delta, stats, interleaved=True)
        except ecc.AideECCError as exc:
            return BlindRecoverResult(False, None, str(exc), prof.name,
                                      stats.corrected_bytes, stats.gmd_codewords)
        ct, tag = protected[:ct_len], protected[ct_len:ct_len + crypto.TAG_LEN]
        try:
            pt = crypto.decrypt(crypto.EncryptedPayload(ct, _nonce_from_salt(salt), tag), salt,
                                shared_secret, kdf=crypto.KDF_SCRYPT)
        except crypto.AideAuthenticationError as exc:
            return BlindRecoverResult(False, None, str(exc), prof.name,
                                      stats.corrected_bytes, stats.gmd_codewords)
        return BlindRecoverResult(True, pt, None, prof.name, stats.corrected_bytes, stats.gmd_codewords)
    except AideBlindDecodeError as exc:
        return BlindRecoverResult(False, None, str(exc))


def max_message_bytes(height: int, width: int, profile: str | RobustnessProfile | None = None) -> int:
    """Exact blind-mode capacity: total keyed-layout bits minus the fixed
    header, inverted through the profile's ECC plan."""
    from .pipeline import max_message_bytes as _max
    prof = get_profile(profile)
    available = capacity_bits(height, width) - HEADER_ECC.codeword_len * 8 * HEADER_REPEATS
    return _max(max(0, available), prof)
