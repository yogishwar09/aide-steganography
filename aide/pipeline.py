"""High-level embed()/recover() orchestration (report sections 2.1-2.3).

Ties together feature extraction, the AIDE decision engine, AES-256-GCM
encryption, Reed-Solomon ECC + interleaving, adaptive LSB/DCT/DWT
embedding, and tamper-localized recovery.

Scope note: this prototype operates on grayscale images. RGB input is
converted to grayscale before embedding (see README) - repeating the same
pipeline per color channel is straightforward future work, out of scope
here.

Side-channel metadata: recovery needs the per-block embedding map,
reference hashes/histograms, RS parameters, and the GCM nonce. Per the
report these are "transmitted alongside" the stego image; this is
implemented literally as a JSON sidecar file (`Sidecar`/`save_sidecar`/
`load_sidecar`), assumed to travel over a channel separate from (and more
reliable than) the possibly-attacked image itself.

Bit-level (not byte-level) accounting: DCT/DWT blocks often carry only a
handful of *bits* of capacity (a 16x16 block can be as low as 4-8 bits).
Rounding each block down to whole bytes before moving to the next block
was measured to waste the large majority of DCT/DWT capacity (e.g. a
4-bit-capacity block contributes 0 usable bytes), pushing the payload back
onto high-capacity LSB blocks - exactly the fragile-under-compression
outcome the method-selection policy is trying to avoid. Instead, the
payload is tracked as a flat bitstream and each block consumes exactly its
own bit capacity (possibly ending mid-byte); only the very last few bits
overall need not fill a whole block, since the total payload length is
always a whole number of bytes by construction (RS codewords).
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import crypto, ecc, tamper
from .embedding import METHODS
from .features import extract_image_features, partition_blocks
from .model import AIDEModel


class AideCapacityError(Exception):
    """Raised when the cover image doesn't have enough safe capacity for the message."""


def load_grayscale(path: Path) -> tuple[np.ndarray, bytes]:
    raw = Path(path).read_bytes()
    gray = np.array(Image.open(path).convert("L"))
    return gray, raw


def save_grayscale(gray: np.ndarray, path: Path) -> None:
    Image.fromarray(gray.astype(np.uint8), mode="L").save(path)


def _bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


@dataclass
class EmbeddingMapEntry:
    block_index: int
    method: str
    bpp: int
    bit_start: int
    bit_end: int


@dataclass
class Sidecar:
    width: int
    height: int
    block_size: int
    nonce: bytes
    image_hash: bytes
    ciphertext_len: int
    tag_len: int
    embedding_map: list[EmbeddingMapEntry]
    references: list[tamper.BlockReference]

    def to_json(self) -> str:
        payload = {
            "version": "aide-0.1",
            "width": self.width,
            "height": self.height,
            "block_size": self.block_size,
            "nonce": base64.b64encode(self.nonce).decode(),
            "image_hash": base64.b64encode(self.image_hash).decode(),
            "ciphertext_len": self.ciphertext_len,
            "tag_len": self.tag_len,
            "embedding_map": [vars(e) for e in self.embedding_map],
            "references": [vars(r) for r in self.references],
        }
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Sidecar":
        payload = json.loads(text)
        return cls(
            width=payload["width"], height=payload["height"], block_size=payload["block_size"],
            nonce=base64.b64decode(payload["nonce"]),
            image_hash=base64.b64decode(payload["image_hash"]),
            ciphertext_len=payload["ciphertext_len"], tag_len=payload["tag_len"],
            embedding_map=[EmbeddingMapEntry(**e) for e in payload["embedding_map"]],
            references=[tamper.BlockReference(**r) for r in payload["references"]],
        )


@dataclass
class EmbedResult:
    stego: np.ndarray
    sidecar: Sidecar


# Fraction of the RS-encoded payload each method may carry, as a share of
# total_bits. DCT is unrestricted (gets whatever's needed) since it
# measured the most broadly robust of the three once its coefficient-pair
# selection was capped to a safe frequency band (aide/embedding/dct.py).
# LSB and DWT are both capped, at very different sizes, for different
# reasons - see each entry below.
DEFAULT_FRAGILE_METHOD_BUDGETS = {
    # LSB: far higher raw capacity than DCT/DWT, but the least tolerant of
    # *any* pixel change - depends on exact pixel values, not a robust
    # coefficient relation. Measured: ~45-55% bit-error rate under every
    # attack tried, including a plain JPEG re-save. That error rate is so
    # disproportionate that even 1% of payload bits from LSB was measured
    # to flip an otherwise-passing median-filter recovery to failing -
    # kept at a token 0.5% purely so the "adaptive, multi-method"
    # embedding map this project is actually about stays visible in
    # practice. Raise it only for cover images/attack scenarios where
    # visual capacity matters more than robustness.
    "LSB": 0.005,
    # DWT: the 2-level approximation-sub-band design (aide/embedding/dwt.py)
    # fixed compression and geometric-attack survival in isolation, but
    # still measures a real (just much smaller than LSB's) error rate
    # under median filtering/scaling - e.g. dropping DWT's share from
    # ~15% to ~2% was the difference between failing and passing a
    # representative median-filter recovery test. Capped small rather
    # than zero for the same "keep the adaptive method mix visible"
    # reason as LSB above.
    "DWT": 0.02,
}


def embed(cover_path: Path, message: bytes, shared_secret: bytes, model: AIDEModel,
          fragile_method_budgets: dict[str, float] | None = None) -> EmbedResult:
    """`fragile_method_budgets` caps how much of the RS-encoded payload may
    be carried by each of a set of "fragile" methods (default: LSB 1% -
    see DEFAULT_FRAGILE_METHOD_BUDGETS above for the measurement behind
    that number). Any method not listed is unrestricted.

    Why cap at all: each method's real *bit* capacity per block is wildly
    uneven (LSB uses whole pixels; DCT/DWT use a handful of coefficient
    pairs), so even when the AIDE engine assigns a fragile method to a
    minority of *blocks* (correctly, per the report's texture/edge rule),
    those blocks can still end up carrying the *majority* of payload
    *bits* - measured empirically at over 50% for LSB alone before this
    cap existed. Combined with how poorly that method survives common
    attacks, interleaving alone can't spread the resulting damage within
    Reed-Solomon's ~6.3% blind-correction budget per codeword. This is a
    real, measured trade-off the report's formula alone doesn't resolve.
    """
    budgets = dict(DEFAULT_FRAGILE_METHOD_BUDGETS if fragile_method_budgets is None
                    else fragile_method_budgets)

    gray, raw = load_grayscale(cover_path)
    image_hash = crypto.hash_image_bytes(raw)

    blocks, features = extract_image_features(gray)
    decisions = model.decide(features)
    kept = sorted((d for d in decisions if d.keep), key=lambda d: d.rank)

    encrypted = crypto.encrypt(message, image_hash, shared_secret)
    protected = encrypted.ciphertext + encrypted.tag
    encoded = ecc.encode(protected)
    interleaved = ecc.interleave(encoded)
    all_bits = _bytes_to_bits(interleaved)
    total_bits = len(all_bits)
    bit_budgets = {m: int(total_bits * f) for m, f in budgets.items()}
    used_bits_by_method: dict[str, int] = {m: 0 for m in budgets}

    stego = gray.copy()
    embedding_map: list[EmbeddingMapEntry] = []
    cursor = 0

    def place(d, max_bits: int | None = None) -> int:
        nonlocal cursor
        block = blocks[d.block_index]
        mod = METHODS[d.method]
        cap_bits = mod.capacity_bits((block.size, block.size), d.bpp)
        use_bits = min(cap_bits, total_bits - cursor, max_bits if max_bits is not None else cap_bits)
        chunk_bits = all_bits[cursor:cursor + use_bits]
        full_bits = np.zeros(cap_bits, dtype=np.uint8)
        full_bits[:use_bits] = chunk_bits

        ys, xs = block.slice()
        stego[ys, xs] = mod.embed(gray[ys, xs], d.bpp, full_bits)
        embedding_map.append(EmbeddingMapEntry(
            block_index=block.index, method=d.method, bpp=d.bpp,
            bit_start=cursor, bit_end=cursor + use_bits,
        ))
        cursor += use_bits
        return use_bits

    for d in kept:
        if cursor >= total_bits:
            break
        block = blocks[d.block_index]
        mod = METHODS[d.method]
        if mod.capacity_bits((block.size, block.size), d.bpp) <= 0:
            continue
        if d.method in bit_budgets:
            # A single block's capacity can dwarf the remaining budget in
            # one step for high-capacity methods, so the budget must also
            # cap *how much* of this block gets used, not just whether
            # it's considered at all - otherwise one block blows the cap.
            remaining_budget = bit_budgets[d.method] - used_bits_by_method[d.method]
            if remaining_budget <= 0:
                continue
            used_bits_by_method[d.method] += place(d, max_bits=remaining_budget)
        else:
            place(d)

    if cursor < total_bits:
        # Deliberately not falling back to unlimited fragile-method usage
        # here: doing so previously let LSB usage spike past 40% of the
        # payload on messages/images where robust-method capacity ran out,
        # silently destroying compression resilience (measured: JPEG q=50
        # recovery failing) without any signal to the caller. Surfacing a
        # clear error lets the caller choose a shorter message, a higher-
        # capacity/more textured cover image, or explicitly relax
        # `fragile_method_budgets` and accept reduced robustness.
        raise AideCapacityError(
            f"cover image capacity insufficient at fragile_method_budgets={budgets}: "
            f"needed {total_bits} bits after ECC encoding, only {cursor} bits available "
            f"(used by method: {used_bits_by_method}) across {len(kept)} AIDE-selected "
            f"blocks. Try a shorter message, a more textured cover image, or relaxed budgets."
        )

    used_blocks = [blocks[e.block_index] for e in embedding_map]
    references = tamper.build_references(stego, used_blocks)

    sidecar = Sidecar(
        width=gray.shape[1], height=gray.shape[0], block_size=blocks[0].size if blocks else 0,
        nonce=encrypted.nonce, image_hash=image_hash,
        ciphertext_len=len(encrypted.ciphertext), tag_len=len(encrypted.tag),
        embedding_map=embedding_map, references=references,
    )
    return EmbedResult(stego=stego, sidecar=sidecar)


@dataclass
class RecoverResult:
    success: bool
    plaintext: bytes | None
    error: str | None
    tamper_report: tamper.TamperReport | None
    erasure_count: int
    total_bytes: int


def recover(stego: np.ndarray, sidecar: Sidecar, shared_secret: bytes) -> RecoverResult:
    if stego.shape != (sidecar.height, sidecar.width):
        return RecoverResult(False, None, "image dimensions do not match sidecar", None, 0, 0)

    blocks = partition_blocks(sidecar.height, sidecar.width, sidecar.block_size)
    used_blocks = [blocks[e.block_index] for e in sidecar.embedding_map]

    report = tamper.analyze(stego, blocks, sidecar.references)

    total_bits = sidecar.embedding_map[-1].bit_end if sidecar.embedding_map else 0
    all_bits = np.zeros(total_bits, dtype=np.uint8)
    unreliable_bits = np.zeros(total_bits, dtype=bool)

    for entry, block in zip(sidecar.embedding_map, used_blocks):
        mod = METHODS[entry.method]
        ys, xs = block.slice()
        n_bits = entry.bit_end - entry.bit_start
        bits = mod.extract(stego[ys, xs], entry.bpp, num_bits=n_bits)
        all_bits[entry.bit_start:entry.bit_end] = bits
        if report.erasure_mask[block.index]:
            unreliable_bits[entry.bit_start:entry.bit_end] = True

    total_bytes = total_bits // 8
    if total_bytes:
        interleaved = np.packbits(all_bits).tobytes()
        unreliable_bytes = unreliable_bits.reshape(-1, 8).any(axis=1)
        encoded = ecc.deinterleave(interleaved)
        unreliable_deint = ecc.deinterleave_mask(unreliable_bytes)
        erase_positions = list(np.where(unreliable_deint)[0])
    else:
        encoded = b""
        erase_positions = []

    try:
        protected = ecc.decode(encoded, erase_positions=erase_positions or None)
    except ecc.AideECCError as exc:
        return RecoverResult(False, None, str(exc), report, len(erase_positions), total_bytes)

    ciphertext = protected[:sidecar.ciphertext_len]
    tag = protected[sidecar.ciphertext_len:sidecar.ciphertext_len + sidecar.tag_len]
    payload = crypto.EncryptedPayload(ciphertext=ciphertext, nonce=sidecar.nonce, tag=tag)
    try:
        plaintext = crypto.decrypt(payload, sidecar.image_hash, shared_secret)
    except crypto.AideAuthenticationError as exc:
        return RecoverResult(False, None, str(exc), report, len(erase_positions), total_bytes)

    return RecoverResult(True, plaintext, None, report, len(erase_positions), total_bytes)


def save_sidecar(sidecar: Sidecar, path: Path) -> None:
    Path(path).write_text(sidecar.to_json())


def load_sidecar(path: Path) -> Sidecar:
    return Sidecar.from_json(Path(path).read_text())
