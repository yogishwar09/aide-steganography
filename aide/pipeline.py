"""High-level embed()/recover() orchestration (report sections 2.1-2.3).

Ties together feature extraction, the AIDE decision engine, AES-256-GCM
encryption, Reed-Solomon ECC + interleaving, adaptive LSB/DCT/DWT
embedding, and tamper-localized recovery.

Color images: the whole pipeline (features, AIDE decisions, embedding,
tamper localization) operates on a single 2D plane. For an RGB cover,
that plane is the Y (luma) channel of a YCbCr conversion - the Cb/Cr
chroma planes are carried through untouched and recombined on save. This
is standard practice (the human eye is far less sensitive to chroma
changes at high spatial frequency than to luma changes, and it keeps the
whole embedding pipeline single-channel rather than 3x more complex).
Attack simulation (`aide.attacks`) and this module's own `recover()`
still operate purely on the luma plane too - a genuinely color-aware
attack model (e.g. chroma subsampling effects from real JPEG encoders)
is not attempted here, since it's a separate can of worms from what this
project's novelty is about.

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
from .profiles import LEGACY_PROFILE, RobustnessProfile, get_profile

SIDECAR_VERSION = "aide-0.2"


class AideCapacityError(Exception):
    """Raised when the cover image doesn't have enough safe capacity for the message."""


def load_grayscale(path: Path) -> tuple[np.ndarray, bytes]:
    """Legacy helper: always converts to grayscale, discarding color. Use
    `load_image` for color-preserving loading (RGB cover -> Y-plane +
    chroma for later recombination)."""
    raw = Path(path).read_bytes()
    gray = np.array(Image.open(path).convert("L"))
    return gray, raw


def save_grayscale(gray: np.ndarray, path: Path) -> None:
    Image.fromarray(gray.astype(np.uint8), mode="L").save(path)


@dataclass
class LoadedImage:
    y: np.ndarray                                    # luma/grayscale plane, always 2D uint8
    mode: str                                         # "L" (grayscale) or "RGB"
    chroma: tuple[np.ndarray, np.ndarray] | None      # (Cb, Cr) if mode == "RGB", else None
    raw_bytes: bytes                                  # original file bytes, for image_hash


def load_image(path: Path) -> LoadedImage:
    """Load any standard-format image (report 2.1 step 1: "RGB or
    grayscale"). RGB is converted to YCbCr and split; everything else in
    this pipeline operates on the returned `y` plane only."""
    raw = Path(path).read_bytes()
    img = Image.open(path)
    if img.mode in ("L", "1", "I", "F"):
        return LoadedImage(y=np.array(img.convert("L")), mode="L", chroma=None, raw_bytes=raw)
    y, cb, cr = (np.array(c) for c in img.convert("RGB").convert("YCbCr").split())
    return LoadedImage(y=y, mode="RGB", chroma=(cb, cr), raw_bytes=raw)


def compose_image(y: np.ndarray, mode: str, chroma: tuple[np.ndarray, np.ndarray] | None) -> Image.Image:
    y_img = Image.fromarray(np.clip(y, 0, 255).astype(np.uint8), mode="L")
    if mode == "L":
        return y_img
    cb, cr = chroma
    ycbcr = Image.merge("YCbCr", [y_img, Image.fromarray(cb.astype(np.uint8)),
                                   Image.fromarray(cr.astype(np.uint8))])
    return ycbcr.convert("RGB")


def save_image(y: np.ndarray, mode: str, chroma: tuple[np.ndarray, np.ndarray] | None, path: Path) -> None:
    compose_image(y, mode, chroma).save(path)


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
    profile: RobustnessProfile = LEGACY_PROFILE
    ecc_params: ecc.ECCParams = ecc.DEFAULT_PARAMS
    kdf: str = crypto.KDF_SHA256

    def to_json(self) -> str:
        payload = {
            "version": SIDECAR_VERSION,
            "profile": self.profile.to_dict(),
            "ecc": self.ecc_params.to_dict(),
            "kdf": self.kdf,
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
    def from_json(cls, text: str) -> Sidecar:
        payload = json.loads(text)
        return cls(
            width=payload["width"], height=payload["height"], block_size=payload["block_size"],
            nonce=base64.b64decode(payload["nonce"]),
            image_hash=base64.b64decode(payload["image_hash"]),
            ciphertext_len=payload["ciphertext_len"], tag_len=payload["tag_len"],
            embedding_map=[EmbeddingMapEntry(**e) for e in payload["embedding_map"]],
            references=[tamper.BlockReference(**r) for r in payload["references"]],
            # aide-0.1 sidecars predate profiles: decode them with the
            # exact fixed parameters they were produced with.
            profile=(RobustnessProfile.from_dict(payload["profile"]) if "profile" in payload
                     else LEGACY_PROFILE),
            ecc_params=ecc.ECCParams.from_dict(payload.get("ecc")),
            kdf=payload.get("kdf", crypto.KDF_SHA256),
        )


@dataclass
class EmbedResult:
    stego: np.ndarray                                 # the embedded luma/grayscale plane
    sidecar: Sidecar
    mode: str = "L"                                   # "L" or "RGB" - the cover's original mode
    chroma: tuple[np.ndarray, np.ndarray] | None = None  # (Cb, Cr), untouched, for RGB covers

    def compose(self) -> Image.Image:
        """The full, ready-to-save stego image (RGB recombined with the
        original chroma if the cover was color; identical to `stego`
        as a grayscale image otherwise)."""
        return compose_image(self.stego, self.mode, self.chroma)


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
          fragile_method_budgets: dict[str, float] | None = None,
          profile: str | RobustnessProfile | None = None) -> EmbedResult:
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

    `profile` (name or `RobustnessProfile`, default "balanced") selects the
    coefficient deltas, Reed-Solomon strength, and default budgets - see
    aide/profiles.py. An explicit `fragile_method_budgets` overrides the
    profile's budgets.
    """
    prof = get_profile(profile)
    budgets = dict(prof.fragile_budgets if fragile_method_budgets is None
                    else fragile_method_budgets)

    loaded = load_image(cover_path)
    gray = loaded.y
    image_hash = crypto.hash_image_bytes(loaded.raw_bytes)

    blocks, features = extract_image_features(gray)
    decisions = model.decide(features)
    kept = sorted((d for d in decisions if d.keep), key=lambda d: d.rank)

    encrypted = crypto.encrypt(message, image_hash, shared_secret, kdf=crypto.DEFAULT_KDF)
    protected = encrypted.ciphertext + encrypted.tag
    ecc_params = ecc.plan(len(protected), nsym=prof.nsym, min_codewords=prof.min_codewords)
    encoded = ecc.encode(protected, ecc_params)
    interleaved = ecc.interleave(encoded, ecc_params.codeword_len)
    all_bits = _bytes_to_bits(interleaved)
    total_bits = len(all_bits)
    bit_budgets = {m: int(total_bits * f) for m, f in budgets.items()}

    def allocate(bpp_cap: int) -> tuple[list[tuple], int, dict[str, int]]:
        """Plan (decision, bpp, bit_start, n_bits) per block without embedding."""
        plan, cursor = [], 0
        used = {m: 0 for m in budgets}
        for d in kept:
            if cursor >= total_bits:
                break
            block = blocks[d.block_index]
            bpp = min(d.bpp, bpp_cap)
            cap_bits = METHODS[d.method].capacity_bits((block.size, block.size), bpp)
            if cap_bits <= 0:
                continue
            limit = cap_bits
            if d.method in bit_budgets:
                # A single block's capacity can dwarf the remaining budget in
                # one step for high-capacity methods, so the budget must also
                # cap *how much* of this block gets used, not just whether
                # it's considered at all - otherwise one block blows the cap.
                limit = min(limit, bit_budgets[d.method] - used[d.method])
                if limit <= 0:
                    continue
            n = min(limit, total_bits - cursor)
            plan.append((d, bpp, cursor, n))
            if d.method in used:
                used[d.method] += n
            cursor += n
        return plan, cursor, used

    # Prefer the profile's bpp cap (fewer, lower-frequency DCT pairs per
    # block measured markedly more robust to median filtering and 0.5x
    # rescaling at equal PSNR - see aide/profiles.py); relax it one step at
    # a time only when the message would not otherwise fit.
    for bpp_cap in range(prof.max_bpp, 5):
        plan, cursor, used_bits_by_method = allocate(bpp_cap)
        if cursor >= total_bits:
            break

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
            f"cover image capacity insufficient for profile {prof.name!r} "
            f"(fragile_method_budgets={budgets}): needed {total_bits} bits after ECC encoding, "
            f"only {cursor} bits available (used by method: {used_bits_by_method}) across "
            f"{len(kept)} AIDE-selected blocks. Try a shorter message, a more textured cover "
            f"image, a lighter profile, or relaxed budgets."
        )

    stego = gray.copy()
    embedding_map: list[EmbeddingMapEntry] = []
    for d, bpp, start, n in plan:
        block = blocks[d.block_index]
        mod = METHODS[d.method]
        cap_bits = mod.capacity_bits((block.size, block.size), bpp)
        full_bits = np.zeros(cap_bits, dtype=np.uint8)
        full_bits[:n] = all_bits[start:start + n]
        ys, xs = block.slice()
        stego[ys, xs] = mod.embed(gray[ys, xs], bpp, full_bits, delta=prof.delta_for(d.method))
        embedding_map.append(EmbeddingMapEntry(
            block_index=block.index, method=d.method, bpp=bpp, bit_start=start, bit_end=start + n,
        ))

    used_blocks = [blocks[e.block_index] for e in embedding_map]
    references = tamper.build_references(stego, used_blocks)

    sidecar = Sidecar(
        width=gray.shape[1], height=gray.shape[0], block_size=blocks[0].size if blocks else 0,
        nonce=encrypted.nonce, image_hash=image_hash,
        ciphertext_len=len(encrypted.ciphertext), tag_len=len(encrypted.tag),
        embedding_map=embedding_map, references=references,
        profile=prof, ecc_params=ecc_params, kdf=crypto.DEFAULT_KDF,
    )
    return EmbedResult(stego=stego, sidecar=sidecar, mode=loaded.mode, chroma=loaded.chroma)


@dataclass
class RecoverResult:
    success: bool
    plaintext: bytes | None
    error: str | None
    tamper_report: tamper.TamperReport | None
    erasure_count: int
    total_bytes: int
    corrected_bytes: int = 0       # byte positions Reed-Solomon actually repaired
    gmd_codewords: int = 0         # codewords rescued by reliability-guided erasures


# Normalised reliability assigned to LSB bits (which carry no analog
# margin): far below any DCT/DWT bit, so reliability-guided decoding
# erases LSB-carried bytes first - matching their measured ~50% error rate
# under every attack.
_LSB_RELIABILITY = 0.05
# Per-method prior on top of the coefficient margin. Measured on BOSSbase
# under 3x3 median filtering: DCT bits flip ~0.5% of the time, DWT bits
# ~13%, and DWT flips frequently happen at *large* margins (median
# filtering resamples the 2x2 neighbourhoods Haar averages over), so a
# DWT margin alone overstates its trustworthiness.
_METHOD_RELIABILITY_WEIGHT = {"DCT": 1.0, "DWT": 0.3}


def _extract_stream(stego: np.ndarray, sidecar: Sidecar, blocks) -> tuple[np.ndarray, np.ndarray]:
    """Hard bits plus a normalised per-bit reliability (|margin| / delta,
    so 1.0 = exactly as embedded, ~0 = on the decision boundary)."""
    total_bits = sidecar.embedding_map[-1].bit_end if sidecar.embedding_map else 0
    bits = np.zeros(total_bits, dtype=np.uint8)
    rel = np.zeros(total_bits, dtype=np.float64)
    for entry in sidecar.embedding_map:
        block = blocks[entry.block_index]
        mod = METHODS[entry.method]
        ys, xs = block.slice()
        n_bits = entry.bit_end - entry.bit_start
        soft = mod.extract_soft(stego[ys, xs], entry.bpp, num_bits=n_bits)
        bits[entry.bit_start:entry.bit_end] = (soft > 0).astype(np.uint8)
        delta = sidecar.profile.delta_for(entry.method)
        rel[entry.bit_start:entry.bit_end] = (
            np.abs(soft) / delta * _METHOD_RELIABILITY_WEIGHT.get(entry.method, 1.0)
            if delta else _LSB_RELIABILITY)
    return bits, rel


def recover(stego: np.ndarray, sidecar: Sidecar, shared_secret: bytes) -> RecoverResult:
    if stego.shape != (sidecar.height, sidecar.width):
        return RecoverResult(False, None, "image dimensions do not match sidecar", None, 0, 0)

    blocks = partition_blocks(sidecar.height, sidecar.width, sidecar.block_size)
    report = tamper.analyze(stego, blocks, sidecar.references)

    all_bits, bit_rel = _extract_stream(stego, sidecar, blocks)
    total_bits = len(all_bits)
    unreliable_bits = np.zeros(total_bits, dtype=bool)
    for entry in sidecar.embedding_map:
        if report.erasure_mask[entry.block_index]:
            unreliable_bits[entry.bit_start:entry.bit_end] = True

    params = sidecar.ecc_params
    n = params.codeword_len
    total_bytes = total_bits // 8
    stats = ecc.DecodeStats()
    if total_bytes:
        interleaved = np.packbits(all_bits).tobytes()
        unreliable_bytes = unreliable_bits.reshape(-1, 8).any(axis=1)
        byte_rel = bit_rel.reshape(-1, 8).min(axis=1)  # a byte is only as good as its weakest bit
        encoded = ecc.deinterleave(interleaved, n)
        unreliable_deint = ecc.deinterleave_mask(unreliable_bytes, n)
        rel_deint = ecc.deinterleave_mask(byte_rel, n)
        erase_positions = list(np.where(unreliable_deint)[0])
    else:
        encoded, rel_deint, erase_positions = b"", None, []

    def _fail(msg: str) -> RecoverResult:
        return RecoverResult(False, None, msg, report, len(erase_positions), total_bytes,
                             stats.corrected_bytes, stats.gmd_codewords)

    try:
        protected = ecc.decode_with_reliability(encoded, rel_deint, erase_positions or None,
                                                params, stats)
    except ecc.AideECCError as exc:
        return _fail(str(exc))

    ciphertext = protected[:sidecar.ciphertext_len]
    tag = protected[sidecar.ciphertext_len:sidecar.ciphertext_len + sidecar.tag_len]
    payload = crypto.EncryptedPayload(ciphertext=ciphertext, nonce=sidecar.nonce, tag=tag)
    try:
        plaintext = crypto.decrypt(payload, sidecar.image_hash, shared_secret, sidecar.kdf)
    except crypto.AideAuthenticationError as exc:
        return _fail(str(exc))

    return RecoverResult(True, plaintext, None, report, len(erase_positions), total_bytes,
                         stats.corrected_bytes, stats.gmd_codewords)


def save_sidecar(sidecar: Sidecar, path: Path) -> None:
    Path(path).write_text(sidecar.to_json())


def load_sidecar(path: Path) -> Sidecar:
    return Sidecar.from_json(Path(path).read_text())


def _encoded_bits(message_len: int, prof: RobustnessProfile) -> int:
    protected = message_len + crypto.TAG_LEN
    params = ecc.plan(protected, nsym=prof.nsym, min_codewords=prof.min_codewords)
    n_cw = -(-(protected + ecc.LENGTH_HEADER) // params.data_len)
    return n_cw * params.codeword_len * 8


def max_message_bytes(available_bits: int, prof: RobustnessProfile) -> int:
    """Largest plaintext length whose encrypted + RS-encoded size fits in
    `available_bits` (binary search over the deterministic ECC plan)."""
    lo, hi = 0, max(0, available_bits // 8)
    if _encoded_bits(0, prof) > available_bits:
        return 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _encoded_bits(mid, prof) <= available_bits:
            lo = mid
        else:
            hi = mid - 1
    return lo


def capacity_report(cover_path: Path, model: AIDEModel) -> dict:
    """Exact sidecar-mode capacity per profile from a single feature pass:
    the bits the AIDE-selected blocks offer (fragile-method budgets
    applied, bpp cap fully relaxed as `embed` would as a last resort),
    inverted through each profile's ECC plan."""
    from .profiles import PROFILES

    gray = load_image(cover_path).y
    blocks, features = extract_image_features(gray)
    kept = [d for d in model.decide(features) if d.keep]
    out = {}
    for name, prof in PROFILES.items():
        robust_bits = 0
        fragile_bits = {m: 0 for m in prof.fragile_budgets}
        for d in kept:
            b = blocks[d.block_index]
            cap = METHODS[d.method].capacity_bits((b.size, b.size), d.bpp)
            if d.method in fragile_bits:
                fragile_bits[d.method] += cap
            else:
                robust_bits += cap
        # Fragile budgets are a share of the *total* payload, so with
        # robust capacity R and share f the payload T must satisfy
        # T * (1 - sum f) <= R (and each f*T <= that method's capacity).
        share = sum(f for m, f in prof.fragile_budgets.items() if fragile_bits.get(m))
        total = int(robust_bits / (1 - share)) if share < 1 else robust_bits
        for m, f in prof.fragile_budgets.items():
            if f > 0:
                total = min(total, int(robust_bits + fragile_bits[m] / f))
        preferred = sum(METHODS[d.method].capacity_bits((blocks[d.block_index].size,) * 2,
                                                        min(d.bpp, prof.max_bpp))
                        for d in kept if d.method not in prof.fragile_budgets
                        or prof.fragile_budgets[d.method] > 0)
        out[name] = {
            "max_message_bytes": max_message_bytes(total, prof),
            "max_bytes_at_preferred_bpp_cap": max_message_bytes(preferred, prof),
        }
    return out
