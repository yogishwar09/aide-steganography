"""AES-256-GCM confidentiality layer with image-bound key derivation.

Report section 2.1 step 5: key = SHA-256(image_hash + shared_secret), so the
same message embedded in two different cover images yields unrelated
ciphertexts, with a GCM tag for integrity verification at recovery time.

Design note on ``image_hash``: the receiver only ever sees the *stego*
image, whose pixels differ from the pristine cover by design (that's the
embedding). So the pristine cover's hash can't be recomputed from the stego
image alone. We compute it once at embed time and carry it in the sidecar
metadata (see ``aide.pipeline``) — functionally it acts as a per-embedding
salt. This preserves the report's actual security goal (ciphertext is bound
to a specific embedding event, not reusable across images) without changing
where the real secrecy lives: that's still entirely in ``shared_secret``,
which is never stored or transmitted alongside the image.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

from Crypto.Cipher import AES

KEY_LEN = 32  # AES-256
NONCE_LEN = 12  # standard GCM nonce size
TAG_LEN = 16


class AideAuthenticationError(Exception):
    """Raised when GCM tag verification fails: message is tampered beyond
    recovery (report section 2.3, recovery sub-pipeline step 3)."""


@dataclass(frozen=True)
class EncryptedPayload:
    ciphertext: bytes
    nonce: bytes
    tag: bytes


def hash_image_bytes(image_bytes: bytes) -> bytes:
    """SHA-256 digest of the pristine cover image's raw file bytes."""
    return hashlib.sha256(image_bytes).digest()


def derive_key(image_hash: bytes, shared_secret: bytes) -> bytes:
    """key = SHA-256(image_hash + shared_secret) -> 32 bytes for AES-256."""
    return hashlib.sha256(image_hash + shared_secret).digest()


def encrypt(plaintext: bytes, image_hash: bytes, shared_secret: bytes) -> EncryptedPayload:
    key = derive_key(image_hash, shared_secret)
    nonce = os.urandom(NONCE_LEN)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return EncryptedPayload(ciphertext=ciphertext, nonce=nonce, tag=tag)


def decrypt(payload: EncryptedPayload, image_hash: bytes, shared_secret: bytes) -> bytes:
    key = derive_key(image_hash, shared_secret)
    cipher = AES.new(key, AES.MODE_GCM, nonce=payload.nonce)
    try:
        return cipher.decrypt_and_verify(payload.ciphertext, payload.tag)
    except ValueError as exc:
        raise AideAuthenticationError(
            "GCM tag verification failed - message tampered beyond recovery"
        ) from exc
