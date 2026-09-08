from __future__ import annotations

import pytest

from aide import crypto


def test_encrypt_decrypt_roundtrip():
    image_hash = crypto.hash_image_bytes(b"some cover image bytes")
    secret = b"shared-secret"
    plaintext = b"Rendezvous at grid 47N 12E at 0300 hours."

    payload = crypto.encrypt(plaintext, image_hash, secret)
    recovered = crypto.decrypt(payload, image_hash, secret)

    assert recovered == plaintext
    assert len(payload.nonce) == crypto.NONCE_LEN
    assert len(payload.tag) == crypto.TAG_LEN


def test_different_images_yield_different_ciphertext_for_same_message():
    secret = b"shared-secret"
    plaintext = b"identical message"
    hash_a = crypto.hash_image_bytes(b"image A bytes")
    hash_b = crypto.hash_image_bytes(b"image B bytes")

    payload_a = crypto.encrypt(plaintext, hash_a, secret)
    payload_b = crypto.encrypt(plaintext, hash_b, secret)

    assert payload_a.ciphertext != payload_b.ciphertext


def test_tampered_ciphertext_fails_authentication():
    image_hash = crypto.hash_image_bytes(b"cover")
    secret = b"secret"
    payload = crypto.encrypt(b"top secret payload", image_hash, secret)

    corrupted_bytes = bytearray(payload.ciphertext)
    corrupted_bytes[0] ^= 0xFF
    tampered = crypto.EncryptedPayload(bytes(corrupted_bytes), payload.nonce, payload.tag)

    with pytest.raises(crypto.AideAuthenticationError):
        crypto.decrypt(tampered, image_hash, secret)


def test_wrong_shared_secret_fails_authentication():
    image_hash = crypto.hash_image_bytes(b"cover")
    payload = crypto.encrypt(b"message", image_hash, b"correct-secret")

    with pytest.raises(crypto.AideAuthenticationError):
        crypto.decrypt(payload, image_hash, b"wrong-secret")
