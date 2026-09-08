from __future__ import annotations

import numpy as np
import pytest

from aide import ecc


def test_encode_decode_roundtrip_no_damage():
    data = b"The quick brown fox jumps over the lazy dog." * 10
    encoded = ecc.encode(data)
    assert len(encoded) % ecc.CODEWORD_LEN == 0
    assert ecc.decode(encoded) == data


def test_encode_output_is_always_a_multiple_of_codeword_len():
    for n in (0, 1, 100, 223, 224, 1000):
        encoded = ecc.encode(b"x" * n)
        assert len(encoded) % ecc.CODEWORD_LEN == 0


def test_interleave_deinterleave_is_identity():
    encoded = ecc.encode(b"payload" * 50)
    interleaved = ecc.interleave(encoded)
    assert len(interleaved) == len(encoded)
    assert ecc.deinterleave(interleaved) == encoded


def test_blind_correction_within_t_errors_per_codeword():
    data = b"secret message padded out a bit for a full codeword worth of data here"
    encoded = bytearray(ecc.encode(data))
    # flip up to t=16 bytes within the first codeword - should self-correct
    for i in range(16):
        encoded[i] ^= 0xFF
    assert ecc.decode(bytes(encoded)) == data


def test_blind_correction_fails_beyond_t_errors():
    data = b"secret message padded out a bit for a full codeword worth of data here"
    encoded = bytearray(ecc.encode(data))
    for i in range(30):  # well past t=16
        encoded[i] ^= 0xFF
    with pytest.raises(ecc.AideECCError):
        ecc.decode(bytes(encoded))


def test_erasure_correction_handles_more_damage_than_blind_correction():
    data = b"secret message padded out a bit for a full codeword worth of data here"
    encoded = bytearray(ecc.encode(data))
    damaged_positions = list(range(28))  # up to 2t=32 erasures is correctable if known
    for i in damaged_positions:
        encoded[i] ^= 0xFF
    assert ecc.decode(bytes(encoded), erase_positions=damaged_positions) == data


def test_interleaving_spreads_a_contiguous_burst_across_codewords():
    data = b"x" * 1000
    encoded = ecc.encode(data)
    interleaved = bytearray(ecc.interleave(encoded))
    n_codewords = len(encoded) // ecc.CODEWORD_LEN
    assert n_codewords >= 2

    # a contiguous burst in *interleaved* (spatial embedding) order...
    burst = slice(10, 10 + n_codewords * 2)
    mask = np.zeros(len(interleaved), dtype=bool)
    mask[burst] = True
    for i in range(burst.start, burst.stop):
        interleaved[i] ^= 0xFF

    deinterleaved_mask = ecc.deinterleave_mask(mask)
    # ...should land in *multiple different* codewords once deinterleaved,
    # rather than being concentrated in just one
    hit_codewords = {pos // ecc.CODEWORD_LEN for pos in np.where(deinterleaved_mask)[0]}
    assert len(hit_codewords) > 1

    recovered = ecc.decode(ecc.deinterleave(bytes(interleaved)),
                            erase_positions=list(np.where(deinterleaved_mask)[0]))
    assert recovered == data
