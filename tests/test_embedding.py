from __future__ import annotations

import numpy as np
import pytest

from aide.embedding import METHODS

BLOCK_SIZES = (8, 16)
BPP_VALUES = (1, 2, 3, 4)


@pytest.mark.parametrize("method_name", sorted(METHODS))
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("bpp", BPP_VALUES)
def test_embed_extract_roundtrip(method_name, block_size, bpp, synthetic_image):
    mod = METHODS[method_name]
    block = synthetic_image[:block_size, :block_size]
    cap = mod.capacity_bits(block.shape, bpp)
    if cap == 0:
        pytest.skip(f"{method_name} bpp={bpp} block={block_size} has zero capacity")

    rng = np.random.default_rng(123)
    bits = rng.integers(0, 2, size=cap).astype(np.uint8)

    stego = mod.embed(block, bpp, bits)
    assert stego.shape == block.shape
    assert stego.dtype == np.uint8

    extracted = mod.extract(stego, bpp, num_bits=cap)
    assert np.array_equal(bits, extracted)


@pytest.mark.parametrize("method_name", sorted(METHODS))
def test_partial_bit_usage_extracts_correct_prefix(method_name, synthetic_image):
    """pipeline.py always pads to full block capacity and only asks for a
    prefix back - this is the behavior that makes bit-level (not
    byte-rounded) capacity accounting possible."""
    mod = METHODS[method_name]
    block = synthetic_image[:16, :16]
    cap = mod.capacity_bits(block.shape, bpp=2)
    if cap < 3:
        pytest.skip("not enough capacity for a meaningful partial-prefix test")

    rng = np.random.default_rng(7)
    real_bits = rng.integers(0, 2, size=cap - 2).astype(np.uint8)
    full_bits = np.zeros(cap, dtype=np.uint8)
    full_bits[:len(real_bits)] = real_bits

    stego = mod.embed(block, 2, full_bits)
    extracted_prefix = mod.extract(stego, 2, num_bits=len(real_bits))
    assert np.array_equal(real_bits, extracted_prefix)


def test_capacity_bits_nondecreasing_in_bpp(synthetic_image):
    block = synthetic_image[:16, :16]
    for name, mod in METHODS.items():
        caps = [mod.capacity_bits(block.shape, bpp) for bpp in BPP_VALUES]
        assert caps == sorted(caps), f"{name} capacity should not decrease as bpp grows"
