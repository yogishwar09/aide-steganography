"""Per-block embedding method registry: LSB, DCT, DWT (report 2.1 step 7).

Each method module exposes the same three functions:
    capacity_bits(block_shape, bpp) -> int
    embed(block, bpp, bits) -> block
    extract(block, bpp, num_bits=None) -> bits
"""
from __future__ import annotations

from . import dct, dwt, lsb

METHODS = {
    "LSB": lsb,
    "DCT": dct,
    "DWT": dwt,
}

__all__ = ["METHODS", "lsb", "dct", "dwt"]
