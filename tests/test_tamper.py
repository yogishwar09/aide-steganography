from __future__ import annotations

import numpy as np

from aide import tamper
from aide.features import extract_image_features


def test_no_change_means_no_mismatch_and_full_confidence(synthetic_image):
    blocks, _ = extract_image_features(synthetic_image, block_size=16)
    refs = tamper.build_references(synthetic_image, blocks)

    report = tamper.analyze(synthetic_image, blocks, refs)

    assert not report.hash_mismatch.any()
    assert not report.erasure_mask.any()
    assert np.all(report.confidence == 1.0)


def test_localized_replacement_is_flagged_as_erasure(synthetic_image):
    blocks, _ = extract_image_features(synthetic_image, block_size=16)
    refs = tamper.build_references(synthetic_image, blocks)

    tampered = synthetic_image.copy()
    rng = np.random.default_rng(0)
    # fully replace one block's pixels with unrelated random noise
    target = blocks[0]
    ys, xs = target.slice()
    tampered[ys, xs] = rng.integers(0, 256, (target.size, target.size), dtype=np.uint8)

    report = tamper.analyze(tampered, blocks, refs)

    assert report.hash_mismatch[target.index]
    assert report.erasure_mask[target.index]
    assert report.confidence[target.index] < 0.5
    # untouched blocks should remain fully trusted
    for b in blocks[1:]:
        assert not report.hash_mismatch[b.index]


def test_mild_uniform_shift_is_not_over_flagged_as_erasure(synthetic_image):
    """A small global shift (like mild compression/noise) should mismatch
    the strict hash check everywhere but should NOT trip the erasure gate
    for most blocks - that budget is reserved for genuinely localized
    tampering (see aide/tamper.py's module docstring)."""
    blocks, _ = extract_image_features(synthetic_image, block_size=16)
    refs = tamper.build_references(synthetic_image, blocks)

    shifted = np.clip(synthetic_image.astype(np.int16) + 2, 0, 255).astype(np.uint8)
    report = tamper.analyze(shifted, blocks, refs)

    assert report.hash_mismatch.all()  # exact hash always breaks on any change
    assert report.erasure_mask.sum() < len(blocks) // 2  # but erasure gate stays conservative
