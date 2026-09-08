from __future__ import annotations

import numpy as np

from aide.features import (
    choose_block_size, extract_block_features, extract_image_features, partition_blocks,
)


def test_choose_block_size_switches_at_resolution_threshold():
    assert choose_block_size(256, 256) == 8
    assert choose_block_size(512, 512) == 16
    assert choose_block_size(300, 900) == 16  # max(dims) decides


def test_partition_blocks_covers_grid_without_overlap():
    blocks = partition_blocks(64, 64, 16)
    assert len(blocks) == 16  # 4x4 grid
    seen = np.zeros((64, 64), dtype=bool)
    for b in blocks:
        ys, xs = b.slice()
        assert not seen[ys, xs].any(), "blocks must not overlap"
        seen[ys, xs] = True
    assert seen.all(), "blocks must fully tile an evenly-divisible image"


def test_partition_blocks_drops_trailing_remainder():
    blocks = partition_blocks(20, 20, 8)  # 20 // 8 = 2 -> 2x2 grid, last 4px dropped
    assert len(blocks) == 4
    for b in blocks:
        assert b.y1 <= 20 and b.x1 <= 20


def test_block_features_are_normalised_and_discriminate_smooth_vs_textured(synthetic_image):
    smooth_block = synthetic_image[0:32, 0:32]
    textured_block = synthetic_image[0:32, 32:64]

    f_smooth = extract_block_features(smooth_block)
    f_textured = extract_block_features(textured_block)

    assert f_smooth.shape == (4,)
    assert np.all((f_smooth >= 0) & (f_smooth <= 1))
    assert np.all((f_textured >= 0) & (f_textured <= 1))

    # variance and entropy should clearly separate a flat block from noise
    variance_idx, entropy_idx = 2, 1
    assert f_textured[variance_idx] > f_smooth[variance_idx]
    assert f_textured[entropy_idx] > f_smooth[entropy_idx]


def test_extract_image_features_matches_block_count(synthetic_image):
    blocks, features = extract_image_features(synthetic_image, block_size=16)
    assert len(blocks) == features.shape[0]
    assert features.shape[1] == 4
    assert np.all(np.isfinite(features))
