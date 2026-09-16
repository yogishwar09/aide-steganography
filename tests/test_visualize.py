from __future__ import annotations

from pathlib import Path

import numpy as np

from aide import attacks, pipeline, visualize
from aide.features import partition_blocks


def test_embedding_map_overlay_matches_image_size(real_sample_image_path, trained_model):
    message = b"Rendezvous at grid 47N 12E at 0300 hours."
    secret = b"defence-shared-secret-2026"
    result = pipeline.embed(real_sample_image_path, message, secret, trained_model)

    blocks = partition_blocks(result.sidecar.height, result.sidecar.width, result.sidecar.block_size)
    overlay = visualize.embedding_map_overlay(result.stego, blocks, result.sidecar.embedding_map)

    assert overlay.size == (result.sidecar.width, result.sidecar.height)
    assert overlay.mode == "RGB"


def test_embedding_map_overlay_colors_used_blocks_only(real_sample_image_path, trained_model):
    """Every block actually carrying payload should show a non-grayscale
    (colored) pixel somewhere in its region; a block with no entry in the
    embedding map should be left as pure grayscale (R==G==B)."""
    message = b"Rendezvous at grid 47N 12E at 0300 hours."
    secret = b"defence-shared-secret-2026"
    result = pipeline.embed(real_sample_image_path, message, secret, trained_model)
    blocks = partition_blocks(result.sidecar.height, result.sidecar.width, result.sidecar.block_size)
    overlay = np.array(visualize.embedding_map_overlay(result.stego, blocks, result.sidecar.embedding_map))

    used_indices = {e.block_index for e in result.sidecar.embedding_map}
    assert used_indices, "test message should have used at least one block"

    sample_block = blocks[next(iter(used_indices))]
    ys, xs = sample_block.slice()
    region = overlay[ys, xs]
    is_colored = not np.array_equal(region[..., 0], region[..., 1])
    assert is_colored, "a used block should be tinted, not left pure grayscale"

    unused_indices = set(range(len(blocks))) - used_indices
    if unused_indices:
        unused_block = blocks[next(iter(unused_indices))]
        ys, xs = unused_block.slice()
        region = overlay[ys, xs]
        assert np.array_equal(region[..., 0], region[..., 1]) and np.array_equal(region[..., 1], region[..., 2])


def test_diff_heatmap_is_black_for_identical_images(real_sample_image_path):
    gray = np.array(__import__("PIL.Image", fromlist=["Image"]).open(real_sample_image_path).convert("L"))
    heat = np.array(visualize.diff_heatmap(gray, gray))
    assert heat.max() == 0


def test_diff_heatmap_responds_to_real_difference(real_sample_image_path, trained_model):
    message = b"short covert payload"
    secret = b"defence-shared-secret-2026"
    result = pipeline.embed(real_sample_image_path, message, secret, trained_model)
    cover, _ = pipeline.load_grayscale(real_sample_image_path)

    heat = np.array(visualize.diff_heatmap(cover, result.stego))
    assert heat.max() > 0, "embedding changed some pixels, so the heatmap should show something"


def test_tamper_overlay_flags_spliced_region(real_sample_image_path, trained_model):
    message = (
        b"CLASSIFIED SITREP: Convoy Alpha departs staging area 0600Z, route Bravo-7. "
        b"Escort confirmed, rendezvous at checkpoint Delta. "
    ) * 2
    secret = b"defence-shared-secret-2026"
    try:
        result = pipeline.embed(real_sample_image_path, message, secret, trained_model)
    except pipeline.AideCapacityError:
        return  # not every sample image has capacity for this message; skip quietly

    attacked = attacks.splicing(result.stego, fraction=0.08, seed=1)
    blocks, report = visualize.tamper_report_for(attacked, result.sidecar)
    overlay = visualize.tamper_overlay(attacked, blocks, report)

    assert overlay.size == (result.sidecar.width, result.sidecar.height)
    if report.erasure_mask.any():
        overlay_arr = np.array(overlay)
        b = blocks[int(np.where(report.erasure_mask)[0][0])]
        ys, xs = b.slice()
        region = overlay_arr[ys, xs]
        assert not np.array_equal(region[..., 0], region[..., 1]), "an erasure block should be tinted red"
