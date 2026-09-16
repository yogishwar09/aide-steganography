"""Visual analysis for the web console: turning the AIDE decision engine's
per-block choices, tamper/erasure status, and raw embedding distortion
into images a human can actually look at, instead of only numbers.

Three views, each answering a different question an evaluator or analyst
would ask:

1. `embedding_map_overlay` - "where did the adaptive engine actually put
   the payload, and with which method?" Colors each used block by
   LSB/DCT/DWT. This is the most direct visual evidence of the report's
   central novelty claim (a learned, per-block, multi-method policy - not
   a fixed rule) - you can see the method assignment track texture/edges
   across the image.
2. `diff_heatmap` - "how much did the image actually change, and where?"
   Real per-pixel |cover - stego| difference, amplified and colorized,
   since the true difference is by design too small to see directly (that
   is the whole point of steganography) but is still worth showing
   honestly rather than just asserting "PSNR is high."
3. `tamper_overlay` - "which blocks does the receiver actually trust after
   an attack?" Colors blocks by aide.tamper's verdict: erasure (red),
   hash-mismatch-but-not-erasure (amber, corrected by blind ECC instead),
   or trusted (left uncolored).
4. `suspicion_heatmap_overlay` - "where in an *arbitrary intercepted*
   image does the blind steganalysis in aide.steganalysis look most
   suspicious?" Colorizes `steganalysis.spatial_suspicion_grid`'s coarse
   chi-square grid (blue=low, red=high) over the analyzed image. Unlike
   the other three views, this one comes with an explicit statistical
   caveat carried over from the module it's built on: see
   aide/steganalysis.py's docstring on why individual hot cells are
   diagnostic, not conclusive on their own.

All four operate on a single 2D grayscale/luma plane - consistent with
the rest of this project's Y-plane-only design (see aide/pipeline.py).
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from . import tamper as tamper_mod
from .features import Block, partition_blocks
from .pipeline import EmbeddingMapEntry

# Matches the web console's CSS accent palette (webapp/templates/index.html)
METHOD_COLORS: dict[str | None, tuple[int, int, int]] = {
    "LSB": (240, 166, 62),    # --warn: least robust, flagged like a caution color
    "DCT": (43, 214, 168),    # --dct-color: the main workhorse
    "DWT": (124, 111, 240),   # --dwt-color / --accent
}


def _base_rgba(gray: np.ndarray) -> Image.Image:
    return Image.fromarray(gray.astype(np.uint8), mode="L").convert("RGBA")


def embedding_map_overlay(gray: np.ndarray, blocks: list[Block],
                          embedding_map: list[EmbeddingMapEntry], alpha: float = 0.45) -> Image.Image:
    """Color every block that carried payload by its assigned method;
    blocks the AIDE engine skipped or didn't need are left untouched."""
    base = _base_rgba(gray)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    a = int(255 * alpha)
    for entry in embedding_map:
        block = blocks[entry.block_index]
        color = METHOD_COLORS.get(entry.method)
        if color is None:
            continue
        draw.rectangle([block.x0, block.y0, block.x1 - 1, block.y1 - 1], fill=(*color, a))
    return Image.alpha_composite(base, overlay).convert("RGB")


def diff_heatmap(cover_gray: np.ndarray, stego_gray: np.ndarray, amplify: float = 10.0) -> Image.Image:
    """|cover - stego| per pixel, amplified and mapped through a simple
    black -> amber -> white heat ramp so an honestly tiny difference (a
    handful of intensity levels) is still visible as an image."""
    diff = np.abs(cover_gray.astype(np.int16) - stego_gray.astype(np.int16)).astype(np.float64)
    amp = np.clip(diff * amplify, 0, 255).astype(np.uint8)
    rgb = np.zeros((*amp.shape, 3), dtype=np.uint8)
    rgb[..., 0] = amp                                   # red ramps up first
    rgb[..., 1] = np.clip(amp.astype(np.int16) - 80, 0, 255).astype(np.uint8)  # green kicks in mid-range -> amber
    rgb[..., 2] = np.clip(amp.astype(np.int16) - 170, 0, 255).astype(np.uint8)  # blue only at the hottest pixels -> white
    return Image.fromarray(rgb, mode="RGB")


def suspicion_heatmap_overlay(gray: np.ndarray, grid: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Colorize `aide.steganalysis.spatial_suspicion_grid`'s coarse p-value
    grid (blue=low suspicion -> red=high) and blend it over the analyzed
    image, upsampled with nearest-neighbor so the grid's real cell
    boundaries stay visible rather than implying false spatial precision.
    Diagnostic only, same caveat as the grid itself: see
    aide/steganalysis.py's module and spatial_suspicion_grid docstrings on
    why individual hot cells are not, on their own, conclusive."""
    base = _base_rgba(gray)
    h, w = gray.shape
    grid = np.asarray(grid, dtype=np.float64)
    gh, gw = grid.shape

    # blue (cool/low) -> amber -> red (hot/high) ramp, matching the app's warn/danger palette
    heat = np.zeros((gh, gw, 3), dtype=np.uint8)
    heat[..., 0] = np.clip(grid * 255, 0, 255).astype(np.uint8)              # red channel rises with p
    heat[..., 1] = np.clip((1 - np.abs(grid - 0.5) * 2) * 140, 0, 255).astype(np.uint8)  # green peaks at mid (amber transition)
    heat[..., 2] = np.clip((1 - grid) * 255, 0, 255).astype(np.uint8)        # blue falls as p rises

    heat_img = Image.fromarray(heat, mode="RGB").resize((w, h), resample=Image.NEAREST).convert("RGBA")
    alpha_channel = Image.new("L", (w, h), int(255 * alpha))
    heat_img.putalpha(alpha_channel)
    return Image.alpha_composite(base, heat_img).convert("RGB")


def tamper_overlay(gray: np.ndarray, blocks: list[Block],
                   report: tamper_mod.TamperReport, alpha: float = 0.55) -> Image.Image:
    """Red = erasure (flagged unreliable, fed to Reed-Solomon as a known
    erasure). Amber = hash mismatch but not flagged as an erasure (left to
    blind error correction). Untouched = trusted."""
    base = _base_rgba(gray)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    erasure_a = int(255 * alpha)
    mismatch_a = int(255 * alpha * 0.55)
    for block in blocks:
        if report.erasure_mask[block.index]:
            color, a = (242, 85, 95), erasure_a      # --danger
        elif report.hash_mismatch[block.index]:
            color, a = (240, 166, 62), mismatch_a    # --warn
        else:
            continue
        draw.rectangle([block.x0, block.y0, block.x1 - 1, block.y1 - 1], fill=(*color, a))
    return Image.alpha_composite(base, overlay).convert("RGB")


def tamper_report_for(gray: np.ndarray, sidecar) -> tuple[list[Block], tamper_mod.TamperReport]:
    """Convenience: reconstruct the block grid from sidecar geometry and
    run tamper analysis, for callers (the web API) that only have the
    received image + sidecar, not an in-memory Block list."""
    blocks = partition_blocks(sidecar.height, sidecar.width, sidecar.block_size)
    report = tamper_mod.analyze(gray, blocks, sidecar.references)
    return blocks, report
