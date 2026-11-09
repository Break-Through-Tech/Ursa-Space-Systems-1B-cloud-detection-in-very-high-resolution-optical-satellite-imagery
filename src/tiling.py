"""Multi-scene tiling helper for the Milestone 2 batch pipeline (issue #18).

Every classical baseline (RGB #9, DCP #16, HOT #17) has to be scored on the *same*
tiles across the *same* scenes, or the Task-5 comparison is apples-to-oranges. This
module is the shared front end that turns one Maxar scene into aligned image/mask
tiles that any ``detect(image)`` can run on.

The alignment problem, and how we solve it
------------------------------------------
A Maxar ARD "visual" COG is full resolution (~17408 px) while its companion cloud
mask ("-clouds.tif") is an ~8x-decimated raster (~2176 px). They share the same CRS
and bounds, so the fix is the one already used in ``scripts/runner_dcp.py``: read the
visual *resampled onto the mask's grid* (``out_shape=mask.shape``,
``Resampling.average``). After that the image and mask are pixel-for-pixel aligned, so
a plain NumPy slice of both at the same offsets is a matching tile.

Mask value convention (Maxar's cloud model, see ``scripts/scene_finder.py``):
``0 = nodata, 1 = clear, 2 = cloud, 3 = cloud shadow``. Callers score with
``evaluate(pred, mask == 2, mask != 0)`` — cloud is value 2, nodata (0) is excluded,
and shadow (3) counts as not-cloud.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.enums import Resampling


def read_scene(visual_href: str, mask_href: str):
    """Load one scene's RGB and cloud mask onto a single, aligned pixel grid.

    Parameters
    ----------
    visual_href, mask_href : str
        URLs/paths to the Maxar "visual" COG and its "-clouds.tif" mask (the two
        columns ``scene_finder.py`` writes to ``data/labeled_scenes.csv``).

    Returns
    -------
    image : np.ndarray
        ``(H, W, 3)`` uint8 RGB, resampled to the mask's ``(H, W)`` so it lines up
        with ``mask`` exactly.
    mask : np.ndarray
        ``(H, W)`` raw Maxar mask (0/1/2/3 as above).
    profile : dict
        The mask's rasterio profile (CRS, transform, H, W). Used to write a
        georeferenced predicted mask back out for GIS.
    """
    with rasterio.open(mask_href) as m:
        mask = m.read(1)
        profile = m.profile
    height, width = mask.shape

    with rasterio.open(visual_href) as v:
        if (v.height, v.width) != (height, width):
            bands = v.read([1, 2, 3], out_shape=(3, height, width),
                           resampling=Resampling.average)
        else:
            bands = v.read([1, 2, 3])
    image = np.transpose(bands, (1, 2, 0))  # (H, W, 3)
    return image, mask, profile


def iter_tiles(image: np.ndarray, mask: np.ndarray, tile: int = 512,
               skip_empty: bool = True):
    """Cut an aligned ``image``/``mask`` pair into matching ``tile x tile`` blocks.

    Yields one block at a time so a whole scene never needs all its tiles in memory
    at once. Edge blocks on the right/bottom are kept at their smaller size rather
    than padded, so stitched predictions line up with the source grid exactly.

    Yields
    ------
    tuple ``(tile_row, tile_col, r0, c0, image_tile, mask_tile)``
        ``tile_row, tile_col`` index the block in the grid; ``r0, c0`` are its
        pixel offsets in the full scene (so the runner can place predictions back);
        ``image_tile`` is ``(h, w, 3)`` and ``mask_tile`` is ``(h, w)``.
    """
    height, width = mask.shape
    for r0 in range(0, height, tile):
        for c0 in range(0, width, tile):
            r1, c1 = min(r0 + tile, height), min(c0 + tile, width)
            mask_tile = mask[r0:r1, c0:c1]
            if skip_empty and not np.any(mask_tile != 0):
                continue  # all nodata — nothing to score or predict
            yield r0 // tile, c0 // tile, r0, c0, image[r0:r1, c0:c1], mask_tile


def write_mask_gis(path, pred: np.ndarray, profile: dict) -> None:
    """Write a full-scene predicted mask as a georeferenced GeoTIFF for GIS.

    ``pred`` is the stitched ``(H, W)`` prediction (1 = cloud) on the mask's grid, so
    the mask's own ``profile`` (CRS + transform) georeferences it. The result opens
    aligned over the source scene in QGIS / leafmap. nodata is 0 and the file is
    LZW-compressed because a binary mask compresses to almost nothing.
    """
    prof = dict(profile)
    prof.update(count=1, dtype="uint8", nodata=0, compress="lzw")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(pred.astype("uint8"), 1)
