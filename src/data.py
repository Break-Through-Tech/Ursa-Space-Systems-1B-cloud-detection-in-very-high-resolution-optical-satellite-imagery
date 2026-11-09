"""Shared, reproducible loader for the Milestone 2 evaluation patch.

Every Task-2 deliverable has to run on the *same* 1024x1024 patch or the scores
don't line up: Task 2.1 (#8) labels the ground-truth mask on it, Task 2.2 (#9)
runs ``detect`` on it, Task 2.3 (#10) sweeps thresholds on it. Rather than pass a
hand-cropped file around on Drive, we pin the patch in code here so anyone can
reproduce the exact pixels from the public Maxar COG.

Patch: ``belize_r11384_c1536``
  * event   : Belize-Wildfires-June24  (Maxar Open Data)
  * capture : 10300100FDC7C400, 2024-05-27
  * tile    : 033131012121  ("visual" 8-bit RGB COG, 17408 x 17408)
  * window  : full-resolution Window(col_off=1536, row_off=11384, 1024, 1024)

The window covers thick opaque cloud (top-left), a thin haze veil, and clear
forest/dirt-road ground (bottom-right) — a deliberate cloud/clear mix so precision
and recall are both meaningful.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.windows import Window

# --- Pinned patch definition (single source of truth for all of Milestone 2) ---
PATCH_ID = "belize_r11384_c1536"
COG_URL = (
    "https://maxar-opendata.s3.amazonaws.com/events/Belize-Wildfires-June24/"
    "ard/16/033131012121/2024-05-27/10300100FDC7C400-visual.tif"
)
ROW_OFF = 11384
COL_OFF = 1536
SIZE = 1024


def load_belize_patch(url: str = COG_URL, row_off: int = ROW_OFF,
                      col_off: int = COL_OFF, size: int = SIZE):
    """Read the pinned evaluation patch straight from the Maxar COG.

    Returns
    -------
    rgb : np.ndarray
        ``(size, size, 3)`` uint8 RGB patch (raw DNs, no contrast stretch — the
        detector needs the true brightness, not a display stretch).
    valid : np.ndarray
        ``(size, size)`` bool array, ``True`` where the pixel is real data. Maxar
        ARD tiles are partially filled; this patch is fully valid, but the mask is
        returned so ``evaluate(...)`` can exclude any nodata uniformly.
    """
    with rasterio.open(url) as src:
        window = Window(col_off, row_off, size, size)
        bands = src.read([1, 2, 3], window=window)          # (3, size, size) uint8
        valid = src.read_masks(1, window=window) > 0         # (size, size)
    rgb = np.transpose(bands, (1, 2, 0))                     # (size, size, 3)
    return rgb, valid
