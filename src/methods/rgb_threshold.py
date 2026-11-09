"""RGB brightness/saturation cloud detector (Milestone 2, Task 2.2 — issue #9).

The baseline the team scores everything else against. The idea is the simplest
cue there is: in visible RGB, thick cloud is **bright** (high value) and **gray**
(low saturation, R=G=B), so a pixel is flagged as cloud when

    brightness > b   AND   saturation < s

where ``brightness`` is the HSV value channel ``max(R, G, B)`` and ``saturation``
is the HSV saturation ``(max - min) / max``. Both are computed in ``[0, 1]`` so the
thresholds ``b`` and ``s`` are also in ``[0, 1]`` and comparable across scenes.

This module is pure NumPy (no scikit-image / OpenCV) so the formula stays fully
transparent and the sweep in Task 2.3 (issue #10) can call ``detect`` tens of
thousands of times cheaply. It is the function imported by:

    * ``src/eval.py``  (issue #8)  -> scores ``detect(...)`` against the mask
    * the threshold sweep (issue #10) -> loops ``detect(image, b, s)`` over a grid

Where it fails (documented so the team knows the baseline's blind spots): anything
that is *also* bright and desaturated beats this test — snow, sand, bare concrete,
bright rooftops, and thin cloud / haze over a bright surface. See ``docs`` for the
write-up.
"""

from __future__ import annotations

import numpy as np

__all__ = ["detect", "brightness", "saturation"]


def _to_hwc_float(image: np.ndarray) -> np.ndarray:
    """Return the image as an ``(H, W, 3)`` float array scaled to ``[0, 1]``.

    Accepts the two layouts the project actually produces:
      * ``(H, W, 3)`` — what you get after ``np.transpose(src.read(...), (1, 2, 0))``
      * ``(3, H, W)`` — the raw ``rasterio`` band-first layout

    and either 8-bit integer DNs (the Maxar "visual" COG) or already-normalised
    floats. Integers are divided by their dtype max; floats with values above 1 are
    assumed to be 8-bit and divided by 255; floats already in ``[0, 1]`` pass through.
    """
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-D RGB image, got shape {arr.shape}")

    # Move a leading 3-band axis to the end. (A genuine 3x3x3 image is not a thing
    # we read here, so preferring the trailing axis when both are 3 is safe.)
    if arr.shape[-1] != 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] != 3:
        raise ValueError(f"expected 3 channels (RGB), got shape {arr.shape}")

    arr = arr.astype(np.float32)
    if np.issubdtype(np.asarray(image).dtype, np.integer):
        arr /= float(np.iinfo(np.asarray(image).dtype).max)
    elif arr.max() > 1.0:
        arr /= 255.0
    return arr


def brightness(image: np.ndarray) -> np.ndarray:
    """HSV *value* channel ``max(R, G, B)`` as an ``(H, W)`` float array in ``[0, 1]``."""
    return _to_hwc_float(image).max(axis=2)


def saturation(image: np.ndarray) -> np.ndarray:
    """HSV *saturation* ``(max - min) / max`` as an ``(H, W)`` float array in ``[0, 1]``.

    Black pixels (value 0, e.g. nodata fill) have no defined hue/saturation and are
    returned as 0 rather than dividing by zero.
    """
    rgb = _to_hwc_float(image)
    v = rgb.max(axis=2)
    c = v - rgb.min(axis=2)  # chroma
    return np.divide(c, v, out=np.zeros_like(v), where=v > 0)


def detect(image: np.ndarray, b: float, s: float) -> np.ndarray:
    """Binary cloud mask from RGB brightness and saturation thresholds.

    cloud  <=>  brightness > ``b``  AND  saturation < ``s``

    Parameters
    ----------
    image : np.ndarray
        RGB image, ``(H, W, 3)`` or ``(3, H, W)``, 8-bit DNs or floats (see
        ``_to_hwc_float``). For the pinned Belize eval patch this is ``1024 x 1024``.
    b : float
        Brightness threshold in ``[0, 1]``. Higher = stricter (fewer pixels pass).
    s : float
        Saturation threshold in ``[0, 1]``. Lower = stricter (only very gray pixels).

    Returns
    -------
    np.ndarray
        ``uint8`` mask with the same ``(H, W)`` as the input; ``1`` = cloud, ``0`` = not.
    """
    rgb = _to_hwc_float(image)
    v = rgb.max(axis=2)
    c = v - rgb.min(axis=2)
    sat = np.divide(c, v, out=np.zeros_like(v), where=v > 0)
    mask = (v > b) & (sat < s)
    return mask.astype(np.uint8)
