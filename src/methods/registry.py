"""Detector registry for the shared batch runner (issue #18).

The three baselines expose different ``detect`` signatures:

    rgb : detect(image, b, s)
    dcp : detect(image, t=0.6, **kwargs)
    hot : detect(image, ...)            # issue #17, not yet implemented

The batch runner only wants one uniform interface: ``detect_fn(image) -> uint8 mask``
(1 = cloud). This module wraps each baseline into that shape, baking the tuned
parameters in with ``functools.partial`` so the runner stays method-agnostic — adding
a method is a line here, not a change in the runner.
"""

from __future__ import annotations

import csv
import functools
import os

from src.methods import rgb_threshold

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RGB_SWEEP = os.path.join(_ROOT, "results", "rgb_sweep.csv")

# Fallbacks if the Task-2.3 sweep output (results/rgb_sweep.csv) isn't present; they
# match the defaults in scripts/preview_rgb_threshold.py.
_DEFAULT_B, _DEFAULT_S = 0.5, 0.2

# Methods the runner can resolve today. HOT joins once issue #17 lands a detect().
METHODS = ("rgb", "dcp")


def best_rgb_thresholds(sweep_csv: str = _RGB_SWEEP):
    """Return the ``(b, s)`` with the highest F1 from the Task-2.3 sweep, if available.

    Reads ``results/rgb_sweep.csv`` (columns ``b, s, precision, recall, f1``) and picks
    the best-F1 row so the RGB rerun uses the *tuned* thresholds rather than the raw
    defaults. Falls back to ``(0.5, 0.2)`` when the file is missing or unreadable.
    """
    try:
        with open(sweep_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        best = max(rows, key=lambda r: float(r["f1"]))
        return float(best["b"]), float(best["s"])
    except (OSError, KeyError, ValueError):
        return _DEFAULT_B, _DEFAULT_S


def get_detector(name: str):
    """Return a ``detect_fn(image) -> uint8 mask`` for the named baseline.

    ``dcp`` is imported lazily so the registry still works for RGB even when
    ``src/methods/dcp_detect.py`` isn't on the branch yet (it arrives with PR #19).
    """
    key = name.lower()
    if key == "rgb":
        b, s = best_rgb_thresholds()
        fn = functools.partial(rgb_threshold.detect, b=b, s=s)
        fn.params = {"b": b, "s": s}  # for logging which thresholds were used
        return fn
    if key == "dcp":
        from src.methods.dcp_detect import detect  # lazy: see docstring
        return detect
    if key == "hot":
        raise NotImplementedError(
            "HOT detector (issue #17) is not implemented yet; register it here once "
            "src/methods/hot_detect.py lands with a detect(image) interface."
        )
    raise ValueError(f"unknown method {name!r}; known methods: {METHODS}")
