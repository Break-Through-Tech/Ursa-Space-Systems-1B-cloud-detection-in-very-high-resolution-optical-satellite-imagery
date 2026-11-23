"""SAM (Segment Anything) zero-shot cloud detector — Milestone 3 foundation-model
baseline.

SAM is the general-purpose segmentation foundation model from Meta AI
(Kirillov et al., ICCV 2023, https://github.com/facebookresearch/segment-anything).
Unlike the classical baselines it was *not* trained for clouds: it just finds
object-like regions in an image. This module runs SAM's automatic mask generator
over the RGB visual tiles already in the pipeline (no NIR, no fine-tuning) and
turns its object masks into a binary cloud mask with the same brightness /
saturation cue the RGB baseline uses:

    for each SAM object mask:
        keep the whole region iff  mean(brightness) > b  AND  mean(saturation) < s

So SAM supplies the *regions* (what counts as one "thing" in the scene) and the
cloud-ness rule decides whether each region is cloud. The rule is deliberately
the simplest one — the point of the comparison is to see how far a generic
segmenter plus a trivial spectral cue gets vs. the purpose-built classical
baselines (and later, vs. a cloud-specific pretrained model).

Running it (Colab GPU — SAM needs a checkpoint and ~4 GB VRAM for ViT-H)
--------------------------------------------------------------------------
    pip install git+https://github.com/facebookresearch/segment-anything
    # download one checkpoint, e.g.:
    #   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    export SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth
    export SAM_MODEL_TYPE=vit_h        # or vit_l / vit_b for the smaller checkpoints
    python scripts/run_baselines.py --method sam

``segment_anything`` and the checkpoint are imported/loaded lazily on the first
``detect()`` call, so ``registry.get_detector("sam")`` and the batch runner work
even on machines without SAM installed — the failure surfaces per-scene in the
results CSV rather than at import time.

Where SAM helps / hurts vs. the purpose-built baselines (see docs/06 for the
full write-up and Ren et al., "Segment Anything, from Space?", WACV 2024):

  * Helps: region-level reasoning groups a cloud deck into coherent objects, so
    the mask tends to have cleaner, closed boundaries than a per-pixel threshold
    (which speckles and leaks through thin cloud).
  * Hurts: SAM's notion of "object" is generic — it happily segments snow, sand,
    bright rooftops, and coastline as objects just as readily as cloud, so the
    brightness/saturation rule still has to reject them; it also tends to
    under-segment diffuse haze/thin cloud that has no sharp boundary (a known
    remote-sensing failure mode), and its grid-based point prompts can miss
    small or low-contrast clouds entirely.
"""

from __future__ import annotations

import os

import numpy as np

from src.methods import rgb_threshold

__all__ = ["detect"]

# Cached on first use; keyed by checkpoint path so switching checkpoints in one
# process (e.g. vit_b smoke test -> vit_h full run) reloads correctly.
_generator = None
_generator_ckpt = None


def _to_uint8_hwc(image: np.ndarray) -> np.ndarray:
    """Return the image as an ``(H, W, 3)`` uint8 RGB array for SAM."""
    rgb = rgb_threshold._to_hwc_float(image)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def _get_generator():
    """Build (once) the ``SamAutomaticMaskGenerator`` from the checkpoint.

    Configuration comes from the environment so the same code runs vit_h on a
    Colab T4 and vit_b on a weaker box:

        SAM_CHECKPOINT  path to the .pth (required)
        SAM_MODEL_TYPE  vit_h | vit_l | vit_b   (default vit_h)
        SAM_POINTS_PER_SIDE  grid density for automatic mask generation
                             (default 32; lower = faster, coarser)
    """
    global _generator, _generator_ckpt
    ckpt = os.environ.get("SAM_CHECKPOINT")
    if not ckpt:
        raise FileNotFoundError(
            "SAM_CHECKPOINT is not set. Download a checkpoint from "
            "https://github.com/facebookresearch/segment-anything#model-checkpoints "
            "and point SAM_CHECKPOINT at it (see module docstring)."
        )
    if _generator is not None and _generator_ckpt == ckpt:
        return _generator

    import torch
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    model_type = os.environ.get("SAM_MODEL_TYPE", "vit_h")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = sam_model_registry[model_type](checkpoint=ckpt)
    model.to(device=device)

    _generator = SamAutomaticMaskGenerator(
        model,
        points_per_side=int(os.environ.get("SAM_POINTS_PER_SIDE", "32")),
    )
    _generator_ckpt = ckpt
    return _generator


def detect(image: np.ndarray, b: float = 0.5, s: float = 0.2) -> np.ndarray:
    """Binary cloud mask from SAM object masks + a brightness/saturation rule.

    Parameters
    ----------
    image : np.ndarray
        RGB tile, ``(H, W, 3)`` or ``(3, H, W)``, 8-bit DNs or floats — the same
        input ``rgb_threshold.detect`` takes (512 x 512 tiles in the batch run).
    b : float
        Mean-brightness threshold in ``[0, 1]`` for a SAM region to count as
        cloud. Higher = stricter.
    s : float
        Mean-saturation threshold in ``[0, 1]`` (cloud is gray, so lower passes).
        Lower = stricter.

    Returns
    -------
    np.ndarray
        ``uint8`` mask, same ``(H, W)`` as the input; ``1`` = cloud, ``0`` = not.
    """
    rgb_u8 = _to_uint8_hwc(image)
    generator = _get_generator()
    sam_masks = generator.generate(rgb_u8)

    brightness = rgb_threshold.brightness(image)
    saturation = rgb_threshold.saturation(image)

    cloud = np.zeros(brightness.shape, dtype=bool)
    for m in sam_masks:
        region = np.asarray(m["segmentation"], dtype=bool)
        if not region.any():
            continue
        # Cloud-ness of the whole object: bright on average and unsaturated on
        # average — the same cue as the RGB baseline, applied per SAM region.
        if (brightness[region].mean() > b) and (saturation[region].mean() < s):
            cloud |= region
    return cloud.astype(np.uint8)
