"""Run the RGB brightness/saturation baseline on the Belize eval patch and save a preview.

Deliverable for issue #9 ("Run it on the Belize patch and save a preview of the mask").
Produces a 3-panel figure — RGB | binary mask | red overlay — plus the raw mask PNG.

Usage:
    python scripts/preview_rgb_threshold.py [--b 0.5] [--s 0.2]

Reads the patch directly from the public Maxar COG (needs network), so it reproduces
anywhere without carrying image files around. The default thresholds are a reasonable
starting point; the *tuned* b and s come from the Task 2.3 sweep (issue #10).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import PATCH_ID, load_belize_patch
from src.methods.rgb_threshold import detect

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def stretch(rgb: np.ndarray) -> np.ndarray:
    """Per-band 2-98 percentile stretch, for *display only* (detect uses raw DNs)."""
    out = rgb.astype(np.float32)
    for i in range(3):
        p2, p98 = np.percentile(out[..., i], (2, 98))
        out[..., i] = np.clip((out[..., i] - p2) / (p98 - p2 + 1e-9), 0, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=float, default=0.5, help="brightness threshold in [0,1]")
    ap.add_argument("--s", type=float, default=0.2, help="saturation threshold in [0,1]")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Loading patch {PATCH_ID} from Maxar ...")
    rgb, valid = load_belize_patch()
    mask = detect(rgb, args.b, args.s)

    cloud_frac = mask[valid].mean() * 100 if valid.any() else mask.mean() * 100
    print(f"patch shape: {mask.shape}  valid: {valid.mean()*100:.1f}%  "
          f"cloud: {cloud_frac:.1f}%  (b={args.b}, s={args.s})")

    disp = stretch(rgb)
    overlay = disp.copy()
    overlay[mask == 1] = [1.0, 0.0, 0.0]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(disp)
    ax[0].set_title(f"{PATCH_ID}\nRGB (display stretch)", fontsize=9)
    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title(f"cloud mask  (b={args.b}, s={args.s})\ncloud={cloud_frac:.0f}%", fontsize=9)
    ax[2].imshow(overlay)
    ax[2].set_title("overlay (red = cloud)", fontsize=9)
    for a in ax:
        a.axis("off")
    plt.tight_layout()

    fig_path = os.path.join(RESULTS_DIR, f"{PATCH_ID}_rgb_threshold.png")
    plt.savefig(fig_path, dpi=90, bbox_inches="tight")
    print(f"saved preview -> {fig_path}")

    # Also save the raw binary mask (same geometry as a ground-truth mask would use).
    mask_path = os.path.join(RESULTS_DIR, f"{PATCH_ID}_rgb_threshold_mask.png")
    plt.imsave(mask_path, mask * 255, cmap="gray", vmin=0, vmax=255)
    print(f"saved mask    -> {mask_path}")


if __name__ == "__main__":
    main()
