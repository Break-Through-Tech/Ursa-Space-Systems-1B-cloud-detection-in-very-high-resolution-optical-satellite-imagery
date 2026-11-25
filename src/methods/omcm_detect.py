# omcm_detect.py: OmniCloudMask (OMCM) cloud detector for Maxar scenes.
#
# OMCM needs Red, Green, and NIR, so it reads the multispectral (-ms) GeoTIFF,
# not the visual RGB tiles the other baselines use. Following OMCM's own Maxar
# example (examples/maxar.ipynb), bands are read at 10 m, the resolution the
# model is built for, then the mask is resized back to the label grid with
# nearest neighbour (masks are categories, so no blending between classes).
#
# Output: uint8 mask, 1 = cloud (thick or thin), 0 = everything else.
# OMCM classes: 0 clear, 1 thick cloud, 2 thin cloud, 3 cloud shadow.

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from omnicloudmask import predict_from_array

# WorldView 8-band order (1-indexed): 1 Coastal, 2 Blue, 3 Green, 4 Yellow,
# 5 Red, 6 RedEdge, 7 NIR1, 8 NIR2. OMCM's example uses NIR2.
BANDS_8 = [5, 3, 8]  # Red, Green, NIR
BANDS_4 = [1, 2, 4]  # 4-band scenes are R, G, B, NIR
TARGET_RES_M = 10
CLOUD_CLASSES = (1, 2)


def read_rgn(ms_path, target_res=TARGET_RES_M):
    """Read Red, Green, NIR from a Maxar -ms GeoTIFF, resampled to target_res metres.

    Returns (rgn, nodata) where rgn is float32 (3, h, w) and nodata is a bool
    (h, w) array that is True where every band is 0 (outside the image footprint).
    """
    with rasterio.open(ms_path) as src:
        bands = BANDS_8 if src.count >= 8 else BANDS_4
        scale = src.res[0] / target_res
        h = max(1, int(round(src.height * scale)))
        w = max(1, int(round(src.width * scale)))
        rgn = src.read(bands, out_shape=(3, h, w), resampling=Resampling.bilinear)
    rgn = rgn.astype(np.float32)  # raw digital numbers, no scaling (as in OMCM's example)
    nodata = np.all(rgn == 0, axis=0)
    return rgn, nodata


def resize_nearest(mask, out_hw):
    """Resize a 2D mask to out_hw = (H, W) with nearest neighbour."""
    H, W = out_hw
    h, w = mask.shape
    rows = np.minimum((np.arange(H) * h / H).astype(int), h - 1)
    cols = np.minimum((np.arange(W) * w / W).astype(int), w - 1)
    return mask[rows[:, None], cols[None, :]]


def predict_rgn(rgn):
    """Run OMCM on a (3, h, w) Red/Green/NIR array. Returns uint8 (h, w), 1 = cloud."""
    pred = predict_from_array(rgn)[0]
    return np.isin(pred, CLOUD_CLASSES).astype(np.uint8)


def detect_ms(ms_path, out_hw=None):
    """Cloud mask for a whole -ms scene.

    out_hw: (H, W) of the grid to return on, e.g. the Maxar label mask's shape
    (2176, 2176). If None, the 10 m prediction grid is returned.
    """
    rgn, nodata = read_rgn(ms_path)
    cloud = predict_rgn(rgn)
    cloud[nodata] = 0  # never call no-data pixels cloud
    if out_hw is not None:
        cloud = resize_nearest(cloud, out_hw)
    return cloud.astype(np.uint8)


def detect(image):
    """Registry entry point.

    image can be a path to a Maxar -ms GeoTIFF (returns the scene mask on the
    10 m grid), or a (3, H, W) Red/Green/NIR array already at ~10 m
    (returns a mask with the same H, W). OMCM cannot run on RGB-only tiles.
    """
    if isinstance(image, (str, Path)):
        return detect_ms(image)
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError("OMCM needs a (3, H, W) Red/Green/NIR array or a -ms GeoTIFF path")
    return predict_rgn(arr)


if __name__ == "__main__":
    import sys

    scene = sys.argv[1] if len(sys.argv) > 1 else "data/scene_1_10300101074E6F00-ms.tif"
    label = sys.argv[2] if len(sys.argv) > 2 else None  # optional Maxar cloud mask .tif

    if label:
        with rasterio.open(label) as src:
            truth = src.read(1)
        mask = detect_ms(scene, out_hw=truth.shape)
        valid = truth != 0  # Maxar: 0 nodata, 1 clear, 2 cloud, 3 shadow
        t = truth[valid] == 2
        p = mask[valid] == 1
        tp = int(np.sum(t & p)); fp = int(np.sum(~t & p)); fn = int(np.sum(t & ~p))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        print(f"truth cloud {t.mean():.1%} | pred cloud {p.mean():.1%}")
        print(f"precision {prec:.3f} recall {rec:.3f} IoU {iou:.3f}")
    else:
        mask = detect_ms(scene, out_hw=(2176, 2176))
        print(mask.shape, mask.dtype, np.unique(mask), f"{mask.mean():.1%} cloud")