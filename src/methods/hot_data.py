"""Image and mask loading helpers shared by the HOT notebook and runner."""

import numpy as np
import rasterio
from rasterio.enums import Resampling


class UnsupportedImageError(ValueError):
    """Raised when a source image lacks the RGB bands required by HOT."""


def load_cloud_mask(mask_href: str) -> np.ndarray:
    with rasterio.open(mask_href) as source:
        return source.read(1)


def load_rgb_at_shape(visual_href: str, target_shape: tuple[int, int]) -> np.ndarray:
    """Read bands 1-3 and resample them to the mask shape."""
    with rasterio.open(visual_href) as source:
        if source.count < 3:
            raise UnsupportedImageError(
                f"HOT requires an RGB visual with at least 3 bands; "
                f"{visual_href} has {source.count}"
            )
        if (source.height, source.width) != target_shape:
            image = source.read(
                [1, 2, 3],
                out_shape=(3, target_shape[0], target_shape[1]),
                resampling=Resampling.average,
            )
        else:
            image = source.read([1, 2, 3])

    rgb = np.transpose(image, (1, 2, 0))
    if rgb.shape[:2] != target_shape:
        raise ValueError(
            f"Resampled image shape {rgb.shape[:2]} does not match mask {target_shape}"
        )
    return rgb
