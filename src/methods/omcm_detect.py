# omcm_detect.py: OmniCloudMask, but make it Maxar.
# Feeds Red, Green, and NIR from a Maxar -ms GeoTIFF into OmniCloudMask,
# upsampled to the 2176 px Maxar mask grid, and returns a uint8 mask (1 = cloud).


import numpy as np
import rasterio
from rasterio.enums import Resampling
from omnicloudmask import predict_from_array

RED, GREEN, NIR = 5, 3, 7


def detect_ms(ms_path, out_size=2176):
    with rasterio.open(ms_path) as src:
        bands = src.read(
            [RED, GREEN, NIR],
            out_shape=(3, out_size, out_size),
            resampling=Resampling.bilinear,
        )

    x = bands.astype(np.float32)
    pred = predict_from_array(x)[0]

    cloud = (pred == 1)| (pred == 2)
    return cloud.astype(np.uint8)


if __name__ == "__main__":
    m = detect_ms("data/scene_1_10300101074E6F00-ms.tif")
    print(m.shape, m.dtype, np.unique(m), round(m.mean() * 100, 1), "% cloud")