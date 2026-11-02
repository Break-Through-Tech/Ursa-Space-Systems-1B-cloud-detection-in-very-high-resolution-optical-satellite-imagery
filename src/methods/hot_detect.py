import numpy as np


def hot_index(
    image: np.ndarray,
    blue_slope: float = 1.0,
    blue_intercept: float = 0.0,
) -> np.ndarray:
    """Calculate HOT scores for an RGB image in HWC or CHW layout."""
    rgb = np.asarray(image)
    if rgb.ndim != 3:
        raise ValueError(f"Expected a 3D RGB image, got shape {rgb.shape}")

    if rgb.shape[-1] == 3:
        pass
    elif rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    else:
        raise ValueError(f"Expected 3 RGB channels, got shape {rgb.shape}")

    rgb = rgb.astype(np.float32, copy=False)
    if not np.isfinite(rgb).all():
        raise ValueError("RGB image contains non-finite values")

    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    red = rgb[:, :, 0]
    blue = rgb[:, :, 2]
    expected_blue = blue_slope * red + blue_intercept
    return blue - expected_blue


def detect_hot(
    image: np.ndarray,
    threshold: float = 0.05,
    blue_slope: float = 1.0,
    blue_intercept: float = 0.0,
) -> np.ndarray:
    """
    Cloud detection using HOT (Hazed Optimized Transformation) index.

    The HOT index measures the deviation of pixels from the clear-sky line
    in blue-red space. Higher values indicate haze/thin cloud.

    Parameters:
    -----------
    image : numpy array
        RGB image with shape (height, width, 3) or (3, height, width)
        Band order: [R, G, B]
    threshold : float
        HOT index threshold. Pixels with HOT > threshold are flagged as cloud.
    blue_slope : float
        Slope of the clear-sky line
    blue_intercept : float
        Intercept of the clear-sky line

    Returns:
    --------
    numpy array : Binary mask where 1 = cloud, 0 = non-cloud
    """
    return (hot_index(image, blue_slope, blue_intercept) > threshold).astype(np.uint8)


if __name__ == "__main__":
    # Test the HOT detection method
    print("HOT Cloud Detection Module")
    print("Available method: detect_hot")
