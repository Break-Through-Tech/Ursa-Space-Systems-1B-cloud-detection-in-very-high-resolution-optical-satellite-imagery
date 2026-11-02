import numpy as np

def detect_hot(image: np.ndarray, threshold: float = 0.05,
               blue_slope: float = 1.0, blue_intercept: float = 0.0) -> np.ndarray:
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
    # Handle different input shapes
    if image.shape[0] == 3:  # (3, height, width)
        rgb = np.transpose(image, (1, 2, 0))
    else:  # (height, width, 3)
        rgb = image.copy()

    # Convert to float
    rgb = rgb.astype(float)

    # Normalize only if the image values are in the 0-255 range
    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    # Extract bands (assume RGB order: [R, G, B])
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    # Calculate expected blue value on clear-sky line
    expected_blue = blue_slope * red + blue_intercept

    # HOT index: deviation from clear-sky line
    hot_index = blue - expected_blue

    # Apply threshold
    cloud_mask = (hot_index > threshold).astype(np.uint8)

    return cloud_mask


if __name__ == "__main__":
    # Test the HOT detection method
    print("HOT Cloud Detection Module")
    print("Available method: detect_hot")
