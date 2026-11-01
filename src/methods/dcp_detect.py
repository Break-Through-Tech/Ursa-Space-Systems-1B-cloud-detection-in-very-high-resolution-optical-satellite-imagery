"""
reference: https://people.csail.mit.edu/kaiming/publications/cvpr09.pdf

Cite the paper: Single Image Haze Removal Using Dark Channel Prior , Kaiming He, Jian sun, Xiaoou Tang, 2009
The prior in one line: clear outdoor patches have a near-zero channel somewhere (Sec. 3)

Why that flags clouds: haze/cloud adds airlight, so the dark channel rises (Eq. 1, end of Sec. 3)

Cloud detection with the Dark Channel Prior (DCP).

Paper: Kaiming He, Jian Sun, Xiaoou Tang, "Single Image Haze Removal Using Dark Channel Prior", CVPR 2009.

The prior (Sec. 3): in clear outdoor images, almost every small patch has at least one pixel that is very dark
in at least one color channel.

Haze and cloud add bright "airlight" to every pixel (Eq. 1), so the dark channel rises where they are. 
We use that rise as a cloud score.
"""
import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter, uniform_filter



def to_float(image, scale=255.0):
    # converts the image into floats (btw 0 and 1)
    # scale is 255 for the 8-bit visual asset and about 10000 for ms_analytic.
    # Maxar rasters have no nodata value, so a pixel that is 0 in every band is fill, and we mark it invalid.
    img = np.clip(image.astype(np.float32) / scale, 0, 1)
    valid = image.max(axis=2) > 0
    return img, valid


def dark_channel(img, window=15, open_edges=True):
    # the dark channel is a min over color channels, then a min over a local patch around each pixel.
    dark = np.min(img, axis=2)                    # min over R, G, B
    dark = minimum_filter(dark, size=window)      # min over the patch
    if open_edges:
        # The min filter shrinks bright regions (clouds) by about half a window at their edges.
        # A max filter of the same size grows them back (a grey opening), which keeps cloud edges and helps recall.
        dark = maximum_filter(dark, size=window)
    return dark


def atmospheric_light(img, valid, window=15, top_frac=0.001):
    # Sec. 4.4: estimate A, the color of the haze/airlight.
    # Take the top 0.1% brightest pixels in the dark channel (the most haze-opaque spots), then pick the brightest of those in the image.
    # This avoids picking a white car or building, which the plain "brightest pixel" method would (Sec. 4.4).
    # The plain dark channel is used here.
    dark = dark_channel(img, window, open_edges=False)
    dark = np.where(valid, dark, -1)              # fill pixels can never be picked
    n = max(1, int(top_frac * np.count_nonzero(valid)))
    idx = np.argpartition(dark.ravel(), -n)[-n:]  # indices of the top n pixels
    candidates = img.reshape(-1, img.shape[2])[idx]
    A = candidates[np.argmax(candidates.sum(axis=1))]
    return np.maximum(A, 1e-6)                    # avoid dividing by zero later


def guided_filter(guide, src, radius=30, eps=1e-3):
    # Refine the blocky map with soft matting, which is slow. The guided filter (He, Sun, Tang, "Guided Image
    # Filtering", ECCV 2010) does the same job fast. It smooths src while making its edges follow the edges in guide (the grayscale image).
    # radius sets the window size; eps sets how strongly edges are kept (smaller eps keeps more edges).
    size = 2 * radius + 1
    mean_g = uniform_filter(guide, size)
    mean_s = uniform_filter(src, size)
    var_g = uniform_filter(guide * guide, size) - mean_g * mean_g
    cov_gs = uniform_filter(guide * src, size) - mean_g * mean_s
    a = cov_gs / (var_g + eps)                    # local linear model: src ~ a*guide + b
    b = mean_s - a * mean_g
    return uniform_filter(a, size) * guide + uniform_filter(b, size)


def dcp_score(image, window=15, scale=255.0, normalize=True, top_frac=0.001, 
              omega=0.95, open_edges=True, refine=False, radius=30, eps=1e-3):
    # Returns a cloud score between 0 and 1 for every pixel (high = cloud).
    # Every technique is a parameter so tuning never needs a code change.
    orig, valid = to_float(image, scale)
    img = orig
    if normalize:
        # Eq. 11 uses the image divided by A (the "normalized haze image").
        A = atmospheric_light(orig, valid, window, top_frac)
        img = np.clip(orig / A, 0, 1)
    # Eq. 12: transmission t = 1 - omega * dark(I / A), with omega = 0.95. 
    # We keep 1 - t = omega * dark, so a high score means thick haze/cloud.
    score = omega * dark_channel(img, window, open_edges)
    if refine:
        score = guided_filter(orig.mean(axis=2), score, radius, eps)
    score = np.clip(score, 0, 1)
    score[~valid] = 0                             # fill pixels are never cloud
    return score


def detect(image, t=0.6, **kwargs):
    # Team interface: detect(image, ...) -> mask, with 1 = cloud.
    # t is the cloud threshold. **kwargs passes any dcp_score setting (window, normalize, refine, ...) straight through.
    # Known false positives: bright objects that look like airlight.
    score = dcp_score(image, **kwargs)
    return (score > t).astype(np.uint8)