"""Pre-processing of user images and synthetic corruptions used in robustness studies."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


# ----------------------------------------------------------------------------
# user input -> digit-like 16x16 image
# ----------------------------------------------------------------------------
def to_digit_image(img: np.ndarray, size: int = 16, margin: int = 2) -> np.ndarray:
    """Normalise an arbitrary grey-scale picture to the convention of the digits dataset.

    * convert to float in [0, 1]
    * if the background is brighter than the stroke, invert (dataset: ink = high value)
    * crop to the bounding box of the stroke, centre it on a square canvas, resize
    """
    a = np.asarray(img, dtype=float)
    if a.ndim == 3:
        a = a[..., :3].mean(-1)
    if a.max() > 1.0:
        a = a / 255.0
    if np.median(a) > 0.5:
        a = 1.0 - a
    a = (a - a.min()) / max(a.max() - a.min(), 1e-9)
    mask = a > 0.25
    if mask.any():
        ys, xs = np.where(mask)
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape
    s = max(h, w)
    canvas = np.zeros((s, s))
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
    inner = size - 2 * margin
    out = ndimage.zoom(canvas, inner / s, order=1)
    res = np.zeros((size, size))
    oh, ow = out.shape
    oy, ox = (size - oh) // 2, (size - ow) // 2
    res[oy:oy + oh, ox:ox + ow] = out[: size - oy, : size - ox]
    return np.clip(res, 0, 1)


# ----------------------------------------------------------------------------
# corruptions
# ----------------------------------------------------------------------------
def add_gaussian_noise(X, sigma, rng):
    return np.clip(X + rng.normal(0, sigma, X.shape), 0, 1)


def add_occlusion(X, frac, rng):
    """Zero out a random square patch whose side is ``frac`` of the image side."""
    X = X.copy()
    n, h, w = X.shape
    s = max(1, int(round(frac * h)))
    for i in range(n):
        y0 = rng.integers(0, h - s + 1)
        x0 = rng.integers(0, w - s + 1)
        X[i, y0:y0 + s, x0:x0 + s] = 0.0
    return X


def rotate(X, degrees, rng=None):
    return np.stack([ndimage.rotate(im, degrees, reshape=False, order=1, mode="constant", cval=0.0) for im in X])


def shift(X, pixels, rng=None):
    return np.stack([ndimage.shift(im, (0, pixels), order=1, mode="constant", cval=0.0) for im in X])


def salt_pepper(X, p, rng):
    X = X.copy()
    r = rng.random(X.shape)
    X[r < p / 2] = 0.0
    X[r > 1 - p / 2] = 1.0
    return X


CORRUPTIONS = {
    "gaussian_noise": (add_gaussian_noise, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], "noise std"),
    "salt_pepper": (salt_pepper, [0.0, 0.05, 0.1, 0.2, 0.3], "corrupted pixel fraction"),
    "occlusion": (add_occlusion, [0.0, 0.125, 0.25, 0.375, 0.5], "patch side / image side"),
    "rotation": (rotate, [0, 5, 10, 15, 20, 30], "rotation (degrees)"),
    "shift": (shift, [0, 1, 2, 3, 4], "horizontal shift (pixels)"),
}
