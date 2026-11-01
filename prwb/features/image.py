"""Image feature extraction implemented from scratch with NumPy.

Descriptors
-----------
raw        flattened pixel intensities
zoning     mean intensity over a g x g grid (average pooling)
profiles   horizontal / vertical projection profiles
stats      first-order statistics (mean, std, skew, kurtosis, entropy, ink ratio)
hog        Histogram of Oriented Gradients (Dalal & Triggs, 2005)
lbp        Local Binary Patterns, uniform / rotation-invariant (Ojala et al., 2002)
glcm       Grey-Level Co-occurrence statistics (Haralick et al., 1973)
hu         Hu's seven moment invariants (Hu, 1962)

All functions take a batch ``(n, h, w)`` and return ``(n, d)``.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats as sstats

EPS = 1e-12


# --------------------------------------------------------------------------- simple descriptors
def raw_pixels(X: np.ndarray) -> np.ndarray:
    return X.reshape(len(X), -1).astype(np.float64)


def zoning(X: np.ndarray, grid: int = 4) -> np.ndarray:
    n, h, w = X.shape
    gh, gw = h // grid, w // grid
    Xc = X[:, : gh * grid, : gw * grid]
    return Xc.reshape(n, grid, gh, grid, gw).mean(axis=(2, 4)).reshape(n, -1)


def projection_profiles(X: np.ndarray) -> np.ndarray:
    return np.concatenate([X.sum(axis=2), X.sum(axis=1)], axis=1)


def intensity_stats(X: np.ndarray) -> np.ndarray:
    flat = X.reshape(len(X), -1)
    hist_feats = []
    for row in flat:
        hist, _ = np.histogram(row, bins=16, range=(0, 1))
        p = hist / max(hist.sum(), 1)
        p = p[p > 0]
        hist_feats.append(-(p * np.log2(p)).sum())
    ink = (flat > 0.5).mean(axis=1)
    sk = np.nan_to_num(sstats.skew(flat, axis=1))
    ku = np.nan_to_num(sstats.kurtosis(flat, axis=1))
    return np.column_stack([flat.mean(1), flat.std(1), sk, ku, hist_feats, ink])


# --------------------------------------------------------------------------- HOG
def image_gradients(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Centred [-1, 0, 1] gradients; returns magnitude and unsigned orientation in [0, pi)."""
    gx = np.zeros_like(X)
    gy = np.zeros_like(X)
    gx[:, :, 1:-1] = X[:, :, 2:] - X[:, :, :-2]
    gx[:, :, 0] = X[:, :, 1] - X[:, :, 0]
    gx[:, :, -1] = X[:, :, -1] - X[:, :, -2]
    gy[:, 1:-1, :] = X[:, 2:, :] - X[:, :-2, :]
    gy[:, 0, :] = X[:, 1, :] - X[:, 0, :]
    gy[:, -1, :] = X[:, -1, :] - X[:, -2, :]
    mag = np.hypot(gx, gy)
    ang = np.arctan2(gy, gx) % np.pi
    return mag, ang


def hog_cells(X: np.ndarray, cell: int = 4, nbins: int = 9) -> np.ndarray:
    """Per-cell orientation histograms (n, ny, nx, nbins) with linear bin interpolation."""
    n, h, w = X.shape
    ny, nx = h // cell, w // cell
    mag, ang = image_gradients(X)
    mag, ang = mag[:, : ny * cell, : nx * cell], ang[:, : ny * cell, : nx * cell]
    binw = np.pi / nbins
    pos = ang / binw - 0.5
    b0f = np.floor(pos)
    frac = pos - b0f
    b0 = b0f.astype(int) % nbins
    b1 = (b0 + 1) % nbins
    cy = (np.arange(ny * cell) // cell)[None, :, None]
    cx = (np.arange(nx * cell) // cell)[None, None, :]
    nidx = np.arange(n)[:, None, None]
    base = (((nidx * ny + cy) * nx + cx) * nbins)
    size = n * ny * nx * nbins
    hist = np.bincount((base + b0).ravel(), weights=(mag * (1 - frac)).ravel(), minlength=size)
    hist += np.bincount((base + b1).ravel(), weights=(mag * frac).ravel(), minlength=size)
    return hist.reshape(n, ny, nx, nbins)


def hog(X: np.ndarray, cell: int = 4, nbins: int = 9, block: int = 2, clip: float = 0.2) -> np.ndarray:
    """Histogram of Oriented Gradients with overlapping L2-Hys block normalisation."""
    cells = hog_cells(X, cell, nbins)
    n, ny, nx, _ = cells.shape
    by, bx = ny - block + 1, nx - block + 1
    if by < 1 or bx < 1:
        raise ValueError("image too small for the requested cell / block size")
    blocks = np.empty((n, by, bx, block * block * nbins))
    for i in range(by):
        for j in range(bx):
            blocks[:, i, j, :] = cells[:, i : i + block, j : j + block, :].reshape(n, -1)
    v = blocks / np.sqrt((blocks ** 2).sum(-1, keepdims=True) + 1e-6)
    v = np.minimum(v, clip)
    v = v / np.sqrt((v ** 2).sum(-1, keepdims=True) + 1e-6)
    return v.reshape(n, -1)


# --------------------------------------------------------------------------- LBP
def _lbp_lookup(P: int, mapping: str) -> Tuple[np.ndarray, int]:
    """Look-up table from raw P-bit code to histogram bin."""
    codes = np.arange(2 ** P)
    if mapping == "default":
        return codes, 2 ** P
    lut = np.zeros(2 ** P, dtype=int)
    mask = (1 << P) - 1
    for c in codes:
        rot = ((c << 1) | (c >> (P - 1))) & mask
        transitions = bin(c ^ rot).count("1")
        ones = bin(c).count("1")
        if mapping == "riu2":
            lut[c] = ones if transitions <= 2 else P + 1
    if mapping == "riu2":
        return lut, P + 2
    if mapping == "u2":
        uniform_codes = [c for c in codes if bin(c ^ (((c << 1) | (c >> (P - 1))) & mask)).count("1") <= 2]
        table = {c: i for i, c in enumerate(uniform_codes)}
        lut = np.array([table.get(c, len(uniform_codes)) for c in codes])
        return lut, len(uniform_codes) + 1
    raise ValueError("mapping must be 'default', 'u2' or 'riu2'")


def lbp_codes(X: np.ndarray, P: int = 8, R: float = 1.0) -> np.ndarray:
    """Raw circular LBP code of every pixel (bilinear sampling of the P neighbours)."""
    n, h, w = X.shape
    pad = int(np.ceil(R)) + 1
    Xp = np.pad(X, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    codes = np.zeros((n, h, w), dtype=np.int64)
    center = X
    for p in range(P):
        dy = -R * np.sin(2 * np.pi * p / P)
        dx = R * np.cos(2 * np.pi * p / P)
        oy, ox = int(np.floor(dy)), int(np.floor(dx))
        fy, fx = round(dy - oy, 9), round(dx - ox, 9)

        def sl(a, b):
            return Xp[:, pad + oy + a: pad + oy + a + h, pad + ox + b: pad + ox + b + w]

        sample = (
            (1 - fy) * (1 - fx) * sl(0, 0)
            + (1 - fy) * fx * sl(0, 1)
            + fy * (1 - fx) * sl(1, 0)
            + fy * fx * sl(1, 1)
        )
        codes |= (np.round(sample - center, 7) >= 0).astype(np.int64) << p
    return codes


def lbp(X: np.ndarray, P: int = 8, R: float = 1.0, mapping: str = "u2", grid: int = 2) -> np.ndarray:
    """Spatially-pooled LBP histogram: image split into grid x grid regions."""
    lut, nb = _lbp_lookup(P, mapping)
    mapped = lut[lbp_codes(X, P, R)]
    n, h, w = mapped.shape
    gh, gw = h // grid, w // grid
    feats = np.zeros((n, grid, grid, nb))
    for i in range(grid):
        for j in range(grid):
            region = mapped[:, i * gh:(i + 1) * gh, j * gw:(j + 1) * gw].reshape(n, -1)
            for k in range(n):
                feats[k, i, j] = np.bincount(region[k], minlength=nb)[:nb]
    feats /= feats.sum(-1, keepdims=True) + EPS
    return feats.reshape(n, -1)


# --------------------------------------------------------------------------- GLCM
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "entropy", "correlation"]


def glcm_features(X: np.ndarray, levels: int = 8, distance: int = 1) -> np.ndarray:
    """Haralick-style statistics of the symmetric, normalised co-occurrence matrix.

    Four directions (0, 45, 90, 135 degrees) are computed; for each property the
    mean and range over directions are returned (rotation-tolerant summary).
    """
    n, h, w = X.shape
    q = np.clip((X * levels).astype(int), 0, levels - 1)
    d = distance
    offsets = [(0, d), (-d, d), (-d, 0), (-d, -d)]
    ii, jj = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
    per_dir = []
    for dy, dx in offsets:
        y0, y1 = max(0, -dy), h - max(0, dy)
        x0, x1 = max(0, -dx), w - max(0, dx)
        a = q[:, y0:y1, x0:x1]
        b = q[:, y0 + dy: y1 + dy, x0 + dx: x1 + dx]
        idx = (np.arange(n)[:, None, None] * levels * levels + a * levels + b).ravel()
        G = np.bincount(idx, minlength=n * levels * levels).reshape(n, levels, levels).astype(float)
        G = G + G.transpose(0, 2, 1)
        G /= G.sum(axis=(1, 2), keepdims=True) + EPS
        diff = (ii - jj).astype(float)
        contrast = (G * diff ** 2).sum((1, 2))
        dissim = (G * np.abs(diff)).sum((1, 2))
        homog = (G / (1 + diff ** 2)).sum((1, 2))
        energy = np.sqrt((G ** 2).sum((1, 2)))
        entropy = -(G * np.log2(G + EPS)).sum((1, 2))
        mu_i = (G * ii).sum((1, 2))
        mu_j = (G * jj).sum((1, 2))
        s_i = np.sqrt((G * (ii - mu_i[:, None, None]) ** 2).sum((1, 2)))
        s_j = np.sqrt((G * (jj - mu_j[:, None, None]) ** 2).sum((1, 2)))
        cov = (G * (ii - mu_i[:, None, None]) * (jj - mu_j[:, None, None])).sum((1, 2))
        corr = np.where(s_i * s_j > 1e-9, cov / (s_i * s_j + EPS), 1.0)
        per_dir.append(np.stack([contrast, dissim, homog, energy, entropy, corr], axis=1))
    P = np.stack(per_dir, axis=0)
    return np.concatenate([P.mean(0), P.max(0) - P.min(0)], axis=1)


# --------------------------------------------------------------------------- Hu moments
def hu_moments(X: np.ndarray) -> np.ndarray:
    """Seven Hu invariants (translation / scale / rotation invariant), log-scaled."""
    n, h, w = X.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    out = np.zeros((n, 7))
    for k in range(n):
        im = X[k]
        m00 = im.sum()
        if m00 < EPS:
            continue
        cx = (im * xx).sum() / m00
        cy = (im * yy).sum() / m00

        def eta(p, q):
            mu = (im * (xx - cx) ** p * (yy - cy) ** q).sum()
            return mu / m00 ** (1 + (p + q) / 2.0)

        n20, n02, n11 = eta(2, 0), eta(0, 2), eta(1, 1)
        n30, n03, n21, n12 = eta(3, 0), eta(0, 3), eta(2, 1), eta(1, 2)
        h1 = n20 + n02
        h2 = (n20 - n02) ** 2 + 4 * n11 ** 2
        h3 = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
        h4 = (n30 + n12) ** 2 + (n21 + n03) ** 2
        h5 = (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) + (
            3 * n21 - n03
        ) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
        h6 = (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + 4 * n11 * (n30 + n12) * (n21 + n03)
        h7 = (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) - (
            n30 - 3 * n12
        ) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
        out[k] = [h1, h2, h3, h4, h5, h6, h7]
    # Invariants that are analytically zero (h7 for mirror-symmetric shapes, ...) come out of the
    # floating-point arithmetic as round-off noise (~1e-23).  Without this guard the log transform
    # blows that noise up to a spurious value of about +/-20 and breaks translation invariance.
    out[np.abs(out) < 1e-18] = 0.0
    return -np.sign(out) * np.log10(np.abs(out) + 1e-30)


# --------------------------------------------------------------------------- registry
def _names(prefix: str, d: int) -> List[str]:
    return [f"{prefix}_{i}" for i in range(d)]


IMAGE_EXTRACTORS: Dict[str, callable] = {
    "raw": raw_pixels,
    "zoning": zoning,
    "profiles": projection_profiles,
    "stats": intensity_stats,
    "hog": hog,
    "lbp": lbp,
    "glcm": glcm_features,
    "hu": hu_moments,
}


def extract_image_features(X: np.ndarray, groups: List[str]) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Concatenate several descriptor groups.

    Returns ``(F, feature_names, group_index)`` where ``group_index[j]`` is the
    index (into ``groups``) of the descriptor that produced column ``j``.
    """
    blocks, names, gidx = [], [], []
    for gi, g in enumerate(groups):
        if g not in IMAGE_EXTRACTORS:
            raise KeyError(f"unknown image descriptor '{g}'. options: {sorted(IMAGE_EXTRACTORS)}")
        F = IMAGE_EXTRACTORS[g](X)
        blocks.append(F)
        if g == "glcm":
            nm = [f"glcm_{p}_{s}" for s in ("mean", "range") for p in GLCM_PROPS]
        else:
            nm = _names(g, F.shape[1])
        names += nm
        gidx += [gi] * F.shape[1]
    return np.hstack(blocks), names, np.asarray(gidx)
