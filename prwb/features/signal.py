"""1-D signal feature extraction: time-domain, Hjorth, spectral and autocorrelation groups."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy import stats as sstats

EPS = 1e-12


def time_domain(X: np.ndarray, fs: float = 1.0):
    mean = X.mean(1)
    std = X.std(1)
    rms = np.sqrt((X ** 2).mean(1))
    ptp = X.max(1) - X.min(1)
    crest = np.abs(X).max(1) / (rms + EPS)
    xc = X - mean[:, None]
    zcr = (np.diff(np.signbit(xc), axis=1) != 0).mean(1)
    mad = np.abs(np.diff(X, axis=1)).mean(1)
    skew = sstats.skew(X, axis=1)
    kurt = sstats.kurtosis(X, axis=1)
    F = np.column_stack([mean, std, rms, ptp, crest, zcr, mad, skew, kurt])
    return F, [
        "mean", "std", "rms", "peak_to_peak", "crest_factor",
        "zero_cross_rate", "mean_abs_diff", "skewness", "kurtosis",
    ]


def hjorth(X: np.ndarray, fs: float = 1.0):
    """Hjorth (1970) activity, mobility and complexity."""
    d1 = np.diff(X, axis=1)
    d2 = np.diff(d1, axis=1)
    v0, v1, v2 = X.var(1), d1.var(1), d2.var(1)
    mobility = np.sqrt(v1 / (v0 + EPS))
    complexity = np.sqrt(v2 / (v1 + EPS)) / (mobility + EPS)
    return np.column_stack([v0, mobility, complexity]), [
        "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
    ]


def spectral(X: np.ndarray, fs: float = 1.0, n_bands: int = 8):
    n, L = X.shape
    Xc = X - X.mean(1, keepdims=True)
    win = np.hanning(L)
    spec = np.abs(np.fft.rfft(Xc * win, axis=1)) ** 2
    freqs = np.fft.rfftfreq(L, 1.0 / fs)
    total = spec.sum(1) + EPS
    p = spec / total[:, None]
    centroid = (p * freqs).sum(1)
    spread = np.sqrt((p * (freqs - centroid[:, None]) ** 2).sum(1))
    entropy = -(p * np.log2(p + EPS)).sum(1) / np.log2(p.shape[1])
    flatness = np.exp(np.log(spec + EPS).mean(1)) / (spec.mean(1) + EPS)
    dom = freqs[np.argmax(spec, axis=1)]
    cum = np.cumsum(p, axis=1)
    rolloff = freqs[np.argmax(cum >= 0.85, axis=1)]
    k0 = np.argmax(spec, axis=1)
    S = spec.shape[1]
    h3 = np.array([spec[i, min(3 * k0[i] - 1, S - 1): min(3 * k0[i] + 2, S)].sum() for i in range(n)]) / total
    h2 = np.array([spec[i, min(2 * k0[i] - 1, S - 1): min(2 * k0[i] + 2, S)].sum() for i in range(n)]) / total
    edges = np.linspace(0, freqs[-1] + 1e-9, n_bands + 1)
    bands = np.stack(
        [p[:, (freqs >= edges[b]) & (freqs < edges[b + 1])].sum(1) for b in range(n_bands)], axis=1
    )
    F = np.column_stack([centroid, spread, entropy, flatness, dom, rolloff, h2, h3, bands])
    names = [
        "spec_centroid", "spec_spread", "spec_entropy", "spec_flatness",
        "dominant_freq", "spec_rolloff85", "harmonic2_ratio", "harmonic3_ratio",
    ] + [f"band_{b}" for b in range(n_bands)]
    return F, names


def autocorr(X: np.ndarray, fs: float = 1.0):
    n, L = X.shape
    Xc = X - X.mean(1, keepdims=True)
    f = np.fft.rfft(Xc, n=2 * L, axis=1)
    ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :L]
    ac /= ac[:, :1] + EPS
    lag_pk = np.zeros(n)
    val_pk = np.zeros(n)
    for i in range(n):
        a = ac[i, : L // 2]
        neg = np.flatnonzero(a < 0)
        start = neg[0] if neg.size else 1
        seg = a[start:]
        if seg.size > 2:
            j = np.argmax(seg)
            lag_pk[i], val_pk[i] = (start + j) / fs, seg[j]
    decay = ac[:, : L // 4].mean(1)
    env = np.abs(X)
    half = L // 2
    env_ratio = env[:, half:].mean(1) / (env[:, :half].mean(1) + EPS)
    env_var = np.sqrt(env.reshape(n, 8, -1).mean(2).var(1)) / (env.mean(1) + EPS)
    F = np.column_stack([lag_pk, val_pk, decay, env_ratio, env_var])
    return F, [
        "acf_peak_lag", "acf_peak_value", "acf_mean_decay",
        "envelope_late_early", "envelope_variation",
    ]


def raw_signal(X: np.ndarray, fs: float = 1.0):
    return X.copy(), [f"x_{i}" for i in range(X.shape[1])]


SIGNAL_EXTRACTORS = {
    "raw": raw_signal,
    "time": time_domain,
    "hjorth": hjorth,
    "spectral": spectral,
    "autocorr": autocorr,
}


def extract_signal_features(
    X: np.ndarray, groups: List[str], fs: float = 128.0
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    blocks, names, gidx = [], [], []
    for gi, g in enumerate(groups):
        if g not in SIGNAL_EXTRACTORS:
            raise KeyError(f"unknown signal descriptor '{g}'. options: {sorted(SIGNAL_EXTRACTORS)}")
        F, nm = SIGNAL_EXTRACTORS[g](X, fs)
        blocks.append(F)
        names += nm
        gidx += [gi] * F.shape[1]
    return np.hstack(blocks), names, np.asarray(gidx)
