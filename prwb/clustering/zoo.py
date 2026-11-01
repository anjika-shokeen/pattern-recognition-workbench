"""Clustering model registry and model-selection helpers (number of clusters)."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from .scratch import FuzzyCMeans, GMMScratch, KMeansScratch


def get_clusterers(k: int, seed: int = 0, eps: float | None = None) -> Dict[str, object]:
    algos = {
        "KMeans (scratch)": KMeansScratch(k, n_init=5, random_state=seed),
        "KMeans (sklearn)": KMeans(k, n_init=5, random_state=seed),
        "GMM-full (scratch)": GMMScratch(k, "full", random_state=seed),
        "GMM-diag (scratch)": GMMScratch(k, "diag", random_state=seed),
        "Fuzzy C-Means": FuzzyCMeans(k, m=1.2, random_state=seed),
        "Agglomerative-Ward": AgglomerativeClustering(k, linkage="ward"),
        "Agglomerative-Average": AgglomerativeClustering(k, linkage="average"),
        "Spectral": SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=12, random_state=seed,
                                       assign_labels="kmeans"),
        "HDBSCAN": HDBSCAN(min_cluster_size=15),
    }
    if eps is not None:
        algos["DBSCAN"] = DBSCAN(eps=eps, min_samples=5)
    return algos


def estimate_eps(X: np.ndarray, k: int = 5, quantile: float = 0.9) -> float:
    """k-distance heuristic for DBSCAN: a high quantile of each point's distance to its k-th neighbour."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    d, _ = nn.kneighbors(X)
    return float(np.quantile(d[:, -1], quantile))


def sweep_k(X: np.ndarray, ks, seed: int = 0, covariance_type: str = "diag") -> dict:
    """Internal criteria versus k: inertia (elbow), silhouette, GMM BIC / AIC."""
    inertia, sil, bic, aic = [], [], [], []
    for k in ks:
        km = KMeansScratch(k, n_init=3, random_state=seed).fit(X)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X, km.labels_) if k > 1 else np.nan)
        g = GMMScratch(k, covariance_type, n_init=1, random_state=seed).fit(X)
        bic.append(g.bic(X))
        aic.append(g.aic(X))
    return {"k": list(ks), "inertia": inertia, "silhouette": sil, "bic": bic, "aic": aic}


def gap_statistic(X: np.ndarray, ks, n_refs: int = 8, seed: int = 0) -> dict:
    """Tibshirani et al. (2001) gap statistic with uniform reference sets in the data bounding box."""
    rng = np.random.default_rng(seed)
    lo, hi = X.min(0), X.max(0)
    gaps, sk = [], []
    for k in ks:
        wk = np.log(KMeansScratch(k, n_init=2, random_state=seed).fit(X).inertia_)
        ref = []
        for _ in range(n_refs):
            Xr = rng.uniform(lo, hi, size=X.shape)
            ref.append(np.log(KMeansScratch(k, n_init=1, random_state=seed).fit(Xr).inertia_))
        gaps.append(np.mean(ref) - wk)
        sk.append(np.std(ref) * np.sqrt(1 + 1 / n_refs))
    return {"k": list(ks), "gap": gaps, "s": sk}
