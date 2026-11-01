"""High-level facade used by the CLI and the dashboard.

Everything here works on a single (dataset, feature-set) pair; the long-running
multi-dataset protocol lives in :mod:`prwb.experiments`.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline

from . import datasets as D
from .classifiers import get_classifiers
from .clustering import estimate_eps, get_clusterers
from .evaluation.metrics import clustering_scores, cross_validate_model
from .features import PCAScratch, Standardizer, build_features
from .features.builder import FEATURE_SETS
from .stats import anova_table, class_separability, correlation_summary, normality_tests


def load_config(path: Optional[str] = None) -> dict:
    path = path or os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def quick_classify(dataset: str, feature_set: str, classifier: str, n_splits: int = 5, n_repeats: int = 1,
                   seed: int = 0, subsample: Optional[int] = None) -> dict:
    ds = D.load_dataset(dataset)
    if subsample:
        ds = ds.subsample(subsample, seed)
    fm = build_features(ds, feature_set)
    clf = get_classifiers(seed, names=[classifier])[classifier]
    res = cross_validate_model(clf, fm.X, ds.y, n_splits, n_repeats, seed)
    res.update(dataset=ds, features=fm)
    return res


def quick_cluster(dataset: str, feature_set: str, algorithm: str, k: Optional[int] = None, pca: int = 30,
                  seed: int = 0, subsample: Optional[int] = None) -> dict:
    ds = D.load_dataset(dataset)
    if subsample:
        ds = ds.subsample(subsample, seed)
    fm = build_features(ds, feature_set)
    Z = Standardizer().fit_transform(fm.X)
    if pca and Z.shape[1] > pca:
        Z = PCAScratch(pca).fit_transform(Z)
    k = k or ds.n_classes
    algos = get_clusterers(k, seed, eps=estimate_eps(Z))
    if algorithm not in algos:
        raise KeyError(f"unknown algorithm '{algorithm}'. options: {list(algos)}")
    labels = algos[algorithm].fit_predict(Z)
    return {"dataset": ds, "Z": Z, "labels": labels, "scores": clustering_scores(Z, ds.y, labels)}


def quick_stats(dataset: str, feature_set: str, top: int = 15, subsample: Optional[int] = None) -> dict:
    ds = D.load_dataset(dataset)
    if subsample:
        ds = ds.subsample(subsample)
    fm = build_features(ds, feature_set)
    an = anova_table(fm.X, ds.y, fm.names)
    return {"anova": an.head(top), "normality_fraction": float(normality_tests(fm.X[:, :min(60, fm.X.shape[1])], fm.names[:60])["normal_at_5pct"].mean()),
            "correlation": {k: v for k, v in correlation_summary(fm.X, fm.names).items() if k not in ("corr", "kept_index")},
            "separability": class_separability(PCAScratch(20).fit_transform(Standardizer().fit_transform(fm.X)), ds.y),
            "features": fm}


def embed_2d(Z: np.ndarray, method: str = "pca", seed: int = 0) -> np.ndarray:
    if method == "pca":
        return PCAScratch(2).fit_transform(Z)
    if method == "tsne":
        return TSNE(2, init="pca", perplexity=30, random_state=seed).fit_transform(Z)
    raise ValueError(method)


def train_final_model(dataset: str = "digits", feature_set: str = "hog+lbp", classifier: str = "SVM-RBF",
                      seed: int = 0) -> Pipeline:
    """Fit a deployable model on the *whole* dataset (for the `predict` command)."""
    ds = D.load_dataset(dataset)
    fm = build_features(ds, feature_set)
    model = get_classifiers(seed, names=[classifier])[classifier]
    model.fit(fm.X, ds.y)
    return model
