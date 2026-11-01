"""Feature transforms implemented from scratch: standardisation, PCA, Fisher LDA, whitening."""
from __future__ import annotations

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin


class Standardizer(BaseEstimator, TransformerMixin):
    """Zero-mean, unit-variance scaling (constant columns are left centred)."""

    def __init__(self, with_std: bool = True):
        self.with_std = with_std

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.mean_ = X.mean(0)
        sd = X.std(0)
        sd[sd < 1e-12] = 1.0
        self.scale_ = sd if self.with_std else np.ones_like(sd)
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mean_) / self.scale_

    def inverse_transform(self, Z):
        return np.asarray(Z) * self.scale_ + self.mean_


class PCAScratch(BaseEstimator, TransformerMixin):
    """Principal Component Analysis via thin SVD of the centred data matrix.

    ``n_components`` may be an int (number of axes) or a float in (0, 1)
    (smallest number of axes explaining that fraction of the variance).
    """

    def __init__(self, n_components=0.95, whiten: bool = False):
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        n = X.shape[0]
        self.mean_ = X.mean(0)
        U, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        var = S ** 2 / max(n - 1, 1)
        ratio = var / var.sum()
        if isinstance(self.n_components, float) and 0 < self.n_components < 1:
            k = int(np.searchsorted(np.cumsum(ratio), self.n_components) + 1)
        else:
            k = int(self.n_components)
        k = max(1, min(k, len(S)))
        self.components_ = Vt[:k]
        self.explained_variance_ = var[:k]
        self.explained_variance_ratio_ = ratio[:k]
        self.all_explained_variance_ratio_ = ratio
        self.n_components_ = k
        return self

    def transform(self, X):
        Z = (np.asarray(X, float) - self.mean_) @ self.components_.T
        if self.whiten:
            Z = Z / np.sqrt(self.explained_variance_ + 1e-12)
        return Z

    def inverse_transform(self, Z):
        Z = np.asarray(Z, float)
        if self.whiten:
            Z = Z * np.sqrt(self.explained_variance_ + 1e-12)
        return Z @ self.components_ + self.mean_

    def reconstruction_error(self, X):
        X = np.asarray(X, float)
        return float(np.mean((X - self.inverse_transform(self.transform(X))) ** 2))


class LDAProjection(BaseEstimator, TransformerMixin):
    """Multi-class Fisher discriminant analysis (supervised dimensionality reduction).

    Solves the generalised eigenproblem  S_b v = lambda S_w v  and keeps the
    ``min(C-1, n_components)`` leading eigenvectors.
    """

    def __init__(self, n_components=None, shrinkage: float = 1e-3):
        self.n_components = n_components
        self.shrinkage = shrinkage

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        classes = np.unique(y)
        d = X.shape[1]
        self.mean_ = X.mean(0)
        Sw = np.zeros((d, d))
        Sb = np.zeros((d, d))
        for c in classes:
            Xc = X[y == c]
            mc = Xc.mean(0)
            Sw += (Xc - mc).T @ (Xc - mc)
            diff = (mc - self.mean_)[:, None]
            Sb += len(Xc) * diff @ diff.T
        Sw /= len(X)
        Sb /= len(X)
        Sw = (1 - self.shrinkage) * Sw + self.shrinkage * np.trace(Sw) / d * np.eye(d)
        vals, vecs = linalg.eigh(Sb, Sw)
        order = np.argsort(vals)[::-1]
        k = min(len(classes) - 1, self.n_components or len(classes) - 1)
        self.scalings_ = vecs[:, order[:k]]
        self.eigenvalues_ = vals[order[:k]]
        self.explained_ratio_ = vals[order] / max(vals.sum(), 1e-12)
        self.n_components_ = k
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mean_) @ self.scalings_
