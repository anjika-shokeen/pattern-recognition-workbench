"""Filter-type feature selection: Fisher score, mutual information, mRMR, correlation pruning."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif


def fisher_scores(X, y) -> np.ndarray:
    """Fisher criterion  F_j = sum_c n_c (mu_cj - mu_j)^2 / sum_c n_c var_cj."""
    X = np.asarray(X, float)
    y = np.asarray(y)
    mu = X.mean(0)
    num = np.zeros(X.shape[1])
    den = np.zeros(X.shape[1])
    for c in np.unique(y):
        Xc = X[y == c]
        num += len(Xc) * (Xc.mean(0) - mu) ** 2
        den += len(Xc) * Xc.var(0)
    return num / (den + 1e-12)


class _TopK(BaseEstimator, TransformerMixin):
    """Base class: subclasses implement ``_rank`` returning an ordered index array."""

    def __init__(self, k: int = 20):
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.ranking_ = self._rank(X, np.asarray(y))
        self.support_ = np.sort(self.ranking_[: min(self.k, X.shape[1])])
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]


class FisherSelector(_TopK):
    def _rank(self, X, y):
        s = fisher_scores(X, y)
        self.scores_ = s
        return np.argsort(-s)


class MISelector(_TopK):
    def __init__(self, k: int = 20, random_state: int = 0):
        super().__init__(k)
        self.random_state = random_state

    def _rank(self, X, y):
        s = mutual_info_classif(X, y, random_state=self.random_state)
        self.scores_ = s
        return np.argsort(-s)


class MRMRSelector(_TopK):
    """Minimum-redundancy / maximum-relevance (Peng et al., 2005), MID scheme.

    Relevance = mutual information with the class, redundancy = mean absolute
    Pearson correlation with the features already chosen (cheap surrogate of MI).
    """

    def __init__(self, k: int = 20, random_state: int = 0):
        super().__init__(k)
        self.random_state = random_state

    def _rank(self, X, y):
        rel = mutual_info_classif(X, y, random_state=self.random_state)
        self.scores_ = rel
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
        C = np.abs(np.corrcoef(Xs, rowvar=False))
        C = np.nan_to_num(C)
        selected = [int(np.argmax(rel))]
        remaining = set(range(X.shape[1])) - set(selected)
        while remaining and len(selected) < min(self.k, X.shape[1]):
            rem = np.array(sorted(remaining))
            red = C[np.ix_(rem, selected)].mean(1)
            score = rel[rem] - red * rel.mean() * 2.0
            j = int(rem[np.argmax(score)])
            selected.append(j)
            remaining.remove(j)
        return np.array(selected + sorted(remaining))


class CorrelationPruner(BaseEstimator, TransformerMixin):
    """Drop one feature of every pair whose |Pearson r| exceeds ``threshold``."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        sd = X.std(0)
        keep = np.flatnonzero(sd > 1e-12)
        C = np.abs(np.nan_to_num(np.corrcoef(X[:, keep], rowvar=False)))
        drop = set()
        for i in range(len(keep)):
            if i in drop:
                continue
            for j in range(i + 1, len(keep)):
                if j not in drop and C[i, j] > self.threshold:
                    drop.add(j)
        self.support_ = np.array([keep[i] for i in range(len(keep)) if i not in drop])
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]
