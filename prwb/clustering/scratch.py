"""Clustering algorithms implemented from scratch: k-means++, EM for Gaussian mixtures, fuzzy c-means."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array


def _sqdist(A, B):
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T, 0)


# ----------------------------------------------------------------------------
class KMeansScratch(ClusterMixin, BaseEstimator):
    """Lloyd's algorithm with k-means++ seeding (Arthur & Vassilvitskii, 2007) and restarts."""

    def __init__(self, n_clusters: int = 8, init: str = "kmeans++", n_init: int = 5, max_iter: int = 200,
                 tol: float = 1e-5, random_state: int = 0):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _seed(self, X, rng):
        n = len(X)
        k = self.n_clusters
        if self.init == "random":
            return X[rng.choice(n, k, replace=False)].copy()
        C = [X[rng.integers(n)]]
        d2 = _sqdist(X, np.array(C))[:, 0]
        for _ in range(1, k):
            probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
            C.append(X[rng.choice(n, p=probs)])
            d2 = np.minimum(d2, _sqdist(X, C[-1][None])[:, 0])
        return np.array(C)

    def _run(self, X, rng):
        C = self._seed(X, rng)
        prev = np.inf
        for it in range(1, self.max_iter + 1):
            D = _sqdist(X, C)
            lab = D.argmin(1)
            inertia = D[np.arange(len(X)), lab].sum()
            for j in range(self.n_clusters):
                m = lab == j
                if m.any():
                    C[j] = X[m].mean(0)
                else:  # re-seed an empty cluster at the point farthest from its centroid
                    C[j] = X[D[np.arange(len(X)), lab].argmax()]
            if abs(prev - inertia) <= self.tol * max(inertia, 1e-12):
                break
            prev = inertia
        D = _sqdist(X, C)
        lab = D.argmin(1)
        return C, lab, D[np.arange(len(X)), lab].sum(), it

    def fit(self, X, y=None):
        X = check_array(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        best = None
        self.inertia_history_ = []
        for _ in range(self.n_init):
            r = self._run(X, rng)
            self.inertia_history_.append(r[2])
            if best is None or r[2] < best[2]:
                best = r
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def predict(self, X):
        return _sqdist(check_array(X, dtype=float), self.cluster_centers_).argmin(1)


# ----------------------------------------------------------------------------
class GMMScratch(ClusterMixin, BaseEstimator):
    """Gaussian mixture model fitted by Expectation-Maximisation.

    covariance_type: 'full' | 'diag' | 'spherical'.  Initialised from k-means.
    """

    def __init__(self, n_components: int = 3, covariance_type: str = "full", max_iter: int = 200,
                 tol: float = 1e-4, reg_covar: float = 1e-4, n_init: int = 2, random_state: int = 0):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.random_state = random_state

    # log N(x | mu_k, Sigma_k) for all k -> (n, K)
    def _log_gauss(self, X):
        n, d = X.shape
        K = self.n_components
        out = np.empty((n, K))
        for k in range(K):
            diff = X - self.means_[k]
            if self.covariance_type == "full":
                L = np.linalg.cholesky(self.covariances_[k])
                z = np.linalg.solve(L, diff.T)
                maha = (z ** 2).sum(0)
                logdet = 2 * np.log(np.diag(L)).sum()
            elif self.covariance_type == "diag":
                v = self.covariances_[k]
                maha = (diff ** 2 / v).sum(1)
                logdet = np.log(v).sum()
            else:
                v = self.covariances_[k]
                maha = (diff ** 2).sum(1) / v
                logdet = d * np.log(v)
            out[:, k] = -0.5 * (d * np.log(2 * np.pi) + logdet + maha)
        return out

    def _estep(self, X):
        lp = self._log_gauss(X) + np.log(self.weights_)
        ll = logsumexp(lp, axis=1)
        return np.exp(lp - ll[:, None]), ll.mean()

    def _mstep(self, X, R):
        n, d = X.shape
        Nk = R.sum(0) + 1e-10
        self.weights_ = Nk / n
        self.means_ = (R.T @ X) / Nk[:, None]
        covs = []
        for k in range(self.n_components):
            diff = X - self.means_[k]
            if self.covariance_type == "full":
                S = (R[:, k, None] * diff).T @ diff / Nk[k] + self.reg_covar * np.eye(d)
            elif self.covariance_type == "diag":
                S = (R[:, k, None] * diff ** 2).sum(0) / Nk[k] + self.reg_covar
            else:
                S = (R[:, k] * (diff ** 2).sum(1)).sum() / (Nk[k] * d) + self.reg_covar
            covs.append(S)
        self.covariances_ = covs

    def _fit_once(self, X, seed):
        km = KMeansScratch(self.n_components, n_init=1, random_state=seed).fit(X)
        R = np.eye(self.n_components)[km.labels_]
        self._mstep(X, R)
        prev, hist = -np.inf, []
        for it in range(1, self.max_iter + 1):
            R, ll = self._estep(X)
            self._mstep(X, R)
            hist.append(ll)
            if abs(ll - prev) < self.tol:
                break
            prev = ll
        return ll, hist, it

    def fit(self, X, y=None):
        X = check_array(X, dtype=float)
        best = None
        for i in range(self.n_init):
            ll, hist, it = self._fit_once(X, self.random_state + i)
            if best is None or ll > best[0]:
                best = (ll, hist, it, self.weights_.copy(), self.means_.copy(), [c.copy() for c in self.covariances_])
        ll, self.loglik_history_, self.n_iter_, self.weights_, self.means_, self.covariances_ = best
        self.lower_bound_ = ll
        self.labels_ = self.predict(X)
        return self

    def predict_proba(self, X):
        return self._estep(check_array(X, dtype=float))[0]

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def score(self, X, y=None):
        return float(self._estep(check_array(X, dtype=float))[1])

    def _n_params(self, d):
        K = self.n_components
        cov = {"full": K * d * (d + 1) / 2, "diag": K * d, "spherical": K}[self.covariance_type]
        return cov + K * d + K - 1

    def bic(self, X):
        X = check_array(X, dtype=float)
        return float(-2 * self.score(X) * len(X) + self._n_params(X.shape[1]) * np.log(len(X)))

    def aic(self, X):
        X = check_array(X, dtype=float)
        return float(-2 * self.score(X) * len(X) + 2 * self._n_params(X.shape[1]))


# ----------------------------------------------------------------------------
class FuzzyCMeans(ClusterMixin, BaseEstimator):
    """Fuzzy c-means (Bezdek, 1981) with fuzzifier m > 1.

    Note: in high dimensions m = 2 makes all memberships collapse towards 1/c; a fuzzifier
    close to 1 (default 1.2) is recommended (see the fuzzifier sweep in the report).
    """

    def __init__(self, n_clusters: int = 3, m: float = 1.2, max_iter: int = 200, tol: float = 1e-5,
                 random_state: int = 0):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        U = rng.random((len(X), self.n_clusters))
        U /= U.sum(1, keepdims=True)
        self.objective_history_ = []
        for it in range(1, self.max_iter + 1):
            Um = U ** self.m
            C = (Um.T @ X) / Um.sum(0)[:, None]
            D2 = _sqdist(X, C) + 1e-12
            inv = D2 ** (-1.0 / (self.m - 1))
            Unew = inv / inv.sum(1, keepdims=True)
            self.objective_history_.append(float((Unew ** self.m * D2).sum()))
            if np.abs(Unew - U).max() < self.tol:
                U = Unew
                break
            U = Unew
        self.cluster_centers_, self.membership_, self.n_iter_ = C, U, it
        self.labels_ = U.argmax(1)
        self.partition_coefficient_ = float((U ** 2).sum() / len(X))
        return self

    def predict(self, X):
        D2 = _sqdist(check_array(X, dtype=float), self.cluster_centers_) + 1e-12
        inv = D2 ** (-1.0 / (self.m - 1))
        return (inv / inv.sum(1, keepdims=True)).argmax(1)
