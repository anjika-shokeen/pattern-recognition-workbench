"""Classifiers implemented from scratch (NumPy only) with a scikit-learn compatible API.

KNNClassifier         k-nearest neighbours (Euclidean / Manhattan / cosine / Mahalanobis)
GaussianBayes         Bayes classifier with Gaussian class-conditionals (QDA / LDA / naive)
MinimumDistance       nearest class mean (optionally Mahalanobis)
ParzenClassifier      kernel-density (Parzen window) Bayes classifier
SoftmaxRegression     multinomial logistic regression trained with Adam
NumpyMLP              multilayer perceptron with back-propagation, dropout, early stopping
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array


def _encode(y):
    classes, yi = np.unique(np.asarray(y), return_inverse=True)
    return classes, yi


def _softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


# ----------------------------------------------------------------------------
# k-NN
# ----------------------------------------------------------------------------
class KNNClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, k: int = 5, metric: str = "euclidean", weights: str = "uniform", chunk: int = 512):
        self.k = k
        self.metric = metric
        self.weights = weights
        self.chunk = chunk

    def fit(self, X, y):
        X = check_array(X, dtype=float)
        self.classes_, self._y = _encode(y)
        if self.metric == "mahalanobis":
            S = np.cov(X, rowvar=False) + 1e-3 * np.eye(X.shape[1])
            w, V = np.linalg.eigh(S)
            self._W = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-9))) @ V.T
            X = X @ self._W
        self._X = X
        return self

    def _prep(self, X):
        X = check_array(X, dtype=float)
        return X @ self._W if self.metric == "mahalanobis" else X

    def _dist(self, A, B):
        if self.metric in ("euclidean", "mahalanobis"):
            d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
            return np.sqrt(np.maximum(d2, 0))
        if self.metric == "manhattan":
            return np.abs(A[:, None, :] - B[None, :, :]).sum(-1)
        if self.metric == "cosine":
            An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
            Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
            return 1 - An @ Bn.T
        raise ValueError(f"unknown metric {self.metric}")

    def kneighbors(self, X):
        Xp = self._prep(X)
        k = min(self.k, len(self._X))
        idxs, dists = [], []
        chunk = self.chunk if self.metric != "manhattan" else 32
        for s in range(0, len(Xp), chunk):
            D = self._dist(Xp[s:s + chunk], self._X)
            part = np.argpartition(D, k - 1, axis=1)[:, :k]
            dd = np.take_along_axis(D, part, 1)
            o = np.argsort(dd, axis=1)
            idxs.append(np.take_along_axis(part, o, 1))
            dists.append(np.take_along_axis(dd, o, 1))
        return np.vstack(dists), np.vstack(idxs)

    def predict_proba(self, X):
        d, idx = self.kneighbors(X)
        lab = self._y[idx]
        w = 1.0 / (d + 1e-9) if self.weights == "distance" else np.ones_like(d)
        P = np.zeros((len(lab), len(self.classes_)))
        for c in range(len(self.classes_)):
            P[:, c] = (w * (lab == c)).sum(1)
        return P / P.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


# ----------------------------------------------------------------------------
# Gaussian Bayes
# ----------------------------------------------------------------------------
class GaussianBayes(ClassifierMixin, BaseEstimator):
    """Bayes decision rule  argmax_c  log p(x|c) + log P(c)  with Gaussian p(x|c).

    mode = "qda"   : full covariance per class (quadratic boundaries)
    mode = "lda"   : pooled covariance (linear boundaries)
    mode = "naive" : diagonal covariance (feature independence)
    ``reg`` shrinks covariances towards a scaled identity (Ledoit-Wolf-like).
    """

    def __init__(self, mode: str = "qda", reg: float = 0.1, priors=None):
        self.mode = mode
        self.reg = reg
        self.priors = priors

    def _shrink(self, S):
        d = S.shape[0]
        S = (1 - self.reg) * S + self.reg * np.trace(S) / d * np.eye(d)
        return S + 1e-6 * np.eye(d)

    def fit(self, X, y):
        X = check_array(X, dtype=float)
        self.classes_, yi = _encode(y)
        C, d = len(self.classes_), X.shape[1]
        self.means_ = np.stack([X[yi == c].mean(0) for c in range(C)])
        counts = np.bincount(yi, minlength=C)
        self.log_priors_ = np.log(np.asarray(self.priors) if self.priors is not None else counts / counts.sum())
        covs = []
        if self.mode == "naive":
            for c in range(C):
                v = X[yi == c].var(0) + 1e-6 + self.reg * X.var(0)
                covs.append(np.diag(v))
        elif self.mode == "lda":
            Sp = sum((X[yi == c] - self.means_[c]).T @ (X[yi == c] - self.means_[c]) for c in range(C)) / (len(X) - C)
            Sp = self._shrink(Sp)
            covs = [Sp] * C
        elif self.mode == "qda":
            for c in range(C):
                Xc = X[yi == c]
                S = np.cov(Xc, rowvar=False) if len(Xc) > 1 else np.eye(d)
                covs.append(self._shrink(np.atleast_2d(S)))
        else:
            raise ValueError("mode must be 'qda', 'lda' or 'naive'")
        self.covs_ = covs
        self._chol = [np.linalg.cholesky(S) for S in covs]
        self._logdet = np.array([2 * np.log(np.diag(L)).sum() for L in self._chol])
        return self

    def decision_log_posterior(self, X):
        X = check_array(X, dtype=float)
        d = X.shape[1]
        out = np.empty((len(X), len(self.classes_)))
        for c in range(len(self.classes_)):
            z = np.linalg.solve(self._chol[c], (X - self.means_[c]).T)
            maha = (z ** 2).sum(0)
            out[:, c] = -0.5 * (maha + self._logdet[c] + d * np.log(2 * np.pi)) + self.log_priors_[c]
        return out

    def predict_proba(self, X):
        L = self.decision_log_posterior(X)
        return np.exp(L - logsumexp(L, axis=1, keepdims=True))

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_log_posterior(X), 1)]


class MinimumDistance(ClassifierMixin, BaseEstimator):
    """Nearest class-mean rule (template matching)."""

    def __init__(self, mahalanobis: bool = False):
        self.mahalanobis = mahalanobis

    def fit(self, X, y):
        X = check_array(X, dtype=float)
        self.classes_, yi = _encode(y)
        self.means_ = np.stack([X[yi == c].mean(0) for c in range(len(self.classes_))])
        if self.mahalanobis:
            Sp = sum((X[yi == c] - self.means_[c]).T @ (X[yi == c] - self.means_[c])
                     for c in range(len(self.classes_))) / (len(X) - len(self.classes_))
            Sp += 1e-3 * np.trace(Sp) / Sp.shape[0] * np.eye(Sp.shape[0])
            self.Sinv_ = np.linalg.inv(Sp)
        return self

    def _d2(self, X):
        X = check_array(X, dtype=float)
        out = np.empty((len(X), len(self.classes_)))
        for c, m in enumerate(self.means_):
            diff = X - m
            out[:, c] = np.einsum("ij,jk,ik->i", diff, self.Sinv_, diff) if self.mahalanobis else (diff ** 2).sum(1)
        return out

    def predict_proba(self, X):
        return _softmax(-0.5 * self._d2(X))

    def predict(self, X):
        return self.classes_[np.argmin(self._d2(X), 1)]


# ----------------------------------------------------------------------------
# Parzen window
# ----------------------------------------------------------------------------
class ParzenClassifier(ClassifierMixin, BaseEstimator):
    """Class-conditional Gaussian kernel density estimates combined by Bayes' rule.

    ``bandwidth`` is h in p(x|c) = 1/n_c sum_i N(x; x_i, h^2 I).  If None it is
    ``bw_scale`` times the mean nearest-neighbour distance of the training set.
    """

    def __init__(self, bandwidth=None, bw_scale: float = 0.8, chunk: int = 256):
        self.bandwidth = bandwidth
        self.bw_scale = bw_scale
        self.chunk = chunk

    def fit(self, X, y):
        X = check_array(X, dtype=float)
        self.classes_, yi = _encode(y)
        self._Xc = [X[yi == c] for c in range(len(self.classes_))]
        counts = np.array([len(a) for a in self._Xc])
        self.log_priors_ = np.log(counts / counts.sum())
        if self.bandwidth is None:
            rng = np.random.default_rng(0)
            sub = X[rng.choice(len(X), min(400, len(X)), replace=False)]
            d2 = (sub ** 2).sum(1)[:, None] + (sub ** 2).sum(1)[None] - 2 * sub @ sub.T
            np.fill_diagonal(d2, np.inf)
            self.h_ = self.bw_scale * float(np.sqrt(np.maximum(d2.min(1), 1e-12)).mean())
        else:
            self.h_ = float(self.bandwidth)
        return self

    def _log_post(self, X):
        X = check_array(X, dtype=float)
        d = X.shape[1]
        h2 = self.h_ ** 2
        out = np.empty((len(X), len(self.classes_)))
        for c, Xc in enumerate(self._Xc):
            for s in range(0, len(X), self.chunk):
                A = X[s:s + self.chunk]
                d2 = (A ** 2).sum(1)[:, None] + (Xc ** 2).sum(1)[None] - 2 * A @ Xc.T
                logk = -np.maximum(d2, 0) / (2 * h2)
                out[s:s + self.chunk, c] = logsumexp(logk, axis=1) - np.log(len(Xc)) - 0.5 * d * np.log(2 * np.pi * h2)
        return out + self.log_priors_

    def predict_proba(self, X):
        L = self._log_post(X)
        return np.exp(L - logsumexp(L, axis=1, keepdims=True))

    def predict(self, X):
        return self.classes_[np.argmax(self._log_post(X), 1)]


# ----------------------------------------------------------------------------
# Softmax regression
# ----------------------------------------------------------------------------
class _Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.p, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


class SoftmaxRegression(ClassifierMixin, BaseEstimator):
    def __init__(self, l2: float = 1e-3, lr: float = 0.05, epochs: int = 80, batch: int = 128, random_state: int = 0):
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs
        self.batch = batch
        self.random_state = random_state

    def fit(self, X, y):
        X = check_array(X, dtype=float)
        self.classes_, yi = _encode(y)
        n, d = X.shape
        C = len(self.classes_)
        rng = np.random.default_rng(self.random_state)
        W = rng.normal(0, 0.01, (d, C))
        b = np.zeros(C)
        Y = np.eye(C)[yi]
        opt = _Adam([W, b], lr=self.lr)
        self.loss_curve_ = []
        for _ in range(self.epochs):
            perm = rng.permutation(n)
            tot = 0.0
            for s in range(0, n, self.batch):
                ix = perm[s:s + self.batch]
                P = _softmax(X[ix] @ W + b)
                G = (P - Y[ix]) / len(ix)
                opt.step([X[ix].T @ G + self.l2 * W, G.sum(0)])
                tot += -np.log(P[np.arange(len(ix)), yi[ix]] + 1e-12).sum()
            self.loss_curve_.append(tot / n + 0.5 * self.l2 * (W ** 2).sum())
        self.W_, self.b_ = W, b
        return self

    def predict_proba(self, X):
        return _softmax(check_array(X, dtype=float) @ self.W_ + self.b_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


# ----------------------------------------------------------------------------
# MLP
# ----------------------------------------------------------------------------
class NumpyMLP(ClassifierMixin, BaseEstimator):
    """Fully-connected network: [Linear -> ReLU -> (Dropout)]* -> Linear -> Softmax.

    Trained with mini-batch Adam, L2 weight decay and early stopping on a
    held-out validation split.  Gradients are verified against finite
    differences in the unit tests (tests/test_classifiers.py).
    """

    def __init__(self, hidden=(64, 32), l2: float = 1e-4, lr: float = 2e-3, epochs: int = 80, batch: int = 64,
                 dropout: float = 0.0, val_fraction: float = 0.1, patience: int = 12, random_state: int = 0):
        self.hidden = hidden
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs
        self.batch = batch
        self.dropout = dropout
        self.val_fraction = val_fraction
        self.patience = patience
        self.random_state = random_state

    # --- helpers ---------------------------------------------------------
    def _init(self, d, C, rng):
        sizes = [d, *self.hidden, C]
        self.Ws_ = [rng.normal(0, np.sqrt(2.0 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.bs_ = [np.zeros(b) for b in sizes[1:]]

    def _forward(self, X, train=False, rng=None):
        acts, masks = [X], []
        a = X
        for i, (W, b) in enumerate(zip(self.Ws_, self.bs_)):
            z = a @ W + b
            if i < len(self.Ws_) - 1:
                a = np.maximum(z, 0)
                if train and self.dropout > 0:
                    m = (rng.random(a.shape) >= self.dropout) / (1 - self.dropout)
                    a = a * m
                    masks.append(m)
                else:
                    masks.append(None)
            else:
                a = _softmax(z)
            acts.append(a)
        return acts, masks

    def _loss_and_grads(self, X, Y, rng=None, train=False):
        acts, masks = self._forward(X, train, rng)
        P = acts[-1]
        n = len(X)
        loss = -np.log((P * Y).sum(1) + 1e-12).mean() + 0.5 * self.l2 * sum((W ** 2).sum() for W in self.Ws_)
        gW, gb = [None] * len(self.Ws_), [None] * len(self.Ws_)
        delta = (P - Y) / n
        for i in range(len(self.Ws_) - 1, -1, -1):
            gW[i] = acts[i].T @ delta + self.l2 * self.Ws_[i]
            gb[i] = delta.sum(0)
            if i > 0:
                delta = delta @ self.Ws_[i].T
                if masks[i - 1] is not None:
                    delta = delta * masks[i - 1]
                delta = delta * (acts[i] > 0)
        return loss, gW, gb

    # --- API -------------------------------------------------------------
    def fit(self, X, y):
        X = check_array(X, dtype=float)
        self.classes_, yi = _encode(y)
        n, d = X.shape
        C = len(self.classes_)
        rng = np.random.default_rng(self.random_state)
        self._init(d, C, rng)
        perm = rng.permutation(n)
        nval = int(self.val_fraction * n) if self.val_fraction > 0 else 0
        va, tr = perm[:nval], perm[nval:]
        Y = np.eye(C)[yi]
        opt = _Adam([*self.Ws_, *self.bs_], lr=self.lr)
        best, best_state, wait = np.inf, None, 0
        self.train_loss_, self.val_loss_ = [], []
        for ep in range(self.epochs):
            order = rng.permutation(tr)
            tot = 0.0
            for s in range(0, len(order), self.batch):
                ix = order[s:s + self.batch]
                loss, gW, gb = self._loss_and_grads(X[ix], Y[ix], rng, True)
                opt.step(gW + gb)
                tot += loss * len(ix)
            self.train_loss_.append(tot / len(tr))
            if nval:
                P = self._forward(X[va])[0][-1]
                vl = -np.log((P * Y[va]).sum(1) + 1e-12).mean()
                self.val_loss_.append(vl)
                if vl < best - 1e-5:
                    best, wait = vl, 0
                    best_state = ([W.copy() for W in self.Ws_], [b.copy() for b in self.bs_])
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
        if best_state is not None:
            self.Ws_, self.bs_ = best_state
        self.n_epochs_ = len(self.train_loss_)
        return self

    def predict_proba(self, X):
        return self._forward(check_array(X, dtype=float))[0][-1]

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]
