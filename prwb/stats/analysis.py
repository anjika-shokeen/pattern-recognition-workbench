"""Statistical analysis of feature spaces.

Descriptive statistics, normality, hypothesis tests (ANOVA / Kruskal-Wallis /
Levene), effect sizes, multiple-testing correction, correlation structure,
class separability (Fisher ratio, Bhattacharyya distance) and multivariate
outlier detection (Mahalanobis).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as ss


# ----------------------------------------------------------------------------
# multiple testing
# ----------------------------------------------------------------------------
def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH step-up adjusted p-values (q-values) controlling the false-discovery rate."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.minimum(ranked, 1.0)
    return q


def bonferroni(p: np.ndarray) -> np.ndarray:
    return np.minimum(np.asarray(p, float) * len(p), 1.0)


# ----------------------------------------------------------------------------
# descriptive
# ----------------------------------------------------------------------------
def descriptive_table(X: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    X = np.asarray(X, float)
    q1, med, q3 = np.percentile(X, [25, 50, 75], axis=0)
    return pd.DataFrame({
        "feature": list(names), "mean": X.mean(0), "std": X.std(0, ddof=1), "min": X.min(0),
        "q1": q1, "median": med, "q3": q3, "max": X.max(0),
        "skewness": np.nan_to_num(ss.skew(X, axis=0)), "kurtosis": np.nan_to_num(ss.kurtosis(X, axis=0)),
    })


def normality_tests(X: np.ndarray, names: Sequence[str], max_n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Shapiro-Wilk and D'Agostino-Pearson K^2 per feature (random subsample if n > max_n)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    if X.shape[0] > max_n:
        X = X[rng.choice(X.shape[0], max_n, replace=False)]
    rows = []
    for j, nm in enumerate(names):
        col = X[:, j]
        if col.std() < 1e-12:
            rows.append((nm, np.nan, np.nan, np.nan, np.nan))
            continue
        w, pw = ss.shapiro(col)
        k2, pk = ss.normaltest(col)
        rows.append((nm, w, pw, k2, pk))
    df = pd.DataFrame(rows, columns=["feature", "shapiro_W", "shapiro_p", "dagostino_K2", "dagostino_p"])
    df["normal_at_5pct"] = (df["shapiro_p"] > 0.05)
    return df


# ----------------------------------------------------------------------------
# group comparison
# ----------------------------------------------------------------------------
def anova_table(X: np.ndarray, y: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    """One-way ANOVA, Kruskal-Wallis and Levene per feature + eta^2 effect size + BH q-values."""
    X = np.asarray(X, float)
    y = np.asarray(y)
    classes = np.unique(y)
    N = len(y)
    k = len(classes)
    rows = []
    for j, nm in enumerate(names):
        groups = [X[y == c, j] for c in classes]
        if X[:, j].std() < 1e-12:
            rows.append((nm, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        F, p = ss.f_oneway(*groups)
        H, pkw = ss.kruskal(*groups)
        L, pl = ss.levene(*groups)
        grand = X[:, j].mean()
        ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        sst = ((X[:, j] - grand) ** 2).sum()
        eta2 = ssb / sst
        omega2 = max((ssb - (k - 1) * (sst - ssb) / (N - k)) / (sst + (sst - ssb) / (N - k)), 0.0)
        rows.append((nm, F, p, H, pkw, pl, eta2, omega2))
    df = pd.DataFrame(rows, columns=["feature", "anova_F", "anova_p", "kruskal_H", "kruskal_p",
                                     "levene_p", "eta_squared", "omega_squared"])
    valid = df["anova_p"].notna()
    df["anova_q_BH"] = np.nan
    df.loc[valid, "anova_q_BH"] = benjamini_hochberg(df.loc[valid, "anova_p"].values)
    df["bonferroni_p"] = np.nan
    df.loc[valid, "bonferroni_p"] = bonferroni(df.loc[valid, "anova_p"].values)
    return df.sort_values("anova_F", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# correlation structure
# ----------------------------------------------------------------------------
def correlation_summary(X: np.ndarray, names: Sequence[str], threshold: float = 0.9) -> dict:
    X = np.asarray(X, float)
    keep = np.flatnonzero(X.std(0) > 1e-12)
    Xk = X[:, keep]
    C = np.corrcoef(Xk, rowvar=False)
    iu = np.triu_indices_from(C, k=1)
    absC = np.abs(C[iu])
    ev = np.linalg.eigvalsh(np.cov(Xk, rowvar=False))
    ev = np.clip(ev, 1e-15, None)
    return {
        "n_features": int(X.shape[1]),
        "n_constant": int(X.shape[1] - len(keep)),
        "mean_abs_corr": float(absC.mean()),
        "n_pairs_above_threshold": int((absC > threshold).sum()),
        "threshold": threshold,
        "condition_number": float(ev.max() / ev.min()),
        "effective_rank": float(np.exp(-(ev / ev.sum() * np.log(ev / ev.sum())).sum())),
        "corr": C, "kept_index": keep,
    }


# ----------------------------------------------------------------------------
# separability
# ----------------------------------------------------------------------------
def bhattacharyya_matrix(X: np.ndarray, y: np.ndarray, shrink: float = 0.1) -> np.ndarray:
    """Pairwise Bhattacharyya distance between Gaussian class models (shrinkage covariances).

    D_B = 1/8 (m1-m2)^T S^-1 (m1-m2) + 1/2 ln( det S / sqrt(det S1 det S2) ),  S = (S1+S2)/2
    A larger value means an easier pair to separate; exp(-D_B) upper-bounds the Bayes error.
    """
    X = np.asarray(X, float)
    classes = np.unique(y)
    d = X.shape[1]
    mus, covs = [], []
    for c in classes:
        Xc = X[y == c]
        S = np.cov(Xc, rowvar=False) + 1e-9 * np.eye(d)
        S = (1 - shrink) * S + shrink * np.trace(S) / d * np.eye(d)
        mus.append(Xc.mean(0))
        covs.append(S)
    k = len(classes)
    D = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            S = (covs[i] + covs[j]) / 2
            dm = mus[i] - mus[j]
            term1 = dm @ np.linalg.solve(S, dm) / 8
            term2 = 0.5 * (np.linalg.slogdet(S)[1] - 0.5 * (np.linalg.slogdet(covs[i])[1] + np.linalg.slogdet(covs[j])[1]))
            D[i, j] = D[j, i] = term1 + term2
    return D


def class_separability(X: np.ndarray, y: np.ndarray) -> dict:
    """Scatter-matrix criteria: trace(Sw^-1 Sb), Davies-Bouldin-style ratio and mean pair distance."""
    X = np.asarray(X, float)
    y = np.asarray(y)
    d = X.shape[1]
    mu = X.mean(0)
    Sw = np.zeros((d, d))
    Sb = np.zeros((d, d))
    for c in np.unique(y):
        Xc = X[y == c]
        mc = Xc.mean(0)
        Sw += (Xc - mc).T @ (Xc - mc)
        Sb += len(Xc) * np.outer(mc - mu, mc - mu)
    Sw = Sw / len(X) + 1e-6 * np.eye(d)
    Sb = Sb / len(X)
    J = float(np.trace(np.linalg.solve(Sw, Sb)))
    return {"J_trace_SwInv_Sb": J, "trace_Sb_over_trace_Sw": float(np.trace(Sb) / np.trace(Sw))}


# ----------------------------------------------------------------------------
# outliers
# ----------------------------------------------------------------------------
def mahalanobis_outliers(X: np.ndarray, alpha: float = 0.001, shrink: float = 0.05):
    """Squared Mahalanobis distance and chi-square based outlier flag."""
    X = np.asarray(X, float)
    keep = X.std(0) > 1e-12
    Xk = X[:, keep]
    mu = Xk.mean(0)
    S = np.cov(Xk, rowvar=False)
    d = Xk.shape[1]
    S = (1 - shrink) * S + shrink * np.trace(S) / d * np.eye(d)
    diff = Xk - mu
    d2 = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(S), diff)
    thr = ss.chi2.ppf(1 - alpha, df=d)
    return d2, thr, d2 > thr


def pairwise_class_ttests(x: np.ndarray, y: np.ndarray, class_names: Sequence[str]) -> pd.DataFrame:
    """Welch t-tests of one feature between every pair of classes, Cohen's d, Bonferroni."""
    rows = []
    classes = np.unique(y)
    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            xa, xb = x[y == a], x[y == b]
            t, p = ss.ttest_ind(xa, xb, equal_var=False)
            sp = np.sqrt(((len(xa) - 1) * xa.var(ddof=1) + (len(xb) - 1) * xb.var(ddof=1)) / (len(xa) + len(xb) - 2))
            rows.append((class_names[a], class_names[b], t, p, (xa.mean() - xb.mean()) / (sp + 1e-12)))
    df = pd.DataFrame(rows, columns=["class_a", "class_b", "t", "p", "cohens_d"])
    df["p_bonferroni"] = bonferroni(df["p"].values)
    return df
