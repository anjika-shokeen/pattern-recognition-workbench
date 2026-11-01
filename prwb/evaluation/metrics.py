"""Performance evaluation: classification, clustering and statistical model comparison."""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as ss
from scipy.optimize import linear_sum_assignment
from sklearn import metrics as skm
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import label_binarize


# ----------------------------------------------------------------------------
# classification
# ----------------------------------------------------------------------------
def expected_calibration_error(proba: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    conf = proba.max(1)
    pred = proba.argmax(1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def multiclass_scores(y, pred, proba=None) -> Dict[str, float]:
    out = {
        "accuracy": skm.accuracy_score(y, pred),
        "balanced_accuracy": skm.balanced_accuracy_score(y, pred),
        "macro_f1": skm.f1_score(y, pred, average="macro"),
        "macro_precision": skm.precision_score(y, pred, average="macro", zero_division=0),
        "macro_recall": skm.recall_score(y, pred, average="macro", zero_division=0),
        "mcc": skm.matthews_corrcoef(y, pred),
        "cohen_kappa": skm.cohen_kappa_score(y, pred),
    }
    if proba is not None:
        try:
            out["log_loss"] = skm.log_loss(y, np.clip(proba, 1e-12, 1), labels=np.arange(proba.shape[1]))
            if proba.shape[1] == 2:
                out["roc_auc"] = skm.roc_auc_score(y, proba[:, 1])
            else:
                out["roc_auc"] = skm.roc_auc_score(y, proba, multi_class="ovr", average="macro")
            out["ece"] = expected_calibration_error(proba, np.asarray(y))
        except ValueError:
            pass
    return out


def cross_validate_model(estimator, X, y, n_splits: int = 5, n_repeats: int = 2, seed: int = 0,
                         keep_oof: bool = True) -> dict:
    """Repeated stratified k-fold CV.  Returns fold scores plus out-of-fold predictions of repeat 0.

    The estimator (a Pipeline containing the scaler) is cloned and re-fitted on
    every training fold, so no statistic of the held-out fold leaks into training.
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    fold_rows = []
    oof_pred = np.full(len(y), -1)
    oof_proba = None
    fit_t, pred_t = 0.0, 0.0
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        m = clone(estimator)
        t0 = time.perf_counter()
        m.fit(X[tr], y[tr])
        t1 = time.perf_counter()
        pred = m.predict(X[te])
        proba = m.predict_proba(X[te]) if hasattr(m, "predict_proba") else None
        t2 = time.perf_counter()
        fit_t += t1 - t0
        pred_t += t2 - t1
        sc = multiclass_scores(y[te], pred, proba)
        sc["fold"] = i
        fold_rows.append(sc)
        if keep_oof and i < n_splits:  # first repeat covers every sample exactly once
            if oof_proba is None and proba is not None:
                oof_proba = np.zeros((len(y), proba.shape[1]))
            oof_pred[te] = pred
            if proba is not None:
                oof_proba[te] = proba
    folds = pd.DataFrame(fold_rows)
    n_f = len(folds)
    return {"folds": folds, "oof_pred": oof_pred, "oof_proba": oof_proba,
            "fit_time": fit_t / n_f, "predict_time": pred_t / n_f}


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


# ----------------------------------------------------------------------------
# hypothesis tests for comparing classifiers
# ----------------------------------------------------------------------------
def mcnemar_test(y, pred_a, pred_b) -> dict:
    """McNemar's test with continuity correction (Dietterich, 1998) on paired predictions."""
    a_ok, b_ok = pred_a == y, pred_b == y
    n01 = int(np.sum(a_ok & ~b_ok))   # A right, B wrong
    n10 = int(np.sum(~a_ok & b_ok))   # A wrong, B right
    if n01 + n10 == 0:
        return {"n01": n01, "n10": n10, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return {"n01": n01, "n10": n10, "chi2": float(chi2), "p": float(ss.chi2.sf(chi2, 1))}


def corrected_resampled_ttest(scores_a, scores_b, n_train: int, n_test: int) -> dict:
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated CV scores."""
    d = np.asarray(scores_a) - np.asarray(scores_b)
    k = len(d)
    var = d.var(ddof=1)
    if var < 1e-18:
        return {"mean_diff": float(d.mean()), "t": 0.0, "p": 1.0}
    t = d.mean() / np.sqrt((1.0 / k + n_test / n_train) * var)
    return {"mean_diff": float(d.mean()), "t": float(t), "p": float(2 * ss.t.sf(abs(t), k - 1))}


def friedman_nemenyi(score_matrix: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Friedman test across blocks (rows = datasets / settings, cols = classifiers) + Nemenyi CD."""
    S = score_matrix.values
    n, k = S.shape
    stat, p = ss.friedmanchisquare(*[S[:, j] for j in range(k)])
    ranks = np.vstack([ss.rankdata(-row) for row in S])
    avg = ranks.mean(0)
    q = ss.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))
    return {"chi2": float(stat), "p": float(p), "avg_ranks": pd.Series(avg, index=score_matrix.columns).sort_values(),
            "cd": float(cd), "n_blocks": n, "k": k}


# ----------------------------------------------------------------------------
# clustering
# ----------------------------------------------------------------------------
def hungarian_map(y_true, y_pred):
    """Optimal one-to-one mapping from clusters to classes (Kuhn-Munkres)."""
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    ct, cp = np.unique(yt), np.unique(yp)
    M = np.zeros((len(cp), len(ct)), dtype=int)
    for i, a in enumerate(cp):
        for j, b in enumerate(ct):
            M[i, j] = np.sum((yp == a) & (yt == b))
    r, c = linear_sum_assignment(-M)
    mapping = {cp[i]: ct[j] for i, j in zip(r, c)}
    return mapping, M


def clustering_accuracy(y_true, y_pred) -> float:
    mapping, _ = hungarian_map(y_true, y_pred)
    mapped = np.array([mapping.get(p, -1) for p in y_pred])
    return float(np.mean(mapped == np.asarray(y_true)))


def purity(y_true, y_pred) -> float:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    return float(sum(np.bincount(yt[yp == c]).max() for c in np.unique(yp)) / len(yt))


def clustering_scores(X, y_true, labels) -> Dict[str, float]:
    """External (needs ground truth) and internal validity indices."""
    labels = np.asarray(labels)
    out = {
        "ARI": skm.adjusted_rand_score(y_true, labels),
        "NMI": skm.normalized_mutual_info_score(y_true, labels),
        "AMI": skm.adjusted_mutual_info_score(y_true, labels),
        "homogeneity": skm.homogeneity_score(y_true, labels),
        "completeness": skm.completeness_score(y_true, labels),
        "V-measure": skm.v_measure_score(y_true, labels),
        "purity": purity(y_true, labels),
        "cluster_acc": clustering_accuracy(y_true, labels),
        "n_clusters": int(len(np.unique(labels[labels >= 0]))),
        "noise_fraction": float(np.mean(labels < 0)),
    }
    m = labels >= 0
    if len(np.unique(labels[m])) > 1:
        out["silhouette"] = skm.silhouette_score(X[m], labels[m])
        out["davies_bouldin"] = skm.davies_bouldin_score(X[m], labels[m])
        out["calinski_harabasz"] = skm.calinski_harabasz_score(X[m], labels[m])
    return out


def bootstrap_stability(make_model, X, n_boot: int = 10, seed: int = 0) -> float:
    """Mean pairwise ARI between clusterings of bootstrap resamples evaluated on the full data."""
    rng = np.random.default_rng(seed)
    labs = []
    for b in range(n_boot):
        idx = rng.choice(len(X), len(X), replace=True)
        m = make_model().fit(X[idx])
        labs.append(m.predict(X))
    aris = [skm.adjusted_rand_score(labs[i], labs[j]) for i in range(n_boot) for j in range(i + 1, n_boot)]
    return float(np.mean(aris)), float(np.std(aris))
