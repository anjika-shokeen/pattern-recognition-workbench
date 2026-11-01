"""Plotting helpers (matplotlib) shared by the CLI, the dashboard and the report generator."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn import metrics as skm
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize

PALETTE = ["#1f5fa8", "#d9822b", "#2a9d6f", "#c0392b", "#7d5ba6", "#8c6d31", "#d4579a", "#4a4a4a", "#17a2b8", "#a3a82e"]
DPI = 130


def set_style():
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": DPI, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE), "legend.frameon": False, "legend.fontsize": 8,
        "figure.constrained_layout.use": True, "axes.titleweight": "bold",
    })


set_style()


def save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def plot_image_grid(X, y, class_names, per_class: int = 8, path: str = "samples.png"):
    C = len(class_names)
    fig, axes = plt.subplots(C, per_class, figsize=(per_class * 0.8, C * 0.8))
    for c in range(C):
        idx = np.flatnonzero(y == c)[:per_class]
        for j in range(per_class):
            ax = axes[c, j]
            ax.axis("off")
            if j < len(idx):
                ax.imshow(X[idx[j]], cmap="gray_r")
            if j == 0:
                ax.set_title(str(class_names[c]), fontsize=8, loc="left")
    return save(fig, path)


def plot_class_balance(datasets: Dict[str, np.ndarray], path: str):
    fig, axes = plt.subplots(1, len(datasets), figsize=(2.6 * len(datasets), 2.4), sharey=False)
    for ax, (name, counts) in zip(np.atleast_1d(axes), datasets.items()):
        ax.bar(range(len(counts)), counts, color=PALETTE[: len(counts)])
        ax.set_title(name)
        ax.set_xlabel("class")
        ax.set_xticks(range(len(counts)))
    np.atleast_1d(axes)[0].set_ylabel("samples")
    return save(fig, path)


def plot_signal_examples(X, y, class_names, fs, path):
    C = len(class_names)
    t = np.arange(X.shape[1]) / fs
    fig, axes = plt.subplots(2, C, figsize=(2.0 * C, 3.6))
    for c in range(C):
        i = np.flatnonzero(y == c)[0]
        axes[0, c].plot(t, X[i], lw=0.8)
        axes[0, c].set_title(class_names[c])
        axes[0, c].set_xlabel("time (s)")
        sp = np.abs(np.fft.rfft((X[i] - X[i].mean()) * np.hanning(len(X[i])))) ** 2
        axes[1, c].plot(np.fft.rfftfreq(len(X[i]), 1 / fs), sp, lw=0.8, color=PALETTE[1])
        axes[1, c].set_xlabel("freq (Hz)")
    axes[0, 0].set_ylabel("amplitude")
    axes[1, 0].set_ylabel("power")
    return save(fig, path)


# ----------------------------------------------------------------------------
# feature visualisation
# ----------------------------------------------------------------------------
def plot_hog_visual(img, cells, path: str, cell: int = 4):
    """img (h,w); cells (ny, nx, nbins)."""
    from ..features.image import image_gradients
    mag, _ = image_gradients(img[None])
    ny, nx, nb = cells.shape
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    axes[0].imshow(img, cmap="gray_r")
    axes[0].set_title("input")
    axes[1].imshow(mag[0], cmap="magma")
    axes[1].set_title("gradient magnitude")
    axes[2].imshow(img, cmap="gray_r", alpha=0.25)
    axes[2].set_title("HOG cell glyphs")
    mx = cells.max() + 1e-9
    for i in range(ny):
        for j in range(nx):
            cy, cx = i * cell + cell / 2 - 0.5, j * cell + cell / 2 - 0.5
            for b in range(nb):
                theta = (b + 0.5) * np.pi / nb + np.pi / 2  # edge direction is perpendicular to gradient
                L = cell / 2 * cells[i, j, b] / mx
                axes[2].plot([cx - L * np.cos(theta), cx + L * np.cos(theta)],
                             [cy - L * np.sin(theta), cy + L * np.sin(theta)], color=PALETTE[0], lw=0.9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    return save(fig, path)


def plot_lbp_visual(img, codes, mapped, nbins, path: str):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    axes[0].imshow(img, cmap="gray_r")
    axes[0].set_title("input")
    axes[1].imshow(codes, cmap="nipy_spectral")
    axes[1].set_title("LBP code image (P=8, R=1)")
    h = np.bincount(mapped.ravel(), minlength=nbins)
    axes[2].bar(range(nbins), h / h.sum(), color=PALETTE[2])
    axes[2].set_title("uniform-LBP histogram")
    axes[2].set_xlabel("bin")
    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    return save(fig, path)


def plot_pca_scree(ratio, path: str, n_show: int = 60, recon=None):
    fig, axes = plt.subplots(1, 2 if recon is None else 3, figsize=(7.4, 2.5))
    k = min(n_show, len(ratio))
    axes[0].bar(range(1, k + 1), ratio[:k], color=PALETTE[0])
    axes[0].set_title("scree plot")
    axes[0].set_xlabel("component")
    axes[0].set_ylabel("explained variance ratio")
    cum = np.cumsum(ratio)
    axes[1].plot(range(1, len(cum) + 1), cum, color=PALETTE[1])
    for thr in (0.9, 0.95, 0.99):
        kk = int(np.searchsorted(cum, thr) + 1)
        axes[1].axhline(thr, ls=":", color="gray", lw=0.8)
        axes[1].annotate(f"{int(thr*100)}%: {kk}", (kk, thr), textcoords="offset points", xytext=(6, -10), fontsize=7)
    axes[1].set_title("cumulative variance")
    axes[1].set_xlabel("components")
    axes[1].set_xlim(0, min(len(cum), 200))
    if recon is not None:
        ks, errs = recon
        axes[2].plot(ks, errs, marker="o", ms=3, color=PALETTE[3])
        axes[2].set_title("reconstruction MSE")
        axes[2].set_xlabel("components")
        axes[2].set_yscale("log")
    return save(fig, path)


def plot_reconstructions(orig, recons: Dict[int, np.ndarray], path: str):
    n = orig.shape[0]
    rows = 1 + len(recons)
    fig, axes = plt.subplots(rows, n, figsize=(n * 0.8, rows * 0.85))
    for j in range(n):
        axes[0, j].imshow(orig[j], cmap="gray_r")
        for i, (k, R) in enumerate(recons.items(), start=1):
            axes[i, j].imshow(R[j], cmap="gray_r")
    for i, lab in enumerate(["original"] + [f"{k} PCs" for k in recons]):
        axes[i, 0].set_ylabel(lab, fontsize=7, rotation=0, ha="right", va="center")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    return save(fig, path)


def plot_embeddings(embs: Dict[str, np.ndarray], y, class_names, path: str):
    n = len(embs)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4))
    for ax, (title, E) in zip(np.atleast_1d(axes), embs.items()):
        for c in np.unique(y):
            m = y == c
            ax.scatter(E[m, 0], E[m, 1], s=5, color=PALETTE[c % len(PALETTE)], label=str(class_names[c]), alpha=0.75)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    np.atleast_1d(axes)[-1].legend(markerscale=2.5, ncol=2, loc="best", fontsize=6)
    return save(fig, path)


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------
def plot_anova_top(anova: pd.DataFrame, path: str, top: int = 20):
    d = anova.head(top).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    axes[0].barh(d["feature"], np.log10(d["anova_F"]), color=PALETTE[0])
    axes[0].set_xlabel("log10 F statistic")
    axes[0].set_title(f"top-{top} features by ANOVA F")
    axes[0].tick_params(axis="y", labelsize=6)
    ok = anova.dropna(subset=["anova_q_BH"])
    sig = ok["anova_q_BH"] < 0.05
    axes[1].scatter(ok["eta_squared"][sig], -np.log10(ok["anova_q_BH"][sig] + 1e-300), s=6, color=PALETTE[0], label="q < 0.05")
    axes[1].scatter(ok["eta_squared"][~sig], -np.log10(ok["anova_q_BH"][~sig] + 1e-300), s=6, color=PALETTE[3], label="n.s.")
    axes[1].axhline(-np.log10(0.05), ls=":", color="gray")
    axes[1].set_xlabel("effect size  eta-squared")
    axes[1].set_ylabel("-log10 q (BH-FDR)")
    axes[1].set_title("significance vs effect size")
    axes[1].legend()
    return save(fig, path)


def plot_heatmap(M, xlabels, ylabels, path, title="", cmap="viridis", fmt="{:.2f}", vmin=None, vmax=None,
                 annotate=True, figsize=(5.2, 4.4), cbar_label=""):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.grid(False)
    ax.set_title(title)
    if annotate:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    r, g, b, _ = im.cmap(im.norm(M[i, j]))
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=6,
                            color="white" if lum < 0.5 else "black")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(cbar_label)
    return save(fig, path)


def plot_correlation(C, path, title="feature correlation"):
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    im = ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    return save(fig, path)


def plot_hist_with_threshold(values, thr, path, xlabel="squared Mahalanobis distance", title=""):
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    ax.hist(values, bins=50, color=PALETTE[0], alpha=0.85)
    ax.axvline(thr, color=PALETTE[3], ls="--", label=f"chi2 threshold = {thr:.1f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    return save(fig, path)


def plot_feature_distributions(X, y, names, class_names, path, ncols=4):
    n = len(names)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.9 * ncols, 1.7 * nrows))
    for ax, j, nm in zip(np.atleast_1d(axes).ravel(), range(n), names):
        data = [X[y == c, j] for c in np.unique(y)]
        ax.boxplot(data, showfliers=False, widths=0.6, patch_artist=True,
                   boxprops=dict(facecolor="#cfe0f3", lw=0.6), medianprops=dict(color=PALETTE[3], lw=0.8),
                   whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
        ax.set_title(nm, fontsize=7)
        ax.set_xticks([])
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis("off")
    return save(fig, path)


# ----------------------------------------------------------------------------
# classification
# ----------------------------------------------------------------------------
def plot_confusion(y, pred, class_names, path, title="", normalize=True):
    cm = skm.confusion_matrix(y, pred)
    M = cm / cm.sum(1, keepdims=True) if normalize else cm
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    ax.grid(False)
    for i in range(len(cm)):
        for j in range(len(cm)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if M[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return save(fig, path)


def plot_roc_multiclass(y, proba, class_names, path, title=""):
    C = proba.shape[1]
    Y = label_binarize(y, classes=np.arange(C))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for c in range(C):
        fpr, tpr, _ = skm.roc_curve(Y[:, c], proba[:, c])
        axes[0].plot(fpr, tpr, lw=1, color=PALETTE[c % 10], label=f"{class_names[c]} ({skm.auc(fpr, tpr):.3f})")
    axes[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("ROC (one-vs-rest) " + title)
    axes[0].legend(fontsize=6, ncol=2)
    for c in range(C):
        p, r, _ = skm.precision_recall_curve(Y[:, c], proba[:, c])
        axes[1].plot(r, p, lw=1, color=PALETTE[c % 10])
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("precision-recall")
    axes[1].set_ylim(0.5, 1.01)
    return save(fig, path)


def plot_calibration(y, probas: Dict[str, np.ndarray], path, n_bins: int = 10):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].plot([0, 1], [0, 1], "k:", label="perfect")
    for i, (name, P) in enumerate(probas.items()):
        conf = P.max(1)
        correct = (P.argmax(1) == y).astype(int)
        frac, mean = calibration_curve(correct, conf, n_bins=n_bins, strategy="quantile")
        axes[0].plot(mean, frac, marker="o", ms=3, lw=1, label=name, color=PALETTE[i % 10])
        axes[1].hist(conf, bins=20, alpha=0.5, label=name, color=PALETTE[i % 10], range=(0, 1))
    axes[0].set_xlabel("mean predicted confidence")
    axes[0].set_ylabel("empirical accuracy")
    axes[0].set_title("reliability diagram")
    axes[0].legend()
    axes[1].set_xlabel("confidence")
    axes[1].set_title("confidence histogram")
    axes[1].set_yscale("log")
    return save(fig, path)


def plot_cv_boxplot(fold_scores: Dict[str, Sequence[float]], path, metric="accuracy", title=""):
    names = list(fold_scores)
    order = np.argsort([np.median(fold_scores[n]) for n in names])
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.boxplot([fold_scores[names[i]] for i in order], tick_labels=[names[i] for i in order], patch_artist=True,
               boxprops=dict(facecolor="#cfe0f3", lw=0.7), medianprops=dict(color=PALETTE[3]))
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=40, labelsize=7)
    return save(fig, path)


def plot_critical_difference(avg_ranks: pd.Series, cd: float, path: str, title: str = ""):
    """Demsar-style critical-difference diagram (Friedman / Nemenyi)."""
    r = avg_ranks.sort_values()
    k = len(r)
    fig, ax = plt.subplots(figsize=(7.2, 0.45 * k + 1.8))
    lo, hi = np.floor(r.min()), np.ceil(r.max())
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.set_ylim(0, k + 3)
    ax.axis("off")
    y0 = k + 1.2
    ax.plot([lo, hi], [y0, y0], color="k", lw=1)
    for t in np.arange(lo, hi + 1):
        ax.plot([t, t], [y0, y0 + 0.15], color="k", lw=1)
        ax.text(t, y0 + 0.35, f"{int(t)}", ha="center", fontsize=8)
    half = (k + 1) // 2
    for i, (name, rank) in enumerate(r.items()):
        left = i < half
        yy = y0 - 1.0 - (i if left else (k - 1 - i)) * 0.9
        xx = lo - 0.25 if left else hi + 0.25
        ax.plot([rank, rank], [y0, yy], color="gray", lw=0.7)
        ax.plot([rank, xx], [yy, yy], color="gray", lw=0.7)
        ax.text(xx + (-0.03 if left else 0.03), yy, f"{name} ({rank:.2f})", ha="right" if left else "left",
                va="center", fontsize=7)
    ax.plot([lo, lo + cd], [k + 2.4, k + 2.4], color=PALETTE[3], lw=2)
    ax.text(lo + cd / 2, k + 2.65, f"CD = {cd:.2f}", ha="center", fontsize=8, color=PALETTE[3])
    # bars joining groups of classifiers not significantly different
    vals = r.values
    ybar = y0 - 0.45
    drawn = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and not any(i >= a and j <= b for a, b in drawn):
            ax.plot([vals[i] - 0.02, vals[j] + 0.02], [ybar, ybar], color="k", lw=2.2)
            ybar -= 0.18
            drawn.append((i, j))
    ax.set_title(title)
    return save(fig, path)


def plot_line_family(x, series: Dict[str, Sequence[float]], path, xlabel, ylabel, title="", logx=False,
                     errs: Optional[Dict[str, Sequence[float]]] = None, ylim=None, figsize=(5.2, 3.2), markers=True):
    fig, ax = plt.subplots(figsize=figsize)
    for i, (name, y) in enumerate(series.items()):
        ax.plot(x, y, marker="o" if markers else None, ms=3, lw=1.3, label=name, color=PALETTE[i % 10])
        if errs and name in errs:
            ax.fill_between(x, np.asarray(y) - np.asarray(errs[name]), np.asarray(y) + np.asarray(errs[name]),
                            alpha=0.15, color=PALETTE[i % 10])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logx:
        ax.set_xscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend()
    return save(fig, path)


def plot_scatter_time_acc(df: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for i, r in df.reset_index().iterrows():
        ax.scatter(r["fit_time"], r["accuracy"], s=35, color=PALETTE[i % 10])
        ax.annotate(r["classifier"], (r["fit_time"], r["accuracy"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("mean training time per fold (s, log)")
    ax.set_ylabel("CV accuracy")
    ax.set_title("accuracy / cost trade-off")
    return save(fig, path)


def plot_training_curves(curves: Dict[str, Sequence[float]], path, ylabel="loss", title=""):
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for i, (n, c) in enumerate(curves.items()):
        ax.plot(c, label=n, color=PALETTE[i % 10])
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return save(fig, path)


# ----------------------------------------------------------------------------
# clustering
# ----------------------------------------------------------------------------
def plot_k_selection(sw: dict, gap: Optional[dict], path: str, true_k: Optional[int] = None):
    n = 4 if gap is not None else 3
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n + 0.4, 2.7))
    ks = sw["k"]
    axes[0].plot(ks, sw["inertia"], marker="o", ms=3)
    axes[0].set_title("elbow (inertia)")
    axes[1].plot(ks, sw["silhouette"], marker="o", ms=3, color=PALETTE[1])
    axes[1].set_title("silhouette")
    axes[2].plot(ks, sw["bic"], marker="o", ms=3, color=PALETTE[2], label="BIC")
    axes[2].plot(ks, sw["aic"], marker="s", ms=3, color=PALETTE[4], label="AIC")
    axes[2].set_title("GMM information criteria")
    axes[2].legend()
    if gap is not None:
        axes[3].errorbar(gap["k"], gap["gap"], yerr=gap["s"], marker="o", ms=3, color=PALETTE[3], capsize=2)
        axes[3].set_title("gap statistic")
    for ax in axes:
        ax.set_xlabel("k")
        if true_k:
            ax.axvline(true_k, color="gray", ls=":", lw=1)
    return save(fig, path)


def plot_contingency(y_true, labels, class_names, path, title=""):
    from .metrics import hungarian_map
    mapping, M = hungarian_map(y_true, labels)
    cp = np.unique(labels)
    # order clusters by the class they map to
    order = sorted(range(len(cp)), key=lambda i: (mapping.get(cp[i], 99), -M[i].max()))
    M = M[order]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(M, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=7)
    ax.set_yticks(range(len(cp)))
    ax.set_yticklabels([f"cluster {cp[i]}" for i in order], fontsize=7)
    ax.set_xlabel("true class")
    ax.set_title(title)
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j]:
                ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=6, color="white" if M[i, j] > M.max() / 2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return save(fig, path)


def plot_silhouette(X, labels, path, title=""):
    s = skm.silhouette_samples(X, labels)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ylo = 0
    for i, c in enumerate(np.unique(labels)):
        v = np.sort(s[labels == c])
        ax.fill_betweenx(np.arange(ylo, ylo + len(v)), 0, v, color=PALETTE[i % 10], alpha=0.85)
        ax.text(-0.07, ylo + len(v) / 2, str(c), fontsize=7)
        ylo += len(v) + 10
    ax.axvline(s.mean(), color=PALETTE[3], ls="--", label=f"mean = {s.mean():.3f}")
    ax.set_xlabel("silhouette coefficient")
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend()
    return save(fig, path)


def plot_dendrogram(X, path, y=None, n_show: int = 150, seed: int = 0):
    from scipy.cluster.hierarchy import dendrogram, linkage
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n_show, len(X)), replace=False)
    Z = linkage(X[idx], "ward")
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=0.45 * Z[-1, 2], above_threshold_color="gray")
    ax.set_title(f"Ward dendrogram ({len(idx)} random samples)")
    ax.set_ylabel("merge distance")
    return save(fig, path)


def plot_metric_bars(df: pd.DataFrame, metrics: List[str], path: str, title: str = ""):
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    w = 0.8 / len(metrics)
    x = np.arange(len(df))
    for i, m in enumerate(metrics):
        ax.bar(x + i * w, df[m], w, label=m, color=PALETTE[i % 10])
    ax.set_xticks(x + w * (len(metrics) - 1) / 2)
    ax.set_xticklabels(df.index, rotation=35, ha="right", fontsize=7)
    ax.set_title(title)
    ax.legend(ncol=len(metrics))
    ax.set_ylim(0, 1.05)
    return save(fig, path)


def plot_cluster_scatter(E, y, labels, path, class_names=None):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for ax, vec, ttl in zip(axes, [y, labels], ["ground-truth classes", "discovered clusters"]):
        for i, c in enumerate(np.unique(vec)):
            m = vec == c
            ax.scatter(E[m, 0], E[m, 1], s=5, color=PALETTE[i % 10] if c >= 0 else "lightgray", alpha=0.8)
        ax.set_title(ttl)
        ax.set_xticks([])
        ax.set_yticks([])
    return save(fig, path)


# ----------------------------------------------------------------------------
# architecture diagrams
# ----------------------------------------------------------------------------
def _box(ax, x, y, w, h, text, fc="#dbe8f7", ec="#1f5fa8", fs=8, bold=False, ls="-", lw=1.1, tc="#1a1a1a"):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def _arrow(ax, x0, y0, x1, y1, color="#444", ls="-", lw=1.1, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0), zorder=1)


def _canvas(w, h, xmax=100, ymax=100):
    fig, ax = plt.subplots(figsize=(w, h), constrained_layout=False)
    fig.subplots_adjust(0.005, 0.005, 0.995, 0.995)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    ax.grid(False)
    return fig, ax


C_DATA, C_FEAT, C_STAT, C_MODEL, C_EVAL, C_IO = "#e9e9e9", "#dbe8f7", "#fbe9d0", "#d8f0e4", "#f5d9d5", "#f4f1de"


def diagram_high_level(path):
    fig, ax = _canvas(7.6, 3.3, 100, 44)
    ax.text(50, 41, "High-level view of the End-to-End Pattern Recognition Workbench", ha="center", fontsize=10, fontweight="bold")
    stages = [("Data\\nacquisition", "datasets.py", C_DATA, "#666"), ("Pre-\\nprocessing", "preprocess.py\\nStandardizer", C_DATA, "#666"),
              ("Feature\\nextraction", "features/", C_FEAT, "#1f5fa8"), ("Statistical\\nanalysis", "stats/", C_STAT, "#d9822b"),
              ("Classification\\n/ Clustering", "classifiers/\\nclustering/", C_MODEL, "#2a9d6f"),
              ("Performance\\nevaluation", "evaluation/", C_EVAL, "#c0392b")]
    w, gap, x0 = 13.6, 2.8, 2.2
    for i, (t, m, c, e) in enumerate(stages):
        x = x0 + i * (w + gap)
        _box(ax, x, 17, w, 17, "", fc=c, ec=e)
        ax.text(x + w / 2, 29.5, t.replace("\\n", "\n"), ha="center", va="center", fontsize=8, fontweight="bold", zorder=3, linespacing=1.15)
        ax.text(x + w / 2, 21.5, m.replace("\\n", "\n"), ha="center", va="center", fontsize=6.3, style="italic", color="#444", zorder=3)
        if i < len(stages) - 1:
            _arrow(ax, x + w, 25.5, x + w + gap, 25.5)
        _arrow(ax, x + w / 2, 17, x + w / 2, 11.2, color="#999")
    _box(ax, 2.2, 3, 95.6, 8, "Interface & reporting:   CLI (python -m prwb)   -   Streamlit dashboard   -   CSV / JSON tables   -   PNG figures   -   model.joblib",
         fc=C_IO, ec="#8c6d31", fs=7.5)
    return save(fig, path)


def diagram_architecture(path):
    fig, ax = _canvas(7.6, 6.9, 100, 92)
    ax.text(50, 89, "Detailed layered architecture of the package prwb", ha="center", fontsize=10, fontweight="bold")
    ax.text(2, 81.3, "INTERFACE", fontsize=6, color="#8c6d31", fontweight="bold")
    ax.text(2, 70.0, "ORCHESTRATION", fontsize=6, color="#7d5ba6", fontweight="bold")
    ax.text(2, 54.3, "CORE MODULES  (every algorithm marked 'scratch' is implemented in NumPy and validated against scikit-learn)", fontsize=6, color="#444", fontweight="bold")
    iface = [(2, 38, "cli.py\npython -m prwb  datasets | features | stats\nclassify | cluster | benchmark | train | predict"),
             (41, 27, "app/streamlit_app.py\ninteractive dashboard\n(5 tabs, digit recogniser)"),
             (69, 29, "scripts/run_experiments.py\nfull reproducible protocol\n(--quick for smoke test)")]
    for x, w, t in iface:
        _box(ax, x, 72, w, 8, t, fc=C_IO, ec="#8c6d31", fs=6.6)
    orch = [(2, 46, "pipeline.py  -  facade for one (dataset, feature-set) pair\nquick_classify | quick_cluster | quick_stats | train_final_model"),
            (50, 29, "experiments.py  -  class Workbench\n9 experiments -> tables, figures, summary.json"),
            (81, 17, "configs/\ndefault.yaml")]
    for x, w, t in orch:
        _box(ax, x, 59, w, 7.5, t, fc="#e8e0f2", ec="#7d5ba6", fs=6.6)
    for x in (21, 55, 83):
        _arrow(ax, x, 72, x, 66.7, color="#999")
    ax.add_patch(Rectangle((1.2, 57.6), 97.6, 10.4, fc="none", ec="#7d5ba6", lw=0.8, ls=":", zorder=1))
    mods = [("datasets.py\npreprocess.py\n\nimage (digits)\nsignal (6 waveforms)\ntabular (wine,\nbreast cancer)", C_DATA, "#666"),
            ("features/\n\nimage.py\nsignal.py\ntransforms.py\nselection.py\nbuilder.py", C_FEAT, "#1f5fa8"),
            ("stats/\nanalysis.py\n\nANOVA, Kruskal,\nBH-FDR, eta^2,\nBhattacharyya,\nMahalanobis", C_STAT, "#d9822b"),
            ("classifiers/\n\nscratch.py\nkNN, Bayes, Parzen,\nSoftmax, MLP\nzoo.py\n(+ SVM, RF, GB)", C_MODEL, "#2a9d6f"),
            ("clustering/\n\nscratch.py\nk-means++, EM-GMM,\nfuzzy c-means\nzoo.py\n(+ Ward, DBSCAN...)", C_MODEL, "#2a9d6f"),
            ("evaluation/\n\nmetrics.py\nCV, ECE, McNemar,\nFriedman, ARI...\nplots.py\n(30+ figures)", C_EVAL, "#c0392b")]
    w = 14.6
    for i, (t, c, e) in enumerate(mods):
        x = 2 + i * (w + 1.6)
        _box(ax, x, 27, w, 26, t, fc=c, ec=e, fs=6.1)
        _arrow(ax, x + w / 2, 57.6, x + w / 2, 53.2, color="#999")
    _box(ax, 2, 15, 96, 7.5, "SHARED CONTRACT:  scikit-learn estimator API  (fit / transform / predict / predict_proba)\nPipeline(Standardizer -> model) re-fitted inside every training fold  =>  no test-fold leakage",
         fc="#ffffff", ec="#444", fs=6.8)
    _box(ax, 2, 3, 96, 8, "FOUNDATION:  NumPy  -  SciPy  -  scikit-learn (reference models, splitters)  -  pandas  -  Matplotlib  -  PyYAML  -  joblib  -  Pillow  -  Streamlit",
         fc="#ececec", ec="#444", fs=6.8)
    _arrow(ax, 50, 27, 50, 22.7, color="#999")
    _arrow(ax, 50, 15, 50, 11.2, color="#999")
    return save(fig, path)


def diagram_operational(path):
    fig, ax = _canvas(7.6, 7.2, 100, 100)
    ax.text(50, 97, "Operational diagram: execution of one benchmark run", ha="center", fontsize=10, fontweight="bold")
    cx, w = 40, 52
    steps = [(88, 6, "START  -  read configs/default.yaml, fix random seed", C_DATA, "#666"),
             (77, 7, "Load dataset  (image | signal | tabular)\nclass-balance check, subsampling if --quick", C_DATA, "#666"),
             (65, 8, "Pre-process and build feature sets  F1 ... Fm\nraw, zoning, HOG, LBP, GLCM, Hu, time, spectral ...", C_FEAT, "#1f5fa8"),
             (52, 9, "Statistical analysis of every feature set\nnormality, ANOVA / Kruskal + BH-FDR, correlation,\nBhattacharyya separability, Mahalanobis outliers", C_STAT, "#d9822b")]
    for y, h, t, c, e in steps:
        _box(ax, cx - w / 2, y, w, h, t, fc=c, ec=e, fs=6.8)
    for (y, h, *_), (y2, h2, *_) in zip(steps[:-1], steps[1:]):
        _arrow(ax, cx, y, cx, y2 + h2 + 0.05)
    # decision
    ax.add_patch(plt.Polygon([[cx, 49], [cx + 12, 44], [cx, 39], [cx - 12, 44]], closed=True, fc="#fff", ec="#444", lw=1.1, zorder=2))
    ax.text(cx, 44, "labels\nused?", ha="center", va="center", fontsize=6.8, zorder=3)
    _arrow(ax, cx, 52, cx, 49.05)
    _box(ax, 1, 20, 44, 14, "SUPERVISED branch\nfor every (feature set, classifier):\nrepeated stratified 5 x 2 CV\ninside each training fold: fit Standardizer\n(+ PCA / selector) -> fit classifier;\nscore the held-out fold", fc=C_MODEL, ec="#2a9d6f", fs=6.5)
    _box(ax, 51, 20, 48, 14, "UNSUPERVISED branch\nStandardizer -> PCA(30) -> fit 10 clusterers\nchoose k: elbow, silhouette, BIC, gap statistic\nlabels are NOT used for fitting;\nonly for external validation (ARI, NMI, purity)", fc=C_MODEL, ec="#2a9d6f", fs=6.5)
    _arrow(ax, cx - 12, 44, 23, 34.1)
    _arrow(ax, cx + 12, 44, 75, 34.1)
    ax.text(14, 38.8, "yes", fontsize=6.5, style="italic")
    ax.text(62, 38.8, "no (clustering)", fontsize=6.5, style="italic")
    _box(ax, 8, 4, 70, 9, "PERFORMANCE EVALUATION\naccuracy, macro-F1, MCC, AUC, ECE + bootstrap 95% CI;  McNemar, corrected resampled t-test,\nFriedman + Nemenyi;  ARI, NMI, V-measure, silhouette, Davies-Bouldin", fc=C_EVAL, ec="#c0392b", fs=6.4)
    _arrow(ax, 23, 20, 30, 13.1)
    _arrow(ax, 75, 20, 62, 13.1)
    _box(ax, 83, 74, 16, 19, "Stress tests\n(side loop)\n\nsweeps, ablation,\nlearning curves,\nnoise, occlusion,\nrotation, shift", fc="#fff", ec="#8c6d31", fs=5.8, ls="--")
    _arrow(ax, cx + w / 2, 69, 83, 80, color="#8c6d31", ls="--", style="<|-|>")
    ax.text(79, 66, "re-uses the\nfeature sets", fontsize=5.6, style="italic", color="#8c6d31", ha="center")
    _box(ax, 81, 4, 18, 14, "OUTPUTS\nCSV tables\nJSON summary\nPNG figures\nmodel.joblib", fc=C_IO, ec="#8c6d31", fs=6.3)
    _arrow(ax, 78, 8.5, 81, 8.5)
    return save(fig, path)


def diagram_cv_protocol(path):
    fig, ax = _canvas(7.6, 3.5, 100, 46)
    ax.text(50, 43, "Leakage-free repeated stratified K-fold protocol (K = 5, 2 repeats)", ha="center", fontsize=10, fontweight="bold")
    for r in range(5):
        y = 33 - r * 6.4
        ax.text(1, y + 2.2, f"fold {r + 1}", fontsize=6.8, va="center")
        for c in range(5):
            test = c == r
            _box(ax, 9 + c * 10.5, y, 10, 4.8, "TEST" if test else "train", fc="#f5b878" if test else "#c9dcf2",
                 ec="#c0702b" if test else "#1f5fa8", fs=6.2, bold=test)
    _box(ax, 64, 10, 35, 23, "In every fold\n\n1. fit Standardizer on the train cells\n2. (fit PCA / selector on train cells)\n3. fit classifier on train cells\n4. transform + predict the TEST cell\n5. store accuracy, F1, MCC, AUC, ECE\n\nThe whole protocol is repeated with a\nsecond random stratified partition\n=> 10 scores per (features, model) cell",
         fc="#fff", ec="#444", fs=6.4)
    ax.text(9, 1.6, "Out-of-fold predictions of repeat 1 cover every sample exactly once -> confusion matrices, ROC, calibration, McNemar.", fontsize=6.4, style="italic")
    return save(fig, path)


def diagram_feature_pipeline(path):
    fig, ax = _canvas(7.6, 4.0, 100, 62)
    ax.set_ylim(8, 62)
    ax.text(50, 59, "Feature-extraction and transformation pipeline", ha="center", fontsize=10, fontweight="bold")
    rows = [("IMAGE  16 x 16", 42, "#dbe8f7", "raw | zoning | profiles | stats\nHOG | LBP (u2, 2x2 grid) | GLCM | Hu"),
            ("SIGNAL  256 samples", 27, "#d8f0e4", "time-domain | Hjorth\nspectral (FFT) | auto-correlation"),
            ("TABULAR  d features", 12, "#fbe9d0", "used as-is\n(already feature vectors)")]
    for name, y, c, desc in rows:
        _box(ax, 1.5, y, 17, 10, name, fc="#e9e9e9", ec="#666", fs=6.8, bold=True)
        _box(ax, 25, y, 30, 10, desc, fc=c, ec="#555", fs=6.4)
        _arrow(ax, 18.5, y + 5, 25, y + 5)
        _arrow(ax, 55, y + 5, 60, 31 if y != 27 else 31)
    ax.text(40, 54.2, "descriptor groups (concatenated)", fontsize=6.4, style="italic", ha="center", color="#444")
    _box(ax, 60, 22, 15, 18, "feature\nmatrix\nF  (n x m)\n\nnamed columns,\ngroup index", fc="#fff", ec="#444", fs=6.4)
    _box(ax, 80, 40, 18, 9, "Standardizer\n(fit on train only)", fc=C_FEAT, ec="#1f5fa8", fs=6.4)
    _box(ax, 80, 27, 18, 9, "PCA | Fisher-LDA\nFisher / MI / mRMR", fc=C_FEAT, ec="#1f5fa8", fs=6.4)
    _box(ax, 80, 12, 18, 11, "classifier\nor clusterer", fc=C_MODEL, ec="#2a9d6f", fs=6.6, bold=True)
    _arrow(ax, 75, 34, 80, 44.5)
    _arrow(ax, 89, 40, 89, 36.1)
    _arrow(ax, 89, 27, 89, 23.1)
    return save(fig, path)


def diagram_class_structure(path):
    """Class hierarchy of the package: every algorithm honours the scikit-learn estimator contract."""
    fig, ax = _canvas(7.6, 5.2, 100, 70)
    ax.text(50, 67, "Class structure: the scikit-learn estimator contract is the common interface", ha="center", fontsize=10, fontweight="bold")
    _box(ax, 36, 55, 28, 7, "sklearn.base.BaseEstimator\nget_params / set_params / clone", fc="#ececec", ec="#444", fs=6.8, bold=True)
    mix = [(3, "TransformerMixin\nfit_transform", C_FEAT, "#1f5fa8"), (37, "ClassifierMixin\nscore (accuracy)", C_MODEL, "#2a9d6f"),
           (71, "ClusterMixin\nfit_predict", C_MODEL, "#2a9d6f")]
    for x, t, c, e in mix:
        _box(ax, x, 41, 26, 7, t, fc=c, ec=e, fs=6.6, bold=True)
        _arrow(ax, x + 13, 48, 50 if x == 37 else (42 if x < 37 else 58), 55, color="#888", style="-|>")
    groups = [(3, 26, C_FEAT, "#1f5fa8", "Standardizer\nPCAScratch\nLDAProjection\nFisherSelector\nMISelector\nMRMRSelector\nCorrelationPruner"),
              (37, 26, C_MODEL, "#2a9d6f", "KNNClassifier\nGaussianBayes (qda|lda|naive)\nMinimumDistance\nParzenClassifier\nSoftmaxRegression  (Adam)\nNumpyMLP  (back-prop, dropout)"),
              (71, 26, C_MODEL, "#2a9d6f", "KMeansScratch  (k-means++)\nGMMScratch  (EM; full|diag|\n   spherical)\nFuzzyCMeans")]
    for x, w, c, e, t in groups:
        _box(ax, x, 6, w, 28, t, fc=c, ec=e, fs=6.3)
        _arrow(ax, x + w / 2, 41, x + w / 2, 34.1, color="#888")
    ax.text(50, 2.2, "Because all components are estimators they compose in sklearn.pipeline.Pipeline and can be cloned, cross-validated and compared with reference models.",
            ha="center", fontsize=6.2, style="italic", color="#444")
    return save(fig, path)
