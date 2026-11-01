"""The full experimental protocol (every table and figure of the project report).

Usage
-----
    python scripts/run_experiments.py --config configs/default.yaml --out results
    python scripts/run_experiments.py --quick          # ~1-2 minute smoke run

Each ``exp_*`` method is independent and writes CSV / JSON tables to
``<out>/tables`` and PNG figures to ``<out>/figures``.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.base import clone
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from . import datasets as D
from . import preprocess as P
from .classifiers import KNNClassifier, GaussianBayes, ParzenClassifier, NumpyMLP, SoftmaxRegression, get_classifiers
from .clustering import (FuzzyCMeans, GMMScratch, KMeansScratch, estimate_eps, gap_statistic, get_clusterers, sweep_k)
from .evaluation import plots as pl
from .evaluation.metrics import (bootstrap_ci, bootstrap_stability, clustering_scores, corrected_resampled_ttest,
                                 cross_validate_model, friedman_nemenyi, mcnemar_test, multiclass_scores)
from .features import (FisherSelector, LDAProjection, MISelector, MRMRSelector, PCAScratch, Standardizer,
                       build_features)
from .features.builder import FEATURE_SETS
from .features.image import hog_cells, lbp_codes, _lbp_lookup
from .stats import (anova_table, bhattacharyya_matrix, class_separability, correlation_summary, descriptive_table,
                    mahalanobis_outliers, normality_tests, pairwise_class_ttests)

warnings.filterwarnings("ignore")


def _fit_eval(estimator, X, y, n_splits, n_repeats, seed):
    return cross_validate_model(estimator, X, y, n_splits, n_repeats, seed)


class Workbench:
    def __init__(self, cfg: dict, out: str = "results", quick: bool = False, verbose: bool = True):
        self.cfg, self.out, self.quick, self.verbose = cfg, out, quick, verbose
        self.seed = cfg.get("seed", 42)
        self.fig_dir = os.path.join(out, "figures")
        self.tab_dir = os.path.join(out, "tables")
        for d in (self.fig_dir, self.tab_dir, os.path.join(out, "json")):
            os.makedirs(d, exist_ok=True)
        self._ds: Dict[str, D.Dataset] = {}
        self._feat: Dict[tuple, object] = {}
        self.cache: Dict[str, object] = {}
        self.summary: Dict[str, object] = {}
        self.cv_splits = cfg["cv"]["n_splits"]
        self.cv_repeats = 1 if quick else cfg["cv"]["n_repeats"]
        self.n_jobs = cfg.get("n_jobs", 2)
        self.t0 = time.time()

    # ------------------------------------------------------------------ utils
    def log(self, msg):
        if self.verbose:
            print(f"[{time.time() - self.t0:7.1f}s] {msg}", flush=True)

    def fig(self, name):
        return os.path.join(self.fig_dir, name)

    def save_table(self, df: pd.DataFrame, name: str, index: bool = False):
        df.to_csv(os.path.join(self.tab_dir, name), index=index)

    def dataset(self, name: str) -> D.Dataset:
        if name not in self._ds:
            if name == "signals":
                ds = D.make_signal_dataset(n_per_class=40 if self.quick else 200, seed=self.seed)
            else:
                ds = D.load_dataset(name)
                if self.quick and name == "digits":
                    ds = ds.subsample(60, self.seed)
            self._ds[name] = ds
        return self._ds[name]

    def feat(self, name: str, set_name: str):
        key = (name, set_name)
        if key not in self._feat:
            self._feat[key] = build_features(self.dataset(name), set_name)
        return self._feat[key]

    def main_set(self, name):
        return self.cfg["datasets"][name]["main"]

    # --------------------------------------------------------------- overview
    def exp_overview(self):
        self.log("overview")
        rows = []
        counts = {}
        for name in self.cfg["datasets"]:
            ds = self.dataset(name)
            counts[name] = ds.class_counts()
            rows.append({"dataset": name, "modality": ds.modality, "samples": ds.n_samples,
                         "pattern_shape": "x".join(map(str, ds.X.shape[1:])), "classes": ds.n_classes,
                         "min_class": int(ds.class_counts().min()), "max_class": int(ds.class_counts().max()),
                         "raw_dims": int(np.prod(ds.X.shape[1:]))})
        self.save_table(pd.DataFrame(rows), "dataset_overview.csv")
        dg, sg = self.dataset("digits"), self.dataset("signals")
        pl.plot_image_grid(dg.X, dg.y, dg.class_names, 10, self.fig("digits_samples.png"))
        pl.plot_class_balance(counts, self.fig("class_balance.png"))
        pl.plot_signal_examples(sg.X, sg.y, sg.class_names, sg.meta["fs"], self.fig("signal_examples.png"))
        pl.diagram_high_level(self.fig("diagram_high_level.png"))
        pl.diagram_architecture(self.fig("diagram_architecture.png"))
        pl.diagram_operational(self.fig("diagram_operational.png"))
        pl.diagram_class_structure(self.fig("diagram_classes.png"))
        pl.diagram_cv_protocol(self.fig("diagram_cv_protocol.png"))
        pl.diagram_feature_pipeline(self.fig("diagram_feature_pipeline.png"))

    # --------------------------------------------------------- feature visuals
    def exp_feature_visuals(self):
        self.log("feature visuals")
        ds = self.dataset("digits")
        i = int(np.flatnonzero(ds.y == 8)[3])
        img = ds.X[i]
        cells = hog_cells(img[None], 4, 9)[0]
        pl.plot_hog_visual(img, cells, self.fig("hog_visual.png"))
        codes = lbp_codes(img[None])[0]
        lut, nb = _lbp_lookup(8, "u2")
        pl.plot_lbp_visual(img, codes, lut[codes], nb, self.fig("lbp_visual.png"))
        # PCA on raw pixels: scree + reconstructions
        Xr = ds.X.reshape(len(ds.X), -1)
        pca = PCAScratch(0.999).fit(Xr)
        ks = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120]
        errs = [PCAScratch(k).fit(Xr).reconstruction_error(Xr) for k in ks if k <= min(Xr.shape)]
        pl.plot_pca_scree(pca.all_explained_variance_ratio_, self.fig("pca_scree.png"), 40,
                          recon=(ks[:len(errs)], errs))
        idx = [int(np.flatnonzero(ds.y == c)[0]) for c in range(10)]
        recons = {k: PCAScratch(k).fit(Xr).inverse_transform(PCAScratch(k).fit(Xr).transform(Xr[idx])).reshape(-1, *ds.X.shape[1:])
                  for k in (5, 15, 40)}
        pl.plot_reconstructions(ds.X[idx], recons, self.fig("pca_reconstruction.png"))
        self.summary["pca_raw_components_95"] = int(PCAScratch(0.95).fit(Xr).n_components_)
        self.summary["pca_raw_components_99"] = int(PCAScratch(0.99).fit(Xr).n_components_)
        # embeddings of the main feature space
        fm = self.feat("digits", self.main_set("digits"))
        Z = Standardizer().fit_transform(fm.X)
        Z30 = PCAScratch(30).fit_transform(Z)
        pca2 = Z30[:, :2]
        lda = LDAProjection(2).fit(Z30, ds.y).transform(Z30)
        tsne = TSNE(2, init="pca", perplexity=30, random_state=self.seed).fit_transform(Z30)
        self.cache["tsne_digits"] = tsne
        pl.plot_embeddings({"PCA (unsupervised, 2-D)": pca2, "Fisher LDA (supervised, 2-D)": lda, "t-SNE (non-linear)": tsne},
                           ds.y, ds.class_names, self.fig("embeddings_digits.png"))
        # glcm distributions, signal distributions
        fg = self.feat("digits", "glcm")
        pl.plot_feature_distributions(fg.X, ds.y, fg.names[:12], ds.class_names, self.fig("glcm_distributions.png"), ncols=4)
        sg = self.dataset("signals")
        fs_ = self.feat("signals", "handcrafted")
        an = anova_table(fs_.X, sg.y, fs_.names)
        top = list(an["feature"].head(8))
        cols = [fs_.names.index(t) for t in top]
        pl.plot_feature_distributions(fs_.X[:, cols], sg.y, top, sg.class_names, self.fig("signal_feature_distributions.png"), ncols=4)

    # ----------------------------------------------------------------- stats
    def exp_stats(self):
        self.log("statistical analysis")
        summ = {}
        for name in self.cfg["datasets"]:
            ds = self.dataset(name)
            sets = ["handcrafted"] if name in ("digits", "signals") else ["raw"]
            for s in sets:
                fm = self.feat(name, s)
                tag = f"{name}_{s}".replace("+", "_")
                desc = descriptive_table(fm.X, fm.names)
                self.save_table(desc.head(60), f"descriptive_{tag}.csv")
                norm = normality_tests(fm.X, fm.names)
                self.save_table(norm, f"normality_{tag}.csv")
                an = anova_table(fm.X, ds.y, fm.names)
                self.save_table(an, f"anova_{tag}.csv")
                corr = correlation_summary(fm.X, fm.names)
                C = corr["corr"]
                if C.shape[0] > 40:
                    order = leaves_list(linkage(np.nan_to_num(C), "average"))
                    C = C[np.ix_(order, order)]
                pl.plot_correlation(C, self.fig(f"correlation_{tag}.png"), f"{name}: |feature correlation| ({s})")
                pl.plot_anova_top(an, self.fig(f"anova_{tag}.png"), 20 if name != "digits" else 25)
                # separability in a decorrelated space
                Zs = Standardizer().fit_transform(fm.X)
                Zp = PCAScratch(min(20, Zs.shape[1] - 1)).fit_transform(Zs) if Zs.shape[1] > 21 else Zs
                B = bhattacharyya_matrix(Zp, ds.y)
                pl.plot_heatmap(B, ds.class_names, ds.class_names, self.fig(f"bhattacharyya_{tag}.png"),
                                f"{name}: pairwise Bhattacharyya distance", cmap="magma", fmt="{:.1f}" if name == "digits" else "{:.1f}",
                                annotate=ds.n_classes <= 10, figsize=(5.2, 4.4), cbar_label="D_B")
                d2, thr, flag = mahalanobis_outliers(Zp)
                pl.plot_hist_with_threshold(d2, thr, self.fig(f"mahalanobis_{tag}.png"), title=f"{name}: Mahalanobis outliers")
                valid = an.dropna(subset=["anova_q_BH"])
                summ[tag] = {
                    "n_features": int(fm.X.shape[1]), "n_constant": corr["n_constant"],
                    "frac_normal_shapiro": float(norm["normal_at_5pct"].mean()),
                    "n_sig_bh_5pct": int((valid["anova_q_BH"] < 0.05).sum()), "n_tested": int(len(valid)),
                    "n_sig_bonferroni_5pct": int((valid["bonferroni_p"] < 0.05).sum()),
                    "frac_levene_het": float((valid["levene_p"] < 0.05).mean()),
                    "median_eta2": float(valid["eta_squared"].median()), "max_eta2": float(valid["eta_squared"].max()),
                    "top_feature": str(an.iloc[0]["feature"]), "top_F": float(an.iloc[0]["anova_F"]),
                    "mean_abs_corr": corr["mean_abs_corr"], "pairs_corr_gt_0.9": corr["n_pairs_above_threshold"],
                    "condition_number": corr["condition_number"], "effective_rank": corr["effective_rank"],
                    "n_mahalanobis_outliers": int(flag.sum()), "mahalanobis_threshold": float(thr),
                    "bhattacharyya_min_pair": float(B[np.triu_indices_from(B, 1)].min()),
                    "bhattacharyya_max_pair": float(B[np.triu_indices_from(B, 1)].max()),
                }
                iu = np.triu_indices_from(B, 1)
                k = int(np.argmin(B[iu]))
                summ[tag]["hardest_pair"] = f"{ds.class_names[iu[0][k]]} vs {ds.class_names[iu[1][k]]}"
        # which descriptor family is most discriminative?
        fm = self.feat("digits", "handcrafted")
        an = anova_table(fm.X, self.dataset("digits").y, fm.names).set_index("feature")
        fam = pd.DataFrame({"group": [fm.groups[g] for g in fm.group_index], "eta2": an.loc[fm.names, "eta_squared"].values,
                            "F": an.loc[fm.names, "anova_F"].values})
        fam_t = fam.groupby("group").agg(n_features=("eta2", "size"), mean_eta2=("eta2", "mean"), max_eta2=("eta2", "max"),
                                         median_F=("F", "median")).sort_values("mean_eta2", ascending=False).reset_index()
        self.save_table(fam_t, "descriptor_family_effect_sizes.csv")
        # separability J per feature set (digits & signals)
        rows = []
        for name in ("digits", "signals"):
            for s in self.cfg["datasets"][name]["feature_sets"]:
                fm = self.feat(name, s)
                Zs = Standardizer().fit_transform(fm.X)
                Zp = PCAScratch(min(30, Zs.shape[1] - 1)).fit_transform(Zs) if Zs.shape[1] > 31 else Zs
                sep = class_separability(Zp, self.dataset(name).y)
                rows.append({"dataset": name, "feature_set": s, "dims": fm.X.shape[1], **sep})
        self.save_table(pd.DataFrame(rows), "separability_by_featureset.csv")
        # pairwise tests for the best signal feature
        sg = self.dataset("signals")
        fs_ = self.feat("signals", "handcrafted")
        an = anova_table(fs_.X, sg.y, fs_.names)
        best = an.iloc[0]["feature"]
        pw = pairwise_class_ttests(fs_.X[:, fs_.names.index(best)], sg.y, sg.class_names)
        pw.insert(0, "feature", best)
        self.save_table(pw, "pairwise_ttests_signals_best_feature.csv")
        self.summary["stats"] = summ

    # ------------------------------------------------- classification benchmark
    def _clf_names_for(self, dim: int):
        names = list(get_classifiers().keys())
        if dim > self.cfg.get("histgb_max_dim", 100) and "HistGB" in names:
            names.remove("HistGB")
        if dim > 300 and "ExtraTrees" in names and self.quick:
            names.remove("ExtraTrees")
        return names

    def exp_classification(self):
        self.log("classification grid")
        long_rows, fold_rows, oof = [], [], {}
        for name in self.cfg["datasets"]:
            ds = self.dataset(name)
            for s in self.cfg["datasets"][name]["feature_sets"]:
                fm = self.feat(name, s)
                names = self._clf_names_for(fm.X.shape[1])
                zoo = get_classifiers(self.seed, names=names)
                t0 = time.time()
                res = Parallel(n_jobs=self.n_jobs)(
                    delayed(_fit_eval)(zoo[n], fm.X, ds.y, self.cv_splits, self.cv_repeats, self.seed) for n in names)
                for n, r in zip(names, res):
                    f = r["folds"]
                    lo, hi = bootstrap_ci(f["accuracy"].values)
                    row = {"dataset": name, "feature_set": s, "dims": fm.X.shape[1], "classifier": n,
                           "fit_time": r["fit_time"], "predict_time": r["predict_time"], "acc_ci_low": lo, "acc_ci_high": hi}
                    for m in f.columns:
                        if m != "fold":
                            row[m] = f[m].mean()
                            row[m + "_std"] = f[m].std()
                    long_rows.append(row)
                    ff = f.copy()
                    ff.insert(0, "classifier", n)
                    ff.insert(0, "feature_set", s)
                    ff.insert(0, "dataset", name)
                    fold_rows.append(ff)
                    if s == self.main_set(name):
                        oof[(name, n)] = (r["oof_pred"], r["oof_proba"])
                best = max(zip(names, res), key=lambda t: t[1]["folds"]["accuracy"].mean())
                self.log(f"  {name:13s} {s:12s} d={fm.X.shape[1]:4d}  best={best[0]} acc={best[1]['folds']['accuracy'].mean():.4f}  ({time.time()-t0:.0f}s)")
        res = pd.DataFrame(long_rows)
        self.save_table(res, "classification_results.csv")
        self.save_table(pd.concat(fold_rows), "classification_folds.csv")
        self.cache["oof"] = oof
        self.cache["cls_res"] = res
        self.cache["cls_folds"] = pd.concat(fold_rows)

    def exp_classification_figures(self):
        self.log("classification figures")
        res, folds, oof = self.cache["cls_res"], self.cache["cls_folds"], self.cache["oof"]
        # heatmaps (accuracy) for digits and signals
        for name in ("digits", "signals"):
            sub = res[res.dataset == name]
            piv = sub.pivot(index="classifier", columns="feature_set", values="accuracy")
            cols = self.cfg["datasets"][name]["feature_sets"]
            piv = piv[cols]
            piv = piv.loc[piv.mean(1).sort_values(ascending=False).index]
            pl.plot_heatmap(piv.values * 100, cols, piv.index, self.fig(f"heatmap_accuracy_{name}.png"),
                            f"{name}: CV accuracy (%) by classifier and feature set", cmap="YlGnBu", fmt="{:.1f}",
                            figsize=(6.6, 5.2), cbar_label="accuracy (%)", vmin=max(np.nanmin(piv.values * 100), 30), vmax=100)
            self.save_table(piv.round(4).reset_index(), f"accuracy_matrix_{name}.csv")
        # tabular bar tables
        for name in ("wine", "breast_cancer"):
            sub = res[res.dataset == name].sort_values("accuracy", ascending=False)
            self.save_table(sub[["classifier", "accuracy", "acc_ci_low", "acc_ci_high", "macro_f1", "roc_auc", "mcc", "fit_time"]], f"results_{name}.csv")
        # boxplots on main feature sets
        for name in self.cfg["datasets"]:
            s = self.main_set(name)
            f = folds[(folds.dataset == name) & (folds.feature_set == s)]
            pl.plot_cv_boxplot({c: g["accuracy"].values for c, g in f.groupby("classifier")}, self.fig(f"cv_boxplot_{name}.png"),
                               "accuracy", f"{name} ({s}): fold-wise accuracy over {self.cv_splits}x{self.cv_repeats} CV")
        # Friedman / Nemenyi over all (dataset, feature set) blocks
        piv = res.pivot_table(index=["dataset", "feature_set"], columns="classifier", values="accuracy").dropna(axis=1)
        fr = friedman_nemenyi(piv)
        pl.plot_critical_difference(fr["avg_ranks"], fr["cd"], self.fig("critical_difference.png"),
                                    f"Friedman chi2={fr['chi2']:.1f} (p={fr['p']:.1e}), {fr['n_blocks']} blocks")
        self.save_table(fr["avg_ranks"].rename("avg_rank").reset_index(), "friedman_avg_ranks.csv")
        self.summary["friedman"] = {k: (float(v) if not isinstance(v, pd.Series) else v.to_dict()) for k, v in fr.items() if k not in ("n_blocks", "k")}
        self.summary["friedman"]["n_blocks"] = int(fr["n_blocks"])
        self.summary["friedman"]["n_classifiers"] = int(fr["k"])
        # time / accuracy scatter
        sub = res[(res.dataset == "digits") & (res.feature_set == self.main_set("digits"))][["classifier", "fit_time", "accuracy"]]
        pl.plot_scatter_time_acc(sub, self.fig("time_vs_accuracy.png"))
        # best model diagnostics on digits (main)
        name, s = "digits", self.main_set("digits")
        ds = self.dataset(name)
        sub = res[(res.dataset == name) & (res.feature_set == s)].sort_values("accuracy", ascending=False)
        best, second, third = list(sub["classifier"][:3])
        yb, pb = ds.y, oof[(name, best)][1]
        pl.plot_confusion(ds.y, oof[(name, best)][0], ds.class_names, self.fig("confusion_digits_best.png"), f"{best} on {s}")
        pl.plot_roc_multiclass(ds.y, pb, ds.class_names, self.fig("roc_digits_best.png"), f"({best})")
        cal = {n: oof[(name, n)][1] for n in [best, "MLP", "Bayes-QDA", "kNN"] if (name, n) in oof and oof[(name, n)][1] is not None}
        pl.plot_calibration(ds.y, cal, self.fig("calibration_digits.png"))
        weak = "DecisionTree"
        pl.plot_confusion(ds.y, oof[(name, weak)][0], ds.class_names, self.fig("confusion_digits_weak.png"), f"{weak} on {s}")
        # signals confusion
        sgs = self.main_set("signals")
        sg = self.dataset("signals")
        sbest = res[(res.dataset == "signals") & (res.feature_set == sgs)].sort_values("accuracy", ascending=False).iloc[0]["classifier"]
        pl.plot_confusion(sg.y, oof[("signals", sbest)][0], sg.class_names, self.fig("confusion_signals_best.png"), f"{sbest} on signals/{sgs}")
        # per-class report of best digits model
        from sklearn.metrics import classification_report
        rep = pd.DataFrame(classification_report(ds.y, oof[(name, best)][0], target_names=ds.class_names, output_dict=True)).T
        self.save_table(rep.round(4).reset_index().rename(columns={"index": "class"}), "per_class_report_digits_best.csv")
        # significance tests between top classifiers
        rows = []
        f = folds[(folds.dataset == name) & (folds.feature_set == s)]
        n = ds.n_samples
        n_te = n // self.cv_splits
        for a, b in [(best, second), (best, third), (best, "MLP"), (best, "Bayes-QDA"), (best, "DecisionTree")]:
            if a == b:
                continue
            mc = mcnemar_test(ds.y, oof[(name, a)][0], oof[(name, b)][0])
            ct = corrected_resampled_ttest(f[f.classifier == a]["accuracy"].values, f[f.classifier == b]["accuracy"].values, n - n_te, n_te)
            rows.append({"A": a, "B": b, "acc_A": f[f.classifier == a]["accuracy"].mean(), "acc_B": f[f.classifier == b]["accuracy"].mean(),
                         "McNemar_n01(A right,B wrong)": mc["n01"], "McNemar_n10(A wrong,B right)": mc["n10"], "McNemar_p": mc["p"],
                         "corrected_t": ct["t"], "corrected_t_p": ct["p"]})
        self.save_table(pd.DataFrame(rows), "significance_tests_digits.csv")
        self.summary["best_digits"] = {"classifier": best, "feature_set": s,
                                       "accuracy": float(sub.iloc[0]["accuracy"]), "ci": [float(sub.iloc[0]["acc_ci_low"]), float(sub.iloc[0]["acc_ci_high"])],
                                       "macro_f1": float(sub.iloc[0]["macro_f1"]), "mcc": float(sub.iloc[0]["mcc"]),
                                       "roc_auc": float(sub.iloc[0]["roc_auc"]), "ece": float(sub.iloc[0]["ece"]), "second": second, "third": third}
        # headline table: best per dataset/featureset
        top = res.sort_values("accuracy", ascending=False).groupby(["dataset", "feature_set"]).head(1)
        self.save_table(top[["dataset", "feature_set", "dims", "classifier", "accuracy", "macro_f1", "mcc"]].sort_values(["dataset", "accuracy"], ascending=[True, False]), "best_per_block.csv")

    # ------------------------------------------------------------- sweeps
    def exp_sweeps(self):
        self.log("hyper-parameter sweeps, learning curves, selection")
        ds = self.dataset("digits")
        y = ds.y
        fm_h = self.feat("digits", "hog")
        fm_m = self.feat("digits", "hog+lbp")
        skf = StratifiedKFold(self.cv_splits, shuffle=True, random_state=self.seed)

        def cv_acc(model, X, folds=None):
            accs = []
            for tr, te in skf.split(X, y):
                m = clone(model).fit(X[tr], y[tr])
                accs.append((m.predict(X[te]) == y[te]).mean())
            return float(np.mean(accs)), float(np.std(accs))

        def scaled(est):
            return Pipeline([("s", Standardizer()), ("c", est)])

        # k-NN: k x metric
        ks = [1, 3, 5, 7, 9, 15, 25]
        metrics = ["euclidean", "manhattan", "cosine"] + (["mahalanobis"] if not self.quick else [])
        series = {m: [cv_acc(scaled(KNNClassifier(k, m)), fm_h.X)[0] for k in ks] for m in metrics}
        pl.plot_line_family(ks, series, self.fig("sweep_knn.png"), "k (neighbours)", "CV accuracy", "k-NN on HOG features", ylim=(0.9, 1.0))
        self.save_table(pd.DataFrame(series, index=ks).rename_axis("k").reset_index(), "sweep_knn.csv")
        # SVM C x gamma
        d = fm_m.X.shape[1]
        Cs = [0.1, 1, 10, 100, 1000]
        gs = [0.03, 0.1, 0.3, 1.0, 3.0]
        M = np.array([[cv_acc(scaled(SVC(C=C, gamma=g / d)), fm_m.X)[0] for g in gs] for C in Cs])
        pl.plot_heatmap(M * 100, [f"{g}/d" for g in gs], [str(c) for c in Cs], self.fig("sweep_svm.png"), "RBF-SVM accuracy (%): C vs gamma (HOG+LBP)",
                        cmap="YlGnBu", fmt="{:.1f}", figsize=(5.0, 3.8), cbar_label="accuracy (%)", vmin=max(M.min() * 100, 50))
        self.save_table(pd.DataFrame(M, index=[f"C={c}" for c in Cs], columns=[f"gamma={g}/d" for g in gs]).reset_index(), "sweep_svm.csv")
        # Parzen bandwidth
        scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0]
        pz = [cv_acc(scaled(ParzenClassifier(bw_scale=s)), fm_m.X)[0] for s in scales]
        pl.plot_line_family(scales, {"Parzen window": pz}, self.fig("sweep_parzen.png"), "bandwidth scale (x mean 1-NN distance)", "CV accuracy",
                            "Parzen-window bandwidth", ylim=(0.5, 1.0))
        # regularisation of Bayes models
        regs = [0.0, 0.01, 0.05, 0.1, 0.3, 0.6, 0.9]
        bs = {m: [cv_acc(scaled(GaussianBayes(m, reg=r)), fm_m.X)[0] for r in regs] for m in ("qda", "lda", "naive")}
        pl.plot_line_family(regs, bs, self.fig("sweep_bayes_reg.png"), "covariance shrinkage", "CV accuracy", "Gaussian Bayes: shrinkage", ylim=(0.5, 1.0))
        self.save_table(pd.DataFrame(bs, index=regs).rename_axis("shrinkage").reset_index(), "sweep_bayes_reg.csv")
        # MLP training curves + architecture sweep
        tr, te = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=self.seed)
        sc = Standardizer().fit(fm_m.X[tr])
        mlp = NumpyMLP((64, 32), epochs=100, dropout=0.1, random_state=self.seed).fit(sc.transform(fm_m.X[tr]), y[tr])
        pl.plot_training_curves({"train": mlp.train_loss_, "validation": mlp.val_loss_}, self.fig("mlp_loss.png"), "cross-entropy", "NumPy MLP (64-32)")
        sm = SoftmaxRegression(epochs=60, random_state=self.seed).fit(sc.transform(fm_m.X[tr]), y[tr])
        self.summary["mlp_epochs_run"] = int(mlp.n_epochs_)
        arch = {"(16)": (16,), "(64)": (64,), "(128)": (128,), "(64,32)": (64, 32), "(128,64)": (128, 64), "(128,64,32)": (128, 64, 32)}
        rows = []
        for nm, h in arch.items():
            m = NumpyMLP(h, epochs=60, dropout=0.1, random_state=self.seed).fit(sc.transform(fm_m.X[tr]), y[tr])
            nparam = sum(W.size + b.size for W, b in zip(m.Ws_, m.bs_))
            rows.append({"hidden": nm, "params": nparam, "test_acc": float((m.predict(sc.transform(fm_m.X[te])) == y[te]).mean()), "epochs": m.n_epochs_})
        self.save_table(pd.DataFrame(rows), "sweep_mlp_architecture.csv")
        # learning curves
        fracs = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
        lc_models = {"kNN": KNNClassifier(5, weights="distance"), "Bayes-LDA": GaussianBayes("lda", reg=0.1), "SVM-RBF": SVC(C=10),
                     "MLP": NumpyMLP((64, 32), epochs=60, dropout=0.1, random_state=self.seed), "MinDist": get_classifiers(names=["MinDist"])["MinDist"].steps[-1][1]}
        lc, lc_sd = {}, {}
        for nm, est in lc_models.items():
            accs = []
            for fr in fracs:
                a = []
                for tr_i, te_i in skf.split(fm_m.X, y):
                    rng = np.random.default_rng(self.seed)
                    sub = rng.choice(tr_i, max(int(fr * len(tr_i)), 2 * ds.n_classes), replace=False)
                    m = scaled(clone(est)).fit(fm_m.X[sub], y[sub])
                    a.append((m.predict(fm_m.X[te_i]) == y[te_i]).mean())
                accs.append(a)
            lc[nm] = [float(np.mean(a)) for a in accs]
            lc_sd[nm] = [float(np.std(a)) for a in accs]
        pl.plot_line_family([int(f * len(y) * 0.8) for f in fracs], lc, self.fig("learning_curves.png"), "training samples", "CV accuracy", "Learning curves (HOG+LBP)", errs=lc_sd, ylim=(0.3, 1.0), logx=True)
        self.save_table(pd.DataFrame(lc, index=fracs).rename_axis("train_fraction").reset_index(), "learning_curves.csv")
        # PCA dimension sweep
        dims = [2, 5, 10, 20, 30, 50, 80, 120, 200]
        pc = {}
        for nm, est in {"kNN": KNNClassifier(5, weights="distance"), "SVM-RBF": SVC(C=10), "Bayes-QDA": GaussianBayes("qda", reg=0.3)}.items():
            pc[nm] = [cv_acc(Pipeline([("s", Standardizer()), ("p", PCAScratch(k)), ("c", clone(est))]), fm_m.X)[0] for k in dims]
        pl.plot_line_family(dims, pc, self.fig("sweep_pca_dims.png"), "PCA components kept", "CV accuracy", "Accuracy vs. PCA dimensionality (HOG+LBP)", logx=True, ylim=(0.5, 1.0))
        self.save_table(pd.DataFrame(pc, index=dims).rename_axis("n_components").reset_index(), "sweep_pca_dims.csv")
        # filter feature selection (selection inside every training fold)
        fm_all = self.feat("digits", "handcrafted")
        ksel = [5, 10, 20, 40, 80, 160, 320]
        sel = {"Fisher": FisherSelector, "Mutual information": MISelector, "mRMR": MRMRSelector}
        acc = {n: {k: [] for k in ksel} for n in sel}
        for tr_i, te_i in skf.split(fm_all.X, y):
            sc = Standardizer().fit(fm_all.X[tr_i])
            Xtr, Xte = sc.transform(fm_all.X[tr_i]), sc.transform(fm_all.X[te_i])
            for n, S in sel.items():
                rank_sel = S(k=max(ksel)).fit(Xtr, y[tr_i])
                for k in ksel:
                    cols = np.sort(rank_sel.ranking_[:k])
                    m = SVC(C=10, gamma="scale").fit(Xtr[:, cols], y[tr_i])
                    acc[n][k].append((m.predict(Xte[:, cols]) == y[te_i]).mean())
        series = {n: [float(np.mean(acc[n][k])) for k in ksel] for n in sel}
        pl.plot_line_family(ksel, series, self.fig("sweep_feature_selection.png"), "number of selected features", "CV accuracy (RBF-SVM)",
                            f"Filter selection from {fm_all.X.shape[1]} hand-crafted features", logx=True, ylim=(0.6, 1.0))
        self.save_table(pd.DataFrame(series, index=ksel).rename_axis("k").reset_index(), "sweep_feature_selection.csv")
        # which descriptor families does mRMR choose?
        sel_all = MRMRSelector(k=40).fit(Standardizer().fit_transform(fm_all.X), y)
        fam = pd.Series([fm_all.groups[fm_all.group_index[j]] for j in sel_all.support_]).value_counts()
        self.save_table(fam.rename("selected_features").reset_index().rename(columns={"index": "group"}), "mrmr_selected_groups.csv")

    # ------------------------------------------------------------- ablation
    def exp_ablation(self):
        self.log("descriptor-group ablation")
        ds = self.dataset("digits")
        fm = self.feat("digits", "handcrafted")
        rows = []
        groups = list(range(len(fm.groups)))
        for clf in ("SVM-RBF", "Bayes-LDA", "kNN"):
            model = get_classifiers(self.seed, names=[clf])[clf]
            full = cross_validate_model(model, fm.X, ds.y, self.cv_splits, 1, self.seed, keep_oof=False)["folds"]["accuracy"].mean()
            rows.append({"classifier": clf, "setting": "all groups", "dims": fm.X.shape[1], "accuracy": full})
            for g in groups:
                keep = fm.group_index != g
                a = cross_validate_model(model, fm.X[:, keep], ds.y, self.cv_splits, 1, self.seed, keep_oof=False)["folds"]["accuracy"].mean()
                rows.append({"classifier": clf, "setting": f"without {fm.groups[g]}", "dims": int(keep.sum()), "accuracy": a})
            for g in groups:
                keep = fm.group_index == g
                a = cross_validate_model(model, fm.X[:, keep], ds.y, self.cv_splits, 1, self.seed, keep_oof=False)["folds"]["accuracy"].mean()
                rows.append({"classifier": clf, "setting": f"only {fm.groups[g]}", "dims": int(keep.sum()), "accuracy": a})
        df = pd.DataFrame(rows)
        self.save_table(df, "ablation_descriptor_groups.csv")
        piv = df.pivot(index="setting", columns="classifier", values="accuracy")
        order = ["all groups"] + [f"without {g}" for g in fm.groups] + [f"only {g}" for g in fm.groups]
        piv = piv.loc[order]
        pl.plot_heatmap(piv.values * 100, list(piv.columns), list(piv.index), self.fig("ablation_groups.png"), "Descriptor-group ablation (accuracy %)",
                        cmap="YlGnBu", fmt="{:.1f}", figsize=(4.6, 5.0), cbar_label="accuracy (%)", vmin=max(piv.values.min() * 100, 30))

    # ------------------------------------------------------------ robustness
    def exp_robustness(self):
        self.log("robustness study")
        ds = self.dataset("digits")
        cfg = self.cfg["robustness"]
        n_seeds = 1 if self.quick else cfg["seeds"]
        rows = []
        for sd in range(n_seeds):
            tr, te = train_test_split(np.arange(ds.n_samples), test_size=0.3, stratify=ds.y, random_state=self.seed + sd)
            from .features.image import extract_image_features
            models = {}
            for s in cfg["feature_sets"]:
                groups = FEATURE_SETS["image"][s]
                Ftr = extract_image_features(ds.X[tr], groups)[0]
                for c in cfg["classifiers"]:
                    models[(s, c)] = get_classifiers(self.seed, names=[c])[c].fit(Ftr, ds.y[tr])
            for cname, (fn, levels, _) in P.CORRUPTIONS.items():
                for lv in levels:
                    rng = np.random.default_rng(1000 * sd + int(lv * 100) + 7)
                    Xc = fn(ds.X[te], lv, rng) if lv != 0 else ds.X[te]
                    for s in cfg["feature_sets"]:
                        Fte = extract_image_features(Xc, FEATURE_SETS["image"][s])[0]
                        for c in cfg["classifiers"]:
                            rows.append({"seed": sd, "corruption": cname, "level": lv, "feature_set": s, "classifier": c,
                                         "accuracy": float((models[(s, c)].predict(Fte) == ds.y[te]).mean())})
        df = pd.DataFrame(rows)
        agg = df.groupby(["corruption", "level", "feature_set", "classifier"]).accuracy.mean().reset_index()
        self.save_table(agg, "robustness_results.csv")
        for cname, (_, levels, xlabel) in P.CORRUPTIONS.items():
            sub = agg[agg.corruption == cname]
            fig_series = {}
            for s in cfg["feature_sets"]:
                fig_series[s] = [float(sub[(sub.level == lv) & (sub.feature_set == s)].accuracy.mean()) for lv in levels]
            pl.plot_line_family(levels, fig_series, self.fig(f"robustness_{cname}.png"), xlabel, "accuracy (mean over classifiers)",
                                cname.replace("_", " "), ylim=(0, 1.0), figsize=(4.4, 3.0))
        # summary: area under degradation curve per feature set
        summ = agg[agg.level != 0].groupby(["corruption", "feature_set"]).accuracy.mean().unstack()
        self.save_table(summ.round(4).reset_index(), "robustness_summary.csv")
        self.summary["robustness_mean_by_featureset"] = agg[agg.level != 0].groupby("feature_set").accuracy.mean().round(4).to_dict()

    # ------------------------------------------------------------ clustering
    def exp_clustering(self):
        self.log("clustering study")
        cc = self.cfg["clustering"]
        ds = self.dataset("digits")
        y = ds.y
        spaces = {}
        for s in ("raw", "hog", "hog+lbp", "handcrafted"):
            Z = Standardizer().fit_transform(self.feat("digits", s).X)
            spaces[f"{s} -> PCA{cc['pca_components']}"] = PCAScratch(cc["pca_components"]).fit_transform(Z)
        Zm = spaces[f"hog+lbp -> PCA{cc['pca_components']}"]
        spaces["hog+lbp -> LDA9 (supervised bound)"] = LDAProjection().fit(Zm, y).transform(Zm)
        rows = []
        labels_store = {}
        for sp, Z in spaces.items():
            eps = estimate_eps(Z, 5, 0.7)
            for an, model in get_clusterers(ds.n_classes, self.seed, eps).items():
                t0 = time.perf_counter()
                lab = model.fit_predict(Z)
                dt = time.perf_counter() - t0
                sc = clustering_scores(Z, y, lab)
                rows.append({"space": sp, "algorithm": an, "time_s": dt, **sc})
                labels_store[(sp, an)] = lab
        res = pd.DataFrame(rows)
        self.save_table(res, "clustering_results.csv")
        self.cache["clu_res"] = res
        # headline figure for main space
        main = f"hog+lbp -> PCA{cc['pca_components']}"
        sub = res[res.space == main].set_index("algorithm")
        pl.plot_metric_bars(sub, ["ARI", "NMI", "cluster_acc", "purity"], self.fig("clustering_metrics_main.png"), f"digits: clustering quality in space '{main}'")
        piv = res.pivot(index="algorithm", columns="space", values="ARI")
        pl.plot_heatmap(piv.values, [c.replace(" -> ", "\n") for c in piv.columns], piv.index, self.fig("clustering_ari_heatmap.png"), "Adjusted Rand Index (digits)", cmap="YlGnBu", fmt="{:.2f}", figsize=(6.6, 4.4), vmin=0, vmax=1, cbar_label="ARI")
        best_alg = sub["ARI"].drop(index=[a for a in sub.index if "LDA" in a], errors="ignore").idxmax()
        lab = labels_store[(main, best_alg)]
        pl.plot_contingency(y, lab, ds.class_names, self.fig("cluster_contingency.png"), f"{best_alg}: clusters vs classes")
        pl.plot_cluster_scatter(self.cache["tsne_digits"], y, lab, self.fig("cluster_scatter_tsne.png"))
        pl.plot_silhouette(Zm, lab, self.fig("cluster_silhouette.png"), f"silhouette: {best_alg}")
        pl.plot_dendrogram(Zm, self.fig("dendrogram.png"), y, 150, self.seed)
        self.summary["clustering_best_main"] = {"algorithm": best_alg, **{k: float(v) for k, v in sub.loc[best_alg][["ARI", "NMI", "cluster_acc", "purity", "silhouette"]].items()}}
        # choosing k
        ks = cc["k_range"] if not self.quick else [2, 4, 6, 8, 10, 12]
        sw = sweep_k(Zm, ks, self.seed)
        gp = gap_statistic(Zm, ks, n_refs=4 if self.quick else 8, seed=self.seed)
        pl.plot_k_selection(sw, gp, self.fig("k_selection.png"), true_k=ds.n_classes)
        self.save_table(pd.DataFrame({**sw, "gap": gp["gap"]}), "k_selection.csv")
        gk = int(ks[int(np.argmax(gp["gap"]))])
        self.summary["k_selection"] = {"best_silhouette_k": int(ks[int(np.nanargmax(sw["silhouette"]))]), "best_bic_k": int(ks[int(np.argmin(sw["bic"]))]),
                                       "best_gap_k": gk}
        # init comparison
        inits = {}
        for init in ("random", "kmeans++"):
            inits[init] = [KMeansScratch(10, init=init, n_init=1, max_iter=300, random_state=s).fit(Zm).inertia_ for s in range(15 if self.quick else 40)]
        pd.DataFrame(inits).describe().round(1).reset_index().to_csv(os.path.join(self.tab_dir, "kmeans_init_comparison.csv"), index=False)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.4, 3.0))
        ax.boxplot([inits["random"], inits["kmeans++"]], tick_labels=["random init", "k-means++ init"], patch_artist=True, boxprops=dict(facecolor="#cfe0f3"), medianprops=dict(color=pl.PALETTE[3]))
        ax.set_ylabel("final inertia (single run)")
        ax.set_title("k-means initialisation")
        pl.save(fig, self.fig("kmeans_init.png"))
        # GMM convergence + covariance type
        g = GMMScratch(10, "full", n_init=1, random_state=self.seed).fit(Zm)
        pl.plot_training_curves({"mean log-likelihood": g.loglik_history_}, self.fig("gmm_convergence.png"), "mean log-likelihood", f"EM convergence ({g.n_iter_} iterations)")
        cov_rows = []
        for ct in ("spherical", "diag", "full"):
            gm = GMMScratch(10, ct, n_init=1, random_state=self.seed).fit(Zm)
            cov_rows.append({"covariance": ct, "BIC": gm.bic(Zm), "AIC": gm.aic(Zm), "ARI": clustering_scores(Zm, y, gm.labels_)["ARI"], "iterations": gm.n_iter_})
        self.save_table(pd.DataFrame(cov_rows), "gmm_covariance_types.csv")
        # FCM fuzzifier sweep
        ms = [1.05, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0]
        fc = [FuzzyCMeans(10, m=m, random_state=self.seed).fit(Zm) for m in ms]
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        ax.plot(ms, [clustering_scores(Zm, y, f.labels_)["ARI"] for f in fc], marker="o", ms=3, label="ARI")
        ax.plot(ms, [f.partition_coefficient_ for f in fc], marker="s", ms=3, label="partition coefficient")
        ax.set_xlabel("fuzzifier m")
        ax.set_title("fuzzy c-means: effect of the fuzzifier")
        ax.legend()
        pl.save(fig, self.fig("fcm_fuzzifier.png"))
        # DBSCAN sweep
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        eps_grid = np.linspace(estimate_eps(Zm, 5, 0.3), estimate_eps(Zm, 5, 0.95), 14)
        ari, ncl, noise = [], [], []
        for e in eps_grid:
            lab = DBSCAN(eps=e, min_samples=5).fit_predict(Zm)
            sc = clustering_scores(Zm, y, lab)
            ari.append(sc["ARI"]); ncl.append(sc["n_clusters"]); noise.append(sc["noise_fraction"])
        ax.plot(eps_grid, ari, marker="o", ms=3, label="ARI")
        ax.plot(eps_grid, noise, marker="s", ms=3, label="noise fraction")
        ax.set_xlabel("eps")
        ax.set_title("DBSCAN sensitivity to eps (min_samples=5)")
        ax2 = ax.twinx()
        ax2.plot(eps_grid, ncl, color=pl.PALETTE[2], ls="--", label="#clusters")
        ax2.set_ylabel("#clusters")
        ax2.grid(False)
        ax.legend(loc="upper left")
        pl.save(fig, self.fig("dbscan_sweep.png"))
        self.save_table(pd.DataFrame({"eps": eps_grid, "ARI": ari, "n_clusters": ncl, "noise_fraction": noise}), "dbscan_sweep.csv")
        # stability
        nb = 4 if self.quick else cc["bootstrap_runs"]
        m, s = bootstrap_stability(lambda: KMeansScratch(10, n_init=2, random_state=self.seed), Zm, nb, self.seed)
        m2, s2 = bootstrap_stability(lambda: GMMScratch(10, "full", n_init=1, random_state=self.seed), Zm, max(3, nb // 2), self.seed)
        self.save_table(pd.DataFrame([{"algorithm": "KMeans (scratch)", "mean_pairwise_ARI": m, "std": s}, {"algorithm": "GMM-full (scratch)", "mean_pairwise_ARI": m2, "std": s2}]), "cluster_stability.csv")
        # other datasets
        rows = []
        for name in ("signals", "wine", "breast_cancer"):
            dsn = self.dataset(name)
            s = self.main_set(name)
            Z = Standardizer().fit_transform(self.feat(name, s).X)
            if Z.shape[1] > 20:
                Z = PCAScratch(min(15, Z.shape[1] - 1)).fit_transform(Z)
            for an, model in get_clusterers(dsn.n_classes, self.seed, estimate_eps(Z, 5, 0.9)).items():
                try:
                    lab = model.fit_predict(Z)
                    rows.append({"dataset": name, "algorithm": an, **clustering_scores(Z, dsn.y, lab)})
                except Exception as e:  # pragma: no cover
                    rows.append({"dataset": name, "algorithm": an, "ARI": np.nan})
        self.save_table(pd.DataFrame(rows), "clustering_other_datasets.csv")
        # signals visual
        sg = self.dataset("signals")
        Zs = Standardizer().fit_transform(self.feat("signals", "handcrafted").X)
        Zp = PCAScratch(10).fit_transform(Zs)
        lab = GMMScratch(6, "full", random_state=self.seed).fit_predict(Zp)
        E = TSNE(2, init="pca", perplexity=30, random_state=self.seed).fit_transform(Zp)
        pl.plot_cluster_scatter(E, sg.y, lab, self.fig("cluster_scatter_signals.png"))

    # --------------------------------------------------------------- driver
    def run_all(self):
        self.exp_overview()
        self.exp_feature_visuals()
        self.exp_stats()
        self.exp_classification()
        self.exp_classification_figures()
        self.exp_sweeps()
        self.exp_ablation()
        self.exp_robustness()
        self.exp_clustering()
        self.summary["runtime_seconds"] = round(time.time() - self.t0, 1)
        with open(os.path.join(self.out, "json", "summary.json"), "w") as f:
            json.dump(self.summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        self.log("done")
        return self.summary
