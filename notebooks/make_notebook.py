"""Build and execute notebooks/PatternRecognitionWorkbench.ipynb (run:  python notebooks/make_notebook.py).

The notebook is a guided, end-to-end tour of the workbench: every stage of the pattern-recognition chain is
executed on real data and its outputs are stored inside the .ipynb, so the file can be uploaded as-is.
"""
import os
import sys

import nbformat as nbf
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "PatternRecognitionWorkbench.ipynb")

cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md("""
# End-to-End Pattern Recognition Workbench - guided tour

**Authors:** Saksham Bakshi and Anjika Shokeen  |  B.Tech final-year mini project (Pattern Recognition)

This notebook walks through the complete chain implemented in the `prwb` package:

1. **Data** - three modalities (handwritten-digit images, synthetic waveform signals, tabular medical/chemical data)
2. **Feature extraction** - HOG, LBP, GLCM, Hu moments, spectral/Hjorth/auto-correlation descriptors (all from scratch in NumPy)
3. **Statistical analysis** - normality, ANOVA/Kruskal-Wallis with FDR control, effect sizes, Bhattacharyya separability, outliers
4. **Classification** - k-NN, Bayes (QDA/LDA/naive), Parzen, softmax regression and an MLP with back-propagation (from scratch) vs. SVM / forests / boosting
5. **Clustering** - k-means++, EM-GMM, fuzzy c-means (from scratch) vs. hierarchical / spectral / DBSCAN / HDBSCAN
6. **Performance evaluation** - repeated stratified CV, ROC / calibration, McNemar, corrected resampled t-test, Friedman-Nemenyi

Everything runs offline (datasets ship with scikit-learn or are synthesised) and is seeded for reproducibility.
""")

code("""
import os, sys, warnings, time
warnings.filterwarnings("ignore")
# make the repository importable whether the notebook is run from notebooks/ or from the repository root
ROOT = os.path.abspath("..") if os.path.exists(os.path.join("..", "prwb")) else os.path.abspath(".")
sys.path.insert(0, ROOT)
FIG = os.path.join(ROOT, "results", "notebook_figs"); os.makedirs(FIG, exist_ok=True)

import numpy as np, pandas as pd
from IPython.display import Image, display
pd.set_option("display.precision", 4); pd.set_option("display.width", 140)

import prwb
from prwb import datasets as D
from prwb.features import build_features, PCAScratch, Standardizer, LDAProjection, FEATURE_SETS
from prwb.features.image import hog_cells, lbp_codes
from prwb.stats import anova_table, normality_tests, bhattacharyya_matrix, class_separability, mahalanobis_outliers, correlation_summary
from prwb.classifiers import get_classifiers
from prwb.clustering import get_clusterers, estimate_eps, sweep_k, gap_statistic
from prwb.evaluation import plots as pl
from prwb.evaluation.metrics import (cross_validate_model, clustering_scores, mcnemar_test, corrected_resampled_ttest,
                                     friedman_nemenyi, bootstrap_ci)
print("prwb", prwb.__version__)
""")

md("## 1. Data - three modalities")
code("""
datasets = {n: D.load_dataset(n) for n in D.list_datasets()}
pd.DataFrame([{"dataset": n, "modality": d.modality, "samples": d.n_samples,
               "pattern shape": "x".join(map(str, d.X.shape[1:])), "classes": d.n_classes,
               "smallest class": int(d.class_counts().min()), "largest class": int(d.class_counts().max())}
              for n, d in datasets.items()])
""")
code("""
dg, sg = datasets["digits"], datasets["signals"]
display(Image(pl.plot_image_grid(dg.X, dg.y, dg.class_names, 10, f"{FIG}/digits.png")))
display(Image(pl.plot_signal_examples(sg.X, sg.y, sg.class_names, sg.meta["fs"], f"{FIG}/signals.png")))
""")

md("""
## 2. Feature extraction

Descriptors are implemented from scratch. The HOG cell histograms and the LBP code map of one digit are shown below.
""")
code("""
i = int(np.flatnonzero(dg.y == 8)[3])
display(Image(pl.plot_hog_visual(dg.X[i], hog_cells(dg.X[i][None], 4, 9)[0], f"{FIG}/hog.png")))
for s in FEATURE_SETS["image"]:
    fm = build_features(dg, s)
    print(f"digits / {s:12s} -> matrix {fm.shape}")
""")
code("""
fm = build_features(dg, "hog+lbp")
Z = PCAScratch(30).fit_transform(Standardizer().fit_transform(fm.X))
emb = {"PCA (unsupervised)": Z[:, :2], "Fisher LDA (supervised)": LDAProjection(2).fit(Z, dg.y).transform(Z)}
display(Image(pl.plot_embeddings(emb, dg.y, dg.class_names, f"{FIG}/emb.png")))
""")

md("## 3. Statistical analysis of the feature space")
code("""
fmh = build_features(dg, "handcrafted")
an = anova_table(fmh.X, dg.y, fmh.names)
print("features:", fmh.X.shape[1], "| significant after BH (q<0.05):", int((an.anova_q_BH < 0.05).sum()))
an.head(8)[["feature", "anova_F", "eta_squared", "omega_squared", "anova_q_BH", "levene_p"]]
""")
code("""
display(Image(pl.plot_anova_top(an, f"{FIG}/anova.png", 20)))
nt = normality_tests(fmh.X, fmh.names)
print("fraction of features that look Gaussian (Shapiro-Wilk, 5%):", round(nt.normal_at_5pct.mean(), 3))
cs = correlation_summary(fmh.X, fmh.names)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in cs.items() if k not in ("corr", "kept_index")})
""")
code("""
Zs = PCAScratch(20).fit_transform(Standardizer().fit_transform(fmh.X))
B = bhattacharyya_matrix(Zs, dg.y)
display(Image(pl.plot_heatmap(B, dg.class_names, dg.class_names, f"{FIG}/bhatt.png", "Pairwise Bhattacharyya distance", cmap="magma",
                              fmt="{:.0f}", annotate=True, figsize=(5.2, 4.4), cbar_label="D_B")))
d2, thr, flag = mahalanobis_outliers(Zs)
print(f"Mahalanobis outliers (chi2, alpha=0.001): {flag.sum()} of {len(flag)}   |  class separability: {class_separability(Zs, dg.y)}")
""")

md("""
## 4. Classification benchmark (repeated stratified cross-validation)

Every model is wrapped in a `Pipeline(Standardizer -> classifier)`, so the scaler is re-fitted inside each training fold (no leakage).
Models marked *scratch* are implemented in NumPy in this repository.
""")
code("""
fm = build_features(dg, "hog+lbp")
names = ["MinDist", "kNN", "Bayes-LDA", "Bayes-QDA", "Parzen", "Softmax", "MLP", "SVM-RBF", "LogReg(sk)", "RandomForest"]
zoo = get_classifiers(0, names=names)
rows, folds, oof = [], {}, {}
for n, m in zoo.items():
    t0 = time.time()
    r = cross_validate_model(m, fm.X, dg.y, n_splits=5, n_repeats=2, seed=0)
    f = r["folds"]; lo, hi = bootstrap_ci(f.accuracy.values)
    rows.append({"classifier": n, "scratch": n in ("MinDist", "kNN", "Bayes-LDA", "Bayes-QDA", "Parzen", "Softmax", "MLP"),
                 "accuracy": f.accuracy.mean(), "ci95_low": lo, "ci95_high": hi, "macro_F1": f.macro_f1.mean(), "MCC": f.mcc.mean(),
                 "AUC": f.roc_auc.mean(), "ECE": f.ece.mean(), "fit_s": r["fit_time"]})
    folds[n] = f.accuracy.values; oof[n] = (r["oof_pred"], r["oof_proba"])
res = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
res
""")
code("""
display(Image(pl.plot_cv_boxplot(folds, f"{FIG}/box.png", "accuracy", "digits (HOG+LBP): fold-wise accuracy, 5x2 CV")))
best = res.classifier[0]
display(Image(pl.plot_confusion(dg.y, oof[best][0], dg.class_names, f"{FIG}/cm.png", f"{best} (out-of-fold)")))
display(Image(pl.plot_roc_multiclass(dg.y, oof[best][1], dg.class_names, f"{FIG}/roc.png", f"({best})")))
""")
code("""
# is the best model *significantly* better than the runner-up?  (McNemar on paired OOF predictions + Nadeau-Bengio corrected t-test)
second = res.classifier[1]
mc = mcnemar_test(dg.y, oof[best][0], oof[second][0])
n = dg.n_samples; nte = n // 5
ct = corrected_resampled_ttest(folds[best], folds[second], n - nte, nte)
print(f"{best} vs {second}:  McNemar p={mc['p']:.3f}  (n01={mc['n01']}, n10={mc['n10']})   corrected-t p={ct['p']:.3f}")
display(Image(pl.plot_calibration(dg.y, {k: oof[k][1] for k in [best, "MLP", "Bayes-QDA", "kNN"]}, f"{FIG}/cal.png")))
""")

md("## 5. Signals: waveform-family recognition (1-D patterns)")
code("""
fs_ = build_features(sg, "handcrafted")
print("signal feature matrix:", fs_.shape)
rows = []
for s in ["raw", "time", "spectral", "time+hjorth", "handcrafted"]:
    f_ = build_features(sg, s)
    r = cross_validate_model(get_classifiers(0, names=["SVM-RBF"])["SVM-RBF"], f_.X, sg.y, 5, 1, 0, keep_oof=False)
    rows.append({"feature set": s, "dims": f_.X.shape[1], "SVM-RBF accuracy": r["folds"].accuracy.mean()})
pd.DataFrame(rows)
""")

md("## 6. Clustering (no labels used for fitting)")
code("""
Zc = PCAScratch(30).fit_transform(Standardizer().fit_transform(fm.X))
rows = []
for name, model in get_clusterers(10, 0, estimate_eps(Zc, 5, 0.7)).items():
    lab = model.fit_predict(Zc)
    rows.append({"algorithm": name, **{k: v for k, v in clustering_scores(Zc, dg.y, lab).items() if k in ("ARI", "NMI", "cluster_acc", "purity", "silhouette", "n_clusters")}})
cl = pd.DataFrame(rows).sort_values("ARI", ascending=False).reset_index(drop=True); cl
""")
code("""
lab = get_clusterers(10, 0)["Agglomerative-Ward"].fit_predict(Zc)
display(Image(pl.plot_contingency(dg.y, lab, dg.class_names, f"{FIG}/cont.png", "Ward clusters vs. digit classes")))
ks = list(range(2, 16))
sw, gp = sweep_k(Zc, ks, 0), gap_statistic(Zc, ks, n_refs=5, seed=0)
display(Image(pl.plot_k_selection(sw, gp, f"{FIG}/ksel.png", true_k=10)))
""")

md("## 7. Train a deployable recogniser and predict a new digit image")
code("""
from prwb.pipeline import train_final_model
from prwb.features.image import extract_image_features
model = train_final_model("digits", "hog+lbp", "SVM-RBF")
i = int(np.flatnonzero(dg.y == 7)[10])
p = model.predict_proba(extract_image_features(dg.X[i][None], FEATURE_SETS["image"]["hog+lbp"])[0])[0]
print("true label: 7 | predicted:", int(np.argmax(p)), f"| confidence {p.max():.2%}")
""")

md("""
## 8. Summary

* The full chain (features -> statistics -> classification -> clustering -> evaluation) runs from one package and one CLI (`python -m prwb --help`).
* From-scratch implementations are numerically validated against scikit-learn / scikit-image / SciPy in the unit-test suite (`pytest`).
* The complete benchmark behind the project report is reproduced by `python scripts/run_experiments.py`.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": HERE}})
client.execute()
nbf.write(nb, OUT)
print("wrote", OUT, os.path.getsize(OUT) // 1024, "KB")
