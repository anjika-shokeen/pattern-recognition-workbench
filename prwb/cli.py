"""Command line interface:  python -m prwb <command> [options]

Commands
--------
datasets                         list datasets and feature sets
features  -d digits -f hog       compute a feature set and print its shape / head
stats     -d digits -f handcrafted     ANOVA ranking, correlation and separability summary
classify  -d digits -f hog+lbp -c SVM-RBF   cross-validated classification report
cluster   -d digits -f hog+lbp -a "Agglomerative-Ward"   clustering + validity indices
benchmark [--quick] [--out results]      full experimental protocol
train     -o model.joblib        fit and save the deployable digit recogniser
predict   -m model.joblib image.png      recognise a hand-written digit picture
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import __version__
from . import datasets as D


def _cmd_datasets(a):
    from .features.builder import FEATURE_SETS
    from .classifiers import get_classifiers
    for n in D.list_datasets():
        ds = D.load_dataset(n)
        print(ds.summary())
        print("   feature sets:", ", ".join(FEATURE_SETS[ds.modality]))
    print("classifiers :", ", ".join(get_classifiers()))
    from .clustering import get_clusterers
    print("clusterers  :", ", ".join(get_clusterers(3, eps=1.0)))


def _cmd_features(a):
    from .features import build_features
    ds = D.load_dataset(a.dataset)
    fm = build_features(ds, a.features)
    print(f"{ds.name}/{a.features}: matrix {fm.shape}, groups={fm.groups}")
    print(pd.DataFrame(fm.X[:5, :8], columns=fm.names[:8]).round(4).to_string())


def _cmd_stats(a):
    from .pipeline import quick_stats
    r = quick_stats(a.dataset, a.features, a.top)
    print("Top features by one-way ANOVA (BH-adjusted q-values):")
    print(r["anova"][["feature", "anova_F", "eta_squared", "anova_q_BH"]].round(4).to_string(index=False))
    print("\nFraction of normal features (Shapiro, 5%):", round(r["normality_fraction"], 3))
    print("Correlation summary:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r["correlation"].items()}))
    print("Separability:", {k: round(v, 3) for k, v in r["separability"].items()})


def _cmd_classify(a):
    from .pipeline import quick_classify
    from sklearn.metrics import classification_report
    r = quick_classify(a.dataset, a.features, a.classifier, a.folds, a.repeats, a.seed, a.subsample)
    f = r["folds"]
    print(f"{a.dataset}/{a.features}/{a.classifier}  ({a.folds}x{a.repeats} CV)")
    print(f[["accuracy", "macro_f1", "mcc"]].agg(["mean", "std"]).round(4).to_string())
    ds = r["dataset"]
    print()
    print(classification_report(ds.y, r["oof_pred"], target_names=ds.class_names, digits=4))


def _cmd_cluster(a):
    from .pipeline import quick_cluster
    r = quick_cluster(a.dataset, a.features, a.algorithm, a.k, a.pca, a.seed, a.subsample)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r["scores"].items()}, indent=2))


def _cmd_benchmark(a):
    from .experiments import Workbench
    from .pipeline import load_config
    wb = Workbench(load_config(a.config), a.out, a.quick)
    wb.run_all()


def _cmd_train(a):
    import joblib
    from .pipeline import train_final_model
    m = train_final_model(a.dataset, a.features, a.classifier)
    joblib.dump(m, a.output, compress=3)
    print("saved", a.output)


def _cmd_predict(a):
    import joblib
    from PIL import Image
    from .features.builder import FEATURE_SETS
    from .features.image import extract_image_features
    from .preprocess import to_digit_image
    model = joblib.load(a.model)
    img = to_digit_image(np.asarray(Image.open(a.image).convert("L")))
    F = extract_image_features(img[None], FEATURE_SETS["image"][a.features])[0]
    p = model.predict_proba(F)[0]
    top = np.argsort(-p)[:3]
    print("prediction:", int(model.classes_[top[0]]) if hasattr(model, "classes_") else int(top[0]))
    for t in top:
        print(f"  digit {t}: {p[t]:.3f}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="prwb", description="End-to-End Pattern Recognition Workbench")
    p.add_argument("--version", action="version", version=__version__)
    sp = p.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("datasets")
    s.set_defaults(fn=_cmd_datasets)

    def common(s, feat="hog+lbp"):
        s.add_argument("-d", "--dataset", default="digits", choices=D.list_datasets())
        s.add_argument("-f", "--features", default=feat)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--subsample", type=int, default=None, help="samples per class (fast demo)")

    s = sp.add_parser("features"); common(s, "hog"); s.set_defaults(fn=_cmd_features)
    s = sp.add_parser("stats"); common(s, "handcrafted"); s.add_argument("--top", type=int, default=15); s.set_defaults(fn=_cmd_stats)
    s = sp.add_parser("classify"); common(s)
    s.add_argument("-c", "--classifier", default="SVM-RBF"); s.add_argument("--folds", type=int, default=5)
    s.add_argument("--repeats", type=int, default=1); s.set_defaults(fn=_cmd_classify)
    s = sp.add_parser("cluster"); common(s)
    s.add_argument("-a", "--algorithm", default="Agglomerative-Ward"); s.add_argument("-k", type=int, default=None)
    s.add_argument("--pca", type=int, default=30); s.set_defaults(fn=_cmd_cluster)
    s = sp.add_parser("benchmark")
    s.add_argument("--config", default=None); s.add_argument("--out", default="results"); s.add_argument("--quick", action="store_true")
    s.set_defaults(fn=_cmd_benchmark)
    s = sp.add_parser("train")
    s.add_argument("-d", "--dataset", default="digits"); s.add_argument("-f", "--features", default="hog+lbp")
    s.add_argument("-c", "--classifier", default="SVM-RBF"); s.add_argument("-o", "--output", default="model.joblib")
    s.set_defaults(fn=_cmd_train)
    s = sp.add_parser("predict")
    s.add_argument("image"); s.add_argument("-m", "--model", default="model.joblib"); s.add_argument("-f", "--features", default="hog+lbp")
    s.set_defaults(fn=_cmd_predict)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
