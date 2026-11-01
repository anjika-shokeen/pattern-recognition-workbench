"""Integration tests: datasets, high-level pipeline, command-line interface and the experiment driver."""
import json
import os

import numpy as np
import pytest

from prwb import datasets as D
from prwb.cli import main as cli_main
from prwb.evaluation.metrics import bootstrap_ci, clustering_accuracy, hungarian_map
from prwb.experiments import Workbench
from prwb.pipeline import load_config, quick_classify, quick_cluster, quick_stats, train_final_model
from prwb.stats import class_separability, correlation_summary, pairwise_class_ttests


# --------------------------------------------------------------------------- datasets
@pytest.mark.parametrize("name,modality,n_classes", [
    ("digits", "image", 10), ("signals", "signal", 6), ("wine", "tabular", 3), ("breast_cancer", "tabular", 2)])
def test_datasets_are_well_formed(name, modality, n_classes):
    ds = D.load_dataset(name)
    assert ds.modality == modality
    assert ds.n_classes == n_classes
    assert ds.X.shape[0] == ds.y.shape[0] == ds.n_samples
    assert set(np.unique(ds.y)) == set(range(n_classes))
    assert np.isfinite(ds.X).all()


def test_signal_dataset_is_deterministic_and_seeded():
    a = D.make_signal_dataset(n_per_class=5, seed=3)
    b = D.make_signal_dataset(n_per_class=5, seed=3)
    c = D.make_signal_dataset(n_per_class=5, seed=4)
    assert np.array_equal(a.X, b.X) and not np.array_equal(a.X, c.X)


def test_subsample_is_class_balanced():
    ds = D.load_dataset("digits").subsample(7, seed=1)
    assert (ds.class_counts() == 7).all()


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        D.load_dataset("imagenet")


# --------------------------------------------------------------------------- pipeline
def test_quick_classify_digits_is_accurate():
    r = quick_classify("digits", "hog", "SVM-RBF", n_splits=3, subsample=30)
    assert r["folds"]["accuracy"].mean() > 0.9
    assert (r["oof_pred"] >= 0).all()          # every sample predicted out-of-fold exactly once


def test_quick_cluster_and_stats_run():
    r = quick_cluster("wine", "raw", "KMeans (scratch)")
    assert r["scores"]["ARI"] > 0.7
    s = quick_stats("wine", "raw", top=5)
    assert len(s["anova"]) == 5 and "separability" in s


def test_train_final_model_predicts_a_digit_image(tmp_path):
    from PIL import Image
    model = train_final_model("digits", "hog+lbp", "SVM-RBF")
    ds = D.load_dataset("digits")
    i = int(np.flatnonzero(ds.y == 3)[0])
    png = tmp_path / "three.png"
    Image.fromarray((255 * ds.X[i]).astype(np.uint8)).save(png)
    import joblib
    joblib.dump(model, tmp_path / "m.joblib")
    cli_main(["predict", str(png), "-m", str(tmp_path / "m.joblib")])


# --------------------------------------------------------------------------- CLI
def test_cli_datasets_lists_everything(capsys):
    cli_main(["datasets"])
    out = capsys.readouterr().out
    for token in ("digits", "signals", "wine", "breast_cancer", "SVM-RBF", "GMM-full (scratch)"):
        assert token in out


def test_cli_classify_and_cluster(capsys):
    cli_main(["classify", "-d", "wine", "-f", "raw", "-c", "Bayes-LDA", "--folds", "3"])
    assert "accuracy" in capsys.readouterr().out
    cli_main(["cluster", "-d", "wine", "-f", "raw", "-a", "KMeans (scratch)"])
    scores = json.loads(capsys.readouterr().out)
    assert scores["n_clusters"] == 3


# --------------------------------------------------------------------------- experiment driver
def test_workbench_quick_overview_writes_artifacts(tmp_path):
    cfg = load_config()
    wb = Workbench(cfg, str(tmp_path), quick=True, verbose=False)
    wb.exp_overview()
    for f in ("digits_samples.png", "class_balance.png", "diagram_high_level.png", "diagram_architecture.png",
              "diagram_operational.png"):
        assert os.path.getsize(tmp_path / "figures" / f) > 5_000
    assert (tmp_path / "tables" / "dataset_overview.csv").exists()


# --------------------------------------------------------------------------- extra statistics / metrics
def test_correlation_summary_detects_duplicated_feature():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(200, 3))
    X = np.column_stack([a, a[:, 0] * 2.0 + 1.0])
    s = correlation_summary(X, list("abcd"), threshold=0.99)
    assert s["n_pairs_above_threshold"] == 1


def test_separability_grows_with_class_distance():
    rng = np.random.default_rng(1)
    y = np.repeat([0, 1], 100)
    near = rng.normal(size=(200, 4)) + y[:, None] * 0.5
    far = rng.normal(size=(200, 4)) + y[:, None] * 3.0
    assert class_separability(far, y)["J_trace_SwInv_Sb"] > class_separability(near, y)["J_trace_SwInv_Sb"]


def test_pairwise_ttests_shape_and_effect_sign():
    rng = np.random.default_rng(2)
    y = np.repeat([0, 1, 2], 50)
    x = rng.normal(size=150) + y * 2.0
    df = pairwise_class_ttests(x, y, ["a", "b", "c"])
    assert len(df) == 3 and (df["cohens_d"] < 0).all() and (df["p_bonferroni"] <= 1).all()


def test_hungarian_accuracy_is_permutation_invariant():
    y = np.repeat(np.arange(4), 10)
    perm = np.array([2, 0, 3, 1])
    assert clustering_accuracy(y, perm[y]) == 1.0
    mapping, M = hungarian_map(y, perm[y])
    assert M.sum() == len(y)


def test_bootstrap_ci_brackets_the_mean():
    v = np.random.default_rng(3).normal(0.9, 0.02, 30)
    lo, hi = bootstrap_ci(v)
    assert lo < v.mean() < hi
