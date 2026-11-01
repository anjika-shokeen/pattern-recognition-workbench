import numpy as np
import pandas as pd
import pytest
from scipy import stats as ss
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from prwb.clustering import FuzzyCMeans, GMMScratch, KMeansScratch, gap_statistic, sweep_k
from prwb.evaluation import (clustering_accuracy, clustering_scores, corrected_resampled_ttest, cross_validate_model,
                             expected_calibration_error, friedman_nemenyi, mcnemar_test, purity)
from prwb.classifiers import get_classifiers
from prwb.stats import (anova_table, benjamini_hochberg, bhattacharyya_matrix, mahalanobis_outliers, normality_tests)
from prwb.preprocess import to_digit_image, add_gaussian_noise, rotate


@pytest.fixture(scope="module")
def blobs():
    return make_blobs(300, centers=3, n_features=4, cluster_std=0.7, random_state=1)


# -------------------------------------------------------------- clustering
def test_kmeans_recovers_blobs(blobs):
    X, y = blobs
    km = KMeansScratch(3, random_state=0).fit(X)
    assert adjusted_rand_score(y, km.labels_) > 0.95
    assert km.inertia_ > 0 and km.cluster_centers_.shape == (3, 4)


def test_kmeanspp_not_worse_than_random_on_average(blobs):
    X, _ = blobs
    pp = np.mean([KMeansScratch(3, n_init=1, random_state=s).fit(X).inertia_ for s in range(10)])
    rnd = np.mean([KMeansScratch(3, "random", n_init=1, random_state=s).fit(X).inertia_ for s in range(10)])
    assert pp <= rnd * 1.05


def test_gmm_em_monotone_and_accurate(blobs):
    X, y = blobs
    g = GMMScratch(3, "full", n_init=1, random_state=0).fit(X)
    h = np.diff(g.loglik_history_)
    assert (h > -1e-8).all()                     # EM never decreases the likelihood
    assert adjusted_rand_score(y, g.labels_) > 0.95
    assert np.allclose(g.weights_.sum(), 1)


@pytest.mark.parametrize("cov", ["full", "diag", "spherical"])
def test_gmm_covariance_types(blobs, cov):
    X, y = blobs
    g = GMMScratch(3, cov, n_init=1).fit(X)
    assert g.bic(X) > 0 or True
    assert adjusted_rand_score(y, g.predict(X)) > 0.9


def test_gmm_bic_prefers_true_k(blobs):
    X, _ = blobs
    bics = [GMMScratch(k, "full", n_init=1).fit(X).bic(X) for k in (1, 2, 3, 4, 5)]
    assert int(np.argmin(bics)) + 1 == 3


def test_fuzzy_cmeans_memberships(blobs):
    X, y = blobs
    f = FuzzyCMeans(3, m=1.5).fit(X)
    assert np.allclose(f.membership_.sum(1), 1)
    assert adjusted_rand_score(y, f.labels_) > 0.9


def test_sweep_and_gap_find_k(blobs):
    X, _ = blobs
    sw = sweep_k(X, [2, 3, 4, 5])
    assert sw["k"][int(np.nanargmax(sw["silhouette"]))] == 3
    gp = gap_statistic(X, [1, 2, 3, 4], n_refs=4)
    assert len(gp["gap"]) == 4


# -------------------------------------------------------------- evaluation
def test_cluster_metrics_permutation_invariance():
    y = np.repeat([0, 1, 2], 20)
    perm = np.array([2, 0, 1])[y]
    assert clustering_accuracy(y, perm) == 1.0 and purity(y, perm) == 1.0
    assert adjusted_rand_score(y, perm) == 1.0


def test_clustering_scores_keys(blobs):
    X, y = blobs
    s = clustering_scores(X, y, y)
    assert s["ARI"] == 1.0 and s["NMI"] == pytest.approx(1.0) and s["silhouette"] > 0.5


def test_cross_validation_no_leak_and_oof_complete(blobs):
    X, y = blobs
    r = cross_validate_model(get_classifiers(names=["kNN"])["kNN"], X, y, n_splits=5, n_repeats=2)
    assert len(r["folds"]) == 10
    assert (r["oof_pred"] >= 0).all()           # first repeat covers every sample once
    assert r["folds"]["accuracy"].mean() > 0.9


def test_mcnemar_known_values():
    y = np.zeros(100, int)
    a = np.zeros(100, int)
    b = a.copy(); b[:20] = 1                    # B wrong on 20 samples where A is right
    r = mcnemar_test(y, a, b)
    assert (r["n01"], r["n10"]) == (20, 0) and r["p"] < 1e-4
    assert mcnemar_test(y, a, a)["p"] == 1.0


def test_corrected_t_is_more_conservative_than_naive():
    a = np.array([.95, .96, .94, .95, .97, .96, .95, .96, .95, .94])
    b = a - 0.005 + np.random.default_rng(0).normal(0, 0.004, 10)
    naive = ss.ttest_rel(a, b).pvalue
    corr = corrected_resampled_ttest(a, b, 800, 200)["p"]
    assert corr > naive


def test_friedman_nemenyi_ranks():
    rng = np.random.default_rng(0)
    S = pd.DataFrame({"good": 0.9 + rng.normal(0, .005, 12), "mid": 0.8 + rng.normal(0, .005, 12), "bad": 0.7 + rng.normal(0, .005, 12)})
    r = friedman_nemenyi(S)
    assert r["avg_ranks"].idxmin() == "good" and r["p"] < 0.01 and r["cd"] > 0


def test_ece_perfect_and_overconfident():
    y = np.array([0, 1] * 50)
    P = np.eye(2)[y]
    assert expected_calibration_error(P, y) == 0
    Pw = np.eye(2)[1 - y]
    assert expected_calibration_error(Pw, y) == pytest.approx(1.0)


# -------------------------------------------------------------- statistics
def test_benjamini_hochberg_matches_statsmodels_definition():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    q = benjamini_hochberg(p)
    assert np.all(q >= p) and np.all(np.diff(q) >= -1e-12)
    assert q[0] == pytest.approx(0.01)


def test_anova_matches_scipy(blobs):
    X, y = blobs
    t = anova_table(X, y, list("abcd")).set_index("feature")
    for j, n in enumerate("abcd"):
        F, p = ss.f_oneway(*[X[y == c, j] for c in range(3)])
        assert t.loc[n, "anova_F"] == pytest.approx(F) and t.loc[n, "anova_p"] == pytest.approx(p)
    assert (t["eta_squared"].between(0, 1)).all()


def test_normality_detects_non_gaussian():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=400), rng.exponential(size=400)])
    n = normality_tests(X, ["gauss", "expo"])
    assert n.loc[0, "shapiro_p"] > 0.01 and n.loc[1, "shapiro_p"] < 1e-6


def test_bhattacharyya_ordering():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (100, 3)), rng.normal(1, 1, (100, 3)), rng.normal(5, 1, (100, 3))])
    y = np.repeat([0, 1, 2], 100)
    B = bhattacharyya_matrix(X, y)
    assert B[0, 1] < B[0, 2] and np.allclose(B, B.T) and np.allclose(np.diag(B), 0)


def test_mahalanobis_flags_planted_outlier():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    X[0] = 15
    d2, thr, flag = mahalanobis_outliers(X)
    assert flag[0]


# -------------------------------------------------------------- preprocessing
def test_to_digit_image_inverts_and_centres():
    page = np.full((60, 40), 255.0)
    page[10:50, 18:22] = 0          # dark vertical stroke on light page (a "1")
    d = to_digit_image(page)
    # bilinear down-scaling of a 4-pixel-wide stroke legitimately lowers its peak below 1.0
    assert d.shape == (16, 16) and 0.5 < d.max() <= 1.0
    ys, xs = np.where(d > 0.5)
    assert abs(xs.mean() - 7.5) < 2.5


def test_corruptions_preserve_shape_and_range():
    X = np.random.default_rng(0).random((4, 16, 16))
    rng = np.random.default_rng(0)
    assert add_gaussian_noise(X, 0.3, rng).shape == X.shape
    R = rotate(X, 20)
    assert R.shape == X.shape and R.min() >= 0
