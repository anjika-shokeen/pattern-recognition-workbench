import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.estimator_checks import parametrize_with_checks

from prwb.classifiers import (GaussianBayes, KNNClassifier, MinimumDistance, NumpyMLP, ParzenClassifier,
                              SoftmaxRegression, get_classifiers)


@pytest.fixture(scope="module")
def blobs():
    X, y = make_blobs(400, centers=4, n_features=5, cluster_std=2.0, random_state=0)
    return X[:300], y[:300], X[300:], y[300:]


def test_knn_matches_sklearn(blobs):
    Xtr, ytr, Xte, yte = blobs
    for metric in ("euclidean", "manhattan"):
        mine = KNNClassifier(5, metric).fit(Xtr, ytr).predict(Xte)
        ref = KNeighborsClassifier(5, metric=metric).fit(Xtr, ytr).predict(Xte)
        assert (mine == ref).mean() > 0.98


def test_knn_k1_perfect_on_train(blobs):
    Xtr, ytr, *_ = blobs
    assert (KNNClassifier(1).fit(Xtr, ytr).predict(Xtr) == ytr).all()


def test_knn_proba_rows_sum_to_one(blobs):
    Xtr, ytr, Xte, _ = blobs
    P = KNNClassifier(7, weights="distance").fit(Xtr, ytr).predict_proba(Xte)
    assert np.allclose(P.sum(1), 1)


def test_gaussian_bayes_lda_matches_sklearn(blobs):
    Xtr, ytr, Xte, _ = blobs
    mine = GaussianBayes("lda", reg=0.0).fit(Xtr, ytr).predict(Xte)
    ref = LinearDiscriminantAnalysis(solver="lsqr").fit(Xtr, ytr).predict(Xte)
    assert (mine == ref).mean() > 0.97


def test_gaussian_bayes_qda_matches_sklearn(blobs):
    Xtr, ytr, Xte, _ = blobs
    mine = GaussianBayes("qda", reg=0.0).fit(Xtr, ytr).predict(Xte)
    ref = QuadraticDiscriminantAnalysis().fit(Xtr, ytr).predict(Xte)
    assert (mine == ref).mean() > 0.97


def test_bayes_posteriors_normalised(blobs):
    Xtr, ytr, Xte, _ = blobs
    for mode in ("qda", "lda", "naive"):
        P = GaussianBayes(mode).fit(Xtr, ytr).predict_proba(Xte)
        assert np.allclose(P.sum(1), 1) and (P >= 0).all()


def test_min_distance_and_parzen_reasonable(blobs):
    Xtr, ytr, Xte, yte = blobs
    assert (MinimumDistance().fit(Xtr, ytr).predict(Xte) == yte).mean() > 0.7
    assert (MinimumDistance(True).fit(Xtr, ytr).predict(Xte) == yte).mean() > 0.7
    assert (ParzenClassifier().fit(Xtr, ytr).predict(Xte) == yte).mean() > 0.7


def test_softmax_loss_decreases(blobs):
    Xtr, ytr, Xte, yte = blobs
    m = SoftmaxRegression(epochs=40).fit(Xtr, ytr)
    assert m.loss_curve_[-1] < m.loss_curve_[0]
    assert (m.predict(Xte) == yte).mean() > 0.7


def test_mlp_gradient_check():
    """Back-propagation vs. central finite differences."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 4))
    y = rng.integers(0, 3, 12)
    m = NumpyMLP(hidden=(5, 4), l2=1e-2, epochs=1, val_fraction=0.0, random_state=0)
    m.classes_ = np.arange(3)
    m._init(4, 3, np.random.default_rng(1))
    Y = np.eye(3)[y]
    _, gW, gb = m._loss_and_grads(X, Y)
    eps = 1e-6
    for layer in range(len(m.Ws_)):
        for idx in [(0, 0), (1, 2), (-1, -1)]:
            W = m.Ws_[layer]
            old = W[idx]
            W[idx] = old + eps; lp = m._loss_and_grads(X, Y)[0]
            W[idx] = old - eps; lm = m._loss_and_grads(X, Y)[0]
            W[idx] = old
            assert abs((lp - lm) / (2 * eps) - gW[layer][idx]) < 1e-6
        b = m.bs_[layer]
        old = b[0]
        b[0] = old + eps; lp = m._loss_and_grads(X, Y)[0]
        b[0] = old - eps; lm = m._loss_and_grads(X, Y)[0]
        b[0] = old
        assert abs((lp - lm) / (2 * eps) - gb[layer][0]) < 1e-6


def test_mlp_learns_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 50, float)
    y = np.array([0, 1, 1, 0] * 50)
    m = NumpyMLP(hidden=(8,), epochs=200, lr=0.02, val_fraction=0.0, random_state=0).fit(X, y)
    assert (m.predict(X) == y).mean() == 1.0


def test_zoo_estimators_clone_and_fit(blobs):
    Xtr, ytr, Xte, yte = blobs
    zoo = get_classifiers(names=["MinDist", "kNN", "Bayes-QDA", "Softmax", "MLP", "SVM-RBF", "Ensemble"])
    for name, est in zoo.items():
        m = clone(est).fit(Xtr, ytr)
        assert (m.predict(Xte) == yte).mean() > 0.6, name
        assert np.allclose(m.predict_proba(Xte).sum(1), 1), name


def test_label_handling_noncontiguous_labels(blobs):
    Xtr, ytr, Xte, _ = blobs
    labels = np.array([10, 20, 30, 40])
    m = KNNClassifier(3).fit(Xtr, labels[ytr])
    assert set(m.predict(Xte)) <= set(labels)
