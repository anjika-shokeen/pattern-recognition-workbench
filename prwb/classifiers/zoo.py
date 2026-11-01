"""The classifier zoo: every model is wrapped in a scaling pipeline so CV is leakage-free."""
from __future__ import annotations

from typing import Dict, List, Optional

from sklearn.base import BaseEstimator
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ..features.transforms import Standardizer
from .scratch import (GaussianBayes, KNNClassifier, MinimumDistance, NumpyMLP, ParzenClassifier,
                      SoftmaxRegression)

SCRATCH = ["MinDist", "kNN", "Bayes-Naive", "Bayes-LDA", "Bayes-QDA", "Parzen", "Softmax", "MLP"]
REFERENCE = ["SVM-RBF", "SVM-Linear", "LogReg(sk)", "DecisionTree", "RandomForest", "ExtraTrees", "HistGB"]


def _calibrated(svc) -> BaseEstimator:
    """SVMs have no native probabilities: Platt scaling fitted by internal 3-fold CV."""
    return CalibratedClassifierCV(svc, method="sigmoid", cv=3, ensemble=False)


def _scaled(est: BaseEstimator) -> Pipeline:
    return Pipeline([("scale", Standardizer()), ("clf", est)])


def get_classifiers(seed: int = 0, names: Optional[List[str]] = None, fast: bool = False) -> Dict[str, Pipeline]:
    """Return ``{name: Pipeline}``.  ``fast=True`` drops the slowest models."""
    zoo = {
        "MinDist": MinimumDistance(),
        "kNN": KNNClassifier(k=5, weights="distance"),
        "Bayes-Naive": GaussianBayes("naive", reg=0.05),
        "Bayes-LDA": GaussianBayes("lda", reg=0.1),
        "Bayes-QDA": GaussianBayes("qda", reg=0.3),
        "Parzen": ParzenClassifier(bw_scale=0.8),
        "Softmax": SoftmaxRegression(l2=1e-3, epochs=60, random_state=seed),
        "MLP": NumpyMLP(hidden=(64, 32), epochs=60, dropout=0.1, random_state=seed),
        "SVM-RBF": _calibrated(SVC(C=10, gamma="scale", random_state=seed)),
        "SVM-Linear": _calibrated(SVC(kernel="linear", C=0.1, random_state=seed)),
        "LogReg(sk)": LogisticRegression(C=1.0, max_iter=500),
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=seed),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, n_jobs=1, random_state=seed),
        "HistGB": HistGradientBoostingClassifier(max_iter=40, learning_rate=0.15, random_state=seed),
    }
    if fast:
        zoo.pop("HistGB")
        zoo.pop("ExtraTrees")
    chosen = names or list(zoo)
    out = {n: _scaled(zoo[n]) for n in chosen if n in zoo}
    if names and "Ensemble" in names:
        out["Ensemble"] = _scaled(VotingClassifier(
            [("svm", _calibrated(SVC(C=10, random_state=seed))),
             ("mlp", NumpyMLP(hidden=(64, 32), epochs=60, dropout=0.1, random_state=seed)),
             ("knn", KNNClassifier(k=5, weights="distance")),
             ("qda", GaussianBayes("qda", reg=0.3))], voting="soft"))
    return out
