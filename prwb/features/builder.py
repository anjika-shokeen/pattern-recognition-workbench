"""Named feature sets: one place that maps (dataset, feature-set name) -> feature matrix."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ..datasets import Dataset
from .image import extract_image_features
from .signal import extract_signal_features


@dataclass
class FeatureMatrix:
    X: np.ndarray
    names: List[str]
    group_index: np.ndarray
    groups: List[str]
    set_name: str

    @property
    def shape(self):
        return self.X.shape


# feature-set name -> descriptor groups
FEATURE_SETS: Dict[str, Dict[str, List[str]]] = {
    "image": {
        "raw": ["raw"],
        "zoning": ["zoning"],
        "hu": ["hu"],
        "glcm": ["glcm"],
        "lbp": ["lbp"],
        "hog": ["hog"],
        "hog+lbp": ["hog", "lbp"],
        "handcrafted": ["hog", "lbp", "glcm", "hu", "profiles", "stats"],
    },
    "signal": {
        "raw": ["raw"],
        "time": ["time"],
        "spectral": ["spectral"],
        "time+hjorth": ["time", "hjorth"],
        "handcrafted": ["time", "hjorth", "spectral", "autocorr"],
    },
    "tabular": {
        "raw": ["raw"],
    },
}


def build_features(ds: Dataset, set_name: str) -> FeatureMatrix:
    """Compute the named feature set for a dataset."""
    sets = FEATURE_SETS[ds.modality]
    if set_name not in sets:
        raise KeyError(f"feature set '{set_name}' not defined for {ds.modality}; options: {list(sets)}")
    groups = sets[set_name]
    if ds.modality == "image":
        F, names, gi = extract_image_features(ds.X, groups)
    elif ds.modality == "signal":
        F, names, gi = extract_signal_features(ds.X, groups, fs=ds.meta.get("fs", 128.0))
    else:
        F = ds.X.astype(float)
        names = list(ds.feature_names or [f"f{i}" for i in range(F.shape[1])])
        gi = np.zeros(F.shape[1], dtype=int)
        groups = ["tabular"]
    return FeatureMatrix(np.nan_to_num(F), names, gi, groups, set_name)
