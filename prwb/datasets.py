"""Dataset layer.

Three data modalities are supported so that the workbench is demonstrably
domain-independent:

* ``image``   - 2-D grey-scale images (handwritten digits, UCI Optical Recognition).
* ``signal``  - 1-D waveforms (synthetic, six waveform families with random
                frequency / phase / amplitude, additive noise and baseline drift).
* ``tabular`` - ready-made feature vectors (UCI Wine, Wisconsin Breast Cancer).

All datasets ship with scikit-learn or are generated locally, so the whole
project runs offline and is exactly reproducible from a random seed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage
from scipy import signal as sps
from sklearn import datasets as skd


@dataclass
class Dataset:
    """Container describing one pattern-recognition problem."""

    name: str
    X: np.ndarray                      # raw patterns: (n,h,w) | (n,L) | (n,d)
    y: np.ndarray                      # integer labels 0..C-1
    class_names: List[str]
    modality: str                      # "image" | "signal" | "tabular"
    feature_names: Optional[List[str]] = None
    meta: Dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.y, minlength=self.n_classes)

    def subsample(self, n_per_class: int, seed: int = 0) -> "Dataset":
        """Class-balanced random subsample (used for fast demos / tests)."""
        rng = np.random.default_rng(seed)
        idx = []
        for c in range(self.n_classes):
            members = np.flatnonzero(self.y == c)
            take = min(n_per_class, members.size)
            idx.append(rng.choice(members, size=take, replace=False))
        idx = np.sort(np.concatenate(idx))
        return Dataset(self.name, self.X[idx], self.y[idx], self.class_names,
                       self.modality, self.feature_names, dict(self.meta))

    def summary(self) -> str:
        shape = "x".join(map(str, self.X.shape[1:]))
        return (f"{self.name}: {self.n_samples} samples, pattern shape {shape}, "
                f"{self.n_classes} classes, modality={self.modality}")


# ----------------------------------------------------------------------------
# images
# ----------------------------------------------------------------------------
def load_digits_images(upscale: int = 2) -> Dataset:
    """UCI Optical Recognition of Handwritten Digits (1,797 images, 8x8, 10 classes).

    Images are rescaled to [0, 1].  A bicubic up-sampling (default x2 -> 16x16)
    gives the gradient / texture descriptors enough spatial support to work with.
    """
    d = skd.load_digits()
    imgs = d.images.astype(np.float64) / 16.0
    if upscale and upscale > 1:
        imgs = np.stack([np.clip(ndimage.zoom(im, upscale, order=3), 0.0, 1.0) for im in imgs])
    return Dataset("digits", imgs, d.target.astype(int), [str(i) for i in range(10)],
                   "image", meta={"source": "UCI Optical Recognition of Handwritten Digits",
                                  "upscale": upscale})


# ----------------------------------------------------------------------------
# signals
# ----------------------------------------------------------------------------
SIGNAL_CLASSES = ["sine", "square", "sawtooth", "chirp", "am-tone", "damped"]


def make_signal_dataset(n_per_class: int = 200, length: int = 256, fs: float = 128.0,
                        noise: float = 0.45, seed: int = 7) -> Dataset:
    """Synthetic waveform-family recognition problem (6 classes).

    Every instance has a random amplitude, phase and frequency drawn from
    *overlapping* ranges, plus white Gaussian noise and a slow baseline drift.
    Because the frequency ranges overlap, no single spectral peak identifies
    a class; the classifier must exploit waveform *shape* (harmonics, envelope,
    frequency sweep, decay).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(length) / fs
    T = t[-1]
    X, y = [], []
    for c, name in enumerate(SIGNAL_CLASSES):
        for _ in range(n_per_class):
            A = rng.uniform(0.7, 1.3)
            ph = rng.uniform(0, 2 * np.pi)
            f = rng.uniform(4.0, 12.0)
            if name == "sine":
                s = np.sin(2 * np.pi * f * t + ph)
            elif name == "square":
                s = sps.square(2 * np.pi * f * t + ph)
            elif name == "sawtooth":
                s = sps.sawtooth(2 * np.pi * f * t + ph)
            elif name == "chirp":
                f0, f1 = rng.uniform(3, 8), rng.uniform(14, 22)
                s = np.sin(2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / T * t ** 2) + ph)
            elif name == "am-tone":
                fm = rng.uniform(1.0, 3.0)
                s = (1 + 0.7 * np.sin(2 * np.pi * fm * t + rng.uniform(0, 6.28))) * np.sin(2 * np.pi * f * t + ph)
            else:  # damped oscillation
                tau = rng.uniform(0.25, 0.6)
                s = np.exp(-t / tau) * np.sin(2 * np.pi * f * t + ph) * 2.0
            drift = rng.uniform(-0.3, 0.3) * np.sin(2 * np.pi * rng.uniform(0.2, 0.8) * t + rng.uniform(0, 6.28))
            s = A * s + drift + rng.normal(0.0, noise, size=length)
            X.append(s)
            y.append(c)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    perm = rng.permutation(len(y))
    return Dataset("signals", X[perm], y[perm], list(SIGNAL_CLASSES), "signal",
                   meta={"fs": fs, "length": length, "noise": noise, "seed": seed})


# ----------------------------------------------------------------------------
# tabular
# ----------------------------------------------------------------------------
def load_wine() -> Dataset:
    d = skd.load_wine()
    return Dataset("wine", d.data.astype(float), d.target.astype(int), list(d.target_names),
                   "tabular", list(d.feature_names), {"source": "UCI Wine (Forina et al., 1991)"})


def load_breast_cancer() -> Dataset:
    d = skd.load_breast_cancer()
    return Dataset("breast_cancer", d.data.astype(float), d.target.astype(int), list(d.target_names),
                   "tabular", list(d.feature_names),
                   {"source": "Wisconsin Diagnostic Breast Cancer (Street et al., 1993)"})


REGISTRY = {
    "digits": load_digits_images,
    "signals": make_signal_dataset,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}


def load_dataset(name: str, **kwargs) -> Dataset:
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset '{name}'. choose from {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


def list_datasets() -> List[str]:
    return sorted(REGISTRY)
