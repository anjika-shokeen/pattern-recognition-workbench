"""PRWB - End-to-End Pattern Recognition Workbench.

A modular toolkit that integrates the complete pattern-recognition chain:

    data -> feature extraction -> statistical analysis -> classification
         -> clustering -> performance evaluation -> reporting

Core algorithms (HOG, LBP, GLCM, Hu moments, PCA, Fisher LDA, k-NN, Gaussian
Bayes, Parzen windows, softmax regression, an MLP with back-propagation,
k-means++, EM-GMM and fuzzy c-means) are implemented from scratch with NumPy
and exposed through a scikit-learn compatible API so they can be benchmarked
side by side with reference implementations.
"""
__version__ = "1.0.0"
__all__ = [
    "datasets",
    "features",
    "stats",
    "classifiers",
    "clustering",
    "evaluation",
    "pipeline",
]
