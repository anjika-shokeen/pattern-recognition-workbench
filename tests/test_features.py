import numpy as np
import pytest
from skimage.feature import hog as sk_hog, local_binary_pattern
from skimage.measure import moments_central, moments_hu, moments_normalized

from prwb.datasets import load_digits_images, make_signal_dataset
from prwb.features import (CorrelationPruner, FisherSelector, LDAProjection, MRMRSelector, PCAScratch, Standardizer,
                           build_features, hog)
from prwb.features.image import glcm_features, hu_moments, lbp, lbp_codes, zoning
from prwb.features.signal import extract_signal_features


@pytest.fixture(scope="module")
def digits():
    return load_digits_images().subsample(15, 0)


def test_lbp_matches_skimage_on_interior():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 256, (10, 16, 16)).astype(float)
    mine = lbp_codes(X)
    ref = np.stack([local_binary_pattern(im.astype(np.uint8), 8, 1, method="default") for im in X])
    assert np.array_equal(mine[:, 1:-1, 1:-1], ref[:, 1:-1, 1:-1].astype(int))


def test_hog_shape_and_correlation_with_skimage(digits):
    X = digits.X[:10]
    h = hog(X, cell=4, nbins=9, block=2)
    ref = np.stack([sk_hog(im, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), block_norm="L2-Hys") for im in X])
    assert h.shape == ref.shape
    assert np.corrcoef(h.ravel(), ref.ravel())[0, 1] > 0.85


def test_hog_blocks_are_normalised(digits):
    h = hog(digits.X[:5]).reshape(5, -1, 36)
    norms = np.linalg.norm(h, axis=-1)
    assert np.all(norms <= 1.0 + 1e-6)


def test_hog_translation_of_constant_image_is_zero():
    assert np.allclose(hog(np.full((1, 16, 16), 0.7)), 0.0)


def test_lbp_histograms_sum_to_one(digits):
    f = lbp(digits.X[:6], grid=2)
    assert f.shape == (6, 4 * 59)
    assert np.allclose(f.reshape(6, 4, 59).sum(-1), 1.0)


def test_lbp_riu2_bins():
    f = lbp(np.random.default_rng(1).random((3, 16, 16)), mapping="riu2", grid=1)
    assert f.shape == (3, 10)


def test_hu_moments_match_skimage_magnitudes(digits):
    X = digits.X[:5]
    mine = hu_moments(X)
    ref = np.array([moments_hu(moments_normalized(moments_central(im))) for im in X])
    ref = -np.sign(ref) * np.log10(np.abs(ref) + 1e-30)
    assert np.allclose(np.abs(mine[:, :6]), np.abs(ref[:, :6]), atol=1e-6)
    assert np.allclose(np.abs(mine[:, 6]), np.abs(ref[:, 6]), atol=1e-6)


def test_hu_translation_invariance():
    im = np.zeros((32, 32)); im[8:14, 10:20] = 1.0; im[10:12, 14:24] = 1.0
    shifted = np.roll(im, (5, 4), axis=(0, 1))
    assert np.allclose(hu_moments(im[None]), hu_moments(shifted[None]), atol=1e-8)


def test_glcm_shapes_and_ranges(digits):
    g = glcm_features(digits.X[:8])
    assert g.shape == (8, 12) and np.isfinite(g).all()
    assert np.all(g[:, 3] <= 1.0 + 1e-9)  # energy


def test_zoning_average_pooling():
    X = np.arange(16, dtype=float).reshape(1, 4, 4)
    assert np.allclose(zoning(X, 2), [[2.5, 4.5, 10.5, 12.5]])


def test_signal_features_finite():
    sg = make_signal_dataset(n_per_class=5, seed=0)
    F, names, gi = extract_signal_features(sg.X, ["time", "hjorth", "spectral", "autocorr"], fs=128.0)
    assert F.shape[1] == len(names) == len(gi) and np.isfinite(F).all()


def test_signal_dominant_frequency():
    fs = 128.0
    t = np.arange(256) / fs
    x = np.sin(2 * np.pi * 10 * t)[None]
    F, names, _ = extract_signal_features(x, ["spectral"], fs=fs)
    assert abs(F[0, names.index("dominant_freq")] - 10) < 0.6


def test_pca_matches_eigendecomposition():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6)) @ rng.normal(size=(6, 6))
    p = PCAScratch(3).fit(X)
    ev = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False)))[::-1][:3]
    assert np.allclose(p.explained_variance_, ev)
    assert np.allclose(p.components_ @ p.components_.T, np.eye(3), atol=1e-10)


def test_pca_full_reconstruction_is_exact():
    X = np.random.default_rng(1).normal(size=(50, 5))
    p = PCAScratch(5).fit(X)
    assert np.allclose(p.inverse_transform(p.transform(X)), X)


def test_pca_variance_threshold():
    X = np.random.default_rng(2).normal(size=(100, 10)) * np.arange(1, 11)
    p = PCAScratch(0.9).fit(X)
    assert p.explained_variance_ratio_.sum() >= 0.9


def test_lda_projection_separates_classes():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(m, 1, (60, 4)) for m in (0, 3, 6)])
    y = np.repeat([0, 1, 2], 60)
    Z = LDAProjection().fit(X, y).transform(X)
    assert Z.shape[1] == 2
    assert abs(Z[y == 0, 0].mean() - Z[y == 2, 0].mean()) > 3 * Z[:, 0].std() / 2


def test_standardizer_zero_mean_unit_var():
    X = np.random.default_rng(0).normal(5, 3, (100, 4))
    Z = Standardizer().fit_transform(X)
    assert np.allclose(Z.mean(0), 0, atol=1e-10) and np.allclose(Z.std(0), 1)


def test_feature_selectors_pick_informative_columns():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    X = rng.normal(size=(300, 10))
    X[:, 2] += 3 * y
    X[:, 7] += 2 * y
    for S in (FisherSelector(2), MRMRSelector(2)):
        sup = S.fit(X, y).support_
        assert 2 in sup


def test_correlation_pruner_drops_duplicates():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(100, 3))
    X = np.hstack([a, a[:, :1] * 2 + 1e-3 * rng.normal(size=(100, 1))])
    assert CorrelationPruner(0.95).fit(X).support_.size == 3


def test_build_features_all_sets(digits):
    from prwb.features.builder import FEATURE_SETS
    for s in FEATURE_SETS["image"]:
        fm = build_features(digits, s)
        assert fm.X.shape[0] == digits.n_samples and np.isfinite(fm.X).all()
        assert len(fm.names) == fm.X.shape[1]
