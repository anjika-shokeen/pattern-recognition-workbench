"""Interactive dashboard:  streamlit run app/streamlit_app.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from prwb import datasets as D
from prwb.classifiers import get_classifiers
from prwb.clustering import estimate_eps, get_clusterers
from prwb.evaluation import plots as pl
from prwb.evaluation.metrics import clustering_scores, cross_validate_model
from prwb.features import PCAScratch, Standardizer, build_features
from prwb.features.builder import FEATURE_SETS
from prwb.features.image import extract_image_features, hog_cells
from prwb.pipeline import embed_2d
from prwb.preprocess import to_digit_image
from prwb.stats import anova_table, class_separability

st.set_page_config(page_title="Pattern Recognition Workbench", layout="wide")
st.title("End-to-End Pattern Recognition Workbench")
st.caption("Feature extraction - statistical analysis - classification - clustering - performance evaluation")


@st.cache_data(show_spinner=False)
def get_ds(name, per_class):
    ds = D.load_dataset(name)
    return ds.subsample(per_class, 0) if per_class and per_class < ds.class_counts().min() else ds


@st.cache_data(show_spinner=False)
def get_fm(name, set_name, per_class):
    return build_features(get_ds(name, per_class), set_name)


with st.sidebar:
    st.header("Problem setup")
    dname = st.selectbox("Dataset", D.list_datasets(), index=0)
    per_class = st.slider("Max samples per class", 20, 200, 80, 10)
    ds = get_ds(dname, per_class)
    fsets = list(FEATURE_SETS[ds.modality])
    fset = st.selectbox("Feature set", fsets, index=fsets.index("hog+lbp") if "hog+lbp" in fsets else len(fsets) - 1)
    st.info(ds.summary())

fm = get_fm(dname, fset, per_class)
tabs = st.tabs(["Data & features", "Statistics", "Classification", "Clustering", "Recognise a digit"])

# ------------------------------------------------------------------ data & features
with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        if ds.modality == "image":
            fig_path = pl.plot_image_grid(ds.X, ds.y, ds.class_names, 8, "/tmp/_grid.png")
            st.image(fig_path, caption="class samples")
        elif ds.modality == "signal":
            st.image(pl.plot_signal_examples(ds.X, ds.y, ds.class_names, ds.meta["fs"], "/tmp/_sig.png"))
        else:
            st.dataframe(pd.DataFrame(ds.X, columns=ds.feature_names).head(10))
    with c2:
        st.metric("Feature matrix", f"{fm.shape[0]} x {fm.shape[1]}")
        st.write("Descriptor groups:", ", ".join(fm.groups))
        E = embed_2d(Standardizer().fit_transform(fm.X), "pca")
        st.image(pl.plot_embeddings({"PCA 2-D": E}, ds.y, ds.class_names, "/tmp/_emb.png"))
    if ds.modality == "image":
        idx = st.slider("Inspect sample #", 0, ds.n_samples - 1, 3)
        cells = hog_cells(ds.X[idx][None], 4, 9)[0]
        st.image(pl.plot_hog_visual(ds.X[idx], cells, "/tmp/_hog.png"), caption=f"HOG of sample {idx} (label {ds.class_names[ds.y[idx]]})")

# ------------------------------------------------------------------ statistics
with tabs[1]:
    an = anova_table(fm.X, ds.y, fm.names)
    st.subheader("One-way ANOVA ranking (Benjamini-Hochberg corrected)")
    st.dataframe(an.head(25).round(4), width="stretch")
    st.image(pl.plot_anova_top(an, "/tmp/_anova.png", 15))
    Z = Standardizer().fit_transform(fm.X)
    Z = PCAScratch(min(20, Z.shape[1] - 1)).fit_transform(Z) if Z.shape[1] > 21 else Z
    st.json({k: round(v, 4) for k, v in class_separability(Z, ds.y).items()})

# ------------------------------------------------------------------ classification
with tabs[2]:
    zoo = list(get_classifiers().keys())
    clf = st.selectbox("Classifier", zoo, index=zoo.index("SVM-RBF"))
    folds = st.slider("CV folds", 3, 10, 5)
    if st.button("Run cross-validation", type="primary"):
        with st.spinner("training..."):
            r = cross_validate_model(get_classifiers(names=[clf])[clf], fm.X, ds.y, folds, 1)
        f = r["folds"]
        c = st.columns(4)
        c[0].metric("Accuracy", f"{f.accuracy.mean():.4f}", f"+/- {f.accuracy.std():.4f}")
        c[1].metric("Macro F1", f"{f.macro_f1.mean():.4f}")
        c[2].metric("MCC", f"{f.mcc.mean():.4f}")
        c[3].metric("fit time / fold", f"{r['fit_time']:.2f}s")
        a, b = st.columns(2)
        a.image(pl.plot_confusion(ds.y, r["oof_pred"], ds.class_names, "/tmp/_cm.png", clf))
        if r["oof_proba"] is not None:
            b.image(pl.plot_roc_multiclass(ds.y, r["oof_proba"], ds.class_names, "/tmp/_roc.png", f"({clf})"))
        st.dataframe(f.round(4))

# ------------------------------------------------------------------ clustering
with tabs[3]:
    k = st.slider("Number of clusters k", 2, 15, ds.n_classes)
    zc = Standardizer().fit_transform(fm.X)
    zc = PCAScratch(min(30, zc.shape[1] - 1)).fit_transform(zc) if zc.shape[1] > 31 else zc
    algos = get_clusterers(k, 0, estimate_eps(zc, 5, 0.7))
    alg = st.selectbox("Algorithm", list(algos))
    if st.button("Cluster"):
        lab = algos[alg].fit_predict(zc)
        sc = clustering_scores(zc, ds.y, lab)
        st.dataframe(pd.Series(sc).round(4).to_frame("value").T)
        E = embed_2d(zc, "pca")
        st.image(pl.plot_cluster_scatter(E, ds.y, lab, "/tmp/_cs.png"))
        st.image(pl.plot_contingency(ds.y, lab, ds.class_names, "/tmp/_ct.png", alg))

# ------------------------------------------------------------------ recognise
with tabs[4]:
    st.write("Upload a picture of a single hand-written digit (dark stroke on a light page, or vice-versa).")
    up = st.file_uploader("image", type=["png", "jpg", "jpeg"])
    if up is not None:
        dg = get_ds("digits", 200)
        fmd = get_fm("digits", "hog+lbp", 200)
        model = get_classifiers(names=["SVM-RBF"])["SVM-RBF"].fit(fmd.X, dg.y)
        img = to_digit_image(np.asarray(Image.open(up).convert("L")))
        F = extract_image_features(img[None], FEATURE_SETS["image"]["hog+lbp"])[0]
        p = model.predict_proba(F)[0]
        c1, c2 = st.columns(2)
        c1.image(np.kron(img, np.ones((12, 12))), caption="normalised 16x16 input", clamp=True)
        c2.metric("Prediction", int(np.argmax(p)), f"confidence {p.max():.2%}")
        c2.bar_chart(pd.Series(p, index=[str(i) for i in range(10)]))
