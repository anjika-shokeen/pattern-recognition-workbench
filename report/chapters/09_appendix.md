
```{=latex}
\appendix
```

# Complete Classification Results {#app:cls}

<<ref('tab:allres')>> lists the result of every (data set, feature set, classifier) cell of the benchmark: mean accuracy over the ten held-out folds, macro-F1, Matthews correlation coefficient, one-vs-rest macro ROC-AUC, expected calibration error (ECE) and the mean fit time of one fold in seconds. Within each block the classifiers are sorted by accuracy.

<<tbl(csv('classification_results.csv')[['dataset', 'feature_set', 'dims', 'classifier', 'accuracy', 'macro_f1', 'mcc', 'roc_auc', 'ece', 'fit_time']].sort_values(['dataset', 'feature_set', 'accuracy'], ascending=[True, True, False]).rename(columns={'feature_set': 'features', 'macro_f1': 'macro-F1', 'roc_auc': 'AUC', 'fit_time': 'fit (s)'}), 'Complete classification results (5 x 2 cross-validation).', 'tab:allres', fmts={'accuracy': '{:.4f}', 'macro-F1': '{:.4f}', 'mcc': '{:.4f}', 'AUC': '{:.4f}', 'ece': '{:.4f}', 'fit (s)': '{:.3f}'}, colfmt='llrlrrrrrr', font='scriptsize', long=True)>>

# Additional Result Tables {#app:extra}

## Descriptor-group ablation

<<ref('tab:abl')>> gives the accuracy of three classifiers when each descriptor family of the 617-dimensional hand-crafted set is removed (*without*) or used alone (*only*).

<<tbl(csv('ablation_descriptor_groups.csv').pivot(index='setting', columns='classifier', values='accuracy').reset_index().rename(columns={'setting': 'setting'}), 'Ablation of the descriptor families (CV accuracy, 5 folds).', 'tab:abl', fmts={'Bayes-LDA': '{:.4f}', 'SVM-RBF': '{:.4f}', 'kNN': '{:.4f}'}, colfmt='lrrr', font='small')>>

## Clustering results in all feature spaces

<<tbl(csv('clustering_results.csv')[['space', 'algorithm', 'ARI', 'NMI', 'cluster_acc', 'purity', 'silhouette', 'time_s']].rename(columns={'cluster_acc': 'acc', 'time_s': 'time (s)'}), 'External and internal validity of ten clustering algorithms in five feature spaces of the digits data (k = 10).', 'tab:cluall', fmts={'ARI': '{:.3f}', 'NMI': '{:.3f}', 'acc': '{:.3f}', 'purity': '{:.3f}', 'silhouette': '{:.3f}', 'time (s)': '{:.2f}'}, colfmt='llrrrrrr', font='scriptsize', long=True)>>

## Clustering of the other data sets

<<tbl(csv('clustering_other_datasets.csv')[['dataset', 'algorithm', 'n_clusters', 'ARI', 'NMI', 'cluster_acc', 'silhouette']].rename(columns={'cluster_acc': 'acc', 'n_clusters': 'clusters'}), 'Clustering of the signal, wine and breast-cancer data sets (main feature set, PCA to at most 15 dimensions).', 'tab:cluother', fmts={'ARI': '{:.3f}', 'NMI': '{:.3f}', 'acc': '{:.3f}', 'silhouette': '{:.3f}'}, colfmt='llrrrrr', font='scriptsize', long=True)>>

## Robustness summary

<<tbl(csv('robustness_summary.csv'), 'Mean accuracy over all severity levels and four classifiers, per corruption and feature set (digits).', 'tab:robsum', fmts={'raw': '{:.3f}', 'lbp': '{:.3f}', 'hog': '{:.3f}', 'hog+lbp': '{:.3f}'}, colfmt='lrrrr', font='small')>>

## Descriptive statistics of the wine data

<<tbl(csv('descriptive_wine_raw.csv')[['feature', 'mean', 'std', 'min', 'median', 'max', 'skewness', 'kurtosis']], 'Descriptive statistics of the 13 wine features.', 'tab:descwine', fmts={'mean': '{:.3f}', 'std': '{:.3f}', 'min': '{:.3f}', 'median': '{:.3f}', 'max': '{:.3f}', 'skewness': '{:.2f}', 'kurtosis': '{:.2f}'}, colfmt='lrrrrrrr', font='scriptsize')>>

# Experiment Configuration {#app:config}

The protocol of the full benchmark is defined by `configs/default.yaml`:

```yaml
<<readfile('configs/default.yaml')>>
```

# Selected Source Listings {#app:code}

The following listings are extracted automatically from the repository at the time the report is built. They show the core of five of the algorithms discussed in Chapter 3.

**HOG descriptor** (`prwb/features/image.py`)

```python
<<source_of('prwb/features/image.py', 'hog_cells')>>

<<source_of('prwb/features/image.py', 'hog')>>
```

**Gaussian Bayes classifier: fit and log-posterior** (`prwb/classifiers/scratch.py`)

```python
<<source_of('prwb/classifiers/scratch.py', 'GaussianBayes.fit')>>

<<source_of('prwb/classifiers/scratch.py', 'GaussianBayes.decision_log_posterior')>>
```

**k-means++ seeding and Lloyd iterations** (`prwb/clustering/scratch.py`)

```python
<<source_of('prwb/clustering/scratch.py', 'KMeansScratch._seed')>>

<<source_of('prwb/clustering/scratch.py', 'KMeansScratch._run')>>
```

**EM for Gaussian mixtures: E-step and M-step** (`prwb/clustering/scratch.py`)

```python
<<source_of('prwb/clustering/scratch.py', 'GMMScratch._estep')>>

<<source_of('prwb/clustering/scratch.py', 'GMMScratch._mstep')>>
```

**Statistical procedures** (`prwb/evaluation/metrics.py`, `prwb/stats/analysis.py`)

```python
<<source_of('prwb/evaluation/metrics.py', 'mcnemar_test')>>

<<source_of('prwb/evaluation/metrics.py', 'corrected_resampled_ttest')>>

<<source_of('prwb/evaluation/metrics.py', 'friedman_nemenyi')>>

<<source_of('prwb/stats/analysis.py', 'benjamini_hochberg')>>
```

# Reproduction Instructions {#app:repro}

The complete source code, results and this report are available in the public repository <<esc(meta['repository'])>>. To reproduce the study:

```
git clone <repository-url> && cd pattern-recognition-workbench
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

python -m pytest                                   # <<S['n_tests']>> tests
python -m prwb datasets                            # list data sets and models
python scripts/run_experiments.py --quick          # smoke run of the whole protocol
python scripts/run_experiments.py                  # full protocol -> results/
streamlit run app/streamlit_app.py                 # dashboard
python notebooks/make_notebook.py                  # execute the demonstration notebook
python report/build_report.py                      # rebuild this report (pandoc + XeLaTeX)
```

The experiments are seeded (seed 42), so a re-run reproduces the tables of this report exactly on the same library versions (<<ref('tab:stack')>>); small last-digit differences can occur with other BLAS back-ends.
