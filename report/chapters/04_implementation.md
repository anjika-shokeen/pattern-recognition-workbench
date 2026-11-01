
# Implementation Description

This chapter describes how the methodology of Chapter 3 was turned into software: the technology stack, the organisation of the code, the implementation techniques that make the from-scratch algorithms fast and numerically safe, the validation strategy, the user interfaces and the setup of the experimental study. The results themselves are presented in <<ref('sec:results')>> and analysed in <<ref('sec:analysis')>>.

## Technology Stack

The workbench is written in Python and depends only on the standard scientific stack, so that it installs with a single `pip` command and runs on a laptop without a GPU. <<ref('tab:stack')>> lists the components and the versions used to produce this report.

<<tbl(env_df(S), 'Technology stack (versions used to generate the results in this report).', 'tab:stack', colfmt='llp{7.2cm}', font='small')>>

Python was chosen because it combines readable code with vectorised numerical kernels (NumPy <<cite('harris2020')>>, SciPy <<cite('virtanen2020')>>) and because its ecosystem provides trusted reference implementations (scikit-learn <<cite('pedregosa2011')>>, scikit-image <<cite('vanderwalt2014')>>) against which the from-scratch code can be tested. Result tables are handled with pandas <<cite('mckinney2010')>>, and Matplotlib <<cite('hunter2007')>> produces every figure, including the architecture diagrams, which are drawn programmatically so that they stay consistent with the code. The development workflow uses *Git* for version control and *GitHub Actions* for continuous integration (the test suite and a quick end-to-end run are executed on every push, with Python 3.10 and 3.12), a `Makefile` for the common tasks, and a `Dockerfile` for a container image. The report itself is generated from Markdown templates with pandoc and XeLaTeX, so that each number in the text is read from the result files.

## Organisation of the Code

The repository is organised so that the directory tree mirrors the stages of <<ref('fig:diagram_high_level')>>:

```
pattern-recognition-workbench/
|-- prwb/                      # the Python package
|   |-- datasets.py            # image / signal / tabular data sets, Dataset container
|   |-- preprocess.py          # digit normalisation, noise / occlusion / rotation / shift
|   |-- features/              # image.py, signal.py, transforms.py, selection.py, builder.py
|   |-- stats/analysis.py      # normality, ANOVA, FDR, correlation, Bhattacharyya, Mahalanobis
|   |-- classifiers/           # scratch.py (6 model families), zoo.py (15 named models)
|   |-- clustering/            # scratch.py (k-means++, EM-GMM, FCM), zoo.py (+ k selection)
|   |-- evaluation/            # metrics.py (CV, tests, validity), plots.py (figures, diagrams)
|   |-- pipeline.py            # facade: quick_classify / quick_cluster / quick_stats / train
|   |-- experiments.py         # class Workbench: the complete experimental protocol
|   `-- cli.py                 # python -m prwb <command>
|-- app/streamlit_app.py       # interactive dashboard
|-- scripts/run_experiments.py # reproduce all tables and figures
|-- configs/default.yaml       # experiment protocol
|-- tests/                     # automated tests
|-- notebooks/                 # demonstration notebook
|-- results/                   # tables (CSV), figures (PNG), summary.json
|-- report/                    # this report (Markdown templates -> PDF)
`-- Makefile, Dockerfile, .github/workflows/ci.yml, requirements.txt, pyproject.toml
```

The code base comprises about <<S['loc']['package']>> lines in the package, <<S['loc']['tests']>> lines of tests and <<S['loc']['app'] + S['loc']['scripts']>> lines of scripts and dashboard (<<ref('tab:loc')>>). In total <<S['n_scratch']>> algorithms are implemented from scratch (<<ref('tab:scratch')>>).

<<tbl(loc_df(S), 'Size of the code base (physical lines of Python).', 'tab:loc', colfmt='lrr', font='small')>>

<<tbl(scratch_df(S), 'Algorithms implemented from scratch.', 'tab:scratch', colfmt='>{\\raggedright}p{3.6cm}>{\\raggedright\\arraybackslash}p{10.3cm}', font='small')>>

## Implementation Techniques

Writing an algorithm "from scratch" is only valuable if the result is fast enough to be used and numerically trustworthy. The following techniques were applied throughout.

### Vectorisation

Loops over pixels or patterns are the main cause of slow Python code, so all descriptors operate on a *batch* of shape $(n,h,w)$. The HOG cell histograms, for example, are accumulated for the whole batch with two calls to `numpy.bincount`, using a flat index that encodes (image, cell row, cell column, bin):

```python
base = (((nidx * ny + cy) * nx + cx) * nbins)          # flat index of the cell histogram
size = n * ny * nx * nbins
hist  = np.bincount((base + b0).ravel(), weights=(mag * (1 - frac)).ravel(), minlength=size)
hist += np.bincount((base + b1).ravel(), weights=(mag * frac).ravel(),       minlength=size)
return hist.reshape(n, ny, nx, nbins)
```

The $k$-NN, Parzen and clustering code uses the identity $\lVert a-b\rVert^2=\lVert a\rVert^2+\lVert b\rVert^2-2a^\top b$ so that the full distance matrix is one matrix multiplication, evaluated in chunks of 256--512 queries to bound memory.

### Numerical Stability

* **Log-domain arithmetic.** In 64--617 dimensions the Gaussian densities underflow to zero. Both the Bayes classifier and the EM algorithm therefore work with log-densities and normalise with `logsumexp`.
* **Cholesky instead of inversion.** The Mahalanobis term of a Gaussian is evaluated by a triangular solve with the Cholesky factor, never by forming $\Sigma^{-1}$:

```python
z = np.linalg.solve(self._chol[c], (X - self.means_[c]).T)       # L^{-1} (x - mu)
maha = (z ** 2).sum(0)
out[:, c] = -0.5 * (maha + self._logdet[c] + d * np.log(2 * np.pi)) + self.log_priors_[c]
```

* **Covariance shrinkage and regularisation.** Sample covariances of 560-dimensional descriptors computed from 150 patterns per class are singular. Shrinkage towards a scaled identity (Bayes), a ridge term $\varepsilon I$ (EM) and a jitter $10^{-6}I$ guarantee positive definiteness.
* **Round-off guards.** Hu invariants that are analytically zero are clipped to zero before the logarithm; empty $k$-means clusters are re-seeded; the softmax subtracts the row maximum.

### Gradient-based Learners

Softmax regression and the MLP share a small `_Adam` optimiser class. The MLP caches the activations of the forward pass and propagates the error backwards with the recursion of Chapter 3; the correctness of the gradients is established by a *numerical gradient check* in the test suite (the analytic gradient of every weight matrix is compared with central finite differences). Inverted dropout, L2 weight decay and early stopping on a validation split (restoring the best weights) are implemented in about 100 lines.

### Leakage-free Pipelines

Every classifier in the zoo is a `Pipeline([("scale", Standardizer()), ("clf", model)])`; the evaluation routine clones the pipeline for each fold, so the means and standard deviations are always those of the training fold. Where PCA or a feature selector is used (<<ref('sec:results')>>) it is placed *inside* the pipeline or inside the fold loop for the same reason.

### Parallel Evaluation

The classifiers of one (data set, feature set) cell are evaluated in parallel with `joblib.Parallel` (`n_jobs=2`); each worker receives a cloned estimator and returns its fold table and out-of-fold predictions. Because the seeds are fixed and the folds are generated before the workers start, the results do not depend on the number of workers.

## Software Validation

A scientific code base is only as credible as its tests. The suite contains <<S['n_tests']>> automated tests (collected by `pytest`) that fall into four categories.

<<tbl(pd.DataFrame([
 ['Agreement with a reference implementation', 'k-NN predictions equal scikit-learn; LDA and QDA predictions equal scikit-learn; HOG correlates with scikit-image HOG; LBP codes equal scikit-image on interior pixels; Hu invariants match scikit-image in magnitude; ANOVA equals SciPy; BH q-values equal the step-up definition'],
 ['Analytic properties', 'PCA axes equal the eigenvectors of the covariance matrix; full-rank PCA reconstructs exactly; HOG blocks have unit norm; LBP histograms sum to one; Hu invariants are translation invariant; posteriors sum to one; ECE is 0 for a perfectly calibrated and large for an over-confident model'],
 ['Behavioural / numerical', 'MLP gradients equal finite differences; MLP learns XOR; softmax loss decreases; EM log-likelihood is monotone; BIC selects the true number of components; k-means++ is not worse than random seeding; the corrected t-test is more conservative than the naive one; metrics are permutation invariant'],
 ['Integration', 'the cross-validation routine does not leak test statistics and its out-of-fold predictions cover every sample; the CLI commands run; a saved model recognises a rendered digit; the experiment driver writes its artefacts; the data sets are deterministic for a fixed seed'],
], columns=['Category', 'Examples']), 'Categories of automated tests.', 'tab:tests', colfmt='>{\\raggedright}p{3.5cm}>{\\raggedright\\arraybackslash}p{10.4cm}', font='small')>>

Comparing against independent implementations also gives a way of *finding* subtle bugs: the Hu-moment test, for example, exposed that the logarithmic transform amplifies floating-point round-off in invariants that are analytically zero, which led to the guard described in Chapter 3.

## User Interfaces

**Command line.** `python -m prwb <command>` exposes the stages individually: `datasets` lists the data sets, feature sets, classifiers and clusterers; `features` computes a feature matrix; `stats` prints the ANOVA ranking, correlation and separability summaries; `classify` runs cross-validated classification and prints a per-class report; `cluster` runs a clustering algorithm and prints the validity indices; `benchmark` executes the full protocol; `train` fits and saves the deployable digit recogniser and `predict` recognises a hand-written digit in an image file. For example:

```
python -m prwb classify -d digits -f hog+lbp -c SVM-RBF --folds 5
python -m prwb cluster  -d digits -f hog+lbp -a Agglomerative-Ward
python -m prwb stats    -d wine   -f raw --top 10
```

**Dashboard.** `streamlit run app/streamlit_app.py` starts an interactive application with five tabs: *Data and features* (samples, HOG visualisation, feature matrix), *Statistics* (ANOVA ranking, correlation map, separability), *Classification* (choose a model and cross-validation, view the confusion matrix and metrics), *Clustering* (choose an algorithm and $k$, view the embedding and validity indices) and *Recognise a digit* (upload a picture and obtain the predicted digit with class probabilities). Screenshots are presented in <<ref('sec:results')>>.

**Programmatic API.** The facade `prwb.pipeline` offers `quick_classify`, `quick_cluster`, `quick_stats` and `train_final_model`, and the notebook `notebooks/PatternRecognitionWorkbench.ipynb` demonstrates them.

## Reproducibility and Deployment

* **Configuration.** `configs/default.yaml` states the data sets, the feature sets, the cross-validation parameters, the clustering parameters and the robustness protocol. Changing a line changes the study.
* **Seeds.** One global seed (42) drives the data generation, the fold assignment and the stochastic algorithms; folds are generated once and shared by all classifiers, which makes the comparisons *paired*.
* **Make targets.** `make test`, `make quick` (a smoke run of the whole protocol), `make benchmark` (the full protocol), `make app`, `make notebook` and `make report`.
* **Continuous integration.** The workflow in `.github/workflows/ci.yml` installs the package, runs the tests and executes the quick protocol on Python 3.10 and 3.12.
* **Container.** The `Dockerfile` builds an image whose default command starts the dashboard on port 8501 (`docker build -t prwb .` followed by `docker run -p 8501:8501 prwb`); the test suite can be run in the same image with `docker run prwb python -m pytest`.
* **Deployable model.** `prwb train` serialises the best digit pipeline (HOG + LBP features, RBF-SVM) with `joblib`; `prwb predict image.png` applies the same geometric normalisation, feature extraction and model to a new picture.

## Experimental Setup

The experiments of <<ref('sec:results')>> were all produced by the single command `python scripts/run_experiments.py --config configs/default.yaml` on a machine with two CPU cores and no GPU. <<ref('tab:experiments')>> lists the nine experiments, which together answer the questions posed by the objectives O1--O8.

<<tbl(pd.DataFrame([
 ['E1', 'Overview', 'Data sets, class balance, sample patterns, architecture diagrams'],
 ['E2', 'Feature visuals', 'HOG / LBP visualisation, PCA scree and reconstruction, PCA / LDA / t-SNE embeddings'],
 ['E3', 'Statistical analysis', 'Normality, ANOVA + FDR, correlation, Bhattacharyya, Mahalanobis, descriptor-family effect sizes'],
 ['E4', 'Classification grid', 'Up to 15 classifiers x 15 (data set, feature set) blocks, 5 x 2 CV, metrics, Friedman-Nemenyi, McNemar, calibration'],
 ['E5', 'Sweeps', 'k and metric for k-NN, C-gamma grid for SVM, Parzen bandwidth, Bayes shrinkage, MLP training and architecture, learning curves, PCA dimension, feature selection'],
 ['E6', 'Ablation', 'Removing / keeping each descriptor family of the hand-crafted set'],
 ['E7', 'Robustness', 'Gaussian noise, salt-and-pepper, occlusion, rotation and shift at increasing severity'],
 ['E8', 'Clustering', '10 algorithms x 5 feature spaces, choice of k, initialisation, EM convergence, fuzzifier, DBSCAN sensitivity, stability'],
 ['E9', 'Other data sets', 'Clustering of signals, wine and breast cancer'],
], columns=['ID', 'Experiment', 'Content']), 'The experimental protocol.', 'tab:experiments', colfmt='>{\\raggedright}p{1cm}>{\\raggedright}p{3cm}>{\\raggedright\\arraybackslash}p{9.8cm}', font='small')>>

The complete protocol finishes in about <<((S.get('runtime_seconds', 1800) / 60)|round(0)|int)>> minutes. The classification grid evaluates <<S['n_cls_cells']>> (feature set, classifier) cells; each cell yields ten held-out scores (five folds, two repeats), i.e. about <<S['n_cls_cells'] * 10>> trained models, besides the models of the sweeps and ablations.
