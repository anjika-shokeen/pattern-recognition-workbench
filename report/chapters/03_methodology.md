
# Proposed Methodology

This chapter presents the proposed workbench at three levels of detail: a *high-level view* that shows the stages and the data that flow between them, a *detailed architecture* that specifies every module together with the mathematics of the algorithms it implements, and an *operational diagram* that follows one complete run of the experimental protocol.

## High-Level View

The workbench transforms raw patterns into decisions and into evidence about the quality of those decisions. <<ref('fig:diagram_high_level')>> shows the six stages and the package that implements each of them. Data flow from left to right: patterns are acquired and pre-processed, converted into feature vectors, analysed statistically, fed to a classifier or a clusterer, and finally evaluated. Every stage is also reachable from the interfaces (command line, dashboard and experiment script), which write tables, figures and a deployable model.

<<fig('diagram_high_level.png', 'High-level view of the End-to-End Pattern Recognition Workbench: six processing stages and the package that realises each one.', 0.98)>>

Four design principles govern the architecture.

1. **Modularity.** Each stage is a package with a narrow interface, so that a descriptor, a classifier or a clustering algorithm can be replaced without touching the rest. All estimators follow the scikit-learn contract (`fit`, `transform`, `predict`, `predict_proba`), which makes them composable with `Pipeline` and interchangeable with reference implementations.
2. **Verifiability.** Every algorithm written from scratch is accompanied by a unit test that compares it with an independent implementation (scikit-learn, scikit-image, SciPy), with an analytic value, or with a numerical derivative.
3. **Leakage-free evaluation.** Every data-dependent step (scaling, PCA, feature selection, density bandwidth) is *inside* the cross-validation loop and is re-fitted on the training fold only.
4. **Reproducibility.** All randomness is seeded; data sets are bundled with scikit-learn or generated deterministically; one command (`python scripts/run_experiments.py`) regenerates every number and figure of this report.

## Data Sets

Three modalities are represented so that the claim of domain independence can be tested (<<ref('tab:datasets')>>).

<<tbl(csv('dataset_overview.csv').rename(columns={'pattern_shape': 'pattern shape', 'min_class': 'min class', 'max_class': 'max class', 'raw_dims': 'raw dims'}), 'The four data sets used in the study.', 'tab:datasets', colfmt='lllrrrrr')>>

* **Handwritten digits (image).** The UCI *Optical Recognition of Handwritten Digits* set <<cite('alpaydin1998')>> contains 1,797 greyscale images of the digits 0--9 written by 43 people, originally $8\times 8$ pixels. The images are rescaled to $[0,1]$ and bicubically up-sampled to $16\times16$ so that gradient and texture operators have enough spatial support. The classes are almost perfectly balanced (174--183 per class).
* **Synthetic waveforms (signal).** Six waveform families -- sine, square, sawtooth, linear chirp, amplitude-modulated tone and damped oscillation -- are generated at a sampling rate of 128 Hz, 256 samples each (200 per class). The fundamental frequency ($4$--$12$ Hz, overlapping between classes), amplitude, phase, baseline drift and white noise ($\sigma=0.45$) are random. Because the frequency ranges overlap, the problem cannot be solved by locating a single spectral peak; the classifier must exploit *shape*: harmonics, envelope, sweep and decay.
* **Wine (tabular).** 178 samples of Italian wines described by 13 chemical analyses, three cultivars <<cite('forina1988')>>.
* **Wisconsin Diagnostic Breast Cancer (tabular).** 569 samples, 30 features computed from digitised images of fine-needle aspirates of breast masses, two classes (benign / malignant) <<cite('street1993')>>.

<<figrow([('digits_samples.png', 'Digit images (16 x 16)'), ('signal_examples.png', 'Waveform families')], 'Examples of the image and signal patterns.', 'fig:examples', width=0.485)>>

## Detailed Architecture

<<ref('fig:diagram_architecture')>> shows the layered structure of the package `prwb`. The *interface layer* (command line, dashboard, experiment script) calls the *orchestration layer* (`pipeline.py` for single runs, `experiments.py` for the full protocol), which composes the *core modules*. All core modules rest on a shared contract, the scikit-learn estimator API, and on the scientific Python foundation.

<<fig('diagram_architecture.png', 'Detailed layered architecture of the package prwb.', 0.98)>>

The class structure that realises the shared contract is shown in <<ref('fig:diagram_classes')>>: transformers inherit from `TransformerMixin`, classifiers from `ClassifierMixin` and clusterers from `ClusterMixin`, all on top of `BaseEstimator`, which provides parameter introspection and cloning.

<<fig('diagram_classes.png', 'Class structure of the package: every algorithm is a scikit-learn estimator.', 0.92)>>

### Pre-processing

Feature scales differ by orders of magnitude (a variance next to a ratio of band powers), and distance-based and gradient-based learners are sensitive to this. The `Standardizer` maps each feature to zero mean and unit variance,

$$ z_j = \frac{x_j-\mu_j}{\sigma_j}, $$

with $\mu_j$ and $\sigma_j$ estimated on the training data only (constant columns are merely centred). For the handwritten-digit images an additional geometric normalisation (`to_digit_image`) centres, scales and inverts an arbitrary user-supplied picture into the format of the training data, which makes the deployable recogniser usable on photographs of digits.

### Image Descriptors

<<fig('diagram_feature_pipeline.png', 'Feature-extraction and transformation pipeline: the raw pattern is converted into a named feature matrix by one or more descriptor groups.', 0.98)>>

**Histogram of Oriented Gradients.** The gradient is computed with the centred derivative $[-1,0,1]$, giving magnitude $m(x,y)=\sqrt{g_x^2+g_y^2}$ and unsigned orientation $\theta(x,y)=\operatorname{atan2}(g_y,g_x)\bmod\pi$. The image is divided into $4\times4$-pixel cells; in every cell the orientation range $[0,\pi)$ is quantised into $B=9$ bins and each pixel votes with weight $m(x,y)$, *split linearly between the two nearest bin centres* to avoid aliasing:

$$ b = \frac{\theta}{\pi/B}-\tfrac12,\qquad h[\lfloor b\rfloor] \mathrel{+}= m\,(1-f),\quad h[\lfloor b\rfloor+1] \mathrel{+}= m\,f,\quad f=b-\lfloor b\rfloor. $$

Overlapping blocks of $2\times2$ cells (stride one cell) are concatenated and normalised with the L2-Hys scheme: $\mathbf v\leftarrow \mathbf v/\sqrt{\lVert\mathbf v\rVert^2+\varepsilon^2}$, clipped at $0.2$, and renormalised. For a $16\times16$ image this gives $3\times3$ blocks $\times\,36$ values $=324$ features. The implementation is fully vectorised: the votes of the whole batch are accumulated with a single `numpy.bincount` call.

<<fig('hog_visual.png', 'HOG of a digit 8: per-cell orientation histograms drawn as oriented glyphs.', 0.8)>>

**Local Binary Patterns.** For each pixel the $P=8$ neighbours on a circle of radius $R=1$ (bilinearly interpolated) are thresholded against the centre value and the bits are packed into a code,

$$ \mathrm{LBP}_{P,R}=\sum_{p=0}^{P-1} s\,(g_p-g_c)\,2^{p},\qquad s(u)=\begin{cases}1&u\ge0\\0&u<0.\end{cases} $$

Codes with at most two circular 0/1 transitions ("uniform" patterns: spots, edges, corners) receive individual histogram bins and all the others share one bin, which reduces $2^8=256$ codes to $59$ bins. The image is divided into a $2\times2$ grid and the four normalised histograms are concatenated ($4\times59=236$ features), which preserves coarse spatial layout. A rotation-invariant mapping (`riu2`, $P+2$ bins) is also available.

<<fig('lbp_visual.png', 'LBP of a digit 8: raw codes and the uniform-pattern map used to build the histogram.', 0.8)>>

**Grey-level co-occurrence statistics.** After quantisation to $G=8$ levels, the symmetric, normalised co-occurrence matrix $P_{\delta,\phi}(i,j)$ is computed at distance $\delta=1$ for the four directions $\phi\in\{0^\circ,45^\circ,90^\circ,135^\circ\}$. From each matrix six Haralick properties are derived,

$$ \text{contrast}=\sum_{i,j}(i-j)^2P_{ij},\quad \text{homogeneity}=\sum_{i,j}\frac{P_{ij}}{1+(i-j)^2},\quad \text{energy}=\sqrt{\sum_{i,j}P_{ij}^2},\quad \text{entropy}=-\sum_{i,j}P_{ij}\log_2 P_{ij}, $$

together with dissimilarity and correlation. The mean and the range over the four directions are kept, giving a rotation-tolerant vector of $2\times6=12$ features.

**Hu moment invariants.** From the normalised central moments $\eta_{pq}=\mu_{pq}/\mu_{00}^{1+(p+q)/2}$ the seven invariants of Hu are computed, for instance

$$ \phi_1=\eta_{20}+\eta_{02},\qquad \phi_2=(\eta_{20}-\eta_{02})^2+4\eta_{11}^2,\qquad \phi_3=(\eta_{30}-3\eta_{12})^2+(3\eta_{21}-\eta_{03})^2 . $$

They are invariant to translation, scale and rotation, and their enormous dynamic range is compressed with $-\operatorname{sign}(\phi)\log_{10}|\phi|$. Invariants that are analytically zero but evaluate to floating-point noise ($\sim10^{-23}$) are set to exactly zero before the logarithm; without this guard the transform amplifies round-off into a spurious value.

**Simple descriptors.** *Zoning* is average pooling over a $4\times4$ grid (16 features), *projection profiles* are the row and column sums (32), and *intensity statistics* are mean, standard deviation, skewness, kurtosis, histogram entropy and ink ratio (6). The composite set `handcrafted` concatenates HOG, LBP, GLCM, Hu, profiles and statistics ($324+236+12+7+32+6=617$ features).

### Signal Descriptors

For a signal $x[n]$, $n=0,\dots,L-1$ sampled at $f_s$ the following groups are computed.

* **Time domain (9):** mean, standard deviation, RMS, peak-to-peak, crest factor $\max|x|/\mathrm{RMS}$, zero-crossing rate, mean absolute first difference, skewness and kurtosis.
* **Hjorth parameters (3):** activity $=\operatorname{var}(x)$, mobility $=\sqrt{\operatorname{var}(x')/\operatorname{var}(x)}$ and complexity $=\text{mobility}(x')/\text{mobility}(x)$ <<cite('hjorth1970')>>.
* **Spectral (16):** from the Hann-windowed periodogram $P_k=|\mathrm{FFT}(w\cdot(x-\bar x))_k|^2$ and its normalisation $p_k=P_k/\sum_iP_i$: centroid $\sum_kf_kp_k$, spread, normalised entropy $-\sum_kp_k\log_2p_k/\log_2K$, flatness (geometric over arithmetic mean of $P_k$), dominant frequency, 85 % roll-off frequency, the energy ratios at two and three times the dominant bin (*harmonic ratios*) and eight linearly spaced band powers.
* **Auto-correlation (5):** $r[\tau]$ is obtained through the FFT (Wiener--Khinchin theorem); the lag and height of the first peak after the first zero crossing, the mean decay over the first quarter, the late-to-early envelope ratio and the envelope variation. The last two distinguish decaying and amplitude-modulated waveforms from stationary ones.

The composite set `handcrafted` has $9+3+16+5=33$ features. A raw 256-sample set is kept as a baseline.

### Dimensionality Reduction and Feature Selection

**PCA.** With $\tilde X=X-\bar X$ the thin singular value decomposition $\tilde X=U\Sigma V^\top$ gives the principal axes (rows of $V^\top$) and the variances $\lambda_i=\sigma_i^2/(n-1)$. Projection on the first $k$ axes is $Z=\tilde XV_k$ and reconstruction is $\hat X=ZV_k^\top+\bar X$; $k$ may be given directly or as the smallest value that explains a prescribed fraction of the variance.

**Fisher LDA.** With within-class scatter $S_W=\sum_c\sum_{i\in c}(\mathbf x_i-\boldsymbol\mu_c)(\mathbf x_i-\boldsymbol\mu_c)^\top$ and between-class scatter $S_B=\sum_cn_c(\boldsymbol\mu_c-\boldsymbol\mu)(\boldsymbol\mu_c-\boldsymbol\mu)^\top$ the discriminant directions solve the generalised eigenproblem

$$ S_B\mathbf v=\lambda S_W\mathbf v,$$

solved with `scipy.linalg.eigh` after a small shrinkage of $S_W$ towards its trace-scaled identity; at most $C-1$ directions have non-zero eigenvalues.

**Filter selection.** The *Fisher score* of feature $j$ is $F_j=\sum_cn_c(\mu_{cj}-\mu_j)^2/\sum_cn_c\sigma_{cj}^2$. The *mutual-information* ranking uses the nearest-neighbour estimator of $I(f_j;y)$. The *mRMR* selector greedily adds the feature that maximises relevance minus redundancy,

$$ j^\ast=\arg\max_{j\notin S}\Bigl[I(f_j;y)-2\,\overline{I}\;\tfrac{1}{|S|}\sum_{i\in S}|\rho_{ij}|\Bigr], $$

where $\overline I$ is the mean relevance (so that both terms have the same scale) and $\rho_{ij}$ is the Pearson correlation. A `CorrelationPruner` drops one feature of every pair with $|\rho|>0.95$. In experiments, selectors are always fitted inside the training fold.

### Statistical Analysis Module

For every feature set the module computes the following (all implemented in `prwb/stats/analysis.py`).

* **Descriptive statistics** and **normality** (Shapiro--Wilk $W$ and D'Agostino--Pearson $K^2$; a random subsample of 500 patterns keeps the tests meaningful for large $n$).
* **One-way ANOVA** per feature, $F=\mathrm{MS}_B/\mathrm{MS}_W$, with the effect sizes
$$\eta^2=\frac{SS_B}{SS_T},\qquad \omega^2=\max\Bigl(0,\frac{SS_B-(k-1)MS_W}{SS_T+MS_W}\Bigr),$$
the non-parametric **Kruskal--Wallis** $H$ and **Levene's** test of equal variances.
* **Multiple-testing control.** With $m$ tests sorted by $p_{(1)}\le\dots\le p_{(m)}$, the Benjamini--Hochberg adjusted value is $q_{(i)}=\min_{j\ge i}\frac{m}{j}p_{(j)}$; Bonferroni's $\min(1,mp)$ is also reported.
* **Correlation structure:** mean absolute correlation, number of pairs with $|\rho|>0.9$, the condition number $\lambda_{\max}/\lambda_{\min}$ of the covariance matrix, and its effective rank $\exp(-\sum_i\hat\lambda_i\ln\hat\lambda_i)$ with $\hat\lambda_i=\lambda_i/\sum\lambda$.
* **Class separability.** The pairwise Bhattacharyya distance between Gaussian class models $\mathcal N(\mathbf m_i,S_i)$, $\mathcal N(\mathbf m_j,S_j)$ with $S=(S_i+S_j)/2$,
$$ D_B=\tfrac18(\mathbf m_i-\mathbf m_j)^\top S^{-1}(\mathbf m_i-\mathbf m_j)+\tfrac12\ln\frac{\det S}{\sqrt{\det S_i\det S_j}}, $$
whose exponential $e^{-D_B}$ bounds the Bayes error of the pair from above, and the scalar criterion $J=\operatorname{tr}(S_W^{-1}S_B)$.
* **Outliers.** The squared Mahalanobis distance $d^2=(\mathbf x-\boldsymbol\mu)^\top S^{-1}(\mathbf x-\boldsymbol\mu)$ follows $\chi^2_d$ for Gaussian data; patterns with $d^2>\chi^2_{d,\,0.999}$ are flagged.
* **Pairwise Welch $t$-tests** between classes for a single feature, with Cohen's $d$ and Bonferroni adjustment <<cite('welch1947','cohen1988')>>.

### Classifiers

Let $\mathbf x\in\mathbb R^d$ be a standardised feature vector and $\omega_c$, $c=1,\dots,C$ the classes.

**Minimum-distance (template) classifier.** $\hat c=\arg\min_c\lVert\mathbf x-\boldsymbol\mu_c\rVert^2$; a Mahalanobis variant uses the pooled covariance. It is the simplest possible model and serves as the lower baseline.

**$k$-nearest neighbours.** The $k$ closest training patterns vote, with weights $1/(d_i+\varepsilon)$ in the distance-weighted version. Euclidean, Manhattan, cosine and Mahalanobis metrics are supported; the Mahalanobis metric is implemented by whitening, $\mathbf x\mapsto S^{-1/2}\mathbf x$. Squared Euclidean distances use the expansion $\lVert a-b\rVert^2=\lVert a\rVert^2+\lVert b\rVert^2-2a^\top b$, which turns the search into one matrix product and is processed in chunks to bound memory.

**Gaussian Bayes classifier (QDA / LDA / naive).** With class-conditional densities $\mathcal N(\boldsymbol\mu_c,\Sigma_c)$ the log-discriminant is
$$ g_c(\mathbf x)=-\tfrac12\Bigl[(\mathbf x-\boldsymbol\mu_c)^\top\Sigma_c^{-1}(\mathbf x-\boldsymbol\mu_c)+\ln|\Sigma_c|+d\ln2\pi\Bigr]+\ln P(\omega_c). $$
The covariances are *shrunk*, $\Sigma_c\leftarrow(1-r)\Sigma_c+r\,\frac{\operatorname{tr}\Sigma_c}{d}I$, to keep them well-conditioned. The quadratic form is evaluated through the Cholesky factor $\Sigma_c=LL^\top$ as $\lVert L^{-1}(\mathbf x-\boldsymbol\mu_c)\rVert^2$ and $\ln|\Sigma_c|=2\sum_i\ln L_{ii}$; posteriors are obtained by a log-sum-exp normalisation, which is numerically stable even when all densities underflow. *LDA* uses the pooled covariance (linear boundaries), *naive Bayes* a diagonal covariance.

**Parzen-window classifier.** The class-conditional density is estimated with Gaussian kernels,
$$ \hat p(\mathbf x\mid\omega_c)=\frac1{n_c}\sum_{i\in c}\frac{1}{(2\pi h^2)^{d/2}}\exp\Bigl(-\frac{\lVert\mathbf x-\mathbf x_i\rVert^2}{2h^2}\Bigr), $$
evaluated in the log domain; the bandwidth $h$ is a multiple (default 0.8) of the mean nearest-neighbour distance of the training set.

**Softmax regression.** $P(\omega_c\mid\mathbf x)=\operatorname{softmax}(W^\top\mathbf x+\mathbf b)_c$ is trained by minimising the regularised cross-entropy $L=-\frac1n\sum_i\ln P(y_i\mid\mathbf x_i)+\frac\lambda2\lVert W\rVert^2$ with mini-batch Adam <<cite('kingma2015')>>. For the output layer the gradient is simply $(P-Y)/n$.

**Multilayer perceptron.** The network $[\text{Linear}\to\text{ReLU}\to\text{Dropout}]^{\,L}\to\text{Linear}\to\text{Softmax}$ is trained by back-propagation <<cite('rumelhart1986')>>. The output error is $\delta^{(L+1)}=(P-Y)/n$ and it is propagated back with $\delta^{(l)}=(\delta^{(l+1)}W^{(l+1)\top})\odot\mathbf 1[a^{(l)}>0]\odot m^{(l)}$, where $m^{(l)}$ is the inverted-dropout mask; $\partial L/\partial W^{(l)}=a^{(l-1)\top}\delta^{(l)}+\lambda W^{(l)}$. Weights use He initialisation ($\mathcal N(0,2/\text{fan}_{in})$), the optimiser is Adam with learning rate $2\cdot10^{-3}$, and 10 % of the training fold is held out for early stopping with a patience of 12 epochs, restoring the best weights.

**Reference classifiers.** For calibration against strong established models the zoo also contains an RBF-SVM and a linear SVM (with Platt-scaled probabilities <<cite('platt1999')>>), logistic regression, a decision tree, random forest, extremely randomised trees and histogram gradient boosting, all from scikit-learn. <<ref('tab:zoo')>> lists the complete zoo.

<<tbl(pd.DataFrame([
 ['MinDist', 'scratch', 'Nearest class mean (Euclidean)'],
 ['kNN', 'scratch', 'k = 5, distance-weighted, Euclidean'],
 ['Bayes-Naive', 'scratch', 'Diagonal Gaussian, shrinkage 0.05'],
 ['Bayes-LDA', 'scratch', 'Pooled covariance, shrinkage 0.1'],
 ['Bayes-QDA', 'scratch', 'Class covariances, shrinkage 0.3'],
 ['Parzen', 'scratch', 'Gaussian kernel, h = 0.8 x mean 1-NN distance'],
 ['Softmax', 'scratch', 'L2 = 1e-3, Adam, 60 epochs'],
 ['MLP', 'scratch', 'Hidden (64, 32), dropout 0.1, early stopping'],
 ['SVM-RBF', 'reference', 'C = 10, gamma = scale, Platt scaling'],
 ['SVM-Linear', 'reference', 'C = 0.1, Platt scaling'],
 ['LogReg(sk)', 'reference', 'L2, C = 1'],
 ['DecisionTree', 'reference', 'CART, unlimited depth'],
 ['RandomForest', 'reference', '200 trees'],
 ['ExtraTrees', 'reference', '200 trees'],
 ['HistGB', 'reference', '40 iterations, learning rate 0.15 (only if dimension <= 100)'],
], columns=['Classifier', 'origin', 'configuration']), 'The classifier zoo. Every model is wrapped in a Standardizer pipeline.', 'tab:zoo', colfmt='lll', font='small')>>

### Clustering Algorithms

**$k$-means++.** Seeding picks the first centre uniformly and every further centre with probability proportional to the squared distance to the nearest centre already chosen <<cite('arthur2007')>>. Lloyd's iterations then alternate $z_i=\arg\min_j\lVert\mathbf x_i-\mathbf c_j\rVert^2$ and $\mathbf c_j=\operatorname{mean}\{\mathbf x_i:z_i=j\}$ until the relative decrease of the inertia $\sum_i\lVert\mathbf x_i-\mathbf c_{z_i}\rVert^2$ falls below $10^{-5}$. An empty cluster is re-seeded at the point farthest from its centroid, and the best of $n_{\text{init}}=5$ restarts is kept.

**Gaussian mixture by EM.** The model is $p(\mathbf x)=\sum_k\pi_k\mathcal N(\mathbf x;\boldsymbol\mu_k,\Sigma_k)$. The E-step computes the responsibilities in the log domain, $r_{ik}\propto\pi_k\mathcal N(\mathbf x_i;\boldsymbol\mu_k,\Sigma_k)$; the M-step updates
$$ N_k=\sum_ir_{ik},\quad \pi_k=\frac{N_k}n,\quad \boldsymbol\mu_k=\frac1{N_k}\sum_ir_{ik}\mathbf x_i,\quad \Sigma_k=\frac1{N_k}\sum_ir_{ik}(\mathbf x_i-\boldsymbol\mu_k)(\mathbf x_i-\boldsymbol\mu_k)^\top+\varepsilon I. $$
Initialisation is from a $k$-means solution; iterations stop when the mean log-likelihood changes by less than $10^{-4}$ (it is guaranteed to be non-decreasing <<cite('dempster1977')>>, which is checked in a test). Covariances may be `full`, `diag` or `spherical`, and the model is scored with $\mathrm{BIC}=-2\ln\hat L+p\ln n$ <<cite('schwarz1978')>>, where $p$ is the number of free parameters.

**Fuzzy $c$-means.** Memberships $u_{ij}=\bigl[\sum_l(d_{ij}/d_{il})^{2/(m-1)}\bigr]^{-1}$ and centres $\mathbf c_j=\sum_iu_{ij}^m\mathbf x_i/\sum_iu_{ij}^m$ are iterated for the fuzzifier $m>1$ <<cite('bezdek1981')>>. In high-dimensional spaces $m=2$ makes all memberships collapse towards $1/c$; the workbench therefore defaults to $m=1.2$ and reports a sweep of the fuzzifier.

**Reference algorithms.** Ward-linkage and average-linkage agglomerative clustering, spectral clustering with a 12-nearest-neighbour affinity, DBSCAN (with the radius $\varepsilon$ set from the 5-NN distance distribution) and HDBSCAN are taken from scikit-learn, together with the library $k$-means that is used to cross-check the scratch implementation.

**Choosing the number of clusters.** Four criteria are computed for $k=2,\dots,15$: the elbow of the inertia curve, the mean silhouette $s(i)=\frac{b(i)-a(i)}{\max\{a(i),b(i)\}}$ (with $a$ the mean intra-cluster and $b$ the mean nearest-other-cluster distance), the BIC of a diagonal GMM, and the gap statistic $\mathrm{Gap}(k)=\mathbb E^\ast[\ln W_k]-\ln W_k$ computed against uniform reference data in the bounding box <<cite('tibshirani2001')>>.

**External validity.** With ground-truth labels available for benchmarking, the adjusted Rand index (ARI) <<cite('hubert1985')>>, normalised mutual information (NMI), homogeneity, completeness, V-measure, purity and the *cluster accuracy* after optimal Hungarian matching <<cite('kuhn1955')>> are reported; the internal silhouette, Davies--Bouldin and Calinski--Harabasz indices are reported for the case without labels. *Stability* is measured as the mean pairwise ARI between clusterings fitted on bootstrap resamples.

### Evaluation Module

<<fig('diagram_cv_protocol.png', 'Leakage-free repeated stratified K-fold protocol used for all classification results.', 0.95)>>

**Protocol.** Repeated stratified $5$-fold cross-validation with $2$ repeats gives ten held-out scores per (feature set, classifier) cell (<<ref('fig:diagram_cv_protocol')>>). The estimator, always a `Pipeline` that contains the scaler, is cloned and re-fitted in every fold. The out-of-fold predictions of the first repeat cover every sample once and are used for confusion matrices, ROC curves and calibration diagrams.

**Metrics.** Accuracy, balanced accuracy, macro-averaged precision, recall and F1, the multiclass Matthews correlation coefficient, Cohen's $\kappa$, the one-vs-rest macro ROC-AUC, the log-loss and the *expected calibration error*
$$ \mathrm{ECE}=\sum_{b=1}^{10}\frac{|B_b|}{n}\bigl|\operatorname{acc}(B_b)-\operatorname{conf}(B_b)\bigr|, $$
where $B_b$ are ten equal-width bins of the predicted confidence <<cite('guo2017')>>. Ninety-five percent *bootstrap* confidence intervals of the mean fold accuracy use 2,000 resamples <<cite('efron1993')>>.

**Significance tests.** For two classifiers on the same out-of-fold predictions, McNemar's test uses the counts $n_{01}$ (A right, B wrong) and $n_{10}$ (A wrong, B right): $\chi^2=(|n_{01}-n_{10}|-1)^2/(n_{01}+n_{10})$ <<cite('dietterich1998')>>. For the fold scores the *corrected resampled $t$-test* is used,
$$ t=\frac{\bar d}{\sqrt{\bigl(\tfrac1k+\tfrac{n_{\text{test}}}{n_{\text{train}}}\bigr)s_d^2}}, $$
whose variance is inflated by the factor $n_{\text{test}}/n_{\text{train}}$ to account for the overlap of training sets <<cite('nadeau2003')>>. Across many blocks (data set $\times$ feature set) the Friedman statistic
$$ \chi^2_F=\frac{12N}{k(k+1)}\Bigl[\sum_jR_j^2-\frac{k(k+1)^2}{4}\Bigr] $$
tests whether the average ranks $R_j$ of the $k$ classifiers differ, and the Nemenyi critical difference $\mathrm{CD}=q_\alpha\sqrt{k(k+1)/(6N)}$ identifies which pairs differ <<cite('demsar2006')>>; $q_\alpha$ is obtained from the studentised-range distribution.

### Computational Complexity

<<ref('tab:complexity')>> summarises the cost of the main algorithms ($n$ patterns, $d$ dimensions, $C$ classes, $k$ clusters, $T$ iterations or epochs, $H$ hidden units). It explains the run-times observed in <<ref('sec:results')>>: $k$-NN and Parzen have no training cost but $O(nd)$ cost *per query*, the Bayes classifiers are dominated by the $O(Cd^3)$ Cholesky factorisations and become expensive for descriptors with hundreds of dimensions, and the agglomerative methods are quadratic in the number of patterns.

<<tbl(pd.DataFrame([
 ['PCA (SVD)', 'O(n d min(n, d))', 'O(n d)', 'one-off'],
 ['Fisher LDA', 'O(n d^2 + d^3)', 'O(d^2)', 'generalised eigenproblem'],
 ['k-NN', 'O(1)', 'O(n d) per query', 'chunked distance matrix'],
 ['Gaussian Bayes (QDA)', 'O(n d^2 + C d^3)', 'O(C d^2) per query', 'Cholesky per class'],
 ['Parzen window', 'O(1)', 'O(n d) per query', 'log-sum-exp'],
 ['Softmax / MLP', 'O(T n d H)', 'O(d H)', 'mini-batch Adam'],
 ['k-means++', 'O(T n k d)', 'O(k d)', 'restarts multiply the cost'],
 ['EM for GMM (full)', 'O(T n k d^2 + T k d^3)', 'O(k d^2)', 'costly for d > 50'],
 ['Ward / average linkage', 'O(n^2 log n) - O(n^3)', 'O(n^2)', 'dense distance matrix'],
 ['DBSCAN', 'O(n^2) (O(n log n) with index)', 'O(n)', 'region queries'],
], columns=['Algorithm', 'Training', 'Prediction', 'Remark']), 'Asymptotic cost of the main algorithms.', 'tab:complexity', colfmt='llll', font='small')>>

## Operational Diagram

<<ref('fig:diagram_operational')>> traces one complete run of the experimental protocol.

<<fig('diagram_operational.png', 'Operational diagram: execution of one benchmark run (supervised and unsupervised branches, stress tests and outputs).', 0.92)>>

1. **Start.** The YAML configuration is read (data sets, feature sets, classifiers, CV parameters, clustering parameters) and the random seed is fixed.
2. **Data loading.** Each data set is loaded or generated; class balance is checked and, in the quick mode, the data set is sub-sampled.
3. **Feature construction.** For every (data set, feature set) pair the descriptors are computed once, producing a named matrix with a *group index* that records which descriptor family each column came from (needed for the ablation study).
4. **Statistical analysis.** Normality, ANOVA/Kruskal with FDR correction, correlation structure, Bhattacharyya separability and Mahalanobis outliers are computed for each feature space.
5. **Branch on the use of labels.** *Supervised branch:* for each classifier the repeated stratified CV is run (in parallel over classifiers), the scaler and the model are fitted in each training fold and the held-out fold is scored. *Unsupervised branch:* the standardised features are reduced to 30 principal components, ten clustering algorithms are fitted *without* labels, and the labels are used only for external validation.
6. **Evaluation.** Fold scores are aggregated with bootstrap confidence intervals; out-of-fold predictions feed the confusion matrices, ROC and calibration plots; McNemar, corrected-$t$ and Friedman--Nemenyi tests are applied.
7. **Stress tests (side loop).** Hyper-parameter sweeps, learning curves, descriptor-group ablation and robustness studies (noise, occlusion, rotation, shift) re-use the cached feature sets.
8. **Outputs.** CSV tables, a JSON summary, PNG figures and the serialised model are written; the report generator reads them so that every number in this document comes from the files.
