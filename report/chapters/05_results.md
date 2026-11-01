## Results (Screenshots) {#sec:results}

This section presents the outputs of the workbench exactly as a user sees them: first the screenshots of the command-line interface and of the interactive dashboard, then the figures and tables produced by the experimental protocol of <<ref('tab:experiments')>>. Every number below is read from the files in `results/` at the moment the report is built; the interpretation follows in <<ref('sec:analysis')>>.

<% set bpb = csv('best_per_block.csv') %>
<% set cls = csv('classification_results.csv') %>
<% set dgb = bpb[bpb.dataset == 'digits'].sort_values('accuracy', ascending=False) %>

### Command-line interface

The package installs a single entry point, `python -m prwb`, with one sub-command for each stage of the chain. <<ref('fig:term_datasets')>> lists the available data sets, feature sets, classifiers and clusterers. <<ref('fig:term_features')>> extracts the HOG+LBP descriptor of the digit images and prints the first rows of the feature matrix.

<<fig('term_datasets.png', 'Screenshot: \\texttt{python -m prwb datasets} lists the data sets with their feature sets, the classifiers and the clusterers.', 0.9)>>

<<fig('term_features.png', 'Screenshot: \\texttt{python -m prwb features -d digits -f hog+lbp} computes a named feature matrix.', 0.9)>>

The statistical-analysis command (<<ref('fig:term_stats')>>) ranks the features of the wine data by one-way ANOVA with Benjamini--Hochberg adjusted $q$-values and prints the normality, correlation and separability summaries. The `classify` command (<<ref('fig:term_classify')>>) runs a cross-validated classification and prints a per-class report, and `cluster` (<<ref('fig:term_cluster')>>) runs a clustering algorithm and prints the internal and external validity indices.

<<fig('term_stats.png', 'Screenshot: \\texttt{python -m prwb stats -d wine -f raw --top 10} (ANOVA ranking, normality, correlation and separability).', 0.9)>>

<<fig('term_classify.png', 'Screenshot: \\texttt{python -m prwb classify -d digits -f hog+lbp -c SVM-RBF --folds 5} (cross-validated metrics and per-class report).', 0.9)>>

<<fig('term_cluster.png', 'Screenshot: \\texttt{python -m prwb cluster -d digits -f hog+lbp -a Agglomerative-Ward} (clustering with validity indices).', 0.9)>>

<<fig('term_pytest.png', 'Screenshot: \\texttt{python -m pytest -q}: the automated test suite passes.', 0.9)>>

### Interactive dashboard

The Streamlit dashboard (`streamlit run app/streamlit_app.py`) wraps the same functions in five tabs. <<ref('fig:app_data')>> shows the *Data and features* tab, <<ref('fig:app_stats')>> the *Statistics* tab, <<ref('fig:app_classification')>> the result of running a cross-validation from the *Classification* tab, <<ref('fig:app_clustering')>> the *Clustering* tab and <<ref('fig:app_recognise')>> the *Recognise a digit* tab, in which an uploaded picture of a digit is classified by the deployed model.

<<fig('app_data.png', 'Dashboard: Data and features tab (sample patterns, HOG visualisation and feature matrix).', 0.95)>>

<<fig('app_stats.png', 'Dashboard: Statistics tab (ANOVA ranking, correlation map and class separability).', 0.95)>>

<<fig('app_classification.png', 'Dashboard: Classification tab after pressing "Run cross-validation".', 0.95)>>

<<fig('app_clustering.png', 'Dashboard: Clustering tab (embedding and validity indices).', 0.95)>>

<<fig('app_recognise.png', 'Dashboard: Recognise a digit tab. An uploaded picture is normalised, described by HOG+LBP features and classified; the class probabilities are shown.', 0.95)>>

### Data sets and extracted features

<<ref('tab:dsover')>> summarises the data sets. Examples of the digit images and of the synthetic waveform signals were shown in <<ref('fig:examples')>>.

<<tbl(csv('dataset_overview.csv').rename(columns={'pattern_shape': 'pattern', 'min_class': 'min', 'max_class': 'max', 'raw_dims': 'raw d'}), 'The data sets used in the study.', 'tab:dsover', colfmt='llrlrrrr', font='small')>>

<<ref('fig:hog_visual')>> and <<ref('fig:lbp_visual')>> visualise two of the image descriptors computed by the from-scratch code, and <<ref('fig:glcm_distributions')>> and <<ref('fig:signal_feature_distributions')>> show how selected texture and signal features distribute over the classes.

<<fig('hog_visual.png', 'HOG of a digit: per-cell orientation histograms drawn as oriented glyphs.', 0.85)>>

<<fig('lbp_visual.png', 'LBP of a digit: raw codes and the uniform-pattern map.', 0.85)>>

<<fig('glcm_distributions.png', 'Distribution of selected GLCM (Haralick) features for the ten digit classes.', 0.85)>>

<<fig('signal_feature_distributions.png', 'Distribution of selected signal features for the waveform classes.', 0.9)>>

The raw $<<S['stats']['digits_handcrafted']['n_features']>>$-dimensional descriptor is highly redundant: the principal-component analysis of the raw pixels needs <<S['pca_raw_components_95']>> components for 95\% and <<S['pca_raw_components_99']>> components for 99\% of the variance (<<ref('fig:pca_row')>>), and <<ref('fig:pca_row')>> shows reconstructions with an increasing number of components. <<ref('fig:embeddings_digits')>> compares PCA, Fisher LDA and t-SNE embeddings of the digit descriptors.

<<figrow([('pca_scree.png', 'Scree and cumulative variance'), ('pca_reconstruction.png', 'Reconstruction')], 'Principal-component analysis of the digit images.', 'fig:pca_row')>>

<<fig('embeddings_digits.png', 'Two-dimensional embeddings of the digit descriptors (PCA, Fisher LDA and t-SNE).', 0.95)>>

### Statistical analysis

<<ref('tab:statsum')>> condenses the statistical analysis of the four main feature spaces. The columns give the share of features that pass the Shapiro--Wilk normality test, the share of features that differ significantly between classes after Benjamini--Hochberg correction, the median and maximum effect size $\eta^2$ of the ANOVA, the strongest feature, the average absolute correlation and the number of strongly correlated pairs, the number of Mahalanobis outliers and the pair of classes with the smallest Bhattacharyya distance.

<<tbl(stats_df(S), 'Summary of the statistical analysis of the main feature spaces.', 'tab:statsum', fmts={'normal %': '{:.1f}', 'signif. %': '{:.1f}', 'median eta2': '{:.3f}', 'max eta2': '{:.3f}', 'mean abs r': '{:.3f}'}, colfmt='lrrrrrlrrrl', resize=True, font='scriptsize')>>

<<figrow([('anova_digits_handcrafted.png', 'Digits: top ANOVA features'), ('anova_signals_handcrafted.png', 'Signals: top ANOVA features')], 'Features with the largest ANOVA $F$-statistic (bars) and their adjusted significance.', 'fig:anova_row')>>

<<figrow([('correlation_digits_handcrafted.png', 'Digits'), ('correlation_signals_handcrafted.png', 'Signals')], 'Absolute pairwise feature correlation of the hand-crafted descriptors, ordered by hierarchical clustering.', 'fig:corr_row')>>

<<figrow([('bhattacharyya_digits_handcrafted.png', 'Digits'), ('bhattacharyya_signals_handcrafted.png', 'Signals')], 'Pairwise Bhattacharyya distance between the classes (small values mark difficult pairs).', 'fig:bhatt_row')>>

<<figrow([('mahalanobis_digits_handcrafted.png', 'Digits'), ('mahalanobis_signals_handcrafted.png', 'Signals'), ('mahalanobis_wine_raw.png', 'Wine')], 'Squared Mahalanobis distances with the chi-square outlier threshold.', 'fig:mahal_row', width=0.31)>>

The descriptor families of the hand-crafted digit descriptor differ in how much class information they carry (<<ref('tab:famef')>>), and <<ref('tab:sep')>> reports two scalar separability criteria for every feature space.

<<tbl(csv('descriptor_family_effect_sizes.csv').sort_values('mean_eta2', ascending=False), 'ANOVA effect size per descriptor family of the digit descriptor.', 'tab:famef', fmts={'mean_eta2': '{:.3f}', 'max_eta2': '{:.3f}', 'median_F': '{:.1f}'}, colfmt='lrrrr', font='small')>>

<<tbl(csv('separability_by_featureset.csv').rename(columns={'J_trace_SwInv_Sb': 'J = tr(Sw^-1 Sb)', 'trace_Sb_over_trace_Sw': 'tr Sb / tr Sw'}), 'Scatter-matrix separability criteria of every feature space.', 'tab:sep', fmts={'J = tr(Sw^-1 Sb)': '{:.2f}', 'tr Sb / tr Sw': '{:.3f}'}, colfmt='llrrr', font='small')>>

### Classification

The classification grid evaluates <<S['n_cls_cells']>> (data set, feature set, classifier) cells with 5-fold stratified cross-validation repeated twice. <<ref('tab:bestblock')>> lists the best classifier of every block, and <<ref('fig:heatmap_accuracy_digits')>> and <<ref('fig:heatmap_accuracy_signals')>> show the complete accuracy matrices as heat maps (feature sets as columns, classifiers as rows). The complete tables are in <<ref('app:cls')>>.

<<tbl(bpb.sort_values(['dataset', 'accuracy'], ascending=[True, False]).rename(columns={'feature_set': 'features', 'macro_f1': 'macro-F1'}), 'Best classifier in each (data set, feature set) block.', 'tab:bestblock', fmts={'accuracy': '{:.4f}', 'macro-F1': '{:.4f}', 'mcc': '{:.4f}'}, colfmt='llrlrrr', font='small')>>

<<fig('heatmap_accuracy_digits.png', 'Cross-validated accuracy (%) of every classifier on every digit feature set.', 0.95)>>

<<fig('heatmap_accuracy_signals.png', 'Cross-validated accuracy (%) of every classifier on every signal feature set.', 0.95)>>

<<ref('fig:cvbox_row')>> shows the distribution of the ten fold accuracies of each classifier. <<ref('fig:conf_row')>> and <<ref('fig:conf_sig')>> show the pooled out-of-fold confusion matrices of the best and of a weak classifier, <<ref('fig:roc_cal_row')>> the one-vs-rest ROC curves and <<ref('fig:roc_cal_row')>> the reliability diagram of the best digit model.

<<figrow([('cv_boxplot_digits.png', 'Digits'), ('cv_boxplot_signals.png', 'Signals')], 'Distribution of the ten fold accuracies for every classifier.', 'fig:cvbox_row')>>

<<figrow([('confusion_digits_best.png', 'Best model'), ('confusion_digits_weak.png', 'Weak model')], 'Pooled out-of-fold confusion matrices on the digit data.', 'fig:conf_row')>>

<<fig('confusion_signals_best.png', 'Confusion matrix of the best signal classifier.', 0.6, 'fig:conf_sig')>>

<<figrow([('roc_digits_best.png', 'One-vs-rest ROC curves'), ('calibration_digits.png', 'Reliability diagram')], 'ROC and calibration of the best digit classifier.', 'fig:roc_cal_row')>>

<<ref('tab:perclass')>> lists the per-class precision, recall and F1-score of the best digit model, and <<ref('fig:time_vs_accuracy')>> relates accuracy to the training time.

<<tbl(csv('per_class_report_digits_best.csv'), 'Per-class report of the best digit classifier (pooled out-of-fold predictions).', 'tab:perclass', fmts={'precision': '{:.4f}', 'recall': '{:.4f}', 'f1-score': '{:.4f}', 'support': '{:.0f}'}, colfmt='lrrrr', font='small')>>

<<fig('time_vs_accuracy.png', 'Accuracy versus training time per fold (digits).', 0.85)>>

**Statistical comparison.** <<ref('tab:friedman')>> gives the average rank of each classifier over the <<S['friedman']['n_blocks']>> blocks, <<ref('fig:critical_difference')>> the corresponding critical-difference diagram, and <<ref('tab:mcnemar')>> compares the three best digit models pairwise with McNemar's test and the corrected resampled $t$-test.

<<tbl(csv('friedman_avg_ranks.csv').sort_values('avg_rank'), 'Average rank of each classifier (Friedman). Lower is better.', 'tab:friedman', fmts={'avg_rank': '{:.2f}'}, colfmt='lr', font='small')>>

<<fig('critical_difference.png', 'Critical-difference diagram of the Nemenyi post-hoc test (classifiers joined by a bar are not significantly different).', 0.95)>>

<<tbl(csv('significance_tests_digits.csv').rename(columns={'McNemar_n01(A right,B wrong)': 'A right, B wrong', 'McNemar_n10(A wrong,B right)': 'A wrong, B right', 'McNemar_p': 'McNemar p', 'corrected_t': 't', 'corrected_t_p': 't-test p'}), 'Pairwise significance tests between the best digit classifiers.', 'tab:mcnemar', fmts={'acc_A': '{:.4f}', 'acc_B': '{:.4f}', 'McNemar p': '{:.3f}', 't': '{:.2f}', 't-test p': '{:.3f}'}, colfmt='llrrrrrrr', resize=True, font='scriptsize')>>

### Hyper-parameter sweeps, learning curves and feature selection

<<ref('fig:sweep_row1')>> shows the influence of the main hyper-parameters of $k$-NN and of the RBF-SVM, and <<ref('fig:sweep_row2')>> that of the Parzen bandwidth and of the covariance shrinkage of the Gaussian Bayes classifiers.

<<figrow([('sweep_knn.png', 'k-NN: k and metric'), ('sweep_svm.png', 'RBF-SVM: C and gamma')], 'Hyper-parameter sweeps for the k-NN and the SVM classifiers (digits, 5-fold CV).', 'fig:sweep_row1')>>

<<figrow([('sweep_parzen.png', 'Parzen bandwidth'), ('sweep_bayes_reg.png', 'Bayes shrinkage')], 'Hyper-parameter sweeps for the Parzen and the Bayes classifiers (digits, 5-fold CV).', 'fig:sweep_row2')>>

The training curves of the NumPy multilayer perceptron are shown in <<ref('fig:mlp_loss')>>, the effect of the architecture in <<ref('tab:mlparch')>>, and the learning curves of five classifiers in <<ref('fig:learning_curves')>>.

<<fig('mlp_loss.png', 'Training and validation loss of the NumPy MLP (training stopped after ' ~ S['mlp_epochs_run'] ~ ' epochs).', 0.7)>>

<<tbl(csv('sweep_mlp_architecture.csv'), 'Test accuracy of MLPs with different architectures (HOG+LBP features).', 'tab:mlparch', fmts={'test_acc': '{:.4f}'}, colfmt='lrrr', font='small')>>

<<fig('learning_curves.png', 'Learning curves: accuracy versus the number of training samples (HOG+LBP).', 0.8)>>

<<figrow([('sweep_pca_dims.png', 'Accuracy vs. PCA dimension'), ('sweep_feature_selection.png', 'Filter feature selection')], 'Effect of dimensionality reduction and of feature selection on accuracy.', 'fig:dimsel_row')>>

<<tbl(csv('mrmr_selected_groups.csv'), 'Descriptor families chosen by the first 40 mRMR selections.', 'tab:mrmr', colfmt='lr', font='small')>>

### Descriptor ablation and robustness

<<ref('fig:ablation_groups')>> gives the accuracy when each descriptor family is removed from, or used alone in, the full descriptor (the numbers are listed in <<ref('app:extra')>>). <<ref('fig:rob_row1')>> and <<ref('fig:rob_row2')>> show how the accuracy decays when the test images are corrupted with increasing severity.

<<fig('ablation_groups.png', 'Descriptor-group ablation: accuracy when a family is removed (without) or used alone (only).', 0.9)>>

<<figrow([('robustness_gaussian_noise.png', 'Gaussian noise'), ('robustness_salt_pepper.png', 'Salt-and-pepper noise')], 'Accuracy under additive noise (mean over four classifiers).', 'fig:rob_row1')>>

<<figrow([('robustness_occlusion.png', 'Occlusion'), ('robustness_rotation.png', 'Rotation'), ('robustness_shift.png', 'Shift')], 'Accuracy under geometric corruptions (mean over four classifiers).', 'fig:rob_row2', width=0.31)>>

### Clustering

Ten clustering algorithms were run on five representations of the digit data. <<ref('tab:clumain')>> compares them on the representation that is also used for the classification (HOG+LBP reduced by PCA to 30 dimensions), and <<ref('fig:clustering_metrics_main')>> and <<ref('fig:clustering_ari_heatmap')>> visualise the metrics for all representations.

<% set cr = csv('clustering_results.csv') %>
<<tbl(cr[cr.space.str.startswith('hog+lbp -> PCA30')][['algorithm', 'ARI', 'NMI', 'cluster_acc', 'purity', 'silhouette', 'n_clusters', 'noise_fraction', 'time_s']].rename(columns={'cluster_acc': 'acc', 'n_clusters': 'k', 'noise_fraction': 'noise', 'time_s': 'time (s)'}).sort_values('ARI', ascending=False), 'Clustering of the digits (HOG+LBP, PCA to 30 dimensions).', 'tab:clumain', fmts={'ARI': '{:.3f}', 'NMI': '{:.3f}', 'acc': '{:.3f}', 'purity': '{:.3f}', 'silhouette': '{:.3f}', 'noise': '{:.2f}', 'time (s)': '{:.2f}'}, colfmt='lrrrrrrrr', resize=True, font='scriptsize')>>

<<fig('clustering_metrics_main.png', 'Internal and external validity indices of the clustering algorithms (HOG+LBP, PCA 30).', 0.95)>>

<<fig('clustering_ari_heatmap.png', 'Adjusted Rand index of each algorithm in each feature space.', 0.9)>>

<<ref('fig:k_selection')>> shows the four criteria used to choose the number of clusters, <<ref('fig:cluster_tsne')>> the contingency table of the best clustering and <<ref('fig:cluster_tsne')>> a t-SNE view with true and discovered labels.

<<fig('k_selection.png', 'Choosing the number of clusters: elbow, silhouette, BIC and the gap statistic.', 0.95)>>

<<figrow([('cluster_contingency.png', 'Contingency table'), ('cluster_scatter_tsne.png', 'True (left) and discovered (right) labels in a t-SNE map')], 'Agreement between the best clustering and the true digit classes.', 'fig:cluster_tsne')>>

<<figrow([('kmeans_init.png', 'k-means initialisation'), ('gmm_convergence.png', 'EM convergence')], 'Initialisation of k-means and convergence of the EM algorithm.', 'fig:init_row')>>

<<tbl(csv('gmm_covariance_types.csv'), 'Gaussian mixtures with different covariance structures.', 'tab:gmmcov', fmts={'BIC': '{:.0f}', 'AIC': '{:.0f}', 'ARI': '{:.3f}'}, colfmt='lrrrr', font='small')>>

<<figrow([('fcm_fuzzifier.png', 'Fuzzy c-means: fuzzifier'), ('dbscan_sweep.png', 'DBSCAN: eps sweep')], 'Sensitivity of fuzzy c-means and DBSCAN to their main parameter.', 'fig:fcm_row')>>

<<figrow([('dendrogram.png', 'Ward dendrogram'), ('cluster_silhouette.png', 'Silhouette plot')], 'Hierarchical clustering and silhouette analysis.', 'fig:dend_row')>>

<<tbl(csv('cluster_stability.csv'), 'Stability of the clusterings under bootstrap resampling (mean pairwise ARI).', 'tab:stab', fmts={'mean_pairwise_ARI': '{:.3f}', 'std': '{:.3f}'}, colfmt='lrr', font='small')>>

Finally, the clustering of the other data sets is summarised in <<ref('app:extra')>> and the signal clustering is visualised in <<ref('fig:cluster_scatter_signals')>>.

<<fig('cluster_scatter_signals.png', 'Gaussian-mixture clustering of the waveform signals (t-SNE map; true classes left, clusters right).', 0.9)>>
