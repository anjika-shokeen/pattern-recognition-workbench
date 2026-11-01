## Conclusion and Future Scope {#sec:conclusion}

<% set bd = S['best_digits'] %>
<% set cb = S['clustering_best_main'] %>
<% set ro = S['robustness_mean_by_featureset'] %>
This project set out to show that the stages of a pattern-recognition system (feature extraction, statistical analysis, classification, clustering and performance evaluation) can be built, validated and studied as one coherent workbench rather than as unrelated exercises. The result is PRWB: a package of <<'{:,}'.format(S['loc']['package'])>> lines of Python in which <<S['n_scratch']>> algorithms are implemented from scratch, <<S['n_tests']>> automated tests check them against reference implementations, and three interfaces (command line, dashboard and notebook) expose the same pipeline.

The experimental study led to four main conclusions.

1. **A good representation matters more than a complex classifier.** With HOG and LBP features, the RBF-SVM reaches <<'%.2f'|format(bd['accuracy']*100)>>\% accuracy on the digits (95\% interval <<'%.2f'|format(bd['ci'][0]*100)>>--<<'%.2f'|format(bd['ci'][1]*100)>>\%, macro-F1 <<'%.3f'|format(bd['macro_f1'])>>), and the simple linear models are close behind; the ranking of the classifiers is consistent over the data sets and is confirmed by the Friedman--Nemenyi analysis.
2. **Robustness separates the descriptors where clean accuracy does not.** Averaged over five corruption types, HOG+LBP features keep <<'%.2f'|format(ro['hog+lbp']*100)>>\% accuracy, against <<'%.2f'|format(ro['raw']*100)>>\% for raw pixels.
3. **Statistical analysis explains the classification results.** Non-normality, heteroscedasticity and near-singular covariance matrices, all measured by the statistics module, explain why quadratic Bayes and Parzen classifiers need regularisation or dimensionality reduction, and why discriminative models are more successful on the high-dimensional descriptors.
4. **Unsupervised structure is recoverable, but its evaluation needs care.** <<cb['algorithm']>> recovers the digit classes with an ARI of <<'%.3f'|format(cb['ARI'])>> and a cluster accuracy of <<'%.1f'|format(cb['cluster_acc']*100)>>\%, whereas internal indices such as the silhouette and the criteria for the number of clusters can favour partitions that disagree with the labels.

### Limitations

The limitations listed in the analysis chapter apply: the digit images are small and clean, the signals are synthetic, a single global seed was used, and the models were trained on clean data only. The conclusions about robustness and rankings are therefore valid for the protocol of this report and should be re-examined on larger, real-world data before they are generalised.

### Future scope

* **Larger and real data.** Apply the workbench to real recordings (ECG, vibration or audio signals) and to larger image collections, adding a plug-in interface for new data loaders.
* **Learned features.** Add convolutional or auto-encoder features as another descriptor group and compare them with the hand-crafted descriptors in the same ablation and robustness protocol.
* **Robust training.** Evaluate data augmentation and noise-aware training, because the present robustness study uses models trained on clean data only.
* **Further statistics.** Add multi-seed repetition with hierarchical significance testing, and Bayesian comparison of classifiers, to replace the independence assumption of the Friedman test.
* **Model compression and deployment.** Export the trained pipeline to ONNX, provide a REST service for the recogniser and measure latency on embedded hardware.
* **Semi-supervised and active learning.** Combine the clustering and classification modules to label a small number of representative patterns per cluster and propagate the labels.
