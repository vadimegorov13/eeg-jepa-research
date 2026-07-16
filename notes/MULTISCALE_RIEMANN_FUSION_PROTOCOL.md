# Multiscale Riemannian Fusion Protocol

## Locked Hypotheses

The primary population is all 50 Liu2024 subjects. The Lv-14 mapping is a secondary restricted-population analysis and must be reported separately. The primary endpoint is each subject's pooled outer out-of-fold balanced accuracy, averaged across subjects.

The prespecified hypotheses are that parameter-free multiscale score fusion improves over the stability-selected Riemannian branch, and, when frozen S-JEPA is enabled, that full parameter-free fusion improves over frozen S-JEPA alone. These comparisons use paired subject-level Wilcoxon tests and bootstrap confidence intervals over subjects. Fold-level significance is prohibited.

The experiment was motivated partly by the prototype artifact `artifacts/liu2024-multiwindow-riemann-sjepa-stacking/20260707_0912_700a2c85/global_metrics.json`, which reported 52.13% Riemannian, 54.25% S-JEPA, and 50.25% fusion. Those values are motivation only and are not results from this locked notebook.

## Methods

Every trial is processed independently. The 0-4 s motor-imagery segment is located from marker value 2, 29 EEG channels are retained after dropping recorded CPz, and no filter crosses a trial boundary. The locked bands are 8-12, 13-20, 20-30, and 8-30 Hz. Temporal scales are four 1 s windows starting at 0, 1, 2, and 3 s; three 2 s windows starting at 0, 1, and 2 s; and the complete 4 s window.

OAS covariance is the default. Average reference and trace normalization are explicit configuration controls. Each covariance view is mapped to its own training-fold tangent space and yields one signed class-oriented decision score. Each fitted branch divides query margins by the median absolute margin on that model's training rows, with an RMS fallback. This preserves the zero decision boundary and makes score magnitudes comparable without outer-test labels. The headline models never concatenate tangent coefficients across all views.

Comparators on identical outer splits are: fixed 8-30 Hz/0-4 s tangent space; nested best view using mean inner balanced accuracy minus lambda times its fold SD; each temporal scale; an equal-weight average of the three scale-level Riemannian scores; stability-aware top-k score fusion; signed homologous asymmetry from C3/C4, FC3/FC4, CP3/CP4, and P3/P4 when available; Riemann-plus-asymmetry fusion; optional frozen local S-JEPA; Riemann-plus-S-JEPA fusion; full parameter-free fusion; and a secondary fixed-hyperparameter elastic-net score stack. The stack's `C` and `l1_ratio` are locked and are not tuned on reused cross-fitted scores.

`riemann_equal_short_scales` is the equal average of the fold-safe 1 s and 2 s scale scores and excludes the 4 s scale. Its **55.95% five-fold result is posthoc exploratory**. The frozen sensitivity confirmation is the motor13, full-trial-then-crop, OAS, tangent-LDA repeated 60/40 protocol with 10 repeats in `src/utils/experiments/configs/sweep_multiscale_riemann_fusion_motor13_short_confirm.json`; that confirmation must be reported separately from the exploratory five-fold value.

## Results So Far

The posthoc exploratory five-fold motor13 result was 55.95% for `riemann_equal_short_scales`; it is internal reuse of the same 50 subjects and is not independent confirmation. The frozen repeated-60/40 confirmation is `artifacts/liu2024-multiscale-riemann-fusion/20260712_145032_844360_89d4f1fc/` (canonical metrics: `global_metrics.json` and `subject_metrics.json`). It used all 50 subjects, 10 stratified 60/40 repeats per subject, motor13, full-trial-then-crop filtering, OAS covariance, and tangent-LDA, with `outer_test_used_for_selection: false`.

In that confirmation, mean subject balanced accuracy was **55.375%** (95% subject-bootstrap CI 52.313-58.588%) for equal 1 s/2 s score fusion, **52.688%** for the fixed 8-30 Hz/0-4 s view, **55.213%** for 1 s, and **55.050%** for 2 s. Paired by subject, short-scale fusion minus fixed view was **+2.688 points** (95% bootstrap CI +0.763 to +4.750; Wilcoxon W=391, two-sided p=0.0275; 28 wins/1 tie/21 losses). Fusion minus 1 s was +0.163 points (CI -0.913 to +1.238; W=531, p=0.7267; 25/3/22), and fusion minus 2 s was +0.325 points (CI -0.388 to +1.050; W=409, p=0.4391; 23/7/20). Thus confirmation supports the improvement over the fixed full-window view, but not superiority over either short scale alone.

There were zero fold failures, zero invalid subject-methods, and zero collapsed subject-repeat predictions among 2,000 method-repeats at the configured 95% single-class threshold (`fold_failures.json`, `predictions.csv`). A descriptive paralysis-side stratification, used only as a proxy for lesion laterality, gave fusion-minus-fixed deltas of +2.835 points for left paralysis (n=28; bootstrap CI +0.067 to +5.849) and +2.500 for right paralysis (n=22; CI approximately 0.000 to +5.284). This subgroup analysis was not prespecified and is not evidence of a lesion-side interaction.

The asymmetry branch first averages each homologous pair's log-power difference across the views belonging to one temporal scale. A fold-local four-feature shrinkage-LDA model emits one scalar score for each of the 1 s, 2 s, and 4 s scales; the headline asymmetry score is their equal-weight average. It never fits one classifier to the flattened 112-feature view-by-pair matrix.

The primary split protocol is persisted deterministic stratified five-fold within subject (32 train, 8 test for the expected 40 trials). A subject contributes to primary inference only with complete, exactly-once coverage of all 40 trial indices and no fold failure; invalid subjects and failures remain explicit artifacts. Repeated stratified 60/40 is a separately labeled sensitivity analysis whose subject metric is mean split-level balanced accuracy, not balanced accuracy over concatenated repeated test appearances.

## Leakage Controls

All inner branches share one deterministic inner split list per outer fold, with the seed formula persisted before model execution. Every tangent reference, scaler, classifier, asymmetry model, and optional S-JEPA downstream model is refit in the applicable inner training fold. View ranking and stability weights use inner OOF scores only. Branch margins are scaled only from rows used to fit that branch. The secondary stack is fit to inner-OOF base scores with fixed locked elastic-net hyperparameters; no second use of those folds for stack tuning occurs. The outer test split is transformed only after all outer-training choices are locked.

Split artifacts include indices and overlap assertions. Every result records `outer_test_used_for_selection: false`. Fold failures are recorded and are never replaced silently by another model or a constant prediction.

Covariance caches are namespaced by a signature over covariance, asymmetry, channel, marker, onset, filtering, and window settings. They validate labels, trial IDs/order, view IDs, expected tensor shapes and dtypes, and a SHA-256 source-file fingerprint. S-JEPA cache manifests match model ID/revision or a local checkpoint SHA-256, preprocessing, channel order, marker/window, hook/pooling, labels, and trial IDs; cached features must be finite floating-point two-dimensional matrices with the expected trial count. Policy `refuse` rejects both missing and incompatible caches without loading or writing a model. Recalculation requires policy `recompute` plus either an explicit non-null model revision or a local checkpoint whose hash is recorded; incompatible checkpoint keys are rejected.

## Dimensionality Rationale

There are only 32 outer-training trials in the primary protocol. Concatenating tangent coefficients from 32 views would create a very high-dimensional headline classifier whose apparent flexibility is disproportionate to the sample size. The locked design reduces every view or branch to one inner-honest scalar score before fusion. The asymmetry branch is restricted to four interpretable homologous-pair features per scale. Learned stacking is secondary, uses an actual elastic-net penalty with locked `C` and `l1_ratio`, and is trained only from inner OOF base scores; it is not hyperparameter-tuned in this notebook.

## Interpretation Boundaries

Results estimate within-subject decoding for these populations and protocols; they do not establish cross-subject or clinical generalization. Lv-14 results cannot be generalized to all 50 patients because the paper's exclusion criteria are not auditable. Repeated 60/40 values are sensitivity results, not interchangeable with five-fold values.

The correct classical context is that honest nested TWFB reproductions are approximately 50-52% balanced accuracy, while test-visible/leaky view selection produces approximately 67-80%. The published 72.21% lies in the leaky range, but the authors' exact selection protocol remains unconfirmed. This experiment therefore does not assume that faithful TWFB honestly performs much better than chance.

## Development-Mode Diagnostics

The measured locked run produced **54.40%** stability-selected top-k, **53.85%** secondary elastic stack, **52.50%** fixed single view, and **51.55%** nested best single view balanced accuracy. These values motivate development diagnostics; they do not alter the locked hypotheses or defaults.

Spatial modes are `full29`, `motor8`, symmetric `motor13`, and `average_reference_subspace`. `motor8` is FC3/FC4, C3/C4, CP3/CP4, and P3/P4. `motor13` honestly names the 13-channel symmetric extension adding F3/F4 and FCz/Cz/Pz. The subspace control average-references all 29 sensors and projects them through an explicit orthonormal 29-by-28 Helmert basis before covariance, removing the artificial reference null direction.

`full_trial_then_crop` filters each complete 8-s trial independently before the marker-relative 0-4 s crop; no filter crosses trials. `cropped_mi` preserves locked behavior. Estimators are OAS, fixed shrinkage, downsample-aware OAS, and block-averaged OAS. Cache identity includes spatial mode, selected sensors, resulting coordinates/basis, filter context, and estimator settings. Artifacts report shrinkage, effective rank, condition, and method complexity.

Fold-local classifiers are tangent shrinkage-LDA, affine-invariant MDM, log-Euclidean MDM, and tangent PCA plus ridge logistic. MDM uses signed distance-to-class-0 minus distance-to-class-1 margins scaled only from training margins. Prototype shrinkage is deliberately **target-only**, toward the unlabeled target outer-training Riemannian mean. Source-informed prototypes are not implemented because fold-safe cross-subject alignment would be substantially more invasive.

This sweep is exploratory internal model diagnosis. Any later configuration choice based on it is exploratory/internal reuse of the same 50 subjects and splits, **not independent confirmation**. Outer-test data must not select methods or hyperparameters.

Run the eight sequential development diagnostics with:

```bash
python src/utils/experiments/experiments.py \
  --notebook src/liu2024/liu2024_multiscale_riemann_fusion.ipynb \
  --configs src/utils/experiments/configs/sweep_multiscale_riemann_fusion_development.json \
  --kernel-name eeg-jepa --daemon
```

## Run Commands

Run the four locked configurations with:

```bash
python src/utils/experiments/experiments.py \
  --notebook src/liu2024/liu2024_multiscale_riemann_fusion.ipynb \
  --configs src/utils/experiments/configs/sweep_multiscale_riemann_fusion_locked.json \
  --kernel-name eeg-jepa --daemon
```

For classical-only execution without weights or downloads, run either of the first three sweep entries or leave `enable_sjepa` false. The optional S-JEPA entry intentionally uses `sjepa_cache_policy: refuse`; it fails before any download or cache write unless every subject has a provenance-matching cache. Recalculation must be an explicit separate configuration with `sjepa_cache_policy: recompute` and either a pinned non-null model revision or a local checkpoint.
