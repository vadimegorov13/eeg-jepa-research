# Cohort Selection Sensitivity Protocol

## Status And Scope

This protocol governs `src/liu2024/liu2024_cohort_selection_sensitivity.ipynb`. The primary input is
the locked ShallowFBCSPNet exactly-once OOF artifact
`artifacts/liu2024-compact-mi-models/20260712_165746_790145_fd8ab986/`. The optional 80-epoch
whole-trial Shallow artifact is exploratory development and must be analyzed in a separate run and
reported separately. This analysis does not retrain either decoder.

The estimand is mean patient balanced accuracy (BA). All uncertainty intervals are percentile
patient-bootstrap intervals. The fixed Lv14 cohort is `sub-01, sub-03, sub-07, sub-09, sub-10,
sub-11, sub-14, sub-15, sub-17, sub-29, sub-31, sub-32, sub-37, sub-41`; it is a sensitivity cohort,
not a data-driven recommendation. Full50, Lv14, and non-Lv36 coverage must be stated exactly.

## Completed Locked Result

The completed artifact is
`EEG_JEPA/artifacts/liu2024-cohort-selection-sensitivity/20260713_005254_206429_9bbef9d8/`.
It validates all 50 expected subjects, 40 trials per subject, and 2,000 exactly-once OOF
predictions from the locked 20-epoch ShallowFBCSPNet artifact.

### Fixed cohorts and QC

| Cohort | Patients / coverage | Mean subject BA (95% patient-bootstrap CI) |
|---|---:|---:|
| Full50 | 50 / 100% | 57.45% (54.15-60.70%) |
| Lv14 fixed | 14 / 28% | 52.86% (47.14-59.46%) |
| non-Lv36 | 36 / 72% | 59.24% (55.56-62.92%) |
| Fixed QC combined | 46 / 92% | 57.77% (54.35-61.14%) |

Lv14 is descriptively worse than Full50 by 4.59 points and worse than non-Lv36 by 6.38 points;
the artifact supplies no paired inferential test for these fixed-cohort contrasts. The fixed QC
policy rejects three patients on marker validity and one different patient on covariance numerical
validity; all 50 pass every other fixed criterion. Retaining 46/50 changes BA by only +0.32 points.
Because these label-blind QC summaries use each patient's full recording and are not fold-local,
this is retrospective sensitivity evidence, not a validated prospective rejection policy.

### Nested selector

| Retained coverage | Patients | Mean subject BA (95% patient-bootstrap CI) |
|---:|---:|---:|
| 100% | 50 | 57.45% (54.20-60.75%) |
| 90% | 45 | 57.72% (54.17-61.11%) |
| 80% | 40 | 57.75% (53.88-61.63%) |
| 70% | 35 | 58.50% (54.43-62.50%) |
| 60% | 30 | 59.17% (54.83-63.50%) |
| 50% | 25 | 60.80% (56.10-65.50%) |

This favorable descriptive coverage curve does not establish a selector. Cross-fitted diagnostics
are weak: R2 = -0.0514, MAE = 9.85 BA points, Spearman rho = 0.1797 (`p = 0.2119`), calibration
slope = 0.2284, and intercept = 0.4443. Inner CV selected Ridge alpha 100 in four outer folds and
0.1 in one. Scores were written before outcome joining (SHA256
`aacd5c5130cfd56d51d03405d06700840502c30c9a275c8479d7a1a7eee6f494`), but the same locked
patient outcomes serve as selector targets across folds and evaluation outcomes. The curve is
nested descriptive evidence, not external validation or a deployable restriction.

### Clinical strata

The six prespecified omnibus Kruskal-Wallis tests form one Holm family. None is significant before
or after correction.

| Family | Raw p | Holm p |
|---|---:|---:|
| Paralysis side | 0.7990 | 1.0000 |
| NIHSS | 0.1213 | 0.6065 |
| Age | 0.2183 | 0.7240 |
| Duration | 0.0905 | 0.5432 |
| mRS | 0.1810 | 0.7240 |
| MBI | 0.5146 | 1.0000 |

Some small strata have high descriptive means, such as duration 4-7 days at 63.39% (`n = 14`),
mRS 5 at 70.00% (`n = 1`), and MBI 100 at 65.00% (`n = 3`). These are not multiplicity-supported
cohorts and must not be promoted into selection rules.

### Trial abstention and invalid oracle

Posthoc ranking by uncalibrated maximum class probability gives pooled selective BA of 57.45,
57.83, 58.88, 59.57, 60.32, 60.87, 60.86, 60.45, 58.16, and 58.00% at 100, 90, 80, 70, 60,
50, 40, 30, 20, and 10% trial coverage, respectively. The apparent maximum is only 60.87% while
discarding half of all trials. At 50% coverage one patient already lacks retained examples from
both classes for defined patient BA; lower coverage creates additional undefined cases. This is
posthoc descriptive abstention, not a calibrated selective classifier.

The outcome-ranked oracle retained-cohort BA is 57.45, 59.67, 61.37, 63.14, 65.08, 67.30, 69.38,
71.00, 73.50, and 77.00% at 100 through 10% coverage in 10-point steps. Most importantly, the
oracle needs only the observed top **7/50** patients (14% coverage) to manufacture **75.36% BA** (95% CI
72.86-77.86%). This ranking directly uses each patient's observed held-out BA, so it is circular,
invalid, and not deployable; it quantifies how outcome-selected exclusion can create a 75% result,
not how to identify patients prospectively.

### Decision

No deployable cohort restriction is justified. Fixed QC has negligible effect and is transductive;
Lv14 is worse; no prespecified clinical family survives Holm correction; the selector has negative
R2, weak nonsignificant rank association, and no external validation; and trial abstention is
posthoc and uncalibrated. Full50 therefore remains the primary estimand and result.

## Clinical Strata

Bins are fixed before examining decoder outcomes:

- Paralysis side: every observed level, including missing as an explicit level.
- NIHSS: 0 no symptoms; 1-4 minor; 5-15 moderate; 16-20 moderate-to-severe; 21-42 severe.
- Age: `<45`, `45-64`, and `>=65` years.
- Duration from stroke onset to enrollment: `1-3`, `4-7`, `8-14`, and `>=15` days.
- mRS: exact grades 0 through 5.
- MBI: 0-20 total dependence; 21-60 severe dependence; 61-90 moderate dependence; 91-99
  slight dependence; 100 independent.

Every declared category is emitted, including empty categories. Six omnibus Kruskal-Wallis tests,
one for each clinical variable, form the only declared clinical multiplicity family and are adjusted
with Holm. These tests describe heterogeneity; no best-performing stratum may be promoted as a
claimed cohort.

## Label-Blind Signal QC

QC uses no task labels. Features are computed from complete raw trials and therefore are
**full-recording transductive patient summaries**. They are suitable for retrospective sensitivity
analysis, not strict prospective fold-local rejection. A strict trial-decoding deployment would fit
every data-dependent transform and threshold using training-fold recordings only, then apply it to
held-out trials without using their aggregate distribution.

The feature signature includes the QC version, source files and sizes, channel order, sampling rate,
marker rule, filters, and thresholds. Cache files live only under
`artifacts/liu2024-cohort-selection-sensitivity/qc_cache/qc_<signature>/`; existing artifacts are
never changed. Features include marker fallback count, robust channel SD/MAD, peak-to-peak,
clipped and robust-extreme sample fractions, flat-channel count, pre-notch 50-Hz line ratio,
30-40/8-30 high-frequency ratio, channel-variance dispersion, motor13 8-30 Hz covariance effective
rank and condition number, and odd/even-trial split-half reliability of unlabeled log-bandpower.

## Fixed QC Policies

Thresholds are physical or numerical plausibility checks, not accuracy-tuned rules:

- At least 95% of trials have a valid marker; at most two marker fallbacks.
- All retained EEG values are finite.
- Median robust channel scale is between 0.1 and 1,000 acquisition units.
- No channel has robust scale below 0.05 acquisition units.
- Median trial-channel peak-to-peak is at most 5,000 acquisition units.
- At most 1% of samples exceed 20 robust standard deviations.
- At most 0.1% of centered samples reach the conservative 16-bit amplitude rail of 32,760 units.
- Motor13 OAS covariance condition number is finite and at most 1e8; effective rank is at least 2.

Line-noise ratio, high-frequency ratio, variance dispersion, and split-half reliability are reported
but are not exclusion rules because physiological and environmental variation makes universal
cutoffs indefensible here. Every criterion's pass count is reported. The combined policy must retain
at least 70% of patients; otherwise execution stops unless every additional rejection is explicitly
classified as a hard hardware failure (non-finite data or flat channels). Thresholds are never
relaxed after seeing BA.

## Learned Reliability Selector

The learned selector is nested across patients. A fixed shuffled five-fold outer patient split
generates one cross-fitted reliability score for every patient. In each outer development set, a
pipeline standardizes at most these eight prespecified QC features and tunes Ridge alpha by inner
patient CV: marker fallback count, median robust SD, median peak-to-peak, extreme fraction,
flat-channel count, 50-Hz line ratio, high-frequency ratio, and split-half reliability. The target is
the locked development-patient OOF BA. No held-out patient's BA enters preprocessing, alpha choice,
or fitting. Held-out scores are saved to `selector_scores_pre_evaluation.csv` before outcomes are
joined. Cross-fitted R2, MAE, Spearman correlation, and calibration slope are descriptive estimates
with only 50 patients.

Coverage-risk rows at 100, 90, 80, 70, 60, and 50% rank patients only by cross-fitted predicted
reliability and report retained and rejected BA, intervals, and clinical composition. They evaluate
a reliability-ranking procedure; because the decoder outcomes used as selector targets are reused
across outer selector folds, they are not an external validation of a deployment policy.

## Invalid Oracle And Trial Abstention

`oracle_invalid_curve.csv` ranks the same patients by observed held-out BA. It is circular,
outcome-selected, invalid, and not deployable. Its sole purpose is to show how selective exclusions
can manufacture a high cohort mean such as 75%. It must remain structurally separate and red-labeled
in figures and must never be recommended.

Trial-level abstention ranks existing OOF predictions by uncalibrated maximum class probability.
It is posthoc descriptive, not a calibrated selective classifier. Rows report achieved coverage,
selective BA, class-specific coverage and sensitivity, and explicit undefined cases when either
class is absent overall or within a patient.

## Required Validation And Outputs

Execution must fail unless subject IDs are exactly `sub-01` through `sub-50`, each subject has
exactly 40 unique test indices equal to `0..39`, labels are consistent per trial, probabilities are
finite two-class distributions, and the participant table matches one-to-one. Required outputs are
`cohort_results.csv`, `qc_features.csv`, `selector_predictions.csv`, `coverage_risk.csv`,
`oracle_invalid_curve.csv`, `clinical_strata.csv`, `trial_abstention.csv`, `global_metrics.json`,
plots, configuration, run metadata, logs, and the immutable pre-evaluation selector-score file.
