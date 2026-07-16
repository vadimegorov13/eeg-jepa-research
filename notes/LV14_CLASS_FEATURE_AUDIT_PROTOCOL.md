# Lv14 Class-Feature Audit Protocol

## Purpose

This audit answers the professor's request for subject-level left-versus-right motor-imagery feature differences in the uniquely mapped Lv et al. cohort: `sub-01, 03, 07, 09, 10, 11, 14, 15, 17, 29, 31, 32, 37, 41`. It asks whether larger within-subject separation is associated with better decoding. It does not treat a same-dataset association as prospective prediction or a causal biomarker.

## Locked Cohort And Labels

- The analysis retains all 14 mapped subjects. `sub-15` is flagged in outputs and is removed only in a separately labeled sensitivity summary.
- Liu labels 1/2 are mapped to 0/1 (left/right). Trial IDs, labels, channel order, marker handling, and source fingerprints are persisted.
- Liu29 means the first 30 recorded EEG channels with CPz/reference (zero-based channel 17) removed, in the project's canonical order.

## Branches

### Physiology

Each full 8-second trial is independently average-referenced and band-pass filtered before its marker-relative 0-4 second MI crop. No filter sees an adjacent trial. Log bandpower is computed for 8-12, 13-20, 20-30, and 8-30 Hz over C3/C4, FC3/FC4, CP3/CP4, and P3/P4, with motor13 regional summaries. Signed left-minus-right asymmetry and the right-hand-minus-left-hand hemisphere interaction are descriptive physiological contrasts. Hedges g, rank-biserial/Cliff effect, bootstrap confidence intervals, and within-subject label-permutation p-values are saved. BH-FDR is applied within declared feature families.

### Fixed Motor13 Riemann

OAS covariances are computed in a fixed 8-30 Hz view at 4 seconds and at prespecified non-overlapping 1-second and 2-second scales. Each outer fold fits tangent references, scaling, and shrinkage LDA using training trials only. Five-fold stratified OOF predictions cover every trial exactly once. Distances to fold-training class centroids are descriptive fold-local quantities.

### Frozen S-JEPA

`SignalJEPA_PreLocal.from_pretrained` is loaded from the project's model ID using an explicit configured revision or a uniquely resolved cached Hugging Face commit. The frozen `feature_encoder` output is mean-pooled to one 64-dimensional vector per trial; classifier logits are rejected. The cache manifest records model ID and resolved revision, local checkpoint digest when applicable, source fingerprints, channels, preprocessing, trial IDs, and labels. A requested but unavailable model fails loudly. Scaling and shrinkage LDA are fold-local, and inference claims use OOF scores only. PCA is an explicitly transductive descriptive visualization.

### Microstates

Four group maps are fit using all 14 subjects, so every microstate result is labeled transductive and is not inductive classifier evidence. Default names are neutral (`MS0`-`MS3`). Optional `proxy_canonical` naming matches learned maps to deterministic geometry seeds, saves the centers and matching matrix, and remains a proxy rather than an exact reproduction of Lv's manual/undocumented canonical assignment. The 24-feature family is exactly 12 duration/occurrence/coverage measures plus 12 directed off-diagonal transitions; self-transitions are excluded. Lv's reported transition family is B-to-A, D-to-A, and D-to-C. Paper expected directions, Table 2 means/effect sizes/FDR values, and Fisher scores are attached only in proxy mode.

The audit uses all retained 4-second trials per condition. Lv states that 60 seconds per condition were used but does not disclose the exact segment-selection rule, so this is a declared protocol difference rather than an exact reproduction.

## Association And Generalization Analyses

- A small prespecified family relates branch separation magnitudes to locked Shallow BA, locked Lv14 PreLocal BA, and same-run Riemann/S-JEPA BA. Spearman and Kendall estimates use exhaustive `2^n` centered-performance sign-flip permutation tests, subject bootstrap confidence intervals, leave-one-subject-out sign/range diagnostics, and Holm correction.
- These are same-dataset associations and are labeled accordingly.
- Repeated balanced split-half analysis estimates separation on one half and evaluates a simple classifier on the disjoint half. Deterministic swaps make both halves serve as train/test. Correlations are summarized over fixed seeds; this is the stronger generalization check but remains internal to 14 subjects.

## Interpretation Limits

- Fourteen selectively retained patients are not the full Liu cohort, and Lv's original inclusion rule remains unavailable.
- The audit is multiplicity-aware but low-powered. Null correlations do not prove that physiology and decoding are unrelated; positive correlations do not establish transportability.
- Microstate maps are cohort-transductive. Neutral states have no A/B/C/D interpretation. Proxy states are geometry matched and are never called exact canonical states.
- Locked external branch rows are loaded only from provenance-declared artifacts and are not silently substituted. Missing optional locked rows are reported as unavailable.
- The S-JEPA embedding branch is a frozen probe, distinct from the fine-tuned PreLocal decoder whose locked performance may be shown as a side bar.

## Primary Outputs

The canonical notebook writes collision-safe artifacts under `artifacts/liu2024-lv14-class-feature-audit/<run_id>/`. `run_metadata.json` indexes every table and plot; `global_metrics.json` records cohort, OOF coverage assertions, multiplicity families, transductive labels, branch availability, and aggregate performance. Existing artifact directories are never modified.

## Full50 Extension Protocol

The retained notebook also supports a prespecified Full50 extension through
`src/utils/experiments/configs/sweep_full50_class_feature_audit.json`. This extension sets
`cohort_name=full50` and `subjects_to_use=null`; the loader must then discover exactly one source
file for every `sub-01` through `sub-50`. The mapped-Lv14 default and its prior timestamped
artifacts remain unchanged. The extension is a new run under the same collision-safe artifact
root, not a replacement or reinterpretation of the completed Lv14 result.

- All branches use the dynamically selected cohort. Four-state neutral microstate maps are fit to
  the entire configured cohort and remain explicitly all-cohort transductive, with exactly 24
  features and no canonical A/B/C/D claim.
- Physiology retains the existing homologous-pair and motor13 summaries and additionally records
  MI-window log bandpower for all Liu29 channels in each configured band. Subject-level channel
  effects and a declared cohort-level family are saved separately.
- Cohort-level physiology inference tests every channel x band plus the prespecified homologous
  asymmetry and hand-by-hemisphere features. The estimand is the across-subject mean of each
  subject's right-hand-minus-left-hand MI-window log-bandpower difference. Outputs include a
  subject-bootstrap 95% CI, one-sample Wilcoxon test, and BH correction across this complete
  declared family.
- Scalp maps use MNE's `standard_1020` montage and the canonical Liu29 names. Their polarity is
  always right-hand minus left-hand. These are MI-window bandpower differences, not
  baseline-normalized ERD. Subject maps use a common robust symmetric scale within each band;
  representative panels use fixed IDs `01, 07, 11, 14, 29, 31, 37, 41` intersected with the
  configured cohort, so selection is auditable and independent of observed effects.
- External locked Shallow and PreLocal metrics are optional annotations. Only available configured
  subjects are merged; missing files or subject rows are recorded in metadata and do not abort the
  audit.
- For more than 20 subjects, correlation sign-flip p-values use the configured deterministic Monte
  Carlo iteration count rather than the infeasible exhaustive `2^n` enumeration used for Lv14.
  The method and iteration count are recorded per test.

The Full50 extension is a protocol declaration only. No Full50 result is reported here until the
new sweep is executed and its timestamped artifact is reviewed.

## Completed Results (2026-07-14)

Canonical neutral-state artifact:

`artifacts/liu2024-lv14-class-feature-audit/20260714_211953_909023_a14ace51/`

Corrected geometry-proxy sensitivity artifact:

`artifacts/liu2024-lv14-class-feature-audit/20260714_212547_367419_f47be39f/`

- The primary neutral microstate analysis retained all 14 subjects and tested exactly 24 features. No paired left/right feature survived BH correction; the minimum adjusted value was `q = 0.6492`. These group maps are transductive and have no canonical A/B/C/D interpretation.
- The geometry-proxy sensitivity agreed with Lv's reported direction for eight of nine named features, but none survived correction (`q >= 0.6492`) and the effects were much smaller than Lv reported. The A-duration proxy was the largest named result (`d_z = -0.534` using right-minus-left, paired t `p = 0.0669`); it is not significant and the proxy labels are not exact Lv canonical states. The proxy implementation asserts that all four seeds remain distinct under polarity invariance.
- The fixed motor13 Riemann branch reached `49.82%` mean exactly-once OOF BA and `49.89%` AUC. Performance was highly heterogeneous, from `30.0%` to `72.5%` BA.
- The frozen 64-dimensional S-JEPA probe reached `51.96%` mean exactly-once OOF BA and `51.43%` AUC. No embedding dimension survived within-subject BH correction, and no subject's linear-MMD permutation test crossed `p < 0.05` (minimum `p = 0.05097`, subject 03). Subject 07 was the clearest held-out S-JEPA case (`70.0%` BA, `76.25%` AUC); subject 29 showed reversed separation (`37.5%` BA, `28.75%` AUC).
- Sixty-one physiological feature rows survived the declared within-subject x band x family BH correction, concentrated in five subjects: 07 (17 rows), 09 (2), 29 (6), 31 (15), and 37 (21). This is patient-specific evidence, not a cohort-level family correction. The strongest representative effects were beta/low-gamma motor or homologous-pair features in those five subjects.
- Across 100 balanced split-halves with both train/test swaps, mean held-out BA was `53.59%` for physiology, `51.89%` for Riemann, and `51.51%` for S-JEPA. The mean cross-subject Spearman association between training-half separation and disjoint held-out BA was `0.259`, `0.205`, and `-0.028`, respectively. These repeat summaries are internal diagnostics, not independent confirmatory p-values.
- Same-data S-JEPA centroid separation correlated with same-run S-JEPA BA (`rho = 0.707`, nominal exact sign-flip `p = 0.0254`), but this did not survive the prespecified eight-test Holm family (`p_adj = 0.203`). Physiology-to-locked-Shallow, microstate-to-locked-Shallow, and Riemann-separation-to-Riemann-BA associations were null after correction.

The defensible answer to the professor's request is therefore: there is no cohort-wide "obvious correlation" across the mapped Lv14. A minority of patients show clear trial-level physiological class differences, while microstate and frozen S-JEPA evidence is weak or inconsistent under multiplicity-aware testing. This heterogeneity is compatible with the broader finding that the acute-stroke dataset is difficult, but it does not reproduce Lv's reported nine-feature inferential pattern.

## Completed Full50 Extension (2026-07-14)

Canonical Full50 artifact:

`artifacts/liu2024-lv14-class-feature-audit/20260714_221949_123930_a845f385/`

Full findings report:

`../Research Documents/My Reports/full50_class_feature_audit_findings.md`

- The extension retained all 50 subjects and evaluated all 2,000 trials exactly once in both audit probes.
- Fixed motor13 Riemann reached 54.05% mean subject BA (95% subject-bootstrap CI 50.25-57.85%); the frozen S-JEPA probe reached 53.55% (51.10-55.90%).
- None of 148 cohort-level channel x band and prespecified asymmetry tests survived BH correction. The minimum q was 0.1595 despite nominal central/parietal beta and broad-band asymmetry effects.
- Local within-subject x band x family corrections identified physiological effects in 18/50 patients. These are patient-specific findings and are not a globally corrected cohort result.
- No neutral-state microstate feature survived BH (minimum q = 0.7003).
- Nine frozen S-JEPA dimensions survived within-subject BH in one patient (`sub-34`); five patients had nominal multivariate embedding MMD p < 0.05. This does not establish a common latent feature across patients.
- Same-data S-JEPA separation correlated with S-JEPA BA after Holm correction (rho = 0.441, adjusted p = 0.0448). Physiology-to-locked-Shallow, Riemann-to-Riemann, and microstate-to-locked-Shallow associations did not survive the eight-test family.
- In repeated disjoint split-halves, mean held-out BA was 53.13% / 52.71% / 52.23% for physiology / Riemann / S-JEPA. Mean separation-to-held-out-BA rho was 0.077 / 0.220 / 0.024.
- The generated 29-channel maps are MI-window right-minus-left log-bandpower differences and signed Hedges g, not baseline-normalized ERD or source localization.

The Full50 result strengthens the heterogeneity interpretation: visually structured average maps coexist with zero BH-significant cohort features because patient maps vary substantially in sign, location, and magnitude. Full50 should be primary; Lv14 remains a selectively retained paper-comparison sensitivity.
