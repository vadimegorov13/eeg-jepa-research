# Lv et al. 2025 Microstate Reproduction: Diagnosis and Honest Protocol

## Current conclusion

The reported 75.00% KNN accuracy is not currently reproducible from the published description and
the Liu2024 source data. This is not evidence that the number is false. The paper leaves several
analysis-defining choices underspecified, and its 14-patient cohort was selected from 50 patients
without falsifiable inclusion criteria. No new accuracy is claimed here because the corrected and
new notebooks have not been run.

The 14 Lv-local patient IDs map uniquely, by the reported clinical variables, to Liu subjects
`01, 03, 07, 09, 10, 11, 14, 15, 17, 29, 31, 32, 37, 41`. This mapping is derived rather than author
confirmed. `sub-15` has conspicuous amplitude/QC behavior, but the corrected default retains it and
records a QC flag. Excluding it after observing the data would change the target cohort and could
inflate an already small-sample estimate.

## Why an exact reproduction is blocked

- Manual ICA component and epoch rejection decisions are unavailable.
- The stated use of 60 seconds per condition is ambiguous: fixed chronological trials, selected
  clean trials, or typical retained duration.
- Canonical A/B/C/D labels require an unreported visual/topographic matching rule. Cluster indices
  are permutation-indeterminate, so paper-named features cannot be assumed from `MS0..MS3`.
- The exact temporal smoothing/backfitting implementation and transition counting conventions are
  not fully specified.
- The nine classifier features are described as significant, but the precise multiplicity family,
  fallback behavior when none survive FDR, and whether selection occurred inside CV are unclear.
- Ten-fold CV over two condition summaries per subject can put the same patient's paired records in
  train and test. Preprocessing, group maps, feature selection, and scaling can add further leakage
  if estimated before splitting.
- KNN neighborhood, scaling, SVM scoring, AUC aggregation, and paired classifier-comparison details
  are insufficient for bitwise protocol reconstruction.

## Corrected notebook protocol

`src/liu2024/liu2024_lv2025_microstate_reproduction.ipynb` keeps two clearly separated modes.

- `paper_style` is a comparability analysis. It can use the exact nine reported features only when
  an explicit A/B/C/D state-label map is supplied. Otherwise it uses declared data-driven features
  and does not pretend cluster order identifies paper states.
- `honest_grouped` holds out whole subjects and repeats feature testing on outer-training subjects.
  FDR-surviving features are labeled `fdr_significant`; if no feature survives, a ranked exploratory
  fallback is labeled `exploratory_fallback_not_significant` everywhere.
- Pooled out-of-fold predictions provide the primary metrics. SVM AUC uses `decision_function`.
  Classifier comparisons use exact McNemar/binomial tests rather than the uncorrected asymptotic
  formula.
- QC is recorded independently of classification performance. The default does not exclude flagged
  subjects.
- Group microstate maps are currently fit once across the cohort. Therefore grouped results are
  explicitly **transductive**, not fully inductive, and require
  `allow_transductive_group_maps=true`. A fully inductive claim requires refitting and matching maps
  separately inside every outer fold.

## Ranked improvement rationale

1. Refit and label group maps inside each outer-training fold. This removes the remaining
   transductive information path and is more important than classifier tuning.
2. Obtain the authors' original preprocessing decisions, state maps, retained-epoch lists, and
   exact nine-feature table. These determine whether `paper_style` is truly comparable.
3. Evaluate fixed OAS covariance views with fold-safe Riemannian alignment. This tests a mature,
   low-variance transfer family without selecting windows or bands on outer-test data.
4. Stabilize S-JEPA's actual transfer bottleneck (`spatial_conv`) using deviation, norm, and
   orthogonality constraints, separate learning rates, clipping, and collapse diagnostics.
5. Report both full-50 and mapped-14 results, per-subject paired statistics, pooled OOF metrics,
   confidence intervals, and all exclusions. Do not tune a method on the 14-subject result and then
   describe it as confirmatory.

## Primary sources

- Barachant, Bonnet, Congedo, and Jutten (2010), *Riemannian geometry applied to BCI
  classification*, LVA/ICA. https://doi.org/10.1007/978-3-642-15995-4_45
- Barachant, Bonnet, Congedo, and Jutten (2012), *Multiclass brain-computer interface
  classification by Riemannian geometry*, IEEE TBME. https://doi.org/10.1109/TBME.2011.2172210
- Yger, Berar, and Lotte (2017), *Riemannian approaches in brain-computer interfaces: a review*,
  IEEE TNSRE. https://doi.org/10.1109/TNSRE.2016.2627016
- He and Wu (2020), *Transfer learning for brain-computer interfaces: a Euclidean space data
  alignment approach*, IEEE TBME. https://doi.org/10.1109/TBME.2019.2913914
- Zanini et al. (2018), *Transfer learning: a Riemannian geometry framework with applications to
  brain-computer interfaces*, IEEE TBME. https://doi.org/10.1109/TBME.2017.2742541
- Rodrigues et al. (2019), *Riemannian Procrustes analysis: transfer learning for brain-computer
  interfaces*, IEEE TBME. https://doi.org/10.1109/TBME.2018.2889705
- Lotte and Guan (2011), *Regularizing common spatial patterns to improve BCI designs: unified
  theory and new algorithms*, IEEE TBME. https://doi.org/10.1109/TBME.2010.2082539
- Guetschel et al., *S-JEPA: Towards Seamless Cross-Dataset Transfer through Dynamic Spatial
  Attention*. Use the local primary paper in `Research Documents/Papers/` for the final bibliography
  entry and exact publication metadata; the released `SignalJEPA_PreLocal` architecture is the
  implementation reference used by these notebooks.
- Lv et al. (2025), *Journal of NeuroEngineering and Rehabilitation* 22:137, and Liu et al. (2024),
  *Scientific Data* 11:131, are the dataset/reproduction primary sources.
