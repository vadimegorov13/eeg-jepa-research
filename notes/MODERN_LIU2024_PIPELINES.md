# Modern Liu2024 Pipelines

## Status

Five canonical notebooks define the modern neural comparisons. The six-run compact target-only benchmark, prespecified two-target LOSO pilot, full-source Shallow smoke test, focused lightweight sweep, S-JEPA LP-FT comparison, and bounded Shallow augmentation smoke test are complete. Exploratory cropped and longer-training ShallowFBCSP runs are also complete. Neither supervised source transfer, strict LP-FT, nor the tested light augmentations improve accuracy. Foundation-model work remains validation-only until pinned official repositories and checkpoints are supplied. No large checkpoints were downloaded.

The current confirmed honest classical best is **55.375% mean subject balanced accuracy** (95% subject-bootstrap CI 52.313-58.588%) from the frozen full-50 motor13 short-scale repeated 60/40 protocol: `artifacts/liu2024-multiscale-riemann-fusion/20260712_145032_844360_89d4f1fc/`. The best comparable S-JEPA PreLocal result is **57.00%**, but 42% of folds collapse and useful accuracy is concentrated in non-collapsed folds; it is weak-to-moderate transfer, not a strong clinical decoder.

## Results

All six locked compact-model runs completed on all 50 subjects. Values below are mean subject balanced accuracy (BA). The paired delta is compact minus the confirmed `riemann_equal_short_scales` subject BA from `artifacts/liu2024-multiscale-riemann-fusion/20260712_145032_844360_89d4f1fc/subject_metrics.json`. Delta intervals are 10,000-draw paired subject-bootstrap 95% CIs (seed 202607); p-values are two-sided subject-level Wilcoxon signed-rank tests, with Holm adjustment across these six prespecified compact-versus-classical comparisons.

| Model | Artifact | BA (95% CI) | Delta vs classical (95% CI), points | Raw p | Holm p | Win/tie/loss | Collapsed folds | Predicted class 1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FBCNet | `artifacts/liu2024-compact-mi-models/20260712_152445_040475_07bbb7c0/` | 54.00% (50.25-57.75) | -1.375 (-4.638 to +1.838) | 0.3929 | 0.5227 | 22/0/28 | 25/250 (10.0%) | 51.90% |
| EEGTCNet | `artifacts/liu2024-compact-mi-models/20260712_153421_569587_eabbd737/` | 49.75% (49.40-50.15) | -5.625 (-8.788 to -2.562) | 0.00222 | **0.01331** | 17/1/32 | 236/250 (94.4%) | 0.75% |
| IFNet | `artifacts/liu2024-compact-mi-models/20260712_154849_319559_a13227b7/` | 52.35% (49.55-55.20) | -3.025 (-6.263 to +0.250) | 0.0830 | 0.3320 | 17/2/31 | 7/250 (2.8%) | 51.65% |
| FBMSNet | `artifacts/liu2024-compact-mi-models/20260712_155420_204470_85397e63/` | 53.25% (49.55-57.10) | -2.125 (-5.288 to +1.013) | 0.2614 | 0.5227 | 20/2/28 | 14/250 (5.6%) | 51.25% |
| EEGNet | `artifacts/liu2024-compact-mi-models/20260712_165320_889869_62765f8c/` | 50.20% (48.90-51.40) | -5.175 (-8.738 to -1.763) | 0.01143 | 0.05713 | 17/4/29 | 171/250 (68.4%) | 14.30% |
| **ShallowFBCSPNet** | `artifacts/liu2024-compact-mi-models/20260712_165746_790145_fd8ab986/` | **57.45% (54.20-60.65)** | **+2.075 (-1.438 to +5.388)** | 0.1689 | 0.5068 | **31/0/19** | **6/250 (2.4%)** | 48.65% |

The classical reference is 55.375% BA (52.313-58.588%). This is a matched-subject comparison, not an identical-resampling comparison: compact BA is pooled exactly-once five-fold OOF performance, whereas classical subject BA is the mean of ten stratified 60/40 repeat-level BAs. The comparison therefore measures subject-level performance under each locked protocol and should not be described as trial-paired or fold-paired. The completed run configs point to a nonexistent `subject_results.csv`, so their built-in paired-test blocks did not run. The notebook default is now corrected to load the canonical `subject_metrics.json`, filter `method == "riemann_equal_short_scales"`, and retain optional CSV support; the table above was computed from that method-filtered JSON.

### Cohort-selection sensitivity

Canonical artifact:
`EEG_JEPA/artifacts/liu2024-cohort-selection-sensitivity/20260713_005254_206429_9bbef9d8/`.
It reuses the locked ShallowFBCSPNet's 2,000 exactly-once OOF predictions and does not retrain the
decoder.

| Analysis | Patients / coverage | Mean subject BA (95% CI) |
|---|---:|---:|
| Full50 | 50 / 100% | 57.45% (54.15-60.70%) |
| Lv14 fixed | 14 / 28% | 52.86% (47.14-59.46%) |
| non-Lv36 | 36 / 72% | 59.24% (55.56-62.92%) |
| Fixed label-blind QC | 46 / 92% | 57.77% (54.35-61.14%) |

Lv14 is worse, not better: -4.59 points versus Full50 and -6.38 versus non-Lv36. Fixed QC rejects
three marker-validity failures and one separate covariance-validity failure but changes BA by only
+0.32 points; its full-recording summaries are transductive rather than prospective fold-local QC.
The nested selector's retained BA rises descriptively from 57.45% at 100% coverage to 57.72,
57.75, 58.50, 59.17, and 60.80% at 90, 80, 70, 60, and 50% coverage. Its diagnostics do not support
deployment: R2 = -0.0514, MAE = 9.85 points, Spearman rho = 0.1797 (`p = 0.2119`), and calibration
slope = 0.2284. It is nested but not externally validated because locked patient outcomes are reused
across selector training and evaluation.

None of the six prespecified clinical omnibus tests is significant after Holm correction: adjusted
`p = 1.0000` for paralysis side, 0.6065 for NIHSS, 0.7240 for age, 0.5432 for duration, 0.7240 for
mRS, and 1.0000 for MBI. Posthoc uncalibrated trial abstention peaks at 60.87% pooled BA while
retaining only 50% of trials and then degrades. The explicitly invalid oracle reaches 67.30% at 50%
patient coverage and needs only the observed top **7/50** patients to manufacture **75.36% BA**.
That oracle ranks by observed held-out BA and is circular, invalid, and not deployable.

No deployable cohort restriction is justified: Full50 remains primary, and fixed QC, clinical
stratification, nested reliability selection, and trial confidence require prospective external
validation before they could define a target population or abstention policy.

### Latest lightweight and Shallow experiments

All successful runs below use the same 50 subjects, 250 five-fold OOF partitions, Liu29 preprocessing, seed 2026, and split hash `801ec1d2c981335f` as the locked 20-epoch ShallowFBCSPNet baseline. Deltas are run minus locked Shallow subject BA. Delta intervals are 10,000-draw paired subject-bootstrap 95% CIs with seed 202607; p-values are exploratory, uncorrected, two-sided subject-level Wilcoxon tests.

| Run | Status / artifact | BA (95% CI) | Delta vs locked Shallow (95% CI), points | Raw p | Win/tie/loss | Collapsed folds | Predicted class 1 |
|---|---|---:|---:|---:|---:|---:|---:|
| FBLightConvNet, 20 epochs | Locked; `artifacts/liu2024-lightweight-mi-models/20260712_175818_418459_9b71ac89/` | 50.35% (46.95-53.80) | -7.10 (-10.45 to -3.80) | 0.000217 | 15/2/33 | 40/250 (16.0%) | 67.45% |
| EEGITNet, 20 epochs | Locked; `artifacts/liu2024-lightweight-mi-models/20260712_180522_285112_d9904883/` | 50.45% (48.95-52.10) | -7.00 (-10.65 to -3.25) | 0.000703 | 12/4/34 | 126/250 (50.4%) | 27.45% |
| SincShallowNet, 20 epochs | Failed; no artifact | - | - | - | - | - | - |
| Cropped Shallow, 20 epochs | Exploratory; `artifacts/liu2024-cropped-shallow-fusion/20260712_181855_586304_de5be280/` | 55.00% (50.45-59.65) | -2.45 (-6.10 to +1.30) | 0.1983 | 19/5/26 | 10/250 (4.0%) | 46.10% |
| Whole-trial Shallow, 80 epochs | Exploratory; `artifacts/liu2024-cropped-shallow-fusion/20260712_192703_359232_73d733d7/` | **57.90% (54.75-61.10)** | +0.45 (-1.251 to +2.20) | 0.6788 | 21/9/20 | 5/250 (2.0%) | 50.30% |

- **The locked neural headline remains 57.45%, not 57.90%.** The 80-epoch whole-trial run was part of exploratory development and is not an independent confirmation. Its +0.45-point paired gain is uncertain and not broad across subjects: 21 improve, 20 worsen, 9 tie, and the median delta is 0. The small positive mean is nevertheless not a single-outlier artifact: every leave-one-subject-out mean remains positive (+0.102 to +0.714 points), and symmetric 1/2/3/5-per-tail trimmed means remain +0.313 to +0.398 points. Collapse is 2.0%, predictions are balanced (49.70% class 0 / 50.30% class 1), and class-0/class-1 sensitivities are 57.6%/58.2%.
- **Cropping did not improve the locked baseline.** Fixed five-crop training and logit averaging reached 55.00%, a nonsignificant -2.45 points versus locked Shallow, with more losses than wins. It has modest class-0 bias (53.90% of predictions; class-0/class-1 sensitivities 58.9%/51.1%) and 4.0% collapse. This is evidence against the tested crop configuration, not against all cropped decoding.
- **The two completed lightweight models are near chance and substantially below locked Shallow.** FBLightConvNet favors class 1 (67.45% of predictions; class-0/class-1 sensitivities 32.9%/67.8%); 38/40 collapsed folds favor class 1. EEGITNet favors class 0 (72.55% of predictions; sensitivities 73.0%/27.9%); 105/126 collapsed folds favor class 0. Their non-collapsed fold BAs are only 50.42% and 50.91%, respectively, so removing collapsed folds does not reveal a strong decoder.
- **SincShallowNet has no performance result.** Run `experiment_results/20260712_1758_sweep_lightweight_mi_models_locked_0e36dabf/` failed on the first training backward pass with PyTorch `RuntimeError: view size is not compatible with input tensor's size and stride ... Use .reshape(...) instead.` This is an implementation/runtime incompatibility, not evidence about SincShallowNet accuracy, and no artifact directory was created.

The current next recommendation is a frozen/probed EEGPT experiment after official code, checkpoint, channel mapping, and digest validation. MIRepNet is the next external-model route after EEGPT. The completed same-fold fusion and source-alignment experiments are null and should not be tuned against their outer results. These results provide no evidence that any route will reach 75%.

### July 19 interrupted Liu2024 SSL pretraining

Training-only montage-aware sensor rotation and 2 microvolt Gaussian noise passed a one-epoch
structural smoke test at
`artifacts/liu2024-s-jepa-pretraining-gacl-aug-smoke/20260719_1133_213d695a/`. This is not an
accuracy result and has no matched unaugmented control.

The frozen full run at
`artifacts/liu2024-s-jepa-pretraining-gacl-aug-full/20260719_1226_dd305e35/` completed 58 epoch
records before failing while overwriting `checkpoint_latest.pt`. The best observed validation loss
was 0.0017907 at epoch 56. The epoch-56 `checkpoint_best.pt` and `student_backbone_best.pt` are
loadable, but the run did not complete, did not write final summary metrics, and has not undergone a
matched downstream evaluation. It is therefore an interrupted checkpoint source, not evidence that
augmentation improves MI decoding. Any downstream use must identify it as an interrupted-run
checkpoint and remain restricted to the held-out Lv14 cohort for patient-independent evaluation.

The notebook now writes checkpoints through an atomic temporary-file replacement, includes the
epoch in student-only exports, keeps the first-batch diagnostic update-free, reports sanity-check
amplitudes in volts, and appends an epoch record only after its checkpoints are persisted.

### July 15 gated follow-ups

The full-source Shallow smoke test used prespecified targets 01/03/07. Each transfer checkpoint used
the other 49 patients (1,960 source trials, balanced 980/980), excluded the complete target patient,
and was reused across the target's five folds. Checkpoint signatures now include preprocessing,
model, optimizer, epoch, batch-size, seed, and software-version settings; source normalization hashes
and source inventories are persisted.

| Shallow mode | BA | Collapsed folds | Artifact |
|---|---:|---:|---|
| Matched target-only | 55.00% | 2/15 | `artifacts/liu2024-compact-mi-loso-transfer/20260715_203008_458234_b2061b8a/` |
| 49-source, classifier-only adaptation | 41.67% | 1/15 | `artifacts/liu2024-compact-mi-loso-transfer/20260715_203139_633256_9823c097/` |
| 49-source, low-rate full fine-tuning | 48.33% | 0/15 | `artifacts/liu2024-compact-mi-loso-transfer/20260715_204345_474412_5eec5fac/` |

Both transfer modes lost to target-only on all three subjects, so the exact configuration is stopped
without a 50-target expansion.

The S-JEPA follow-up pins revision `213876ea30f0764fd25c055efcb55d1d1652a371` and matches the
historical 57% preprocessing/configuration. Strict LP trains only `final_layer.*` (2,050 parameters),
asserts every frozen tensor is unchanged, then unfreezes all 16,010 parameters after reinitializing
the optimizer. The first phase necessarily uses the checkpoint encoder behind a frozen random
`spatial_conv`, so it is reported as strict LP-FT rather than as a natural PreLocal adapter warm-up.

| S-JEPA strategy | Full50 BA | Collapsed folds | Predicted class 1 | Artifact |
|---|---:|---:|---:|---|
| Adapter/head only (`new`) | 57.00% | 105/250 | 74.00% | `artifacts/liu2024-sjepa-prelocal/20260715_2103_8f79f840/` |
| Adapter/head warm-up then full FT | 55.65% | 60/250 | 69.85% | `artifacts/liu2024-sjepa-prelocal/20260715_2117_e374556f/` |
| Strict classifier LP then full FT | 55.55% | 61/250 | 69.65% | `artifacts/liu2024-sjepa-prelocal/20260715_2125_6cf8105e/` |

Strict LP-FT minus adapter-only is -1.45 points (95% subject-bootstrap CI -3.70 to +0.65;
Wilcoxon `p=0.3195`; 19/9/22 W/T/L). Strict LP-FT and adapter-warmup full FT are effectively tied
(-0.10 points; CI -0.85 to +0.60; `p=0.4805`; 14/24/12). Reduced collapse and class bias do not
translate into improved balanced accuracy.

Finally, modest online training-only Shallow augmentation was smoke-tested on subjects 01/03/07.
Per-trial/channel SD-scaled Gaussian noise (`p=0.5`, fraction 0.1) reached 52.50% BA, and whole-trial
amplitude scaling (`p=0.5`, interval 0.9-1.1) reached 54.17%, versus 55.00% matched control. All
three had 2/15 collapsed folds. Artifacts are respectively
`artifacts/liu2024-compact-mi-models/20260715_214108_991747_c575e8eb/`,
`artifacts/liu2024-compact-mi-models/20260715_214208_676643_46ef90c8/`, and
`artifacts/liu2024-compact-mi-models/20260715_214010_146912_f58cef10/`. These fixed settings are
stopped without full50 expansion.

The same-fold fusion experiment reuses immutable locked Shallow probabilities and recomputes every
motor13 covariance, tangent reference, scaler, LDA, and training-margin scale on the exact locked
five-fold partitions. It does not consume any repeated-60/40 prediction or transform. Fixed view
scores are averaged before a fixed sigmoid probability map; primary fusion is a prespecified 50/50
probability average.

| Same-fold method | Full50 BA (95% subject-bootstrap CI) | Delta vs locked Shallow | Artifact |
|---|---:|---:|---|
| Locked Shallow | 57.45% (54.25-60.65) | reference | `artifacts/liu2024-compact-mi-models/20260712_165746_790145_fd8ab986/` |
| Riemann 1 s | 55.40% (51.75-59.05) | -2.05 points | `artifacts/liu2024-shallow-riemann-same-fold-fusion/20260715_2158_59cfff7f/` |
| Riemann 2 s | 55.45% (51.60-59.40) | -2.00 points | same |
| Equal short-scale Riemann | 55.95% (52.15-59.70) | -1.50 points | same |
| **Fixed 50/50 fusion** | **57.90% (54.75-61.15)** | **+0.45 points** | same |

Fusion's paired CI is -0.95 to +1.90 points, Wilcoxon `p=0.6635`, and W/T/L is 21/9/20. It is a
small exploratory descriptive change, not evidence of improvement and not a replacement headline.

The corrected source-aligned pilot uses the same views and locked folds. Each target is excluded
from 49 labeled source subjects. Per-view source centers use only each source subject's 40 trials;
target centers use only the target fold's 32 training trials. The prespecified 01/03/07 smoke was
descriptively positive, but the frozen full50 expansion did not confirm it.

| Source-transfer method | Full50 BA (95% CI) | Delta vs target-only (95% paired CI) | p | W/T/L |
|---|---:|---:|---:|---:|
| Target-only | 55.95% (52.15-59.70) | reference | - | - |
| Unaligned source pooling | 54.85% (52.75-57.00) | -1.10 (-5.05 to +2.90) | 0.7548 | 23/3/24 |
| Riemannian source recentering | 56.10% (53.55-58.75) | +0.15 (-3.15 to +3.35) | 0.7188 | 25/3/22 |

Full artifact: `artifacts/liu2024-source-aligned-short-scale-riemann-pilot/20260716_071629_196536_aaebd199/`.
The superseded smoke artifact was removed during the July 20 cleanup; its prespecified summary is
retained here because the smoke-to-full reversal is a warning against promoting favorable
three-patient pilots.

### LOSO pilot

The prespecified pilot used targets 01 and 03 from the four-subject pool 01/03/07/09. Excluding each target leaves only two effective supervised source subjects, not a representative full LOSO source cohort. Values are mean subject BA across the two targets and fold-collapse rate across ten target folds.

| Model | Adaptation | Artifact | BA | Collapsed folds |
|---|---|---|---:|---:|
| FBCNet | Target-only | `artifacts/liu2024-compact-mi-loso-transfer/20260712_172611_514408_ccc1b93a/` | 47.50% | 70% |
| FBCNet | Frozen encoder | `artifacts/liu2024-compact-mi-loso-transfer/20260712_172637_596316_5841224b/` | 51.25% | 80% |
| FBCNet | Full fine-tune | `artifacts/liu2024-compact-mi-loso-transfer/20260712_172649_047295_0d656011/` | 51.25% | 90% |
| EEGTCNet | Target-only | `artifacts/liu2024-compact-mi-loso-transfer/20260712_172710_833967_d74f4d66/` | 50.00% | 100% |
| EEGTCNet | Frozen encoder | `artifacts/liu2024-compact-mi-loso-transfer/20260712_172726_178103_5fd640b6/` | 50.00% | 100% |
| EEGTCNet | Full fine-tune | `artifacts/liu2024-compact-mi-loso-transfer/20260712_172735_615232_9daf3bd5/` | 50.00% | 100% |

This pilot is too small to judge full LOSO transfer. With only two source subjects per target, near-chance BA, and 70-100% collapse, it does not support expansion as currently configured.

### Compact-model comparisons

Entries are row minus column in BA points, with raw two-sided paired Wilcoxon p-values. These 15 exploratory model-to-model tests are not part of the prespecified six-test Holm family.

| Row model | Column model | Delta (95% paired CI), points | Raw p | Win/tie/loss |
|---|---|---:|---:|---:|
| FBCNet | EEGTCNet | +4.25 (+0.60 to +7.90) | 0.0436 | 27/5/18 |
| FBCNet | IFNet | +1.65 (-1.80 to +5.10) | 0.4034 | 24/6/20 |
| FBCNet | FBMSNet | +0.75 (-2.05 to +3.70) | 0.7912 | 22/4/24 |
| FBCNet | EEGNet | +3.80 (-0.10 to +7.75) | 0.1324 | 25/4/21 |
| FBCNet | ShallowFBCSPNet | -3.45 (-6.85 to -0.10) | 0.0581 | 18/3/29 |
| EEGTCNet | IFNet | -2.60 (-5.30 to +0.05) | 0.0508 | 15/8/27 |
| EEGTCNet | FBMSNet | -3.50 (-7.15 to +0.10) | 0.0800 | 17/2/31 |
| EEGTCNet | EEGNet | -0.45 (-1.80 to +0.95) | 0.1975 | 11/19/20 |
| EEGTCNet | ShallowFBCSPNet | -7.70 (-10.85 to -4.55) | 0.000055 | 11/6/33 |
| IFNet | FBMSNet | -0.90 (-4.40 to +2.60) | 0.6300 | 21/3/26 |
| IFNet | EEGNet | +2.15 (-0.75 to +5.15) | 0.2883 | 26/5/19 |
| IFNet | ShallowFBCSPNet | -5.10 (-8.25 to -1.85) | 0.00462 | 16/3/31 |
| FBMSNet | EEGNet | +3.05 (-1.00 to +7.30) | 0.2115 | 27/3/20 |
| FBMSNet | ShallowFBCSPNet | -4.20 (-7.10 to -1.35) | 0.00897 | 15/6/29 |
| EEGNet | ShallowFBCSPNet | -7.25 (-10.80 to -3.80) | 0.000361 | 12/7/31 |

### Robustness and diagnostics

- ShallowFBCSPNet's gain over the classical result is broad, not produced by a few favorable outliers: 31/50 subjects improve, the median paired delta is +4.375 points, and every leave-one-subject-out mean delta remains positive (+1.607 to +2.666 points). Removing 1, 2, 3, or 5 observations from each tail gives mean deltas of +2.201, +2.351, +2.500, and +2.625 points. The five largest gains are subjects 04 (+25.0), 39 (+23.75), 05 (+21.25), 27 (+21.25), and 45 (+20.0); substantial negative outliers also exist (subjects 28 -26.875, 50 -26.25, and 29 -23.125 points). The aggregate advantage remains uncertain because the paired CI crosses zero and Wilcoxon/Holm tests are nonsignificant.
- Collapse explains the near-chance EEGTCNet and EEGNet results. EEGTCNet predicts class 0 on 99.25% of trials (class-0/class-1 sensitivity 99.0%/0.5%); all 236 collapsed folds favor class 0. EEGNet predicts class 0 on 85.70% of trials (85.9%/14.5% sensitivity), and 161/171 collapsed folds favor class 0. ShallowFBCSPNet is balanced (58.8%/56.1% sensitivity), with only six collapsed folds split 3/3 by direction; its non-collapsed fold BA is 57.63%. FBCNet, IFNet, and FBMSNet also have near-balanced aggregate predictions and low-to-moderate collapse (10.0%, 2.8%, and 5.6%).
- All runs contain 50 subjects, 250 folds, and 2,000 predictions. For every subject and model, test indices are unique and exactly `0..39`; all six inventories carry split hash `801ec1d2c981335f`. Thus every trial has exactly one OOF prediction and the model comparisons use identical folds.
- All six `run_metadata.json` files record the same ordered Liu29 montage: `Fp1, Fp2, Fz, F3, F4, F7, F8, FCz, FC3, FC4, FT7, FT8, Cz, C3, C4, T3, T4, CP3, CP4, TP7, TP8, Pz, P3, P4, T5, T6, Oz, O1, O2`. This matches raw indices 0-16 and 18-29, dropping reference CPz at index 17. Code verification in `src/liu2024/modern_mi_common.py` confirms marker channel 32/value 2, independent full-8-s-trial fourth-order 4-40 Hz Butterworth zero-phase filtering, marker-relative 0-4 s cropping, Fourier resampling 500 to 128 Hz, and training-fold-only channel standardization. The classical reference instead uses motor13, average reference, OAS covariance, 8-30 Hz subbands, and full-trial-then-crop processing; preprocessing is internally correct but not identical across model families.

## Evidence And Scope

| Model | Primary source / official implementation | Role here | Data-regime and parameter caveat |
|---|---|---|---|
| FBCNet | Mane et al. (2021), arXiv `2104.01233`; official `https://github.com/ravikiran-mane/FBCNet`; Braindecode `FBCNet` | Primary target-only and LOSO | 12,513 parameters at 29x512; built-in filter bank and max-norm; Braindecode reimplementation is not author-verified |
| EEGTCNet | Ingolfsson et al. (2020), arXiv `2006.00622`; Braindecode `EEGTCNet` | Primary target-only and LOSO | 4,282 parameters; compact EEGNet plus TCN |
| IFNet | Braindecode 1.4 `IFNet` implementation and linked model documentation | Primary target-only | 11,586 parameters; internal filter assumptions must match 128 Hz |
| FBMSNet | Braindecode 1.4 `FBMSNet` implementation and linked model documentation | Primary target-only | 16,932 parameters; filter-bank model, small-sample overfit remains possible |
| EEGNet | Lawhern et al. (2018), DOI `10.1088/1741-2552/aace8c`; Braindecode `EEGNet` | Primary target-only smoke-tested baseline | 2,082 parameters; strongest size/data-regime match, not stroke-specific evidence |
| ShallowFBCSPNet | Schirrmeister et al. (2017), DOI `10.1002/hbm.23730`; Braindecode | Primary target-only | 49,762 parameters; materially larger than available 32 fold-training trials |
| FBLightConvNet | Ma et al. (2023), *IEEE TNSRE*, "A temporal dependency learning CNN with attention mechanism for MI-EEG decoding"; Braindecode adaptation of `https://github.com/Ma-Xinzhi/LightConvNet` | Locked lightweight benchmark | `win_len=256` divides the 512-sample input exactly; the Braindecode adaptation is not author-verified |
| SincShallowNet | Borra, Fantozzi, and Magosso (2020), *Neural Networks* 129:55-74, "Interpretable and lightweight convolutional neural network for EEG decoding" | Locked lightweight benchmark | Learns constrained sinc band-pass filters; evidence is not stroke-specific |
| EEGITNet | Salami, Andreu-Perez, and Gillmeister (2022), *IEEE Access*, DOI `10.1109/ACCESS.2022.3161489` | Locked lightweight benchmark | Explainable inception-TCN architecture; Braindecode notes that its reimplementation is not author-verified |
| ATCNet | Altaheri et al. (2023), DOI `10.1109/TII.2022.3197419`; Braindecode | Optional secondary only | 74,968 parameters at this input; default architecture is adjusted for 512 samples |
| EEGConformer | Song et al. (2023), DOI `10.1109/TNSRE.2022.3230250`; Braindecode | Optional secondary only | 462,786 parameters; unfavorable 32-trial target-only regime |
| EEGNeX | Chen et al. (2024), Braindecode `EEGNeX` | Optional secondary only | 56,162 parameters; no primary status |
| Tensor-CSPNet / TSMNet | Ju and Guan (2022), DOI `10.1109/TNNLS.2022.3151790`; official code should be pinned before use | Literature context only | No neural SPD package is installed in the project environment; not implemented or claimed |
| EEGPT | Wang et al. (2024), official repository `https://github.com/BINE022/EEGPT` | Fail-closed frozen-probe framework | Expected official input documented as 58 channels, 256 Hz, 4 s; no direct stroke evidence |
| CBraMod | Wang et al. (2025), official repository `https://github.com/wjq-learning/CBraMod` | Fail-closed frozen-probe framework | 200 Hz patch input; no direct stroke evidence |

Repository URLs identify upstream projects but are not dependency pins. An experiment is valid only when its config records a local repository path, immutable revision, checkpoint path, full SHA256 digest, import module, factory, and feature method.

## Exact Protocols

`src/liu2024/liu2024_compact_mi_models.ipynb` runs the six primary Braindecode 1.4 models one model per run. Each original 8 s trial is band-pass filtered independently, then cropped 0-4 s relative to its own marker, selected to Liu29 or motor13, and resampled to 128 Hz. Fold normalization is fit on the 32 outer-training trials. A deterministic persisted five-fold split gives each of 40 trials exactly one OOF prediction. AdamW, batch size 8, gradient clipping, a fixed source-locked epoch budget, and last-epoch weights avoid outer-test checkpoint selection. Fixed seeds are probability-averaged, never selected.

`src/liu2024/liu2024_compact_mi_loso_transfer.ipynb` excludes target subject `s` from all supervised source training. Its source-only checkpoint is selected without target data and cached under a source-set/config signature. The same compatible checkpoint is reused across the target's five folds. Target-only initialization, frozen encoder plus identified classifier head, and low-rate full fine-tuning are separate modes. Target normalization and BatchNorm adaptation see outer-training data only. The completed pilot fixes targets 01 and 03 within source pool 01/03/07/09; expansion is not supported under the current setup.

`src/liu2024/liu2024_foundation_model_probes.ipynb` defaults to `validate_only`. Missing assets, incomplete provenance, digest mismatch, absent explicit channel mapping/interpolation, incompatible official input, import failure, state-dict mismatch, or missing feature API all stop execution. It does not auto-download. Frozen mean-pooled embeddings may later feed fold-local shrinkage LDA or logistic regression. A low-rank/spatial adapter is permitted only around validated frozen token features; there is no guessed internal adapter API.

`src/liu2024/liu2024_lightweight_mi_models.ipynb` uses the exact completed compact-benchmark Liu29 preprocessing, fold-local normalization, five-fold partitions, and split hash `801ec1d2c981335f`. FBLightConvNet, SincShallowNet, and EEGITNet are fixed at 20 epochs and seed 2026 for direct comparison. EEGInceptionMI, MSVTNet, and CTNet are reduced optional models and require explicit enablement; they are not in the focused locked sweep. The notebook logs parameter counts, losses, collapse, probabilities, original-trial predictions, subject statistics, and an exploratory paired comparison with the completed ShallowFBCSP run.

`src/liu2024/liu2024_cropped_shallow_fusion.ipynb` is exploratory development, not independent confirmation. It splits original trials before constructing five fixed 2 s crops at starts 0, 0.5, 1, 1.5, and 2 s, weights each original training trial equally through exactly five crops, and averages logits to produce one held-out prediction per original trial. Strict TTA uses only that fixed crop set with no confidence selection or BatchNorm updates. A `final_conv_length=1` dense path is available, but fixed raw crops are primary because trial ownership is explicit. This follows the cropped-training and dense-prediction framework of Schirrmeister et al. (2017), DOI `10.1002/hbm.23730`, while using a stricter original-trial evaluation unit.

Riemannian score fusion is currently disabled by a fail-closed guard. It must not consume the existing repeated-60/40 artifact. Enabling it requires recomputing motor13 OAS/tangent 1 s and 2 s scores on these identical five-fold outer partitions, fitting all transforms on outer-training trials, and calibrating each branch through training-only cross-fitting before fixed equal fusion.

### Fair comparison with Lv et al.'s 75%

Lv et al.'s reported 75% KNN result is not a 40-trial-per-patient, exactly-once OOF decoding result comparable to these notebooks. It is based on aggregate condition-level microstate records from a selectively retained 14-patient cohort after undisclosed or unaudited quality screening. The modern notebooks classify individual original MI trials over all 50 Liu subjects (and report any Lv-14 restriction separately). Therefore 75% is useful literature context, not a like-for-like target or evidence that trial-level stroke MI decoding should reach 75%. The current attempted Lv reproduction did not recover that number, and unresolved manual ICA/QC, segment assignment, smoothing, and fold details prevent an exact protocol match.

In particular, neither the locked 57.45% Shallow result nor the exploratory 57.90% longer-training result can be described as approaching a comparable 75% endpoint: the cohort, prediction unit, screening, and validation protocol differ materially.

## Leakage Checklist

- Filter every complete trial independently; never concatenate trials before filtering.
- Crop only after trial-level filtering and relative to that trial's marker.
- Persist identical deterministic outer splits across compact models.
- Produce exactly one pooled OOF prediction for every original trial.
- Fit normalization, alignment, feature transforms, and probes on outer-training data only.
- Never select epochs, checkpoints, hyperparameters, models, seeds, windows, or bands using outer-test results.
- Average prespecified seeds; do not report the best seed.
- In LOSO, exclude the target from source training, source checkpoint selection, and source normalization.
- Do not expose target-test samples to BatchNorm updates.
- Refuse checkpoint-cache metadata mismatches.
- Report subject-level balanced accuracy and subject-bootstrap intervals; use paired subject tests, not folds as independent samples.
- Apply Holm correction across the six primary compact-model comparisons after all six subject tables exist.

## Commands

```bash
./.venv/bin/python src/utils/experiments/experiments.py \
  --notebook src/liu2024/liu2024_compact_mi_models.ipynb \
  --configs src/utils/experiments/configs/sweep_compact_mi_models_locked.json \
  --kernel-name eeg_jepa --daemon

./.venv/bin/python src/utils/experiments/experiments.py \
  --notebook src/liu2024/liu2024_compact_mi_loso_transfer.ipynb \
  --configs src/utils/experiments/configs/sweep_compact_mi_loso_transfer_pilot.json \
  --kernel-name eeg_jepa --daemon

./.venv/bin/python src/utils/experiments/experiments.py \
  --notebook src/liu2024/liu2024_foundation_model_probes.ipynb \
  --configs src/utils/experiments/configs/sweep_foundation_model_probes_validate.json \
  --kernel-name eeg_jepa

./.venv/bin/python src/utils/experiments/experiments.py \
  --notebook src/liu2024/liu2024_lightweight_mi_models.ipynb \
  --configs src/utils/experiments/configs/sweep_lightweight_mi_models_locked.json \
  --kernel-name eeg_jepa --daemon

./.venv/bin/python src/utils/experiments/experiments.py \
  --notebook src/liu2024/liu2024_cropped_shallow_fusion.ipynb \
  --configs src/utils/experiments/configs/sweep_cropped_shallow_development.json \
  --kernel-name eeg_jepa --daemon
```

The foundation validation sweep is expected to fail until each entry is overridden with real pinned assets and explicit mapping. That refusal is the intended current result, not a failed model experiment.
