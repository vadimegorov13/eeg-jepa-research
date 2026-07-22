# Liu2024 S-JEPA PreLocal on the SSL-Held-Out Lv14

## Purpose

`src/liu2024/liu2024_sjepa_prelocal_lv14_ssl_holdout.ipynb` evaluates the local feature encoder from
the corrected Liu2024 SSL run on the 14 patients excluded from SSL training and validation.

This is within-subject calibrated decoding on SSL-unseen patients: each outer fold trains the
PreLocal spatial adapter and classifier on 32 trials from the target patient and tests the other
eight exactly once. It is not zero-shot patient transfer.

## After SSL Pretraining Completes

Locate the completed export:

`artifacts/liu2024-s-jepa-pretraining-gacl-aug-full/<run-id>/student_backbone_best.pt`

Compute its digest:

```bash
shasum -a 256 artifacts/liu2024-s-jepa-pretraining-gacl-aug-full/<run-id>/student_backbone_best.pt
```

Edit only these fields in
`src/utils/experiments/configs/sweep_sjepa_prelocal_lv14_ssl_holdout_template.json`:

```json
"enabled": true,
"pretrained_checkpoint_path": "/absolute/path/to/student_backbone_best.pt",
"pretrained_checkpoint_sha256": "<64-character SHA256>"
```

Do not change the cohort, split, preprocessing, or downstream hyperparameters after inspecting Lv14
outcomes.

## Launch

From the `EEG_JEPA/` repository root:

```bash
python src/utils/experiments/experiments.py \
  --notebook src/liu2024/liu2024_sjepa_prelocal_lv14_ssl_holdout.ipynb \
  --configs src/utils/experiments/configs/sweep_sjepa_prelocal_lv14_ssl_holdout_template.json \
  --kernel-name eeg-jepa \
  --daemon
```

Use the kernel that imports this repository's `mne`, `torch`, and `braindecode` environment. The
notebook fails before loading EEG outcomes if the checkpoint digest, epoch, SSL subject split,
channel order, sampling rate, window length, FIR preprocessing, units, or tensor finiteness differs
from the locked contract.

## Expected Completion

- 14 subjects.
- 70 outer folds.
- 560 exactly-once original-trial predictions.
- Seven `feature_encoder.*` keys loaded into PreLocal.
- Only `spatial_conv.*` and `final_layer.*` trainable.
- Full checkpoint and ignored contextual-key provenance saved at run level.
- Patient-bootstrap confidence interval, subject-level Wilcoxon test versus chance, and collapse
  diagnostics saved in `global_metrics.json`.
