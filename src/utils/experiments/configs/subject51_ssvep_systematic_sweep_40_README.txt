Subject 51 SSVEP systematic sweep (40 runs)

Run groups:
1-10   baseline seed stability
11-16  learning-rate sweep
17-22  batch-size sweep
23-32  normalization-method sweep
33-36  convert_to_uV ablation
37-40  bandpass sweep

Notes:
- This file is intentionally compact.
- It does not repeat notebook defaults such as pretrained_url, device, cv_folds, val_split,
  n_epochs, early_stopping_patience, sfreq, or batch_size unless a run explicitly changes them.
- All runs target only subject 51 and only the SSVEP paradigm.
