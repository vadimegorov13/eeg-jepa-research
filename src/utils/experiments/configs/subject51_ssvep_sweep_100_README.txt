Subject 51 SSVEP fresh 100-run sweep

Format
- JSON array of config overrides
- Each entry includes only settings that change, plus:
  - seed
  - subjects_to_use
  - paradigm

Assumed notebook defaults
- sfreq = 128
- bandpass_low = 0.5
- bandpass_high = 40.0
- convert_to_uV = True
- normalize = False
- normalization_method = None
- learning_rate = 1e-3
- batch_size = 16
- cv_folds = 5
- val_split = 0.2
- n_epochs = 5000
- early_stopping_patience = 50

Run groups
1-20
  Baseline seed sweep at notebook defaults.
21-44
  Focused learning-rate x batch-size grid around the strongest region.
45-64
  Normalization / scaling sweep:
  methods in {none, robust, exponential_moving, exponential_moving_slow, zscore}
  convert_to_uV in {True, False}
  batch_size in {8, 24}
65-80
  Bandpass sweep around the paper-aligned default without repeating 0.5-40.0.
81-92
  Validation split / early stopping sweep.
93-100
  Integrated best-of-best combos.

Judgment from prior runs
- Kept strong emphasis on no normalization, robust, and exponential_moving.
- Did not include demean because prior runs looked weak.
- Kept sfreq fixed at 128.
- Focused LR search around 1e-3 to 2.5e-3 and batch sizes around 8 to 32.
- Included only a limited amount of more aggressive bandpass exploration.
