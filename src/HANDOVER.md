# Project Handover — Predictive Representation Learning for Stroke MI Decoding (Liu2024 + S-JEPA)

> Purpose of this file: bring a fresh Claude (or collaborator) fully up to speed so they can continue
> without re-reading the whole history. Read top to bottom once; it is ordered from "what the project is"
> to "what to do next." Attach the files listed in §11 to the new chat.

---

## 1. Project in one paragraph

Thesis direction: **Learning Intentional State Representations from EEG Using Predictive World Models.**
S-JEPA (Signal-JEPA, a joint-embedding predictive architecture for EEG) is the *methodological backbone*,
not the whole topic. The application target is **motor-imagery (MI) intention decoding on the Liu2024 acute
stroke EEG dataset** (left-hand vs right-hand imagined movement). The open question the advisor (me) and the
user have been working: *does S-JEPA-style predictive latent learning actually help decode MI intention on
this very small, clinically hard dataset, or should S-JEPA be demoted to a representation/regularization
branch inside a hybrid with classical Riemannian methods?*

Current honest answer: S-JEPA transfers **weakly** to Liu2024 (best ~56% balanced accuracy). The plan is to
build strong baselines first, then a hybrid. The likely thesis conclusion is **not** "S-JEPA fixes stroke MI"
— it is "predictive representations add modest value as one branch of a Riemannian-anchored hybrid."

---

## 2. Source material (5 papers, already digested)

| Short name | What it is | What we take from it |
|---|---|---|
| **S-JEPA** (Guetschel et al. 2024, arXiv:2403.11772) | Signal-JEPA: channel-wise local CNN encoder + transformer contextual encoder, spatial **block masking**, EMA target encoder, L1 latent loss. Three downstream heads: contextual / post-local / **pre-local**. | The **pre-local** head (spatial filtering *before* the local encoder, contextual encoder discarded) is best for BCI. Long pretraining windows (16 s) help. Contextual attention needs lots of data and fails when calibration data is small. SOTA on Lee2019-MI is **Riemannian**, not S-JEPA. |
| **Liu2024** (Liu et al., *Scientific Data* 11:131, 2024; doi 10.1038/s41597-023-02787-8) | The dataset paper: 50 acute stroke patients, L/R-hand MI. | Dataset facts (§4). Their baselines (§5). Documented warnings: motion artifacts in 13 subjects, non-stationarity / covariance shift, only ~42–46% of subjects show clean contralateral C3/C4 ERD. |
| **Pérez-Velasco et al. 2024** (Graz BCI Conf.) | Inter-task transfer ME→MI with EEGSym on 109 healthy Physionet subjects. | ME→MI transfer works **in healthy subjects** (~85%). Liu2024 has **no ME data**, so this is not directly usable — but motivates "use large healthy MI as a prior." |
| **Kenneweg et al. 2025** (JEPA for RL, arXiv:2504.16591) | JEPA adapted to RL; **model collapse** analysis + variance regularization (VICReg-style). | Conceptual: JEPA collapse is a real failure mode; batch-variance regularization and task-gradient propagation prevent it. Relevant to why some S-JEPA folds collapse to one class. |
| (the user's two **pilot reports**) | `Report__SJEPA_Transfer_to_Liu2024.pdf` and `liu2024_sjepa_stroke_mi_report.pdf` | The actual S-JEPA-on-Liu2024 results so far (§5). |

---

## 3. The big advisory analysis already delivered (summary so you don't redo it)

A full deep-dive was produced earlier in the conversation. Do **not** rewrite it; just build on it. Its conclusions:

**Why S-JEPA underperforms on Liu2024 (diagnosis):**
- 40 trials/subject → ~24–32 training trials per fold. Far too little for fine-tuning a transformer-scale model.
- Domain gap: pretrained on **healthy** EEG (Lee2019, 128 Hz); applied to **acute stroke** cortical dynamics.
- **Lesion-side confound:** 27/50 left-hemiplegic, 23/50 right-hemiplegic. Imagining the affected vs unaffected
  hand produces qualitatively different neural patterns → a latent class×lesion confound. Likely source of the
  observed **left-hand prediction bias**.
- Non-stationarity + covariance shift (the dataset paper says so explicitly).
- Contextual attention learns nothing on so few trials → **pre-local spatial filtering matters more than attention**.
- Full fine-tuning is unstable (8% collapse rate); freezing the encoder and adapting only the spatial projection is more stable.

**Recommended final method — PRISM** (*Predictive-Riemannian Integration for Stroke Motor-imagery decoding*):
late/feature fusion of (Branch A) frozen S-JEPA pre-local embeddings + (Branch B) filter-bank Riemannian
tangent-space features, with a small **shrinkage-LDA** classifier on top, Euclidean Alignment preprocessing,
and strict per-fold leakage control. S-JEPA is the *predictive-representation* branch; Riemannian geometry
carries the classification. **Build baselines before the hybrid.**

**Augmentation guidance:** Safe = Gaussian noise (~5% per-channel std), temporal jitter (±50 ms),
channel dropout, amplitude scaling, **Riemannian geodesic interpolation** between same-class covariances
(most principled), tangent-space mixup. Risky / label-as-exploratory = frequency perturbation, GAN/VAE/diffusion
synthetic trials (NOT a primary result — cannot train a generator on 32 trials). Apply augmentation **inside
training folds only.**

---

## 4. Dataset facts (Liu2024 source `.mat`)

- 50 subjects; **40 trials each**; binary labels **1 = left-hand MI, 2 = right-hand MI** (balanced).
- `rawdata` shape per subject: **(40 trials × 33 channels × 4000 samples)**, 500 Hz, 8-second trials.
- 33 channels = 30 EEG + 2 EOG (ch 31 HEOL, 32 VEOR) + 1 marker (ch 33). EEG channel **18 = CPz = reference**.
- **Keep 29 EEG channels**: drop CPz (index 17 zero-based / channel 18 one-based), drop the 2 EOG, drop marker.
- MI analysis window: **0–4 s** after MI onset (unless a notebook is specifically sweeping windows).
- Downsample target for S-JEPA/EEGNet: **128 Hz**; bandpass **0.5–40 Hz** (broadband) or 8–30 Hz (MI band) depending on method.

**Channel-order note to verify:** the dataset paper's Table 2 prints "FT7" twice (a likely typo). The notebooks
correct index 20 to **TP7**. For Riemannian methods a consistent channel permutation does not change accuracy
(covariance is permutation-equivariant), but for **S-JEPA (uses 3-D channel coordinates) and EEGNet, channel
identity matters** — so confirm the montage names against the dataset's `channel_location` files before trusting
topographies / spatial maps.

---

## 5. Numbers to anchor against

**Liu2024 paper, Table 4** (their protocol: 60/40 split, 10-fold, per subject):

| Method | Avg accuracy | Kappa |
|---|---|---|
| CSP + LDA | 55.57% | 0.111 |
| FBCSP + SVM | 57.57% | 0.151 |
| TSLDA + DGFMDRM | 61.20% | 0.224 |
| **TWFB + DGFMDRM** (their best) | **72.21%** | 0.444 |

**⚠ Critical caveat on the 72.21% (see §7):** the provided MATLAB selects the best frequency band using
**test-set accuracy** (oracle selection), with a different random split per band. This almost certainly
**inflates** the number. A leakage-safe reproduction (inner-CV band selection) will likely land **below 72%**,
plausibly in the ~60–68% range. Do not treat 72.21% as a fair target for an honest pipeline; treat the honest
Riemannian result you reproduce as the real ceiling.

**User's S-JEPA pilot results** (their protocol: 5-fold per subject, 32 train / 8 test; balanced accuracy):

| Run | Bal. acc | Collapse rate | Notes |
|---|---|---|---|
| full + random | 52.55% | 8.0% | strong left-hand bias |
| full + pretrained | 54.60% | 2.8% | left bias reduced |
| new + random | 53.20% | 1.6% | moderate left bias |
| **new + pretrained** | **56.05%** | 2.0% | **best & most balanced** (only run with right recall > 50%) |

User's own CSP/FBCSP reproduction was weak: **CSP+LDA 50.6%, FBCSP+SVM 51.2%** (below Liu's targets — treat as
sanity baseline, not a faithful reproduction).

Takeaway: best S-JEPA (56.05%) ≈ Liu's CSP+LDA target, below FBCSP, far below TWFB. The gain over random-init
controls (~+2.85 pp) is the real "S-JEPA adds something" signal — modest but present.

---

## 6. What was done in THIS session (file inspection + MATLAB reverse-engineering)

This session did **not** produce new notebooks. It (a) reverse-engineered the MATLAB, (b) discovered two of the
three requested notebooks already exist as drafts, and (c) mapped the environment. Details below.

---

## 7. MATLAB reverse-engineering — `TWFB_DGFMDM.m` (the most important technical findings)

The provided `TWFB_DGFMDM.m` (+ helpers `NotchFilter.m`, `filter_param.m`) is what Liu2024 *actually ran*, and it
**differs from the paper's prose**. Faithfully matching the MATLAB matters more than matching the paper text.

1. **Channels:** `channel = [1:17 19:30]` → 29 channels, drops channel 18 = CPz. (= drop 0-based index 17 in Python.) ✓ consistent with notebooks.
2. **Trigger / onset:** `trigger = find(MIEEGData(:,33)==2)` — marker channel (col 33), value **2 marks MI onset (t=0)**.
   The MATLAB treats `MIEEGData` as a **2-D continuous (time × 33)** array, NOT the (40×33×4000) trial tensor the
   notebooks load. **Action item:** confirm where t=0 sits inside each 4000-sample trial in the segmented `.mat`
   (the notebooks currently assume a fixed sample offset for t=0 — verify this against the marker channel).
3. **Window:** `MIEEGData(trig-800 : trig-800+2799, …)` = 2800 samples; after filtering keeps `BandpassData(801:2800,:)`
   = **2000 samples = 4 s @ 500 Hz**. The first 800 samples (1.6 s) are a **pre-trigger filter warm-up buffer**, then
   discarded. Net MI window = **0–4 s after onset.** (Good practice — replicate the warm-up-then-trim idea to avoid edge artifacts.)
4. **Frequency bands — DIFFERENT FROM THE PAPER:** MATLAB uses **8 bands**:
   `{[8,12],[8,20],[8,30],[12,20],[15,20],[15,30],[20,30],[8,15]}`.
   The paper (and the user's task prompt) lists **19 overlapping 4-Hz bands** (8–12 … 26–30). They do **not** match.
   The provided MATLAB is a **reduced 8-band version**. The pyRiemann notebook already documents this correctly.
5. **No time-window search in the MATLAB.** The window is fixed at 0–4 s. The paper's "7 time windows + backtracking
   search" is **not implemented** in the provided `.m`. So "TWFB" as actually shipped ≈ filter-band selection + DGFMDM,
   *without* the time-window part.
6. **Covariance:** `COV = SS' * SS` where `SS` is (2000 samples × 29 chans) → 29×29 **scatter matrix**, not divided by N,
   not mean-centered. Fine for affine-invariant Riemannian geometry (globally scale-invariant; bandpass removes DC).
   In pyRiemann use `Covariances(estimator='scm')` (or 'oas' for stability) — add ε·I regularization for SPD safety.
7. **Classifier = `fgmdm(...)` = Fisher Geodesic MDM** = discriminant geodesic filtering (FGDA) + MDM. This **is**
   the paper's "DGFMDRM." **Direct pyRiemann equivalent: `pyriemann.classification.FgMDM(metric='riemann')`.** ← key mapping.
8. **🚩 Leakage in the original MATLAB:** `[acc0_value, acc0_index] = max(acc_temp)` picks the band with the highest
   **test** accuracy, and `randperm` makes each band use a **different** 24/16 split. This is oracle band selection +
   luckiest-split selection → inflates the reported 72.21%. **Do not reproduce this leakage.** Use inner-CV selection.
9. **Helpers:** `NotchFilter.m` = 50 Hz IIR notch (`iirnotch`, Q=6) applied **before** bandpass. `filter_param.m` =
   Butterworth via one-pass `filter` (has phase lag). The Python notebook uses zero-phase `filtfilt` and (check) may
   omit the notch. Minor faithfulness deltas to note: consider adding a 50 Hz notch for a "matlab_faithful_attempt" mode.

---

## 8. State of the three notebooks

The user asked for three notebooks. **Two already exist** (drafted in a prior session, structurally complete);
**one is missing.** Run order recommended by the user: **TWFB/pyRiemann → S-JEPA+LDA → EEGNet.**

### 8a. `liu2024_twfb_dgfmrdm_pyriemann.ipynb` — EXISTS (25 cells), looks complete
- Kernel `python3` (3.10). Imports pyRiemann (`MDM`, `TangentSpace`, presumably `FgMDM`), scipy signal, sklearn.
- 4 modes via `CONFIG["mode"]`: `broad_8_30_mdm`, `filterbank_tangent_lda`, `twfb_inner_selection`, `matlab_faithful_attempt`.
- Correctly documents the MATLAB 8-band vs paper 19-band discrepancy and the channel mapping.
- Leakage-safe: bandpass is a fixed transform; TangentSpace ref point + covariance means fit on **train only**;
  inner band/window selection uses **train only**.
- **To improve for faithfulness:** make `matlab_faithful_attempt` actually use `FgMDM` (not just MDM) + the 8 MATLAB
  bands + 50 Hz notch; ensure band selection there is honest (inner CV), and add a clearly-labeled "oracle (leaky,
  paper-style)" diagnostic column purely to show how much the 72.21% is inflated — never as a reported result.

### 8b. `liu2024_sjepa_embeddings_lda.ipynb` — EXISTS (26 cells), looks complete BUT has a key limitation
- Kernel `eeg-jepa-1` (3.11). Uses **braindecode** `SignalJEPA_PreLocal` from HuggingFace
  `braindecode/signal-jepa_without-chans`. Freezes local encoder; fine-tunes only `spatial_conv` per fold (the
  "new" strategy); then trains shrinkage-LDA / L2-logistic on extracted features. PCA fit on train only.
- **⚠ Embedding limitation (flagged in the notebook's own cell 13):** `SignalJEPA_PreLocal` outputs only **2-D logits**
  by default, so the current "embedding" is essentially the 2-class decision score, **not** a rich representation.
  LDA on a 2-D feature is weak and partly defeats the purpose. **Top priority fix:** add a forward hook to capture a
  richer embedding — the `spatial_conv` output or the **mean-pooled local-encoder token sequence** (e.g. C×t×d → pool
  over time → per-channel d, or pool to a fixed vector). This is what makes the S-JEPA branch meaningfully different
  from the existing fine-tuned classifier, and it is exactly the feature Branch A of PRISM needs.
- `save_embeddings=True` already intends to cache per-trial embeddings for later reuse by PRISM. Keep that.

### 8c. `liu2024_eegnet_baseline.ipynb` — **DOES NOT EXIST YET (the missing third notebook)**
- Needs: a clean PyTorch EEGNet implemented in-notebook, kept small (40 trials/subject), input-shape asserts,
  early stopping on an **internal** train-fold split only, class-balanced metrics, optional **disabled-by-default**
  augmentation (Gaussian noise / tiny temporal jitter / channel dropout), per-fold training-curve logging.
- Same evaluation harness, CONFIG style, artifact outputs, and leakage controls as the other two.
- Suggested CONFIG was specified in the user's task prompt (reuse it). Default `augmentation.enabled = False`.

### Common output contract for all three (keep consistent)
`fold_results.csv`, `subject_summary.csv`, `global_summary.json`, `config.json`, `confusion_matrix.png`,
`subject_accuracy_plot.png`, plus notebook-specific artifacts (EEGNet training logs; cached S-JEPA embeddings;
TWFB selected bands/windows). Artifact folders: `artifacts/<notebook_name>/`.

### Common evaluation harness (already used by the two existing notebooks — match it in EEGNet)
Per subject, `StratifiedShuffleSplit(n_splits=n_repeats, test_size=0.40, random_state=2026)`
(≈ 24 train / 16 test, matching Liu2024's 60/40). Metrics: accuracy, **balanced accuracy**, left recall, right recall,
confusion matrix, **collapse flag** (`max class share > 0.95`), predicted-class distribution, subject mean/std,
global mean/std. Aggregate a global confusion matrix across all folds.

---

## 9. Environment constraints (this sandbox)

- Present: **numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0**.
- **Missing: torch, pyriemann, mne, nbformat.** Network is **disabled** → cannot `pip install`.
- Consequence: the existing notebooks **cannot be executed here**; they target the user's local kernels
  (`eeg-jepa-1`, `python3`) where braindecode/torch/pyriemann/mne are installed. **The user must run them locally.**
- For a fresh Claude in the same sandbox: you can author/edit `.ipynb` JSON directly (use the `json` module, not
  `nbformat`), and you can unit-test pure-numpy/scipy/sklearn logic (bandpass, covariance, windowing, splits,
  shrinkage LDA). You **cannot** import torch/pyriemann to smoke-test those parts — flag that in a notebook markdown cell.

---

## 10. Recommended next steps (in order)

1. **Finish & run the TWFB pyRiemann notebook first.** Establish the *honest* Riemannian ceiling on Liu2024.
   Compare `filterbank_tangent_lda` vs `twfb_inner_selection` vs `matlab_faithful_attempt` (with `FgMDM` + 8 bands +
   notch). Report the leaky-oracle number only as a "how inflated is 72.21%?" diagnostic, never as a result.
2. **Fix the S-JEPA embedding hook** (richer embedding, not 2-D logits), then run S-JEPA-embeddings + shrinkage LDA.
   Cache embeddings (`save_embeddings`) for PRISM.
3. **Create and run the EEGNet baseline** (the missing notebook), augmentation off by default.
4. With all three baselines + the honest Riemannian ceiling in hand, **build PRISM** (Branch A frozen S-JEPA embeddings
   + Branch B filter-bank Riemannian tangent-space, feature- or late-fusion, shrinkage LDA, Euclidean Alignment).
   Then ablations (remove A; remove B; random vs pretrained S-JEPA inside the hybrid; ±augmentation; ±EA).
5. Statistical comparison across methods: **Wilcoxon signed-rank on per-subject balanced accuracy** + effect sizes
   (Holm–Bonferroni corrected). Per-subject accuracy vs NIHSS as a secondary clinical analysis.

Do **not** build PRISM before step 1 — the hybrid is only meaningful once the Riemannian-only number is known.

---

## 11. Leakage rules (must be preserved in every notebook)

Inside each train/test split: fit scalers, PCA, covariance means, tangent-space projector, S-JEPA `spatial_conv`,
and the classifier on **training trials only**; select TWFB bands/windows via **inner CV on training only**; never
fit on or select using the test fold. Per-trial normalization (z-score within a single trial) is fine before
splitting; any **across-trial** statistic must be computed post-split on train only. Bandpass/notch are fixed
transforms (no learned params) and are safe to apply before splitting.

---

## 12. Open questions / assumptions to verify before trusting full runs

1. **t=0 sample offset** inside each segmented 4000-sample trial (reconcile the notebooks' fixed offset with the
   MATLAB's marker-based onset). Wrong offset → wrong 0–4 s window → silently degraded results.
2. **Channel name/order** (TP7 vs the paper's duplicated "FT7") — matters for S-JEPA coordinates & EEGNet, not for Riemannian.
3. **S-JEPA checkpoint** loads cleanly from `braindecode/signal-jepa_without-chans` (or the user's local checkpoint),
   and the pre-local `n_times`/window-sample expectation matches the 0–4 s @128 Hz input.
4. **Whether to exclude the 13 artifact subjects** (Liu2024 lists subjects 4,5,13,14,18,24,28,33,42,43,47,48,49) —
   run primary on all 50 and secondary on the clean 37; report both.
5. **Notch filter** inclusion for the "matlab_faithful_attempt" mode (the MATLAB applies 50 Hz notch; the Python may not).
6. **Reproduction gap**: even an honest TWFB likely < 72.21% because of the MATLAB's oracle band selection — set
   expectations accordingly in the write-up.

---

## 13. File inventory (attach these to the new chat)

| File | Role |
|---|---|
| `liu2024_source_mat_sjepa_prelocal_augmented.ipynb` | The user's main existing S-JEPA downstream notebook (data loading / style / metrics reference; ~4.5 MB with outputs). |
| `liu2024_twfb_dgfmrdm_pyriemann.ipynb` | **Existing draft** — TWFB/DGFMDM pyRiemann reproduction (25 cells). Needs faithfulness polish per §8a. |
| `liu2024_sjepa_embeddings_lda.ipynb` | **Existing draft** — S-JEPA embeddings + shrinkage LDA (26 cells). Needs the richer-embedding hook per §8b. |
| `liu2024_eegnet_baseline.ipynb` | **MISSING** — to be created (§8c). |
| `TWFB_DGFMDM.m` | The real Liu2024 TWFB/DGFMDM implementation (reverse-engineered in §7). |
| `NotchFilter.m`, `filter_param.m` | MATLAB filtering helpers (50 Hz notch; Butterworth one-pass). |
| The 5 PDFs (S-JEPA, Liu2024 dataset, Pérez-Velasco, Kenneweg JEPA-for-RL, + 2 pilot reports) | Grounding/context — already digested; do not re-summarize at length. |

---

*End of handover. The single most load-bearing facts: (1) `fgmdm` == pyRiemann `FgMDM`; (2) the MATLAB uses 8 bands +
no time-window search + oracle test-set band selection, so 72.21% is likely inflated and the honest Riemannian ceiling
is the real target; (3) the S-JEPA embeddings notebook currently uses 2-D logits as "embeddings" and needs a richer
hook; (4) EEGNet notebook still needs to be built; (5) S-JEPA should be framed as a representation branch in a hybrid
(PRISM), not the primary classifier.*
