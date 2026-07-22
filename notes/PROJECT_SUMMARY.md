# Project Summary — Learning Intentional State Representations from EEG Using Predictive World Models

> **Historical summary:** this document predates several completed July 2026 experiments. Its “next
> steps” are not current run instructions. Check the workspace-root `AGENTS.md` §2e closed-experiment
> registry before proposing or launching work.

*Working summary of everything done so far. Self-contained handover: context, methods, the central
leakage finding, all key numbers, deliverables, and next steps.*

---

## 1. Thesis and central question

**Thesis:** *Learning Intentional State Representations from EEG Using Predictive World Models.*
S-JEPA (Signal-JEPA) is the methodological backbone — **not** the whole topic. Application: left- vs
right-hand **motor-imagery (MI)** decoding on the **Liu2024 acute-stroke EEG dataset**.

**Original question:** does S-JEPA-style predictive latent learning help on stroke MI, or should it be
demoted to a representation/regularization branch beside classical Riemannian methods?

**How the question evolved:** the honest reproduction of the classical "state-of-the-art" (TWFB-DGFMDM)
turned out to perform *at chance*. That reframed the project from "can S-JEPA beat a strong Riemannian
baseline" to "**there is no honest strong Riemannian baseline — the published number is a leakage artifact,
and S-JEPA is competitive with classical methods once both are evaluated fairly.**"

---

## 2. Dataset (Liu2024)

- 50 stroke subjects; **40 trials each** (binary, balanced: left=0, right=1). Raw `(40, 33, 4000)`, 500 Hz, 8-s trials.
- 33 channels = 30 EEG + 2 EOG + 1 marker (col index 32, 0-based; value **2 marks MI onset ≈ 2.0 s** into the trial).
- Keep **29 EEG channels** (drop CPz, 0-based index 17, the source reference).
- MI analysis window = **0–4 s after onset**.
- Known difficulty: only ~42–46% of subjects show clean contralateral C3/C4 ERD; 13 artifact-affected subjects;
  a class×lesion confound (27/50 left-hemiplegic, 23/50 right) likely drives a left-hand prediction bias.

---

## 3. What we did (chronological)

1. **S-JEPA pilot diagnosis.** Early S-JEPA runs hovered near chance with frequent prediction collapse; best
   pilot was *new + pretrained* PreLocal at ~56%. Classical CSP/FBCSP reproductions were weak (~50–51%).
2. **Reverse-engineered the published MATLAB code** (`code.zip`, from the authors' figshare = **reference #40**).
   Found the shipped `TWFB_DGFMDM.m` does **not** implement the paper's described method: it uses 8 hand-picked
   bands (not 19), no time-window search, no LTSA, and **selects the filter band by accuracy on the test split**
   (data leakage). The sister `TSLDA_DGFMDM.m` similarly picks the better classifier on test.
3. **Established code ≠ paper.** The paper's Table 3 describes a fuller method: 7 time windows × 19 overlapping
   4-Hz bands, a "backtracking search" to select window+band, an **LTSA** dimensionality-reduction step, then
   discriminant geodesic filtering + Riemann minimum distance (DGFMDM). The implementation that produced the
   published 72.21% is **not** in the public release.
4. **Two explanatory PDFs** written (shipped code; and the paper's Table-3 method), in accessible language.
5. **Fixed the S-JEPA embeddings notebook** (rich forward-hook embeddings instead of 2-D logits; channel/trial
   reshape bug fixed; deterministic frozen-embedding cache).
6. **Built two hybrid notebooks + sweeps** (Notebook A: embedding probe + feature fusion; Notebook B: honest
   TWFB-DGFMDM × S-JEPA late fusion).
7. **Ran the Notebook A sweep** (9 experiments) — S-JEPA probe ≈ 54%, fusion null (see §5).
8. **Ran an honest TWFB reproduction** (`liu2024_twfb_dgfmdm_timewindow_faithful`) with leaky vs honest selection
   — the pivotal result (see §4).
9. **Built a paper-faithful TWFB notebook** (19 bands × 7 windows + **real LTSA**, leaky/honest/fixed) to confirm
   the leakage finding is not an artifact of the simplified 8-band code.

---

## 4. The central finding — the published 72.21% is a selection-leakage artifact

Running the **same** time-window × filter-bank pipeline and changing only **how the (window, band) view is
chosen** moves the result from chance to far above the published number:

| Variant (same pipeline) | Balanced acc | Selection scope |
|---|---|---|
| TWFB **leaky** (7×8, selection sees test) | **77.1% ± 4.9** | peeks at test (oracle) |
| Shipped `.m` FB leaky (8 bands) | 67.2% ± 9.4 | peeks at test |
| *Published Liu Table 4* | *72.21%* | — (sits **between** the leaky variants) |
| Best fixed band, **no selection** (8–30 Hz) | 53.7% | none |
| **TWFB honest** (nested inner-CV selection) | **52.2% ± 6.9** | train-only (leakage-free) |

- **Leakage gap = 24.9 points** (77.1% → 52.2%), attributable solely to selection scope.
- Honest TWFB is only marginally above chance (per-subject t-test p = 0.026, Wilcoxon p = 0.051; 27/50 subjects > 0.5).
- Three independent honest Riemannian estimates agree at the chance floor: honest TWFB 52.2%, best fixed band
  53.7%, and Notebook A's canonical-band Riemann 50.6%.
- The leaky version proves the pipeline is *not* broken — it reaches 77% when allowed to cheat.

**Interpretation:** the published 72.21% is consistent with the very common BCI pitfall of selecting the
time-window/filter-band using the test set (or the full dataset). Properly nested, the method is at chance on
this dataset. (Framed as a methodological correction, not misconduct.)

---

## 5. Key numbers

### Published Liu2024 Table 4 (their 60/40, 10-fold)
CSP+LDA 55.57% · FBCSP+SVM 57.57% · TSLDA+DGFMDRM 61.20% · **TWFB+DGFMDRM 72.21%**.

### S-JEPA embedding probe + fusion sweep (Notebook A; 50 subjects, balanced accuracy)
| Experiment | Feature set | Classifier | Bal. acc |
|---|---|---|---|
| A_probe_lda_frozen_mean | sjepa | LDA | **53.95%** |
| A_fusion_lda | sjepa+riemann | LDA | 53.85% |
| A_probe_lda_meanmax | sjepa | LDA | 53.70% |
| A_fusion_lda_liu_holdout | sjepa+riemann | LDA (60/40) | 52.21% |
| A_probe_lda_finetuned | sjepa (finetuned) | LDA | 52.25% |
| A_probe_logreg_frozen_mean | sjepa | logreg | 52.10% |
| A_probe_lda_randominit | sjepa (random init) | LDA | 51.70% |
| A_riemann_only_lda | riemann (canonical) | LDA | 50.60% |
| A_fusion_logreg | sjepa+riemann | logreg | 49.75% |

- Best S-JEPA probe = **53.95%**, significantly above chance (per-subject Wilcoxon p = 0.006; subject sd 9.2%;
  30/50 subjects > 0.5, 16 ≥ 0.6). Strong subjects: S7 80%, S34 72%, S10 70%. Weak: S25 38%, S8/S12 40%.
- Pretraining contributes ~+2.3 pts (53.95% vs random-init 51.70%); modest, needs a per-subject paired test.
- Classifier/pooling barely matter; **fine-tuning spatial_conv per fold does not help** (adds variance on 32 trials).
- **Feature-level fusion is null:** the fixed-band Riemann branch is at chance (50.6%), so fusion ≈ S-JEPA alone.

### Honest head-to-head (same 50 subjects, paired)
- Honest TWFB **52.2%** vs S-JEPA probe **54.0%** → +1.7 pts, **not significant** (Wilcoxon p = 0.27; S-JEPA > TWFB on 30/50).
- Statistically **tied**, both near chance — but S-JEPA is the only one individually significant vs chance and
  has **no selection knob to leak through**.

---

## 6. Reframed thesis narrative (defensible, honest)

1. **Headline = leakage demonstration.** An honest reproduction shows the published TWFB-DGFMDM 72.21% is
   reproducible only with test-set view selection; nested properly, it is at chance (~52%). The controlled
   leaky-vs-honest toggle on one pipeline gives a clean 25-point causal demonstration.
2. **S-JEPA is competitive with the honest classical SOTA.** Frozen S-JEPA + simple LDA reaches ~54%, matching
   honest TWFB and clearing significance vs chance, with no leakage-prone selection step.
3. **The dataset is genuinely hard.** Acute stroke, ~40 trials/subject, ~42–46% clean-ERD — an honest hard
   ceiling; large subject heterogeneity is expected and observed.

This is stronger than the original "S-JEPA helps a bit on top of a strong Riemannian baseline," because there is
no strong *honest* Riemannian baseline to begin with.

---

## 7. Deliverables (files)

**Notebooks**
- `liu2024_sjepa_embed_probe_fusion.ipynb` — Notebook A: S-JEPA embedding probe (LDA/logreg) + optional filter-bank Riemannian feature fusion.
- `liu2024_twfb_dgfmdm_sjepa_hybrid.ipynb` — Notebook B: honest TWFB-DGFMDM (7×19, nested in-fold selection, tangent+PCA, FgMDM/ts_lda) × frozen-S-JEPA late fusion (weighted/stacking) + decorrelation diagnostic.
- `liu2024_twfb_dgfmdm_paper_faithful.ipynb` — paper Table-3 method faithfully: 19 bands × 7 windows + **real LTSA** (sklearn), leaky/honest/fixed modes, mirrors the `timewindow_faithful` summary shape.

**Config sweeps (JSON arrays for the apply tool)**
- `sweeps_notebook_A.json` (9), `sweeps_notebook_B.json` (10), `sweeps_notebook_paper_faithful.json` (6).

**Explanatory PDFs**
- `TWFB_DGFMDM_explained.pdf` — the shipped code, plain + technical + glossary.
- `TWFB_DGFMDM_paper_method_explained.pdf` — the paper's honest Table-3 method.

All notebooks follow the `liu2024_source_mat_sjepa_prelocal_augmented` logging/artifact conventions: timestamped
`print` tee'd to `run.log`, `RUN_ID` hashed from CONFIG, `config.json`, and the CSV/JSON artifact set; each has a
**single editable CONFIG cell**.

---

## 8. Historical open items

The paper-faithful confirmation is complete, and the old Notebook-B batch completed four branches
before a kernel failure; do not rerun those branches. Later fusion experiments were null, so weighted
fusion is not a current priority without a new prespecified hypothesis. Remaining genuinely open work
is limited to the exact protocol-matched S-JEPA/TWFB comparison, any missing per-subject thesis
statistics, and author clarification of the Table-3 selection boundary. See workspace-root `AGENTS.md`
§2e and §3 for authoritative status.

---

## 9. Caveats and methodological notes

- The leaky vs honest difference is a controlled, single-variable demonstration — strong evidence, but the
  original authors' exact selection protocol is unknown; frame as a known evaluation pitfall, not misconduct.
- `liu2024_twfb_dgfmdm_timewindow_faithful` used the **8 shipped bands** (not the paper's 19) and **no LTSA**; the
  paper-faithful notebook closes both gaps.
- LTSA on ~24–32 trials in 435-d tangent space is noisy; the faithful notebook pre-reduces with PCA and falls back
  to tangent features if LTSA is unstable on a fold. This is itself a fair critique of the method on tiny data.
- Verify per-environment: MI onset sample (marker==2), channel order, and that the S-JEPA checkpoint loads.
- Fold counts: `sjepa_5fold` → 32/8; `liu_repeated_holdout` (60/40 ×10) → 24/16.
