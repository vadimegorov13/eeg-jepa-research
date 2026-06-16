# TWFB + DGFMDRM on Liu2024 — Research & Planning Note

*Purpose: explain the Liu2024 strongest baseline (TWFB + DGFMDRM) well enough to (a) present
it to an advisor and (b) decide how it should inform the next S-JEPA experiments. This is a
conceptual note; the companion starter notebook
`liu2024_twfb_shifting_window_diagnostics.ipynb` implements the parts that are practical now
and is explicit about what is approximated.*

> **Scope honesty up front.** This note describes the standard, well-established machinery
> behind each acronym. Where a specific detail of the Liu2024 implementation (exact band
> edges, window set, the precise "discriminant geodesic filtering" formulation, and the
> selection/CV protocol) would need to be read off the paper to reproduce faithfully, that is
> flagged rather than asserted. Treat the band/window specifics below as the *typical* MI
> decoding choices, to be confirmed against the paper before claiming an exact reproduction.

---

## 1. What TWFB means (Time-Window + Filter-Bank)

TWFB is a **search over where (in time) and in which frequency band the motor-imagery signal
is most discriminative**, instead of committing to one fixed window and one fixed band.

### 1a. Time-window search
Motor imagery produces **event-related desynchronization/synchronization (ERD/ERS)**: a drop
(then rebound) in sensorimotor-rhythm power over the contralateral motor cortex, *time-locked
to the cue*. The discriminative information is therefore **localized in time** — typically it
builds up a fraction of a second after cue onset, peaks somewhere in the imagery period, and
fades. A fixed window that starts too early (pre-ERD) or is too long (diluting the
informative segment with rest) throws away signal-to-noise. Time-window search slides and/or
resizes the analysis window and keeps the window(s) that separate the classes best.

### 1b. Filter-bank search
ERD/ERS is also **band-specific** and varies across people: the **mu rhythm (~8–12 Hz)** and
**beta band (~13–30 Hz)** carry most of the hand-MI information, but the exact sub-band that
discriminates left vs right differs per subject (and, in a stroke cohort, per lesion). A
**filter bank** is a set of overlapping band-pass filters (e.g., several ~4 Hz-wide bands
tiling roughly 8–30 Hz). Features are computed per band, and the bands that help are kept or
weighted (classic FBCSP uses a mutual-information criterion to select bands).

### 1c. Why MI EEG depends so strongly on time/frequency windows
- The informative physiology (ERD/ERS) is intrinsically a **time × frequency** phenomenon, so
  the right slice of the time–frequency plane is where the class difference actually lives.
- EEG is **non-stationary and low-SNR**; outside the ERD window/band you are mostly adding
  noise, which a small-data classifier cannot average away.
- **Subject heterogeneity**: peak frequency and timing shift between people, so a per-subject
  search is a strong, cheap form of personalization.

The practical upshot: a method that *finds* the best window/band has a large built-in
advantage on small MI datasets, before any "model capacity" is spent.

---

## 2. What DGFMDRM means (Discriminant Geodesic Filtering + Minimum Distance to Riemannian Mean)

DGFMDRM is a **Riemannian-geometry classifier on EEG covariance matrices**. Its three ideas:

### 2a. Covariance matrices as features
For a windowed, band-filtered trial `X` (channels × time), the **spatial covariance matrix**
`C = X Xᵀ / (T−1)` summarizes how channels co-vary. For oscillatory MI, band-power and its
spatial distribution — exactly the ERD/ERS signature — are captured by `C`. Covariance is a
**physiologically apt, low-parameter** representation: no weights to learn to *form* the
feature.

### 2b. Riemannian geometry of SPD matrices
Covariance matrices are **symmetric positive-definite (SPD)**. SPD matrices do not live in a
flat Euclidean space; they form a **curved manifold**. Using the **affine-invariant Riemannian
metric**, distances and averages respect that curvature. Two consequences matter:
- The **geometric (Fréchet) mean** of a set of covariances is the manifold analogue of an
  average and is far more faithful than the elementwise mean.
- The metric is **invariant to invertible linear transforms** of the channels, which buys
  robustness to some electrode/scaling nuisances.
A common companion is the **tangent-space mapping**: project covariances onto the (flat)
tangent plane at the global mean, vectorize the upper triangle, and feed a plain linear
classifier — turning manifold features into something a logistic regression / LDA can use.

### 2c. Discriminant geodesic filtering (DGF)
This is a **supervised, geometry-aware filtering/projection** step: reduce the covariance
representation to the directions (on or along the manifold's geodesics) that **maximize
class separability**, analogous to how CSP finds discriminative spatial filters but performed
consistently with the Riemannian structure. It improves separation and denoises before
classification. *The precise DGF formulation used in Liu2024 should be read from the paper;
the starter notebook does not reproduce DGF and says so explicitly.*

### 2d. Minimum Distance to Riemannian Mean (MDRM / MDM)
The classifier itself is almost parameter-free: compute the **Riemannian mean covariance of
each class** from the training data, then label a test trial by the **class whose mean is the
nearest** under the Riemannian distance. With DGF in front, this is DGFMDRM. MDM has
essentially **no trainable weights** beyond the per-class means — which is precisely why it
generalizes from very few trials.

---

## 3. What Liu2024 did (as framed in the brief)

- **0–4 s MI decoding**: classify left- vs right-hand motor imagery from the imagery period.
- **8–30 Hz band**: focus on the mu+beta sensorimotor range where hand MI lives.
- **Shifting time windows**: evaluate multiple (shifted, possibly resized) sub-windows within
  the imagery period rather than one fixed window.
- **Overlapping filter banks**: tile 8–30 Hz with overlapping band-passes and compute
  covariance features per band.
- **Selection of optimal time/frequency windows**: pick, per subject, the (window, band)
  combination(s) that decode best — then classify with the Riemannian (DGFMDRM) pipeline.

The combination — *personalized time/frequency selection* feeding a *low-parameter Riemannian
classifier* — is a strong, well-matched recipe for small MI datasets.

*(Confirm the exact window set, band edges/overlap, selection criterion, and CV protocol
against the paper before describing this as an exact reproduction.)*

---

## 4. Why this may outperform S-JEPA right now

This is the crux for the advisor conversation. None of these points say deep learning "cannot"
work here — they explain why the TWFB+Riemannian recipe is hard to beat *in this regime*.

- **Very few trials per subject (~40).** MDM has near-zero trainable parameters; it estimates
  a couple of class-mean covariances and is done. A deep S-JEPA head has orders of magnitude
  more parameters to constrain with the same 40 trials → high variance, overfitting, or the
  collapse/one-class behaviour you have been seeing.
- **Strong, correct inductive bias.** Covariance + Riemannian geometry *encodes* the right
  structure (oscillatory power + spatial pattern, with linear-transform invariance). The
  network must *learn* that structure from data it does not have enough of.
- **Spatial-covariance features match MI physiology.** ERD/ERS is a change in band-limited
  spatial covariance — almost exactly what `C` measures — so the feature is close to a
  sufficient statistic for the task.
- **Parameter efficiency / no representation collapse.** A classifier with no learned
  representation cannot suffer representation collapse; MDM either separates the means or it
  does not, transparently.
- **TWFB does the personalization the network is not getting.** Per-subject window/band
  selection is a powerful adaptation step. An S-JEPA run on a fixed, possibly sub-optimal
  window/band is competing without that advantage.

A useful one-line framing: *"The Riemannian baseline wins because it spends its limited data
on estimating two class means in a representation that already matches the physiology, while
S-JEPA spends the same data trying to learn both the representation and the decision."*

---

## 5. How this can help S-JEPA (concrete next steps)

The goal is to let TWFB/Riemannian insight **improve and diagnose** S-JEPA, not replace it.

1. **Use TWFB windows to choose better S-JEPA windows.** Take the per-subject (and group)
   best time window and band from the shifting-window analysis and use them to set the S-JEPA
   input window start/length and pre-filter band. This directly addresses the
   sampling-rate/window audit work: feed S-JEPA the segment where MI signal actually is.
2. **Benchmark S-JEPA embeddings against Riemannian features under a fair probe.** Freeze the
   S-JEPA encoder, extract embeddings, and train the *same* simple linear classifier on (a)
   S-JEPA embeddings and (b) Riemannian tangent-space vectors. If the linear probe on
   Riemannian features wins, the S-JEPA representation is not yet capturing the discriminative
   structure — a representation-quality verdict independent of the head.
3. **Hybrid feature fusion.** Concatenate the S-JEPA embedding with the vectorized
   tangent-space (or selected band-covariance) features and train one linear head. This tests
   whether S-JEPA adds anything *beyond* the Riemannian baseline, and often stabilizes small-
   data training because the Riemannian part carries a strong, ready-made signal.
4. **Shifting-window analysis as a diagnostic for where MI signal is strongest.** The
   per-subject time/frequency sensitivity maps (from the starter notebook) tell you whether a
   subject has *any* decodable MI and *when/where* — which both explains "impossible" subjects
   from the supervised-diagnostics notebook and tells you whether S-JEPA is even being shown
   the informative segment.

Together these turn the TWFB/Riemannian result from "an embarrassing baseline that beat us"
into "a measurement instrument and a source of inductive bias for the deep model."

---

## 6. Companion notebook: what is implemented vs approximated vs remaining

The starter notebook `liu2024_twfb_shifting_window_diagnostics.ipynb` is intentionally a
*first version* focused on the time-window × filter-band sensitivity analysis.

**Faithfully implemented now**
- Shifting **time-window** sweep (configurable start grid) and **filter-bank** sweep
  (configurable overlapping bands), reusing the project's exact preprocessing/CV/artifacts.
- **Covariance features + Minimum Distance to Riemannian Mean** classification: the
  affine-invariant MDM via `pyriemann` when available, with a dependency-free **log-Euclidean
  MDM** fallback (a legitimate, if simpler, Riemannian metric).
- Per-subject **best (window, band)** selection and **subject-level time/frequency sensitivity
  heatmaps**; artifacts saved in the same style as the other notebooks.

**Approximated / simplified (clearly, not silently)**
- **No discriminant geodesic filtering (DGF).** The notebook classifies with plain MDM (or
  tangent-space + linear), i.e. the "MDRM" half of DGFMDRM; the "DGF" half is omitted.
- The **filter bank and window grid are reasonable defaults**, not necessarily Liu2024's exact
  bands/windows.
- The **selection criterion** is "best balanced accuracy on the within-subject CV," which is a
  transparent stand-in for whatever selection/nested-CV protocol the paper uses (and is itself
  a mild optimistic bias to keep in mind — see the notebook's caveat).

**Remaining to reproduce Liu2024 faithfully**
- The exact **DGF** formulation and its integration with MDM.
- The paper's exact **band edges/overlap, window set, and the 0–4 s framing**.
- The exact **selection + cross-validation protocol** (e.g., nested selection to avoid the
  optimistic bias of choosing the window/band on the same CV that scores it).
- Any **feature fusion / weighting across selected windows/bands** the paper performs.

---

### TL;DR for the advisor
TWFB finds *when* and *in which band* the motor-imagery signal is strongest, per subject;
DGFMDRM classifies the resulting band-limited spatial **covariances** with a near-parameter-
free **Riemannian nearest-mean** rule. On ~40 trials/subject that combination of strong,
physiology-matched inductive bias and minimal parameters is exactly what S-JEPA lacks right
now — so the immediate value of TWFB/Riemannian work is to **choose S-JEPA's window/band, to
benchmark and diagnose its embeddings, and potentially to fuse the two**, rather than to keep
scaling the deep model on data that cannot constrain it.
