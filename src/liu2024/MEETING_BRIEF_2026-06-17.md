# Meeting brief — Wed 2026-06-17, 4:30 pm

Maps to the four items in `HELLO.md`. Headline result this week: **a faithful, runnable TWFB+DGFMDM
reproduction that explains where Liu2024's 72.21% comes from**, plus a data bug found and fixed that
was silently hurting every classical baseline.

---

## 0. The one slide to lead with

> Liu2024's headline **72.21%** is **oracle-inflated**. Their shipped MATLAB picks the frequency band by
> **test-set** accuracy, and the *time-window* half of "TWFB" isn't even in the shipped code. When I add
> the time-window search and keep their leaky selection I get **~75–80%** (i.e. I can reproduce / exceed
> 72%); when I select windows/bands **honestly** (inner-CV on train only) the same pipeline lands
> **~52–58% balanced accuracy**. **That honest number — not 72% — is the real Riemannian ceiling S-JEPA
> must beat.** Most subjects sit near chance; a handful (e.g. 7, 44, 45) are genuinely decodable (73–82%).

---

## 1. "Pretrain on the 500 Hz — check this"  ✅ done, works

- The self-contained 500 Hz S-JEPA pretraining (`liu2024_sjepa_500hz_optionB_pretraining_corrected.ipynb`
  → `…_monitored_2.ipynb`) **completed**: 147 epochs, smooth loss (val 0.41 → 0.075), early-stopped,
  checkpoints saved (`student_backbone_with_chans_best.pt`, etc.). The old `probe_pos` NaN crash is gone
  — fixed by a finite-by-construction positional encoder (channel coords centered + scaled to unit ball).
- SSL-quality monitoring (linear probe on known-decodable subjects, every 5 epochs): peaks **~56–60%**,
  effective rank drops 10 → ~2–3 (mild, not catastrophic collapse). **Verdict: the encoder learns real
  structure but does not densely capture MI**; transfer to unseen subjects is weak — expected.
- Honest framing for the meeting: 500 Hz is *semantically valid only because pretraining + downstream +
  tokenization are all consistently 500 Hz*; it buys robustness vs. resample artifacts, not new info
  (MI is band-limited ≤30 Hz; 128 Hz already covers it).

## 2. "Reduce parameters"  ✅ done — size is **not** the bottleneck

From `liu2024_sjepa_parameter_reduction_sweep.ipynb` (within-subject 5-fold, 50 subjects):

| variant | params | bal. acc | collapse | note |
|---|---|---|---|---|
| tiny | 4.1k | 51.6% | 1.2% | too small |
| small | 16k | 52.6% | 0.8% | stable |
| medium | 59k | 52.6% | 2.4% | no gain over small |
| baseline **pretrained** | 16k | **57.2%** | **40%** | best mean but heavy right-hand collapse |

Takeaway: shrinking the model neither helps nor hurts much (~52%); the limit is **~24 training trials
per fold**, not capacity. Pretraining adds ~+5 pts but destabilises (collapse) → needs the balance/loss
controls already in the augmented notebook. **Recommend: keep `small` (16k) as the standard backbone.**

## 3. "Try a normal classifier again — understand what's going on"  ✅ done

Same data / splits, simple baselines (`liu2024_supervised_classifier_diagnostics.ipynb`) + my TWFB work:

| method | bal. acc (within-subject) |
|---|---|
| CSP + LDA | ~55% |
| FBCSP + LDA | ~53% |
| logreg band-power | ~54% |
| Riemann MDM | ~52% |
| **TWFB+DGFMDM, honest nested** | **~52–58%** |
| S-JEPA pre-local (best) | ~56–57% |

**What's going on:** every honest method clusters at **52–58%**. ~9 subjects are consistently un-decodable
(≤45%); a few are strongly decodable. So this is a **hard, small, heterogeneous dataset**, not a broken
model. The right scientific story is a careful *diagnostic + hybrid*, not a SOTA claim.

## 4. "Learn TWFB + summarize it / shifting-window technique"  ✅ done — main deliverable

**New runnable notebook:** `liu2024_twfb_dgfmdm_timewindow_faithful.ipynb` (CONFIG-cell driven, same
structure as the augmented notebook). It is a faithful Python port of `TWFB_DGFMDM.m` **plus** the
time-window search the shipped code omits.

**What TWFB is:** per-subject search over **time windows** (where in 0–4 s the ERD/ERS is strongest) ×
**filter bank** (which mu/beta sub-band separates L/R), feeding **DGFMDM** = Fisher geodesic discriminant
filtering + minimum-distance-to-Riemannian-mean on band-limited spatial **covariances** (`pyriemann.FgMDM`).
Near-zero trainable parameters → strong on tiny data.

**Three findings (all reproduced in the notebook; numbers below are the full all-50 runs):**
1. The shipped `.m` does the **filter-bank half only** (8 bands), and selects the band by **max test
   accuracy** with a different random split per band → **double leakage**. Faithful port = **67.2%** (n=50).
2. Adding the **7 shifting 1-s time windows** (paper's real TWFB) and keeping the leaky max-over-(window×band)
   on test → **77.1%** (n=50), i.e. the 72.21% is reproduced/exceeded as an *oracle* number.
3. **Honest** selection (inner-CV on train only) of window+band → **~52–58% balanced accuracy**. Notably,
   honest *search* barely beats a single fixed sensible band — with 24 train trials the selection variance
   eats the gains. **→ 72.21% is not an honest target; ~55–60% is.**

**Data bug found & fixed (important):** the MI cue (`marker==2`) sits at **sample ~1003 (≈2.0 s)** inside
each 4000-sample trial, *not* at sample 0. Any pipeline that treated "0–4 s" as samples 0–2000 analysed
the **pre-imagery** period → chance. Also, artifact subjects 8/13/14 carry a spurious `marker==2` at
sample ~1; the new loader rejects implausible onsets and falls back to the subject's median. (The S-JEPA
notebooks' `start≈2.0 s` was accidentally correct — that's why they reached ~56%.)

---

## What to ask / decide at the meeting
- Agree the **honest Riemannian ceiling (~55–60%)** as the benchmark, and that we report 72.21% only as a
  "how inflated" diagnostic.
- Greenlight the **hybrid** (S-JEPA representation branch + TWFB/Riemannian geometry branch, fold-safe
  fusion) as the thesis method — see `what_is_possible` notes.
- Primary on all 50 + secondary on the clean 37 (drop the 13 artifact subjects), report both.

*Numbers above are from validated probes; the full all-50 clean run (`liu2024_twfb_dgfmdm_timewindow_faithful.ipynb`)
was executing at write time — rerun it for the exact table.*
