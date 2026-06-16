"""Generate liu2024_twfb_dgfmdm_timewindow_faithful.ipynb.

A faithful, RUNNABLE Python reproduction of Liu2024's TWFB + DGFMDM (FgMDM), built on
native-500Hz source data with marker-based onset detection, plus the *time-window* search
that the shipped TWFB_DGFMDM.m omits. Reports, clearly separated:
  - shipped-.m faithful : filter-bank-only, leaky max-over-bands (reproduces the .m, ~60-64%)
  - paper TWFB          : time-window x filter-bank, leaky max-over-combos (reproduces ~72%)
  - honest nested-CV    : inner-CV selection of (window,band) on TRAIN only (the real number)

Structure mirrors liu2024_source_mat_sjepa_prelocal_augmented.ipynb (sectioned CONFIG cell,
data section, core, run-all, results+artifacts). Kernel: the project .venv 'python3'.
"""
import json, os

cells = []
def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.rstrip("\n").splitlines(keepends=True)})

md("""# Liu2024 — TWFB + DGFMDM (FgMDM): faithful reproduction + time-window extension

A **runnable** Python port of the Liu2024 strongest baseline, built to settle three questions:

1. Does the project's data + labels reproduce Liu2024's classical result with their *own* pipeline?
2. Where does the paper's headline **72.21%** actually come from?
3. What is the **honest, leakage-free** Riemannian ceiling on this dataset?

**What the shipped `TWFB_DGFMDM.m` actually does** (reverse-engineered from the MATLAB):
- 29 channels `[1:17 19:30]` (drops CPz); native **500 Hz**.
- Per trial: onset = first sample where marker channel (col 33) `== 2`; window = `onset-800 : onset+2000`
  (2800 samp), **50 Hz notch + order-2 Butterworth one-pass** band-pass, then keep samples `801:2800`
  = **0–4 s after onset** (2000 samp). The 800-sample pre-roll is filter warm-up, then discarded.
- Covariance = `Xᵀ X` (unnormalized **scatter matrix**).
- Classifier = `fgmdm` = **FgMDM** (Fisher geodesic discriminant filtering + MDM), affine-invariant.
- **8 bands** `{[8,12],[8,20],[8,30],[12,20],[15,20],[15,30],[20,30],[8,15]}` — *not* the paper's 19 bands.
- Protocol: 10 repeats; **a fresh random 24/16 split per band**, then **`max` over bands** — i.e. the band
  is chosen by **test-set** accuracy (oracle), and the winning band also gets the luckiest split.

**The critical gap:** the shipped `.m` does the **FB** (filter-bank) part only. The paper's *TWFB* is
**Time-Window + Filter-Bank** — a search over 7 shifting 1-s windows in 0–4 s *and* the bands. The
time-window search is **not in the shipped code**. Re-adding it (and keeping the same leaky
max-over-everything-on-test selection) is what pushes ~60% up toward the paper's ~72%.

This notebook implements the full TW×FB and reports the leaky vs honest numbers side by side, so the
**72.21% is shown to be oracle-inflated** and the honest Riemannian ceiling is reported as the real target.

> Onset note: in the segmented trials the MI cue (`marker==2`) sits at **sample ~1003 (≈2.0 s)**, *not*
> sample 0. Measuring the 0–4 s window from sample 0 analyses the pre-imagery prep period → chance. This
> notebook detects the onset per trial from the marker channel, exactly like the MATLAB.
""")

md("# 1. Setup")
code('''import os, sys, glob, json, time, hashlib, platform
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, iirnotch, lfilter
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception as e:
    HAVE_MPL = False; print("matplotlib unavailable:", e)

from pyriemann.classification import FgMDM   # == MATLAB fgmdm (DGFMDM)

print("python", platform.python_version(), "| numpy", np.__version__)
import pyriemann; print("pyriemann", pyriemann.__version__)''')

md("# 2. Configuration\n\nSingle CONFIG cell. Sweep the band set, the time-window grid, the regularization, or the protocol here.")
code('''WORKING_DIR = Path.cwd().resolve().parent.parent

CONFIG = {
    # ---- paths / identity ----
    "experiment_name": "twfb_dgfmdm_timewindow_faithful",
    # Native-500Hz continuous source .mat (what the MATLAB runs on). Falls back to the
    # figshare copy under liu2024_data if the matlab_code tree is absent.
    "source_roots": [
        str(WORKING_DIR.parent / "Liu2024_matlab_code" / "sourcedata"),
        str(WORKING_DIR / "liu2024_data" / "liu2024_figshare" / "sourcedata"),
    ],
    "artifact_dir": str(WORKING_DIR / "artifacts" / "liu2024-twfb-dgfmdm-timewindow-faithful"),

    # ---- dataset / channels (faithful to TWFB_DGFMDM.m) ----
    "native_sfreq": 500,
    "channel_indices": list(range(0,17)) + list(range(18,30)),  # [1:17 19:30] 1-based -> 29 ch, drop CPz
    "marker_channel_index": 32,         # col 33 (1-based)
    "onset_marker_value": 2,            # marker==2 marks MI onset (t=0)
    "onset_plausible_range": [800, 1300],  # accept onset in this sample range (~1.6-2.6 s)
    "onset_fallback_sample": 1003,      # ~2.0 s, used if a trial's onset is implausible/missing
    "subjects_to_use": None,            # None = all found; or e.g. list(range(1,11))

    # ---- windowing (faithful: 0-4 s after onset, 800-sample warm-up pre-roll) ----
    "preroll_samples": 800,             # 1.6 s pre-onset, included for filtering then trimmed
    "mi_window_samples": 2000,          # 4 s @ 500 Hz, kept after trim

    # ---- filtering (faithful) ----
    "notch_freq": 50.0, "notch_Q": 6.0,
    "butter_order": 2,                  # MATLAB BandpassFilter default order
    "freq_bands": [(8,12),(8,20),(8,30),(12,20),(15,20),(15,30),(20,30),(8,15)],

    # ---- time-window search (the part missing from the shipped .m) ----
    # 1-s windows shifted by 0.5 s across 0-4 s, paper-style.
    "time_window_starts_s": [0.0,0.5,1.0,1.5,2.0,2.5,3.0],
    "time_window_len_s": 1.0,

    # ---- covariance regularization (numerical stability for FgMDM Riemannian mean) ----
    # Trace-normalize then shrink toward scaled identity. Bounds condition number; the
    # affine-invariant metric is invariant to the per-matrix trace scaling. gamma=0 -> raw scatter.
    "cov_trace_normalize": True,
    "cov_shrinkage": 0.10,

    # ---- protocols to run ----
    "run_shipped_fb_leaky": True,       # filter-bank only, leaky max-over-bands (reproduces the .m)
    "run_twfb_leaky": True,             # time-window x filter-bank, leaky (reproduces ~72%)
    "run_twfb_honest": True,            # nested inner-CV selection on train only (honest ceiling)
    "run_perband_fixed": True,          # per-band fixed (no selection), honest 60/40

    # ---- evaluation protocol ----
    "n_repeats": 10,
    "train_size": 24, "test_size": 16,  # MATLAB 24/16 = 60/40
    "honest_test_frac": 0.40,
    "inner_folds": 3,
    "random_state": 2026,
}

BANDS = [tuple(b) for b in CONFIG["freq_bands"]]
FS = CONFIG["native_sfreq"]
CH = CONFIG["channel_indices"]
NCH = len(CH)
print(f"channels={NCH}  bands={len(BANDS)}  time_windows={len(CONFIG['time_window_starts_s'])}"
      f"  TWxFB combos={len(BANDS)*len(CONFIG['time_window_starts_s'])}")
for k in ["native_sfreq","freq_bands","time_window_starts_s","time_window_len_s",
          "cov_trace_normalize","cov_shrinkage","n_repeats"]:
    print(f"  {k:22s}: {CONFIG[k]}")''')

md("# 3. Data loading and feature construction")
code('''def find_subject_files(roots):
    for root in roots:
        files = sorted(glob.glob(os.path.join(root, "sub-*", "sub-*_eeg.mat")))
        if files:
            print(f"using source root: {root}  ({len(files)} subjects)")
            return files
    raise FileNotFoundError(f"No sub-*_eeg.mat under any of: {roots}")

def load_subject(path):
    """Return rawdata (n,33,4000), labels (n,), per-trial onset sample list."""
    m = sio.loadmat(path); eeg = m["eeg"][0,0]
    raw = np.asarray(eeg["rawdata"], dtype=np.float64)
    lab = np.asarray(eeg["label"]).ravel().astype(int)
    mark = raw[:, CONFIG["marker_channel_index"], :]
    lo, hi = CONFIG["onset_plausible_range"]
    # Some trials (esp. artifact subjects 8/13/14) carry a spurious marker==2 at sample ~1;
    # prefer the first marker inside the plausible MI-onset range, else fall back to the
    # subject's median plausible onset. Avoids negative/empty windows that poison covariances.
    first = []
    for t in range(raw.shape[0]):
        idx = np.where(mark[t] == CONFIG["onset_marker_value"])[0]
        in_range = idx[(idx >= lo) & (idx <= hi)]
        first.append(int(in_range[0]) if in_range.size else (int(idx[0]) if idx.size else -1))
    plausible = [o for o in first if lo <= o <= hi]
    med = int(np.median(plausible)) if plausible else CONFIG["onset_fallback_sample"]
    onsets = [o if lo <= o <= hi else med for o in first]
    return raw, lab, onsets

# Filters (one-pass, like MATLAB 'filter')
_wo = CONFIG["notch_freq"]/(FS/2); _bw = _wo/CONFIG["notch_Q"]
NOTCH_B, NOTCH_A = iirnotch(_wo, _wo/_bw)
BAND_BA = {bd: butter(CONFIG["butter_order"], [bd[0]/(FS/2), bd[1]/(FS/2)], btype="band") for bd in BANDS}

def _regularize(c):
    if CONFIG["cov_trace_normalize"]:
        c = c / np.trace(c)
    g = CONFIG["cov_shrinkage"]
    if g > 0:
        scale = (np.trace(c)/NCH) if not CONFIG["cov_trace_normalize"] else (1.0/NCH)
        c = (1-g)*c + g*scale*np.eye(NCH)
    return c

def band_mi_segments(raw, onsets, bd):
    """Faithful 0-4 s MI segment per trial for one band: (n, 29, mi_window_samples)."""
    b, a = BAND_BA[bd]; pre = CONFIG["preroll_samples"]; L = CONFIG["mi_window_samples"]
    n = raw.shape[0]; out = np.zeros((n, NCH, L))
    for t in range(n):
        s0 = onsets[t] - pre
        seg = raw[t][:, s0:s0+pre+L][CH, :]      # (29, pre+L)
        seg = lfilter(NOTCH_B, NOTCH_A, seg, axis=1)
        seg = lfilter(b, a, seg, axis=1)
        out[t] = seg[:, pre:pre+L]               # trim warm-up -> 0-4 s
    return out

def cov_full(seg):
    """Scatter covariance over the whole 0-4 s window (shipped .m): (n,29,29)."""
    n = seg.shape[0]; C = np.zeros((n, NCH, NCH))
    for t in range(n):
        X = seg[t]; C[t] = _regularize(X @ X.T)
    return C

def cov_window(seg, wstart_s, wlen_s):
    """Scatter covariance over a shifted sub-window (time-window search): (n,29,29)."""
    s = int(round(wstart_s*FS)); e = s + int(round(wlen_s*FS))
    n = seg.shape[0]; C = np.zeros((n, NCH, NCH))
    for t in range(n):
        X = seg[t][:, s:e]; C[t] = _regularize(X @ X.T)
    return C''')

md("""# 4. DGFMDM core and evaluation protocols

`FgMDM` is the Python equivalent of the MATLAB `fgmdm`. On a few subjects its internal Fisher
geodesic step can hit a non-PD matrix; we wrap fits to return `NaN` and aggregate with nan-aware
max/mean so a single bad (window,band) never aborts the run.""")
code('''def _acc(Ctr, ytr, Cte, yte):
    """One FgMDM fit -> test accuracy. Robust to occasional FgMDM numerical failures."""
    try:
        clf = FgMDM(metric="riemann"); clf.fit(Ctr, ytr)
        return float(np.mean(clf.predict(Cte) == yte))
    except Exception:
        return np.nan

def _acc_bacc(Ctr, ytr, Cte, yte):
    try:
        clf = FgMDM(metric="riemann"); clf.fit(Ctr, ytr)
        pred = clf.predict(Cte)
        return float(np.mean(pred == yte)), float(balanced_accuracy_score(yte, pred))
    except Exception:
        return np.nan, np.nan

def leaky_oracle(combo_covs, y, n_repeats, rng):
    """MATLAB leaky protocol: per combo a fresh random 24/16 split, report max-over-combos
    TEST accuracy (oracle). combo_covs: dict key -> (n,ch,ch)."""
    n = len(y); reps = []
    for _ in range(n_repeats):
        accs = []
        for cov in combo_covs.values():
            p = rng.permutation(n)
            tr = p[:CONFIG["train_size"]]; te = p[CONFIG["train_size"]:CONFIG["train_size"]+CONFIG["test_size"]]
            accs.append(_acc(cov[tr], y[tr], cov[te], y[te]))
        reps.append(np.nanmax(accs))
    return float(np.mean(reps))

def honest_nested(combo_covs, y, splits, inner_folds, rng_state):
    """Leakage-free: select (window,band) by inner CV on TRAIN only; eval on held-out test."""
    accs, baccs = [], []
    keys = list(combo_covs.keys())
    for tr, te in splits:
        skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=rng_state)
        best_key, best_v = keys[0], -np.inf
        for k in keys:
            cov = combo_covs[k]
            iv = [_acc(cov[tr[a]], y[tr[a]], cov[tr[b]], y[tr[b]]) for a, b in skf.split(tr, y[tr])]
            mv = np.nanmean(iv)
            if mv > best_v: best_v, best_key = mv, k
        a, ba = _acc_bacc(combo_covs[best_key][tr], y[tr], combo_covs[best_key][te], y[te])
        accs.append(a); baccs.append(ba)
    return float(np.nanmean(accs)), float(np.nanmean(baccs))''')

md("# 5. Run all subjects")
code('''def run_all():
    files = find_subject_files(CONFIG["source_roots"])
    keep = CONFIG["subjects_to_use"]
    rows = []
    rng = np.random.RandomState(CONFIG["random_state"])
    t0 = time.time()
    for path in files:
        sid = int(os.path.basename(path).split("-")[1][:2])
        if keep is not None and sid not in keep: continue
        raw, y, onsets = load_subject(path)
        n = raw.shape[0]; idx = np.arange(n)
        segs = {bd: band_mi_segments(raw, onsets, bd) for bd in BANDS}
        fb_covs   = {bd: cov_full(segs[bd]) for bd in BANDS}                       # 8 band-only
        twfb_covs = {(bd,ws): cov_window(segs[bd], ws, CONFIG["time_window_len_s"])
                     for bd in BANDS for ws in CONFIG["time_window_starts_s"]}     # 56 TWxFB
        sss = StratifiedShuffleSplit(n_splits=CONFIG["n_repeats"], test_size=CONFIG["honest_test_frac"],
                                     random_state=CONFIG["random_state"])
        splits = list(sss.split(idx, y))
        row = {"subject": sid, "n_trials": n}
        if CONFIG["run_shipped_fb_leaky"]:
            row["shipped_fb_leaky"] = leaky_oracle(fb_covs, y, CONFIG["n_repeats"], rng)
        if CONFIG["run_twfb_leaky"]:
            row["twfb_leaky"] = leaky_oracle(twfb_covs, y, CONFIG["n_repeats"], rng)
        if CONFIG["run_twfb_honest"]:
            a, ba = honest_nested(twfb_covs, y, splits, CONFIG["inner_folds"], 0)
            row["twfb_honest_acc"], row["twfb_honest_bacc"] = a, ba
        if CONFIG["run_perband_fixed"]:
            for bd in BANDS:
                row[f"fixed_{bd[0]}_{bd[1]}"] = float(np.nanmean([_acc(fb_covs[bd][tr], y[tr], fb_covs[bd][te], y[te]) for tr,te in splits]))
        rows.append(row)
        msg = f"sub-{sid:02d}: " + "  ".join(
            f"{k}={row[k]*100:5.1f}" for k in ["shipped_fb_leaky","twfb_leaky","twfb_honest_bacc"] if k in row)
        print(msg + f"   ({time.time()-t0:5.1f}s)", flush=True)
    return pd.DataFrame(rows)

RESULTS = run_all()
RESULTS.head()''')

md("# 6. Results, diagnostics, artifacts")
code('''def summary(df):
    print("="*64); print("TWFB + DGFMDM (FgMDM) — Liu2024, n=%d subjects" % len(df)); print("="*64)
    def line(col, label):
        if col in df: print(f"  {label:42s}: {df[col].mean()*100:5.2f}%  ± {df[col].std()*100:4.2f}")
    line("shipped_fb_leaky", "Shipped .m  (FB-only, leaky max-over-bands)")
    line("twfb_leaky",       "Paper TWFB  (TWxFB, leaky max-over-combos)")
    line("twfb_honest_bacc", "Honest nested-CV (bal. acc, leakage-free)")
    print("\\n  Liu2024 Table 4 TWFB+DGFMDRM (reported)   : 72.21%")
    fixed_cols = [c for c in df.columns if c.startswith("fixed_")]
    if fixed_cols:
        print("\\n  Per-band fixed (honest 60/40, no selection):")
        for c in sorted(fixed_cols): print(f"    {c.replace('fixed_','').replace('_','-'):8s}: {df[c].mean()*100:5.2f}%")
    if "twfb_honest_bacc" in df:
        dec = df.sort_values("twfb_honest_bacc", ascending=False)
        print("\\n  Most decodable subjects (honest bal.acc):")
        for _, r in dec.head(8).iterrows(): print(f"    sub-{int(r['subject']):02d}: {r['twfb_honest_bacc']*100:5.1f}%")
        print(f"  subjects with honest bal.acc > 60%: {(df['twfb_honest_bacc']>0.60).sum()} / {len(df)}")
summary(RESULTS)''')

code('''# Artifacts
art = Path(CONFIG["artifact_dir"]); art.mkdir(parents=True, exist_ok=True)
RESULTS.to_csv(art/"subject_results.csv", index=False)
summ = {k: (float(RESULTS[k].mean()), float(RESULTS[k].std())) for k in
        ["shipped_fb_leaky","twfb_leaky","twfb_honest_acc","twfb_honest_bacc"] if k in RESULTS}
json.dump({"config": {k:(str(v) if isinstance(v,Path) else v) for k,v in CONFIG.items()},
           "summary": summ, "n_subjects": int(len(RESULTS))},
          open(art/"summary.json","w"), indent=2, default=str)
print("saved ->", art)

if HAVE_MPL and "twfb_leaky" in RESULTS and "twfb_honest_bacc" in RESULTS:
    fig, ax = plt.subplots(figsize=(7,4))
    means = {"shipped .m\\n(FB leaky)": RESULTS.get("shipped_fb_leaky"),
             "paper TWFB\\n(TWxFB leaky)": RESULTS.get("twfb_leaky"),
             "honest\\nnested-CV": RESULTS.get("twfb_honest_bacc")}
    labels = [k for k,v in means.items() if v is not None]
    vals = [means[k].mean()*100 for k in labels]; errs=[means[k].std()*100 for k in labels]
    ax.bar(labels, vals, yerr=errs, capsize=5, color=["#888","#c44","#4a4"])
    ax.axhline(72.21, ls="--", c="k", label="Liu Table 4 (72.21%)"); ax.axhline(50, ls=":", c="gray", label="chance")
    ax.set_ylabel("accuracy / balanced accuracy (%)"); ax.set_title("TWFB+DGFMDM: leaky vs honest"); ax.legend()
    plt.tight_layout(); plt.savefig(art/"twfb_leaky_vs_honest.png", dpi=120); plt.show()''')

md("""## TL;DR

- The shipped `TWFB_DGFMDM.m` reproduces to a **filter-bank-only, oracle** number (~60–65%), **not** the
  paper's 72.21%. The headline 72.21% requires the **time-window × filter-bank** search selected on the
  **test set** (oracle) — which this notebook reproduces in `twfb_leaky`.
- The **honest, leakage-free** ceiling (inner-CV selection of window+band on train only) is the number to
  carry into the thesis as the real Riemannian target — and the bar S-JEPA must clear.
- Most subjects sit near chance; a minority are strongly decodable. Report both the all-50 number and the
  decodable-subset number, and use the per-subject honest bal.acc as the anchor for Wilcoxon comparisons.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name":"Python 3 (.venv)","language":"python","name":"python3"},
                   "language_info": {"name":"python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = "/home/vadimegorov/eeg/eeg-jepa-research/src/liu2024/liu2024_twfb_dgfmdm_timewindow_faithful.ipynb"
json.dump(nb, open(out,"w"), indent=1)
print("wrote", out, "cells:", len(cells))
