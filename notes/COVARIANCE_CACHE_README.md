# Covariance Cache — format and how to use it

> **Archived format note.** The covariance caches were removed during maximum cleanup and the TWFB
> protocols that consumed them are CLOSED. Do not rebuild caches or rerun those grids; see
> workspace-root `AGENTS.md` section 2e. Examples below document the historical file format only.

The notebook `liu2024_twfb_dgfmdm_faithful_v2.ipynb` caches the **per-trial covariance matrices** (the expensive
part of the pipeline) so every protocol — and any other notebook or collaborator — can reuse them instantly
instead of re-filtering the raw EEG. This document explains the layout, how to load it, and how to control it.

---

## Why cache covariances

Filtering each trial into 8 bands × (1 full window + 7 sub-windows) and forming scatter covariances is the slow
step. The classification protocols (leaky, honest, fixed) only consume these covariances. Historically,
caching avoided repeated filtering while auditing those protocols. It must not now be used to
reconstruct completed grids merely because the cache was deleted.

---

## Where it lives

```
{covariance_cache_dir}/
  cov_{signature}/
    cache_manifest.json
    sub-01_cov.npz
    sub-02_cov.npz
    ...
    sub-50_cov.npz
```

- `covariance_cache_dir` is a CONFIG path (default `artifacts/covariance_cache`).
- `{signature}` is a 10-char md5 hash of **every covariance-affecting setting**: channels, onset/window,
  notch + Butterworth order + filter phase, the band list, the time-window grid, and the covariance
  regularization (`cov_trace_normalize`, `cov_shrinkage`). Change any of those and you get a *different* folder —
  so incompatible caches never silently mix. Identical config → identical signature → the same reusable cache.

---

## What's inside each `sub-XX_cov.npz`

A flat set of named arrays:

| Key | Shape | Meaning |
|---|---|---|
| `fb__{lo}_{hi}` | `(n_trials, 29, 29)` | covariance over the **full 0–4 s** window for band `[lo,hi]` |
| `tw__{lo}_{hi}__{wstart}` | `(n_trials, 29, 29)` | covariance over the **1-s window** starting at `{wstart}` s, band `[lo,hi]` |
| `y` | `(n_trials,)` | labels (0 = left, 1 = right) |
| `onsets` | `(n_trials,)` | per-trial MI onset sample |
| `_cov_sig` | scalar string | the signature, for a sanity check on load |

Example keys with the default 8 bands × 7 windows: `fb__8_12`, `fb__8_30`, …, `tw__8_12__0.0`, `tw__8_12__0.5`,
…, `tw__20_30__3.0`. The covariance matrices are **regularized SPD** (trace-normalized + shrinkage by default).

`cache_manifest.json` (written once per folder) records the signature, the full config subset that produced the
cache, the band/window lists, and the key scheme — so the folder is self-describing and shareable.

---

## Historical control — `covariance_cache_mode`

Set in the CONFIG cell:

| Mode | Behaviour |
|---|---|
| `auto` (default) | use the cache if a compatible one exists; otherwise compute and **save** it |
| `readonly` | **must** load from cache; raises if a subject's file is missing (no computing) |
| `rebuild` | always recompute and **overwrite** the cache (use after changing a covariance setting, or to refresh) |
| `off` | always compute, never read or write |

These modes describe the retained implementation. There is no current cache-building workflow; the
completed TWFB experiments and deleted caches must not be regenerated without a new prespecification.

---

## Historical loading example

```python
import numpy as np

z = np.load("artifacts/covariance_cache/cov_ab12cd34ef/sub-07_cov.npz")

cov_mu_full   = z["fb__8_12"]        # (40, 29, 29) full 0-4 s covariance, mu band
cov_mu_win2   = z["tw__8_12__2.0"]   # (40, 29, 29) 1-s window starting at 2.0 s, mu band
y             = z["y"]               # (40,) labels
onsets        = z["onsets"]

# e.g. classify with pyriemann directly:
from pyriemann.classification import FgMDM
clf = FgMDM(metric="riemann").fit(cov_mu_full[:24], y[:24])
pred = clf.predict(cov_mu_full[24:])
```

To enumerate everything available:
```python
fb_keys = [k for k in z.files if k.startswith("fb__")]   # full-window views
tw_keys = [k for k in z.files if k.startswith("tw__")]   # time-window views
```

---

## Reusing the cache from another notebook

Point that notebook's `covariance_cache_dir` at the same folder and set `covariance_cache_mode="readonly"`, **and
keep the covariance-affecting CONFIG identical** (so the signature matches). If the signature differs, you'll get a
new `cov_{signature}` folder rather than a silent mismatch. The `_cov_sig` check inside each `.npz` will also warn
if a file was moved into the wrong folder.

---

## Sharing with collaborators

Zip the whole `cov_{signature}/` folder (npz files + `cache_manifest.json`) and share it. The manifest documents
exactly how the covariances were built, so anyone can load and use them without the raw EEG or the filtering code.

---

## Caveats

- The covariances are **regularized** (trace-normalize + 0.1 shrinkage) by default. For a bit-for-bit raw-scatter
  cache, set `cov_trace_normalize=False` and `cov_shrinkage=0.0` — that produces a *different signature/folder*.
- Covariances are tied to the onset detection and 0–4 s window; if you change onset settings, rebuild.
- Arrays are stored uncompressed-per-key inside a compressed npz; ~tens of MB per subject for 8×8 views.
