"""Generate liu2024_sjepa_twfb_hybrid.ipynb — the novel TWFB-guided hybrid.

Branch A: frozen S-JEPA pre-local RICH embeddings (forward hook on feature_encoder).
Branch B: multi-band Riemannian tangent-space features at a per-subject/fold TWFB-selected TIME WINDOW
          (inner-CV on train only) — the differentiator vs the fixed-window hybrid.
Fusion : standardize each block (train-fit) -> concat -> shrinkage-LDA (train-fit).
Eval   : within-subject 10x StratifiedShuffleSplit 60/40 (project standard); Wilcoxon vs branch-only.
Ablations (CONFIG): branch {sjepa,riemann,fusion} x personalize {twfb,fixed} x sjepa_init {pretrained,random}.
Leakage: every learnable step (tangent ref, scalers, LDA, window selection) fit on TRAIN only.
"""
import json
cells = []
def md(s): cells.append({"cell_type":"markdown","metadata":{},"source":s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.rstrip("\n").splitlines(keepends=True)})

md("""# Liu2024 — S-JEPA × TWFB Riemannian hybrid (TWFB-guided personalization)

A geometry-anchored, predictively-regularized decoder for Liu2024 left/right MI. Two branches, fused with a
shrinkage-LDA; everything fold-safe.

- **Branch A — S-JEPA (predictive representation).** Frozen `SignalJEPA_PreLocal` pretrained encoder; a forward
  hook on `feature_encoder` yields a **rich** per-trial embedding (token tensor pooled to a fixed vector),
  *not* the 2-D logits.
- **Branch B — TWFB Riemannian (geometry).** Per fold, an **inner-CV on train** selects the most discriminative
  **time window** (within 0–4 s after MI onset); the feature is the multi-band spatial-covariance
  **tangent-space** vector at that window. This per-subject time personalization is the novelty over the
  existing fixed-window hybrid.
- **Fusion.** Standardize each block (fit on train) → concatenate → shrinkage-LDA (fit on train).

Honest framing (from this project's TWFB analysis): the Riemannian/geometry branch carries the
classification; S-JEPA is the predictive-representation branch. We report fusion **vs** each branch alone
(Wilcoxon on per-subject balanced accuracy) and the `twfb` vs `fixed` ablation, so the contribution of
time personalization and of S-JEPA are both measured, not assumed.
""")

md("# 1. Setup")
code('''import os, sys, glob, json, time, copy, platform
from pathlib import Path
import numpy as np, pandas as pd
import scipy.io as sio
from scipy.signal import butter, iirnotch, lfilter
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
import mne; mne.set_log_level("ERROR")
import torch
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
try:
    import matplotlib.pyplot as plt; HAVE_MPL=True
except Exception: HAVE_MPL=False
from braindecode.models import SignalJEPA_PreLocal
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())''')

md("# 2. Configuration")
code('''WORKING_DIR = Path.cwd().resolve().parent.parent
CONFIG = {
    "experiment_name": "sjepa_twfb_hybrid",
    "source_roots": [
        str(WORKING_DIR.parent / "Liu2024_matlab_code" / "sourcedata"),
        str(WORKING_DIR / "liu2024_data" / "liu2024_figshare" / "sourcedata"),
    ],
    "artifact_dir": str(WORKING_DIR / "artifacts" / "liu2024-sjepa-twfb-hybrid"),
    "subjects_to_use": None,                 # None=all; or e.g. [1,2,7]

    # ---- channels / onset (faithful to TWFB) ----
    "channel_indices": list(range(0,17))+list(range(18,30)),   # 29 ch, drop CPz
    "marker_channel_index": 32, "onset_marker_value": 2,
    "onset_plausible_range": [800,1300], "onset_fallback_sample": 1003,
    "native_sfreq": 500, "preroll_samples": 800, "mi_segment_samples": 2000,   # 0-4 s @500

    # ---- Branch A: S-JEPA ----
    "sjepa_repo_id": "braindecode/signal-jepa_without-chans",
    "sjepa_checkpoint_path": None,
    "sjepa_init": "pretrained",              # 'pretrained' | 'random'  (ablation)
    "sjepa_sfreq": 128, "sjepa_window_start_s": 1.5, "sjepa_window_samples": 537,
    "sjepa_bandpass_hz": (0.5,40.0),
    "embedding_hook": "feature_encoder", "embedding_pool": "mean",
    "sjepa_embedding_cache_dir": str(WORKING_DIR / "artifacts" / "liu2024_sjepa_embeddings_lda" / "embeddings"),

    # ---- Branch B: TWFB Riemannian ----
    "freq_bands": [(8,12),(8,20),(8,30),(12,20),(15,20),(15,30),(20,30),(8,15)],
    "notch_freq": 50.0, "notch_Q": 6.0, "butter_order": 2,
    "tw_starts_s": [0.0,1.0,2.0], "tw_len_s": 2.0,        # candidate windows for 'twfb' selection
    "fixed_window_s": (0.0,4.0),                          # window used by 'fixed' control
    "cov_shrinkage": 0.10, "tangent_metric": "logeuclid",

    # ---- method / ablations ----
    "personalize": "twfb",                   # 'twfb' (select time window, inner-CV) | 'fixed'
    "run_branches": ["sjepa","riemann","fusion"],
    "classifier": "shrinkage_lda",

    # ---- eval ----
    "pca_components": 30,                     # train-fit PCA per branch before LDA (keeps fits fast & regularized)
    "n_repeats": 10, "test_size": 0.40, "inner_folds": 3, "random_state": 2026,
    "device": "auto",
}
BANDS=[tuple(b) for b in CONFIG["freq_bands"]]; FS=CONFIG["native_sfreq"]
CH=CONFIG["channel_indices"]; NCH=len(CH)
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if CONFIG["device"]=="auto" else CONFIG["device"]
np.random.seed(CONFIG["random_state"]); torch.manual_seed(CONFIG["random_state"])
print(f"device={DEVICE} channels={NCH} bands={len(BANDS)} tw_starts={CONFIG['tw_starts_s']} personalize={CONFIG['personalize']}")''')

md("# 3. Data: loaders for both branches")
code('''def find_files(roots):
    for r in roots:
        f=sorted(glob.glob(os.path.join(r,"sub-*","sub-*_eeg.mat")))
        if f: print("source:",r,f"({len(f)} subjects)"); return f
    raise FileNotFoundError(roots)

def load_raw(path):
    """rawdata (40,33,4000), labels 0/1, robust per-trial onset sample."""
    m=sio.loadmat(path); eeg=m["eeg"][0,0]
    raw=np.asarray(eeg["rawdata"],dtype=np.float64); lab=np.asarray(eeg["label"]).ravel().astype(int)
    if set(np.unique(lab)).issubset({1,2}): lab=lab-1
    mark=raw[:,CONFIG["marker_channel_index"],:]; lo,hi=CONFIG["onset_plausible_range"]
    first=[]
    for t in range(raw.shape[0]):
        idx=np.where(mark[t]==CONFIG["onset_marker_value"])[0]; ir=idx[(idx>=lo)&(idx<=hi)]
        first.append(int(ir[0]) if ir.size else (int(idx[0]) if idx.size else -1))
    pl=[o for o in first if lo<=o<=hi]; med=int(np.median(pl)) if pl else CONFIG["onset_fallback_sample"]
    onsets=[o if lo<=o<=hi else med for o in first]
    return raw,lab,onsets

# ---- Branch A preprocessing: MNE avg-ref + resample 128 + bandpass, onset-aligned window ----
EEG_NAMES=["Fp1","Fp2","Fz","F3","F4","F7","F8","FCz","FC3","FC4","FT7","FT8","Cz","C3","C4","T3","T4",
           "CP3","CP4","TP7","TP8","Pz","P3","P4","T5","T6","Oz","O1","O2"]   # 29, CPz dropped
def sjepa_windows(raw, onsets):
    """(40,29,window_samples) at sjepa_sfreq, onset-aligned (start = onset + (window_start - 2.0 s))."""
    sf=CONFIG["sjepa_sfreq"]; W=CONFIG["sjepa_window_samples"]
    eeg=raw[:,CH,:]*1e-6
    n_tr,_,n_t=eeg.shape
    cont=eeg.transpose(1,0,2).reshape(NCH, n_tr*n_t)
    info=mne.create_info(EEG_NAMES, FS, ["eeg"]*NCH)
    r=mne.io.RawArray(cont, info); r.set_eeg_reference("average", projection=False)
    r.resample(sf); lo,hi=CONFIG["sjepa_bandpass_hz"]; r.filter(lo,hi,method="fir",phase="zero")
    d=r.get_data()*1e6; nt2=int(round(n_t*sf/FS)); d=d.reshape(NCH,n_tr,nt2).transpose(1,0,2)
    # onset is ~2.0 s into the 8 s trial; align window start to that, shifted by (start_s - 2.0)
    start=int(round(CONFIG["sjepa_window_start_s"]*sf))
    start=max(0,min(start, nt2-W))
    return d[:,:,start:start+W].astype(np.float32)

# ---- Branch B: native-500 filtered 0-4 s MI segment per band, then covariance over a sub-window ----
_wo=CONFIG["notch_freq"]/(FS/2); NB,NA=iirnotch(_wo,_wo/CONFIG["notch_Q"])
BBA={bd:butter(CONFIG["butter_order"],[bd[0]/(FS/2),bd[1]/(FS/2)],btype="band") for bd in BANDS}
def band_segment(raw,onsets,bd):
    b,a=BBA[bd]; pre=CONFIG["preroll_samples"]; L=CONFIG["mi_segment_samples"]; n=raw.shape[0]
    out=np.zeros((n,NCH,L))
    for t in range(n):
        s0=onsets[t]-pre; seg=raw[t][:,s0:s0+pre+L][CH,:]
        seg=lfilter(NB,NA,seg,axis=1); seg=lfilter(b,a,seg,axis=1); out[t]=seg[:,pre:pre+L]
    return out
def covs_window(seg, w_start_s, w_len_s):
    s=int(round(w_start_s*FS)); e=s+int(round(w_len_s*FS)); n=seg.shape[0]; C=np.zeros((n,NCH,NCH))
    g=CONFIG["cov_shrinkage"]
    for t in range(n):
        X=seg[t][:,s:e]; c=X@X.T; c=c/np.trace(c); c=(1-g)*c+g*np.eye(NCH)/NCH; C[t]=c
    return C
print("data loaders defined")''')

md("# 4. Branch A — S-JEPA model + rich embeddings")
code('''def build_sjepa(n_times):
    kw=dict(n_chans=NCH, chs_info=None, n_times=n_times, n_outputs=2)
    if CONFIG["sjepa_init"]=="random":
        m=SignalJEPA_PreLocal(**kw)
    elif CONFIG["sjepa_checkpoint_path"]:
        m=SignalJEPA_PreLocal(**kw); m.load_state_dict(torch.load(CONFIG["sjepa_checkpoint_path"],map_location="cpu"),strict=False)
    else:
        m=SignalJEPA_PreLocal.from_pretrained(CONFIG["sjepa_repo_id"], **kw, strict=False)
    for p in m.parameters(): p.requires_grad=False
    return m.to(DEVICE).eval()

@torch.no_grad()
def rich_embeddings(model, X):
    pool=CONFIG["embedding_pool"]; hook=CONFIG["embedding_hook"]
    target=getattr(model,hook); cap={}
    h=target.register_forward_hook(lambda mod,i,o: cap.__setitem__("z",o.detach()))
    out=[]
    try:
        xb=torch.from_numpy(np.asarray(X,dtype=np.float32))
        for i in range(0,len(xb),32):
            _=model(xb[i:i+32].to(DEVICE)); z=cap["z"]
            if z.dim()==2: f=z
            else:
                ax=2 if hook=="spatial_conv" else 1
                f = z.flatten(1) if pool=="flatten" else (z.max(ax).values if pool=="max" else z.mean(ax)).flatten(1)
            out.append(f.cpu().numpy())
    finally: h.remove()
    return np.concatenate(out,0).astype(np.float32)

_SJEPA=[None]
def get_sjepa_features(sid, raw, onsets):
    """(40,D) frozen S-JEPA embeddings. Uses cache if present (pretrained only)."""
    cache=Path(CONFIG["sjepa_embedding_cache_dir"])/f"sub-{sid:02d}.npz"
    if CONFIG["sjepa_init"]=="pretrained" and cache.exists():
        d=np.load(cache); return d["X"].astype(np.float32)
    if _SJEPA[0] is None: _SJEPA[0]=build_sjepa(CONFIG["sjepa_window_samples"])
    Xw=sjepa_windows(raw,onsets)
    return rich_embeddings(_SJEPA[0], Xw)
print("S-JEPA branch defined")''')

md("# 5. Branch B — fold-safe tangent features + TWFB window selection")
code('''from sklearn.decomposition import PCA

def tangent_block(cov_by_band, tr, te):
    """Per-band TangentSpace fit on TRAIN; concat band vectors. cov_by_band: {bd:(n,ch,ch)}."""
    Ftr,Fte=[],[]
    for bd in BANDS:
        ts=TangentSpace(metric=CONFIG["tangent_metric"]); ts.fit(cov_by_band[bd][tr])
        Ftr.append(ts.transform(cov_by_band[bd][tr])); Fte.append(ts.transform(cov_by_band[bd][te]))
    return np.concatenate(Ftr,1), np.concatenate(Fte,1)

def featurize(Ftr, Fte):
    """StandardScaler + train-fit PCA (compress high-dim tangent so LDA stays fast & regularized)."""
    sc=StandardScaler().fit(Ftr); Ftr=sc.transform(Ftr); Fte=sc.transform(Fte)
    nc=min(CONFIG["pca_components"], Ftr.shape[1], max(1, Ftr.shape[0]-1))
    if Ftr.shape[1] > nc:
        pca=PCA(n_components=nc, random_state=0).fit(Ftr)
        Ftr, Fte = pca.transform(Ftr), pca.transform(Fte)
    return Ftr, Fte

def make_lda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")

def select_time_window(seg_by_band, y, tr, rng_state):
    """Inner-CV on TRAIN only: pick the time-window start maximising multi-band tangent+LDA bal-acc."""
    starts=CONFIG["tw_starts_s"]; wlen=CONFIG["tw_len_s"]
    skf=StratifiedKFold(n_splits=CONFIG["inner_folds"], shuffle=True, random_state=rng_state)
    best_s, best_v = starts[0], -np.inf
    for s in starts:
        covb={bd:covs_window(seg_by_band[bd][tr], s, wlen) for bd in BANDS}
        accs=[]
        for ia,ib in skf.split(np.arange(len(tr)), y[tr]):
            Ftr,Fva=featurize(*tangent_block(covb, ia, ib))
            clf=make_lda().fit(Ftr, y[tr][ia])
            accs.append(balanced_accuracy_score(y[tr][ib], clf.predict(Fva)))
        if np.mean(accs)>best_v: best_v, best_s = np.mean(accs), s
    return best_s
print("Riemannian branch defined")''')

md("# 6. Hybrid per-subject CV runner")
code('''def collapse_flag(pred):
    c=np.bincount(pred,minlength=2); return bool(c.max()/c.sum()>0.95)

def run_subject(sid, raw, y, onsets):
    seg_by_band={bd:band_segment(raw,onsets,bd) for bd in BANDS}     # 500Hz filtered 0-4s
    E=get_sjepa_features(sid, raw, onsets)                           # (40,D)
    n=len(y); idx=np.arange(n)
    sss=StratifiedShuffleSplit(n_splits=CONFIG["n_repeats"], test_size=CONFIG["test_size"],
                               random_state=CONFIG["random_state"])
    rows=[]
    for fi,(tr,te) in enumerate(sss.split(idx,y)):
        # ---- choose window (fold-safe) ----
        if CONFIG["personalize"]=="twfb":
            w_s, w_len = select_time_window(seg_by_band, y, tr, CONFIG["random_state"]+fi), CONFIG["tw_len_s"]
        else:
            w_s, w_len = CONFIG["fixed_window_s"][0], CONFIG["fixed_window_s"][1]-CONFIG["fixed_window_s"][0]
        covb={bd:covs_window(seg_by_band[bd], w_s, w_len) for bd in BANDS}
        # ---- branch features (train-fit) ----
        Rtr,Rte=tangent_block(covb, tr, te)
        sR=StandardScaler().fit(Rtr); Rtr,Rte=sR.transform(Rtr),sR.transform(Rte)
        sE=StandardScaler().fit(E[tr]); Etr,Ete=sE.transform(E[tr]),sE.transform(E[te])
        feats={"riemann":(Rtr,Rte), "sjepa":(Etr,Ete),
               "fusion":(np.concatenate([Rtr,Etr],1), np.concatenate([Rte,Ete],1))}
        row={"subject":sid,"fold":fi,"window_start_s":float(w_s)}
        for br in CONFIG["run_branches"]:
            Ftr,Fte=feats[br]; pred=make_lda().fit(Ftr,y[tr]).predict(Fte)
            row[f"{br}_acc"]=accuracy_score(y[te],pred)
            row[f"{br}_bacc"]=balanced_accuracy_score(y[te],pred)
            row[f"{br}_collapse"]=collapse_flag(pred)
        rows.append(row)
    return rows

def run_all():
    files=find_files(CONFIG["source_roots"]); keep=CONFIG["subjects_to_use"]; out=[]; t0=time.time()
    for path in files:
        sid=int(os.path.basename(path).split("-")[1][:2])
        if keep is not None and sid not in keep: continue
        raw,y,onsets=load_raw(path)
        rows=run_subject(sid,raw,y,onsets); out.extend(rows)
        msg=" ".join(f"{br}={np.mean([r[f'{br}_bacc'] for r in rows])*100:4.1f}" for br in CONFIG["run_branches"])
        print(f"sub-{sid:02d}: {msg}  ({time.time()-t0:5.1f}s)", flush=True)
    return pd.DataFrame(out)

RESULTS=run_all(); RESULTS.head()''')

md("# 7. Results, Wilcoxon, artifacts")
code('''def summary(df):
    print("="*60); print(f"S-JEPA x TWFB hybrid (personalize={CONFIG['personalize']}, init={CONFIG['sjepa_init']}) n={df['subject'].nunique()}")
    subj=df.groupby("subject").mean(numeric_only=True)
    for br in CONFIG["run_branches"]:
        print(f"  {br:8s}: bal-acc {subj[f'{br}_bacc'].mean()*100:5.2f}% ± {subj[f'{br}_bacc'].std()*100:4.2f}  "
              f"acc {subj[f'{br}_acc'].mean()*100:5.2f}%  collapse {df[f'{br}_collapse'].mean()*100:3.0f}%")
    if {"fusion","riemann"}<=set(CONFIG["run_branches"]):
        s=subj["fusion_bacc"]-subj["riemann_bacc"]
        try: w,p=wilcoxon(subj["fusion_bacc"],subj["riemann_bacc"]); print(f"  Wilcoxon fusion>riemann: dMean={s.mean()*100:+.2f}pp p={p:.4f}")
        except Exception as e: print("  wilcoxon n/a:",e)
    if {"fusion","sjepa"}<=set(CONFIG["run_branches"]):
        try: w,p=wilcoxon(subj["fusion_bacc"],subj["sjepa_bacc"]); print(f"  Wilcoxon fusion>sjepa : dMean={(subj['fusion_bacc']-subj['sjepa_bacc']).mean()*100:+.2f}pp p={p:.4f}")
        except Exception as e: print("  wilcoxon n/a:",e)
    return subj
SUBJ=summary(RESULTS)''')

code('''art=Path(CONFIG["artifact_dir"]); art.mkdir(parents=True,exist_ok=True)
RESULTS.to_csv(art/"fold_results.csv",index=False)
SUBJ.to_csv(art/"subject_summary.csv")
RESULTS[["subject","fold","window_start_s"]].to_csv(art/"selected_window.csv",index=False)
gsum={br:{"bacc_mean":float(SUBJ[f"{br}_bacc"].mean()),"bacc_std":float(SUBJ[f"{br}_bacc"].std())} for br in CONFIG["run_branches"]}
json.dump({"config":{k:(str(v) if isinstance(v,Path) else v) for k,v in CONFIG.items()},
           "summary":gsum,"n_subjects":int(RESULTS["subject"].nunique())},
          open(art/"global_summary.json","w"),indent=2,default=str)
print("saved ->",art)
if HAVE_MPL:
    fig,ax=plt.subplots(figsize=(6,4))
    brs=CONFIG["run_branches"]; m=[SUBJ[f"{b}_bacc"].mean()*100 for b in brs]; e=[SUBJ[f"{b}_bacc"].std()*100 for b in brs]
    ax.bar(brs,m,yerr=e,capsize=5,color=["#4a4","#88c","#c44"][:len(brs)])
    ax.axhline(50,ls=":",c="gray"); ax.set_ylabel("balanced accuracy (%)")
    ax.set_title(f"Hybrid ({CONFIG['personalize']}, {CONFIG['sjepa_init']})"); plt.tight_layout()
    plt.savefig(art/"branch_balacc.png",dpi=120); plt.show()''')

md("""## Notes
- **Leakage:** per fold, TWFB time-window selection (inner CV), TangentSpace reference, StandardScalers, and
  LDA are all fit on the train split only. S-JEPA embeddings are frozen/label-free (deterministic per trial).
- **Ablations:** set `personalize="fixed"` (no time personalization, full 0–4 s) and compare to `"twfb"`;
  set `sjepa_init="random"` to test whether pretraining helps; restrict `run_branches` for branch-only runs.
- **Reading it:** if `fusion` ≈ `riemann` (Wilcoxon n.s.), the honest story is "S-JEPA adds little beyond
  geometry here" — a credible, defensible thesis result, not a failure. `selected_window.csv` should vary
  across folds/subjects (proof selection is data-driven, not global).
""")

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3 (.venv)","language":"python","name":"python3"},
    "language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
out="/home/vadimegorov/eeg/eeg-jepa-research/src/liu2024/liu2024_sjepa_twfb_hybrid.ipynb"
json.dump(nb,open(out,"w"),indent=1); print("wrote",out,"cells:",len(cells))
