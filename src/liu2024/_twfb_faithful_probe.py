"""
Faithful Python port of Liu2024 TWFB_DGFMDM.m, run as a quick probe to verify
the reproduction. Mirrors the MATLAB exactly:
  - channels [1:17 19:30] (0-based [0..16,18..29]) = 29 ch
  - per-trial onset = first sample where marker(ch33==index32)==2
  - window = onset-800 : onset+2000 (2800 samp), notch(50,Q6,one-pass) + butter(order2,one-pass) bandpass,
    then keep samples 800:2800 -> 2000 samp = 0..4s post onset
  - covariance = SS.T @ SS  (SS is 2000x29), unnormalized scatter matrix
  - classifier = FgMDM (riemann/riemann) == matlab fgmdm
  - 8 bands {[8,12],[8,20],[8,30],[12,20],[15,20],[15,30],[20,30],[8,15]}

Protocols:
  A) MATLAB-faithful (LEAKY): 10 repeats; for EACH band a fresh random 24/16 split;
     report max test-acc over bands (oracle). This is what produces ~72%.
  B) HONEST nested: 10 stratified 60/40 splits; band chosen by 3-fold inner CV on TRAIN only.
  C) Per-band fixed (no selection): mean test acc per band.
"""
import sys, time, glob, os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, iirnotch, lfilter
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from pyriemann.classification import FgMDM

FS = 500
BANDS = [(8,12),(8,20),(8,30),(12,20),(15,20),(15,30),(20,30),(8,15)]
CH = list(range(0,17)) + list(range(18,30))   # 29 channels, drop index 17 (CPz)
DATA = sorted(glob.glob('/home/vadimegorov/eeg/Liu2024_matlab_code/sourcedata/sub-*/sub-*_eeg.mat'))

def make_filters():
    wo = 50/(FS/2)
    bw = wo/6.0
    bn, an = iirnotch(wo, wo/bw)   # Q = wo/bw = 6
    band = {}
    for (f1,f2) in BANDS:
        b,a = butter(2, [f1/(FS/2), f2/(FS/2)], btype='band')
        band[(f1,f2)] = (b,a)
    return (bn,an), band

def load_subject(path):
    m = sio.loadmat(path); eeg = m['eeg'][0,0]
    raw = eeg['rawdata'].astype(np.float64)   # (40,33,4000)
    lab = eeg['label'].ravel().astype(int)    # (40,)
    mark = raw[:,32,:]
    onsets = []
    for t in range(raw.shape[0]):
        idx = np.where(mark[t]==2)[0]
        onsets.append(int(idx[0]) if len(idx) else 1003)
    return raw, lab, onsets

def covs_for_band(raw, onsets, notchf, bf):
    bn,an = notchf; b,a = bf
    n = raw.shape[0]
    C = np.zeros((n,29,29))
    for t in range(n):
        on = onsets[t]
        s0 = on-800
        seg = raw[t][:, s0:s0+2800][CH,:]          # (29,2800)
        seg = lfilter(bn,an,seg,axis=1)            # notch, one-pass
        seg = lfilter(b,a,seg,axis=1)              # band, one-pass
        seg = seg[:, 800:2800]                     # (29,2000) -> 0..4s
        SS = seg.T                                 # (2000,29)
        cov = SS.T @ SS                            # (29,29) scatter
        cov += 1e-3*np.trace(cov)/29*np.eye(29)    # tiny ridge for SPD safety
        C[t] = cov
    return C

def fgmdm_acc(Ctr, ytr, Cte, yte):
    clf = FgMDM(metric='riemann')
    clf.fit(Ctr, ytr)
    pred = clf.predict(Cte)
    return np.mean(pred==yte)

def run():
    notchf, bandf = make_filters()
    rng = np.random.RandomState(2026)
    leaky, honest = [], []
    perband = {bd:[] for bd in BANDS}
    t0=time.time()
    for path in DATA:
        sid = os.path.basename(path)[:6]
        raw, lab, onsets = load_subject(path)
        # precompute covariances per band once
        bandcovs = {bd: covs_for_band(raw,onsets,notchf,bandf[bd]) for bd in BANDS}
        n = raw.shape[0]; idx_all = np.arange(n)
        # ---- A) leaky oracle (matlab) ----
        sub_leaky=[]
        for h in range(10):
            accs=[]
            for bd in BANDS:
                perm = rng.permutation(n); tr=perm[:24]; te=perm[24:40]
                a = fgmdm_acc(bandcovs[bd][tr], lab[tr], bandcovs[bd][te], lab[te])
                accs.append(a)
            sub_leaky.append(max(accs))
        leaky.append(np.mean(sub_leaky))
        # ---- C) per-band fixed, honest splits ----
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.40, random_state=2026)
        splits = list(sss.split(idx_all, lab))
        for bd in BANDS:
            accs=[fgmdm_acc(bandcovs[bd][tr],lab[tr],bandcovs[bd][te],lab[te]) for tr,te in splits]
            perband[bd].append(np.mean(accs))
        # ---- B) honest nested band selection ----
        sub_h=[]
        for tr,te in splits:
            # inner CV on train to pick band
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
            best_bd, best_acc = BANDS[0], -1
            for bd in BANDS:
                inner=[]
                for itr,iva in skf.split(tr, lab[tr]):
                    a=fgmdm_acc(bandcovs[bd][tr[itr]],lab[tr[itr]],bandcovs[bd][tr[iva]],lab[tr[iva]])
                    inner.append(a)
                mi=np.mean(inner)
                if mi>best_acc: best_acc, best_bd = mi, bd
            a=fgmdm_acc(bandcovs[best_bd][tr],lab[tr],bandcovs[best_bd][te],lab[te])
            sub_h.append(a)
        honest.append(np.mean(sub_h))
        print(f'{sid}: leaky_oracle={leaky[-1]*100:5.1f}  honest_nested={honest[-1]*100:5.1f}   ({time.time()-t0:5.1f}s)',flush=True)
    print('\n==== SUMMARY (n=%d subjects) ===='%len(DATA))
    print('A) MATLAB leaky oracle :  %.2f%% ± %.2f'%(np.mean(leaky)*100, np.std(leaky)*100))
    print('B) Honest nested-CV    :  %.2f%% ± %.2f'%(np.mean(honest)*100, np.std(honest)*100))
    print('C) Per-band fixed (honest 60/40):')
    for bd in BANDS:
        print('   %-8s : %.2f%%'%(str(bd), np.mean(perband[bd])*100))

if __name__=='__main__':
    run()
