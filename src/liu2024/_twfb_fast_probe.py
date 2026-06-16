"""Fast probe: faithful MATLAB TWFB_DGFMDM, LEAKY-oracle + per-band only (no nested).
Decisive question: does the faithful Python port reproduce Liu2024's ~72% under the
MATLAB's own (leaky) max-over-bands protocol?  Per-band-fixed gives the honest floor.
"""
import time, glob, os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, iirnotch, lfilter
from sklearn.model_selection import StratifiedShuffleSplit
from pyriemann.classification import FgMDM

FS=500
BANDS=[(8,12),(8,20),(8,30),(12,20),(15,20),(15,30),(20,30),(8,15)]
CH=list(range(0,17))+list(range(18,30))
DATA=sorted(glob.glob('/home/vadimegorov/eeg/Liu2024_matlab_code/sourcedata/sub-*/sub-*_eeg.mat'))

wo=50/(FS/2); bw=wo/6.0
BN,AN=iirnotch(wo,wo/bw)
BAND={bd:butter(2,[bd[0]/(FS/2),bd[1]/(FS/2)],btype='band') for bd in BANDS}

def load(path):
    m=sio.loadmat(path); e=m['eeg'][0,0]
    raw=e['rawdata'].astype(np.float64); lab=e['label'].ravel().astype(int)
    mark=raw[:,32,:]; lo,hi=800,1300; first=[]
    for t in range(raw.shape[0]):
        idx=np.where(mark[t]==2)[0]; ir=idx[(idx>=lo)&(idx<=hi)]
        first.append(int(ir[0]) if ir.size else (int(idx[0]) if idx.size else -1))
    pl=[o for o in first if lo<=o<=hi]; med=int(np.median(pl)) if pl else 1003
    ons=[o if lo<=o<=hi else med for o in first]
    return raw,lab,ons

def covs(raw,ons,bd):
    b,a=BAND[bd]; n=raw.shape[0]; C=np.zeros((n,29,29))
    for t in range(n):
        s0=ons[t]-800
        seg=raw[t][:,s0:s0+2800][CH,:]
        seg=lfilter(BN,AN,seg,axis=1); seg=lfilter(b,a,seg,axis=1)
        seg=seg[:,800:2800]; SS=seg.T; c=SS.T@SS
        c=c/np.trace(c); c=0.90*c+0.10*np.eye(29)/29; C[t]=c
    return C

def acc(Ctr,ytr,Cte,yte):
    try:
        clf=FgMDM(metric='riemann'); clf.fit(Ctr,ytr); return float(np.mean(clf.predict(Cte)==yte))
    except Exception:
        return np.nan

t0=time.time(); leaky=[]; perband={bd:[] for bd in BANDS}
rng=np.random.RandomState(2026)
for path in DATA:
    sid=os.path.basename(path)[:6]; raw,lab,ons=load(path)
    bc={bd:covs(raw,ons,bd) for bd in BANDS}
    n=raw.shape[0]; idx=np.arange(n)
    # leaky oracle (matlab): per band fresh random 24/16 split, max over bands
    sub=[]
    for h in range(10):
        a=[acc(bc[bd][p[:24]],lab[p[:24]],bc[bd][p[24:40]],lab[p[24:40]]) for bd in BANDS for p in [rng.permutation(n)]]
        sub.append(np.nanmax(a))
    leaky.append(np.mean(sub))
    # per-band fixed, honest stratified 60/40 x10
    sss=StratifiedShuffleSplit(n_splits=10,test_size=0.4,random_state=2026); spl=list(sss.split(idx,lab))
    for bd in BANDS:
        perband[bd].append(np.nanmean([acc(bc[bd][tr],lab[tr],bc[bd][te],lab[te]) for tr,te in spl]))
    print(f'{sid}: leaky={leaky[-1]*100:5.1f}  best_fixed_band={max(np.mean(perband[bd][-1:]) for bd in BANDS)*100:5.1f}  ({time.time()-t0:5.1f}s)',flush=True)

print('\n==== SUMMARY n=%d ===='%len(DATA))
print('A) MATLAB leaky oracle (max-over-8-bands): %.2f%% ± %.2f'%(np.nanmean(leaky)*100,np.nanstd(leaky)*100))
print('C) Per-band fixed (honest 60/40 x10):')
for bd in BANDS: print('   %-8s : %.2f%%'%(str(bd),np.nanmean(perband[bd])*100))
best=max(BANDS,key=lambda bd:np.mean(perband[bd]))
print('   best single fixed band: %s = %.2f%%'%(str(best),np.mean(perband[best])*100))
# honest "fixed-best-band-per-subject by TRAIN, naive" approximation = mean over subjects of their best fixed band (still mildly optimistic)
subj_best=[max(perband[bd][i] for bd in BANDS) for i in range(len(DATA))]
print('   mean of per-subject best fixed band (optimistic): %.2f%%'%(np.mean(subj_best)*100))
