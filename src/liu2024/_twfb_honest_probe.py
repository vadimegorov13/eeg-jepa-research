"""Honest leakage-free TWFB+DGFMDM ceiling on all 50 subjects:
inner-CV selection of (time-window, band) on TRAIN only, eval on held-out test.
Reports balanced accuracy. n_repeats=5 for speed."""
import time, glob, os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, iirnotch, lfilter
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from pyriemann.classification import FgMDM

FS=500
BANDS=[(8,12),(8,20),(8,30),(12,20),(15,20),(15,30),(20,30),(8,15)]
WIN_STARTS=[0.0,0.5,1.0,1.5,2.0,2.5,3.0]; WIN_LEN=1.0
CH=list(range(0,17))+list(range(18,30)); NREP=5; NINNER=3
DATA=sorted(glob.glob('/home/vadimegorov/eeg/Liu2024_matlab_code/sourcedata/sub-*/sub-*_eeg.mat'))
wo=50/(FS/2); BN,AN=iirnotch(wo,wo/6.0)
BAND={bd:butter(2,[bd[0]/(FS/2),bd[1]/(FS/2)],btype='band') for bd in BANDS}

def load(path):
    m=sio.loadmat(path); e=m['eeg'][0,0]
    raw=e['rawdata'].astype(np.float64); lab=e['label'].ravel().astype(int)
    mark=raw[:,32,:]; lo,hi=800,1300; first=[]
    for t in range(raw.shape[0]):
        idx=np.where(mark[t]==2)[0]; ir=idx[(idx>=lo)&(idx<=hi)]
        first.append(int(ir[0]) if ir.size else (int(idx[0]) if idx.size else -1))
    pl=[o for o in first if lo<=o<=hi]; med=int(np.median(pl)) if pl else 1003
    return raw,lab,[o if lo<=o<=hi else med for o in first]

def seg_band(raw,ons,bd):
    b,a=BAND[bd]; n=raw.shape[0]; out=np.zeros((n,29,2000))
    for t in range(n):
        s0=ons[t]-800; seg=raw[t][:,s0:s0+2800][CH,:]
        seg=lfilter(BN,AN,seg,axis=1); seg=lfilter(b,a,seg,axis=1); out[t]=seg[:,800:2800]
    return out

def cov_win(seg,ws):
    s=int(round(ws*FS)); e=s+int(round(WIN_LEN*FS)); n=seg.shape[0]; C=np.zeros((n,29,29))
    for t in range(n):
        X=seg[t][:,s:e]; c=X@X.T; c=c/np.trace(c); c=0.90*c+0.10*np.eye(29)/29; C[t]=c
    return C

def acc(Ctr,ytr,Cte,yte,bacc=False):
    try:
        clf=FgMDM(metric='riemann'); clf.fit(Ctr,ytr); p=clf.predict(Cte)
        return balanced_accuracy_score(yte,p) if bacc else float(np.mean(p==yte))
    except Exception:
        return np.nan

t0=time.time(); res=[]
for path in DATA:
    sid=int(os.path.basename(path).split('-')[1][:2]); raw,y,ons=load(path); n=raw.shape[0]; idx=np.arange(n)
    combos={(bd,ws):cov_win(seg_band(raw,ons,bd),ws) for bd in BANDS for ws in WIN_STARTS}
    sss=StratifiedShuffleSplit(n_splits=NREP,test_size=0.4,random_state=2026); spl=list(sss.split(idx,y))
    baccs=[]
    for tr,te in spl:
        skf=StratifiedKFold(n_splits=NINNER,shuffle=True,random_state=0); best=None;bv=-np.inf
        for k in combos:
            iv=[acc(combos[k][tr[a]],y[tr[a]],combos[k][tr[b]],y[tr[b]]) for a,b in skf.split(tr,y[tr])]
            mv=np.nanmean(iv)
            if mv>bv: bv,best=mv,k
        baccs.append(acc(combos[best][tr],y[tr],combos[best][te],y[te],bacc=True))
    res.append((sid,np.nanmean(baccs)))
    print(f'sub-{sid:02d}: honest_bacc={res[-1][1]*100:5.1f}  ({time.time()-t0:5.1f}s)',flush=True)
b=np.array([r[1] for r in res])
print('\n==== HONEST nested TWFB+DGFMDM, n=%d ===='%len(res))
print('balanced accuracy: %.2f%% ± %.2f'%(np.nanmean(b)*100,np.nanstd(b)*100))
print('subjects > 60%%: %d/%d ; > 70%%: %d/%d'%((b>0.60).sum(),len(b),(b>0.70).sum(),len(b)))
top=sorted(res,key=lambda r:-r[1])[:8]
print('top decodable:', ', '.join(f'sub{s:02d}={v*100:.0f}' for s,v in top))
