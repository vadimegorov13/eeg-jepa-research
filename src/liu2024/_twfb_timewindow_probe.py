"""Confirm hypothesis: the shipped TWFB_DGFMDM.m does FILTER-BANK selection only.
The paper's TWFB = Time-Window + Filter-Bank. Adding the 7 shifting time windows and
taking max-over-(window x band) on TEST (the paper's leaky protocol) should push the
~60% (band-only) up toward the paper's 72.21%.

Subset run (first N subjects, fewer reps) to confirm the trend quickly.
Reports:
  - band-only leaky (8 combos)   ~ shipped .m
  - TW+FB leaky    (7x8=56 combos) ~ paper TWFB
  - TW+FB honest nested (inner-CV select window+band on train)
"""
import time, glob, os, sys
import numpy as np
import scipy.io as sio
from scipy.signal import butter, iirnotch, lfilter
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from pyriemann.classification import FgMDM

FS=500
BANDS=[(8,12),(8,20),(8,30),(12,20),(15,20),(15,30),(20,30),(8,15)]
# 7 shifting 1-s windows within the 0-4s MI segment (start seconds), paper-style
WIN_STARTS=[0.0,0.5,1.0,1.5,2.0,2.5,3.0]; WIN_LEN=1.0
CH=list(range(0,17))+list(range(18,30))
DATA=sorted(glob.glob('/home/vadimegorov/eeg/Liu2024_matlab_code/sourcedata/sub-*/sub-*_eeg.mat'))
NSUB=int(sys.argv[1]) if len(sys.argv)>1 else 12
NREP=int(sys.argv[2]) if len(sys.argv)>2 else 10

wo=50/(FS/2); bw=wo/6.0; BN,AN=iirnotch(wo,wo/bw)
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

def band_segments(raw,ons,bd):
    """Return filtered 0-4s MI segment (n,29,2000) for a band."""
    b,a=BAND[bd]; n=raw.shape[0]; out=np.zeros((n,29,2000))
    for t in range(n):
        s0=ons[t]-800
        seg=raw[t][:,s0:s0+2800][CH,:]
        seg=lfilter(BN,AN,seg,axis=1); seg=lfilter(b,a,seg,axis=1)
        out[t]=seg[:,800:2800]
    return out

def covs_from_seg(seg2000, wstart):
    """Covariance over a 1-s sub-window. seg2000:(n,29,2000)."""
    s=int(round(wstart*FS)); e=s+int(round(WIN_LEN*FS))
    n=seg2000.shape[0]; C=np.zeros((n,29,29))
    for t in range(n):
        X=seg2000[t][:,s:e]            # (29,500)
        c=X@X.T                        # (29,29)
        c=c/np.trace(c); c=0.90*c+0.10*np.eye(29)/29; C[t]=c
    return C

def full_cov(seg2000):
    n=seg2000.shape[0]; C=np.zeros((n,29,29))
    for t in range(n):
        X=seg2000[t]; c=X@X.T; c=c/np.trace(c); c=0.90*c+0.10*np.eye(29)/29; C[t]=c
    return C

def acc(Ctr,ytr,Cte,yte):
    try:
        clf=FgMDM(metric='riemann'); clf.fit(Ctr,ytr); return float(np.mean(clf.predict(Cte)==yte))
    except Exception:
        return np.nan

t0=time.time(); band_leaky=[]; twfb_leaky=[]; twfb_nested=[]
rng=np.random.RandomState(2026)
for path in DATA[:NSUB]:
    sid=os.path.basename(path)[:6]; raw,lab,ons=load(path); n=raw.shape[0]; idx=np.arange(n)
    segs={bd:band_segments(raw,ons,bd) for bd in BANDS}
    # precompute combo covariances: band-only (full 0-4s) and TWxFB (7 windows)
    bandcov={bd:full_cov(segs[bd]) for bd in BANDS}            # 8 band-only
    combos={(bd,ws):covs_from_seg(segs[bd],ws) for bd in BANDS for ws in WIN_STARTS}  # 56
    # band-only leaky (shipped .m)
    bl=[]
    for h in range(NREP):
        a=[acc(bandcov[bd][p[:24]],lab[p[:24]],bandcov[bd][p[24:40]],lab[p[24:40]]) for bd in BANDS for p in [rng.permutation(n)]]
        bl.append(np.nanmax(a))
    band_leaky.append(np.mean(bl))
    # TW+FB leaky (paper): max over 56 combos, fresh split per combo
    tl=[]
    for h in range(NREP):
        a=[acc(combos[k][p[:24]],lab[p[:24]],combos[k][p[24:40]],lab[p[24:40]]) for k in combos for p in [rng.permutation(n)]]
        tl.append(np.nanmax(a))
    twfb_leaky.append(np.mean(tl))
    # TW+FB honest nested: inner 3-fold select combo on train only (skip if SKIP_NESTED)
    if os.environ.get('SKIP_NESTED'):
        twfb_nested.append(float('nan'))
    else:
        sss=StratifiedShuffleSplit(n_splits=NREP,test_size=0.4,random_state=2026); spl=list(sss.split(idx,lab))
        nn=[]
        for tr,te in spl:
            skf=StratifiedKFold(n_splits=3,shuffle=True,random_state=0); best=None;bestv=-1
            for k in combos:
                iv=[acc(combos[k][tr[a]],lab[tr[a]],combos[k][tr[b]],lab[tr[b]]) for a,b in skf.split(tr,lab[tr])]
                if np.mean(iv)>bestv: bestv=np.mean(iv); best=k
            nn.append(acc(combos[best][tr],lab[tr],combos[best][te],lab[te]))
        twfb_nested.append(np.mean(nn))
    print(f'{sid}: band_leaky={band_leaky[-1]*100:5.1f}  TWFB_leaky={twfb_leaky[-1]*100:5.1f}  TWFB_nested={twfb_nested[-1]*100:5.1f}  ({time.time()-t0:5.1f}s)',flush=True)

print('\n==== SUMMARY n=%d, reps=%d ===='%(NSUB,NREP))
print('band-only leaky (shipped .m, 8 combos)   : %.2f%%'%(np.nanmean(band_leaky)*100))
print('TW+FB    leaky (paper, 56 combos)        : %.2f%%'%(np.nanmean(twfb_leaky)*100))
print('TW+FB    honest nested-CV                : %.2f%%'%(np.mean(twfb_nested)*100))
