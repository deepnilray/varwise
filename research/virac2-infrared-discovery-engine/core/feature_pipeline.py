"""Feature definitions used by the fitted 33-feature model."""
import numpy as np
from scipy.stats import rankdata
windows=(0,3,15,60,180)

def features(night,mag,error,exposures):
    if len(night)<2:return None
    p10,median,p90=np.percentile(mag,[10,50,90])
    peak=int(np.argmin(mag))
    post=mag[peak+1:]
    pre=mag[:peak]
    bright=median-mag[peak]
    mad=float(np.median(np.abs(mag-median)))
    rank=rankdata(mag)
    t=np.arange(len(night)); t=t-t.mean(); rank=rank-rank.mean()
    denom=np.linalg.norm(t)*np.linalg.norm(rank)
    nearby=(np.arange(len(night))!=peak)&(mag<=mag[peak]+max(.4,bright*.4))
    gap=np.min(np.abs(night[nearby]-night[peak])) if np.any(nearby) else 4000.
    return list(map(float,[bright,p90-median,p90-p10,bright/max(.05,mad, np.median(error)),
            float(np.median(post)-mag[peak]) if len(post) else 0.,
            float(np.median(pre)-mag[peak]) if len(pre) else 0.,
            float(np.dot(t,rank)/denom) if denom else 0.,mad,np.median(np.abs(np.diff(mag))),
            float(np.median(mag[len(mag)//2:])-np.median(mag[:len(mag)//2])),
            float(median-np.partition(mag,1)[1]),np.log1p(gap),np.log1p(exposures[peak])]))

def flare_features(t,m,e,k):
    # Use a robust reference from measurements outside each candidate event window.
    # The nightly error and a robust scatter floor avoid assigning infinite weight to an isolated point.
    n=len(t)
    if n<2:raise ValueError('At least two good nights are required')
    peak=int(np.argmin(m))
    result=[]
    for width in windows:
        end=np.searchsorted(t,t+width,side='right')
        candidates=[]
        # The brightest nights are sufficient to propose windows without trying all n² intervals.
        for anchor in np.argsort(m)[:min(n,8)]:
            for start,stop in ((anchor,end[anchor]),(np.searchsorted(t,t[anchor]-width),anchor+1)):
                outside=np.r_[m[:start],m[stop:]]
                if len(outside)==0:continue
                baseline=float(np.median(outside))
                scatter=float(1.4826*np.median(np.abs(outside-baseline)))
                var=e[start:stop]**2+max(.06,scatter)**2
                weights=1/var
                contrast=baseline-float(np.average(m[start:stop],weights=weights))
                baseline_error=max(.06,scatter,float(np.median(e)))/np.sqrt(len(outside))
                snr=contrast/np.sqrt(1/weights.sum()+baseline_error**2)
                candidates.append((float(snr),contrast,stop-start))
        result.extend(max(candidates,default=(0.,0.,0),key=lambda x:x[0]))
    ref=np.delete(m,peak)
    center=np.median(ref)
    spread=max(.06,1.4826*np.median(np.abs(ref-center)),np.median(e))
    result += [(center-m[peak])/spread,
               (np.median(m[peak+1:])-m[peak]) if peak<n-1 else 0.,
               (np.median(m[:peak])-m[peak]) if peak>0 else 0.,
               np.log1p(n-peak-1),np.log1p(peak)]
    return [float(x) for x in result]
