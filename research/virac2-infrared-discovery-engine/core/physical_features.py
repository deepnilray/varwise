"""Frozen ten physical light-curve features for the 43-feature VIRAC2 model.
Do not change feature mathematics without training an entirely new model and validation.
Only the inference function is included here; abandoned original training routines were removed.
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

def physical(frame):
    f=frame.copy()
    for c in ['mag','emag','mjdobs','phot_flag']:f[c]=pd.to_numeric(f[c],errors='coerce')
    good=(f.phot_flag==0)&f.ambiguous_match.astype(str).str.lower().isin(['0','false','f'])&(f.emag>0)&(f.emag<=.3)&np.isfinite(f.mag)&np.isfinite(f.mjdobs)
    if 'filter' in f:good&=f['filter'].astype(str).str.strip()=='Ks'
    f=f[good].sort_values('mjdobs');t=f.mjdobs.to_numpy();mag=f.mag.to_numpy();err=f.emag.to_numpy()
    if len(t)<5:return None
    reference=np.median(mag);flux=10**(-.4*(mag-reference));sigma=np.maximum(.4*np.log(10)*flux*err,.02)
    weighted=np.average(flux,weights=1/sigma**2);chi0=float(np.sum(((flux-weighted)/sigma)**2))
    def model(v):
        t0,logte,logu,fs,fb=v;u=np.sqrt(np.exp(2*logu)+((t-t0)/np.exp(logte))**2)
        return fs*(u*u+2)/(u*np.sqrt(u*u+4))+fb
    bounds=([t.min()-100,np.log(.5),np.log(.01),0,-2],[t.max()+100,np.log(2000),np.log(3),5,5])
    peaks=np.argsort(mag)[:min(4,len(mag))];best=None
    for i in peaks:
        for te in [20,100]:
            z=least_squares(lambda v:(model(v)-flux)/sigma,[t[i],np.log(te),np.log(.3),.5,.5],bounds=bounds,max_nfev=120)
            chi=float(np.sum(z.fun**2))
            if best is None or chi<best[0]:best=(chi,z)
    chi,z=best;t0,lt,lu,fs,fb=z.x;te=np.exp(lt);res=(flux-model(z.x))/sigma
    event=np.abs(t-t0)<=2*te;outside=~event
    eta=float(np.mean(np.diff(flux)**2)/max(np.var(flux),1e-10))
    vals=[np.log1p(max(0,chi0-chi)),np.log1p(chi/max(1,len(t)-5)),eta,
          np.log1p(te),np.exp(lu),float(event.mean()),
          float(np.sqrt(np.mean(res[event]**2))) if event.any() else 100.,
          float(np.sqrt(np.mean(res[outside]**2))) if outside.any() else 100.,
          float(z.success),float(min(t0-t.min(),t.max()-t0)/max(1,te))]
    return vals,{'t0_mjd':float(t0),'tE_days':float(te),'delta_aic':chi0-chi-8,
                 'fit_converged':bool(z.success),'good_observations':len(t)}
