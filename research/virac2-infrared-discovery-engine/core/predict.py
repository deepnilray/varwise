"""Frozen 33-feature extraction from real VIRAC2 Ks photometry.
This module is a feature extractor, not a standalone discovery predictor.
"""
import numpy as np
import pandas as pd
from feature_pipeline import features,flare_features

def feature_row(frame):
    f=frame.copy()
    for c in ['mjdobs','mag','emag','catid','phot_flag']:
        f[c]=pd.to_numeric(f[c],errors='coerce')
    unambiguous=f.ambiguous_match.astype(str).str.lower().isin(['0','false','f'])
    keep=(f.phot_flag==0)&unambiguous&np.isfinite(f.mjdobs)&np.isfinite(f.mag)&(f.emag>0)&(f.emag<=.3)&np.isfinite(f.catid)
    if 'filter' in f:keep&=f['filter'].astype(str).str.strip()=='Ks'
    f=f[keep].copy();f['night']=np.floor(f.mjdobs).astype(int)
    n=f.groupby('night',sort=True).agg(mag=('mag','median'),error=('emag','median'),exposures=('catid','nunique'))
    if len(n)<2:return None,len(n)
    args=n.index.to_numpy(),n.mag.to_numpy(),n.error.to_numpy(),n.exposures.to_numpy()
    x=np.array(features(*args)+flare_features(*args))
    if not np.isfinite(x).all():return None,len(n)
    return x,len(n)
