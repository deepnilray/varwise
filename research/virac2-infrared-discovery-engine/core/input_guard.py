"""Fail-closed input and post-fit evidence checks; does NOT retrain/redefine the 43 ML features.
Scientific review rules are provisional. Never treat them as independently validated selection cuts.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

ID_PATTERN=re.compile(r"[1-9][0-9]{0,18}\Z")
REQUIRED_COLUMNS={"sourceid", "catid", "mjdobs", "filter", "mag", "emag", "phot_flag", "ambiguous_match"}


def validate_input(frame):
    missing=REQUIRED_COLUMNS-set(frame.columns)
    if missing:
        raise ValueError("Required VIRAC2 columns missing: "+", ".join(sorted(missing)))
    if len(frame)==0:
        raise ValueError("Input CSV is empty")
    # Never allow pandas to round source or exposure IDs through binary floating point.
    for field in ("sourceid", "catid"):
        ids=frame[field]
        ok=ids.notna() & ids.astype(str).map(lambda x: ID_PATTERN.fullmatch(x) is not None)
        if not ok.all():
            raise ValueError(f"{field} must contain unambiguous ASCII integer IDs; invalid rows: {int((~ok).sum())}")
        if field=="catid" and any(int(x)>2**53 for x in ids.unique()):
            # The frozen 43-feature extractor converts catid to numeric: precision loss would be silent.
            raise ValueError("catid > 2**53 cannot be represented exactly by the frozen training extractor")
    if frame["filter"].isna().any():
        raise ValueError("Missing filter values; refuse to assume that observations are Ks")
    filters=set(frame["filter"].astype(str).str.strip())
    if not filters.intersection({"Ks"}):
        raise ValueError("No explicitly identified Ks measurements in CSV")
    return {"raw_rows":len(frame),"sources":frame.sourceid.nunique(),"observed_filters":sorted(filters)}


def source_preflight(raw, quality):
    """Track observation-ID duplication, missing values, and astrometry; never fabricate a discovery."""
    flags=[]
    if quality.empty:
        return ['no_valid_photometry'],None
    if quality.catid.astype(str).duplicated().any():
        flags.append('duplicate_exposure_ids')
    raw_mag=pd.to_numeric(raw.mag,errors='coerce')
    raw_time=pd.to_numeric(raw.mjdobs,errors='coerce')
    observed=(raw['filter'].astype(str).str.strip()=='Ks')
    if ((raw_mag<0)|(raw_mag>30)).loc[observed].any():
        flags.append('unphysical_magnitude')
    if ((raw_time<55200)|(raw_time>59000)).loc[observed].any():
        flags.append('outside_virac2_observation_era')
    # User can provide coordinates from the ESO LC or attached source catalog.
    ra_col='ra' if 'ra' in raw else None
    dec_col='de' if 'de' in raw else ('dec' if 'dec' in raw else None)
    if ra_col is None or dec_col is None:
        flags.append('no_astrometric_coordinates')
        center=None
    else:
        ra=pd.to_numeric(quality[ra_col],errors='coerce').to_numpy(dtype=float)
        dec=pd.to_numeric(quality[dec_col],errors='coerce').to_numpy(dtype=float)
        v=np.isfinite(ra)&np.isfinite(dec)&(ra>=0)&(ra<360)&(dec>=-90)&(dec<=90)
        if v.sum()<min(5,len(quality)):
            flags.append('insufficient_valid_coordinates')
            center=None
        else:
            rr=np.deg2rad(ra[v]); dd=np.deg2rad(dec[v])
            xyz=np.column_stack([np.cos(dd)*np.cos(rr),np.cos(dd)*np.sin(rr),np.sin(dd)])
            c=np.mean(xyz,axis=0);c=c/np.linalg.norm(c)
            dist=np.rad2deg(np.arccos(np.clip(xyz@c,-1,1)))*3600
            center=(float(np.mod(np.rad2deg(np.arctan2(c[1],c[0])),360)),float(np.rad2deg(np.arcsin(c[2]))))
            if np.nanpercentile(dist,95)>5:
                flags.append('astrometric_scatter_over_5_arcsec')
    return flags,center


def audit_physical_evidence(quality, fit):
    """Additional *qualitative* checks, not calibrated purity/precision estimates."""
    flags=[]
    if not bool(fit.get('fit_converged')):
        flags.append('optimizer_did_not_converge')
    t0,te=fit.get('t0_mjd'),fit.get('tE_days')
    if not (np.isfinite(t0) and np.isfinite(te) and .5 < te < 1900):
        flags.append('unconstrained_microlensing_timescale')
        return flags,{}
    nights=np.floor(quality.mjdobs.to_numpy(dtype=float)).astype(int)
    tt=quality.mjdobs.to_numpy(dtype=float)
    pre=int(np.unique(nights[tt < t0-2*te]).size)
    post=int(np.unique(nights[tt > t0+2*te]).size)
    inside=int(np.unique(nights[np.abs(tt-t0)<=te]).size)
    if pre<3 or post<3:
        flags.append('inadequate_pre_or_post_event_baseline')
    if inside<2:
        flags.append('fewer_than_two_event_nights')
    if not np.isfinite(fit.get('delta_aic',float('nan'))) or fit['delta_aic']<=10:
        flags.append('weak_physical_model_improvement')
    return flags,{"nights_before_2tE":pre,"nights_after_2tE":post,"event_nights_within_tE":inside}


def fit_stability_check(raw, initial_fit, recompute_physical):
    """Stress check: a tiny deterministic perturbation should not drastically change the fitted model.
    This is NOT a parameter uncertainty interval or substitute for a complete posterior.
    """
    t0=float(initial_fit['t0_mjd']);te=float(initial_fit['tE_days'])
    if not initial_fit.get('fit_converged') or not np.isfinite(te) or te<=0:
        return ['fit_stability_not_testable'],{}
    changed=raw.copy()
    # +/- 1e-5 mag is far below normal photometric uncertainties, probing optimizer sensitivity.
    for i in range(len(changed)):
        if i%2==0:
            val=pd.to_numeric(pd.Series([changed.iloc[i]['mag']]),errors='coerce').iloc[0]
            if np.isfinite(val):
                changed.iat[i,changed.columns.get_loc('mag')]=val+(1e-5 if i%4==0 else -1e-5)
    try:
        alt=recompute_physical(changed)
        if alt is None:return ['perturbed_fit_failed'],{}
        f=alt[1]
        if not f['fit_converged']:return ['perturbed_fit_did_not_converge'],{'perturbed_tE_days':f.get('tE_days')}
        ratio=max(float(f['tE_days'])/te,te/float(f['tE_days']))
        drift=abs(float(f['t0_mjd'])-t0)
        details={'perturbed_tE_days':float(f['tE_days']),
                 'timescale_ratio':ratio,'peak_drift_days':drift}
        if ratio>3 or drift>max(1,te):
            return ['perturbation_sensitive_physical_fit'],details
        return [],details
    except (ValueError,TypeError,ArithmeticError, FloatingPointError, OverflowError) as exc:
        return ['perturbed_fit_failed'],{'exception_type':type(exc).__name__}
