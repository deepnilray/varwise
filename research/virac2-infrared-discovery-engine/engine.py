#!/usr/bin/env python3
"""VIRAC2 real-photometry candidate ranking. NOT a discovery confirmation system.

Input: ESO VIRAC2 DR3 Ks time-series records (one row per exposure).
Output: scored catalogue, HTML light-curve report, full JSON audit.
Trained on real labeled microlensing/other variable classes; no fresh blind validation.
"""
from __future__ import annotations
import argparse,hashlib,html,json,sys,urllib.parse,tempfile,shutil,datetime,platform
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'core'))

REQUIRED={'sourceid','catid','mjdobs','filter','mag','emag','phot_flag','ambiguous_match'}
OUTPUT_FILES=('candidate_audit.json','candidate_ranking.csv','candidate_report.html','summary.json','review_checklist.csv','run_provenance.json')

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(2**20),b''):
            h.update(block)
    return h.hexdigest()


def _load():
    import joblib
    manifest=json.loads((ROOT/'model/manifest.json').read_text())
    if manifest.get('model_file')!='virac2_microlensing_v12_retrained.joblib':
        raise RuntimeError('Model manifest points to an unexpected file')
    path=ROOT/'model'/manifest['model_file']
    actual=sha256_file(path)
    if actual!=manifest['sha256']:
        raise RuntimeError('Model-integrity mismatch: cannot load altered weights')
    weights=joblib.load(path)  # Only load the trusted model shipped with this package.
    if bool(weights.get('release_validated',False)) or bool(manifest.get('release_validated',False)):
        raise RuntimeError('Unexpected status: development model must remain unvalidated')
    if weights.get('feature_count')!=43:
        raise RuntimeError('Incompatible expected feature count')
    if manifest.get('training_sources')!=366 or manifest.get('training_exposure_sky_groups')!=138:
        raise RuntimeError('Unexpected training cohort/model manifest')
    for key,relative in [('training_ids_sha256','model/training_source_ids.txt'),
                         ('training_rows_sha256','training/training_rows.json'),
                         ('fold_predictions_sha256','training/fold_predictions.json')]:
        if manifest.get(key) and sha256_file(ROOT/relative)!=manifest[key]:
            raise RuntimeError('Frozen training-record integrity mismatch: '+relative)
    if not hasattr(weights['model'],'predict_proba') or list(weights['model'].classes_)!=[0,1] or weights['model'].n_features_in_!=43:
        raise RuntimeError('Saved model is not the expected 43-feature binary classifier')
    return manifest,weights


def _quality_nights(frame):
    import pandas as pd,numpy as np
    f=frame.copy()
    for c in ['mag','emag','mjdobs','phot_flag','catid']:
        f[c]=pd.to_numeric(f[c],errors='coerce')
    mask=(f['phot_flag']==0)&(f['ambiguous_match'].astype(str).str.lower().isin(['0','false','f']))&\
         (f['emag']>0)&(f['emag']<=.3)&np.isfinite(f['mag'])&np.isfinite(f['mjdobs'])&np.isfinite(f['catid'])&(f['mjdobs'].between(55200,59000))&(f['mag'].between(0,30))
    if 'filter' in f: mask &= f['filter'].astype(str).str.strip().eq('Ks')
    f=f.loc[mask].copy()
    f['night']=np.floor(f.mjdobs).astype(int)
    return f


def score(input_path:Path,out_dir:Path,top:int=30,max_rows:int=800_000):
    import numpy as np,pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from predict import feature_row
    from physical_features import physical
    from input_guard import validate_input,source_preflight,audit_physical_evidence,fit_stability_check

    manifest,weights=_load()
    if input_path.suffix.lower()!='.csv':
        raise ValueError('Only uncompressed plain .csv files are accepted for this auditable scorer')
    if input_path.stat().st_size>250_000_000:
        raise ValueError('CSV exceeds 250MB; process one region at a time')
    frame=pd.read_csv(input_path,dtype={'sourceid':str,'catid':str},low_memory=False,nrows=max_rows+1)
    missing=REQUIRED-set(frame)
    if missing:raise ValueError('Missing input columns: '+', '.join(sorted(missing)))
    if len(frame)>max_rows: raise ValueError(f'Too many rows ({len(frame):,}); split into batches of <= {max_rows:,}')
    preflight=validate_input(frame)
    if out_dir.resolve()==input_path.resolve() or input_path.resolve().is_relative_to(out_dir.resolve()):
        raise ValueError('Choose an output directory separate from input data')
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError('Output directory already contains files. Use a new output path to preserve the audit trail.')
    out_dir.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.virac2_stage_',dir=str(out_dir.parent)))
    try:
        return _render_score_stage(frame,input_path,out_dir,stage,manifest,weights,top,preflight,plt,feature_row,physical,source_preflight,audit_physical_evidence,fit_stability_check)
    finally:
        if stage.exists():shutil.rmtree(stage,ignore_errors=True)

def _render_score_stage(frame,input_path,out_dir,stage,manifest,weights,top,preflight,plt,feature_row,physical,source_preflight,audit_physical_evidence,fit_stability_check):
    import numpy as np,pandas as pd
    known=set((ROOT/'model/training_source_ids.txt').read_text().split())
    frozen_rows=[r for r in json.loads((ROOT/'training/training_rows.json').read_text()) if r.get('measurement_status')=='scorable']
    if len(frozen_rows)!=366:raise RuntimeError('Frozen training fingerprint registry corrupted')
    def fingerprint(v):
        return hashlib.sha256(np.round(np.asarray(v,dtype=np.float64)[:33],8).astype('<f8').tobytes()).hexdigest()
    known_fingerprints={fingerprint(r['features']):str(r['sourceid']) for r in frozen_rows}
    records=[]
    curves={}
    groups=frame.groupby('sourceid',sort=False,dropna=True)
    for idx,(oid,f) in enumerate(groups,1):
        oid=str(oid)
        good=_quality_nights(f)
        quality_flags,coords=source_preflight(f,good)
        ra,dec=coords if coords is not None else (None,None)
        n_nights=int(good['night'].nunique()) if len(good) else 0
        r=dict(sourceid=oid,ra_deg=ra,dec_deg=dec,raw_observations=len(f),good_observations=len(good),
               good_nights=n_nights,training_source=oid in known,score=None,
               data_quality_flags=quality_flags,physical_evidence_flags=[],
               training_lightcurve_fingerprint_match=None,
               status='abstain_insufficient_valid_nights',original_image_verified=False,
               independently_verified_discovery=False,novelty_catalog_checked=False)
        if quality_flags and set(quality_flags)-{'no_astrometric_coordinates','insufficient_valid_coordinates'}:
            r['status']='abstain_data_quality_failure'
        elif n_nights>=5 and good.catid.nunique()>=5:
            try:
                x,n=feature_row(f)
                if x is not None:
                    match=known_fingerprints.get(fingerprint(x))
                    if match is not None and match!=oid:
                        r['training_lightcurve_fingerprint_match']=match
                        r['data_quality_flags'].append('photometry_identical_to_known_training_feature_vector')
                fit=physical(f)
                if x is not None and n>=5 and fit is not None:
                    values,diagnostic=fit
                    X=np.asarray(x.tolist()+values,dtype=float)[None,:]
                    if X.shape==(1,43) and np.isfinite(X).all():
                        s=float(weights['model'].predict_proba(X)[0,1])
                        pflags,coverage=audit_physical_evidence(good,diagnostic)
                        stability={}
                        if s>=manifest['cutoff'] and oid not in known:
                            stability_flags,stability=fit_stability_check(f,diagnostic,physical)
                            pflags+=stability_flags
                        r.update(score=s,threshold=manifest['cutoff'],physical_fit=diagnostic,
                                 physical_evidence_flags=pflags,coverage=coverage,fit_stability=stability)
                        if oid in known:
                            r['status']='known_training_control_not_discovery'
                        elif r['training_lightcurve_fingerprint_match'] is not None:
                            r['status']='blocked_training_lightcurve_reuse'
                        elif s>=manifest['cutoff'] and (pflags or quality_flags):
                            r['status']='blocked_high_score_requires_data_or_fit_review'
                        elif s>=manifest['cutoff']:
                            r['status']='unvetted_high_score_needs_images_and_novelty_checks'
                        elif diagnostic['fit_converged'] and diagnostic['delta_aic']>10:
                            r['status']='below_cutoff_possible_physical_anomaly'
                        else:
                            r['status']='below_development_threshold_not_nonvariable'
                    else:r['status']='abstain_nonfinite_features'
                else:r['status']='abstain_feature_extraction'
            except (ValueError,TypeError,ArithmeticError,FloatingPointError) as ex:
                r['status']='abstain_fit_error';r['error']=str(ex)[:250]
        records.append(r)
        if r['score'] is not None:
            curves[oid]=good.copy()
    records.sort(key=lambda x: (-1 if x['score'] is None else -x['score'],x['sourceid']))
    (stage/'candidate_audit.json').write_text(json.dumps({'warning':'UNVALIDATED. Development scores only. Images and catalog novelty unverified.',
        'model_sha256':manifest['sha256'],'source_file_sha256':sha256_file(input_path),
        'selection_rules_validated':False,'independent_validation_passed':False,
        'scores':records},indent=2,allow_nan=False),encoding='utf-8')
    flat=[]
    for r in records:
        flat.append({k:v for k,v in r.items() if k not in ['physical_fit','error']})
    pd.DataFrame(flat).to_csv(stage/'candidate_ranking.csv',index=False)
    checklist=[]
    for r in records:
        if (r['score'] is not None and not r['training_source'] and
            r['training_lightcurve_fingerprint_match'] is None and r['score']>=manifest['cutoff']):
            checklist.append({'sourceid':r['sourceid'],'score':r['score'],'review_status':r['status'],
              'original_fits_images_checked':'','at_least_two_independent_epochs':'',
              'detector_artifact_ruled_out':'','neighbor_blend_ruled_out':'',
              'survey_catalogs_checked':'','independent_photometry_available':'',
              'reviewer':'','evidence_links':'','notes':'','scientific_verdict':'UNREVIEWED'})
    pd.DataFrame(checklist,columns=['sourceid','score','review_status','original_fits_images_checked',
      'at_least_two_independent_epochs','detector_artifact_ruled_out','neighbor_blend_ruled_out',
      'survey_catalogs_checked','independent_photometry_available','reviewer','evidence_links','notes',
      'scientific_verdict']).to_csv(stage/'review_checklist.csv',index=False)
    candidates=[r for r in records if r['score'] is not None and not r['training_source'] and r['training_lightcurve_fingerprint_match'] is None and r['score']>=manifest['cutoff']]
    candidates.sort(key=lambda x:-x['score'])
    render_report(records,candidates,curves,stage,top,manifest,plt)
    summary={'input_rows':len(frame),'sources':len(records),'scored_sources':sum(x['score'] is not None for x in records),
             'known_training_sources':sum(x['training_source'] for x in records),
             'training_lightcurve_reuse_suspected':sum(x['training_lightcurve_fingerprint_match'] is not None for x in records),
             'model_threshold_exceedances':sum(x['score'] is not None and x['score']>=manifest['cutoff'] and not x['training_source'] for x in records),
             'review_leads':sum(x['status']=='unvetted_high_score_needs_images_and_novelty_checks' for x in records),
             'blocked_high_score':sum(x['status']=='blocked_high_score_requires_data_or_fit_review' for x in records),
             'discovery_candidates_confirmed':0,'independent_validation_passed':False,
             'not_confirmed':True,'report':str(out_dir/'candidate_report.html')}
    (stage/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    import scipy,sklearn,numpy,pandas
    provenance={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'input_sha256':sha256_file(input_path),'model_sha256':manifest['sha256'],
      'input_preflight':preflight,
      'pipeline_code_sha256':{key:sha256_file(ROOT/key) for key in
        ['engine.py','core/feature_pipeline.py','core/predict.py','core/physical_features.py','core/input_guard.py','model/manifest.json']},
      'software':{'python':platform.python_version(),
      'numpy':numpy.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'pandas':pandas.__version__},
      'validation_type':'development_grouped_CV_only','discovery_validated':False,
      'selection_rules_validated':False,'output_files_sha256':{x:sha256_file(stage/x) for x in OUTPUT_FILES if (stage/x).exists()}}
    (stage/'run_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    # Commit the completed run atomically. Refuse to overwrite any prior scientific record.
    if out_dir.exists():
        if any(out_dir.iterdir()):raise FileExistsError('Output directory became nonempty while processing')
        out_dir.rmdir()
    stage.replace(out_dir)
    return summary


def render_report(records,candidates,curves,out_dir,top,manifest,plt):
    import base64,io,numpy as np
    def e(x):return html.escape(str(x),quote=True)
    def fmt(x):return '—' if x is None else f'{x:.4f}'
    allplots=[]
    rankables=candidates[:top] or [r for r in records if r['score'] is not None][:top]
    for r in rankables:
        f=curves[r['sourceid']]
        fig,ax=plt.subplots(figsize=(7.6,2.7),layout='constrained')
        colors=('tab:blue','tab:orange')
        ax.errorbar(f.mjdobs,f.mag,yerr=f.emag,fmt='.',markersize=2.5,alpha=.65,
                    color=colors[0],ecolor='lightgray',elinewidth=.5,capsize=0)
        ax.invert_yaxis();ax.set_xlabel('MJD (days)');ax.set_ylabel('Ks magnitude (brighter upwards)')
        ax.set_title('Source '+r['sourceid']+' | ML ranking score '+fmt(r['score']),fontsize=10)
        ax.grid(alpha=.2)
        buf=io.BytesIO();fig.savefig(buf,format='png',dpi=125);plt.close(fig)
        img=base64.b64encode(buf.getvalue()).decode()
        coords=''
        if r['ra_deg'] is not None:
            q=f'{r["ra_deg"]} {r["dec_deg"]}'
            simbad='https://simbad.cds.unistra.fr/simbad/sim-coo?Coord='+urllib.parse.quote(q)+'&Radius=5&Radius.unit=arcsec'
            coords=f'<a href="{e(simbad)}" target="_blank" rel="noreferrer">SIMBAD coordinate check</a>'
        allplots.append(f'''<article><div class="h"><strong>{e(r['sourceid'])}</strong><span>{e(r['status'])}</span></div>
        <div class="metadata">Score {fmt(r['score'])} / cutoff {fmt(manifest['cutoff'])} · {r['good_nights']} valid nights · {r['good_observations']} observations · {coords}</div>
        <img alt="IR light curve for {e(r['sourceid'])}" src="data:image/png;base64,{img}" />
        <p class="warning">Unconfirmed lead. Check original FITS images, nearby artifacts, literature and independent photometry.</p></article>''')
    table=''.join('<tr><td>'+e(r['sourceid'])+'</td><td>'+fmt(r['score'])+'</td><td>'+str(r['good_nights'])+'</td><td>'+e(r['status'])+'</td></tr>' for r in records[:500])
    htmltext=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VIRAC2 infrared candidate review</title><style>body{{font:15px system-ui,sans-serif;color:#203145;background:#f3f6f9;max-width:1050px;margin:auto;padding:32px}}
.card,article{{background:#fff;border:1px solid #d8e1eb;border-radius:12px;padding:20px;margin:15px 0}}h1{{font-size:30px}}h2{{margin:32px 0 12px}}.danger{{border-left:4px solid #d97706;padding:15px;background:#fff8ec}}.kpis{{display:flex;gap:20px;flex-wrap:wrap}}.kpis b{{font-size:24px;display:block}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #dce5ee}}.h{{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}}.metadata{{color:#546375;margin:8px 0}}img{{display:block;max-width:100%;height:auto}}.warning{{color:#865a17}}a{{color:#0a6399}}small{{color:#596a7a}}</style></head>
<body><h1>Infrared discovery candidate review</h1><p class="danger"><strong>UNVALIDATED REVIEW ONLY — not a confirmed discovery.</strong> Model trained on real published examples but has no fresh independent blind validation. Original images have not been inspected; no automated novelty check is complete.</p>
<div class="card"><div class="kpis"><div><b>{len(records):,}</b> sources loaded</div><div><b>{sum(r['score'] is not None for r in records):,}</b> scored</div><div><b>{sum(r['status']=='unvetted_high_score_needs_images_and_novelty_checks' for r in records)}</b> review leads</div></div><p><small>VIRAC2 DR3 Ks · development threshold {manifest['cutoff']:.4f} · retained known training controls are not discoveries</small></p></div>
<h2>Ranked observations (first 500 only; complete CSV contains every source)</h2><div class="card"><table><thead><tr><th>Source ID</th><th>Score</th><th>Good nights</th><th>Status</th></tr></thead><tbody>{table}</tbody></table></div>
<h2>Light-curve evidence</h2>{''.join(allplots)}<h2>How to establish a discovery</h2><div class="card"><p>1. Confirm that several clean exposures show a repeatable point-source signal. 2. Compare original ESO images and detector positions; reject saturated wings, streaks and blends. 3. Check SIMBAD, VizieR, VSX, the literature and appropriate transient catalogues. 4. Seek independent photometric evidence. 5. Only then prepare a scientific report.</p></div>
<footer><small>Generated offline from the supplied CSV; this report does not query external catalogs.</small></footer></body></html>'''
    (out_dir/'candidate_report.html').write_text(htmltext,encoding='utf-8')


def fetch(ids:list[str],target:Path,table:str='main'):
    """Fetch real Ks light curves for *given* object IDs; not a blind discovery scan.

    Network operation requires running on the user's internet-connected computer.
    """
    import requests,pandas as pd,io
    if not ids:raise ValueError('Supply at least one source ID')
    if len(ids)>30:raise ValueError('Limit each request to 30 IDs')
    if len(set(ids))!=len(ids):raise ValueError('Duplicate source IDs in request')
    from input_guard import ID_PATTERN
    for oid in ids:
        if ID_PATTERN.fullmatch(oid) is None:
            raise ValueError('Source IDs must be positive canonical ASCII integers <= 19 digits')
    tab={'main':'vvvx_virac_v2_lc','rejected':'vvvx_virac_v2_rejected_lc'}[table]
    columns='sourceid,catid,mjdobs,filter,mag,emag,phot_flag,ambiguous_match,ra,de,x,y,ext,chi'
    query=f"SELECT TOP 100001 {columns} FROM {tab} WHERE sourceid IN ({','.join(ids)}) AND filter='Ks'"
    url='https://archive.eso.org/tap_cat/sync'
    try:
        response=requests.post(url,data={'REQUEST':'doQuery','LANG':'ADQL','FORMAT':'csv','QUERY':query,'MAXREC':'100001'},timeout=110)
        response.raise_for_status()
    except requests.RequestException as ex:
        raise RuntimeError(f'ESO archive request failed: {type(ex).__name__}') from ex
    if '<VOTABLE' in response.text[:2000] and 'QUERY_STATUS' in response.text:
        raise RuntimeError('ESO returned an error VOTable. See telescope archive response: '+response.text[:600])
    try: frame=pd.read_csv(io.StringIO(response.text),dtype={'sourceid':str,'catid':str})
    except Exception as exc:raise RuntimeError('Could not parse ESO response as CSV; inspect TAP service status') from exc
    if not REQUIRED <= set(frame):
        raise RuntimeError('ESO returned unexpected columns: '+str(frame.columns.tolist())+'; response: '+response.text[:250])
    if len(frame)>=100000:
        raise RuntimeError('ESO response reached the row ceiling and may be truncated; refuse incomplete download')
    if len(frame)==0:raise RuntimeError('No records returned for requested source IDs')
    # Parse identifiers as strings before checking; reject unexpected rows or mixed filters.
    frame['sourceid']=frame['sourceid'].astype(str)
    if not set(frame.sourceid).issubset(set(ids)):
        raise RuntimeError('ESO returned an unexpected source ID')
    if not frame['filter'].astype(str).str.strip().eq('Ks').all():
        raise RuntimeError('ESO returned measurements outside Ks band')
    if target.exists():raise FileExistsError('Refusing to overwrite a previously fetched ESO dataset')
    target.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(target,index=False)
    return {'downloaded_rows':len(frame),'matched_sources':int(frame.sourceid.nunique()),
            'missing_source_ids':sorted(set(ids)-set(frame.sourceid)),'output':str(target),
            'origin':'ESO TAP VIRAC2 DR3','hash_sha256':sha256_file(target)}


def main():
    p=argparse.ArgumentParser(description='Rank VIRAC2 infrared microlensing leads from real light curves (experimental)')
    sub=p.add_subparsers(dest='cmd',required=True)
    s=sub.add_parser('score',help='Score a real-ESO-lightcurve CSV, generate ranked HTML report')
    s.add_argument('--input',type=Path,required=True);s.add_argument('--out',type=Path,default=Path('results/new_run'))
    s.add_argument('--top',type=int,default=30)
    f=sub.add_parser('fetch',help='Fetch real VIRAC2 light curves by known IDs from official ESO catalog')
    f.add_argument('--ids',required=True,help='Comma-separated numeric source IDs (max 30)')
    f.add_argument('--table',choices=['main','rejected'],default='main')
    f.add_argument('--out',type=Path,default=Path('results/fetched_virac2.csv'))
    sub.add_parser('info',help='Show training status and provenance')
    a=p.parse_args()
    try:
        if a.cmd=='score':
            if a.top<=0 or a.top>200:raise ValueError('--top must be from 1 to 200')
            r=score(a.input,a.out,a.top)
        elif a.cmd=='fetch':r=fetch([x.strip() for x in a.ids.split(',') if x.strip()],a.out,a.table)
        else:r=_load()[0]
        print(json.dumps(r,indent=2))
    except (ValueError,RuntimeError,OSError) as ex:
        print('ERROR:',ex,file=sys.stderr);sys.exit(2)
if __name__=='__main__':main()
