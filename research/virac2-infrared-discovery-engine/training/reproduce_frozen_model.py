#!/usr/bin/env python3
"""Reproduce the currently frozen development model from supplied *real* training feature rows.
This is a reproducibility check, NOT retraining with new labels or independent validation.
The original validation cohorts are not touched and no model receives discovery authorization.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=ROOT/'training'/'reproduced_development_model.joblib')
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError('Refusing to overwrite existing model output')
    rows=[r for r in json.loads((ROOT/'training/training_rows.json').read_text()) if r.get('measurement_status')=='scorable']
    if len(rows)!=366 or len({r['sourceid'] for r in rows})!=366 or len({r['group'] for r in rows})!=138:
        raise RuntimeError('Frozen training dataset unexpectedly modified')
    ids={str(r['sourceid']) for r in rows}
    right=set((ROOT/'model/training_source_ids.txt').read_text().split())
    if ids!=right:
        raise RuntimeError('Training-source identity list mismatch')
    X=np.asarray([r['features'] for r in rows],dtype=np.float64)
    y=np.asarray([r['label'] for r in rows],dtype=int)
    if X.shape!=(366,43) or set(y.tolist())!={0,1} or not np.isfinite(X).all():
        raise RuntimeError('Unexpected training feature schema')
    m=ExtraTreesClassifier(n_estimators=400,min_samples_leaf=2,max_features=.7,
        class_weight='balanced',random_state=20261112,n_jobs=1)
    m.fit(X,y)
    old=joblib.load(ROOT/'model/virac2_microlensing_v12_retrained.joblib')
    exact=bool(np.array_equal(m.predict_proba(X),old['model'].predict_proba(X)))
    if not exact:raise RuntimeError('Reproduced classifier fails frozen training-row score parity')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'model':m,'threshold':old['threshold'],'feature_count':43,
      'release_validated':False},args.out,compress=3)
    print(json.dumps({'reproduced_training_sources':366,'feature_count':43,
      'same_training_row_predictions':exact,'scientific_validation':'NOT PERFORMED',
      'release_validated':False,'saved_path':str(args.out),
      'saved_sha256':hashlib.sha256(args.out.read_bytes()).hexdigest()},indent=2))

if __name__=='__main__':main()
