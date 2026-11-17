"""Recheck archived real-data training record, integrity and grouped scores."""
from pathlib import Path
import hashlib,json,collections
import numpy as np
from sklearn.metrics import average_precision_score,confusion_matrix
ROOT=Path(__file__).resolve().parent
rows=[r for r in json.loads((ROOT/'training/training_rows.json').read_text()) if r.get('measurement_status')=='scorable']
result=json.loads((ROOT/'training/retraining_report.json').read_text())
folds=json.loads((ROOT/'training/fold_predictions.json').read_text())
manifest=json.loads((ROOT/'model/manifest.json').read_text())
model=ROOT/'model'/manifest['model_file']
assert hashlib.sha256(model.read_bytes()).hexdigest()==manifest['sha256']
assert len(rows)==366 and len(set(r['sourceid'] for r in rows))==len(rows)
assert len({r['group'] for r in rows})==138
assert len({len(r['features']) for r in rows})==1 and len(rows[0]['features'])==43
idx_all=set(range(len(rows)));assigned=[];probs=np.zeros(len(rows));pred=np.zeros(len(rows),dtype=bool)
for f in folds:
    held=set(f['indices']);train=idx_all-held
    assert held and train and not (held&train)
    assert not {rows[i]['group'] for i in held}&{rows[i]['group'] for i in train}
    assert len(held)==len(f['scores'])==len(f['predicted'])
    for i,s,p in zip(f['indices'],f['scores'],f['predicted']):probs[i]=s;pred[i]=p
    assigned.extend(f['indices'])
assert sorted(assigned)==list(range(len(rows)))
y=np.array([r['label'] for r in rows])
assert confusion_matrix(y,pred).tolist()==result['nested_development_confusion_matrix']
assert abs(average_precision_score(y,probs)-result['nested_development_ap'])<1.e-12
assert result['external_tests_used_for_training_or_thresholds'] is False
assert manifest['release_validated'] is False
print(json.dumps({'tests_passed':10,'real_labeled_sources':len(rows),'unique_groups':len({r['group'] for r in rows}),
                  'nested_grouped_development_average_precision':average_precision_score(y,probs),
                  'nested_confusion_matrix':confusion_matrix(y,pred).tolist(),
                  'new_independent_validation':False,'weights_sha256':manifest['sha256']},indent=2))
