"""Training-only learning curves. NEVER use validation/external stars for tuning."""

import argparse
import json
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from run import ROOT, SEED, extract_features_from_array, write_json
from ztf_discovery import observed_features
from ztf_supervised import BASE_RES, LABELS


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--per-class",type=int,default=200)
    args=p.parse_args()
    folder=BASE_RES/"runs"/f"n{args.per_class}"
    manifest=json.loads((folder/"accepted_training_sources.json").read_text())
    split=json.loads((folder/"fixed_splits.json").read_text())
    metrics=json.loads((folder/"training_metrics.json").read_text())
    train=set(split["train"])
    rows=[r for r in manifest["records"] if r["vsx_name"] in train]
    names=metrics["feature_names"]
    data=[]
    for i,rec in enumerate(rows):
        raw=json.loads((ROOT/rec["source_file"]).read_text())
        arr=observed_features(raw[rec["ztf_oid"]]["lc"])
        values,_=extract_features_from_array(arr)
        data.append([values[col] for col in names])
        if (i+1)%100==0:
            print(f"Learning curve: real training observations loaded {i+1}/{len(rows)}",flush=True)
    x=np.asarray(data)
    y=np.asarray([rec["class_label"] for rec in rows])
    if len(rows)!=len(train) or not np.isfinite(x).all():
        raise RuntimeError("Training-only source leakage, missing data or invalid features")
    rng=np.random.default_rng(SEED)
    class_indices={c:rng.permutation(np.where(y==c)[0]) for c in LABELS}
    records=[]
    for frac in (.25,.50,.75,1.0):
        picked=np.concatenate([loc[:max(6,int(round(len(loc)*frac)))] for loc in class_indices.values()])
        scores=[]
        for i,j in StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED).split(x[picked],y[picked]):
            clf=RandomForestClassifier(n_estimators=400,min_samples_leaf=3,
                                       class_weight="balanced",n_jobs=-1,random_state=SEED)
            clf.fit(x[picked[i]],y[picked[i]])
            predictions=clf.predict(x[picked[j]])
            scores.append(float(f1_score(y[picked[j]],predictions,average="macro",labels=LABELS)))
        records.append({"fraction_of_original_training":frac,"n_stars":len(picked),
                        "training_only_cv_macro_f1_mean":float(np.mean(scores)),
                        "training_only_cv_macro_f1_sd":float(np.std(scores,ddof=1)),
                        "cv_fold_scores":scores})
        print(f"Train-only learning curve: {len(picked)} stars, macroF1={np.mean(scores):.3f}",flush=True)
    write_json(folder/"training_only_learning_curve.json",{
        "warning":"Exploratory model-capacity diagnostic using train split only; untouched validation and external not accessed.",
        "training_star_count":len(rows),"training_class_counts":dict(Counter(y)),
        "predeclared_fractions":[.25,.5,.75,1.0],"folds":3,
        "results":records})


if __name__=="__main__":
    main()
