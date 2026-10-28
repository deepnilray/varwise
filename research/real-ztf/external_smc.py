"""External field test on *genuine* SMC stars (no retraining or tuning)."""

import hashlib
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report

import run


def main():
    # Isolation: original model and 80/20 LMC split remain unchanged.
    run.LIGHTCURVES = run.ROOT / "data" / "lightcurves" / "smc"
    run.LIGHTCURVES.mkdir(parents=True, exist_ok=True)
    records = []
    rej = Counter()
    for name,(filename,subdir) in run.SOURCES.items():
        lines = (run.CATALOG / "smc" / filename).read_text().splitlines()
        candidates=[]
        for line in lines:
            f=line.split()
            if len(f) < 4 or not f[0].startswith("OGLE-SMC-"):
                continue
            try:
                confirmed_period=float(f[3])
            except ValueError:
                rej["missing_catalog_period"]+=1
                continue
            candidates.append({"source_id":f[0], "class_label":name,
                "catalog_period_days":confirmed_period,
                "source_url":f"https://ftp.astrouw.edu.pl/ogle/ogle4/OCVS/smc/{subdir}/phot/I/{f[0]}.dat",
                "catalog_path":f"data/catalogs/smc/{filename}"})
        rng=random.Random(run.SEED+700+len(name)*307)
        rng.shuffle(candidates)
        accepted=[]
        attempted=0
        while len(accepted)<100 and attempted<len(candidates):
            batch=candidates[attempted:attempted+64]
            attempted+=len(batch)
            with ThreadPoolExecutor(max_workers=10) as pool:
                fs=[pool.submit(run.get_one,r,80) for r in batch]
                for future in as_completed(fs):
                    r=future.result()
                    if r["status"]=="ok":
                        accepted.append(r)
                    else:
                        rej[r["status"]]+=1
            accepted=sorted(accepted,key=lambda a:a["source_id"])[:100]
            print(f"SMC real data {name}: {len(accepted)}/100, attempted {attempted}",flush=True)
        if len(accepted)<100:
            raise RuntimeError(f"Insufficient valid external samples: {name}")
        records.extend(accepted)

    base_manifest=json.loads((run.RESULTS/"source_manifest.json").read_text())
    lmc_ids={r["source_id"] for r in base_manifest["records"]}
    assert len(records)==300
    assert not lmc_ids.intersection({r["source_id"] for r in records})
    bundle=joblib.load(run.RESULTS/"model.joblib")
    feature_names=bundle["feature_names"]
    x=[]
    for i,rec in enumerate(records):
        feats,_=run.extract_features(rec)
        x.append([feats[k] for k in feature_names])
        if (i+1)%100==0:
            print(f"SMC features: {i+1}/300",flush=True)
    x=np.asarray(x,dtype=float)
    assert np.isfinite(x).all()
    truth=np.array([r["class_label"] for r in records])
    pred=bundle["model"].predict(x)
    labels=list(run.SOURCES)
    metrics={
        "status":"completed, independent SMC field only, no retraining",
        "actual_external_star_count":len(records),
        "train_field":"OGLE-IV LMC", "external_field":"OGLE-IV SMC",
        "class_counts":dict(Counter(truth)),
        "accuracy":float(accuracy_score(truth,pred)),
        "balanced_accuracy":float(balanced_accuracy_score(truth,pred)),
        "macro_f1":float(f1_score(truth,pred,average="macro")),
        "confusion_labels":labels,
        "confusion_matrix":confusion_matrix(truth,pred,labels=labels).tolist(),
        "report":classification_report(truth,pred,labels=labels,output_dict=True,zero_division=0),
        "rejected":dict(rej),
        "limitations":["Both fields observed by OGLE; this is an out-of-region test, not an independent telescope.",
                       "Catalog labels are published types but may contain cataloging mistakes."],
    }
    run.write_json(run.RESULTS/"external_smc_metrics.json",metrics)
    run.write_json(run.RESULTS/"external_smc_manifest.json",records)
    run.write_json(run.RESULTS/"external_smc_predictions.json",[
        {"source_id":r["source_id"],"catalog_label":r["class_label"],"predicted_label":str(p)}
        for r,p in zip(records,pred)
    ])
    print("EXTERNAL SMC METRICS "+json.dumps({k:metrics[k] for k in
          ("actual_external_star_count","accuracy","balanced_accuracy","macro_f1","confusion_matrix")}),flush=True)


if __name__=="__main__":
    main()
