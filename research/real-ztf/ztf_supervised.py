"""Real-label ZTF DR24 supervised training. No simulated or pseudo-labeled stars.

Ground truth: exact AAVSO VSX published types RRAB, DCEP, EW.
Photometry: public ZTF DR24, ZTF-r band only, as served by SNAD.

Windows: python ztf_supervised.py --per-class 200 --workers 3
Resume:  repeat the same command; raw responses are cached with checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from run import ROOT, SEED, extract_features_from_array, write_json
from ztf_catalog_check import tsv_rows
from ztf_discovery import angular_arcsec, observed_features


DATA=ROOT/"data"/"ztf_supervised"
BASE_RES=ROOT/"results"/"ztf_supervised"
RES=BASE_RES  # replaced with a unique n-per-class run directory in main()
VSX="https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
ZTF="https://db.ztf.snad.space/api/v3/data/dr24/circle/full/json"
LABELS=("RRAB","DCEP","EW")
SECTORS=tuple(range(12))   # RA sectors of width 30 degrees
EXTERNAL_SECTORS=frozenset((10,11))  # RA 300-360 deg, never used for fitting
# Deliberately exclude observation-count and time-baseline shortcut features.
EXCLUDED_FEATURES=frozenset(("log_n_obs","log_timespan"))


def request_bytes(url,params,timeout=50):
    last=None
    for attempt in range(3):
        try:
            response=requests.get(url,params=params,timeout=timeout,
                                  headers={"User-Agent":"Deepnil-real-ZTF-research/1.0"})
            if response.status_code==429 or response.status_code>=500:
                raise requests.HTTPError(f"HTTP {response.status_code}")
            response.raise_for_status()
            return response.content,response.url
        except requests.RequestException as exc:
            last=exc
            if attempt<2:
                time.sleep(1.5*(2**attempt))
    raise RuntimeError(f"Real public archive unavailable after retries: {last}")


def catalog_rows(label,sector,per_sector):
    path=DATA/"vsx_catalogs"/f"{label}_RA{sector:02}_max{per_sector}.tsv"
    params={"-source":"B/vsx/vsx","-out.max":str(per_sector),
            "-out":"Name,RAJ2000,DEJ2000,Type,Period",
            "Type":label,"RAJ2000":f"{sector*30}..{(sector+1)*30}",
            "DEJ2000":">-25"}
    if path.exists():
        content=path.read_bytes()
    else:
        content,url=request_bytes(VSX,params,timeout=60)
        if not b"RAJ2000" in content or b"DEJ2000" not in content:
            raise ValueError("VSX did not return a usable catalog response")
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(content)
    collected=[]
    for row in tsv_rows(content):
        if row.get("Type","").strip()!=label:
            continue
        try:
            ra=float(row["RAJ2000"])
            dec=float(row["DEJ2000"])
            period=float(row["Period"])
        except (KeyError,TypeError,ValueError):
            continue
        if not (math.isfinite(ra) and math.isfinite(dec) and 0.2<=period<=80
                and -25<dec<=90 and sector*30<=ra<=(sector+1)*30):
            continue
        name=row["Name"].strip()
        if not name:
            continue
        collected.append({"vsx_name":name,"class_label":label,"ra":ra,"dec":dec,
                          "published_period_audit_only_days":period,"sector":sector,
                          "source_catalog_file":str(path.relative_to(ROOT)),
                          "source_catalog_sha256":hashlib.sha256(content).hexdigest()})
    return collected


def collect_vsx_candidates(per_sector):
    pool={label:{} for label in LABELS}
    for label in LABELS:
        for sector in SECTORS:
            rows=catalog_rows(label,sector,per_sector)
            pool[label][sector]=rows
            print(f"VSX {label} RA sector {sector}: {len(rows)} exact real labels",flush=True)
    # Globally deduplicate within 2 arcseconds, and reject any spatially
    # conflicting published class labels. This occurs before splitting.
    flat=[item for label in LABELS for sector in SECTORS for item in pool[label][sector]]
    if flat:
        sky=SkyCoord([r["ra"] for r in flat]*u.deg,[r["dec"] for r in flat]*u.deg)
        left,right,_,_=sky.search_around_sky(sky,2*u.arcsec)
        parents=list(range(len(flat)))
        def find(i):
            while parents[i]!=i:
                parents[i]=parents[parents[i]]
                i=parents[i]
            return i
        for a,b in zip(left,right):
            a=int(a);b=int(b)
            if a!=b:
                parents[find(b)]=find(a)
        groups={}
        for index,item in enumerate(flat):
            groups.setdefault(find(index),[]).append(index)
        keep=[]
        conflict_count=0
        for group in groups.values():
            if len({flat[i]["class_label"] for i in group})>1:
                conflict_count+=len(group)
                continue
            keep.append(flat[min(group)])
    else:
        keep=[];conflict_count=0
    buckets={label:{sector:[] for sector in SECTORS} for label in LABELS}
    for item in keep:
        buckets[item["class_label"]][item["sector"]].append(item)
    # Advance each RA sector one item per round, preventing one catalog strip
    # from supplying nearly all the examples of a class.
    ordered={}
    for label in LABELS:
        for split,sectors in (("main",tuple(s for s in SECTORS if s not in EXTERNAL_SECTORS)),
                              ("external",tuple(sorted(EXTERNAL_SECTORS)))):
            maximum=max((len(buckets[label][s]) for s in sectors),default=0)
            ordered[label,split]=[buckets[label][s][ix] for ix in range(maximum)
                                  for s in sectors if ix<len(buckets[label][s])]
    write_json(RES/"vsx_pool_audit.json",{"labels":list(LABELS),"catalog_per_sector":per_sector,
              "unique_published_labeled_sources":len(keep),
              "rejected_conflicting_label_rows":conflict_count,
              "external_ra_deg":[300,360],
              "catalog_source":"AAVSO VSX via VizieR B/vsx/vsx exact Type, period 0.2-80 days"})
    return ordered


def raw_path(rec):
    key=f'{rec["vsx_name"]}|{rec["ra"]:.7f}|{rec["dec"]:.7f}'.encode()
    return DATA/"ztf_lightcurves"/rec["class_label"]/(hashlib.sha256(key).hexdigest()+".json")


def fetch_one(rec):
    location=raw_path(rec)
    params={"ra":rec["ra"],"dec":rec["dec"],"radius_arcsec":2}
    if location.exists():
        raw=location.read_bytes()
        source_url=f"{ZTF}?ra={rec['ra']}&dec={rec['dec']}&radius_arcsec=2"
    else:
        raw,source_url=request_bytes(ZTF,params,timeout=50)
        try:
            data=json.loads(raw)
            if not isinstance(data,dict):
                raise ValueError("bad ZTF JSON response")
        except (ValueError,TypeError):
            raise RuntimeError("ZTF service returned no valid JSON measurements")
        location.parent.mkdir(parents=True,exist_ok=True)
        location.write_bytes(raw)
    records=json.loads(raw)
    if not isinstance(records,dict):
        return None,"bad_cached_json"
    options=[]
    for oid,item in records.items():
        try:
            if item["meta"]["filter"]!="zr":
                continue
            c=item["meta"]["coord"]
            sep=angular_arcsec((rec["ra"],rec["dec"]),(float(c["ra"]),float(c["dec"])))
            if sep>1.5:
                continue
            points=observed_features(item["lc"])
            if points is None:
                continue
            options.append((len(points),-sep,str(oid),points))
        except (KeyError,TypeError,ValueError):
            continue
    if not options:
        return None,"no_usable_matching_zr"
    options.sort(key=lambda r:(-r[0],-r[1],r[2]))
    n,negsep,oid,points=options[0]
    return dict(rec,ztf_oid=oid,filter="zr",n_quality_points=n,
                separation_from_catalog_arcsec=-negsep,
                source_url=source_url,source_file=str(location.relative_to(ROOT)),
                source_sha256=hashlib.sha256(raw).hexdigest()),"accepted"


def collect_real_records(ordered,per_class,workers,max_attempts):
    selected=[]
    rejection=Counter()
    ext_target=max(5,math.ceil(per_class*.2))
    main_target=per_class-ext_target
    if main_target<20:
        raise ValueError("Require --per-class >= 25 for any supervised metrics; larger is better")
    for label in LABELS:
        for split,target in (("main",main_target),("external",ext_target)):
            candidates=ordered[label,split]
            accepted=[]
            tested=0
            limit=min(len(candidates),max_attempts)
            while len(accepted)<target and tested<limit:
                batch=candidates[tested:min(tested+max(workers*3,8),limit)]
                tested+=len(batch)
                done={}
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures={pool.submit(fetch_one,rec):i for i,rec in enumerate(batch)}
                    for future in as_completed(futures):
                        index=futures[future]
                        try:
                            done[index]=future.result()
                        except Exception as exc:
                            done[index]=(None,"network_or_parse_"+type(exc).__name__)
                for index in sorted(done):
                    item,status=done[index]
                    if status=="accepted":
                        accepted.append(dict(item,split=split))
                    else:
                        rejection[f"{label}_{split}_{status}"]+=1
                accepted=accepted[:target]
                print(f"ZTF {label} {split}: {len(accepted)}/{target} valid real stars; "+
                      f"catalog candidates checked {tested}/{limit}",flush=True)
            selected.extend(accepted)
            write_json(RES/"collection_progress.json",{"accepted_so_far":len(selected),
                       "rejections":dict(rejection),"per_class_target":per_class,
                       "in_progress":{"class":label,"split":split,"accepted":len(accepted),
                                      "required":target,"tested":tested,"available":len(candidates)}})
            if len(accepted)<target:
                raise RuntimeError(f"Only {len(accepted)} {label}/{split} real stars passed; "
                                   f"need {target}. Increase --catalog-per-sector or --max-attempts; "
                                   "no fake samples or substitute labels will be inserted.")
    ids=[r["vsx_name"] for r in selected]
    if len(set(ids))!=len(ids):
        raise AssertionError("duplicate VSX source identifiers: abort")
    if len({r["ztf_oid"] for r in selected})!=len(selected):
        raise AssertionError("same ZTF light curve selected twice: abort")
    # The exact original downloaded responses and published class strings
    # remain in data/ ; no undocumented generated observation values.
    write_json(RES/"accepted_training_sources.json",{
        "source":"Published AAVSO VSX classes and real ZTF DR24 observations",
        "label_policy":"exact RRAB,DCEP,EW; no synthetic, guessed or pseudo-labels",
        "validation_design":"80/20 by distinct source in RA<300deg; external RA 300-360deg",
        "requested_per_class":per_class,"records":selected,
        "rejections":dict(rejection),"selection_seed":SEED})
    return selected


def train(records,per_class):
    feats=[]
    for i,rec in enumerate(records):
        source=json.loads((ROOT/rec["source_file"]).read_text())[rec["ztf_oid"]]
        arr=observed_features(source["lc"])
        if arr is None or len(arr)!=rec["n_quality_points"]:
            raise RuntimeError(f"Accepted raw photometry changed: {rec['vsx_name']}")
        feature,_=extract_features_from_array(arr)
        feats.append(feature)
        if (i+1)%50==0:
            print(f"Extracted real ZTF-r features {i+1}/{len(records)}",flush=True)
    columns=sorted(k for k in feats[0] if k not in EXCLUDED_FEATURES)
    x=np.asarray([[f[k] for k in columns] for f in feats])
    if not np.isfinite(x).all():
        raise RuntimeError("Real-data feature array contains NaNs; abort, never impute fake observations")
    labels=np.array([r["class_label"] for r in records])
    main=np.array([i for i,r in enumerate(records) if r["split"]=="main"])
    external=np.array([i for i,r in enumerate(records) if r["split"]=="external"])
    idx_train,idx_val=train_test_split(main,test_size=.2,random_state=SEED,stratify=labels[main])
    if set(idx_train)&set(idx_val) or set(main)&set(external):
        raise AssertionError("split overlap")
    model=RandomForestClassifier(n_estimators=400,min_samples_leaf=3,
                                 class_weight="balanced",random_state=SEED,n_jobs=-1)
    model.fit(x[idx_train],labels[idx_train])
    def evaluate(indices):
        predicted=model.predict(x[indices])
        return {"count":len(indices),"accuracy":float(accuracy_score(labels[indices],predicted)),
                "macro_f1":float(f1_score(labels[indices],predicted,labels=LABELS,average="macro")),
                "confusion_labels":list(LABELS),
                "confusion_matrix":confusion_matrix(labels[indices],predicted,labels=LABELS).tolist(),
                "classification_report":classification_report(labels[indices],predicted,
                                           labels=LABELS,output_dict=True,zero_division=0),
                "individual_predictions":[{"vsx_name":records[i]["vsx_name"],
                     "ztf_oid":records[i]["ztf_oid"],"true_published_label":str(labels[i]),
                     "model_predicted_label":str(p),"ra_deg":records[i]["ra"],
                     "dec_deg":records[i]["dec"]} for i,p in zip(indices,predicted)]}
    val=evaluate(idx_val)
    external_eval=evaluate(external)
    metrics={"status":"COMPLETED_REAL_ZTF_SUPERVISED_RUN",
             "training_count":len(idx_train),"training_class_counts":dict(Counter(labels[idx_train])),
             "validation_count":len(idx_val),"validation_class_counts":dict(Counter(labels[idx_val])),
             "external_count":len(external),"external_class_counts":dict(Counter(labels[external])),
             "validation":val,"external_ra_300_to_360":external_eval,
             "feature_names":columns,"removed_survey_shortcut_features":sorted(EXCLUDED_FEATURES),
             "seed":SEED,"source_labels":"Exact AAVSO VSX published types: RRAB,DCEP,EW",
             "limits":["These are published catalog labels, which can contain mistakes.",
                       "Tested only 3 known variable classes, without nonvariable or unknown class.",
                       "No ZTF object is claimed as newly discovered.",
                       "The external RA sky-sector split is not an independent telescope.",
                       "Running this pipeline does not prove novelty or survey-wide completeness."]}
    RES.mkdir(parents=True,exist_ok=True)
    write_json(RES/"training_metrics.json",metrics)
    write_json(RES/"fixed_splits.json",{"train":[records[i]["vsx_name"] for i in idx_train],
                                       "validation":[records[i]["vsx_name"] for i in idx_val],
                                       "external":[records[i]["vsx_name"] for i in external]})
    joblib.dump({"model":model,"feature_names":columns,"classes":list(LABELS),
                 "training_source":"Published AAVSO VSX x original public ZTF DR24 r-band",
                 "seed":SEED},RES/"ztf_supervised_model.joblib")
    print(json.dumps({"validation_count":val["count"],"validation_macro_f1":val["macro_f1"],
                      "external_count":external_eval["count"],
                      "external_macro_f1":external_eval["macro_f1"]},indent=2),flush=True)


def main():
    global RES
    parser=argparse.ArgumentParser()
    parser.add_argument("--per-class",type=int,default=200,help="real accepted ZTF sources each class (at least 25)")
    parser.add_argument("--catalog-per-sector",type=int,default=200)
    parser.add_argument("--max-attempts",type=int,default=2500,help="maximum real catalog candidates per class/split")
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--train-only",action="store_true",help="reuse existing accepted_training_sources.json")
    args=parser.parse_args()
    if args.per_class<25 or args.catalog_per_sector<10 or not 1<=args.workers<=6:
        parser.error("Require per-class>=25, catalog-per-sector>=10, workers 1..6")
    RES=BASE_RES/"runs"/f"n{args.per_class}"
    prior=RES/"vsx_pool_audit.json"
    if prior.exists() and not args.train_only:
        previous=json.loads(prior.read_text()).get("catalog_per_sector")
        if previous!=args.catalog_per_sector:
            parser.error(f"Run n{args.per_class} was created with catalog-per-sector={previous}. "
                         "Use that exact parameter to reproduce it or a new --per-class value for expansion; "
                         "existing experiment cannot silently change selection policy.")
    RES.mkdir(parents=True,exist_ok=True)
    if args.train_only:
        source=json.loads((RES/"accepted_training_sources.json").read_text())
        rows=source["records"]
        if source["requested_per_class"]!=args.per_class:
            parser.error("--per-class must match saved accepted dataset")
    else:
        pool=collect_vsx_candidates(args.catalog_per_sector)
        rows=collect_real_records(pool,args.per_class,args.workers,args.max_attempts)
    train(rows,args.per_class)


if __name__=="__main__":
    main()
