"""Independently re-read every ZTF training star and its VSX class source."""

import hashlib
import json
import argparse
from collections import Counter

from run import ROOT, write_json
from ztf_catalog_check import tsv_rows
from ztf_discovery import angular_arcsec, observed_features
from ztf_supervised import LABELS, BASE_RES


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--per-class",type=int,default=200)
    args=parser.parse_args()
    res=BASE_RES/"runs"/f"n{args.per_class}"
    dataset=json.loads((res/"accepted_training_sources.json").read_text())
    rows=dataset["records"]
    splits=json.loads((res/"fixed_splits.json").read_text())
    metrics=json.loads((res/"training_metrics.json").read_text())
    errors=[]
    catcache={}
    for row in rows:
        name=row["vsx_name"]
        source=ROOT/row["source_file"]
        catalog=ROOT/row["source_catalog_file"]
        if not source.is_file() or not catalog.is_file():
            errors.append("missing_original_file:"+name)
            continue
        raw=source.read_bytes()
        catraw=catalog.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row["source_sha256"]:
            errors.append("ztf_checksum_mismatch:"+name)
        if hashlib.sha256(catraw).hexdigest()!=row["source_catalog_sha256"]:
            errors.append("vsx_checksum_mismatch:"+name)
        key=str(catalog)
        if key not in catcache:
            catcache[key]=tsv_rows(catraw)
        matches=[x for x in catcache[key] if x.get("Name","").strip()==name]
        if not matches or matches[0].get("Type","").strip()!=row["class_label"]:
            errors.append("no_exact_published_label:"+name)
        try:
            obj=json.loads(raw)[row["ztf_oid"]]
            coordinate=obj["meta"]["coord"]
            if obj["meta"]["filter"]!="zr" or angular_arcsec((row["ra"],row["dec"]),
                     (float(coordinate["ra"]),float(coordinate["dec"])))>1.5:
                errors.append("wrong_filter_or_position:"+name)
            arr=observed_features(obj["lc"])
            if arr is None or len(arr)!=row["n_quality_points"]:
                errors.append("observed_point_count_mismatch:"+name)
        except (KeyError,ValueError,TypeError):
            errors.append("broken_source_record:"+name)
    names=[row["vsx_name"] for row in rows]
    if len(names)!=len(set(names)):
        errors.append("duplicate_source_names")
    if len({r["ztf_oid"] for r in rows})!=len(rows):
        errors.append("duplicate_ztf_lightcurves")
    train=set(splits["train"])
    val=set(splits["validation"])
    ext=set(splits["external"])
    if train&val or train&ext or val&ext or (train|val|ext)!=set(names):
        errors.append("source_leakage_or_incomplete_split")
    for row in rows:
        desired="external" if row["ra"]>=300 else "main"
        if row["split"]!=desired:
            errors.append("sky_sector_mismatch:"+row["vsx_name"])
    if (len(train)!=metrics["training_count"] or len(val)!=metrics["validation_count"] or
            len(ext)!=metrics["external_count"]):
        errors.append("metrics_split_count_mismatch")
    target=dataset["requested_per_class"]
    if Counter(r["class_label"] for r in rows)!=Counter({x:target for x in LABELS}):
        errors.append("wrong_class_balance")
    out={"status":"PASS" if not errors else "FAIL",
         "errors":errors,"verified_real_source_count":len(rows),
         "verified_training_count":len(train),"verified_validation_count":len(val),
         "verified_external_count":len(ext),
         "label_origin":"Published AAVSO VSX exact class strings, checked against archived catalog rows",
         "photometry_origin":"Original public ZTF DR24 measurements in archived SNAD query responses",
         "no_generated_lightcurves":True if not errors else "not established"}
    write_json(res/"training_integrity_audit.json",out)
    print(json.dumps(out,indent=2))
    if errors:
        raise SystemExit(1)


if __name__=="__main__":
    main()
