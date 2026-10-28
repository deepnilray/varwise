"""Independently audit every published model data point against raw OGLE files."""

from collections import Counter
from hashlib import sha256
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent
R=ROOT/"results"


def read(name):
    return json.loads((R/name).read_text())


def main():
    lmc=read("source_manifest.json")["records"]
    smc=read("external_smc_manifest.json")
    split=read("split.json")
    metrics=read("metrics.json")
    external=read("external_smc_metrics.json")
    errors=[]
    seen={}
    for record in lmc+smc:
        source_id=record["source_id"]
        path=ROOT/record["local_path"]
        if not path.is_file():
            errors.append("missing:"+source_id)
            continue
        if sha256(path.read_bytes()).hexdigest()!=record["sha256"]:
            errors.append("checksum_mismatch:"+source_id)
        if source_id in seen:
            errors.append("duplicate_source_id:"+source_id)
        seen[source_id]=record["class_label"]
        if not record["source_url"].startswith("https://ftp.astrouw.edu.pl/ogle/ogle4/OCVS/"):
            errors.append("unexpected_data_origin:"+source_id)

    train=set(split["train"])
    validation=set(split["validation"])
    lmc_ids={r["source_id"] for r in lmc}
    smc_ids={r["source_id"] for r in smc}
    if train & validation:
        errors.append("train_validation_overlap")
    if train | validation != lmc_ids:
        errors.append("lmc_split_mismatch")
    if (train | validation) & smc_ids:
        errors.append("external_data_leakage")
    if len(train)!=1200 or len(validation)!=300 or len(smc_ids)!=300:
        errors.append("wrong_group_counts")
    if Counter(seen[i] for i in train)!=Counter({"RRab":400,"Cepheid_F":400,"Eclipsing":400}):
        errors.append("incorrect_training_balance")
    if Counter(seen[i] for i in validation)!=Counter({"RRab":100,"Cepheid_F":100,"Eclipsing":100}):
        errors.append("incorrect_validation_balance")
    if metrics["validation_count"]!=300 or external["actual_external_star_count"]!=300:
        errors.append("metrics_manifest_mismatch")

    result={
        "status":"PASS" if not errors else "FAIL",
        "errors":errors,
        "checked_source_files":len(lmc)+len(smc),
        "unique_source_ids":len(seen),
        "correct_sha256_count":len(lmc)+len(smc)-sum(x.startswith("checksum_mismatch") for x in errors),
        "lmc_training_records":len(train),
        "lmc_validation_records":len(validation),
        "smc_external_records":len(smc_ids),
        "all_sources_are_official_ogle_urls":not any(x.startswith("unexpected_data_origin") for x in errors),
    }
    (R/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if errors:
        raise SystemExit(1)


if __name__=="__main__":
    main()
