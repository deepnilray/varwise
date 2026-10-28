"""Make the complete input-audited Windows research bundle (no sample invention)."""

from hashlib import sha256
from pathlib import Path
import json
import zipfile


ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/"Deepnil_Real_Astronomy_Discovery_Screening_v2_Windows.zip"
R=ROOT/"results"


def read(path):
    return json.loads(path.read_text())


def main():
    if read(R/"audit.json")["status"]!="PASS":
        raise RuntimeError("OGLE 1,800-file audit did not pass")
    if read(R/"ztf_integrity.json")["status"]!="PASS":
        raise RuntimeError("ZTF live-pilot mirror check did not pass")
    run_paths={}
    raw_training=set()
    for n,expected in ((25,75),(200,600)):
        folder=R/"ztf_supervised"/"runs"/f"n{n}"
        if read(folder/"training_integrity_audit.json")["status"]!="PASS":
            raise RuntimeError(f"Real-ZTF n{n} training audit did not pass")
        selected=read(folder/"accepted_training_sources.json")["records"]
        if len(selected)!=expected:
            raise RuntimeError(f"ZTF n{n} source quota not verified")
        for row in selected:
            raw_training.add(row["source_file"])
            raw_training.add(row["source_catalog_file"])
        run_paths[n]=folder
    lmc=read(R/"source_manifest.json")["records"]
    smc=read(R/"external_smc_manifest.json")
    raw_ogle={item["local_path"] for item in lmc+smc}
    if len(raw_ogle)!=1800:
        raise RuntimeError("OGLE source manifest not distinct 1,800")
    file_set=set(raw_training|raw_ogle)
    input_files=set(raw_training|raw_ogle)
    # Exact published catalog texts, original ZTF pilot, official IPAC audit,
    # and original code, without packing temporary/smoke-test sources.
    file_set.update(str(f.relative_to(ROOT)) for f in (ROOT/"data"/"catalogs").rglob("*.dat"))
    file_set.update(str(f.relative_to(ROOT)) for f in (ROOT/"data"/"catalog_crossmatches").glob("*.tsv"))
    file_set.update(("data/ztf_dr24_field_298_29_45arcsec.json",
                     "data/ztf_verified_irsa_oid686103400034440.dat"))
    input_files.update(f for f in file_set if f.startswith("data/"))
    # Preserve real attempted/catalog download provenance for the two completed
    # supervised experiments; source observations not selected aren't required
    # to rerun an archived fixed split.
    file_set.update(str(f.relative_to(ROOT)) for f in ROOT.glob("*.py"))
    file_set.update(str(f.relative_to(ROOT)) for f in ROOT.glob("*.bat"))
    file_set.update(str(f.relative_to(ROOT)) for f in ROOT.glob("*.md"))
    file_set.add("requirements.txt")
    input_files.update(f for f in file_set if not f.startswith("results/"))
    file_set.update(str(f.relative_to(ROOT)) for f in R.glob("*") if f.is_file())
    for folder in run_paths.values():
        file_set.update(str(f.relative_to(ROOT)) for f in folder.glob("*") if f.is_file())
    # Source selection manifest acts as immutable input to --train-only.
    for n in run_paths:
        input_files.add(f"results/ztf_supervised/runs/n{n}/accepted_training_sources.json")
    file_set.update(input_files)
    if any(not (ROOT/name).is_file() for name in file_set):
        raise FileNotFoundError("At least one actual audited input is missing")
    checksums={name:sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(input_files)}
    checksum_path=R/"package_input_checksums.json"
    checksum_path.write_text(json.dumps({"source":"Only immutable measured inputs, exact published labels, and executable code",
            "number_ogle_verified_lightcurves":1800,
            "number_verified_ztf_training_rows":600,
            "sha256_by_relative_path":checksums},indent=2)+"\n")
    file_set.add(str(checksum_path.relative_to(ROOT)))
    with zipfile.ZipFile(OUT,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name in sorted(file_set):
            archive.write(ROOT/name,arcname="ogle_research/"+name)
    with zipfile.ZipFile(OUT) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Bad ZIP CRC")
    print(json.dumps({"archive":str(OUT),"bytes":OUT.stat().st_size,
                      "number_files":len(file_set),
                      "immutable_inputs_with_sha256":len(input_files),
                      "original_ogle_lightcurves":len(raw_ogle),
                      "actual_ztf_training_raw_response_and_catalog_files":len(raw_training)},indent=2))


if __name__=="__main__":
    main()
