"""Package the audited, actually-used observations and code for handoff."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "Deepnil_OGLE_ZTF_Real_Object_Hunt_Windows.zip"


def main():
    audit = json.loads((ROOT / "results" / "audit.json").read_text())
    if audit["status"] != "PASS" or audit["checked_source_files"] != 1800:
        raise RuntimeError("Run audit.py successfully before packaging")
    ztf_audit=json.loads((ROOT/"results"/"ztf_integrity.json").read_text())
    if ztf_audit["status"]!="PASS" or ztf_audit["actual_photometry_rows"]!=175419:
        raise RuntimeError("Run ztf_integrity.py successfully before packaging")
    lmc = json.loads((ROOT / "results" / "source_manifest.json").read_text())["records"]
    smc = json.loads((ROOT / "results" / "external_smc_manifest.json").read_text())
    observed_files = {r["local_path"] for r in lmc + smc}
    assert len(observed_files) == 1800
    supporting_files = {
        "README.md", "requirements.txt", "run.py", "external_smc.py", "audit.py", "package.py",
        "ztf_discovery.py", "ztf_catalog_check.py", "ztf_integrity.py", "ztf_plot.py",
        "windows_run.bat", "windows_retrain_ogle.bat", "windows_hunt_new_field.bat",
        "data/ztf_dr24_field_298_29_45arcsec.json",
        "data/ztf_verified_irsa_oid686103400034440.dat",
    }
    supporting_files.update(str(p.relative_to(ROOT)) for p in (ROOT / "data" / "catalog_crossmatches").glob("*.tsv"))
    supporting_files.update(str(p.relative_to(ROOT)) for p in (ROOT / "data" / "catalogs").rglob("*.dat"))
    supporting_files.update(str(p.relative_to(ROOT)) for p in (ROOT / "results").glob("*") if p.is_file())
    files = sorted(observed_files | supporting_files)
    if any(not (ROOT / rel).is_file() for rel in files):
        raise FileNotFoundError("Manifest references an absent file")
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel in files:
            zf.write(ROOT / rel, arcname=f"ogle_research/{rel}")
    with zipfile.ZipFile(OUT) as zf:
        if zf.testzip() is not None:
            raise RuntimeError("Archive CRC check failed")
        raw = [n for n in zf.namelist() if "/data/lightcurves/" in n]
        assert len(raw) == 1800
    print(json.dumps({"archive": str(OUT), "bytes": OUT.stat().st_size,
                      "actual_observation_files": len(observed_files), "total_files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
