"""Audit a public ZTF lightcurve mirror against actual original IRSA rows."""

import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.table import Table

from run import ROOT, RESULTS, write_json


def main():
    raw_file=ROOT/"data"/"ztf_dr24_field_298_29_45arcsec.json"
    original_file=ROOT/"data"/"ztf_verified_irsa_oid686103400034440.dat"
    raw=raw_file.read_bytes()
    items=json.loads(raw)
    mirror=items["686103400034440"]["lc"]
    observed=np.array(sorted((float(a["mjd"]),float(a["mag"]),float(a["magerr"])) for a in mirror))
    original=Table.read(original_file,format="ascii.ipac")
    clean=original[original["catflags"]==0]
    official=np.array(sorted(zip(np.asarray(clean["hjd"],float)-2400000.5,
                                 np.asarray(clean["mag"],float),np.asarray(clean["magerr"],float))))
    max_deltas=np.max(np.abs(observed-official),axis=0) if observed.shape==official.shape else [float("inf")]*3
    passed=(observed.shape==official.shape and len(mirror)==724 and len(original)==838 and
            max_deltas[0]<1e-5 and max_deltas[1]<1e-5 and max_deltas[2]<1e-7)
    previous=json.loads((RESULTS/"ztf_scan_results.json").read_text())
    counts=previous["counts"]
    total_rows=sum(len(v["lc"]) for v in items.values())
    passed=passed and (len(items)==750 and total_rows==175419 and
                       counts["actual_measurement_rows"]==total_rows and
                       previous["archive_sha256"]==hashlib.sha256(raw).hexdigest())
    result={
        "status":"PASS" if passed else "FAIL",
        "actual_ztf_source_lightcurves":len(items),
        "actual_photometry_rows":total_rows,
        "original_irsa_rows_for_checked_oid":len(original),
        "original_irsa_good_catflags_zero_rows_for_checked_oid":len(clean),
        "mirror_rows_for_checked_oid":len(mirror),
        "max_numeric_differences_time_magnitude_error":list(map(float,max_deltas)),
        "mirror_json_sha256":hashlib.sha256(raw).hexdigest(),
        "official_ipac_file_sha256":hashlib.sha256(original_file.read_bytes()).hexdigest(),
        "limits":"One cross-source check; other SNAD entries were not individually re-fetched from IRSA. No proof of novelty or model generalization.",
    }
    write_json(RESULTS/"ztf_integrity.json",result)
    print(json.dumps(result,indent=2))
    if not passed:
        raise SystemExit(1)


if __name__=="__main__":
    main()
