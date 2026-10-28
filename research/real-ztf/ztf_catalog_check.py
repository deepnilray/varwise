"""Crossmatch *existing* ZTF rankings to real published variable catalogs.

An absence from these catalogs is NEVER sufficient evidence of a new object.
The query snapshots shipped with this project are valid only for the original
50-arcsecond field. For a new field, run --fetch --ra ... --dec ... .
"""

import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path

import requests

from run import ROOT, RESULTS, write_json
from ztf_discovery import angular_arcsec

FIELD = ROOT / "data" / "catalog_crossmatches"
CATALOGS = {
    "vsx":("B/vsx/vsx","vsx_50arcsec.tsv"),
    "gaia_dr3_variable_summary":("I/358/varisum","gaia_dr3_variables_50arcsec.tsv"),
    "ztf_2020_periodic":("J/ApJS/249/18","ztf_2020_periodic_50arcsec.tsv"),
}


def query_catalog(source, ra, dec, radius):
    base = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"
    params = {"-source":source,"-c":f"{ra} {dec:+}","-c.rs":str(radius),"-out.max":"10000"}
    response = requests.get(base,params=params,timeout=35)
    response.raise_for_status()
    return response.content, response.url


def tsv_rows(raw):
    lines = raw.decode("utf-8",errors="replace").splitlines()
    # VizieR ASU TSV: metadata comment block; table header; units; separator;
    # then observations/catalog entries. Rows are not synthesized or inferred.
    for i,line in enumerate(lines):
        if line.startswith("#") or "\t" not in line:
            continue
        keys = line.split("\t")
        has_coords = any(c in keys for c in ("RAJ2000","RA_ICRS","_RAJ2000"))
        if has_coords:
            entries = [s for s in lines[i+3:] if s.strip() and not s.startswith("#")]
            return list(csv.DictReader(io.StringIO("\n".join([line]+entries)),delimiter="\t"))
    return []


def catalog_coords(row):
    for rakey,deckey in (("RAJ2000","DEJ2000"),("RA_ICRS","DE_ICRS"),("_RAJ2000","_DEJ2000")):
        if rakey in row and deckey in row:
            try:
                return float(row[rakey]),float(row[deckey])
            except ValueError:
                return None
    return None


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fetch",action="store_true",help="fetch catalog cone results for a new sky patch")
    p.add_argument("--ra",type=float,default=298.0025)
    p.add_argument("--dec",type=float,default=29.87147)
    p.add_argument("--radius",type=float,default=50,help="query radius, arcsec")
    p.add_argument("--tag",help="same run tag used with ztf_discovery.py")
    args=p.parse_args()
    if args.tag and not re.fullmatch(r"[A-Za-z0-9_-]{1,40}",args.tag):
        p.error("Invalid tag")
    if args.fetch and not args.tag:
        p.error("Use --tag when fetching to preserve original catalog snapshots")
    results_dir=RESULTS/"fields"/args.tag if args.tag else RESULTS
    cache_dir=FIELD/args.tag if args.tag else FIELD
    if not args.fetch and (args.ra!=298.0025 or args.dec!=29.87147 or args.radius!=50):
        p.error("Cached crossmatches only cover the default field; use --fetch for another field")
    if args.tag and not args.fetch:
        p.error("Tagged runs need --fetch for catalog checks")
    scan=json.loads((results_dir/"ztf_scan_results.json").read_text())
    if scan.get("field_query"):
        center=scan["field_query"]
        if angular_arcsec((center["ra"],center["dec"]),(args.ra,args.dec))>.5:
            p.error("Catalog query center does not match lightcurve field center")
        if args.radius < center["radius_arcsec"] + 2:
            p.error("Catalog query radius must exceed lightcurve radius by at least 2 arcsec")
    catalogs={}
    audit={}
    for label,(identifier,filename) in CATALOGS.items():
        location=cache_dir/filename
        if args.fetch:
            data,url=query_catalog(identifier,args.ra,args.dec,args.radius)
            location.parent.mkdir(parents=True,exist_ok=True)
            location.write_bytes(data)
        else:
            data=location.read_bytes()
            url=next((line.split("request=",1)[1].split("\t")[0] for line in data.decode(errors="replace").splitlines() if line.startswith("#INFO\trequest=")),"archived ASU query")
        rows=tsv_rows(data)
        catalogs[label]=[(row,catalog_coords(row)) for row in rows if catalog_coords(row) is not None]
        audit[label]={"catalog_id":identifier,"entries_in_queried_field":len(rows),
                      "source_url":url,"saved_query_response":str(location.relative_to(ROOT)),
                      "raw_sha256":hashlib.sha256(data).hexdigest()}
    groups=scan["positional_groups_ranked"]
    out=[]
    for group in groups:
        position=(group["ra_deg"],group["dec_deg"])
        matches=[]
        for label,entries in catalogs.items():
            for row,coord in entries:
                sep=angular_arcsec(position,coord)
                if sep<=2.0:
                    matches.append({"catalog":label,"angular_separation_arcsec":sep,
                                    "catalog_record":row})
        g=dict(group)
        g["published_catalog_matches_within_2arcsec"]=matches
        g["catalog_status"]=("MATCHES_PUBLISHED_CATALOG" if matches else
                             "NO_MATCH_IN_THESE_THREE_CATALOGS__NOT_PROOF_OF_DISCOVERY")
        pday=g["representative_period_days"]
        g["strong_daily_alias_risk"]=any(abs(pday-k)/k <.025 for k in (.2,.25,1/3,.5,1.,2.))
        out.append(g)
    write_json(results_dir/"ztf_catalog_check.json",{
        "warning":"Catalog incompleteness, blends, period aliases and artifacts remain; NOT a discovery catalog.",
        "search_field":{"ra":args.ra,"dec":args.dec,"catalog_radius_arcsec":args.radius},
        "catalog_query_provenance":audit,
        "count_candidate_positions":len(out),
        "positions_matching_at_least_one_published_catalog":sum(bool(g["published_catalog_matches_within_2arcsec"]) for g in out),
        "positions_without_match_in_these_three_catalogs":sum(not g["published_catalog_matches_within_2arcsec"] for g in out),
        "ranked_positions":out})
    print(json.dumps({"count_candidate_positions":len(out),
                      "known_catalog_matches":sum(bool(g["published_catalog_matches_within_2arcsec"]) for g in out),
                      "top_ranked":{"oid":out[0]["representative_oid"],
                                    "catalog_status":out[0]["catalog_status"],
                                    "daily_alias_risk":out[0]["strong_daily_alias_risk"]}},indent=2))


if __name__=="__main__":
    main()
