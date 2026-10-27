#!/usr/bin/env python3
"""
varwise_gaia_hunt.py - search for optical variability in the already-published
VarWISE infrared-variable catalogue. This CANNOT discover a previously unseen
object by definition; it may flag optical behaviour not yet separately listed.

RUN HERE OR ON YOUR COMPUTER (internet needed for REAL training):
    pip install astropy scikit-learn pandas scipy matplotlib joblib pyvo astroquery requests
    python varwise_gaia_hunt.py selftest                 # offline logic test, ~2 min
    python varwise_gaia_hunt.py brutetest --out stress   # unseen signal shapes (synthetic)
    python varwise_gaia_hunt.py probe                    # tests every archive + shows VarWISE table/columns
    python varwise_gaia_hunt.py train --out training_only  # training+sealed validation, no VarWISE
    python varwise_gaia_hunt.py auto --out strict_search --hours 2 --target 5
        # first fit real model, query withheld Gaia/ZTF ONLY AFTER FIT,
        # validate, then (and ONLY then) search untouched VarWISE tiles
    python varwise_gaia_hunt.py run --ra0 150 --dec0 30 --size 4 --out training_only
        # requires existing validated model; refuses training/validation tiles

Pipeline (what "auto" does)
  1. Select TRAINING sky tiles and entirely SEALED VALIDATION sky tiles.
     Query Gaia training labels and ZTF training curves. Fit model. ONLY THEN
     query validation Gaia labels and ZTF light curves; require held-out AUC
     >=0.85 and shuffled-label control near 0.5, else HARD STOP. Approx 80/20.
  2. AFTER successful real-sky validation, query VarWISE Extended target tiles;
     recursively split
     every dense tile, count rows, reject incomplete TAP responses.
  3. By default search all VarWISE positions, NO Gaia match or labels required.
     With --require-gaia: crossmatch G=12..19 unique Gaia DR3 counterparts.
     Flag whether Gaia already has a variability solution (gaiadr3.vari_summary).
  4. ZTF light curves (IRSA ZTF API), Lomb-Scargle features per source.
  5. Score target stars ONLY with the separately fitted, validated model;
     no untrained fallback. Proxy negatives = Gaia objects without published
     variability solutions, NOT proven constant stars. Independent temporal
     period confirmation is required for periodic candidates.
  6. Prior-art check: VSX, ASAS-SN, ZTF periodic, Gaia, SIMBAD, NED.
     Any known counterpart -> known_objects.csv; outages -> UNVERIFIED only.
  7. Writes candidates.csv (all UNCONFIRMED), known_objects.csv, unverified.csv,
     anomaly_leads.csv, plots; save validation report and sealed model.

Status of claims: selftest and brutetest measure SIMULATED signals, not real-sky
performance. `probe`/`smoke` test actual archive access. Source entries are
ALREADY in VarWISE; candidates are never confirmed discoveries, TNS alert
matching and imaging still require external human checks. A full 1.9M-source
scan is impractical with per-target online light-curve queries: use bulk ZTF
ingest and dedicated compute before claiming exhaustive survey completeness.
Nothing here submits or posts anything anywhere.
"""
import argparse, os, sys, time, json, hashlib, pickle, warnings, tempfile
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

IRSA_TAP = "https://irsa.ipac.caltech.edu/TAP"
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap"
ZTF_API = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"
ZTF_HATS_OBJECTS = "s3://ipac-irsa-ztf/ztf/enhanced/dr24/objects/hats"
MATCH_ARCSEC = 3.0

FEATURES = ["log_fap", "power", "amp", "amp_over_err", "chi2red", "log_period",
            "log_n", "mean_mag", "skew", "iqr_over_err"]
VALIDATION_PROTOCOL = "sealed-tiles-after-fit-v2"


def code_sha256():
    with open(__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


CODE_LOADED_SHA256 = code_sha256()

# ---------------------------------------------------------------- features ---
def alias_flag(period):
    """True if the period sits on a common sampling alias (1 d, 0.5 d, 2 d, lunar ~29.5 d)."""
    for a in (0.5, 1.0, 2.0, 29.53, 365.25):
        if abs(period - a) / a < 0.02:
            return True
    return False


def lc_features(t, m, e, fmin=1 / 200.0, fmax=10.0, spp=5):
    """Lomb-Scargle features. Returns dict or None if too few points."""
    from astropy.timeseries import LombScargle
    from scipy.stats import skew
    t = np.asarray(t, float); m = np.asarray(m, float); e = np.asarray(e, float)
    ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(e) & (e > 0)
    t, m, e = t[ok], m[ok], e[ok]
    if len(t) < 40:
        return None
    # 5-sigma clip about the median
    med = np.median(m); mad = 1.4826 * np.median(np.abs(m - med)) + 1e-6
    k = np.abs(m - med) < 5 * mad + 3 * np.median(e)
    t, m, e = t[k], m[k], e[k]
    if len(t) < 40:
        return None
    ls = LombScargle(t, m, e)
    freq, power = ls.autopower(minimum_frequency=fmin, maximum_frequency=fmax,
                               samples_per_peak=spp)
    i = int(np.argmax(power))
    pbest = 1.0 / freq[i]
    fap = float(ls.false_alarm_probability(power[i], method="baluev",
                                           minimum_frequency=fmin, maximum_frequency=fmax))
    fap = min(max(fap, 1e-300), 1.0)
    chi2red = float(np.sum(((m - np.average(m, weights=1 / e ** 2)) / e) ** 2) / (len(m) - 1))
    p5, p95 = np.percentile(m, [5, 95])
    q1, q3 = np.percentile(m, [25, 75])
    return dict(log_fap=np.log10(fap), power=float(power[i]), amp=float(p95 - p5),
                amp_over_err=float((p95 - p5) / np.median(e)), chi2red=chi2red,
                log_period=float(np.log10(pbest)), log_n=float(np.log10(len(t))),
                mean_mag=float(np.mean(m)), skew=float(skew(m)),
                iqr_over_err=float((q3 - q1) / np.median(e)),
                period=float(pbest), alias=bool(alias_flag(pbest)), n=int(len(t)))


def outlier_lead_features(t, m, e):
    """Model-free, NON-PERIODIC variability lead: repeated significant excursions.

    Flags both previously untrained shapes and instrumental contamination.
    ALWAYS export to an unverified queue; this is not astrophysical proof.
    """
    t, m, e = (np.asarray(x, float) for x in (t, m, e))
    ok = np.isfinite(t) & np.isfinite(m) & np.isfinite(e) & (e > 0)
    t, m, e = t[ok], m[ok], e[ok]
    if len(t) < 60:
        return dict(flag=False, excursion_count=0, excursion_nights=0, max_excursion_sigma=0)
    center = np.median(m)
    threshold = np.maximum(5 * e, 0.12)  # both statistical and absolute floor
    high, low = m - center > threshold, center - m > threshold
    ind = high if high.sum() >= low.sum() else low
    n = int(ind.sum())
    nights = len(np.unique(np.floor(t[ind]).astype(int)))
    max_z = float(np.max(np.abs(m - center) / e))
    # A short real event may occupy only a few epochs and hence have a modest
    # global chi-squared. Do not discard it for that reason.
    return dict(flag=bool(n >= 4 and nights >= 3 and max_z >= 6),
                excursion_count=n, excursion_nights=int(nights),
                max_excursion_sigma=max_z)


# ------------------------------------------------------------------ model ---
def sky_cell(ra, dec, cell=1.0):
    return f"{int(np.floor(ra / cell))}_{int(np.floor(dec / cell))}"


def is_test_cell(cellname):
    """Deterministic 20% of sky cells go to the held-out set."""
    h = int(hashlib.md5(cellname.encode()).hexdigest(), 16)
    return (h % 5) == 0


def spatial_test_mask(df, seed=0):
    """Find an ~80/20 SOURCE split while NEVER mixing sky cells.

    A tiny patch may contain only a few giant cells; then a scientifically
    meaningful blocked 80/20 split does not exist and training must refuse.
    """
    from sklearn.model_selection import GroupShuffleSplit
    group = df["cell"].astype(str).to_numpy()
    label = df["label"].to_numpy()
    n = len(df)
    if n < 120 or len(np.unique(group)) < 10:
        raise RuntimeError(f"insufficient spatial diversity for blocked 80/20: "
                           f"{n} sources across {len(np.unique(group))} cells; "
                           "sample larger, separated sky regions")
    splitter = GroupShuffleSplit(n_splits=200, test_size=0.20, random_state=seed)
    best = None
    for tr, te in splitter.split(np.zeros(n), label, groups=group):
        if len(tr) < 90 or len(te) < 20:
            continue
        if np.sum(label[tr] == 1) < 20 or np.sum(label[tr] == 0) < 20:
            continue
        if np.sum(label[te] == 1) < 10 or np.sum(label[te] == 0) < 10:
            continue
        fraction = len(te) / n
        if abs(fraction - 0.20) > 0.05:
            continue
        imbalance = abs(float(label[te].mean()) - float(label.mean()))
        quality = abs(fraction - .20) + .1 * imbalance
        if best is None or quality < best[0]:
            best = (quality, te)
    if best is None:
        raise RuntimeError("cannot construct a geographically disjoint 80/20 "
                           "split with enough examples of both labels; "
                           "enlarge the training sky regions")
    mask = np.zeros(n, dtype=bool)
    mask[best[1]] = True
    return mask


def train_eval(df, seed=0, verbose=True):
    """df needs FEATURES, 'label' (1 known variable, 0 constant), 'cell'.
    Returns (model, auc_test, auc_shuffled_control)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    test = spatial_test_mask(df, seed)
    tr, te = df[~test], df[test]
    if tr["label"].nunique() < 2 or te["label"].nunique() < 2 or len(te) < 10:
        raise RuntimeError(f"split unusable: train={len(tr)} test={len(te)} "
                           f"(need both classes in both). Use a larger box or more stars.")
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08,
                                         max_iter=200, random_state=seed)
    clf.fit(tr[FEATURES], tr["label"])
    auc = roc_auc_score(te["label"], clf.predict_proba(te[FEATURES])[:, 1])
    # planted-failure control: shuffled training labels, mean over 10 shuffles, must be ~0.5
    rng = np.random.default_rng(seed)
    a0 = []
    for _ in range(10):
        c0 = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08,
                                            max_iter=100, random_state=seed)
        c0.fit(tr[FEATURES], rng.permutation(tr["label"].values))
        a0.append(roc_auc_score(te["label"], c0.predict_proba(te[FEATURES])[:, 1]))
    auc0 = float(np.mean(a0))
    if verbose:
        print(f"[80/20] train n={len(tr)} ({len(tr)/len(df):.1%})  "
              f"held-out n={len(te)} ({len(te)/len(df):.1%}) "
              f"over disjoint sky cells")
        print(f"[80/20] held-out AUC = {auc:.3f}   shuffled-label control AUC = {auc0:.3f}")
    return clf, auc, auc0


# ---------------------------------------------------------------- selftest ---
def synth_lc(rng, periodic, n=160, span=1000.0):
    t = np.sort(rng.uniform(0, span, n))
    e = np.full(n, 0.03) * rng.uniform(0.8, 1.3)
    m0 = rng.uniform(13, 18)
    if periodic:
        P = 10 ** rng.uniform(np.log10(0.25), np.log10(20))
        A = rng.uniform(0.08, 0.5)
        m = m0 + A * np.sin(2 * np.pi * t / P + rng.uniform(0, 6.28))
    else:
        P = np.nan
        m = np.full(n, m0)
    return t, m + rng.normal(0, e), e, P


def selftest():
    rng = np.random.default_rng(42)
    # (a) planted period must be recovered
    t, m, e, P = synth_lc(rng, True)
    P = 3.7; m = 15 + 0.3 * np.sin(2 * np.pi * t / P) + rng.normal(0, e)
    f = lc_features(t, m, e)
    rec = abs(f["period"] - P) / P < 0.01
    print(f"[test 1] planted P=3.7 d recovered as {f['period']:.4f} d -> {'PASS' if rec else 'FAIL'}")
    # (b) flat star must NOT look significant
    t, m, e, _ = synth_lc(rng, False)
    f0 = lc_features(t, m, e)
    flat_ok = f0["log_fap"] > -4
    print(f"[test 2] flat star log10 FAP = {f0['log_fap']:.1f} -> {'PASS' if flat_ok else 'FAIL'}")
    # (c) model with sky-cell split, plus shuffled-label control
    rows = []
    for k in range(600):
        periodic = k % 2 == 0
        t, m, e, P = synth_lc(rng, periodic)
        ft = lc_features(t, m, e, fmax=10.0, spp=4)
        ft["label"] = int(periodic)
        ft["cell"] = sky_cell(rng.uniform(0, 30), rng.uniform(0, 30))
        rows.append(ft)
    df = pd.DataFrame(rows)
    clf, auc, auc0 = train_eval(df)
    ok_model = auc > 0.9 and abs(auc0 - 0.5) < 0.1
    print(f"[test 3] held-out AUC>0.9 and shuffled control within 0.5+-0.1 -> {'PASS' if ok_model else 'FAIL'}")
    # (d) the control itself must be able to fail: feed it labels equal to a feature
    df_leak = df.copy(); df_leak["power"] = df_leak["label"] * 1.0
    _, a_l, a_l0 = train_eval(df_leak, verbose=False)
    ctrl_ok = a_l > 0.95 and abs(a_l0 - 0.5) < 0.1
    print(f"[test 4] leak planted: AUC {a_l:.2f} vs shuffled {a_l0:.2f} (control separates) -> "
          f"{'PASS' if ctrl_ok else 'FAIL'}")
    allok = rec and flat_ok and ok_model and ctrl_ok
    print("SELFTEST", "PASSED" if allok else "FAILED")
    return 0 if allok else 1


# ----------------------------------------------------------------- network ---
def tap(url):
    import pyvo
    svc = pyvo.dal.TAPService(url)
    orig = svc.run_sync

    def run_sync_retry(*a, **k):
        for i in range(5):
            try:
                return orig(*a, **k)
            except Exception as ex:
                if type(ex).__name__ in ("DALQueryError", "DALFormatError") and "ProxyError" not in str(ex) \
                        and "Connection" not in str(ex) and "Timeout" not in str(ex):
                    raise                          # a genuinely bad query: do not retry
                if i == 4:
                    raise
                time.sleep(3 * (i + 1))
    svc.run_sync = run_sync_retry
    return svc


def probe():
    import requests
    hosts = {
        "IRSA TAP": IRSA_TAP + "/capabilities",
        "Gaia TAP": GAIA_TAP + "/capabilities",
        "ZTF API": ZTF_API + "?POS=CIRCLE%20150%2030%200.001&FORMAT=csv",
        "VizieR": "https://vizier.cds.unistra.fr/viz-bin/votable",
        "SIMBAD": "https://simbad.cds.unistra.fr/simbad/sim-tap/capabilities",
    }
    for k, u in hosts.items():
        try:
            r = requests.get(u, timeout=30)
            print(f"{k:9s} HTTP {r.status_code}")
        except Exception as ex:
            print(f"{k:9s} FAILED {type(ex).__name__}: {ex}")
    irsa = tap(IRSA_TAP)
    q = ("SELECT table_name, description FROM TAP_SCHEMA.tables "
         "WHERE lower(table_name) LIKE '%varwise%' OR lower(description) LIKE '%varwise%'")
    t = irsa.run_sync(q).to_table().to_pandas()
    print("\nVarWISE tables found:\n", t.to_string())
    for name in t["table_name"].head(3):
        c = irsa.run_sync(f"SELECT column_name, datatype, description FROM TAP_SCHEMA.columns "
                          f"WHERE table_name='{name}'").to_table().to_pandas()
        print(f"\n--- columns of {name} ---\n", c.to_string())


def pick_radec(cols):
    low = {c.lower(): c for c in cols}
    for a, b in (("ra", "dec"), ("ra_deg", "dec_deg"), ("raj2000", "dej2000"),
                 ("w1_ra", "w1_dec"), ("ra_mean", "dec_mean")):
        if a in low and b in low:
            return low[a], low[b]
    raise RuntimeError(f"cannot guess RA/Dec columns from {list(cols)}; pass --ra-col/--dec-col")


def ra_where(column, low, high):
    """Half-open RA interval, including intervals crossing 0 degrees."""
    if high - low >= 360:
        return "1=1"
    a, b = low % 360, high % 360
    if a < b:
        return f"({column}>={a:.12f} AND {column}<{b:.12f})"
    return f"({column}>={a:.12f} OR {column}<{b:.12f})"


def catalog_rectangle(column_ra, column_dec, ra_low, ra_high, dec_low, dec_high):
    """Exact, nonoverlapping coordinate rectangles, not overlapping TAP BOX cones."""
    d0, d1 = max(-90.0, dec_low), min(90.0, dec_high)
    high_op = "<=" if d1 == 90.0 else "<"
    return (f"{ra_where(column_ra, ra_low, ra_high)} AND "
            f"{column_dec}>={d0:.12f} AND {column_dec}{high_op}{d1:.12f}")


def rectangle_intersects(a, b):
    """Open-area overlap of (ra_low, ra_high, dec_low, dec_high), RA circular."""
    if max(a[2], b[2]) >= min(a[3], b[3]):
        return False
    def intervals(lo, hi):
        if hi-lo >= 360:
            return [(0., 360.)]
        first = lo % 360
        last = first + (hi-lo)
        return [(first, min(last, 360.))] + ([(0., last-360.)] if last > 360 else [])
    return any(max(x0,y0) < min(x1,y1)
               for x0,x1 in intervals(a[0],a[1])
               for y0,y1 in intervals(b[0],b[1]))


def fetch_complete_rectangle(service, table, rac, decc, bounds, page_size, depth=0):
    """Count every source, recursively subdivide dense tiles, and refuse truncation.

    Avoid TAP OFFSET (not portable across providers); every leaf is disjoint.
    """
    r0, r1, d0, d1 = bounds
    where = catalog_rectangle(rac, decc, r0, r1, d0, d1)
    c = service.run_sync(f"SELECT COUNT(*) AS n FROM {table} WHERE {where}")
    count = int(c.to_table().to_pandas().iloc[0, 0])
    if count == 0:
        return []
    if count > page_size:
        if depth >= 35 or min(r1-r0, d1-d0) < 1e-9:
            raise RuntimeError(f"cannot safely subdivide dense VarWISE tile {bounds}: {count} rows")
        if (r1-r0)*max(np.cos(np.radians((d0+d1)/2)), 0.05) >= d1-d0:
            mid = (r0+r1)/2
            children = [(r0,mid,d0,d1), (mid,r1,d0,d1)]
        else:
            mid = (d0+d1)/2
            children = [(r0,r1,d0,mid), (r0,r1,mid,d1)]
        frames = []
        for child in children:
            frames.extend(fetch_complete_rectangle(service, table, rac, decc,
                                                    child, page_size, depth+1))
        return frames
    # A count without a complete extraction is a hard failure, not a success.
    q = f"SELECT TOP {count} * FROM {table} WHERE {where}"
    df = service.run_sync(q, maxrec=page_size+1).to_table().to_pandas()
    if len(df) != count:
        raise RuntimeError(f"VarWISE TAP returned {len(df)} of {count} rows for {bounds}; "
                           "query incomplete, refusing to mark tile done")
    return [df]


def get_varwise(args):
    irsa = tap(IRSA_TAP)
    table = args.table
    if not table:
        t = irsa.run_sync("SELECT table_name FROM TAP_SCHEMA.tables WHERE "
                          "lower(table_name) LIKE '%varwise%'").to_table().to_pandas()
        if t.empty:
            raise RuntimeError("no VarWISE table found in IRSA TAP; run `probe` and pass --table")
        print("VarWISE tables:", list(t["table_name"]))
        # The associations table contains detection IDs but NO sky positions.
        # It can sort first, causing the original live smoke test to fail.
        names = list(t["table_name"])
        prefer = next((x for x in names if x.lower().split(".")[-1] == "varwiseext"), None)
        table = prefer or next((x for x in names if "assoc" not in x.lower()), None)
        if table is None:
            raise RuntimeError("VarWISE has no positional catalogue; specify --table explicitly")
    cols = irsa.run_sync(f"SELECT column_name FROM TAP_SCHEMA.columns WHERE table_name='{table}'"
                         ).to_table().to_pandas()["column_name"].tolist()
    rac, decc = (args.ra_col, args.dec_col) if args.ra_col else pick_radec(cols)
    if args.max_vw < 2:
        raise ValueError("--max-vw must be >= 2 (it is a page size, not a survey limit)")
    ra_width = getattr(args, "ra_width", None)
    dec_height = getattr(args, "dec_height", None)
    ra_width = (args.size / max(np.cos(np.radians(args.dec0)), .05)
                if ra_width is None else ra_width)
    dec_height = args.size if dec_height is None else dec_height
    bounds = (args.ra0-ra_width/2, args.ra0+ra_width/2,
              args.dec0-dec_height/2, args.dec0+dec_height/2)
    print(f"VarWISE complete rectangle {bounds} in {table} (leaf page size {args.max_vw})")
    parts = fetch_complete_rectangle(irsa, table, rac, decc, bounds, args.max_vw)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    df = df.rename(columns={rac: "ra", decc: "dec"})
    print(f"VarWISE objects in box: {len(df)}")
    return df


GAIA_JOIN = """
SELECT m.uid AS uid, g.source_id, g.ra AS g_ra, g.dec AS g_dec, g.phot_g_mean_mag,
       g.bp_rp, g.ruwe, g.phot_variable_flag,
       3600.0*DISTANCE(POINT('ICRS',m.ra,m.dec),POINT('ICRS',g.ra,g.dec)) AS sep
FROM TAP_UPLOAD.t AS m
JOIN gaiadr3.gaia_source AS g
  ON 1=CONTAINS(POINT('ICRS',g.ra,g.dec),CIRCLE('ICRS',m.ra,m.dec,{r}))
WHERE g.phot_g_mean_mag BETWEEN 12 AND 19
"""


def gaia_match(vw):
    from astropy.table import Table
    if vw.empty:
        return pd.DataFrame(columns=["uid", "source_id", "sep", "vari_id", "n_gaia",
                                     "g_ra", "g_dec", "phot_g_mean_mag", "bp_rp", "ruwe"])
    gaia = tap(GAIA_TAP)
    up = Table.from_pandas(vw[["uid", "ra", "dec"]])
    out = []
    for i in range(0, len(up), 5000):
        chunk = up[i:i + 5000]
        try:
            # Run the spatial join separately from vari_summary. The original
            # three-table join was repeatedly aborted by the real Gaia TAP.
            res = gaia.run_sync(GAIA_JOIN.format(r=MATCH_ARCSEC / 3600.0),
                                uploads={"t": chunk}, maxrec=200000)
            frame = res.to_table().to_pandas()
        except Exception as ex:
            # Cones are indexed in Gaia and worked live even when table-upload
            # joins did not. Slow but reproducible fallback for small boxes.
            if len(chunk) > 150:
                raise RuntimeError("Gaia bulk join unavailable; avoid thousands of slow "
                                   "individual queries with --skip-gaia") from ex
            print(f"  Gaia bulk match failed ({type(ex).__name__}: {ex}); "
                  f"falling back to {len(chunk)} individual cones")
            rows = []
            for r in chunk:
                q = ("SELECT TOP 20 g.source_id,g.ra AS g_ra,g.dec AS g_dec,"
                     "g.phot_g_mean_mag,g.bp_rp,g.ruwe,g.phot_variable_flag "
                     "FROM gaiadr3.gaia_source AS g "
                     f"WHERE 1=CONTAINS(POINT('ICRS',g.ra,g.dec),"
                     f"CIRCLE('ICRS',{float(r['ra'])},{float(r['dec'])},{MATCH_ARCSEC / 3600.0})) "
                     "AND g.phot_g_mean_mag BETWEEN 12 AND 19")
                sub = gaia.run_sync(q).to_table().to_pandas()
                if len(sub):
                    sub["uid"] = int(r["uid"])
                    sub["sep"] = 3600 * np.degrees(np.arccos(np.clip(
                        np.sin(np.radians(float(r["dec"]))) * np.sin(np.radians(sub["g_dec"])) +
                        np.cos(np.radians(float(r["dec"]))) * np.cos(np.radians(sub["g_dec"])) *
                        np.cos(np.radians(float(r["ra"]) - sub["g_ra"])), -1, 1)))
                    rows.append(sub)
            frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        out.append(frame)
        print(f"  Gaia match chunk {i // 5000 + 1}: {len(out[-1])} rows")
    g = pd.concat(out, ignore_index=True)
    if g.empty:
        return pd.DataFrame(columns=["uid", "source_id", "sep", "vari_id", "n_gaia",
                                     "g_ra", "g_dec", "phot_g_mean_mag", "bp_rp", "ruwe"])
    # Gaia variable labels are only queried AFTER finding sparse counterparts.
    known = []
    ids = g["source_id"].dropna().astype("int64").drop_duplicates().tolist()
    for i in range(0, len(ids), 400):
        id_csv = ",".join(str(x) for x in ids[i:i + 400])
        q = f"SELECT source_id AS vari_id FROM gaiadr3.vari_summary WHERE source_id IN ({id_csv})"
        known.append(gaia.run_sync(q).to_table().to_pandas())
    v = pd.concat(known, ignore_index=True) if known else pd.DataFrame(columns=["vari_id"])
    g = g.merge(v, left_on="source_id", right_on="vari_id", how="left")
    n = g.groupby("uid")["source_id"].transform("count")
    g["n_gaia"] = n
    g = g.sort_values("sep").drop_duplicates("uid")
    return g


def gaia_training_stars(args, n_each):
    """Known Gaia variables (positives) and non-variables (negatives) in the same box."""
    gaia = tap(GAIA_TAP)
    rw = getattr(args, "ra_width", None)
    dh = getattr(args, "dec_height", None)
    rw = args.size / max(np.cos(np.radians(args.dec0)), .05) if rw is None else rw
    dh = args.size if dh is None else dh
    sky = catalog_rectangle("g.ra", "g.dec", args.ra0-rw/2, args.ra0+rw/2,
                            args.dec0-dh/2, args.dec0+dh/2)
    base = f"({sky}) AND g.phot_g_mean_mag BETWEEN 12 AND 19"
    pos = gaia.run_sync(
        f"SELECT TOP {n_each} g.source_id,g.ra,g.dec,g.phot_g_mean_mag FROM gaiadr3.gaia_source g "
        f"JOIN gaiadr3.vari_summary v ON v.source_id=g.source_id WHERE {base} "
        f"AND v.num_selected_g_fov > 40 ORDER BY random_index").to_table().to_pandas()
    neg = gaia.run_sync(
        f"SELECT TOP {n_each} g.source_id,g.ra,g.dec,g.phot_g_mean_mag FROM gaiadr3.gaia_source g "
        f"LEFT JOIN gaiadr3.vari_summary v ON v.source_id=g.source_id WHERE {base} "
        f"AND v.source_id IS NULL AND g.phot_variable_flag <> 'VARIABLE' ORDER BY random_index"
    ).to_table().to_pandas()
    pos["label"] = 1; neg["label"] = 0
    return pd.concat([pos, neg], ignore_index=True)


def angular_separation_arcsec(ra, dec, obj_ra, obj_dec):
    """Accurate local small-angle separation, with RA wraparound."""
    dra = (np.asarray(obj_ra, float) - ra + 180) % 360 - 180
    ddec = np.asarray(obj_dec, float) - dec
    return 3600.0 * np.hypot(dra * np.cos(np.radians(dec)), ddec)


def select_nearest_ztf_oid(raw, ra, dec, max_sep_arcsec=1.2):
    """Select a geometrically matched light curve; reject crowded ambiguity."""
    required = {"oid", "ra", "dec", "mjd", "mag", "magerr", "catflags"}
    if not required.issubset(raw.columns):
        raise RuntimeError(f"ZTF response missing astrometry/quality columns: {required-set(raw.columns)}")
    data = raw.copy()
    for c in ("ra", "dec", "mjd", "mag", "magerr", "catflags"):
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.loc[(data["catflags"] == 0) & (data["magerr"] > 0) &
                    np.isfinite(data[["ra", "dec", "mjd", "mag", "magerr"]]).all(axis=1)]
    if data.empty:
        return None
    objects = []
    for oid, group in data.groupby("oid"):
        # The median of [359.9999, 0.0001] is NOT 180 degrees.
        offsets = (group["ra"].to_numpy() - ra + 180) % 360 - 180
        objra = float((ra + np.median(offsets)) % 360)
        objdec = float(group["dec"].median())
        sep = float(angular_separation_arcsec(ra, dec, objra, objdec))
        objects.append((sep, oid, objra, objdec, len(group), float(group["mag"].median())))
    objects.sort(key=lambda item: item[0])
    closest = objects[0]
    if closest[0] > max_sep_arcsec:
        return None
    # ZTF overlapping fields produce *different* OIDs for the SAME source:
    # live confirmed AGN case had 814+93 r-band visits with centroid distance
    # ~0.01 arcsec. Group coincident astrometry first, then prefer more epochs.
    coincident = []
    for item in objects:
        offset = float(angular_separation_arcsec(closest[2], closest[3], item[2], item[3]))
        same_source = offset <= 0.3 and abs(item[5] - closest[5]) <= 1.5
        if same_source:
            coincident.append(item)
        elif item[0] <= max_sep_arcsec and item[0] - closest[0] < 0.5:
            raise RuntimeError(f"ambiguous ZTF counterpart: distinct nearby OIDs "
                               f"at ({ra},{dec}); requires imaging")
    selected = max(coincident, key=lambda item: item[4])
    out = data.loc[data["oid"] == selected[1], ["mjd", "mag", "magerr"]].copy()
    out.attrs = dict(ztf_oid=str(selected[1]), obj_ra=selected[2], obj_dec=selected[3],
                     separation_arcsec=selected[0], coincident_oids=len(coincident))
    return out


def ztf_lc(ra, dec, cache):
    os.makedirs(cache, exist_ok=True)
    # Cache only successful, astrometrically checked results, never missing data.
    key = os.path.join(cache, f"matched_v3_{ra:.5f}_{dec:.5f}.pkl")
    if os.path.exists(key):
        with open(key, "rb") as f:
            cached = pickle.load(f)
        if cached is not None:
            return cached
    r = ztf_single_band(ra, dec, "r", cache)
    # Avoid a second internet round trip for every training star when the
    # r-band already supplies a robust 60+ point time series. g-band is still
    # independently fetched for the cross-filter anomaly validation stage.
    g = None if r is not None and len(r) >= 60 else ztf_single_band(ra, dec, "g", cache)
    good = [x for x in (r, g) if x is not None]
    if not good:
        return None
    best = max(good, key=len)
    with open(key, "wb") as f:
        pickle.dump(best, f)
    return best


def ztf_single_band(ra, dec, band, cache):
    """Fetch ZTF g or r independently (not a second observatory)."""
    import requests
    from io import StringIO
    os.makedirs(cache, exist_ok=True)
    key = os.path.join(cache, f"matched_v3_band_{ra:.5f}_{dec:.5f}_{band}.pkl")
    if os.path.exists(key):
        with open(key, "rb") as f:
            cached = pickle.load(f)
        if cached is not None:
            return cached
    failure = None
    for attempt in range(3):
        try:
            response = requests.get(ZTF_API, params={"POS": f"CIRCLE {ra} {dec} 0.0006",
                          "BANDNAME": band, "FORMAT": "csv", "BAD_CATFLAGS_MASK": 32768}, timeout=80)
            response.raise_for_status()
            if not response.text.strip() or response.text.lstrip().startswith("\\"):
                return None
            raw = pd.read_csv(StringIO(response.text))
            selected = select_nearest_ztf_oid(raw, ra, dec)
            if selected is not None:
                with open(key, "wb") as f:
                    pickle.dump(selected, f)
            return selected
        except requests.RequestException as ex:
            failure = ex
            # IRSA's position endpoint generates a preliminary TAP query.
            # On 2026-10-27 it returned "invalid TAP query" (HTTP 400) even
            # for positions that had worked earlier. The documented ID-based
            # endpoint skips that TAP lookup. Resolve nearby ZTF IDs from the
            # public DR24 HATS Objects Table and query those IDs directly.
            if isinstance(ex, requests.HTTPError) and getattr(ex.response, "status_code", None) == 400:
                return ztf_single_band_via_hats(ra, dec, band, cache)
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    if isinstance(failure, requests.HTTPError):
        try:
            return ztf_single_band_via_hats(ra, dec, band, cache)
        except Exception as ex:
            raise RuntimeError(f"ZTF {band}-band archive unavailable after retries "
                               f"({failure}) and HATS/ID fallback ({ex})") from ex
    raise RuntimeError(f"ZTF {band}-band archive unavailable after retries: {failure}")


def ztf_single_band_via_hats(ra, dec, band, cache):
    """Official public ZTF DR24 backup: HATS source IDs -> direct ID light curves.

    Fetching by ID bypasses the ZTF positional API's internal TAP lookup.
    Astrometric source selection is still performed on the returned epoch data.
    No Gaia labels or VarWISE discovery data are read here.
    """
    from io import StringIO
    import requests
    id_cache = os.path.join(cache, f"hats_dr24_oids_{ra:.5f}_{dec:.5f}_{band}.json")
    if os.path.exists(id_cache):
        with open(id_cache, encoding="utf-8") as stream:
            oids = json.load(stream)
    else:
        try:
            import lsdb
            import dask
        except ImportError as exc:
            raise RuntimeError("ZTF position API failed; HATS backup requires "
                               "`pip install lsdb s3fs pyarrow`") from exc
        # Public anonymous AWS access. A 2.16" cone corresponds to the
        # original positional radius; source matching below remains <=1.2".
        search = lsdb.ConeSearch(ra=float(ra), dec=float(dec), radius_arcsec=2.16)
        query = lsdb.open_catalog(
            ZTF_HATS_OBJECTS, search_filter=search,
            columns=["oid", "ra", "dec", "filtercode", "ngoodobsrel"],
            storage_options={"anon": True},
        )
        with dask.config.set(scheduler="single-threaded"):
            objects = query.compute()
        objects = objects.loc[objects["filtercode"].astype(str).str.lower().isin(
            (band, "z" + band))]
        if len(objects) > 20:
            raise RuntimeError("crowded HATS cone: more than 20 ZTF OIDs; "
                               "requires individual astrometric review")
        oids = sorted({int(x) for x in objects["oid"].dropna()})
        if not oids:
            return None
        # Only known real IDs are cached. No failed API response is cached.
        atomic_json(id_cache, oids)
    if not oids:
        return None
    params = [("ID", str(oid)) for oid in oids]
    params += [("BAD_CATFLAGS_MASK", 32768), ("FORMAT", "csv")]
    response = requests.get(ZTF_API, params=params, timeout=80)
    response.raise_for_status()
    if not response.text.strip() or response.text.lstrip().startswith(("\\", "<")):
        raise RuntimeError("ZTF ID endpoint returned no usable CSV")
    selected = select_nearest_ztf_oid(pd.read_csv(StringIO(response.text)), ra, dec)
    if selected is not None:
        path = os.path.join(cache, f"matched_v3_band_{ra:.5f}_{dec:.5f}_{band}.pkl")
        with open(path, "wb") as stream:
            pickle.dump(selected, stream)
    return selected


def ztf_two_band_check(ra, dec, cache):
    """Year-scale agreement in g and r; supporting evidence, not confirmation."""
    from scipy.stats import spearmanr
    try:
        r = ztf_single_band(ra, dec, "r", cache)
        g = ztf_single_band(ra, dec, "g", cache)
        if r is None or g is None:
            raise ValueError("one filter has no valid data")
        if "obj_ra" not in r.attrs or "obj_ra" not in g.attrs or \
                float(angular_separation_arcsec(r.attrs["obj_ra"], r.attrs["obj_dec"],
                                                g.attrs["obj_ra"], g.attrs["obj_dec"])) > 1.2:
            raise ValueError("g and r ZTF centroids disagree; possible blended/mismatched source")
        def annual(lc):
            d = lc.copy()
            d["bin"] = ((d["mjd"] - 58000) // 365).astype(int)
            group = d.groupby("bin")["mag"]
            return group.median()[group.size() >= 5]
        rb, gb = annual(r), annual(g)
        common = sorted(set(rb.index) & set(gb.index))
        if len(common) < 5:
            raise ValueError("fewer than 5 jointly observed year bins")
        rv, gv = rb.loc[common], gb.loc[common]
        rho = float(spearmanr(rv, gv).statistic)
        ramp, gamp = float(rv.max() - rv.min()), float(gv.max() - gv.min())
        consistent = bool(np.isfinite(rho) and rho >= 0.7 and
                          ramp >= 0.3 and gamp >= 0.3)
        return dict(crossband_consistent=consistent, crossband_rho=rho,
                    r_year_amp_mag=ramp, g_year_amp_mag=gamp,
                    common_year_bins=len(common), crossband_error="")
    except Exception as ex:
        return dict(crossband_consistent=False, crossband_rho=np.nan,
                    r_year_amp_mag=np.nan, g_year_amp_mag=np.nan,
                    common_year_bins=0, crossband_error=f"{type(ex).__name__}: {ex}")


def featurize(df, cache, label_col=None, workers=3, strict=True):
    """Fetch ZTF; discovery scans fail on errors, labelled training may skip bad matches."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    rows, failures = [], []

    def one(r):
        ra = getattr(r, "g_ra", None) if hasattr(r, "g_ra") else r.ra
        de = getattr(r, "g_dec", None) if hasattr(r, "g_dec") else r.dec
        lc = ztf_lc(float(ra), float(de), cache)
        if lc is None:
            return None
        f = lc_features(lc["mjd"], lc["mag"], lc["magerr"])
        if f is None:
            return None
        f["source_id"] = int(r.source_id)
        f["ra"] = float(ra); f["dec"] = float(de)
        f["cell"] = sky_cell(float(ra), float(de))
        if label_col:
            f["label"] = int(getattr(r, label_col))
        return f

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as pool:
        futures = [pool.submit(one, r) for r in df.itertuples()]
        for k, fut in enumerate(as_completed(futures), start=1):
            try:
                f = fut.result()
                if f is not None:
                    rows.append(f)
            except Exception as ex:
                failures.append(ex)
                print(f"  light-curve error ({type(ex).__name__}): {str(ex)[:100]}")
            if k % 10 == 0 or k == len(futures):
                print(f"  light curves processed: {k}/{len(df)} (usable {len(rows)})", flush=True)
    if failures and strict:
        raise RuntimeError(f"{len(failures)} ZTF fetches failed; this tile MUST be retried, "
                           f"not recorded as fully surveyed. First error: {failures[0]}")
    if failures and not strict:
        print(f"  EXCLUDED {len(failures)} unusable training/validation light curves "
              f"(never imputed or treated as negatives)")
    return pd.DataFrame(rows)


# ------------------------------------------------------------- prior art ---
VAR_OTYPES = {"V*", "Mi*", "Ir*", "RR*", "Ce*", "Pu*", "EB*", "El*", "RS*", "BY*", "Er*", "Or*",
              "TT*", "Fl*", "LP*", "SX*", "dS*", "gD*", "bC*", "WV*", "Al*", "Psr", "cC*", "RV*",
              "AB*", "SB*", "Ro*", "Eb*", "V*?", "EB?", "RR?", "Mi?", "Pu?", "Ir?", "Or?", "LP?"}
VIZIER_CATS = {"VSX": "B/vsx/vsx", "ASAS-SN variables": "II/366",
               "ZTF periodic (Chen+2020)": "J/ApJS/249/18", "Gaia DR3 vari classifier": "I/358/vclassre"}


def prior_art(ra, dec):
    """Return known-object/catalogue hits and fail-closed lookup failures.

    Any SIMBAD/NED counterpart means the *object* is already documented; its
    optical behaviour could still be new, but cannot be called a new object.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astroquery.vizier import Vizier
    from astroquery.simbad import Simbad
    c = SkyCoord(ra * u.deg, dec * u.deg)
    hits = []
    for name, cat in VIZIER_CATS.items():
        try:
            r = Vizier(row_limit=5).query_region(c, radius=3 * u.arcsec, catalog=cat)
            if len(r) and len(r[0]):
                hits.append(name)
        except Exception as ex:
            hits.append(f"{name}:ERROR({type(ex).__name__})")      # an error blocks acceptance
    try:
        simbad = Simbad()
        simbad.add_votable_fields("otype", "otypes")
        s = simbad.query_region(c, radius=3 * u.arcsec)
        if s is not None and len(s):
            col = next((name for name in s.colnames if name.lower() == "otype"), None)
            more = next((name for name in s.colnames if name.lower() == "otypes"), None)
            for i in range(len(s)):
                primary = str(s[col][i]).strip() if col else "unknown"
                secondary = str(s[more][i]).strip() if more else ""
                hits.append(f"SIMBAD:KNOWN({primary};{secondary[:160]})")
    except Exception as ex:
        hits.append(f"SIMBAD:ERROR({type(ex).__name__})")
    try:
        from astroquery.ipac.ned import Ned
        ned = Ned.query_region(c, radius=3 * u.arcsec)
        if ned is not None and len(ned):
            hits.append(f"NED:KNOWN({len(ned)} counterparts)")
    except Exception as ex:
        hits.append(f"NED:ERROR({type(ex).__name__})")
    return list(dict.fromkeys(hits))


# -------------------------------------------------------------------- run ---
def fold_plot(ra, dec, period, cache, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lc = ztf_lc(ra, dec, cache)
    ph = (lc["mjd"].values % period) / period
    plt.figure(figsize=(6, 3.5))
    plt.errorbar(np.r_[ph, ph + 1], np.r_[lc["mag"], lc["mag"]],
                 np.r_[lc["magerr"], lc["magerr"]], fmt=".", ms=3, alpha=0.6)
    plt.gca().invert_yaxis(); plt.xlabel(f"phase (P = {period:.5f} d)"); plt.ylabel("ZTF mag")
    plt.title(f"RA {ra:.5f} Dec {dec:.5f}"); plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def raw_plot(ra, dec, cache, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lc = ztf_lc(ra, dec, cache)
    plt.figure(figsize=(7, 3.5))
    plt.errorbar(lc["mjd"], lc["mag"], lc["magerr"], fmt=".", ms=3, alpha=0.7)
    plt.gca().invert_yaxis()
    plt.xlabel("MJD"); plt.ylabel("ZTF magnitude")
    plt.title(f"Unverified light-curve anomaly RA {ra:.5f} Dec {dec:.5f}")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def two_band_plot(ra, dec, cache, path):
    """Show independent ZTF g and r light curves on one aligned time axis."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, band, color in zip(axes, ("g", "r"), ("#17a398", "#d57b32")):
        lc = ztf_single_band(ra, dec, band, cache)
        if lc is None:
            ax.text(.5, .5, f"ZTF {band}: no data", ha="center", transform=ax.transAxes)
            continue
        ax.errorbar(lc["mjd"], lc["mag"], lc["magerr"], fmt=".",
                    ms=2.5, alpha=.35, color=color)
        d = lc.copy()
        d["bin"] = ((d["mjd"] - 58000) // 365).astype(int)
        annual = d.groupby("bin")["mag"].median()
        ax.plot(58000 + (annual.index.to_numpy() + .5) * 365,
                annual.to_numpy(), color=color, lw=2, marker="o", ms=3)
        ax.invert_yaxis()
        ax.set_ylabel(f"ZTF {band} magnitude")
        ax.grid(alpha=.15)
    axes[-1].set_xlabel("Modified Julian Date")
    fig.suptitle(f"Unverified two-filter optical variability | RA {ra:.5f}, Dec {dec:.5f}")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def collect_anomaly_leads(unk, args, cache, out, already_handled=None):
    """Second discovery channel for shapes missed by Lomb-Scargle/ML.

    Never labels these as verified: multi-band/imaging confirmation is required.
    """
    if not args.include_anomalies:
        return [], []
    already_handled = set(already_handled or [])
    prospects = unk[unk["amp_over_err"] >= 4]
    prospects = prospects.sort_values("amp_over_err", ascending=False)
    leads, known = [], []
    for r in prospects.itertuples():
        if int(r.source_id) in already_handled:
            continue
        lc = ztf_lc(r.ra, r.dec, cache)
        if lc is None:
            continue
        signature = outlier_lead_features(lc["mjd"], lc["mag"], lc["magerr"])
        if not signature["flag"]:
            continue
        hits = prior_art(r.ra, r.dec)
        rec = r._asdict(); rec.pop("Index", None)
        rec.update(signature)
        if any("ERROR" not in h for h in hits):
            rec.update(prior_art=";".join(hits), status="ALREADY CATALOGUED - anomaly only")
            known.append(rec)
            continue
        bands = ztf_two_band_check(r.ra, r.dec, cache)
        rec.update(bands)
        stage = ("correlated g/r filters" if bands["crossband_consistent"] else
                 "g/r evidence missing or inconsistent")
        rec.update(prior_art=";".join(hits), status="UNVERIFIED ANOMALY - " + stage +
                   "; still requires imaging and astrophysical confirmation")
        leads.append(rec)
        raw_plot(r.ra, r.dec, cache, os.path.join(out, f"raw_{r.source_id}.png"))
        if not bands["crossband_error"]:
            two_band_plot(r.ra, r.dec, cache, os.path.join(out, f"bands_{r.source_id}.png"))
        if len(leads) >= args.max_prior:
            break
    return leads, known


VARWISE_PROVENANCE = ("cluster_id", "designation", "vartype", "confidence",
                      "blended_source", "latent_artifact", "period1", "w1_amp", "w2_amp")


def attach_varwise_provenance(g, vw):
    """Keep the original catalogue identifier and artifact warnings in exports."""
    cols = [c for c in VARWISE_PROVENANCE if c in vw.columns]
    renamed = vw[["uid"] + cols].rename(columns={c: "varwise_" + c for c in cols})
    return g.merge(renamed, on="uid", how="left")


def gaia_export_columns(g):
    return [c for c in ("source_id", "phot_g_mean_mag", "bp_rp", "ruwe") if c in g.columns] + [
        c for c in g.columns if c.startswith("varwise_")]


def varwise_targets_without_gaia(vw):
    """Optical target positions directly from VarWISE; Gaia is NOT mandatory.

    Negative source_id values identify VarWISE clusters, never Gaia IDs.
    Existing optical catalogues are checked later before accepting a candidate.
    """
    d = vw.copy()
    # VarWISE cluster IDs are catalogue-global; row uids reset for every sky
    # box and would overwrite plots when a resumable survey spans many boxes.
    if "cluster_id" in d.columns:
        d["source_id"] = -(d["cluster_id"].astype("int64").abs() + 1)
    else:
        # Position-derived IDs stay stable across adjacent sky boxes and runs.
        d["source_id"] = [-(int(hashlib.sha256(f"{ra:.7f},{de:.7f}".encode()).hexdigest()[:15], 16)+1)
                          for ra, de in zip(d["ra"], d["dec"])]
    d["g_ra"] = d["ra"]
    d["g_dec"] = d["dec"]
    d["vari_id"] = np.nan
    d["n_gaia"] = 0
    d["phot_g_mean_mag"] = np.nan
    d["bp_rp"] = np.nan
    d["ruwe"] = np.nan
    for c in VARWISE_PROVENANCE:
        if c in d.columns:
            d["varwise_" + c] = d[c]
    return d


def load_validated_model(path):
    """Reject old, proxy-validated, or incomplete model packages."""
    if not os.path.exists(path):
        return None
    import joblib
    try:
        pack = joblib.load(path)
        report = pack.get("report", {})
        if not (pack.get("validated_for_discovery") is True and
                pack.get("validation_protocol") == VALIDATION_PROTOCOL and
                report.get("validated_for_discovery") is True and
                report.get("code_sha256") == code_sha256() and
                report.get("auc_heldout", 0) >= .85 and
                abs(report.get("auc_shuffled_control", 999) - .5) <= .12 and
                pack.get("features") == FEATURES):
            return None
        return pack
    except Exception:
        return None


def run(args):
    # HARD BOUNDARY: no VarWISE query, not even a count, until a separately
    # trained and genuinely validated model already exists.
    if args.search_mode == "signal" and not getattr(args, "allow_untrained_diagnostics", False):
        print("TRAIN-FIRST LOCK: untrained signal-only search disabled. "
              "No VarWISE data was touched. Train and validate first.")
        return 2
    if args.search_mode != "signal":
        pack = load_validated_model(os.path.join(args.out, "model.joblib"))
        if pack is None:
            print("TRAINING REQUIRED: no sealed-20%-validated model. "
                  "NOT querying VarWISE. Run `train` or `auto` first.")
            return 2
        rw = args.size/max(np.cos(np.radians(args.dec0)), .05)
        sought = (args.ra0-rw/2,args.ra0+rw/2,
                  args.dec0-args.size/2,args.dec0+args.size/2)
        reserved = pack["report"]["training_tiles"] + pack["report"].get("validation_tiles", [])
        for ra0, dec0, w, h in reserved:
            training_rect = (ra0-w/2,ra0+w/2,dec0-h/2,dec0+h/2)
            if rectangle_intersects(sought, training_rect):
                print("STOP: requested discovery box overlaps a reserved TRAINING tile. "
                      "Choose a disjoint sky region. No VarWISE data was accessed.")
                return 2
        clf = pack["model"]
    else:
        clf = None  # explicit legacy diagnostic, not the default science search
    os.makedirs(args.out, exist_ok=True)
    cache = os.path.join(args.out, "ztf_cache"); os.makedirs(cache, exist_ok=True)
    vw = get_varwise(args)
    vw = vw.reset_index(drop=True); vw["uid"] = np.arange(len(vw))
    if args.skip_gaia:
        g_unk = varwise_targets_without_gaia(vw)
        print(f"Gaia match deliberately skipped: searching {len(g_unk)} VarWISE positions "
              "(optical catalogue novelty checked later).")
    else:
        g = gaia_match(vw)
        g = attach_varwise_provenance(g[(g["sep"] <= MATCH_ARCSEC)], vw)
        print(f"VarWISE objects with a Gaia G 12-19 match: {len(g)}")
        g_known = g[g["vari_id"].notna()]
        g_unk = g[g["vari_id"].isna() & (g["n_gaia"] == 1)]      # unique match only
        print(f"  already have a Gaia variability solution: {len(g_known)}   unknown: {len(g_unk)}")

    unk = featurize(g_unk, cache)
    if unk.empty:
        print("No targets have usable optical light curves.")
        pd.DataFrame().to_csv(os.path.join(args.out, "candidates.csv"), index=False)
        return 0
    unk["score"] = clf.predict_proba(unk[FEATURES])[:, 1] if clf is not None else np.nan
    unk = unk.merge(g_unk[gaia_export_columns(g_unk)], on="source_id", how="left")
    sel = unk[passes_cuts(unk, require_model=clf is not None)]
    sel = sel.sort_values("score" if clf is not None else "log_fap",
                          ascending=clf is None)
    print(f"optical periodic-signal selections: {len(sel)} of {len(unk)} scored" +
          (" (Gaia variability unknown; match skipped)" if args.skip_gaia else ""))
    sel = sel.head(args.max_prior)
    keep, unverified, rejected, known = [], [], [], []
    for r in sel.itertuples():
        temporal = split_epoch_confirmation(r.ra, r.dec, r.period, cache)
        rec = r._asdict(); rec.pop("Index", None)
        rec.update(temporal)
        rec["search_mode"] = "ml" if clf is not None else "untrained_signal"
        if not temporal["confirmed"]:
            rejected.append(rec)
            continue
        hits = prior_art(r.ra, r.dec)
        ident = f"VarWISE local {r.source_id}" if r.source_id < 0 else f"Gaia {r.source_id}"
        print(f"  {ident}  P={r.period:.5f} d  score={r.score:.3f}  prior-art hits: {hits or 'none'}")
        rec["prior_art"] = ";".join(hits)
        if any("ERROR" in x for x in hits):
            if any("ERROR" not in x for x in hits):
                rec["status"] = "ALREADY CATALOGUED - positive prior-art match"
                known.append(rec)
            else:
                rec["status"] = "UNVERIFIED - prior-art service unavailable"
                unverified.append(rec)
        elif not hits:
            rec["status"] = "CANDIDATE - manual imaging and literature checks required"
            keep.append(rec)
        else:
            rec["status"] = "ALREADY CATALOGUED - positive prior-art match"
            known.append(rec)
    out = pd.DataFrame(keep).drop(columns=["Index"], errors="ignore")
    out.to_csv(os.path.join(args.out, "candidates.csv"), index=False)
    pd.DataFrame(unverified).to_csv(os.path.join(args.out, "unverified.csv"), index=False)
    pd.DataFrame(rejected).to_csv(os.path.join(args.out, "failed_temporal_check.csv"), index=False)
    anomalies, known_anomalies = collect_anomaly_leads(
        unk, args, cache, args.out,
        already_handled=[x["source_id"] for x in keep + unverified + known])
    pd.DataFrame(anomalies).to_csv(os.path.join(args.out, "anomaly_leads.csv"), index=False)
    pd.DataFrame(known + known_anomalies).to_csv(os.path.join(args.out, "known_objects.csv"), index=False)
    print(f"additional unverified anomaly leads: {len(anomalies)}")
    for r in out.itertuples():
        fold_plot(r.ra, r.dec, r.period, cache, os.path.join(args.out, f"fold_{r.source_id}.png"))
    print(f"\n{len(out)} candidates written to {args.out}/candidates.csv")
    print("MANUAL STEPS STILL REQUIRED before calling any of them new: check the folded PNG by eye, "
          "PanSTARRS/Legacy imaging for blends and a nearby bright star, a second period search "
          "(PDM / double the period for eclipsing binaries), MPChecker (asteroids), "
          "and a final VSX name search on the web site.")
    return 0


def smoke(args):
    """Run every network stage on a tiny box. Prints PASS/FAIL per stage with the error."""
    import traceback
    args.size = 2.0; args.max_vw = 300; args.n_train = 25
    state = {}
    fails, warnings_seen = [], []

    def stage(name, fn, optional=False):
        try:
            r = fn(); print(f"[PASS] {name}"); return r
        except Exception as ex:
            if optional:
                warnings_seen.append(name)
            else:
                fails.append(name)
            print(f"[{'WARN' if optional else 'FAIL'}] {name}: {type(ex).__name__}: {ex}")
            print("       " + traceback.format_exc().strip().splitlines()[-3].strip())
            return None
    vw = stage("1 VarWISE query (IRSA TAP)", lambda: get_varwise(args))
    if vw is not None:
        vw = vw.reset_index(drop=True); vw["uid"] = np.arange(len(vw))
        print("       columns:", list(vw.columns)[:15])
    if args.skip_gaia:
        g = None
        print("[SKIP] 2 Gaia DR3 cross-match (not needed with --skip-gaia)")
    else:
        g = stage("2 Gaia DR3 cross-match + vari_summary join",
                  lambda: gaia_match(vw)) if vw is not None else None
    if args.search_mode == "signal":
        tr = None
        print("[SKIP] 3 Gaia training-star query (label-free signal search)")
    else:
        tr = stage("3 Gaia training-star query", lambda: gaia_training_stars(args, args.n_train),
                   optional=args.search_mode != "ml")
    def z():
        d = ztf_lc(150.0, 30.0, args.out or ".")
        return d
    os.makedirs(args.out, exist_ok=True)
    def z2():
        row = tr.iloc[0] if tr is not None and len(tr) else None
        ra, de = (float(row["ra"]), float(row["dec"])) if row is not None else (args.ra0, args.dec0)
        d = ztf_lc(ra, de, args.out)
        if d is None: raise RuntimeError("ZTF API returned no rows for this star (try another)")
        print(f"       ZTF rows: {len(d)}"); return d
    d = stage("4 ZTF light-curve API", z2)
    if d is not None:
        stage("5 Lomb-Scargle features", lambda: lc_features(d["mjd"], d["mag"], d["magerr"]))
    def pa():
        h = prior_art(150.0, 2.2)
        errs = [x for x in h if "ERROR" in x]
        print("       hits:", h)
        if errs:
            raise RuntimeError("services unreachable or failing: " + "; ".join(errs))
    # A transient SIMBAD outage cannot authorize a candidate, but should not
    # prevent searching: each candidate will instead enter unverified.csv.
    stage("6 Prior-art services (VizieR + SIMBAD)", pa, optional=True)
    if fails:
        print("\nPaste the FAIL lines back to Claude and that stage gets fixed.")
    else:
        print("\nAll required stages PASS." +
              (" Prior-art outage: any affected results remain UNVERIFIED." if warnings_seen else ""))
    return len(fails)


def train_only(args):
    """Fit on labelled Gaia/ZTF only, then unlock sealed held-out validation."""
    os.makedirs(args.out, exist_ok=True)
    boxes = generate_boxes(args.box_size, 0, -90, 90)
    pack = train_and_validate(args, boxes, args.out)
    if pack is None:
        print("TRAINING/VALIDATION DID NOT PASS. No target data searched.")
        return 2
    print(f"VALIDATED model saved to {args.out}/model.joblib; targets remain untouched")
    return 0


# ------------------------------------------------------------------- auto ---
class Tee:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1); self.o = sys.stdout
    def write(self, x):
        self.o.write(x); self.f.write(x)
    def flush(self):
        self.o.flush(); self.f.flush()
    def close(self):
        self.f.close()


def clone_args(a, **kw):
    d = dict(vars(a)); d.update(kw)
    return argparse.Namespace(**d)


def atomic_json(path, data):
    """Never expose a half-written checkpoint, including on interruption."""
    def ready(value):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {str(k): ready(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [ready(v) for v in value]
        return value
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".pending_", suffix=".json", dir=folder)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(ready(data), f, indent=2, allow_nan=False)
            f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def atomic_csv(path, rows):
    """Rebuild each survey output from committed box records (idempotent)."""
    folder = os.path.dirname(os.path.abspath(path))
    fd, temp = tempfile.mkstemp(prefix=".pending_", suffix=".csv", dir=folder)
    try:
        with os.fdopen(fd, "w") as f:
            if rows:
                d = pd.DataFrame(rows)
                keys = [x for x in ("source_id", "ra", "dec") if x in d.columns]
                if keys:
                    d = d.drop_duplicates(subset=keys)
                d.to_csv(f, index=False)
            else:
                f.write("\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def survey_fingerprint(args):
    """Search identity changes whenever a result-affecting setting changes."""
    opts = ("table", "ra_col", "dec_col", "max_vw", "box_size", "bmin",
            "dec_min", "dec_max", "search_mode", "skip_gaia", "include_anomalies",
            "max_prior", "n_train", "train_per_box", "min_train")
    config = {x: getattr(args, x, None) for x in opts}
    config.update(code_sha256=code_sha256(),
                  method="complete-rectangle-v3", cuts=SEARCH_CUTS)
    token = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return token, config


def initialize_survey(out, token, config):
    confpath = os.path.join(out, "survey_config.json")
    if os.path.exists(confpath):
        with open(confpath) as f:
            previous = json.load(f)
        if previous.get("fingerprint") != token:
            raise RuntimeError("OUTPUT DIRECTORY HAS DIFFERENT SURVEY SETTINGS OR CODE; "
                               "use a NEW --out directory. Old search results were not touched.")
    else:
        # Never treat legacy outputs as safely resumable.
        legacy = ("state.json", "candidates.csv", "unverified.csv", "anomaly_leads.csv")
        if any(os.path.exists(os.path.join(out, x)) for x in legacy):
            raise RuntimeError("legacy results without a safe configuration fingerprint; "
                               "use a NEW --out directory")
        atomic_json(confpath, dict(fingerprint=token, config=config))


def survey_checkpoint(out, token):
    """Use committed per-tile JSON as the sole source of resume truth."""
    folder = os.path.join(out, "committed_tiles")
    os.makedirs(folder, exist_ok=True)
    records = {}
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(folder, filename)) as f:
            item = json.load(f)
        if item.get("fingerprint") != token:
            raise RuntimeError("committed tile belongs to another survey configuration")
        records[item["key"]] = item
    for field, dest in (("candidates", "candidates.csv"),
                        ("unverified", "unverified.csv"),
                        ("anomalies", "anomaly_leads.csv"),
                        ("known", "known_objects.csv")):
        rows = [row for item in records.values() for row in item[field]]
        atomic_csv(os.path.join(out, dest), rows)
    state = dict(fingerprint=token, done=sorted(records),
                 n_candidates=sum(len(v["candidates"]) for v in records.values()),
                 scanned=sum(v["scanned"] for v in records.values()))
    atomic_json(os.path.join(out, "state.json"), state)
    return state


def generate_boxes(size, bmin=0.0, dec_min=-90.0, dec_max=90.0, seed=1):
    """Nonoverlapping RA/Dec rectangles that exactly tile the requested sky.

    Returns (ra_center, dec_center, ra_width, dec_height). Optional bmin>0
    knowingly discards tiles by their centers and is NOT complete coverage.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    if not (0 < size <= 60 and -90 <= dec_min < dec_max <= 90 and 0 <= bmin < 90):
        raise ValueError("invalid tiling: require 0<size<=60, -90<=dec_min<dec_max<=90, 0<=bmin<90")
    boxes = []
    dec_low = float(dec_min)
    while dec_low < dec_max - 1e-10:
        dec_high = min(dec_low + size, dec_max)
        dec_center = (dec_low + dec_high)/2
        # Every longitude in [0,360) is covered exactly once, even at poles.
        raw_width = size / max(np.cos(np.radians(dec_center)), 0.05)
        n_ra = max(1, int(np.ceil(360 / raw_width)))
        ra_width = 360 / n_ra
        for j in range(n_ra):
            boxes.append(((j + .5)*ra_width, dec_center, ra_width, dec_high-dec_low))
        dec_low = dec_high
    if bmin:
        arr = np.asarray(boxes)
        b = SkyCoord(arr[:, 0]*u.deg, arr[:, 1]*u.deg).galactic.b.deg
        boxes = [x for x, keep in zip(boxes, np.abs(b) >= bmin) if keep]
    rng = np.random.default_rng(seed); rng.shuffle(boxes)
    return boxes


SEARCH_CUTS = dict(score=0.9, log_fap=-6.0, amp_over_err=5.0)


def passes_cuts(df, require_model=True):
    """A conservative optical signal gate; classifier confidence is optional.

    This enables an UNTRAINED signal search on a target without labelled stars.
    Such results must never be represented as calibrated ML discoveries.
    """
    signal = ((df["log_fap"] <= SEARCH_CUTS["log_fap"])
              & (~df["alias"].astype(bool))
              & (df["amp_over_err"] >= SEARCH_CUTS["amp_over_err"]))
    return signal & (df["score"] >= SEARCH_CUTS["score"]) if require_model else signal


def temporal_consistency_lc(lc, period, tolerance=0.03):
    """Test whether independent temporal halves both recover the proposed period."""
    if lc is None or len(lc) < 80:
        return dict(confirmed=False, early_period=None, late_period=None,
                    reason="fewer than 80 observations")
    lc = lc.sort_values("mjd")
    first, second = np.array_split(np.arange(len(lc)), 2)
    periods = []
    for idx in (first, second):
        q = lc.iloc[idx]
        ft = lc_features(q["mjd"], q["mag"], q["magerr"])
        periods.append(ft["period"] if ft is not None else None)
    def near(p, ref):
        return p is not None and any(abs(p / (ref * k) - 1) <= tolerance
                                     for k in (0.5, 1.0, 2.0))
    ok = all(near(p, period) for p in periods)
    return dict(confirmed=bool(ok), early_period=periods[0], late_period=periods[1],
                reason="period independently seen in both halves" if ok else
                       "period absent or inconsistent in a temporal half")


def split_epoch_confirmation(ra, dec, period, cache, tolerance=0.03):
    """Independent first-half / second-half check on actual ZTF observations."""
    lc = ztf_lc(float(ra), float(dec), cache)
    return temporal_consistency_lc(lc, period, tolerance)


def injection_recovery(neg, cache, clf, n=160, seed=7):
    """Inject sinusoids into REAL constant-star ZTF light curves (their own sampling and noise)
    and measure how often the whole search chain recovers them."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in neg.sample(min(n, len(neg)), random_state=seed).itertuples():
        lc = ztf_lc(float(r.ra), float(r.dec), cache)
        if lc is None or len(lc) < 40:
            continue
        for A in (0.05, 0.1, 0.2, 0.4):
            P = 10 ** rng.uniform(np.log10(0.3), np.log10(20))
            m = lc["mag"].values + A * np.sin(2 * np.pi * lc["mjd"].values / P + rng.uniform(0, 6.28))
            f = lc_features(lc["mjd"].values, m, lc["magerr"].values)
            if f is None:
                continue
            f["score"] = float(clf.predict_proba(pd.DataFrame([f])[FEATURES])[:, 1][0])
            ratio = f["period"] / P
            good_p = any(abs(ratio - k) < 0.01 * k for k in (1.0, 0.5, 2.0))
            rows.append(dict(amp=A, period=P, ok=bool(good_p and passes_cuts(pd.DataFrame([f])).iloc[0])))
    d = pd.DataFrame(rows)
    return d


def train_and_validate(args, boxes, out):
    """STRICT: split Gaia metadata, FIT on training ZTF, THEN open holdout ZTF.

    No VarWISE target query occurs anywhere in this function. On failure no
    discovery-valid model is produced, and the calling search must stop.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    cache = os.path.join(out, "ztf_cache"); os.makedirs(cache, exist_ok=True)
    frames, used_tiles = [], []
    # Sample training labels where ZTF has comparatively good coverage. These
    # tiles are RESERVED, never used as discovery targets later.
    eligible = [box for box in boxes if -15 <= box[1] <= 75]
    # The validation sky tiles are selected *before* training, then completely
    # untouched: NO Gaia labels, ZTF curves or features from them are fetched.
    train_boxes = eligible[:16]
    validation_boxes = eligible[16:24]
    if len(train_boxes) < 8 or len(validation_boxes) < 4:
        print("NO MODEL: need 24 separated training/holdout sky tiles; target untouched")
        return None
    for ra0, dec0, ra_width, dec_height in train_boxes:
        a = clone_args(args, ra0=ra0, dec0=dec0, size=args.box_size,
                       ra_width=ra_width, dec_height=dec_height)
        try:
            tr = gaia_training_stars(a, args.train_per_box)
        except Exception as ex:
            print(f"  training box ({ra0:.1f},{dec0:.1f}) skipped: {type(ex).__name__}: {str(ex)[:120]}")
            continue
        tr = tr.rename(columns={"ra": "g_ra", "dec": "g_dec"})
        if len(tr):
            frames.append(tr)
            used_tiles.append((ra0, dec0, ra_width, dec_height))
        pool = pd.concat(frames, ignore_index=True).drop_duplicates("source_id") if frames else pd.DataFrame()
        tot = len(pool)
        n1 = int(pool["label"].sum()) if tot else 0
        print(f"  Gaia TRAINING METADATA ONLY: {tot} sources ({n1} variable, "
              f"{tot-n1} proxy negatives), {len(used_tiles)} tiles")
        if tot >= max(args.min_train, 120) and min(n1, tot - n1) >= 0.25 * tot:
            break
    if not frames:
        print("NO MODEL: no independent Gaia training metadata."); return None
    train_meta = pd.concat(frames, ignore_index=True).drop_duplicates("source_id").copy()
    train_meta["cell"] = [sky_cell(r, d) for r, d in zip(train_meta["g_ra"], train_meta["g_dec"])]
    if len(train_meta) < max(args.min_train, 120):
        print(f"NO MODEL: only {len(train_meta)} Gaia training labels, need "
              f"{max(args.min_train, 120)}. No discovery target was accessed.")
        return None
    print(f"[SEALED] {len(train_meta)} TRAINING Gaia labels; "
          f"{len(validation_boxes)} held-out sky tiles remain COMPLETELY UNQUERIED")
    try:
        training_features = featurize(train_meta, cache, label_col="label", strict=False)
    except Exception as ex:
        print(f"NO MODEL: TRAINING download failed: {ex}"); return None
    if len(training_features) < 90 or training_features["label"].value_counts().min() < 20:
        print("NO MODEL: too few usable TRAINING light curves. HOLDOUT untouched.")
        return None
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=.08,
                                         max_iter=200, random_state=0)
    clf.fit(training_features[FEATURES], training_features["label"])
    print("[MODEL FIT COMPLETE] Only NOW allowed to query validation Gaia/ZTF")
    # Shuffled baselines are fit only on TRAINING curves, never the held-out data.
    rng = np.random.default_rng(0)
    shuffled_models = []
    for _ in range(10):
        null_clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=.08,
                                                  max_iter=100, random_state=0)
        null_clf.fit(training_features[FEATURES],
                     rng.permutation(training_features["label"].to_numpy()))
        shuffled_models.append(null_clf)
    # FIRST AND ONLY ACCESS to held-out sky data is here, AFTER classifier.fit.
    holdout_frames, used_validation_tiles = [], []
    desired = max(20, int(round(len(training_features)/4)))
    for ra0, dec0, ra_width, dec_height in validation_boxes:
        a = clone_args(args, ra0=ra0, dec0=dec0, size=args.box_size,
                       ra_width=ra_width, dec_height=dec_height)
        try:
            val = gaia_training_stars(a, args.train_per_box)
        except Exception as ex:
            print(f"  sealed validation tile ({ra0:.1f},{dec0:.1f}) unavailable: {ex}")
            continue
        if len(val):
            holdout_frames.append(val.rename(columns={"ra":"g_ra", "dec":"g_dec"}))
            used_validation_tiles.append((ra0, dec0, ra_width, dec_height))
        if sum(len(x) for x in holdout_frames) >= desired:
            break
    if not holdout_frames:
        print("NO MODEL: sealed validation Gaia metadata unavailable"); return None
    holdout_meta = pd.concat(holdout_frames, ignore_index=True).drop_duplicates("source_id")
    holdout_meta["cell"] = [sky_cell(r, d) for r, d in zip(holdout_meta["g_ra"],holdout_meta["g_dec"])]
    holdout_meta = holdout_meta[~holdout_meta["cell"].isin(set(train_meta["cell"]))]
    if len(holdout_meta) > desired:
        holdout_meta = holdout_meta.sample(n=desired, random_state=0)
    assert not set(train_meta["cell"]) & set(holdout_meta["cell"])
    try:
        te = featurize(holdout_meta, cache, label_col="label", strict=False)
    except Exception as ex:
        print(f"NO MODEL: HOLDOUT download failed; refusing search: {ex}"); return None
    if len(te) < 20 or te["label"].nunique() != 2 or te["label"].value_counts().min() < 10:
        print("NO MODEL: holdout lacked enough usable objects in both classes")
        return None
    heldout_fraction = len(te)/(len(te)+len(training_features))
    if not .15 <= heldout_fraction <= .25:
        print(f"NO MODEL: usable validation share {heldout_fraction:.1%}, "
              "outside permitted 15%-25%; cannot claim 80/20")
        return None
    auc = float(roc_auc_score(te["label"], clf.predict_proba(te[FEATURES])[:, 1]))
    auc0 = float(np.mean([roc_auc_score(te["label"], p.predict_proba(te[FEATURES])[:, 1])
                          for p in shuffled_models]))
    te = te.copy()
    te["score"] = clf.predict_proba(te[FEATURES])[:, 1]
    sel = passes_cuts(te)
    tp = int(((te["label"] == 1) & sel).sum()); fp = int(((te["label"] == 0) & sel).sum())
    fn = int(((te["label"] == 1) & ~sel).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    rep = dict(n_train=int(len(training_features)), n_heldout=int(len(te)),
               metadata_train=int(len(train_meta)), metadata_heldout=int(len(holdout_meta)),
               split_strategy="sky-disjoint training/holdout tiles PRESELECTED; "
                              "fit model BEFORE any held-out Gaia or ZTF query",
               actual_heldout_fraction=float(heldout_fraction),
               labels="known Gaia variables vs sources lacking Gaia variability solutions (proxy negatives)",
               auc_heldout=float(auc),
               auc_shuffled_control=float(auc0), precision_at_search_cuts=float(prec),
               recall_at_search_cuts=float(rec), false_positives_heldout=fp,
               training_tiles=[list(x) for x in used_tiles],
               validation_tiles=[list(x) for x in used_validation_tiles],
               validation_protocol=VALIDATION_PROTOCOL,
               code_sha256=CODE_LOADED_SHA256,
               validated_for_discovery=bool(auc >= .85 and abs(auc0-.5) <= .12))
    atomic_json(os.path.join(out, "validation_report.json"), rep)
    training_features.to_csv(os.path.join(out, "training_features.csv"), index=False)
    te.to_csv(os.path.join(out, "sealed_validation_features.csv"), index=False)
    if not rep["validated_for_discovery"]:
        print(f"NO MODEL: true holdout AUC {auc:.3f}, null {auc0:.3f}; "
              "validation FAILED. Discovery targets remain untouched.")
        return None
    import joblib
    pack = dict(model=clf, features=FEATURES, report=rep,
                validated_for_discovery=True, validation_protocol=VALIDATION_PROTOCOL)
    joblib.dump(pack, os.path.join(out, "model.joblib"))
    print("\n=== VALIDATION PASSED (real data, sealed sky-disjoint holdout) ===")
    for k, v in rep.items():
        print(f"  {k}: {v}")
    print("====================================================\n")
    return pack


def process_box(args, clf, ra0, dec0, out, cache, ra_width=None, dec_height=None):
    a = clone_args(args, ra0=ra0, dec0=dec0, size=args.box_size,
                   ra_width=ra_width, dec_height=dec_height)
    vw = get_varwise(a)
    if vw is None or len(vw) == 0:
        return [], [], [], [], 0
    vw = vw.reset_index(drop=True); vw["uid"] = np.arange(len(vw))
    if args.skip_gaia:
        g_unk = varwise_targets_without_gaia(vw)
    else:
        g = gaia_match(vw)
        g = attach_varwise_provenance(g[g["sep"] <= MATCH_ARCSEC], vw)
        g_unk = g[g["vari_id"].isna() & (g["n_gaia"] == 1)]
    unk = featurize(g_unk, cache)
    if len(unk) == 0:
        return [], [], [], [], len(vw)
    unk["score"] = clf.predict_proba(unk[FEATURES])[:, 1] if clf is not None else np.nan
    unk = unk.merge(g_unk[gaia_export_columns(g_unk)], on="source_id", how="left")
    sel = unk[passes_cuts(unk, require_model=clf is not None)]
    sel = sel.sort_values("score" if clf is not None else "log_fap",
                          ascending=clf is None).head(args.max_prior)
    good, unverified, known = [], [], []
    for r in sel.itertuples():
        temporal = split_epoch_confirmation(r.ra, r.dec, r.period, cache)
        if not temporal["confirmed"]:
            print(f"   Gaia {r.source_id} rejected by independent temporal check: "
                  f"{temporal['reason']}")
            continue
        hits = prior_art(r.ra, r.dec)
        real = [h for h in hits if "ERROR" not in h]
        errs = [h for h in hits if "ERROR" in h]
        rec = r._asdict(); rec.pop("Index", None)
        rec.update(box_ra=ra0, box_dec=dec0, prior_art=";".join(hits),
                   search_mode="ml" if clf is not None else "untrained_signal", **temporal)
        if real:
            rec["status"] = "ALREADY CATALOGUED - positive prior-art match"
            known.append(rec)
            continue
        if errs:
            rec["status"] = "UNVERIFIED - prior-art service unavailable"
            unverified.append(rec)
        else:
            rec["status"] = "CANDIDATE - manual imaging and literature checks required"
            good.append(rec)
            fold_plot(r.ra, r.dec, r.period, cache, os.path.join(out, f"fold_{r.source_id}.png"))
    anomalies, known_anomalies = collect_anomaly_leads(
        unk, args, cache, out,
        already_handled=[x["source_id"] for x in good + unverified + known])
    return good, unverified, anomalies, known + known_anomalies, len(vw)


def auto(args):
    if args.search_mode == "signal":
        print("TRAIN-FIRST SURVEY: untrained signal mode is not allowed in auto. "
              "Use --search-mode ml. No VarWISE data was accessed.")
        return 2
    os.makedirs(args.out, exist_ok=True)
    token, config = survey_fingerprint(args)
    try:
        initialize_survey(args.out, token, config)
    except RuntimeError as ex:
        print(f"STOP: {ex}")
        return 2
    original_stdout = sys.stdout
    log = Tee(os.path.join(args.out, "auto.log"))
    sys.stdout = log
    def finish(code):
        sys.stdout = original_stdout
        log.close()
        return code
    t0 = time.time()
    print(f"=== AUTO START {time.strftime('%Y-%m-%d %H:%M:%S')}  out={args.out} ===")
    print("Data source: real archives only (IRSA VarWISE, Gaia DR3, ZTF). No synthetic data is used.")
    # NO preflight touching VarWISE. The first network actions are exclusively
    # external Gaia training metadata and their ZTF observations.
    print("\n[1/3] TRAIN on external Gaia/ZTF sources. Discovery dataset sealed.")
    boxes = generate_boxes(args.box_size, args.bmin, args.dec_min, args.dec_max)
    print(f"sky tiles: {len(boxes)} (Dec {args.dec_min}..{args.dec_max}, |b| >= {args.bmin})")
    if args.bmin > 0 or args.dec_min > -90 or args.dec_max < 90:
        print("WARNING: YOU REQUESTED INCOMPLETE SKY COVERAGE by filtering latitude/declination")
    print("NOTE: ZTF optical coverage is NOT full sky even when VarWISE tiling is full sky")
    # 2. validate before obtaining even a count of VarWISE target objects
    mp = os.path.join(args.out, "model.joblib")
    pack = None if args.retrain else load_validated_model(mp)
    if pack is None:
        print("\n[2/3] CREATE SEALED 80/20 HOLDOUT, FIT FIRST, TEST AFTER")
        external_training_grid = generate_boxes(args.box_size, 0.0, -90.0, 90.0)
        pack = train_and_validate(args, external_training_grid, args.out)
    else:
        print("[2/3] Using previously sealed-and-validated model from this survey")
    if pack is None:
        print("\nHARD STOP: model absent or failed real held-out validation. "
              "No VarWISE target data was queried.")
        return finish(2)
    clf = pack["model"]
    # 3. search, resumable
    print("\n[3/3] SEARCH")
    state = survey_checkpoint(args.out, token)
    cache = os.path.join(args.out, "ztf_cache"); os.makedirs(cache, exist_ok=True)
    cpath = os.path.join(args.out, "candidates.csv")
    try:
        # Reserve ALL metadata-sampled training tiles, even if sampling took
        # place outside the first 16 survey tiles.
        training_tiles = {tuple(x) for x in pack["report"]["training_tiles"]}
        training_tiles.update(tuple(x) for x in pack["report"]["validation_tiles"])
        search_boxes = [box for box in boxes if box not in training_tiles]
        for ra0, dec0, ra_width, dec_height in search_boxes:
            key = f"{ra0:.8f},{dec0:.8f},{ra_width:.8f},{dec_height:.8f}"
            if key in state["done"]:
                continue
            if state["n_candidates"] >= args.target:
                break
            if time.time() - t0 > args.hours * 3600:
                print("time limit reached"); break
            print(f"\n-- box {key}  (done {len(state['done'])}, candidates {state['n_candidates']}/{args.target})")
            try:
                good, unver, anomal, known, nvw = process_box(
                    args, clf, ra0, dec0, args.out, cache, ra_width, dec_height)
            except Exception as ex:
                print(f"   box failed: {type(ex).__name__}: {str(ex)[:200]}  (will retry next run)")
                continue
            # COMMIT: one atomic tile result; rebuild CSVs from committed tiles.
            # A crash at ANY point cannot double-count a candidate on resume.
            tilepath = os.path.join(args.out, "committed_tiles",
                                    hashlib.sha256(key.encode()).hexdigest()+".json")
            atomic_json(tilepath, dict(key=key, fingerprint=token, scanned=nvw,
                                       candidates=good, unverified=unver,
                                       anomalies=anomal, known=known))
            state = survey_checkpoint(args.out, token)
            print(f"   VarWISE objects {nvw}; new candidates {len(good)}; "
                  f"unverified {len(unver)}; anomaly leads {len(anomal)}; known {len(known)}")
    except KeyboardInterrupt:
        print("\ninterrupted - progress saved; rerun the same command to resume")
    print(f"\n=== DONE: {state['n_candidates']} candidates in {cpath} "
          f"after {len(state['done'])} boxes, {state['scanned']} VarWISE objects ===")
    print("These are UNVERIFIED candidates, NOT verified discoveries. Still required: eyeball fold_*.png, "
          "imaging for blends, double-period check for eclipsing binaries, MPChecker, VSX website search.")
    return finish(0)


def brutetest(args):
    """Adversarial, reproducible SIMULATION: train on sinusoids; hide new morphologies.

    This is a falsification test, NOT a measurement of real-sky precision or
    proof that any astrophysical object is previously undiscovered.
    """
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(20261027)
    os.makedirs(args.out, exist_ok=True)
    n = args.brutetest_n
    pool = []
    print("[BRUTE] generating known sine and constant stars in independent sky cells", flush=True)
    for k in range(8 * n):
        label = int(k % 2 == 0)
        t, m, e, _ = synth_lc(rng, bool(label), n=150)
        f = lc_features(t, m, e, spp=4)
        f["label"] = label
        f["cell"] = sky_cell(rng.uniform(0, 30), rng.uniform(0, 30), cell=3)
        f["source_id"] = k
        pool.append(f)
    df = pd.DataFrame(pool)
    clf, auc, auc0 = train_eval(df)
    test = spatial_test_mask(df)
    te = df[test].copy()
    te["score"] = clf.predict_proba(te[FEATURES])[:, 1]
    te["found"] = passes_cuts(te)
    rows = []
    for kind in ("sine", "eclipse", "period_drift", "single_event", "constant",
                 "instrumental_glitches"):
        print(f"[BRUTE] unseen target morphology: {kind}", flush=True)
        for _ in range(n):
            t = np.sort(rng.uniform(0, 1000, 160))
            e = np.full(len(t), rng.uniform(0.022, 0.038))
            p = 10 ** rng.uniform(np.log10(0.6), np.log10(15))
            amp = rng.uniform(0.18, 0.45)
            phase = rng.uniform(0, 1)
            m = np.full(len(t), rng.uniform(13.5, 17.5))
            if kind == "sine":
                m += amp * np.sin(2 * np.pi * (t / p + phase))
            elif kind == "eclipse":
                ph = (t / p + phase) % 1
                m += amp * (ph < 0.12)
            elif kind == "period_drift":
                m += amp * np.sin(2 * np.pi * (t / p + 0.6 * (t / 1000) ** 2 + phase))
            elif kind == "single_event":
                center = rng.uniform(200, 800)
                m += amp * np.exp(-0.5 * ((t - center) / 20) ** 2)
            elif kind == "instrumental_glitches":
                # Adversarial 8-night calibration failures: not astrophysics.
                # If flagged, this quantifies a limitation of the anomaly gate.
                bad = rng.choice(len(t), size=8, replace=False)
                m[bad] += amp
            m += rng.normal(0, e)
            f = lc_features(t, m, e, spp=4)
            if f is None:
                continue
            f["score"] = float(clf.predict_proba(pd.DataFrame([f])[FEATURES])[0, 1])
            ft = pd.DataFrame([f])
            phys = bool(passes_cuts(ft, require_model=False).iloc[0])
            ml = bool(passes_cuts(ft, require_model=True).iloc[0])
            temp = temporal_consistency_lc(pd.DataFrame({"mjd": t, "mag": m,
                                                          "magerr": e}), f["period"])["confirmed"]
            anomaly = outlier_lead_features(t, m, e)
            pgood = any(abs(f["period"] / p / k - 1) < 0.03 for k in (0.5, 1, 2))
            rows.append(dict(kind=kind, target_label_available_to_model=False,
                             score=f["score"], signal_gate=phys, ml_gate=ml,
                             temporal_gate=bool(temp), signal_and_temporal=bool(phys and temp),
                             ml_and_temporal=bool(ml and temp),
                             anomaly_lead=anomaly["flag"],
                             recovered_period=bool(pgood) if kind in
                             ("sine", "eclipse", "period_drift") else None))
    d = pd.DataFrame(rows)
    summary = {}
    for kind, g in d.groupby("kind", sort=False):
        summary[kind] = dict(n=len(g),
                             mean_model_score=round(float(g["score"].mean()), 4),
                             signal_fraction=round(float(g["signal_and_temporal"].mean()), 4),
                             ml_fraction=round(float(g["ml_and_temporal"].mean()), 4),
                             unverified_anomaly_fraction=round(float(g["anomaly_lead"].mean()), 4))
    checks = dict(training_auc_at_least_0_90=bool(auc >= .90),
                  shuffled_auc_near_chance=bool(abs(auc0 - .5) <= .15),
                  sine_recovery=bool(summary["sine"]["signal_fraction"] >= .8),
                  drifting_period_recovery=bool(summary["period_drift"]["signal_fraction"] >= .6),
                  unseen_eclipses_anomaly=bool(summary["eclipse"]["unverified_anomaly_fraction"] >= .6),
                  unseen_events_anomaly=bool(summary["single_event"]["unverified_anomaly_fraction"] >= .5),
                  constant_false_leads=bool(summary["constant"]["unverified_anomaly_fraction"] <= .1))
    report = dict(provenance="SYNTHETIC - not real-sky performance",
                  training_morphologies=["sine", "constant"],
                  withheld_target_morphologies=["eclipse", "period_drift", "single_event"],
                  training_sources=int((~test).sum()), validation_sources=int(test.sum()),
                  validation_auc=float(auc), shuffled_auc=float(auc0),
                  heldout_positive_signal_recall=float(te.loc[te.label == 1, "found"].mean()),
                  heldout_negative_false_positive_rate=float(te.loc[te.label == 0, "found"].mean()),
                  target_classes=summary, screening_checks=checks,
                  screening_pass=all(checks.values()),
                  artifact_discrimination_pass=bool(
                      summary["instrumental_glitches"]["unverified_anomaly_fraction"] <= .1),
                  discovery_validated=False,
                  important_limitation="instrumental_glitches are NOT astrophysical; "
                  "if anomaly gate flags them, human imaging/quality checks remain mandatory")
    d.to_csv(os.path.join(args.out, "brutetest_individual_trials.csv"), index=False)
    with open(os.path.join(args.out, "brutetest_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\n=== BRUTE RESULTS: synthetic only ===")
    print(json.dumps(report, indent=2))
    print("This does not validate detection of unknown physical classes or prove catalog novelty.")
    return 0 if report["screening_pass"] else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["selftest", "brutetest", "probe", "smoke", "train", "run", "auto"])
    ap.add_argument("--ra0", type=float, default=150.0)
    ap.add_argument("--dec0", type=float, default=30.0)
    ap.add_argument("--size", type=float, default=4.0, help="box side, degrees")
    ap.add_argument("--out", default="out")
    ap.add_argument("--table", default=None)
    ap.add_argument("--ra-col", default=None); ap.add_argument("--dec-col", default=None)
    ap.add_argument("--max-vw", type=int, default=20000,
                    help="maximum VarWISE rows per complete TAP leaf; automatically subdivides denser tiles")
    ap.add_argument("--n-train", type=int, default=400, help="stars per class")
    ap.add_argument("--max-prior", type=int, default=60, help="max survivors sent to prior-art")
    ap.add_argument("--target", type=int, default=5, help="auto: stop after this many UNVERIFIED candidates")
    ap.add_argument("--hours", type=float, default=12.0, help="auto: wall-clock limit for the search")
    ap.add_argument("--box-size", type=float, default=6.0, help="auto: box side in degrees")
    ap.add_argument("--bmin", type=float, default=0.0,
                    help="auto: optional Galactic-plane exclusion, 0 means NO exclusion")
    ap.add_argument("--dec-min", type=float, default=-90.0, help="auto: minimum survey declination")
    ap.add_argument("--dec-max", type=float, default=90.0, help="auto: maximum survey declination")
    ap.add_argument("--train-per-box", type=int, default=150)
    ap.add_argument("--min-train", type=int, default=400, help="auto: usable training stars required")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--search-mode", choices=["auto", "ml", "signal"], default="ml",
                    help="ml (default) and auto: require externally fitted model and sealed "
                         "held-out 80/20 validation; signal: explicitly disabled unless diagnostic override")
    ap.add_argument("--allow-untrained-diagnostics", action="store_true",
                    help="explicit research-only override for untrained `run`/`smoke`; "
                         "NOT supported by strict `auto` survey")
    ap.add_argument("--skip-gaia", action="store_true", default=True,
                    help="default: search every VarWISE sky position, even without a Gaia match; "
                         "uses negative local IDs, never pretends they are Gaia IDs")
    ap.add_argument("--require-gaia", dest="skip_gaia", action="store_false",
                    help="restrict to VarWISE positions with a unique Gaia G=12..19 counterpart")
    ap.add_argument("--include-anomalies", action="store_true",
                    help="also queue nonperiodic/eclipsing outlier patterns as unverified "
                         "anomaly_leads.csv (not confirmed discoveries)")
    ap.add_argument("--brutetest-n", type=int, default=50,
                    help="trials per target morphology in the synthetic adversarial test")
    a = ap.parse_args()
    if a.cmd == "selftest":
        sys.exit(selftest())
    if a.cmd == "brutetest":
        sys.exit(brutetest(a))
    if a.cmd == "probe":
        probe(); return
    if a.cmd == "smoke":
        if not a.allow_untrained_diagnostics and load_validated_model(
                os.path.join(a.out, "model.joblib")) is None:
            print("TRAIN-FIRST LOCK: smoke would query VarWISE before validation; refused.")
            sys.exit(2)
        sys.exit(1 if smoke(a) else 0)
    if a.cmd == "auto":
        sys.exit(auto(a))
    if a.cmd == "train":
        sys.exit(train_only(a))
    sys.exit(run(a))


if __name__ == "__main__":
    main()
