"""Real-data diagnostic of a known ZTF periodic variable and a cadence alias."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run import ROOT, RESULTS


def main():
    field=json.loads((ROOT/"data"/"ztf_dr24_field_298_29_45arcsec.json").read_text())
    check=json.loads((RESULTS/"ztf_catalog_check.json").read_text())
    top=check["ranked_positions"][0]
    matches=[x for x in top["published_catalog_matches_within_2arcsec"] if x["catalog"]=="ztf_2020_periodic"]
    if not matches:
        raise RuntimeError("Top source not found in ZTF periodic reference catalog")
    known_period=float(matches[0]["catalog_record"]["Per-r"])
    inferred_period=top["representative_period_days"]
    samples=[]
    for desc in top["filters_and_oids"]:
        item=field[desc["oid"]]
        arr=np.array([[x["mjd"],x["mag"],x["magerr"]] for x in item["lc"]],dtype=float)
        keep=(np.isfinite(arr).all(axis=1)&(arr[:,2]>0)&(arr[:,2]<.2))
        samples.append((desc["filter"],arr[keep]))
    fig,ax=plt.subplots(1,3,figsize=(15,4),layout="constrained")
    colors={"zg":"#188b95","zr":"#e6802c","zi":"#ac5eb5"}
    for filt,arr in samples:
        color=colors.get(filt,"#63748b")
        ax[0].scatter(arr[:,0],arr[:,1],s=4,alpha=.55,label=f"{filt}: {len(arr)} rows",c=color)
        ax[1].scatter((arr[:,0]/known_period)%1,arr[:,1],s=5,alpha=.55,c=color,label=filt)
        ax[2].scatter((arr[:,0]/inferred_period)%1,arr[:,1],s=5,alpha=.55,c=color,label=filt)
    ax[0].set(xlabel="Heliocentric MJD (days)",ylabel="ZTF magnitude",title="Real ZTF DR24 observations")
    ax[1].set(xlabel="Phase",title=f"Published r-band period: {known_period:.5f} d")
    ax[2].set(xlabel="Phase",title=f"Our coarse-grid peak: {inferred_period:.5f} d")
    for a in ax:
        a.invert_yaxis()
        a.grid(alpha=.15)
    ax[0].legend(loc="best",fontsize=8)
    fig.suptitle("Known periodic object, NOT a new discovery — illustrating period alias risk",fontsize=12)
    out=RESULTS/"ztf_known_source_period_alias.png"
    fig.savefig(out,dpi=150)
    print(f"Saved {out}")


if __name__=="__main__":
    main()
