"""
Cross-match the lens & NN objects from lens_neighbors_final_figure against
SIMBAD and NED. We do a 3" cone for SIMBAD (per Carol's snippet) and a 5"
cone for NED to catch nearby catalogued sources.
"""
import json, time, sys
from pathlib import Path

from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.simbad import Simbad
from astroquery.ipac.ned import Ned

HERE = Path(__file__).resolve().parent
OBJ_PATH = HERE / "objects.json"
OUT_PATH = HERE / "results.json"
TXT_PATH = HERE / "results.txt"

SIMBAD_RADIUS = 3 * u.arcsec
NED_RADIUS    = 5 * u.arcsec

simbad = Simbad()
simbad.add_votable_fields("otype", "ids")  # bibcodelist not needed for now

def query_simbad(ra, dec):
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    try:
        r = simbad.query_region(coord, radius=SIMBAD_RADIUS)
    except Exception as e:
        return {"error": str(e)}
    if r is None or len(r) == 0:
        return {"hits": []}
    hits = []
    cols = {c.lower(): c for c in r.colnames}
    main_id = cols.get("main_id", "MAIN_ID")
    otype   = cols.get("otype", "OTYPE")
    ids_col = cols.get("ids", "IDS")
    for row in r:
        hits.append({
            "main_id": str(row[main_id]),
            "otype":   str(row[otype]) if otype in r.colnames else "",
            "ids":     str(row[ids_col]) if ids_col in r.colnames else "",
        })
    return {"hits": hits}


def query_ned(ra, dec):
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    try:
        r = Ned.query_region(coord, radius=NED_RADIUS)
    except Exception as e:
        return {"error": str(e)}
    if r is None or len(r) == 0:
        return {"hits": []}
    hits = []
    for row in r:
        hits.append({
            "name": str(row["Object Name"]),
            "type": str(row["Type"]) if "Type" in r.colnames else "",
            "z":    (None if "Redshift" not in r.colnames or
                     row["Redshift"] is None or
                     (hasattr(row["Redshift"], "mask") and row["Redshift"].mask)
                     else float(row["Redshift"])),
        })
    return {"hits": hits}


def main():
    with open(OBJ_PATH) as f:
        objs = json.load(f)

    results = []
    for o in objs:
        ra, dec = o["ra"], o["dec"]
        print(f"[L{o['lens']:>2} {o['label']:<7} {o['survey']:<6}] "
              f"RA={ra:9.5f} Dec={dec:+9.5f} ...", flush=True)
        sim = query_simbad(ra, dec)
        time.sleep(0.4)
        ned = query_ned(ra, dec)
        time.sleep(0.4)
        r = dict(o, simbad=sim, ned=ned)
        results.append(r)

        # Print short status
        n_sim = len(sim.get("hits", []))
        n_ned = len(ned.get("hits", []))
        print(f"  SIMBAD={n_sim}  NED={n_ned}", flush=True)
        if n_sim:
            for h in sim["hits"]:
                print(f"    SIMBAD  {h['main_id']:30s}  otype={h['otype']}", flush=True)
        if n_ned:
            for h in ned["hits"]:
                print(f"    NED     {h['name']:30s}  type={h['type']}  z={h['z']}", flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved JSON: {OUT_PATH}")

    # Plain-text summary
    lines = []
    lines.append(f"{'Lens':<5}{'Label':<7}{'Survey':<8}{'RA':>10}{'Dec':>10}  {'obj_id':<22}  SIMBAD                          NED")
    for r in results:
        sim_hits = r["simbad"].get("hits", [])
        ned_hits = r["ned"].get("hits", [])
        sim_str = "—" if not sim_hits else "; ".join(f"{h['main_id']} [{h['otype']}]" for h in sim_hits)
        ned_str = "—" if not ned_hits else "; ".join(
            f"{h['name']} [{h['type']}]" + (f" z={h['z']}" if h['z'] is not None else "")
            for h in ned_hits
        )
        lines.append(
            f"{r['lens']:<5}{r['label']:<7}{r['survey']:<8}"
            f"{r['ra']:10.5f}{r['dec']:10.5f}  "
            f"{r['obj_id']:<22}  "
            f"SIMBAD: {sim_str}  |  NED: {ned_str}"
        )
    TXT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Saved text: {TXT_PATH}")


if __name__ == "__main__":
    main()
