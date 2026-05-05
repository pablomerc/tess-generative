#!/bin/bash
#SBATCH -J lens-simbad-check
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/job.%j.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/job.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 00:30:00
#SBATCH -p devel

source ~/.bashrc
conda activate torchenv

PY=/work1/jeroenaudenaert/pablomer/miniconda3/envs/torchenv/bin/python
DISCORD="https://discord.com/api/webhooks/1488692651334177071/8b8KvACfQIVYCNY3ovee04BixCEWiqbqp1iQk4z9sXHlgR29kMkGIjl1pahV5uEPSbxe"
HERE="/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/neighbor_search_lenses/check_objects"

notify_file() {
    # Send a file's contents (or arbitrary text) as a Discord message,
    # chunked to <2000 char to satisfy webhook limits.
    local payload="$1"
    "$PY" - "$DISCORD" <<PY
import json, sys, urllib.request
url = sys.argv[1]
text = ${payload@Q}
CHUNK = 1900
def post(content):
    data = json.dumps({"content": content}).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type":"application/json",
                 "User-Agent":"claude-code-slurm/1.0"},
    )
    try:
        urllib.request.urlopen(req, timeout=20).read()
    except Exception as e:
        print("discord notify failed:", e)
for i in range(0, len(text), CHUNK):
    post(text[i:i+CHUNK])
PY
}

notify_file "Lens SIMBAD/NED check started (job $SLURM_JOB_ID)"

cd "$HERE"
"$PY" check_simbad_ned.py
status=$?

if [ $status -eq 0 ] && [ -f "$HERE/results.json" ]; then
    SUMMARY=$("$PY" - <<'PY'
import json
with open("/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/neighbor_search_lenses/check_objects/results.json") as f:
    rs = json.load(f)
lines = ["Lens neighbour SIMBAD/NED cross-match (3\" SIMBAD, 5\" NED)"]
for r in rs:
    sim = r["simbad"].get("hits", [])
    ned = r["ned"].get("hits", [])
    sim_str = "—" if not sim else "; ".join(f"{h['main_id']}[{h['otype']}]" for h in sim)
    ned_str = "—" if not ned else "; ".join(
        f"{h['name']}[{h['type']}]" + (f" z={h['z']:.3f}" if h['z'] is not None else "")
        for h in ned
    )
    lines.append(f"L{r['lens']:>2} {r['label']:<6} {r['survey']:<6} RA={r['ra']:.4f} Dec={r['dec']:+.4f} | SIMBAD: {sim_str} | NED: {ned_str}")
print("\n".join(lines))
PY
)
    notify_file "$SUMMARY"
    notify_file "Lens SIMBAD/NED check finished OK (job $SLURM_JOB_ID). Files: $HERE/results.{json,txt}"
else
    ERR_TAIL=$(tail -40 "/work1/jeroenaudenaert/pablomer/logs/job.${SLURM_JOB_ID}.err" 2>/dev/null)
    notify_file "Lens SIMBAD/NED check FAILED (job $SLURM_JOB_ID, exit $status).
stderr tail:
$ERR_TAIL"
fi

exit $status
