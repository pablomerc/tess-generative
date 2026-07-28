#!/bin/bash
#SBATCH --job-name=stage-ram48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --partition=mit_normal
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#
# Copy the ram48 image store off the contended pool volume onto the group allocation.
#
# Measured 2026-07-27 from the login node, uncached O_DIRECT reads:
#   /orcd/pool/007/pablomer      (hstor012-n1, 1 TB, 91% full)  ->   3.9 MB/s
#   /orcd/data/mki_aryeh/001     (hstor003-n1, 200 TB, 1% full) ->   1.4 GB/s
# Buffered sequential reads on pool manage ~25 MB/s thanks to readahead, which is why
# every job that touches this 17 GB store has been paying a ~45 min startup tax.
#
# CPU-only job: this is pure I/O, so it must not sit on a GPU.
#
# Submit (ideally once the pool-reading jobs are done, so they don't fight for
# bandwidth on hstor012):
#   sbatch --dependency=afterany:<JOBIDS> stage_store_to_group.slurm

set -eo pipefail

: "${SRC:=/orcd/pool/007/pablomer/efficient_neighs_48}"
: "${DST_PARENT:=/orcd/data/mki_aryeh/001/pablomer}"
DST="$DST_PARENT/$(basename "$SRC")"
WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"

_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }

SRC_BYTES=$(du -sb "$SRC" | cut -f1)
_ping "stage-ram48 🚀 start · job ${SLURM_JOB_ID:-?} · $(numfmt --to=iec "$SRC_BYTES") · \`${SRC}\` → \`${DST}\`"
cleanup() {
  local rc=$?
  if [[ ${rc} -ne 0 ]]; then
    _ping "stage-ram48 ❌ failed (exit ${rc}) · job ${SLURM_JOB_ID:-?} · \`${DST}\` left INCOMPLETE — do not point jobs at it"
  fi
}
trap cleanup EXIT

mkdir -p "$DST_PARENT"
echo "[stage] src=$SRC ($(numfmt --to=iec "$SRC_BYTES"))"
echo "[stage] dst=$DST"
df -hT "$SRC" "$DST_PARENT" | sed 's/  */ /g'

t0=$(date +%s)
# --partial + a .inprogress staging dir: a killed job can resume rather than restart, and
# nothing ever appears at the final path unless the whole copy verified.
STAGE="$DST.inprogress"
mkdir -p "$STAGE"
rsync -a --partial --info=progress2 "$SRC"/ "$STAGE"/
t1=$(date +%s)
elapsed=$(( t1 - t0 ))
rate=$(python3 -c "print(f'{$SRC_BYTES/1048576/max(1,$elapsed):.0f}')")
echo "[stage] copied in ${elapsed}s (~${rate} MB/s)"

echo "[stage] verifying byte counts per file..."
fail=0
while IFS= read -r rel; do
    a=$(stat -c %s "$SRC/$rel")
    b=$(stat -c %s "$STAGE/$rel" 2>/dev/null || echo -1)
    if [[ "$a" != "$b" ]]; then
        echo "[stage] MISMATCH $rel: src=$a dst=$b"; fail=1
    else
        echo "[stage] ok $rel ($(numfmt --to=iec "$a"))"
    fi
done < <(cd "$SRC" && find . -type f -printf '%P\n')
if [[ $fail -ne 0 ]]; then
    echo "[stage] verification FAILED — leaving $STAGE in place for inspection"
    exit 1
fi

# Only now does the destination become visible under its real name.
rm -rf "$DST"
mv "$STAGE" "$DST"

echo "[stage] read-back check from the new location:"
dd if="$DST/hsc48.bin" of=/dev/null bs=1M count=512 iflag=direct 2>&1 | tail -1

_ping "stage-ram48 ✅ done · job ${SLURM_JOB_ID:-?} · ${elapsed}s (~${rate} MB/s) · \`${DST}\` verified · point DATA_DIR here"
echo "[stage] done at $(date)"
echo
echo "To use it, set DATA_DIR / efficient_data_dir to:"
echo "  $DST"
