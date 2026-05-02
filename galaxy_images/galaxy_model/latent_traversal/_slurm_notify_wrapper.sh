#!/bin/bash
# Source this from a slurm script to wrap it with Discord start/finish/fail notifications.
# Expects $LATENT_DIR set to the directory containing discord_notify.py.

NOTIFY_PY="python3 ${LATENT_DIR}/discord_notify.py"
JOB_ID="${SLURM_JOB_ID:-local}"
JOB_NAME="${SLURM_JOB_NAME:-script}"
HOST="$(hostname)"
START_TS=$(date +%s)

discord_notify() { $NOTIFY_PY "$1" || true; }

_on_exit() {
    local code=$?
    local end_ts=$(date +%s)
    local elapsed=$((end_ts - START_TS))
    local mins=$((elapsed / 60))
    local secs=$((elapsed % 60))
    if [ "$code" -eq 0 ]; then
        discord_notify "✅ \`${JOB_NAME}\` (job ${JOB_ID}) finished OK in ${mins}m${secs}s on ${HOST}"
    else
        discord_notify "❌ \`${JOB_NAME}\` (job ${JOB_ID}) FAILED exit=${code} after ${mins}m${secs}s on ${HOST} — check /work1/jeroenaudenaert/pablomer/logs/"
    fi
}
trap _on_exit EXIT

discord_notify "🟢 \`${JOB_NAME}\` (job ${JOB_ID}) started on ${HOST}"
