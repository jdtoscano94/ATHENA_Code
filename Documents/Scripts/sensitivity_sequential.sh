#!/bin/bash
# Run N isolated ATHENA runs one after another on the current node (e.g. the single H100 for the
# PINN case), each in its own box (athena_box_run.sh). Same one-folder layout as sensitivity_run.sbatch:
#   Results/Sensitivity/<team>/<case>/run<N>/   job_info, stdout, logs, usage, dbs, results, box
#
# Usage: bash Documents/Scripts/sensitivity_sequential.sh <request.json> <team> <case> <first> <last> [wait_pid]
#   wait_pid: optional PID to wait for before starting (keeps the GPU to one run at a time)
set -u
REQUEST="$1"; TEAM="$2"; CASE="$3"; FIRST="$4"; LAST="$5"; WAIT_PID="${6:-}"
REPO="$(readlink -f "$(dirname "$0")/../..")"
cd "$REPO" || exit 1

if [ -n "$WAIT_PID" ]; then
    echo "$(date -Is) waiting for PID $WAIT_PID to finish"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi

for RUN in $(seq "$FIRST" "$LAST"); do
    RUN_DIR="$REPO/Results/Sensitivity/$TEAM/$CASE/run$RUN"
    if [ -e "$RUN_DIR/stdout.txt" ]; then
        echo "$(date -Is) run$RUN exists, skipping (never overwrite): $RUN_DIR"
        continue
    fi
    echo "$(date -Is) starting $CASE run$RUN"
    bash "$REPO/Documents/Scripts/athena_box_run.sh" "$REQUEST" "$TEAM" "$RUN_DIR"
    echo "$(date -Is) finished $CASE run$RUN exit=$?"
done
echo "$(date -Is) all runs done"
