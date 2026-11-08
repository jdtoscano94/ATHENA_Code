#!/bin/bash
# Run one generated Julia script. Usage: run_julia.sh /path/to/script.jl /path/to/stderr.log
#
# Inside the box (ATHENA_IN_BOX=1, set by athena_box_run.sh) it runs directly with the image's Julia.
# Outside a box, by default it runs inside the ATHENA release container (container/images/athena_release.sif), a box:
#   - Julia 1.11.9 + the frozen ATHENA project (Trixi 0.13.15), precompiled, read-only in the image
#   - only the script's folder is visible and writable (--containall), plus a per-job writable
#     Julia depot, so a Pkg.add or a recompile lands there and never touches a shared environment
#   - the reference data (Documents/codes/Data) and earlier runs (Results/) are visible read-only
# ATHENA_EXEC=host runs on the host instead (no container): ATHENA_JULIA_ENV is the Julia project
#   folder (instantiated from container/julia/{Project,Manifest}.toml) and ATHENA_JULIA_EXEC the
#   julia binary (default: $ATHENA_JULIA_ENV/julia-1.11.9/bin/julia if present, else julia on PATH).

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ATHENA_EXEC="${ATHENA_EXEC:-container}"
ATHENA_SIF="${ATHENA_SIF:-$REPO/container/images/athena_release.sif}"
DATA_DIR="$REPO/Documents/codes/Data"
RESULTS_DIR="$REPO/Results"

# Host mode only: the Julia project and binary.
FROZEN_ENV="${ATHENA_JULIA_ENV:-}"
JULIA_EXEC="${ATHENA_JULIA_EXEC:-}"
if [ -z "$JULIA_EXEC" ]; then
    if [ -n "$FROZEN_ENV" ] && [ -x "$FROZEN_ENV/julia-1.11.9/bin/julia" ]; then
        JULIA_EXEC="$FROZEN_ENV/julia-1.11.9/bin/julia"
    else
        JULIA_EXEC="$(command -v julia 2>/dev/null || true)"
    fi
fi
JULIA_CPU_TARGET='generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1)'

# Writable depot, one per job: concurrent runs never compile into the same cache.
RUN_DEPOT="${ATHENA_JULIA_RUN_DEPOT:-$HOME/.athena_julia_run_depots/${SLURM_JOB_ID:-local_$$}}"

# Number of threads: prefer the SLURM allocation, else the cgroup-visible CPU
# count (nproc respects the cores actually granted to this job).
JULIA_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-$(nproc)}}"

set -e

# --- Argument Handling ---
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 /path/to/script.jl /path/to/stderr.log" >&2
    exit 1
fi
JULIA_SCRIPT="$(readlink -f "$1")"
STDERR_FILE="$2"
SCRIPT_DIR=$(dirname "$JULIA_SCRIPT")

if [ ! -f "$JULIA_SCRIPT" ]; then
    echo "Error: Julia script not found at $JULIA_SCRIPT" >&2
    exit 1
fi
[ "${ATHENA_IN_BOX:-0}" = 1 ] || mkdir -p "$RUN_DEPOT"
cd "$SCRIPT_DIR" || { echo "Error: Failed to change directory to '$SCRIPT_DIR'." >&2; exit 1; }

# --- Execution (stderr goes to the log file the Python side reads) ---
# The generated code never sees the API keys.
NOKEYS=(env -u OLLAMA_API -u OLLAMA_API_KEY -u GOOGLE_API_KEY -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u XAI_API_KEY)
if [ "${ATHENA_IN_BOX:-0}" = 1 ]; then
    # already inside the box (athena_box_run.sh): the image's Julia and frozen project, the
    # run's own writable depot in front of the read-only image depot
    "${NOKEYS[@]}" JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_NUM_THREADS="$JULIA_THREADS" \
        julia --threads="$JULIA_THREADS" --color=no "$JULIA_SCRIPT" 2> "$STDERR_FILE"
elif [ "$ATHENA_EXEC" = container ]; then
    [ -f "$ATHENA_SIF" ] || { echo "Error: container image not found at $ATHENA_SIF" >&2; exit 1; }
    DATA_BIND=()
    [ -d "$DATA_DIR" ] && DATA_BIND=(-B "$DATA_DIR:$DATA_DIR:ro")
    # earlier runs under Results/ read-only (a later stage reads an earlier stage's output); the
    # script's own folder is bound after it, so it stays writable even when it is inside Results/
    RESULTS_BIND=()
    [ -d "$RESULTS_DIR" ] && RESULTS_BIND=(-B "$RESULTS_DIR:$RESULTS_DIR:ro")
    apptainer exec --containall --cleanenv \
        "${RESULTS_BIND[@]}" \
        -B "$SCRIPT_DIR:$SCRIPT_DIR" -B "$RUN_DEPOT:/box/julia_depot" \
        "${DATA_BIND[@]}" \
        --pwd "$SCRIPT_DIR" \
        --env ATHENA_JULIA_DEPOT=/box/julia_depot \
        --env JULIA_PKG_PRECOMPILE_AUTO=0 \
        --env JULIA_NUM_THREADS="$JULIA_THREADS" \
        "$ATHENA_SIF" \
        julia --threads="$JULIA_THREADS" --color=no "$JULIA_SCRIPT" 2> "$STDERR_FILE"
else
    [ -n "$FROZEN_ENV" ] && [ -d "$FROZEN_ENV" ] || { echo "Error: host mode needs ATHENA_JULIA_ENV (the Julia project folder)" >&2; exit 1; }
    [ -n "$JULIA_EXEC" ] && [ -x "$JULIA_EXEC" ] || { echo "Error: julia not found; set ATHENA_JULIA_EXEC" >&2; exit 1; }
    export JULIA_DEPOT_PATH="$RUN_DEPOT:$FROZEN_ENV/depot:${ATHENA_JULIA_USER_DEPOT:-$HOME/.julia}"
    export JULIA_CPU_TARGET JULIA_PKG_PRECOMPILE_AUTO=0
    "$JULIA_EXEC" --project="$FROZEN_ENV" --threads="$JULIA_THREADS" --color=no "$JULIA_SCRIPT" 2> "$STDERR_FILE"
fi
