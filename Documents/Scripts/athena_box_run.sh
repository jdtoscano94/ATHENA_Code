#!/bin/bash
# One ATHENA run in one box. Everything the run makes lives in ONE folder:
#
#   <RUN_DIR>/
#     job_info.txt     node, GPU, start/end, git commit, exit code
#     stdout.txt       the full console output
#     logs/            agent transcripts (hena_*.txt)
#     usage/           token / time log, one record per LLM call and code execution
#     dbs/             private copies of the vector DBs this run used
#     results/         generated code, solver / training output, Experiment_Report.md per iteration
#     box/             the box's own home and writable Julia depot (anything installed lands here)
#
# The box is the release container (container/images/athena_release.sif): the orchestrator, the
# agents and the generated code all run inside it. The repository is visible read-only; RUN_DIR is
# the only writable place. API keys enter through an env file outside the box and are removed from
# the environment of the generated code (run_julia.sh / run_python.sh).
#
# Usage: athena_box_run.sh <request.json> <team> <RUN_DIR> [extra run_athena.py options, e.g. --resume DIR]
set -u
REQUEST="$1"; TEAM="$2"; RUN_DIR="$3"; shift 3
REPO="$(readlink -f "$(dirname "${BASH_SOURCE[0]}")/../..")"
SIF="${ATHENA_SIF:-$REPO/container/images/athena_release.sif}"
[ -f "$SIF" ] || { echo "container image not found: $SIF" >&2; exit 1; }
[ -f "$REPO/$REQUEST" ] || [ -f "$REQUEST" ] || { echo "request not found: $REQUEST" >&2; exit 1; }

# --resume DIR: resolve to the real path (the box sees the repo only under its real path, not
# through the /users/... symlink)
RESUME=0; ARGS=(); prev=""
for a in "$@"; do
    if [ "$prev" = --resume ]; then
        a="$(readlink -f "$a")"
        case "$a/" in "$REPO"/Results/*) ;; *) echo "--resume DIR must be under $REPO/Results/" >&2; exit 1 ;; esac
    fi
    [ "$a" = --resume ] && RESUME=1
    ARGS+=("$a"); prev="$a"
done
set -- "${ARGS[@]}"
mkdir -p "$RUN_DIR"
RUN_DIR="$(readlink -f "$RUN_DIR")"
case "$RUN_DIR/" in "$REPO"/Results/*) ;; "$REPO"/*) echo "RUN_DIR inside the repo must be under Results/" >&2; exit 1 ;; esac
OUT="$RUN_DIR/stdout.txt"
if [ -e "$OUT" ]; then
    [ "$RESUME" = 1 ] || { echo "Refusing to overwrite an existing run: $RUN_DIR (use --resume)" >&2; exit 1; }
    OUT="$RUN_DIR/stdout_resume_$(date +%Y%m%d_%H%M%S).txt"
fi
mkdir -p "$RUN_DIR"/{logs,usage,dbs,results,box/home,box/julia_depot}

# keys: read on the host from crunch_llm/code/.env (or ATHENA_ENV_FILE), handed to apptainer
# through a private file outside the box, never printed
ENV_FILE="${ATHENA_ENV_FILE:-$REPO/crunch_llm/code/.env}"
[ -f "$ENV_FILE" ] || { echo "API key file not found: $ENV_FILE (copy crunch_llm/code/.env_template)" >&2; exit 1; }
SECRET="$(mktemp "${TMPDIR:-/tmp}/athena_box_env.XXXXXX")"; chmod 600 "$SECRET"
trap 'rm -f "$SECRET"' EXIT
getkey() { grep -E "^$1=" "$2" 2>/dev/null | head -1 | cut -d= -f2- | tr -d "\"' \r"; }
for k in OLLAMA_API GOOGLE_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY XAI_API_KEY; do
    v="$(getkey "$k" "$ENV_FILE")"; [ -n "$v" ] && echo "$k=$v"
done > "$SECRET"
grep -q '^OLLAMA_API=' "$SECRET" || echo "warning: no OLLAMA_API in $ENV_FILE; the Ollama Cloud roles will fail" >&2

{
    echo "job=${SLURM_JOB_ID:-none} node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} gpu=${CUDA_VISIBLE_DEVICES:-none} start=$(date -Is)"
    echo "request=$REQUEST team=$TEAM args=$* box=$SIF"
    echo "image_sha256=$(cut -d' ' -f1 "$SIF.sha256" 2>/dev/null)"
    echo "git=$(git -C "$REPO" rev-parse --short HEAD) branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD) dirty=$(git -C "$REPO" status --porcelain | wc -l)"
} >> "$RUN_DIR/job_info.txt"

GPU_ENV=(); [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && GPU_ENV=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
apptainer exec --nv --containall --cleanenv \
    -B "$REPO:$REPO:ro" \
    -B /dev/null:"$REPO/crunch_llm/code/.env" \
    -B "$RUN_DIR:$RUN_DIR" \
    --home "$RUN_DIR/box/home" \
    --pwd "$REPO" \
    --env-file "$SECRET" \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env ATHENA_IN_BOX=1 \
    --env ATHENA_OLLAMA_CLOUD_ONLY=1 \
    --env ATHENA_JULIA_DEPOT="$RUN_DIR/box/julia_depot" \
    --env ATHENA_RESUME_PREFIX="bash $REPO/Documents/Scripts/athena_box_run.sh $REQUEST $TEAM $RUN_DIR" \
    --env SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-$(nproc)}" \
    "${GPU_ENV[@]}" \
    "$SIF" \
    /opt/envs/athena/bin/python -u run_athena.py "$REQUEST" \
        --team "$TEAM" \
        --results-dir "$RUN_DIR/results" \
        --log-base "$RUN_DIR/logs" \
        --usage-dir "$RUN_DIR/usage" \
        --db-copy-dir "$RUN_DIR/dbs" \
        "$@" \
    > "$OUT" 2>&1
rc=$?
echo "exit=$rc end=$(date -Is)" >> "$RUN_DIR/job_info.txt"
exit $rc
