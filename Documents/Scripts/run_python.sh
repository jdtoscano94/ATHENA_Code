#!/bin/bash
# Run one generated Python script. Usage: run_python.sh /path/to/script.py /path/to/stderr.log
#
# Inside the box (ATHENA_IN_BOX=1, set by athena_box_run.sh) it runs directly with the image's env.
# Outside a box, by default it runs inside the ATHENA release container (container/images/athena_release.sif), a box:
#   the image's NABLA_JAX env (JAX 0.8.2 CUDA 12, Flax, Optax, Crunch), the host GPU via --nv,
#   the script's folder visible and writable (--containall), and the reference data
#   (Documents/codes/Data) visible read-only. A pip install lands in the container's throwaway
#   home, never in a shared environment.
# ATHENA_EXEC=host runs with a host conda env instead: ATHENA_CONDA_SH (path to conda.sh; default
#   from `conda info --base`) and ATHENA_CONDA_ENV (default NABLA_JAX).

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ATHENA_EXEC="${ATHENA_EXEC:-container}"
ATHENA_SIF="${ATHENA_SIF:-$REPO/container/images/athena_release.sif}"
DATA_DIR="$REPO/Documents/codes/Data"
NOKEYS=(env -u OLLAMA_API -u OLLAMA_API_KEY -u GOOGLE_API_KEY -u OPENAI_API_KEY -u ANTHROPIC_API_KEY -u XAI_API_KEY)
if [ "${ATHENA_IN_BOX:-0}" = 1 ]; then
    # already inside the box (athena_box_run.sh): the image's NABLA_JAX env; the generated code
    # never sees the API keys
    set -e
    [ -n "$1" ] && [ -n "$2" ] || { echo "Usage: $0 /path/to/script.py /path/to/stderr.log" >&2; exit 1; }
    PYTHON_SCRIPT="$(readlink -f "$1")"; STDERR_FILE="$2"
    [ -f "$PYTHON_SCRIPT" ] || { echo "Error: Python script not found at $PYTHON_SCRIPT" >&2; exit 1; }
    cd "$(dirname "$PYTHON_SCRIPT")"
    rm -rf "./__pycache__/"
    "${NOKEYS[@]}" /opt/envs/nabla_jax/bin/python -u "$PYTHON_SCRIPT" 2> "$STDERR_FILE"
    exit $?
fi
if [ "$ATHENA_EXEC" = container ]; then
    set -e
    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "Usage: $0 /path/to/script.py /path/to/stderr.log" >&2
        exit 1
    fi
    PYTHON_SCRIPT="$(readlink -f "$1")"
    STDERR_FILE="$2"
    SCRIPT_DIR=$(dirname "$PYTHON_SCRIPT")
    [ -f "$PYTHON_SCRIPT" ] || { echo "Error: Python script not found at $PYTHON_SCRIPT" >&2; exit 1; }
    [ -f "$ATHENA_SIF" ] || { echo "Error: container image not found at $ATHENA_SIF" >&2; exit 1; }
    cd "$SCRIPT_DIR"
    rm -rf "./__pycache__/"
    GPU_ENV=()
    [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && GPU_ENV=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
    DATA_BIND=()
    [ -d "$DATA_DIR" ] && DATA_BIND=(-B "$DATA_DIR:$DATA_DIR:ro")
    apptainer exec --nv --containall --cleanenv \
        -B "$SCRIPT_DIR:$SCRIPT_DIR" --pwd "$SCRIPT_DIR" \
        "${DATA_BIND[@]}" \
        "${GPU_ENV[@]}" \
        "$ATHENA_SIF" \
        /opt/envs/nabla_jax/bin/python -u "$PYTHON_SCRIPT" 2> "$STDERR_FILE"
    exit $?
fi

# --- Host mode (ATHENA_EXEC=host) ---
CONDA_ENV_NAME="${ATHENA_CONDA_ENV:-NABLA_JAX}"
CONDA_ACTIVATION_SCRIPT="${ATHENA_CONDA_SH:-}"
if [ -z "$CONDA_ACTIVATION_SCRIPT" ] && command -v conda >/dev/null 2>&1; then
    CONDA_ACTIVATION_SCRIPT="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
fi

# --- Safety ---
set -e

# --- Argument Handling ---
if [ -z "$1" ]; then
    echo "Error: No Python script path provided." >&2
    echo "Usage: $0 /path/to/script.py /path/to/stderr.log" >&2
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: No stderr log path provided." >&2
    echo "Usage: $0 /path/to/script.py /path/to/stderr.log" >&2
    exit 1
fi

PYTHON_SCRIPT="$1"
STDERR_FILE="$2"
SCRIPT_DIR=$(dirname "$PYTHON_SCRIPT")

# --- Pre-run Checks ---
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT" >&2
    exit 1
fi

if [ ! -f "$CONDA_ACTIVATION_SCRIPT" ]; then
    echo "Error: conda activation script not found ('$CONDA_ACTIVATION_SCRIPT'); set ATHENA_CONDA_SH" >&2
    exit 1
fi

# --- Environment Activation ---
# (Silently activate)
source "$CONDA_ACTIVATION_SCRIPT"
if ! conda activate "$CONDA_ENV_NAME" 2>/dev/null; then
    # Fallback: activate by hand when the conda binary cannot run (e.g. a shared
    # install whose bin/conda is not executable).
    ENV_PREFIX="$HOME/.conda/envs/$CONDA_ENV_NAME"
    if [ ! -x "$ENV_PREFIX/bin/python" ]; then
        echo "Error: conda activate failed and no env at $ENV_PREFIX" >&2
        exit 1
    fi
    export CONDA_PREFIX="$ENV_PREFIX" CONDA_DEFAULT_ENV="$CONDA_ENV_NAME"
    export PATH="$ENV_PREFIX/bin:$PATH"
    set +e
    for _hook in "$ENV_PREFIX"/etc/conda/activate.d/*.sh; do
        [ -r "$_hook" ] && . "$_hook" >/dev/null 2>&1
    done
    set -e
fi

# --- Change Directory ---
cd "$SCRIPT_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Failed to change directory to '$SCRIPT_DIR'." >&2
    exit 1
fi

# --- Execution ---
rm -rf "./__pycache__/"
python -u "$PYTHON_SCRIPT" 2> "$STDERR_FILE"
conda deactivate 2>/dev/null || true