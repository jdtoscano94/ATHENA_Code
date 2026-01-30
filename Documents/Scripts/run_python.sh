#!/bin/bash

CONDA_ACTIVATION_SCRIPT="/gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh"
CONDA_ENV_NAME="JAX_311"

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
    echo "Error: Conda activation script not found at $CONDA_ACTIVATION_SCRIPT" >&2
    exit 1
fi

# --- Environment Activation ---
# (Silently activate)
source "$CONDA_ACTIVATION_SCRIPT"
conda activate "$CONDA_ENV_NAME"

# --- Change Directory ---
cd "$SCRIPT_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Failed to change directory to '$SCRIPT_DIR'." >&2
    exit 1
fi

# --- Execution ---
rm -rf "./__pycache__/"
python -u "$PYTHON_SCRIPT" 2> "$STDERR_FILE"
conda deactivate