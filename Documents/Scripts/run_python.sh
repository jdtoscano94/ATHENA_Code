#!/bin/bash

CONDA_ACTIVATION_SCRIPT="/oscar/rt/9.6/25/spack/x86_64_v3/anaconda3-2023.09-0-aqbcryind6ewgctu7wijluakv5mo3lo5/etc/profile.d/conda.sh"
CONDA_ENV_NAME="NABLA_JAX"

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