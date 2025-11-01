#!/bin/bash

# --- Configuration ---
# Path to the conda activation script
CONDA_ACTIVATION_SCRIPT="/gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh"
# Name of the conda environment to activate
CONDA_ENV_NAME="JAX_311"

# --- Safety ---
# Exit immediately if a command exits with a non-zero status.
set -e

# --- Argument Handling ---

# $1 will be the Python script path (e.g., /path/to/script.py)
# $2 will be the stderr log path (e.g., /path/to/script.err)

if [ -z "$1" ]; then
    # Send errors to stderr (>&2) so they are captured
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
STDERR_FILE="$2" # New variable for the error log
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

echo "Activating Conda environment: $CONDA_ENV_NAME"
# Source the conda script to make 'conda' command available
source "$CONDA_ACTIVATION_SCRIPT"
conda activate "$CONDA_ENV_NAME"

echo "Conda environment activated."
echo "Active Python executable: $(which python)"

# --- CRITICAL: Change Directory ---
echo "Changing directory to $SCRIPT_DIR"
cd "$SCRIPT_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Failed to change directory to '$SCRIPT_DIR'." >&2
    exit 1
fi
echo "Current working directory: $(pwd)"

# --- Execution ---

# Clean the __pycache__ directory (now relative to the new CWD)
echo "Cleaning __pycache__ from $(pwd)"
rm -rf "./__pycache__/"

echo "Starting Python script..."
echo "  Script: $PYTHON_SCRIPT"
echo "  Errors to: $STDERR_FILE"
echo "-------------------------------------"

# --- CORE MODIFICATION ---
# Run the Python script.
# We can now just use the base filename, but $PYTHON_SCRIPT (full path)
# is safer and will also work.
# Add -u for unbuffered stdout, which allows for live streaming.
# STDOUT (standard output) will go to the calling Python function.
# STDERR (standard error) will be redirected (2>) to the error file.
python -u "$PYTHON_SCRIPT" 2> "$STDERR_FILE"

# Python will log the exit code and output.
echo "-------------------------------------"
echo "Python script finished."

# Deactivate the environment (good practice)
conda deactivate
echo "Conda environment deactivated."

