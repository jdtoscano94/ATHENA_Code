#!/bin/bash

# --- Configuration ---
# Path to the conda activation script
CONDA_ACTIVATION_SCRIPT="/gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh"
# Name of the conda environment to activate
CONDA_ENV_NAME="JAX_311"

# --- Argument Handling ---

# Check if a script path was provided as an argument
if [ -z "$1" ]; then
    echo "Error: No Python script path provided."
    echo "Usage: $0 /path/to/your/script.py"
    exit 1
fi

# Set the script to run from the first argument
PYTHON_SCRIPT="$1"
# Get the directory of the script to clean __pycache__
SCRIPT_DIR=$(dirname "$PYTHON_SCRIPT")

# --- Pre-run Checks ---

# Check if the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Check if the conda activation script exists
if [ ! -f "$CONDA_ACTIVATION_SCRIPT" ]; then
    echo "Error: Conda activation script not found at $CONDA_ACTIVATION_SCRIPT"
    exit 1
fi

# --- Execution ---

echo "Activating Conda environment: $CONDA_ENV_NAME"
# Source the conda script to make 'conda' command available
source "$CONDA_ACTIVATION_SCRIPT"
conda activate "$CONDA_ENV_NAME"

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate Conda environment '$CONDA_ENV_NAME'."
    exit 1
fi

echo "Conda environment activated."
echo "Active Python executable: $(which python)"

# Clean the __pycache__ directory relative to the script
echo "Cleaning __pycache__ from $SCRIPT_DIR"
rm -rf "$SCRIPT_DIR/__pycache__/"

echo "Starting Python script..."
echo "  Script: $PYTHON_SCRIPT"
echo "-------------------------------------"

# Run the Python script using the environment's python
# The Python function 'run_shell_script' will capture stdout and stderr.
python "$PYTHON_SCRIPT"

# Python will log the exit code and output.
echo "-------------------------------------"
echo "Python script finished."

# Deactivate the environment (good practice)
conda deactivate
echo "Conda environment deactivated."
