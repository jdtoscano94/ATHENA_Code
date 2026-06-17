#!/bin/bash

# --- Configuration ---
# Using the specific paths from your environment
# 2. Dynamically grab the path to the executable
JULIA_EXEC="/users/jdtoscan/.juliaup/bin/julia"

PROJECT_ENV="/oscar/data/gk/jdtoscan/Learning_Julia"

# Number of threads: prefer the SLURM allocation, else the cgroup-visible CPU
# count (nproc respects the cores actually granted to this job).
JULIA_THREADS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-$(nproc)}}"

# --- Safety ---
# Exit immediately if a command exits with a non-zero status
set -e

# --- Argument Handling ---
if [ -z "$1" ]; then
    echo "Error: No Julia script path provided." >&2
    echo "Usage: $0 /path/to/script.jl /path/to/stderr.log" >&2
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: No stderr log path provided." >&2
    echo "Usage: $0 /path/to/script.jl /path/to/stderr.log" >&2
    exit 1
fi

JULIA_SCRIPT="$1"
STDERR_FILE="$2"
SCRIPT_DIR=$(dirname "$JULIA_SCRIPT")

# --- Pre-run Checks ---
if [ ! -f "$JULIA_SCRIPT" ]; then
    echo "Error: Julia script not found at $JULIA_SCRIPT" >&2
    exit 1
fi

if [ ! -x "$JULIA_EXEC" ]; then
    echo "Error: Julia executable not found or not executable at $JULIA_EXEC" >&2
    exit 1
fi

if [ ! -d "$PROJECT_ENV" ]; then
    echo "Error: Project environment directory not found at $PROJECT_ENV" >&2
    exit 1
fi

# --- Change Directory ---
# CRITICAL: Switch to the directory containing the script so relative paths work
cd "$SCRIPT_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Failed to change directory to '$SCRIPT_DIR'." >&2
    exit 1
fi

# --- Execution ---
# 1. Use the configured Project Environment
# 2. Run the script
# 3. Redirect stderr (2>) to the log file provided by Python
"$JULIA_EXEC" --project="$PROJECT_ENV" --threads="$JULIA_THREADS" --color=no "$JULIA_SCRIPT" 2> "$STDERR_FILE"