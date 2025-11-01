#!/bin/bash

# --- Configuration ---
JULIA_EXEC="/oscar/runtime/software/external/julia/1.11.1/bin/julia"
PROJECT_ENV="/users/jdtoscan/data/jdtoscan/Learning_Julia"

# --- Argument Handling ---

# Check if a script path was provided as an argument
if [ -z "$1" ]; then
    echo "Error: No Julia script path provided."
    echo "Usage: $0 /path/to/your/script.jl"
    exit 1
fi

# Set the script to run from the first argument
JULIA_SCRIPT="$1"

# --- Pre-run Checks ---

# Check if the Julia script exists
if [ ! -f "$JULIA_SCRIPT" ]; then
    echo "Error: Julia script not found at $JULIA_SCRIPT"
    exit 1
fi

# Check if the Julia executable exists
if [ ! -x "$JULIA_EXEC" ]; then
    echo "Error: Julia executable not found or not executable at $JULIA_EXEC"
    exit 1
fi

# Check if the project environment directory exists
if [ ! -d "$PROJECT_ENV" ]; then
    echo "Error: Project environment directory not found at $PROJECT_ENV"
elif [ ! -f "$PROJECT_ENV/Project.toml" ]; then
    echo "Warning: Project directory found, but Project.toml is missing in $PROJECT_ENV"
fi

# --- Execution ---

echo "Starting Julia script..."
echo "  Executable: $JULIA_EXEC"
echo "  Environment: $PROJECT_ENV"
echo "  Script: $JULIA_SCRIPT"
echo "-------------------------------------"

# Run the Julia script.
# The Python function 'run_shell_script' will capture stdout and stderr.
"$JULIA_EXEC" --project="$PROJECT_ENV" "$JULIA_SCRIPT"

# Python will log the exit code and output.
echo "-------------------------------------"
echo "Julia script finished."