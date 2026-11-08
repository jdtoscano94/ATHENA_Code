#!/bin/bash
#SBATCH --partition=batch2
#SBATCH --account=batch2
#SBATCH --qos=batch2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=3-00:00:00
# One isolated ATHENA run for the model-sensitivity study, in one box (the release container):
# the orchestrator, the agents and the generated code all run inside it.
#
# Submit (one job per run; the slurm log goes into the run folder too):
#   R=Results/Sensitivity/open/RT/run1; mkdir -p $R
#   sbatch --job-name=ath_open_RT_r1 -o $R/slurm-%j.out Documents/Scripts/sensitivity_run.sbatch \
#          Conceptualization_team/ATHENA_Request_Rayleigh_Taylor_2D_Euler.json open RT 1
#
# Everything of a run lives in ONE folder (see athena_box_run.sh for the layout):
#   Results/Sensitivity/<team>/<case>/run<N>/
set -u
REQUEST="$1"; TEAM="$2"; CASE="$3"; RUN="$4"; shift 4
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO" || exit 1
exec bash "$REPO/Documents/Scripts/athena_box_run.sh" "$REQUEST" "$TEAM" \
    "$REPO/Results/Sensitivity/$TEAM/$CASE/run$RUN" "$@"
