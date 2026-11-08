#!/bin/bash
# Build the ATHENA release image (container/athena_release.def) into container/images/.
# Usage (from anywhere):  sbatch container/build_release.sh      on Oscar (a CPU node, about 1-2 h)
#                         bash   container/build_release.sh      on a machine with apptainer + fakeroot
#SBATCH -J athena_release_build
#SBATCH -p batch
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o container/logs/build_%j.out
set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f "$REPO/container/athena_release.def" ] || REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
mkdir -p container/images container/logs
OUT=container/images/athena_release.sif
[ -e "$OUT" ] && { echo "$OUT exists; move it away first (never overwrite a built image)" >&2; exit 1; }

# build scratch on the node's local disk when it has room (fakeroot builds need a local filesystem)
LOCAL="${TMPDIR:-/tmp}/$USER/athena_release_build"
mkdir -p "$LOCAL"
free_gb=$(df -BG --output=avail "$LOCAL" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 60 ] || { echo "only ${free_gb}G free in $LOCAL; need 60G" >&2; exit 1; }
export APPTAINER_TMPDIR="$LOCAL/tmp" APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$LOCAL/cache}"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

echo "host=$(hostname) start=$(date -Is) git=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
S=$(date +%s)
apptainer build --fakeroot "$OUT.partial" container/athena_release.def
mv "$OUT.partial" "$OUT"
echo "rc=0 wall_s=$(( $(date +%s) - S )) end=$(date -Is)"
ls -lh "$OUT"
sha256sum "$OUT" | tee "$OUT.sha256"
rm -rf "$LOCAL"
