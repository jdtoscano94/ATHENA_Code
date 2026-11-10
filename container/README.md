# ATHENA release container

One Apptainer image with everything ATHENA and its generated codes need. Nothing is taken from the host except the GPU driver (`--nv`) and the repository itself.

| Inside the image | Version | Used by |
|---|---|---|
| Julia | 1.11.9 | the numerical-methods codes (Trixi, OrdinaryDiffEq, Gridap, StartUpDG) |
| Julia project | `container/julia/{Project,Manifest}.toml` (frozen; Trixi 0.13.15) | same |
| `Crunch.jl` | `third_party/Crunch.jl` | local Julia package listed in the project |
| NABLA_JAX env | Python 3.12.12, `container/requirements_nabla_jax.txt` (JAX 0.8.2, CUDA 12 wheels) | the PINN and operator codes |
| NABLA `Crunch` | `third_party/NABLA` (from NABLA-SciML) | `from Crunch.Models.layers import ...` in the PINN codes |
| ATHENA env | Python 3.12.12, `container/requirements_athena.txt` | the orchestrator (`run_athena.py`) |
| OpenCode | 1.18.33 | installed in the image; not used by ATHENA or the paper's experiments |

The Julia depot inside the image is precompiled for a portable CPU target and is read-only. A writable depot sits in front of it (`$ATHENA_JULIA_DEPOT`, default `~/.athena_julia_depot`), so any package an agent adds, and any recompile, lands there and never changes the image.

## Build

```bash
mkdir -p container/logs                 # SLURM writes the build log here
sbatch container/build_release.sh        # Oscar, from the repository root
bash   container/build_release.sh        # any machine with apptainer and --fakeroot
```

The image goes to `container/images/athena_release.sif` with its `sha256`. A built image is never overwritten.

## Run

Use the launcher, from the repository root:

```bash
bash Documents/Scripts/athena_box_run.sh <request.json> <team> <RUN_DIR> [run_athena.py options]
```

It sets up what a raw `apptainer exec` does not: the read-only repository and writable run
folder, the API keys for the orchestrator only, Ollama Cloud for the utility roles, and
`ATHENA_IN_BOX=1`, which tells the bridges to run the generated code in the same box instead of
starting another container. See the main `README.md` ("Running ATHENA") for the options.

Inside the image, `$ATHENA_PY` is the orchestrator's Python and `$NABLA_PY` the generated codes' Python; `julia` is on the `PATH` with the frozen project active.
