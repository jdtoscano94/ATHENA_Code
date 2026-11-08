# ATHENA: Agentic Team for Hierarchical Evolutionary Numerical Algorithms

<img src="Agent_Framework.png" width="50%" alt="ATHENA Agent Framework">

Progress in computational science depends on complex numerical workflows that must faithfully encode physical laws, yet translating conceptual insight into reliable code remains a major bottleneck. Although large language models can generate isolated code fragments, they lack the structured reasoning required to design, verify, and iteratively refine complete scientific pipelines. Here we introduce ATHENA, an agentic framework explicitly designed to emulate scientific research modeled as a knowledge-driven contextual bandit process. Its core loop separates conceptual policy from numerical realization through expert-derived conceptual scaffolding, enabling principled diagnosis, reformulation, and repair of computational strategies. Across scientific computing and scientific machine learning tasks, ATHENA autonomously discovers exact analytical solutions, constructs stable numerical solvers, diagnoses ill-posed formulations, and orchestrates hybrid symbolic–numeric workflows. Quantitatively, ATHENA exceeds expert-designed workflows, achieving superhuman accuracy and robustness on canonical benchmarks. By reframing computation as an object of agentic reasoning, the framework enables autonomous orchestration of heterogeneous algorithms across scientific domains.


# Use a coding agent (highly encouraged)

**We highly encourage running ATHENA through an external agentic platform, such as
[Claude Code](https://claude.com/claude-code) or [Codex](https://openai.com/codex), to ensure a
proper installation.** The repository ships a `CLAUDE.md` (read by Claude Code; `AGENTS.md` is the
same file for Codex and other agents) that takes the agent through the whole workflow:

1. **Install.** It checks Apptainer, the GPU and the cluster, gets the container image, checks that
   your API keys are set (you fill in `crunch_llm/code/.env` yourself; the agent never reads the
   values), and runs a test script in each language inside the container.
2. **Formalize.** It acts as ATHENA's Conceptualization Team, following the same Coordinator and
   Gatekeeper instructions as `Formalize.py`: it asks about your problem one question at a time,
   proposes the families (A, B, C, or a hybrid), and, once you approve the specification, writes
   the request JSON.
3. **Run.** It launches the request in the container and monitors it.
4. **Report.** It tells you how the run ended, where the best trial and its figures are, and what
   to do if the run stopped and asked for help.

Open the agent in the repository root and ask, for example, *"Set up ATHENA on this machine"* or
*"I want to solve the 2D Helmholtz equation with ATHENA"*. The manual steps below are what the
agent does; you can also follow them yourself.

> **Scope:** the agent drives ATHENA; it does not replace it. Choosing the method and writing,
> debugging and judging the code remain the work of ATHENA's own agents. Results are reproduced
> from the request JSONs, `run_athena.py`, the container and the logged transcripts, not from
> decisions the coding agent makes mid-run.


# Installation

ATHENA's agents write code in three environments: Python with JAX for the PINN and operator
codes (families **B** and **C**), Julia for the numerical solvers (family **A**), and Python for
the orchestrator itself. All three ship in **one Apptainer container**, and the container is the
default execution environment: every generated code runs inside it, whether you launch a run with
`athena_box_run.sh` or with `run_athena.py`. The container fixes every package version, hides
your API keys from the generated code, lets that code write only to its run folder, and takes
only the GPU driver from the host, so the same environment runs on any machine with Apptainer.

| Inside the container | Version |
|---|---|
| Julia + frozen project (`container/julia/`) | Julia 1.11.9, Trixi.jl 0.13.15, OrdinaryDiffEq, Gridap, StartUpDG |
| Python for the generated PINN / operator codes | Python 3.12.12, JAX 0.8.2 (CUDA 12), Flax, Optax, NABLA (`third_party/NABLA`) |
| Python for the orchestrator | Python 3.12.12, LangChain 1.2 + LangGraph 1.0, ChromaDB 1.5 |
| OpenCode (single-agent baselines only) | 1.18.33 |

## 1. Clone the repository

```bash
git clone https://github.com/jdtoscano94/ATHENA_Agentic_Team_for_Hierarchical_Evolutionary_Numerical_Algorithms.git ATHENA
cd ATHENA
```

## 2. Get the container image

Build it from the definition in `container/` (about 1 to 2 hours; needs Apptainer with
`--fakeroot` and about 60 GB of local scratch):

```bash
sbatch container/build_release.sh     # on a SLURM cluster, from the repository root
bash   container/build_release.sh     # on any machine with apptainer
```

The image is written to `container/images/athena_release.sif` together with its `sha256`
(a built image is never overwritten). The image used for the paper has sha256
`7dc4537f9c14bae42d4ae9f582bcaa92dd45a52e126c47a20746436030b030e7`
[download link to be added]. See `container/README.md` for details.

## 3. Configure the API keys

```bash
cp crunch_llm/code/.env_template crunch_llm/code/.env
```

Fill in the keys you use. `.env` is gitignored and stays on your machine; the launch scripts read
it on the host and pass the keys only to the orchestrator, never to the generated code.

| Key | Needed for |
|---|---|
| `OLLAMA_API` | the open-weight roles (e.g. `gpt-oss:120b`) and the open team, served by Ollama Cloud |
| `GOOGLE_API_KEY` | Gemini agents and the retrieval embeddings (`gemini-embedding-001`) |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY` | agents on those providers |
| `CLAUDE_CODE_OAUTH_TOKEN` | only the Claude Code baseline (`bare_agent_run.sh`) |

## 4. Check the setup

Each bridge runs one generated script inside the container. A one-line test in each language
should print its output and exit cleanly:

```bash
mkdir -p Results/Tests/bridge && cd Results/Tests/bridge
echo 'import jax; print("python OK", jax.devices())' > toy.py
echo 'println("julia OK")' > toy.jl
bash ../../../Documents/Scripts/run_python.sh toy.py toy.py.err
bash ../../../Documents/Scripts/run_julia.sh  toy.jl toy.jl.err
```

On a GPU node the first line prints `python OK [CudaDevice(id=0)]`.

# Running ATHENA

There are two ways to launch a run. Both execute the generated code inside the container.

**A. One run in one box: `athena_box_run.sh` (recommended).** The orchestrator, the agents and
the generated code all run inside the container. The repository is visible read-only, and the
run folder is the only writable place, so everything a run produces lives in one folder:

```bash
bash Documents/Scripts/athena_box_run.sh \
    Conceptualization_team/ATHENA_Request_Helmholtz_PINN.json original Results/MyRuns/Helmholtz/run1
```

```
Results/MyRuns/Helmholtz/run1/
  job_info.txt   node, GPU, start/end, git commit, image sha256, exit code
  stdout.txt     the full console output
  logs/          agent transcripts
  usage/         token and time log (one record per LLM call and code execution)
  dbs/           private copies of the vector DBs this run used
  results/       generated code, outputs and Experiment_Report.md per iteration
  box/           the box's home and writable Julia depot
```

A run folder inside the repository must be under `Results/`, and an existing run is never
overwritten (continue it with `--resume`). On a SLURM cluster, submit the same command as a batch
job; `Documents/Scripts/sensitivity_run.sbatch` is a worked example (adapt its `#SBATCH` lines to
your cluster).

**B. `run_athena.py` directly.** The orchestrator runs in a Python environment of your own (see
[Running the orchestrator outside the container](#running-the-orchestrator-outside-the-container)),
and each generated code still runs inside the container through the bridges:

```bash
python run_athena.py Conceptualization_team/ATHENA_Request_Helmholtz_PINN.json
```

Outside the box, the `gpt-oss:120b` roles of the original team call a local Ollama server unless
you set `ATHENA_OLLAMA_CLOUD_ONLY=1`, which serves them through Ollama Cloud instead (the box
sets this for you).

## Options of `run_athena.py`

`athena_box_run.sh` passes any extra option through to `run_athena.py`.

| Option | Effect |
|---|---|
| `--team T` | agent team: `original` (the published team, default, families A, B and C), `open` (open-weight models on Ollama Cloud, families A and B) or `closed` (original team with its utility roles on Gemini 3.1 Pro, families A and B) |
| `--runs N` | repeat the whole pipeline `N` times (regret / variance analysis) |
| `--max-iters N` | cap the HENA loop at `N` iterations per stage (quick test) |
| `--stage N` | run only pipeline stage `N` |
| `--resume DIR` | continue from the newest `Experiment_Report.md` at or under `DIR` (needs `--stage` for multi-stage requests) |
| `--results-dir`, `--log-base`, `--usage-dir`, `--db-copy-dir` | where results, transcripts and usage logs go, and a folder for private copies of the vector DBs (set by `athena_box_run.sh`) |

## When a run stops

A run ends in one of four ways:

* **Approved.** The Advisor approves a trial.
* **Score gate** (families A and B). Once a trial scores at least 90, the run continues until
  five iterations have run counting from that trial (crashed iterations included), then stops
  and returns the best-scoring trial.
* **Iteration limit.** `max_iterations_all` (50, 25 and 100 for families A, B and C) is reached.
* **Help needed.** The Debugger classifies an error as unrecoverable: a missing package, a
  broken environment or an invalid input. The agents are told never to modify the environment,
  so the run stops, writes `UNRECOVERABLE_ERROR.md` in the project folder with the diagnosis,
  and prints the exact command that resumes it from the last checkpoint once the problem is
  fixed (for a box run, the `athena_box_run.sh ... --resume ...` command).

## Running the orchestrator outside the container

Only needed for option B. Create a Python 3.12 environment with the orchestrator's packages:

```bash
conda create -n athena_env python=3.12 && conda activate athena_env
pip install -r requirements.txt
pip install -e . --no-deps
```

> **Note:** the agent layer (`crunch_llm/code/agents.py`) is built on the
> `langchain.agents.create_agent` API, which requires `langchain>=1.0` and `langgraph`.

## Running the generated code on the host (advanced)

`ATHENA_EXEC=host` makes the bridges run the generated code on the host instead of in the
container. The host environments must then match the container's:

* **Python / JAX** (`run_python.sh`): a conda env with
  `pip install -r container/requirements_nabla_jax.txt && pip install --no-deps third_party/NABLA`.
  Set `ATHENA_CONDA_ENV` (default `NABLA_JAX`) and, if `conda` is not on your `PATH`,
  `ATHENA_CONDA_SH` (the path to `conda.sh`).
* **Julia** (`run_julia.sh`): Julia 1.11.9 and the frozen project,
  `julia --project=container/julia -e 'using Pkg; Pkg.instantiate()'`.
  Set `ATHENA_JULIA_ENV=$PWD/container/julia` and, if `julia` is not on your `PATH`,
  `ATHENA_JULIA_EXEC`.

In host mode the generated code shares your environment, so the isolation of the container
(read-only packages, no access to the keys or the repository) no longer applies.

# Configuring the Models

Every agent in ATHENA is provider-agnostic: it is created with a `model` name and a
`model_provider` string, and the framework dispatches to the right SDK. You change which
LLM an agent uses simply by editing those two arguments.

**Supported providers** (the `model_provider` value): `google`, `openai`, `anthropic`
(alias `claude`), `xai` (alias `grok`), `ollama_cloud` (Ollama Cloud) and `ollama` (a local
Ollama server, or Ollama Cloud when `ATHENA_OLLAMA_CLOUD_ONLY=1`). Each provider reads its key
from `crunch_llm/code/.env` (`GOOGLE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`XAI_API_KEY`, `OLLAMA_API`).

**Changing an agent's model.** Anywhere an agent is constructed (in a notebook or script),
set the two arguments:

```python
agent = StructuredAgent(
    model='gemini-3-flash-preview',   # any model id valid for the provider
    model_provider='google',          # google | openai | anthropic | grok | ollama_cloud | ollama
    system_prompt=...,
)
```

To switch the same agent to Grok or OpenAI, change only those two fields:

```python
# Grok
StructuredAgent(model='grok-4-1-fast-reasoning', model_provider='grok',  ...)
# OpenAI
StructuredAgent(model='gpt-5.1',                 model_provider='openai', ...)
```

**Two kinds of models to keep in mind.**
1. **Generation models** — the LLMs the agents reason with (analyzer, strategist, advisor, …).
2. **Embedding models** — used to build and query the vector database. These are set with
   `provider` / `model_name` in `initialize_local_data(...)` and with
   `embeddings_provider` / `embeddings_model` in the retriever agents.

> **Important — embeddings must match.** The embedding model used to **ingest** a collection
> must be the same one used to **query** it (different models produce different,
> incompatible vector dimensions). If you change the embedding model, re-ingest the
> affected databases.


The teams shipped in `crunch_llm/code/HENA.py` (the `build_agents_*` factories, selected with
`--team`) are:

| Team | Generation models | Embeddings |
|---|---|---|
| `original` (published) | Gemini, GPT, Claude and Grok in the reasoning roles; `gpt-oss:120b` in the utility roles (retriever, parsers, patcher, naming) | `models/gemini-embedding-001` (`google`) |
| `open` | open-weight models on Ollama Cloud (`OPEN_TEAM` in `HENA.py`) | same |
| `closed` | the original team with its `gpt-oss:120b` roles on Gemini 3.1 Pro | same |

The `gpt-oss:120b` roles use `model_provider='ollama'`. Inside the box they are served by
Ollama Cloud (`ATHENA_OLLAMA_CLOUD_ONLY=1`), so no local model server or GPU is needed for the
agents; outside the box, set the same variable or start a local Ollama server
(`ollama serve`, `ollama pull gpt-oss:120b`).


# Tutorial

The workflow has three stages: **(1) Ingestion** builds the reference-code knowledge bases,
**(2) Conceptualization** formalizes a problem, and **(3) Evaluation** runs the HENA loop.

## Step 1 — Ingestion

Ingestion reads the curated reference code under `Documents/codes/` and indexes it into
local [Chroma](https://www.trychroma.com/) vector databases. At run time the agents retrieve
from these databases to ground their solutions in working examples.

> **Verify every reference file runs on _your_ stack before ingesting it.**
> The agents ground each solution in the retrieved templates, so a template that
> is broken on your machine propagates that breakage into every generated
> solution — and the failure surfaces deep in the HENA loop (or as a silently
> wrong plot), not at ingestion time.
>
> This matters **especially for Julia**, where package APIs drift across versions.
> Run each `Documents/codes/Numerical_Methods/Scripts/*.jl` through your
> `Documents/Scripts/run_julia.sh` bridge — with **your** Julia version, **your**
> `--project` environment, and the **thread count you will actually use** — and
> confirm it exits cleanly *and* produces a correct `summary_all.png` before
> ingesting. (Trixi.jl is a common offender: kernels can throw
> `UndefRefError`/`TaskFailedException` under `--threads > 1`, and plotting calls
> change between releases.) Apply the same check to the Python/JAX templates in
> your `NABLA-SciML` environment. Re-run `--clear` ingestion after fixing a
> template so the corrected version replaces the stale one.

### What gets generated

A single script, `Ingestion/ingest.py`, builds **four knowledge bases** — one per `--class`:

| `--class` | Source files | DB path | Chroma collection |
|---|---|---|---|
| `numerical_methods` | `Documents/codes/Numerical_Methods/Scripts/` (Julia/Python solvers) | `Documents/database_NM` | `Numerical_Methods_opt` |
| `pinns` | `Documents/codes/SciML/PINNs/` | `Documents/database_PIML` | `PINNs_Codes` |
| `deeponet` | `Documents/codes/SciML/Modules/layers.py` (network layers) | `Documents/database_Modules_ollama` | `Layers` |
| `operators` | `Documents/codes/SciML/Operator_Learning/DeepONets/Files/` | `Documents/database_Operators` | `Operators` |

For each source file the script: runs an **analyzer LLM** to extract structured metadata
(method, equation, dimensionality, optimizer, BCs, …), splits the code into cells/chunks,
prepends an analysis header, embeds each chunk, and writes it to the collection. A per-class
report is saved to `Ingestion/ingest_report_<class>.json`.

### How to run it

From the repository root, with `athena_env` active and your `GOOGLE_API_KEY` set:

```bash
# Build each knowledge base
python Ingestion/ingest.py --class numerical_methods
python Ingestion/ingest.py --class pinns
python Ingestion/ingest.py --class deeponet
python Ingestion/ingest.py --class operators
```

This uses the cloud-API defaults (analyzer `gemini-3-flash-preview`, embeddings
`models/gemini-embedding-001`) — no Ollama needed.

**Useful flags:**

```bash
python Ingestion/ingest.py --class pinns \
    --model gemini-3-flash-preview \      # analyzer LLM
    --provider google \                   # analyzer provider
    --embed-provider google \             # embeddings provider (google | openai | ollama)
    --embed-model models/gemini-embedding-001 \
    --path Documents/codes/SciML/PINNs \  # override the source dir/file
    --db-path Documents/database_PIML \   # override the DB location
    --collection PINNs_Codes \            # override the collection name
    --clear                               # wipe the collection before ingesting
```

> **Re-ingesting:** a second run **appends** to the collection. Pass `--clear` to rebuild
> cleanly, or `--clear-only` to just empty a collection. If you change the **embedding
> model**, delete the DB directory first (vector dimension is fixed at creation).

**Verifying:** each run prints `✅ Ingested: <file> (N chunks)`. You can also inspect a
collection directly:

```python
import chromadb
c = chromadb.PersistentClient(path="Documents/database_NM").get_collection("Numerical_Methods_opt")
print(c.count(), "chunks")
```

### Adding your own knowledge

To index additional code (new PINN architectures, solvers, operator variants), drop the
files into the relevant `Documents/codes/...` folder and re-run the matching `--class`
(use `--path` to point at a different directory). For best retrieval, organize code into
**distinct, well-commented cells** with clear docstrings — the analyzer and the cell
splitter rely on that structure.

## Step 2 — Setting up New Problems (Running the Conceptualization Team)

**With a coding agent (recommended),** just describe your problem: following `CLAUDE.md`, the
agent plays the Coordinator and the Gatekeeper below with the same instructions
(`crunch_llm/code/teams_conceptualization.py`), writes the request JSON once you approve the
specification, and records the conversation in `Conceptualization_team/Formalization_logs/`.

**Without one,** run the formalization script, which plays the same two roles with its own models:

```bash
cd Conceptualization_team && python Formalize.py
```

This script initiates a conversational agent that will guide you through defining your problem statement.
* **Process:** A *Coordinator* agent interviews you about the physics, constraints, and objectives, proposes a hybrid strategy (e.g. *infer with a PINN, then verify numerically*), and writes a formal specification. A *Gatekeeper* agent then routes that specification into an ordered execution pipeline.
* **Output:** A **JSON file** `ATHENA_Request_<problem_name>.json`, containing the standardized `problem_name`, an ordered `pipeline` of stages (each tagged `A` = SciC, `B` = PINN, `C` = Operator), a `description`, and the full formalized spec under `user_request`.
* **Transcript:** The whole conversation is mirrored to `Conceptualization_team/Formalization_logs/formalization_<timestamp>.txt` for auditing.
* **Usage:** This JSON is formatted and ready to be fed directly into `run_athena.py` (Step 3, autonomous mode).

**Non-interactive (batch) mode.** If you already have a complete problem description in a text file, you can skip the chat and generate the JSON in one shot:

```bash
cd Conceptualization_team && python Formalize.py --request-file my_problem.txt
```

### (Optional) Skipping the Conceptualization Team

The Conceptualization Team is a **convenience layer** — its only job is to turn a loosely-worded idea into a complete, well-posed specification. **It is entirely optional.** If you already know your problem precisely, you can bypass it and drive the evaluation directly: hand-author a request JSON (copy any `Conceptualization_team/ATHENA_Request_*.json` as a template) with your `pipeline` and a self-contained `user_request`, then run it with `python run_athena.py <your.json>`. The `user_request` string is fed verbatim to the HENA loop as the problem spec.

For the loop to succeed without the Coordinator's help, just make sure your `user_question` is **self-contained**. Include:

* **Governing equation(s)** — the PDE/ODE in explicit form.
* **Domain** — spatial (and temporal, if transient) extent.
* **Boundary conditions** and, for time-dependent problems, **initial conditions**.
* **Physics parameters** — all constants (e.g. `Re`, `ν`, `ϵ`), or which ones are *unknown* for an inverse problem.
* **Data** — path and format of any reference/observation data, plus how to sample it (e.g. number of sensors, noise level).
* **Targets, validation, and outputs** — what to infer, which error metrics to report, and the expected figure/file names.

In short: the Conceptualization Team *produces* this specification, but if you can write it yourself, you can feed the equation directly. The interactive step exists to help when the problem is still vague — not as a required gate.

## Step 3 — Evaluation

There are **two ways to run a problem**, suited to different situations:

* **Autonomous mode — `run_athena.py`.** Reads a request JSON and runs every
  pipeline stage end-to-end through the HENA loop, hands-off. This is the path for a
  **new problem**: first let the **Conceptualization Team** frame it properly
  (`Formalize.py`, Step 2) into a request JSON — highly encouraged, since it scopes
  the physics, picks the family (A/B/C), and writes a clean request — then run it.

  ```bash
  python run_athena.py Conceptualization_team/ATHENA_Request_Rayleigh_Taylor_2D_Euler.json
  ```

* **Interactive mode — the example notebooks.** When you **already understand the
  problem** and don't need the Conceptualization Team to contextualize it, it is
  easier to run the per-problem notebook in `Examples_<family>/` directly. Each runs
  **only the HENA loop**, seeded with your `user_question`, and is the preferred path
  for human-in-the-loop work and checkpoint-driven runs. See
  [**Interactive mode (the example notebooks)**](#interactive-mode-running-the-hena-loop-directly-the-example-notebooks)
  below.

> **Why some problems ship in *both* modes (e.g. `Viscous_Burgers_DeepONet`).**
> For this DeepONet case we provide two demonstrations of the *same* problem, one per mode:
> * **Manual / interactive (HENA only).** The example notebook in `Examples_DeepONet/` drives
>   `run_hena` directly, seeded with a hand-written `user_question`. It shows the inner HENA
>   loop in isolation — strategy → retrieval → implementation → execution → advising — for a
>   problem you already understand.
> * **Autonomous (full end-to-end).** `run_athena.py` runs the *entire* pipeline hands-off,
>   **including the Conceptualization stage**: the request JSON frames the physics, picks the
>   family, and feeds the HENA loop, which then climbs the escalation ladder and **self-terminates**
>   on `APPROVED`. This is the complete agentic story — conceptualization through final model.
>
> Both reach **comparable accuracy** (the operator solves the full 2D space-time field
> `u₀(x) → u(t,x)` either way); the difference is scope of demonstration, not result. The
> autonomous run is the one to read when you want to see ATHENA's full reasoning trace, from
> problem framing to the terminating decision.

### Reference data

Most examples are self-contained (they generate their own reference). The data-driven examples
load a file from `Documents/codes/Data/`, which the bridges mount read-only inside the container:

| Example | File | Size | Where it comes from |
|---|---|---|---|
| `Allen_Cahn_PINN` | `allen_cahn.mat` | 1.8 MB | included in the repo |
| `1D_Viscous_Burgers_PINN_BareProblem` | `burgers.mat` | 15 MB | included in the repo |
| `Inviscid_Burgers_PINN` | `inviscid_burgers_sinpix.npz` | 0.4 MB | included in the repo |
| `Inverse_2D_NavierStokes_…_ReInference` | `ldc_Re500.npz` | 3.0 MB | included in the repo |
| `Viscous_Burgers_DeepONet` | `Burger_1000_5000.mat` | 374 MB | **download from [Google Drive](https://drive.google.com/file/d/1Tr6a22SiMpgUlzbIHiBFAt8DRGXQ30E6/view)** into `Documents/codes/Data/` (too large for git) |

The request JSONs name these files as `{REPO}/Documents/codes/Data/<file>`; `run_athena.py`
replaces `{REPO}` with the repository root, so the requests work wherever the repo is cloned.

### How `run_athena.py` works

`run_athena.py` reads a request JSON and runs **every stage in its `pipeline`,
in order**, dispatching each stage to the HENA loop for its family
(`A`=SciC/Julia, `B`=PINN/JAX, `C`=Operator). You do not edit any code to run a
case — you point the script at the JSON:

```bash
python run_athena.py Conceptualization_team/ATHENA_Request_<name>.json
```

See [Options of `run_athena.py`](#options-of-run_athenapy) for the flags.

Each run writes:
* generated code, data, and per-iteration reports → `Results/<FAMILY>_codes/<ProjectName>/<timestamp>/`
* the full multi-agent transcript → `Examples_<family>/<ProblemName>_log.txt`, mirroring the
  notebook convention: **A → `Examples_NM`**, **B → `Examples_PINNs`**, **C → `Examples_DeepONet`**
* a token and time usage log → `run_logs/usage/`

A box run (`athena_box_run.sh`) writes all of these into its own run folder instead.

> **Family-A analytical cases.** Family A covers both numerical-validation problems
> (`Examples_NM`) and analytical-exact ones (`Examples_analytical_exact`), so its logs default to
> `Examples_NM`. If the problem is analytical-exact (e.g. `Viscous_Burgers_1D_Analytical`,
> `Inviscid_Burgers_2D`, `BlackScholes_1D`, `Wave_Equation_3D_Spherical`), move the
> resulting `<ProblemName>_log.txt` into `Examples_analytical_exact/` afterwards.

(Passing an explicit `log_base=` to `run_hena` restores the old
`crunch_llm/code/HENA_logs/<FAMILY>/hena_<family>_<name>_<timestamp>.txt` layout instead.)

`APPROVED` from the advisor ends a stage; the project directory and log path are printed at the end.

### Single-stage cases (most problems) — one command

For every single-stage case, just run its JSON. Family `A` runs Julia, `B`/`C` run Python/JAX.

| Case (run `python run_athena.py Conceptualization_team/ATHENA_Request_<file>.json`) | Family | Extra data |
|---|---|---|
| `Rayleigh_Taylor_2D_Euler` | A | — |
| `Rayleigh_Taylor_2D_NavierStokes` | A | — |
| `Kelvin_Helmholtz_2D_Euler` | A | — |
| `Inviscid_Burgers_2D` | A | — |
| `Viscous_Burgers_1D_Analytical` | A | — |
| `Viscous_Burgers_1D_Numerical` | A | — |
| `Anisotropic_Diffusion_2D` | A | — |
| `BlackScholes_1D` | A | — |
| `Cahn_Hilliard` | A | — |
| `Wave_Equation_3D_Spherical` | A | — |
| `Helmholtz_PINN` | B | — |
| `Poisson_PINN` | B | — |
| `KdV_PINN` | B | — |
| `Inviscid_Burgers_PINN` | B | — |
| `Allen_Cahn_PINN` | B | `allen_cahn.mat` (in repo) |
| `Viscous_Burgers_DeepONet` | C | `Burger_1000_5000.mat` (download — see *Reference data*) |

Example:

```bash
python run_athena.py Conceptualization_team/ATHENA_Request_Helmholtz_PINN.json
```

### Multi-stage cases (when there are many steps)

Two cases chain families. `run_athena.py` will execute their stages in order from
a single command, **but a later stage that consumes an earlier stage's output
needs that output's path** — and the JSON ships with the path from the paper's
original run, which won't match your fresh timestamped `Results/` folder. So for
multi-stage cases run **stage by stage** and wire the path through:

**`Inverse_2D_NavierStokes_LidDrivenCavity_ReInference` — Stage 1 (B, PINN) → Stage 2 (A, numerical)**

This is the inverse Navier–Stokes case: Stage 1 trains an inverse PINN on the
lid-driven-cavity data (`ldc_Re500.npz`, in repo) to infer the Reynolds number
and the hidden v-field; Stage 2 runs a high-fidelity forward Julia solve at the
inferred `Re` and validates against the same ground truth.

Run each stage in its own run folder under `Results/`: the generated code of a later stage sees
earlier runs under `Results/` read-only (in the box and through the bridges alike). The example
uses `athena_box_run.sh`; with `run_athena.py` the same steps use `--stage 1` and `--stage 2`.

```bash
REQ=Conceptualization_team/ATHENA_Request_Inverse_2D_NavierStokes_LidDrivenCavity_ReInference.json

# 1) Run Stage 1 (the inverse PINN). Note the printed project_dir.
bash Documents/Scripts/athena_box_run.sh $REQ original Results/MyRuns/LDC/stage1 --stage 1
#    -> produces Results/MyRuns/LDC/stage1/results/<Project>/<timestamp>/results.npz

# 2) Edit the JSON: in the "### STAGE 2" section, set the Stage-1 output path
#    ("path:" under "The data was saved as:") to that results.npz, written as
#    {REPO}/Results/MyRuns/LDC/stage1/results/<Project>/<timestamp>/results.npz

# 3) Run Stage 2 (the forward Julia verification at the inferred Re).
bash Documents/Scripts/athena_box_run.sh $REQ original Results/MyRuns/LDC/stage2 --stage 2
```

> Running the JSON without `--stage` executes Stage 1 then Stage 2 back-to-back,
> but Stage 2 will read whatever Stage-1 output path is currently written in the
> JSON — so do one full run first (or pre-set the path) before relying on the
> straight-through form.

**`1D_Viscous_Burgers_PINN` — Stage 1 (A) → Stage 2 (B) → Stage 3 (B)**

Generate a high-fidelity numerical reference (A), then train a PINN against it
(B), then evaluate (B). This spec feeds the full description to each stage, so one box run
without `--stage` runs all three; if you want the PINN stages to read the exact Stage-1
reference file, run `--stage 1` first and point the PINN stages at its output, as above.

### Interactive mode: running the HENA loop directly (the example notebooks)

Every example also ships as a self-contained notebook in its `Examples_<family>/`
folder (e.g. `Examples_PINNs/Poisson.ipynb`, `Examples_NM/Cahn-Hilliard_Equation.ipynb`).
Unlike `run_athena.py`, a notebook runs **only the HENA loop** — it skips the
Conceptualization Team and is seeded directly with a `user_question`. The whole
notebook is just a handful of cells: imports → paths → agent setup →
`user_question` → run the loop → (optional) refinement.

**When to use this instead of `run_athena.py`.** For a **new problem**, prefer the
autonomous path and let the Conceptualization Team frame it first — this is highly
encouraged, as it scopes the physics and routes the problem properly. But when you
**already understand the problem** and don't need it contextualized, editing a
notebook's `user_question` and running the loop directly is faster and gives a
cell-by-cell view of the run.

**Human-in-the-loop demos.** A few cases are kept specifically because they benefit
from a human in the loop, and the notebooks make that hand-off explicit:

* **Multi-step orchestration (PINN → numerical) — two notebooks.** The inverse
  lid-driven-cavity case is split across a PINN notebook and a numerical notebook,
  in the order the agents themselves prescribe:
  1. `Examples_PINNs/Lid_driven_500_inverse.ipynb` — the inverse PINN infers the
     Reynolds number (`Re ≈ 525`) and the hidden velocity field from sparse, noisy
     data, and saves `results.npz`.
  2. `Examples_NM/Validation_NS.ipynb` — a high-fidelity numerical solve at the
     inferred `Re` verifies the physics against the true `Re = 500`. You carry
     Step 1's output forward by setting `PINN_parameter` (the inferred `Re`) and
     `path_data` to Step 1's `results.npz`.

* **Failure + a one-sentence human hint.** When the autonomous run stalls, a single
  sentence in the request rescues it (the agents interpret and integrate the hint):
  * `Examples_analytical_exact/VB1D.ipynb` (Viscous Burgers) — the agents find the
    Cole–Hopf transformation but the integral fails under standard quadrature; the
    request carries the hint *"use Gauss–Hermite quadrature,"* which resolves it to
    machine precision.
  * `Examples_PINNs/Inviscid_burgers_Human_Wavelet.ipynb` (Inviscid Burgers) —
    periodic Fourier embeddings oscillate near the shock; a hint to use periodic
    wavelet embeddings with a refined viscosity schedule extends shock capturing to
    lower viscosity.

**Easy checkpointing.** The notebooks expose checkpointing as a single editable
cell — set `checkpoint_project_dir` to a prior `Results/<run>/` folder and load it
with `load_checkpoint_data(checkpoint_project_dir, load_code_only=...)`:

* `load_code_only=True` — restart from a previously **successful script** only
  (e.g. `Lid_driven_500_inverse.ipynb`).
* `load_code_only=False` — restore the **full prior state**: strategy, history, and
  error report (e.g. `Inviscid_burgers_Human_Wavelet.ipynb`, resuming a human-guided
  run).

Leave `checkpoint_project_dir=''` to start fresh.

**Provenance.** Most of the paper's results were produced in this interactive
notebook mode — including the **ablation studies** (`Examples_*/Ablation_Study/`)
and the **regret analyses** (`Examples_NM/Regret_analysis/`,
`Examples_analytical_exact/Regret_analysis/`, `Examples_PINNs/Regret_Analysis_HM/`).
Each notebook pins the **exact LLM versions used for that result** in its agent-setup
cell (e.g. `gemini-3-pro-preview`, `gpt-5.1`, `claude-sonnet-4-5`,
`grok-4-1-fast-reasoning`, with `gpt-oss:120b` and Google `text-embedding-004` for
retrieval), so each figure can be traced to the specific models that generated it.

> The notebooks ship with the authors' absolute paths (the `my_database_path…`,
> `base_results_dir`, and `checkpoint_project_dir` cells point at
> `/users/jdtoscan/.../Crunch_LLM/...`). Edit those to your own vector-DB and
> `Results/` locations before running.

### Resuming from a checkpoint

Every scored iteration writes `Experiment_Report.md` (strategy, Advisor outcome and the full
trial history) next to its code, and any run can continue from it:

```bash
python run_athena.py <request.json> --resume Results/<FAMILY>_codes/<Project>/<timestamp>
bash Documents/Scripts/athena_box_run.sh <request.json> <team> <RUN_DIR> --resume <RUN_DIR>/results
```

The loop picks up at the next iteration with the earlier trials in its history. In Python,
`run_hena` takes the same checkpoint:

```python
from crunch_llm.code.HENA import run_hena

# Continue a prior run (restore strategy, history, error report):
run_hena("A", user_question, checkpoint_project_dir="Results/NM_codes/<Project>/<timestamp>",
         load_code_only=False)

# Or start fresh from a previously-successful script only:
run_hena("A", user_question, checkpoint_project_dir="Results/NM_codes/<Project>/<timestamp>",
         load_code_only=True)
```

### Reproducing the paper figures

The figures in the paper are assembled by the four notebooks in `Plots/`
(`Analytical.ipynb`, `NM.ipynb`, `PINNs.ipynb`, `Regret_analysis.ipynb`). They
read precomputed result arrays from the per-run folders under
`Results/<FAMILY>_codes/<Project>/<timestamp>/` using **repo-relative paths**
(`../Results/...`), so they work wherever the repo is cloned.

The data each notebook reads is committed under `Results/`, so the figures
reproduce directly from a clone — just run the notebooks. For the large
time-resolved solves (the compressible Euler / Navier–Stokes / Kelvin–Helmholtz
runs, and the 2-D diffusion field) only the time slices the figures actually
display are stored, which keeps each file small; the colorbars are computed from
those same slices. The DeepONet plot file keeps the test set and training
diagnostics it plots, dropping the unused training trajectories.

The generated script (`.jl`/`.py`) that produced each run folder is also tracked,
so you can regenerate any result from scratch instead of using the stored data:

```bash
# Julia (family A): writes its .npz back into the same folder (uses @__DIR__)
bash Documents/Scripts/run_julia.sh  Results/NM_codes/<Project>/<timestamp>/<script>.jl  <script>.jl.err

# Python / PINN (families B, C): run FROM the folder so np.savez lands there
cd Results/PINN_codes/<Project>/<timestamp> && \
  bash <repo>/Documents/Scripts/run_python.sh <script>.py <script>.py.err
```

Julia solves are deterministic; the PINN/operator scripts carry a fixed seed
(`--SEED 9999`), so reruns reproduce the figures up to library/floating-point
nondeterminism.

### A Note on GPU Requirements

The generated PINN and operator codes need an NVIDIA GPU (the container uses the host driver
through `--nv`); the Julia solvers run on CPUs. The agents themselves need no local GPU: every
model, including `gpt-oss:120b`, is reached through a cloud API (Ollama Cloud inside the box).


### A Note on Auditing the Agent Interaction
Auditing the full, multi-turn interaction between the agents (Strategist, Advisor,
Critic, etc.) matters as much as the final solution. Every run mirrors its entire
multi-agent transcript to `Examples_<family>/<ProblemName>_log.txt` (A → `Examples_NM`,
B → `Examples_PINNs`, C → `Examples_DeepONet`), so the complete reasoning trace is preserved as a
plain-text record kept right next to each example notebook — the same place the notebooks
themselves store their captured transcripts.

### Citation

If you use ATHENA in your research or find this framework helpful, please cite our preprint:

```bibtex
@article{toscano2025athena,
  title={ATHENA: Agentic Team for Hierarchical Evolutionary Numerical Algorithms},
  author={Toscano, Juan Diego and Chen, Daniel T and Karniadakis, George Em},
  journal={arXiv preprint arXiv:2512.03476},
  year={2025}
}
```

**Please also cite the libraries ATHENA builds on.** The physics-informed / operator-learning
(families **B** and **C**) path uses **NABLA-SciML** (the variational residual-based-adaptivity
framework where vRBA / SSBroyden was introduced), and the numerical-methods (family **A**) path
runs its solvers in **Julia**:

```bibtex
@article{toscano2025variational,
  title={A variational framework for residual-based adaptivity in neural PDE solvers and operator learning},
  author={Toscano, Juan Diego and Chen, Daniel T and Ooomen, Vivek and Darbon, J{\'e}r{\^o}me and Karniadakis, George Em},
  journal={npj Artificial Intelligence},
  volume={2},
  number={1},
  pages={32},
  year={2026},
  publisher={Nature Publishing Group UK London}
}

@article{bezanson2017julia,
  title={Julia: A fresh approach to numerical computing},
  author={Bezanson, Jeff and Edelman, Alan and Karpinski, Stefan and Shah, Viral B},
  journal={SIAM review},
  volume={59},
  number={1},
  pages={65--98},
  year={2017},
  publisher={SIAM},
  url={https://doi.org/10.1137/141000671}
}
```

## A Note on Model Availability (deprecated model IDs)

The example notebooks and the `build_agents_*` factories in `HENA.py` pin the **exact
LLM versions used to produce the paper's results** (e.g. `gemini-3-pro-preview`,
`gpt-5.1`, `claude-sonnet-4-5`, `grok-4-1-fast-reasoning`, with `gpt-oss:120b` and
Google `text-embedding-004` for retrieval). These pins are deliberate — they are the
provenance record of which models generated each figure.

However, **LLM providers periodically retire model IDs**, especially *preview*
versions. A retired model surfaces as a run-time error such as:

```
NOT_FOUND: 404 ... This model models/gemini-3-pro-preview is no longer available.
Please update your code to use a newer model ...
```

If you hit an error like this (`NOT_FOUND` / `model not found` / `404` / `model is no
longer available`), the model ID simply needs to be **swapped for a current
equivalent from the same provider**. Edit the `model=` argument wherever that agent is
constructed:

* **In a notebook** — the agent-setup cell (Strategy / Retrieval / Advisor teams).
* **For `run_athena.py`** — the matching `build_agents_*` factory in
  `crunch_llm/code/HENA.py`.

For example, replace a retired `model='gemini-3-pro-preview'` with the latest
available Gemini (e.g. `model='gemini-2.5-pro'`, or whichever is current). The agents
are provider-agnostic, so only the `model` / `model_provider` strings change — see
[**Configuring the Models**](#configuring-the-models). Swapping a model may slightly
change the generated solution (different models reason differently), but the HENA loop
and the framework behave identically.
