# CLAUDE.md

Guidance for the coding agent (Claude Code, Codex, or any agent that reads this file; `AGENTS.md`
points here) working in this repository.

**People use ATHENA through you.** A user opens a coding agent in this repository and you take
them through the whole workflow:

1. **Install**: get the container and the API keys in place, and check that the code runs.
2. **Formalize**: talk with the user until their problem is a complete, approved specification,
   and write it as an ATHENA request JSON. You act as ATHENA's Conceptualization Team.
3. **Run**: launch the request in the container and monitor it.
4. **Report**: tell the user how the run ended, where the results are, and what to do next.

You drive ATHENA; you do not replace it. Choosing and designing the method, writing, debugging and
judging the code is the work of ATHENA's own agents inside the HENA loop. Results are reproduced
from the request JSON, `run_athena.py`, the release container and the logged transcripts, never
from decisions you make mid-run.

---

## Rules (read first)

1. **The container is the execution environment.** Every generated code runs inside the release
   container (`container/images/athena_release.sif`), whether the run is launched with
   `athena_box_run.sh` or `run_athena.py`. Use `ATHENA_EXEC=host` only if the user asks for it.
2. **Never modify the environment to make a run work.** Do not install, update or pin packages in
   the container, a host env or the Julia project, and do not rebuild the image, without the user's
   explicit approval. When a run stops for help (STOP_UNRECOVERABLE), explain the diagnosis to the
   user; the fix is their decision.
3. **Never touch the API keys.** They live only in `crunch_llm/code/.env`, which the user fills in
   themselves. Never ask for a key in the chat, never print the file or a value, never copy them
   elsewhere, never `git add` the file. Check that a key is set by its name only
   (`grep -c '^OLLAMA_API=' crunch_llm/code/.env`).
4. **Do not do ATHENA's job.** Do not put methods, discretizations, loss functions or code into a
   request, and do not edit the generated code or the run's reports while a run is going.
5. **Never overwrite anything.** Not a request JSON, a run folder, a vector DB or a report. Each run
   gets a new folder; a stopped run continues with `--resume`. Two exceptions, both only with the
   user's approval: filling the previous stage's output path into a multi-stage request (Step 3),
   and ingesting new reference code into the vector DBs (Reference).
6. **Do not edit code or scripts that a running job uses** (`crunch_llm/code/`, `run_athena.py`,
   `Documents/Scripts/`, the request JSONs). Check `squeue -u $USER` or the running processes first.
7. **Models are reached through APIs.** Inside the box the `gpt-oss:120b` roles go to Ollama Cloud;
   no local model server is needed. Do not start `ollama serve` or pull models unless asked.
8. **Keep runs comparable.** Do not change agent prompts, team models, `max_iterations_all` or the
   score gate unless the user asks (e.g. to try other models; see "Models" in Reference).
   `--max-iters` is for quick tests the user asked for.
9. **Git:** commit or push only when the user asks. Never add `.env`, the image (`container/images/`),
   build logs or heavy run outputs under `Results/`.

---

## Step 1: Install (once per machine)

Check, in this order, and fix only what is missing:

1. **Apptainer and hardware.** `apptainer --version`; `nvidia-smi` for a GPU (needed by the PINN
   and operator codes of families B and C; the Julia solvers of family A run on CPUs); `sinfo` to
   see whether this is a SLURM cluster.
2. **The container image** at `container/images/athena_release.sif` (or wherever `ATHENA_SIF`
   points). If it is missing, ask the user whether to download it or build it.
   - Download (9.4 GB zip with the repo and the image), from the repository root:
     `wget -O ATHENA_release.zip "https://zenodo.org/records/24096756/files/ATHENA_release_4f2e92c.zip?download=1"`, then
     `unzip -j ATHENA_release.zip 'ATHENA/container/images/*' -d container/images/`,
     `sha256sum -c container/images/athena_release.sif.sha256` (must print OK), `rm ATHENA_release.zip`.
   - Build: `mkdir -p container/logs && sbatch container/build_release.sh` on SLURM, `bash container/build_release.sh`
     elsewhere. It takes 1 to 2 hours and needs `--fakeroot` and about 60 GB of scratch; it never
     overwrites an image and writes `athena_release.sif.sha256` next to it.
   - The paper image has sha256 `7dc4537f9c14bae42d4ae9f582bcaa92dd45a52e126c47a20746436030b030e7`.
   - Contents: Julia 1.11.9 with the frozen project in `container/julia/` (Trixi 0.13.15,
     precompiled, read-only); the `nabla_jax` env (Python 3.12, JAX 0.8.2 CUDA 12, Flax, Optax,
     NABLA from `third_party/NABLA`) for generated Python; the `athena` env for the orchestrator.
     See `container/README.md`.
3. **The API keys.** If `crunch_llm/code/.env` is missing, run
   `cp crunch_llm/code/.env_template crunch_llm/code/.env` and ask the user to fill it in
   themselves (rule 3). Which keys are needed:
   - `OLLAMA_API`: Ollama Cloud, for the `gpt-oss:120b` roles of the original team and the open team.
   - `GOOGLE_API_KEY`: the Gemini agents and the retrieval embeddings (`gemini-embedding-001`);
     always needed.
   - `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`: the original team's GPT, Claude and Grok
     roles.
   - `CLAUDE_CODE_OAUTH_TOKEN`: only for the Claude Code baseline.

   The launch scripts read the file on the host and pass the keys to the orchestrator only; the
   generated code never sees them.
4. **The bridges.** One toy script per language, each through its bridge in the container:
   ```bash
   mkdir -p Results/Tests/bridge && cd Results/Tests/bridge
   echo 'import jax; print("python OK", jax.devices())' > toy.py
   echo 'println("julia OK")' > toy.jl
   bash ../../../Documents/Scripts/run_python.sh toy.py toy.py.err
   bash ../../../Documents/Scripts/run_julia.sh  toy.jl toy.jl.err
   ```
   On a GPU node the first prints `python OK [CudaDevice(id=0)]`. Run it on a compute node, not a
   login node.
5. **Large reference data**, only when a request needs it: `Burger_1000_5000.mat` (374 MB, for
   `Viscous_Burgers_DeepONet`) is downloaded into `Documents/codes/Data/` from Zenodo:
   `wget -O Documents/codes/Data/Burger_1000_5000.mat "https://zenodo.org/records/24096756/files/Burger_1000_5000.mat?download=1"`.

---

## Step 2: Formalize the problem with the user

You play ATHENA's Conceptualization Team: first the **Coordinator**, who talks with the user, then
the **Gatekeeper**, who writes the request. Their exact instructions are
`coordinator_system_prompt` and `gatekeeper_system_prompt` in
`crunch_llm/code/teams_conceptualization.py`. **Read them before formalizing and follow them**; the
summary below is not a substitute.

If the user wants a problem that already exists, use its request in `Conceptualization_team/`
(these are the paper's problems) and skip to Step 3.

### As the Coordinator

- **Ask one question at a time.** First: is it a forward or an inverse problem? Then the
  equation(s), physics parameters, domain (space and time), boundary conditions, initial
  conditions, and whether reference data exists (where, and in what format).
- **Classify** it: forward or inverse, steady or transient, elliptic / parabolic / hyperbolic /
  mixed, simple or complex geometry.
- **Propose the methodology** with the families **A** (SciC, numerical methods: the accurate choice
  for forward problems), **B** (PINN: inverse and data-driven problems) and **C** (neural operators:
  parametric studies), including the hybrids the prompt describes: A→B (a numerical reference to
  validate a PINN), A→C (numerical data to train an operator), B→A (verify an inferred parameter
  with a numerical solve). The user decides.
- **Ask how the result will be validated** (reference data, a numerical solution, MMS,
  conservation) and **check well-posedness** (enough BCs and ICs, nothing missing).
- **Present the specification** in the prompt's markdown format: Problem Statement, Equation(s),
  Domain, Boundary Conditions, Initial Conditions, Physics Parameters, Execution Strategy,
  Validation Strategy, Reference Data, Special Instructions. Include what to compute, plot and
  save.
- **Keep the Execution Strategy high-level**: "generate reference data with a numerical method",
  "train a PINN", "compare". No loss formulas, discretization schemes, named numerical methods or
  code (rule 4).
- **Get explicit approval** of the specification before writing anything. A correction means
  revising and presenting it again.

### As the Gatekeeper

Turn the approved specification into
`Conceptualization_team/ATHENA_Request_<problem_name>.json`:

```json
{
  "problem_name": "Poisson_PINN",
  "pipeline": [
    { "stage": 1, "path": "B", "task": "PINN setup, training, and evaluation." }
  ],
  "description": "PINN solution of the 2D Poisson equation.",
  "user_request": "<the approved specification, verbatim>"
}
```

- **Minimal pipeline.** A stage is one handoff to one family. Drop implementation sub-steps
  (refinement, evaluation, error metrics, saving, plotting); merge consecutive stages of the same
  family; number from 1. Almost always 1 or 2 stages (e.g. infer with B, verify with A).
- **`problem_name`**: a short `Standardized_Name` (letters, digits, underscores). If a request with
  that name exists, choose another name (rule 5).
- **`user_request`** is what the HENA loop receives, so it must be self-contained. For a multi-stage
  request, split the specification under `### STAGE 1`, `### STAGE 2`, ... headings, one
  self-contained section per stage; `run_athena.py` hands each stage its own section. A stage that
  reads the previous stage's output says so and leaves the path to be filled in (Step 3).
- **Reference data** must be in `Documents/codes/Data/` (the only data folder the container
  mounts). Ask before copying a user's file there, then name it as
  `{REPO}/Documents/codes/Data/<file>` and describe its format (keys, shapes, layout).
  `run_athena.py` replaces `{REPO}` with the repository root; never write an absolute machine path.
- Show the JSON to the user before Step 3.
- **Keep a record**, as `Formalize.py` does: write the questions, the user's answers and the
  approved specification to `Conceptualization_team/Formalization_logs/formalization_<YYYYmmdd_HHMMSS>.md`.

If the user prefers, `cd Conceptualization_team && python Formalize.py` runs the same team with
its own models (GPT-5.1 Coordinator, Grok Gatekeeper) and writes the JSON and its transcript; it
needs the orchestrator env (see "Reference").

---

## Step 3: Run

### Recommended: one run in one box (`athena_box_run.sh`)

```bash
bash Documents/Scripts/athena_box_run.sh <request.json> <team> <RUN_DIR> [run_athena.py options]
# e.g.
bash Documents/Scripts/athena_box_run.sh \
    Conceptualization_team/ATHENA_Request_Helmholtz_PINN.json original Results/MyRuns/Helmholtz/run1
```

The orchestrator, the agents and the generated code all run inside the container
(`--nv --containall --cleanenv`). The repository is mounted read-only, `.env` is masked, and
`RUN_DIR` is the only writable place. A `RUN_DIR` inside the repository must be under `Results/`; an
existing run is refused unless `--resume` is given. Layout:

```
<RUN_DIR>/
  job_info.txt   node, GPU, start/end, git commit, image sha256, exit code
  stdout.txt     the full console output (stdout_resume_<stamp>.txt after a resume)
  logs/          agent transcripts (hena_*.txt)
  usage/         token / time log, one record per LLM call and code execution
  dbs/           private copies of the vector DBs this run used
  results/       generated code, outputs and Experiment_Report.md per iteration
  box/           the box's home and writable Julia depot
```

- **Team.** `original` (the paper's team allocation, current model versions) unless the user asks for another (options table below).
- **Where.** Families B and C need a GPU node; family A needs CPUs (Julia uses every core of the
  allocation). A run takes hours, so on SLURM submit it as a batch job that calls
  `athena_box_run.sh`, with the partition, GPU and time limit agreed with the user.
  `Documents/Scripts/sensitivity_run.sbatch` is a worked example (its run folder is
  `Results/Sensitivity/<team>/<case>/run<N>` and its `#SBATCH` lines are for the authors'
  cluster), and `sensitivity_sequential.sh` runs several boxes one after another on one node.
- **Multi-stage requests** run one `--stage` at a time, each into its own `RUN_DIR` under
  `Results/`. After a stage finishes, and with the user's approval, fill its output path into the
  next stage's section of the request as `{REPO}/Results/...` (e.g. the LDC Re-inference: Stage 1
  writes `results.npz`, Stage 2 reads it), then run `--stage 2`. The generated code sees earlier
  runs under `Results/` read-only, in the box and in the direct mode below.

### Direct: `run_athena.py`

```bash
ATHENA_OLLAMA_CLOUD_ONLY=1 python run_athena.py Conceptualization_team/ATHENA_Request_<name>.json [options]
```

The orchestrator runs in the current Python env (see "Reference"); each generated code still runs
in the container through the bridges. Without `ATHENA_OLLAMA_CLOUD_ONLY=1`, `gpt-oss:120b` calls a
local Ollama server.

| Option | Effect |
|---|---|
| `--team T` | `original` (the paper's team allocation with current model versions, default, families A/B/C), `open` (open-weight models on Ollama Cloud, `OPEN_TEAM` in `HENA.py`, families A/B), `closed` (original team with its `gpt-oss:120b` utility roles on Gemini 3.1 Pro, families A/B) |
| `--runs N` | repeat the whole pipeline N times |
| `--max-iters N` | cap `max_iterations_all` (quick tests only) |
| `--stage N` | run only pipeline stage N |
| `--resume DIR` | continue from the newest `Experiment_Report.md` at or under DIR (needs `--stage` for multi-stage requests; no `--runs`) |
| `--results-dir`, `--log-base`, `--usage-dir`, `--db-copy-dir` | output locations and private DB copies (the box sets them) |

`athena_box_run.sh` passes extra options through to `run_athena.py`.

### Monitoring

Tail `<RUN_DIR>/stdout.txt` and tell the user at natural points (each iteration's Advisor outcome
and score, a crash, the end). Do not intervene in the loop (rule 4).

---

## Step 4: Report

A stage ends in one of four ways; `run_athena.py` prints `approved=... project_dir=... log=...`.

- **Approved**: the Advisor approved a trial.
- **Score gate** (families A and B): once a trial scored at least 90, the run continued until five
  iterations had run counting from that trial (crashes included), then stopped and returned the
  best-scoring trial.
- **Iteration limit**: `max_iterations_all` (50 / 25 / 100 for A / B / C) was reached.
- **Help needed** (exit code 3): the Debugger classified the error as unrecoverable (missing
  package, broken environment, invalid input). Read `UNRECOVERABLE_ERROR.md` in the project folder,
  explain the problem to the user, and give them the resume command it contains. After the user
  fixes the problem, run that command; it continues from the last checkpoint.

Then report: the outcome, the best trial's folder (`Experiment_Report.md` there has the strategy,
the Advisor's assessment and the full trial history), its figure (`summary_all.png`), the error
metric it reports against the reference, and the transcript. For a multi-stage request, say what
the next stage needs.

---

## Reference

### What ATHENA is

ATHENA turns a scientific-computing problem statement into working numerical code through an
iterative multi-agent loop (the HENA loop). LLM agents (built on LangChain) propose a strategy,
retrieve reference code from a vector DB, implement and patch the code, debug it, **execute it in
the container through two shell bridges**, and critique the result, repeating until a solution is
approved or the run stops. Three problem families:

| Family | Code | Domain | Teams module | Example notebooks |
|--------|------|--------|--------------|-------------------|
| **A** | `SciC` | Classical numerical methods / Julia solvers | `teams_SciC.py` | `Examples_NM/`, `Examples_analytical_exact/` |
| **B** | `PINN` | Physics-informed ML (JAX) | `teams_SciML_PIML.py` | `Examples_PINNs/` |
| **C** | `Operator` | DeepONets / operator learning (JAX) | `teams_SciML_Operators.py` | `Examples_DeepONet/` |

(`teams_SciML_plain.py` is a fourth, plain-SciML variant kept for reference.) The family code is
the request's `path` and exactly the argument `HENA.run_hena()` dispatches on; keep this mapping
consistent everywhere.

### One HENA iteration

```
1. STRATEGY        strategist ⇄ critic loop                 → strategist_report
2. RETRIEVAL       pull a code template from the vector DB  → initial_code
                   (B/C also pull auxiliary functions: collector → librarian → clean_functions)
3. IMPLEMENTATION  planner reads the template cell by cell, patcher applies line-numbered patches
4. INSPECTION      inspector checks the code matches the strategy (else loop back, strategy held)
5. FILING          naming agent names the project; code written under the results folder
6. EXECUTION       run_code() calls the bridge, which runs the code in the container
7. DEBUGGING       language-aware traceback extraction; on failure loop back with an error_report,
                   or STOP_UNRECOVERABLE for an environment / package / input error
8. ADVISING        multimodal critic reads summary_all.png + log tail, scores the trial
                   → APPROVED  |  REJECTED_FOR_REFINEMENT (+ error_report for the next iteration)
```

Every scored iteration writes `Experiment_Report.md` next to its code; this is the checkpoint
`--resume` reloads. Family **C** fuses stages 1–4 into `code_creation_team_ON` and uses
`advisor_team_ON`; **B** adds the auxiliary-function retrieval in stage 2. There is no test suite,
linter or build step for the Python code.

### Execution bridges (`Documents/Scripts/run_python.sh`, `run_julia.sh`)

`utils.run_code(project_dir, filename, final_filepath)` picks the bridge by extension (`.py` →
`run_python.sh`, `.jl` → `run_julia.sh`), resolves it repo-relative, and captures stdout/stderr to
`<file>.out` / `<file>.err`. Both take `<script> <stderr.log>`. Three modes:

| Mode | When | What runs |
|---|---|---|
| in box | `ATHENA_IN_BOX=1` (set by `athena_box_run.sh`) | the image's Python / Julia directly, keys stripped |
| **container** (default) | `ATHENA_EXEC` unset or `container` | `apptainer exec --containall --cleanenv` (Python adds `--nv`): only the script's folder is writable, `Documents/codes/Data` and `Results/` (earlier runs) are mounted read-only, the rest of the repository is invisible, Julia gets a per-job writable depot in front of the image's read-only one |
| host | `ATHENA_EXEC=host` (only on request; install: README, "Installing without the container") | Python: `ATHENA_CONDA_ENV` (env name, default `NABLA_JAX`, or the env folder's path), `ATHENA_CONDA_SH` (path to `conda.sh`, default from `conda info --base`). Julia: `ATHENA_JULIA_ENV` (project instantiated from `container/julia/`, required), `ATHENA_JULIA_EXEC` (default `julia` on PATH) |

Julia threads come from `SLURM_CPUS_PER_TASK` → `SLURM_CPUS_ON_NODE` → `nproc`. The bridges hold no
machine-specific paths; do not hardcode any.

### Orchestrator env (only for `run_athena.py` or `Formalize.py` outside the box)

Python 3.12 with `pip install -r container/requirements_athena.txt && pip install -e . --no-deps` (`langchain>=1.0`,
`langgraph`, `chromadb`). The box uses the image's own `athena` env.

### The HENA API (`crunch_llm/code/HENA.py`)

`run_athena.py` is the normal caller. `HENA.py` is a **dispatcher over three preserved loops** (not
a merged loop), because the families differ in vector-DB config, agents and loop body.

```python
from crunch_llm.code.HENA import build_agents, get_config, run_hena

cfg    = get_config("B", {"max_iterations_all": 5})   # family default + overrides
agents = build_agents("B", cfg, team="original")
result = run_hena("B", user_request, agents=agents, cfg=cfg)
print(result["is_approved"], result["project_dir"], result["log_path"])
```

| `run_hena` arg | meaning |
|-----|---------|
| `agents` | prebuilt agent dict (from `build_agents`); built fresh if `None`. |
| `cfg` | full config dict; defaults to `get_config(family, config_overrides)`. |
| `config_overrides` | partial dict merged onto the family default (`max_iterations_all`, `base_results_dir`, `score_gate`, ...). |
| `checkpoint_project_dir` | resume from a prior run's folder (holding `Experiment_Report.md`). |
| `load_code_only` | `True` = start fresh from a previously successful script; `False` = restore full prior state. |
| `specific_requests` | extra instruction string appended to the strategist prompt. |
| `log` / `log_base` / `run_name` | transcript logging (on by default). |

Returns a dict: `family`, `is_approved`, `project_dir`, `filename`, `initial_code`, `history`,
`history_log_string`, `error_report`, `refinement_iterations`, `log_path`, and the stop fields
`stop_reason`, `gate_stop`, `last_checkpoint_dir`.

**Transcripts.** By default `Examples_<family>/<ProblemName>_log.txt` (A → `Examples_NM`, B →
`Examples_PINNs`, C → `Examples_DeepONet`); with `log_base` (the box sets `--log-base`)
timestamped `<log_base>/<A_SciC|B_PINN|C_Operator>/hena_<family>_<problem>_<stamp>.txt`.

**Models.** Each agent's model is set in the `build_agents_*` factories in `HENA.py`
(`model=` / `model_provider=`); the alternative teams are `OPEN_TEAM` and `TEAM_BUILDERS`.
Providers: `google`, `openai`, `anthropic` (`claude`), `xai` (`grok`), `ollama_cloud`, `ollama`
(local server, or Ollama Cloud when `ATHENA_OLLAMA_CLOUD_ONLY=1`). `StructuredAgent` roles need a
provider with reliable structured output. Embeddings must stay `gemini-embedding-001` unless the
DBs are re-ingested.

### Reference knowledge (`Ingestion/`)

Retrieved reference code lives in Chroma vector DBs (`Documents/database_NM`, `database_PIML`,
`database_Operators`, `database_Modules_ollama`), pre-loaded from the templates in
`Documents/codes/`; the paper's problems need no ingestion. To add knowledge, use
`Ingestion/ingest.py` and `Ingestion/{PINNs,Numerical_methods,DeepONet}/`, with the DB path and
`COLLECTION_NAME` matching the family's `DB_CONFIG`. Retrieval depends on cell granularity:
distinct, well-docstringed cells. Ingesting writes the shared DBs: only on the user's request, never
while a run uses them.

### Architecture (`crunch_llm/code/`)

1. **`agents.py`**: thin LangChain wrappers with `.run()`, `.clear_history()`, `.get_history()`,
   `model` + `model_provider`: `BasicAgent` / `StructuredAgent` / `BasicStructuredAgent`,
   `memoryAgent` (short-term memory), `aRAGAgent*` / `Retriever_Agent*` (retrieval; the multimodal
   one reads the result plot); the Ollama Cloud client and the `ATHENA_OLLAMA_CLOUD_ONLY` routing.
   Keys load from `Path(__file__).parent / '.env'`.
2. **`teams_*.py`**: the HENA teams. The four variants define the **same function names**
   (`strategy_team`, `retrieval_team`, `implementation_team`, `debugging_team_PINN`,
   `advisor_team`, `filing_team_PINN`, ...) with **different signatures**, so `HENA.py` imports them
   as namespaces (`TA`/`TB`/`TC`), never `from teams_* import *`. B adds `retrieval_team_direct` +
   `librarian_agent` → `clean_functions` and a 5-arg `debugging_team_PINN`; C fuses the front half
   into `code_creation_team_ON` and uses `advisor_team_ON`. The Planner and Debugger prompts forbid
   changing the environment. ⚠️ **A change to loop behavior usually belongs in the other
   `teams_*.py` files and the matching `run_hena_*` loop too.**
3. **`utils.py`**: vector-DB init, checkpoint I/O (`load_checkpoint_data`,
   `update_and_save_report`), plot discovery, image encoding, `run_code`, and the **patch/cell
   machinery** (`split_code_into_cells`, `add_line_numbers_to_code`, `apply_patch`,
   `remove_line_number_tokens`). This line-numbered patch contract is how agents edit code;
   **preserve it.**
4. **`HENA.py`**: `DB_CONFIG` (per-family DBs, `base_results_dir`, `max_iterations_all`), with
   `DB_BASE` / `RESULTS_BASE` resolved from the repo root; the `build_agents_*` factories and
   `TEAM_BUILDERS` (`open` and `closed`, the model-sensitivity teams); the
   `run_hena_*` loops; `score_gate`; the run logger; the public API.
5. **`usage_tracker.py`**: one JSONL record per LLM call, embedding call and code execution, plus a
   summary JSON per stage.
6. **`teams_conceptualization.py`**: the Coordinator and Gatekeeper prompts (Step 2).

At the repo root, `run_athena.py` drives a request through its stages.

### Baselines without ATHENA (`bare_agent_run.sh`)

`bash Documents/Scripts/bare_agent_run.sh <request.json> claude/claude-opus-5-5 <RUN_DIR>` gives only
the request's `user_request` to Claude Code (Opus 5.5), the single-agent baseline of the paper,
inside the same container, with no ATHENA code, templates, DBs or web tools. The script header has
the folder layout.

### Repo directories

- `Conceptualization_team/`: `Formalize.py`, the request JSONs, `Formalization_logs/`.
- `Examples_PINNs/`, `Examples_NM/`, `Examples_analytical_exact/`, `Examples_DeepONet/`: the original notebooks (kept as specs) and the default transcripts.
- `Documents/codes/`: reference templates; `Documents/codes/Data/`: reference data (`allen_cahn.mat`, `burgers.mat`, `inviscid_burgers_sinpix.npz`, `ldc_Re500.npz`; `Burger_1000_5000.mat` downloaded); `Documents/Scripts/`: bridges and launchers; `Documents/database_*`: Chroma stores.
- `container/`: image definition, build script, pinned requirements, the frozen Julia project; `images/` and `logs/` stay local.
- `third_party/`: `NABLA` and `Crunch.jl`, installed into the image.
- `Ingestion/`: knowledge-base ingestion.
- `Results/`: generated code and reports, box runs, `Sensitivity/`, `Tests/`. Only `.py`/`.jl` and curated figure data are tracked.
- `Ablation_studies/`, `Similarity_Analysis/`, `Plots/`: paper analysis and figures (`Plots/README.md`).

### Conventions & gotchas

- **A/B/C is the contract**: request `path` == `run_hena` family == teams module == `DB_CONFIG` key.
- **Never `from teams_* import *`.**
- **No machine-specific paths in the code.** Repo paths resolve from the repo root, requests use
  `{REPO}`, the bridges read environment variables. Only the image location (`ATHENA_SIF`), `.env`
  and the `#SBATCH` headers are machine-specific.
- Agents' short-term memories are cleared each iteration; durable state is `Experiment_Report.md`
  and the checkpoint files in the results folder.
- `open` and `closed` exist for families A and B only; the score gate runs in the A and B loops.
