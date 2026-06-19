# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ATHENA is an agentic framework that turns a scientific-computing problem statement into working numerical code through an iterative, multi-agent loop ("HENA" loop). LLM agents (built on LangChain) propose a strategy, retrieve reference code from a vector DB, implement/patch the code, debug it, **execute it in an external compute environment via shell bridges**, and then critique the result — repeating until a solution is approved. The framework targets three problem families:

| Family | Code | Domain | Teams module | Example notebooks |
|--------|------|--------|--------------|-------------------|
| **A** | `SciC` | Classical numerical methods / Julia solvers | `teams_SciC.py` | `Examples_NM/`, `Examples_analytical_exact/` |
| **B** | `PINN` | Physics-informed ML | `teams_SciML_PIML.py` | `Examples_PINNs/` |
| **C** | `Operator` | DeepONets / operator learning | `teams_SciML_Operators.py` | `Examples_DeepONet/` |

(`teams_SciML_plain.py` is a fourth, plain-SciML variant kept for reference.)

The family code (A/B/C) is what the Conceptualization Team's **Gatekeeper** emits as the `path` for each pipeline stage, and it is exactly the argument `HENA.run_hena()` dispatches on. This A/B/C mapping is the spine of the whole system — keep it consistent everywhere.

---

## What solving a problem involves (two levels of stages)

Solving a problem in ATHENA happens at **two nested levels of stages**. Keep them distinct: the *macro* stages are the lifecycle of one problem; the *inner* stages are one iteration of the HENA loop that repeats until the work is approved.

### Macro stages — the lifecycle of one problem

```
Stage 0 · SETUP (once per machine)
     install + API keys (crunch_llm/code/.env) + point the two execution bridges
     (Documents/Scripts/run_python.sh, run_julia.sh) at your conda env / Julia project.
        │
Stage 1 · CONCEPTUALIZE
     Formalize.py  ──►  Conceptualization_team/ATHENA_Request_<name>.json
     { problem_name, pipeline:[{stage, path, task}], user_request, ... }
     The request JSON's `pipeline` is ITSELF a list of stages, each tagged with a
     family `path` (A=SciC, B=PINN, C=Operator). Single-stage for most problems;
     multi-stage chains families (e.g. B→A: PINN solve, then numerical validation).
        │
Stage 2 · RUN  (one call per pipeline stage)
     for stage in pipeline:  HENA.run_hena(stage.path, stage.user_request)
        └─ runs the inner HENA loop below until the advisor APPROVES
        │
Stage 3 · COLLECT
     generated code + data + per-iteration reports  ──►  Results/<run>/
     full multi-agent transcript                    ──►  crunch_llm/code/HENA_logs/<family>/
```

### Inner stages — one HENA iteration (repeats until APPROVED)

Each `run_hena` call loops these stages, feeding the advisor's `error_report` back into the next iteration's strategy, until the advisor returns APPROVED or `max_iterations_all` is hit:

```
1. STRATEGY        strategist ⇄ critic loop                 → strategist_report
2. RETRIEVAL       pull a code template from the vector DB  → initial_code
                   (B/C also pull auxiliary functions: collector → librarian → clean_functions)
3. IMPLEMENTATION  planner reads template cell-by-cell, patcher applies line-numbered patches
4. INSPECTION      inspector checks the code matches the strategy (else loop back, strategy held)
5. FILING          naming agent names the project; code written under Results/
6. EXECUTION       run_code() shells out to the bridge, captures stdout/stderr
7. DEBUGGING       language-aware traceback extraction; on failure loop back with an error_report
8. ADVISING        multimodal critic reads summary_all.png + log tail
                   → APPROVED  |  REJECTED_FOR_REFINEMENT (+ error_report for the next iteration)
```

Family **C (Operator)** fuses inner stages 1–4 into a single `code_creation_team_ON` call and uses `advisor_team_ON`; **B (PINN)** adds the auxiliary-function retrieval in stage 2. The stage *meaning* is the same across families — only the plumbing differs (see "Architecture").

There is **no test suite, linter, or build step.** Historically each problem was a hand-built notebook; `HENA.py` now consolidates those notebooks into one callable dispatcher so you drive runs from Python (a notebook cell or a thin script) instead of copy-editing notebooks.

---

## Setup & commands

```bash
pip install -e .                                        # installs the `crunch_llm` package (editable)
cp crunch_llm/code/.env_template crunch_llm/code/.env   # then add API keys
```

- **Two conda envs, distinct roles.** (1) The **orchestrator** env runs the agentic process itself (`run_athena.py` / `HENA.py`) and needs `chromadb` + `langchain>=1.0`/`langgraph` — this is the env from README step 1 (`athena_env`; on this machine it is **`ATHENA_2_env`**). (2) The **execution-bridge** env is the one `run_python.sh` activates to run *generated* Python/JAX code (`CONDA_ENV_NAME`, here **`NABLA_JAX`**). Activate the orchestrator env before launching a run; point the bridge at the execution env.
- **Ollama server required for the default agents.** The as-shipped `build_agents_*` factories in `HENA.py` run most roles on **`gpt-oss:120b` via a local Ollama server** (embeddings use Google). Start it before any run — on HPC that provides it: `module load ollama`, then `ollama serve` (leave running; listens on `127.0.0.1:11434`), then `ollama pull gpt-oss:120b` once, and `ollama list` to verify. **No local GPU?** `gpt-oss:120b` needs one, but it's an easy swap: the agents are provider-agnostic, so point them at **any provider** — Gemini (`model='gemini-3-flash-preview'`, `model_provider='google'`), ChatGPT/OpenAI (`'openai'`), Anthropic (`'claude'`), or Grok (`'grok'`) — in the `build_agents_*` factories (see "Swapping models"). Embeddings already run on cloud, so an all-cloud setup needs neither GPU nor Ollama.
- **API keys** (`GOOGLE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`) live in `crunch_llm/code/.env`. They are loaded explicitly relative to `agents.py`'s own directory (`Path(__file__).parent / '.env'`), **not** from the repo root or CWD. Which keys you need depends on which providers the agents in your family use (see "Swapping models" below). `ollama` needs no key — just the running server above.
- **Pre-flight before any run:** the two execution bridges (`Documents/Scripts/run_python.sh`, `run_julia.sh`) must point at *your* conda env / Julia project. See "Execution bridges" and the README "Preparing the Run Scripts" section.

---

## How to use each part

### Stage 1 — Define a problem — `Conceptualization_team/Formalize.py`

Interactive Conceptualization Team (Coordinator ⇄ User ⇄ Gatekeeper, built on `teams_conceptualization.py`). Run it and answer prompts:

```bash
cd Conceptualization_team
python Formalize.py
```

It clarifies the problem with you, then the Gatekeeper routes it and writes a **request JSON** to `Conceptualization_team/ATHENA_Request_<problem_name>.json`. The full session transcript is saved under `Conceptualization_team/Formalization_logs/`.

You do **not** have to run `Formalize.py` — the request JSONs can be (and the paper's were) authored by hand. The schema is:

```json
{
  "problem_name": "Poisson_PINN",
  "pipeline": [
    { "stage": 1, "path": "B", "task": "PINN setup, training, and evaluation." }
  ],
  "description": "PINN solution of the 2D Poisson equation.",
  "user_request": "Lets solve the 2D Poisson equation ...",   // the verbatim spec fed to the loop
  "source_notebook": "Examples_PINNs/Poisson.ipynb"
}
```

- `pipeline` is a **list of stages**, each with a `path` (A/B/C) and a `task`. Most problems are single-stage. Multi-stage examples chain families — e.g. `ATHENA_Request_Inverse_2D_NavierStokes_LidDrivenCavity_ReInference.json` is a 2-stage **B → A** pipeline (PINN solve, then numerical validation); its `user_request` concatenates the per-stage specs under `### STAGE 1` / `### STAGE 2`.
- `user_request` is the string handed to the HENA loop as `user_question`. Make it self-contained (equation, domain, BCs/ICs, what to plot/save).

The 18 `ATHENA_Request_*.json` files in `Conceptualization_team/` are the exact problem set from the paper — use them as templates.

### Stage 2 — Run the HENA loop — `crunch_llm/code/HENA.py`

This is the central entry point. It is a **dispatcher over three preserved loops** (not a merged loop), because the three families differ in (1) vector-DB config, (2) which agents are built, and (3) the loop body itself.

Minimal use:

```python
from crunch_llm.code.HENA import run_hena

result = run_hena("B", user_request)     # builds agents + runs the PINN loop, logs the run
print(result["is_approved"], result["project_dir"], result["log_path"])
```

Reuse agents across several problems of the same family (avoids re-instantiating LLM clients):

```python
from crunch_llm.code.HENA import build_agents, get_config, run_hena

cfg    = get_config("A")                 # family A (SciC) default config
agents = build_agents("A", cfg)
for q in questions_A:
    run_hena("A", q, agents=agents, cfg=cfg)
```

`run_hena(family, user_question, *, ...)` keyword arguments:

| arg | meaning |
|-----|---------|
| `agents` | prebuilt agent dict (from `build_agents`); built fresh if `None`. |
| `cfg` | full config dict; defaults to `get_config(family, config_overrides)`. |
| `config_overrides` | partial dict merged onto the family default (e.g. `{"max_iterations_all": 10}` for a quick test, or a different `base_results_dir`). |
| `checkpoint_project_dir` | resume from a prior run's `Results/` folder. |
| `load_code_only` | `True` = start fresh from a previously-successful script; `False` = restore full prior state (strategy, history, error report). |
| `specific_requests` | extra instruction string appended to the strategist prompt. |
| `log` / `log_base` / `run_name` | run-transcript logging (on by default). |

**Return value** (dict): `family`, `is_approved`, `project_dir`, `filename`, `initial_code`, `history`, `history_log_string`, `error_report`, `refinement_iterations`, and `log_path` (when logging is on).

**To drive a request JSON across its pipeline stages**, read the JSON and call `run_hena` once per stage, feeding each stage's `path` and the (stage-scoped) `user_request`. There is currently **no `run_athena.py` caller** that does this automatically — if asked to "run a request file," that thin loop is the thing to write.

**Per-family run logs.** Every run mirrors its full stdout transcript to
`crunch_llm/code/HENA_logs/<FAMILY_LABEL>/hena_<family>_<problem_name>_<timestamp>.txt`
(`FAMILY_LABEL` = `A_SciC` / `B_PINN` / `C_Operator`). The file is opened with a timestamp name, then renamed to include the discovered problem name when the run names its project. This is the auditable record that replaces the notebook chat history.

**Swapping models.** The model/provider for every agent is chosen in the `build_agents_*` factories in `HENA.py` (one factory per family, lifted from each notebook's agent-setup cell). To change which LLM plays a role (strategist, critic, debugger, advisor, …), edit the `model=`/`model_provider=` on that agent there. `StructuredAgent` roles need a provider that supports structured output (e.g. `gpt-oss:120b`, Grok). Providers dispatch by string: `ollama` | `google` | `openai` | `claude` | `grok`.

### Stage 0/2 — Execute generated code — execution bridges (critical)

ATHENA does **not** run generated code in-process. Inside the loop, `utils.run_code(project_dir, filename, final_filepath)` selects a shell script by file extension and shells out, capturing stdout/stderr to `<file>.out` / `<file>.err`:

- `.py` → `Documents/Scripts/run_python.sh` — sources a conda activation script, `conda activate`s an env, runs `python -u`.
- `.jl` → `Documents/Scripts/run_julia.sh` — runs `julia --project=<env> --threads=<N>`.

`run_code()` now resolves these scripts **repo-relative** (`<repo>/Documents/Scripts/`), so `utils.py` no longer needs per-machine edits. But the **scripts themselves hold machine-specific paths you must set**:

| Script | Fields to set |
|--------|---------------|
| `run_python.sh` | `CONDA_ACTIVATION_SCRIPT` (path to `conda.sh`), `CONDA_ENV_NAME` |
| `run_julia.sh` | `JULIA_EXEC` (julia binary), `PROJECT_ENV` (Julia project dir) |

`run_julia.sh` sets `--threads` from `SLURM_CPUS_PER_TASK` → `SLURM_CPUS_ON_NODE` → `nproc`, so Julia uses every core in the current allocation automatically.

**Verify the bridges before a real run** with a toy script through `run_code()`:

```python
import crunch_llm.code.utils as U
ok, log = U.run_code("scratch/bridge_test", "toy.py", "scratch/bridge_test/toy.py")   # or toy.jl
```

> Historical gotcha (now fixed in this checkout): the run scripts and `run_code()` used to point at the author's macOS layout (`/Users/jdtoscan/...`, env `JAX_311`). If launches fail, re-check these four script fields first.

### Optional — Add reference knowledge — `Ingestion/`

Reference code retrieved during the loop lives in a **Chroma vector DB**. The standard templates in `Documents/codes/` (Julia solvers, PINN/DeepONet Python modules) are pre-loaded — no ingestion needed for the paper's problems.

To add custom knowledge, use `Ingestion/ingest.py` and the per-family folders `Ingestion/{PINNs,Numerical_methods,DeepONet}/` (set the DB path and `COLLECTION_NAME` to match the family's `DB_CONFIG` in `HENA.py`). Ingestion reports are written as `Ingestion/ingest_report_*.json`. **Retrieval accuracy depends on cell granularity** — organize ingested code into distinct, well-docstring'd cells.

---

## Architecture (file-by-file)

All core code is under `crunch_llm/code/`:

1. **`agents.py` — agent primitives.** Thin LangChain wrappers, each with `.run()`, `.clear_history()`, `.get_history()` and `model` + `model_provider` arguments. Key classes:
   - `BasicAgent` / `StructuredAgent` / `BasicStructuredAgent` — plain and JSON/structured-output LLM calls. `StructuredAgent` is used wherever output must be machine-parsed.
   - `memoryAgent` — conversational agent with short-term memory (used for strategist/critic/planner/librarian roles).
   - `aRAGAgent`, `aRAGAgent_single_chunk`, `aRAGAgent_multimodal`, `Retriever_Agent*` — retrieval-augmented agents over the Chroma store (the multimodal one reads the result plot).

2. **`teams_*.py` — the HENA orchestration teams.** Four near-parallel variants (`teams_SciC` = A, `teams_SciML_PIML` = B, `teams_SciML_Operators` = C, `teams_SciML_plain`). Each defines the **same team-function NAMES** — `strategy_team`, `retrieval_team`, `implementation_team`, `debugging_team_PINN`, `advisor_team`, `filing_team_PINN`, … — but with **DIFFERENT signatures**, so `HENA.py` imports them as namespaces (`TA`/`TB`/`TC`) and never `from teams_* import *`.

   One loop iteration (A, the canonical shape): `strategy_team` (proposer–critic loop → `strategist_report`) → `retrieval_team` (pull reference code from the vector DB) → `implementation_team` (planner studies code cell-by-cell, then `patcher_agent` applies structured patches) → inspector + `filing_team_PINN` (name + write the project) → **`run_code`** → `debugging_team_PINN` (language-aware traceback extraction from the logs) → `advisor_team` (multimodal: reads `summary_all.png` + log tail, returns APPROVED / REJECTED_FOR_REFINEMENT). The advisor's `error_report` feeds the next iteration's `strategy_team`.

   **Family differences that matter** (these are why the loops are *not* interchangeable):
   - **B (PINN)** adds an auxiliary-module path: `retrieval_team_direct` (via `collector_agent`) + `librarian_agent` → `clean_functions`, and `implementation_team` takes the extra `clean_functions` arg. B can preload `my_functions` from a checkpoint dir (`functions_checkpoint_dir` in its config). Its `debugging_team_PINN` is **5-arg** (extra `filename`); A's and C's are 4-arg.
   - **C (Operator)** fuses the front half of the loop into `code_creation_team_ON` and uses `advisor_team_ON` instead of `advisor_team`.

   ⚠️ Because the variants are largely duplicated, **when you change loop behavior, check whether the same edit belongs in the other `teams_*.py` files** — and whether `HENA.py`'s corresponding `run_hena_*` loop needs the matching change.

3. **`utils.py` — shared mechanics.** Vector DB init (`initialize_local_data`), checkpoint I/O (`load_checkpoint_data`, `update_and_save_report`, report consolidation), plot discovery (`find_plots`), image encoding for multimodal agents, the execution bridge (`run_code`), and the **patch/cell machinery**: code is split into named cells (`split_code_into_cells`), line-numbered (`add_line_numbers_to_code`), patched by line number (`apply_patch`), then stripped back (`remove_line_number_tokens`). This line-numbered-patch contract is how agents edit code surgically — **preserve it.**

4. **`HENA.py` — the dispatcher** (see "Run the HENA loop" above). Holds `DB_CONFIG` (per-family vector-DB paths, `base_results_dir`, `max_iterations_all` = 50/25/100 for A/B/C), the `build_agents_*` factories, the `run_hena_*` loops (lifted verbatim from the notebooks), the per-family run logger, and the `run_hena` / `build_agents` / `get_config` public API. **Vector-DB and Results base paths are machine-specific** (`DB_BASE`, `RESULTS_BASE` near the top) — override per-call via `config_overrides` or edit the defaults.

5. **`teams_conceptualization.py`** — agents/prompts for the Conceptualization Team driven by `Formalize.py`.

### Repo directories

- `Conceptualization_team/` — `Formalize.py`, the `ATHENA_Request_*.json` problem set, `Formalization_logs/`.
- `Examples_PINNs/`, `Examples_NM/`, `Examples_analytical_exact/`, `Examples_DeepONet/` — the original per-problem notebooks (now superseded by `HENA.py` for running, but kept as the authoritative specs).
- `Documents/codes/` — pre-loaded reference templates; `Documents/Scripts/` — the execution bridges; `Documents/database_*` — Chroma stores.
- `Ingestion/` — knowledge-base ingestion.
- `Results/` — generated code, data, and per-iteration reports (one subfolder per run).
- `crunch_llm/code/HENA_logs/` — per-family run transcripts.
- `Ablation_studies/`, `Similarity_Analysis/`, `Plots/`, `Reviews/` — paper analysis artifacts.

---

## Conventions & gotchas

- **A/B/C is the contract.** Family code from the request JSON `path` == `run_hena` family == teams module == `DB_CONFIG` key. Don't break this mapping.
- **Never `from teams_* import *`** — the modules share function names with different signatures. Import as `TA`/`TB`/`TC`.
- **Machine-specific paths live in exactly three places:** the two `Documents/Scripts/*.sh` bridges, and `DB_BASE`/`RESULTS_BASE` in `HENA.py`. `utils.run_code()` is now path-agnostic.
- **`StructuredAgent` roles need structured-output-capable providers.** Don't point them at a model that can't return JSON reliably.
- Agents' short-term memories are cleared each iteration (`clear_history()`); durable state flows through the checkpoint/report files in `Results/`.
