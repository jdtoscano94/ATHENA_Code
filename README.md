# ATHENA: Agentic Team for Hierarchical Evolutionary Numerical Algorithms

<img src="Agent_Framework.png" width="50%" alt="ATHENA Agent Framework">

Progress in computational science depends on complex numerical workflows that must faithfully encode physical laws, yet translating conceptual insight into reliable code remains a major bottleneck. Although large language models can generate isolated code fragments, they lack the structured reasoning required to design, verify, and iteratively refine complete scientific pipelines. Here we introduce ATHENA, an agentic framework explicitly designed to emulate scientific research modeled as a knowledge-driven contextual bandit process. Its core loop separates conceptual policy from numerical realization through expert-derived conceptual scaffolding, enabling principled diagnosis, reformulation, and repair of computational strategies. Across scientific computing and scientific machine learning tasks, ATHENA autonomously discovers exact analytical solutions, constructs stable numerical solvers, diagnoses ill-posed formulations, and orchestrates hybrid symbolic–numeric workflows. Quantitatively, ATHENA exceeds expert-designed workflows, achieving superhuman accuracy and robustness on canonical benchmarks. By reframing computation as an object of agentic reasoning, the framework enables autonomous orchestration of heterogeneous algorithms across scientific domains.


# Installation

ATHENA is an agentic framework that orchestrates workflows across three distinct computational environments: **Python (PyTorch)**, **JAX**, and **Julia**.

Please follow the steps below to set up each environment.

## 1. Primary Environment (Python + PyTorch)
This is the main agentic environment (ATHENA).

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/ATHENA.git](https://github.com/your-username/ATHENA.git)
    cd ATHENA
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    # Using Conda (Recommended)
    conda create -n athena_env python=3.12
    conda activate athena_env
    ```

3.  **Install PyTorch:**
    Install the version of PyTorch compatible with your CUDA version (see [pytorch.org](https://pytorch.org/get-started/locally/)). The pinned environment uses CUDA 12.6.
    ```bash
    # Example for CUDA 12.6
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    ```

4.  **Install ATHENA (Editable Mode):**
    This installs the `crunch_llm` package and its dependencies (LangChain 1.x + LangGraph, on which the agent layer depends).
    ```bash
    pip install -e .
    ```

    For an exact, reproducible environment, install the pinned lockfile instead of (or after) the editable install:
    ```bash
    pip install -r requirements.txt
    pip install -e . --no-deps
    ```
    > **Note:** The agent layer (`crunch_llm/code/agents.py`) is built on the modern `langchain.agents.create_agent` API, which requires **`langchain>=1.0` and `langgraph`**. Older `langchain` 0.3.x installs (which only provide `AgentExecutor`/`create_tool_calling_agent`) will not work.


5.  **Configure API Keys:**
    You must set up your LLM provider keys. A template file is provided in the source code.

    a. Navigate to the code directory:
    ```bash
    cd crunch_llm/code
    ```

    b. Create the `.env` file from the template:
    ```bash
    cp .env_template .env
    ```

    c. Open `.env` and paste your API keys:
    ```bash
    GOOGLE_API_KEY="YOUR_GOOGLE_KEY"
    OPENAI_API_KEY="YOUR_OPENAI_KEY"
    ANTHROPIC_API_KEY="YOUR_ANTHROPIC_KEY"
    XAI_API_KEY="YOUR_XAI_KEY"
    ```
---

## 2. JAX Environment (NABLA-SciML)
The JAX environment handles specific differentiation and optimization tasks using `NABLA-SciML` and `SSBroyden`.

### Setup Instructions (Required for vRBA)

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/jdtoscano94/NABLA-SciML.git](https://github.com/jdtoscano94/NABLA-SciML.git)
    cd NABLA-SciML
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    conda create -n nabla_env python=3.10
    conda activate nabla_env
    ```

3.  **Install JAX with GPU support:**
    *(This is required for all modules. We recommend installing it explicitly first to ensure CUDA support)*
    ```bash
    pip install -U "jax[cuda12]"
    ```

4.  **Install the NABLA package:**
    This installs the library in editable mode, which is necessary to import the vRBA models and the SSBroyden optimizer.
    ```bash
    pip install -e .
    ```

---

## 3. Julia Environment
High-performance numerical solvers are executed in Julia. You need to create an environment with `Trixi`, `StartUpDG`, `Gridap`, and `NPZ`.

1.  **Install Julia:**
    Download and install Julia from [julialang.org](https://julialang.org/downloads/).

2.  **Setup the Environment:**
    Open the Julia REPL and run the following commands to create a dedicated environment and install the required packages:

    ```julia
    using Pkg

    # 1. Activate a new environment (creates a folder named 'athena_julia_env')
    Pkg.activate("athena_julia_env")

    # 2. Add the required packages
    Pkg.add([
        "Trixi",
        "StartUpDG",
        "Gridap",
        "NPZ"
    ])

    # 3. Precompile dependencies
    Pkg.instantiate()
    ```


## 4. Connecting Environments

To enable ATHENA to control the external solvers, you must configure the bridge scripts to point to your specific installations.

### Julia Connection (`run_julia.sh`)

Navigate to the script located at:
`ATHENA/Documents/Scripts/run_julia.sh`

Open this file and modify the **Configuration** section to match your system paths.

**What to change:**
1.  **`JULIA_EXEC`**: Ensure this finds your `julia` executable (or load your module).
2.  **`PROJECT_ENV`**: Set this to the absolute path of the `athena_julia_env` folder you created in Step 3.

**Example `run_julia.sh` configuration:**

```bash
#!/bin/bash

# --- Configuration ---
# 1. Load your system module (if on HPC) or ensure julia is in PATH
# module load julia/1.12.4s-adme

# 2. Dynamically grab the path to the executable
JULIA_EXEC=$(which julia)

# 3. SET THIS PATH: The location of the Julia environment you created in Step 3
# Example: PROJECT_ENV="/home/username/athena_julia_env"
PROJECT_ENV="/users/your_name/data/athena_julia_env"

# --- Safety ---
set -e
# ... (rest of script) ...
```

### Python / JAX Connection (`run_python.sh`)

Navigate to the script located at:
`ATHENA/Documents/Scripts/run_python.sh`

This is the bridge ATHENA uses for **all Python code** — PINNs, DeepONets, and the JAX/NABLA-SciML (vRBA / SSBroyden) path. Open this file and modify the **Configuration** section so ATHENA can activate the correct Python/JAX environment.

**What to change:**
1.  **`CONDA_ACTIVATION_SCRIPT`**: The path to your system's `conda.sh` file. This is required to activate environments from within a script.
2.  **`CONDA_ENV_NAME`**: The name of the environment you created in Section 2 (e.g., `nabla_env`).

**Example `run_python.sh` configuration:**

```bash
#!/bin/bash

# --- Configuration ---
# 1. Path to your conda activation script (Check your ~/.bashrc or system paths)
# Example: "/gpfs/runtime/opt/anaconda/etc/profile.d/conda.sh"
CONDA_ACTIVATION_SCRIPT="/path/to/your/anaconda3/etc/profile.d/conda.sh"

# 2. Name of the JAX environment created in Section 2
CONDA_ENV_NAME="nabla_env"

# --- Safety ---
set -e

# ... (rest of script) ...
# The script will verify the paths, activate the environment, and execute the Python code.
```

# Configuring the Models

Every agent in ATHENA is provider-agnostic: it is created with a `model` name and a
`model_provider` string, and the framework dispatches to the right SDK. You change which
LLM an agent uses simply by editing those two arguments.

**Supported providers** (the `model_provider` value): `google`, `openai`, `anthropic`
(alias `claude`), `xai` (alias `grok`), and `ollama` (local). Each provider reads its key
from `crunch_llm/code/.env` (`GOOGLE_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`XAI_API_KEY`); `ollama` needs a local Ollama server instead of a key.

**Changing an agent's model.** Anywhere an agent is constructed (in a notebook or script),
set the two arguments:

```python
agent = StructuredAgent(
    model='gemini-3-flash-preview',   # any model id valid for the provider
    model_provider='google',          # google | openai | anthropic | grok | ollama
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

The defaults used in this repository are:

| Role | Default | Provider |
|---|---|---|
| Analyzer / agents | `gemini-3-flash-preview` | `google` |
| Embeddings | `models/gemini-embedding-001` (3072-dim) | `google` |

These run entirely through cloud APIs — **no local GPU or Ollama server is required.**


# Preparing the Run Scripts (pre-flight)

ATHENA does not execute generated code in-process — it shells out to two bridge
scripts that activate your environments and run the code. **Before running the
Tutorial, update these two scripts to match your machine.** If you cloned this repo
you already have them; you just need to edit the paths.

Both live in `Documents/Scripts/`:

| Script | Used for | Fields to set |
|---|---|---|
| `run_python.sh` | All Python: PINNs (**B**), DeepONets (**C**), JAX/NABLA-SciML | `CONDA_ACTIVATION_SCRIPT` (your `conda.sh`), `CONDA_ENV_NAME` (your JAX/PINN env) |
| `run_julia.sh` | Julia numerical solvers (**A**, SciC) | `JULIA_EXEC` (your `julia` binary), `PROJECT_ENV` (your Julia project dir) |

```bash
# Documents/Scripts/run_python.sh
CONDA_ACTIVATION_SCRIPT="/path/to/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV_NAME="NABLA_JAX"          # the env from Installation Section 2

# Documents/Scripts/run_julia.sh
JULIA_EXEC="/path/to/.juliaup/bin/julia"   # or $(which julia)
PROJECT_ENV="/path/to/Learning_Julia"      # the project from Installation Section 3
```

> **Note:** you no longer need to edit any absolute path inside `crunch_llm/code/utils.py`.
> `run_code()` now resolves both scripts **relative to the repo** (`<repo>/Documents/Scripts/`),
> so only the two script bodies above are machine-specific.

**Verify the bridges before launching a run.** Drop a one-line script in each language
and execute it through `run_code()`; both should exit cleanly and write a `.out` log:

```python
import os
import crunch_llm.code.utils as U

d = os.path.abspath("scratch/bridge_test"); os.makedirs(d, exist_ok=True)
open(f"{d}/toy.py", "w").write("import jax; print('python OK', jax.devices())")
open(f"{d}/toy.jl", "w").write('println("julia OK")')

for fn in ("toy.py", "toy.jl"):
    ok, log = U.run_code(d, fn, f"{d}/{fn}")
    print(fn, "->", ok); print(open(log).read())
```

A healthy result prints `python OK [CudaDevice(id=0)]` and `julia OK` with empty
`.err` files. If a run fails to launch, this is the first place to look (it is the
most common setup issue — see the path table above).


# Tutorial

The workflow has three stages: **(1) Ingestion** builds the reference-code knowledge bases,
**(2) Conceptualization** formalizes a problem, and **(3) Evaluation** runs the HENA loop.

## Step 1 — Ingestion

Ingestion reads the curated reference code under `Documents/codes/` and indexes it into
local [Chroma](https://www.trychroma.com/) vector databases. At run time the agents retrieve
from these databases to ground their solutions in working examples.

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

To set up a new scientific problem through an interactive dialogue with ATHENA, run the formalization script:

```bash
python Conceptualization_team/Formalize.py
```

This script initiates a conversational agent that will guide you through defining your problem statement.
* **Process:** A *Coordinator* agent interviews you about the physics, constraints, and objectives, proposes a hybrid strategy (e.g. *infer with a PINN, then verify numerically*), and writes a formal specification. A *Gatekeeper* agent then routes that specification into an ordered execution pipeline.
* **Output:** A **JSON file** `ATHENA_Request_<problem_name>.json`, containing the standardized `problem_name`, an ordered `pipeline` of stages (each tagged `A` = SciC, `B` = PINN, `C` = Operator), a `description`, and the full formalized spec under `user_request`.
* **Transcript:** The whole conversation is mirrored to `Conceptualization_team/Formalization_logs/formalization_<timestamp>.txt` for auditing.
* **Usage:** This JSON is formatted and ready to be fed directly into the evaluation notebooks.

**Non-interactive (batch) mode.** If you already have a complete problem description in a text file, you can skip the chat and generate the JSON in one shot:

```bash
python Conceptualization_team/Formalize.py --request-file my_problem.txt
```

### (Optional) Skipping the Conceptualization Team

The Conceptualization Team is a **convenience layer** — its only job is to turn a loosely-worded idea into a complete, well-posed specification. **It is entirely optional.** If you already know your problem precisely, you can bypass it and drive the evaluation directly: each example notebook simply reads a `user_question` string (see the *User Query* cell), so you can paste your equation there verbatim and run the HENA loop.

For the loop to succeed without the Coordinator's help, just make sure your `user_question` is **self-contained**. Include:

* **Governing equation(s)** — the PDE/ODE in explicit form.
* **Domain** — spatial (and temporal, if transient) extent.
* **Boundary conditions** and, for time-dependent problems, **initial conditions**.
* **Physics parameters** — all constants (e.g. `Re`, `ν`, `ϵ`), or which ones are *unknown* for an inverse problem.
* **Data** — path and format of any reference/observation data, plus how to sample it (e.g. number of sensors, noise level).
* **Targets, validation, and outputs** — what to infer, which error metrics to report, and the expected figure/file names.

In short: the Conceptualization Team *produces* this specification, but if you can write it yourself, you can feed the equation directly. The interactive step exists to help when the problem is still vague — not as a required gate.

## Step 3 — Evaluation

Examples are driven from their request JSONs via `run_athena.py`, which dispatches
each pipeline stage to the HENA loop:

```bash
python run_athena.py Conceptualization_team/ATHENA_Request_Rayleigh_Taylor_2D_Euler.json
```

(The original per-problem notebooks have been retired; their extracted run
transcripts are kept alongside each example as `*_log.txt`.)

### Reference data

Most examples are self-contained (they generate their own reference). Three
data-driven examples load an external file from `Documents/codes/Data/`:

| Example | File | Size | Where it comes from |
|---|---|---|---|
| `Allen_Cahn_PINN` | `allen_cahn.mat` | 1.8 MB | included in the repo |
| `Inverse_2D_NavierStokes_…_ReInference` | `ldc_Re500.npz` | 3.0 MB | included in the repo |
| `Viscous_Burgers_DeepONet` | `Burger_1000_5000.mat` | 374 MB | **download from [Google Drive](https://drive.google.com/file/d/1Tr6a22SiMpgUlzbIHiBFAt8DRGXQ30E6/view)** into `Documents/codes/Data/` (too large for git) |

The request JSONs reference these by absolute path under
`Documents/codes/Data/`; adjust if you relocate the repo.

To replicate the examples and benchmarks presented in the paper, navigate to the corresponding example folders.

**Example: 2D Inviscid Burgers (IB2D)**
**Notebook Location:** `ATHENA/Examples_analytical_exact/IB2D.ipynb`

**Running the Evaluation:**
1.  Open the notebook.
2.  **Critical:** Update all file paths in the notebook configuration cells to match your local machine's directory structure.
3.  Run the cells. This process will generate a `Results` folder containing:
    * Generated Code
    * Simulation Data
    * Run Reports for each iteration of the HENA loop.

**Resuming from Checkpoints:**
It is possible to start or continue experiments from earlier checkpoints saved in the `Results` folder. You can load the state of a previous run using the following code snippet inside the notebook:

```python
checkpoint_project_dir = '/path/to/your/Results/specific_run_folder'

# Load the checkpoint data
strategist_report, error_report, history_log_string, history, initial_code, filename = load_checkpoint_data(
    checkpoint_project_dir,
    load_code_only=False
)

print(error_report)
```

By providing the path to a checkpoint directory:
* **Continue:** You can use the last `strategist_report` and `error_report` to seamlessly continue the optimization loop.
* **New Run:** You can set `load_code_only=True` to extract just the `initial_code` and start a fresh run based on a previously successful script.

### A Note on GPU Requirements

The system is fully functional through cloud APIs and requires **no local GPU**. Historically,
some structured-output agents defaulted to local models (e.g. `gpt-oss:120b`) with significant
VRAM requirements; you can point any agent at a cloud provider instead — see
[**Configuring the Models**](#configuring-the-models) for how to change the `model` /
`model_provider` of any agent (for example, switching a structured agent to
`grok-4-1-fast-reasoning` on `grok`).

### A Note on Interface Choice
We prioritized using **Jupyter Notebooks** over standard Python scripts for the core workflow. The primary reason for this choice is that notebooks automatically preserve the entire **chat history** and state. This enables researchers to visually inspect the full, multi-turn interaction between the diverse agents (Strategist, Advisor, Critic, etc.) and audit the reasoning process that led to the final solution.

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
