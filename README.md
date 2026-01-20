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
    conda create -n athena_env python=3.10
    conda activate athena_env
    ```

3.  **Install PyTorch:**
    Install the version of PyTorch compatible with your CUDA version (see [pytorch.org](https://pytorch.org/get-started/locally/)).
    ```bash
    # Example for CUDA 12.1
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```

4.  **Install ATHENA (Editable Mode):**
    This installs the `crunch_llm` package and core dependencies.
    ```bash
    pip install -e .
    ```


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

### JAX Connection (`run_nabla.sh`)

Navigate to the script located at:
`ATHENA/Documents/Scripts/run_nabla.sh`

Open this file and modify the **Configuration** section so ATHENA can activate the correct JAX environment.

**What to change:**
1.  **`CONDA_ACTIVATION_SCRIPT`**: The path to your system's `conda.sh` file. This is required to activate environments from within a script.
2.  **`CONDA_ENV_NAME`**: The name of the environment you created in Step 2 (e.g., `nabla_env`).

**Example `run_nabla.sh` configuration:**

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

# Using Athena

### 1. Ingestion

**Standard Templates:**
The repository already comes pre-loaded with a comprehensive set of standard numerical templates located in:
`ATHENA/Documents/codes`

You do not need to run ingestion for these standard templates. They are ready to use.

**Adding Custom Knowledge:**
The following steps are only required if you wish to ingest **additional** domain-specific knowledge (such as new PINNs architectures, DeepONet variants, or custom solvers) into the vector database.

#### Example: Ingesting PINNs Code
**Notebook Location:** `ATHENA/Ingestion/PINNs/Codes.ipynb`

Open this notebook to process and index new code files. Update the configuration to point to your specific database path:

```python
# 1. Define the exact path where you want to save the database
my_database_path = "/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Documents/database_PIML"

COLLECTION_NAME = 'PINNs_Codes'

# 2. Pass that path to your function
vector_store = initialize_local_data(db_path=my_database_path, collection_name=COLLECTION_NAME)

analyzer_agent = BasicAgent(
                     model='gemini-2.5-flash',
                     model_provider='google',
                     system_prompt=analyzer_system_prompt_scic,
                     verbose=False)
```

**Best Practices for Ingestion:**
* **Cell Organization:** We strongly suggest organizing new code into distinct cells. This makes it easier for the agents to identify and retrieve specific functional blocks.
* **Documentation:** Adding detailed docstrings and comments within the code is strongly suggested to improve retrieval accuracy.

*Note: Follow a similar procedure for **Numerical Methods** or **DeepONets** by navigating to their respective ingestion notebooks.*

### 2. Setting up New Problems (Running the Conceptualization Team)

To set up a new scientific problem through an interactive dialogue with ATHENA, run the formalization script:

```bash
python ATHENA/Conceptualization_team/Formalize.py
```

This script initiates a conversational agent that will guide you through defining your problem statement.
* **Process:** You will discuss the physics, constraints, and objectives of your problem.
* **Output:** After the conversation concludes, the system will generate a **JSON file**.
* **Usage:** This JSON file contains a formalized, modified user request that is formatted and ready to be fed directly into the evaluation notebooks.

### 3. Evaluation

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

### 4. LLM Configuration & GPU Requirements

The system is fully functional with diverse APIs. By default, some agents (specifically those handling structured output) may be configured to use `gpt-oss:120b` or other local models, which can have significant GPU VRAM requirements.

If you do not have sufficient local GPU resources, or if you prefer a different provider, you can easily switch these agents to use other LLMs capable of structured output, such as **Grok**.

**Example: Switching to Grok**
Locate the agent initialization in the python scripts or notebooks and update the `model` and `model_provider` arguments as shown below:

```python
# Advisor Agent (Parser) - Switched to Grok
advisor_parser = StructuredAgent(
                     model='grok-4-1-fast-reasoning',
                     model_provider='grok',  # Explicitly set provider
                     verbose=False,
                     system_prompt=advisor_parser_system_prompt_ON,
                 )

# Register Agent (Parser) - Switched to Grok
register_agent_parser = StructuredAgent(
                     model='grok-4-1-fast-reasoning',
                     model_provider='grok',  # Explicitly set provider
                     system_prompt=register_agent_system_prompt_numerical,
                     verbose=False
                 )
```

### 5. Note on Interface Choice
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
