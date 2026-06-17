#!/usr/bin/env python3
"""
Standalone ingestion script for the ATHENA reference-code vector DB.

Replaces the three Jupyter notebooks:
    Ingestion/PINNs/Codes.ipynb
    Ingestion/Numerical_methods/Codes.ipynb
    Ingestion/DeepONet/Modules.ipynb

One script, selected by --class:
    pinns              walk a dir of .py/.jl, split on "# Cell N:", analyze each file once
    numerical_methods  same pipeline as pinns, different analyzer prompt / collection
    deeponet           single file, split on "# %% [markdown]", analyze each chunk

Usage:
    python Ingestion/ingest.py --class pinns
    python Ingestion/ingest.py --class numerical_methods --path Documents/codes/Numerical_Methods/Scripts
    python Ingestion/ingest.py --class deeponet --clear
    python Ingestion/ingest.py --class pinns --db-path /my/db --collection MyCodes --model gemini-2.5-flash

Defaults (db paths, collections, target paths, prompts) are bundled per class
in CONFIGS below; any of them can be overridden on the command line.
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path

# --- Make the project importable (repo root = parent of Ingestion/) ---
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_core.documents import Document

from crunch_llm.code.agents import BasicAgent
from crunch_llm.code.utils import initialize_local_data


# ============================================================================
# ANALYZER PROMPTS
# ============================================================================

PINN_PROMPT = """
You are an expert Scientific Machine Learning (SciML) code analyst. Your task is to analyze the provided code snippet (typically PyTorch, JAX, or TensorFlow) and extract key information into a structured JSON format. You must be highly specific and infer the values from the model definitions, loss functions, and training loops.

Your final output must be a single, valid JSON object. Do not add any explanatory text outside the JSON.

### Fields to Extract

1.  **`method`**: Identify the specific neural architecture or operator learning scheme.
2.  **`equation`**: Identify the governing PDE being solved.
3.  **`dimensionality`**: Specify spatial/temporal dimensions and the domain type.
4.  **`solver_or_optimizer`**: Identify the optimization strategy.
5.  **`enhancements`**: List specific SciML tricks found in the code (a JSON list).
6.  **`BCs`**: Identify how Boundary/Initial Conditions are enforced.
7.  **`problem`**: Describe the physical task (Forward vs Inverse).
8.  **`description`**: Provide a concise 1-2 sentence summary of the code's methodology.
"""

NM_PROMPT = """
You are an expert Scientific Computing and Numerical Methods code analyst. Your task is to analyze the provided code snippet (often Julia or Python) and extract key information into a structured JSON format. You must be highly specific and infer the values from the code structures, library calls (like Trixi.jl, DiffEq), and variable names.

Your final output must be a single, valid JSON object. Do not add any explanatory text outside the JSON.

### Fields to Extract

1.  **`method`**: Identify the specific spatial discretization or numerical scheme.
2.  **`equation`**: Identify the specific governing equations.
3.  **`dimensionality`**: Specify the spatial and temporal dimensions.
4.  **`solver_or_optimizer`**: Identify the time-integration scheme or the linear solver/optimizer.
5.  **`enhancements`**: List specific advanced numerical features found in the code (a JSON list).
6.  **`BCs`**: Identify the boundary conditions.
7.  **`problem`**: Describe the high-level physical scenario in one sentence.
8.  **`description`**: Provide a concise 1-2 sentence summary of the code's methodology.
"""

DEEPONET_PROMPT = """
You are an expert code analyst. Your task is to analyze the provided Python class definition and extract its name and high-level purpose into a structured JSON format.

Your final output must be a single, valid JSON object. Do not add any explanatory text outside the JSON.

### Fields to Extract

1.  **`function_name`**: The name of the Python class being defined.
2.  **`purpose`**: Infer the primary *categorical* purpose of this class. Use one of:
    * `Feature Expansion`
    * `Architecture Enhancement`
    * `KAN Layer`
    * `KKAN Architecture`
    * `Unknown`
"""


OPERATOR_PROMPT = """
You are an expert scientific machine learning code analyst. Your task is to analyze the provided code snippet and extract key information into a structured JSON format. You must be highly specific and infer the values from the code.

Your final output must be a single, valid JSON object. Do not add any explanatory text outside the JSON.

### Fields to Extract

1.  **`method`**: Infer the primary SciML method (e.g. `PINN`, `DeepONet`, `QR-DeepONet`, `FNO`, `PI-DeepONet`). Look for class names (`DeepONetModel`, `PINN`), loss structures (`loss_f`, `loss_bc` imply PINN), or branch/trunk inputs (imply DeepONet).
2.  **`equation`**: The specific PDE being solved (e.g. `Burgers`, `Allen-Cahn`, `Navier-Stokes`). If not specified, use `Unknown`.
3.  **`dimensionality`**: Problem dimensions (e.g. `1D`, `2D`, `1D+t`, `2D+t`, `3D+t`). If unclear, use `Unknown`.
4.  **`Optimizer`**: The optimization scheme. Only `optax.adam` -> `Adam`. `optax.adam` followed by `scipy.minimize`/`SSBroyden` -> `Adam + Hybrid SSBroyden`. Only `LBFGS`/`scipy.minimize` -> `LBFGS`.
5.  **`BCs`**: Boundary conditions (e.g. `Periodic`, `Dirichlet`, `Neumann`). If not present, use `N/A`.
6.  **`Problem`**: The high-level task in one sentence (e.g. `Solve the forward problem`, `Learn the operator that maps ICs to the full field`).
7.  **`Description`**: A concise 1-2 sentence summary of the code's overall objective.

### Example output
{
  "method": "DeepONet",
  "equation": "Unknown",
  "dimensionality": "1D+t",
  "Optimizer": "Adam",
  "BCs": "N/A",
  "Problem": "Learn the operator that maps initial conditions to the full spatio-temporal field.",
  "Description": "A DeepONet trained to map from a function space (ICs) to a solution space (full field) using an Adam optimizer."
}
"""


# ============================================================================
# PER-CLASS CONFIG
# ============================================================================

CONFIGS = {
    "pinns": {
        "prompt": PINN_PROMPT,
        "db_path": "Documents/database_PIML",
        "collection": "PINNs_Codes",
        "target": "Documents/codes/SciML/PINNs",
        "mode": "cells",
        "schema": "rich",
    },
    "numerical_methods": {
        "prompt": NM_PROMPT,
        "db_path": "Documents/database_NM",
        "collection": "Numerical_Methods_opt",
        "target": "Documents/codes/Numerical_Methods/Scripts",
        "mode": "cells",
        "schema": "rich",
    },
    "deeponet": {
        "prompt": DEEPONET_PROMPT,
        "db_path": "Documents/database_Modules_ollama",
        "collection": "Layers",
        "target": "Documents/codes/SciML/Modules/layers.py",
        "mode": "markdown",
        "schema": "simple",
    },
    "operators": {
        "prompt": OPERATOR_PROMPT,
        "db_path": "Documents/database_Operators",
        "collection": "Operators",
        "target": "Documents/codes/SciML/Operator_Learning/DeepONets/Files",
        "mode": "cells",
        "schema": "operator",
    },
}

LANGUAGE_MAP = {'.py': 'python', '.m': 'matlab', '.jl': 'julia', '.cpp': 'cpp', '.c': 'c'}
COMMENT_STYLES = {'python': '#', 'matlab': '%', 'julia': '#', 'c': '//', 'cpp': '//'}


# ============================================================================
# HELPERS
# ============================================================================

def parse_agent_json(raw: str) -> dict:
    """Strip markdown fences and parse a JSON object from the agent output."""
    clean = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)


def comment_for(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    language = LANGUAGE_MAP.get(ext, 'unknown')
    return COMMENT_STYLES.get(language, '#')


def gather_files(target: Path) -> list:
    """Return list of .py/.jl files: walk if dir, single file if a file."""
    if target.is_file():
        return [str(target)]
    files = []
    for root, _dirs, names in os.walk(target):
        for name in names:
            if name.endswith('.py') or name.endswith('.jl'):
                files.append(os.path.join(root, name))
    return sorted(files)


# ============================================================================
# MODE: "cells"  (pinns / numerical_methods)
#   walk dir -> analyze whole file once -> split on "# Cell N:" -> one doc/cell
# ============================================================================

def ingest_cells(file_path, code_content, agent, vector_store, schema="rich"):
    filename = os.path.basename(file_path)

    # --- 1. Analyze the whole file once ---
    print("  - Running Analyzer Agent...")
    agent_result = agent.run(code_content)
    if isinstance(agent_result, dict) and 'output' in agent_result:
        summary_string = agent_result['output'].strip()
    else:
        summary_string = str(agent_result).strip()

    try:
        data = parse_agent_json(summary_string)
        if schema == "operator":
            # keys the DeepONet retriever reads: equation, dimensionality,
            # optimizer_scheme, problem_type (mapped from Optimizer / Problem).
            analysis_result = {
                "method": str(data.get("method", "Unknown")),
                "equation": str(data.get("equation", "Unknown")),
                "dimensionality": str(data.get("dimensionality", "Unknown")),
                "optimizer_scheme": str(data.get("Optimizer", data.get("optimizer_scheme", "Unknown"))),
                "BCs": str(data.get("BCs", "Unknown")),
                "problem_type": str(data.get("Problem", data.get("problem_type", "Unknown"))),
                "description": str(data.get("Description", data.get("description", "Unknown"))),
                "filename": filename,
            }
        else:  # "rich" (pinns / numerical_methods)
            analysis_result = {
                "method": str(data.get("method", "Unknown")),
                "equation": str(data.get("equation", "Unknown")),
                "dimensionality": str(data.get("dimensionality", "Unknown")),
                "solver_or_optimizer": str(data.get("solver_or_optimizer", "Unknown")),
                "enhancements": ", ".join(data.get("enhancements", [])),
                "BCs": str(data.get("BCs", "Unknown")),
                "problem": str(data.get("problem", "Unknown")),
                "description": str(data.get("description", "Unknown")),
                "filename": filename,
            }
    except Exception as e:
        print(f"  - Warning: Could not parse JSON '{summary_string[:50]}...'. Error: {e}")
        analysis_result = {
            "method": "ParseError", "equation": "Unknown",
            "enhancements": "", "filename": filename,
        }

    print(f"  - Analysis: {analysis_result['method']} | {analysis_result['equation']}")

    # --- 2. Split on "# Cell N:" markers ---
    print("  - Splitting code into cells...")
    split_pattern = r'(^\s*[#\s]+Cell\s*\d+:.*)'
    parts = re.split(split_pattern, code_content, flags=re.MULTILINE)

    cleaned_cells = []
    if parts[0].strip():
        cleaned_cells.append(parts[0].strip())
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            cleaned_cells.append((parts[i] + parts[i + 1]).strip())
        else:
            cleaned_cells.append(parts[i].strip())

    num_cells = len(cleaned_cells)
    print(f"  - Found {num_cells} cells. Ingesting each as a 'part'...")

    # --- 3. Common metadata + header ---
    base_metadata = {
        'source_file': file_path,
        'filename': filename,
        'type': 'code-notebook',
        **analysis_result,
    }
    comment = comment_for(file_path)
    analysis_str = json.dumps(analysis_result, indent=2).replace('{', '').replace('}', '').replace('"', '')

    docs_to_add = []
    for i, cell_content in enumerate(cleaned_cells):
        part_num = i + 1

        cell_description = "File Prologue / Imports"
        if cell_content.startswith('# ## Cell'):
            cell_description = cell_content.split('\n', 1)[0].strip().lstrip('#').strip()
        else:
            for line in cell_content.split('\n'):
                if line.strip() and not line.strip().startswith('#'):
                    cell_description = line.strip()
                    break

        chunk_metadata = base_metadata.copy()
        chunk_metadata['part'] = f"{part_num}/{num_cells}"
        chunk_metadata['cell_description'] = cell_description

        header = f"""{comment} CODE ANALYSIS (PART {part_num} of {num_cells})
{comment} DESCRIPTION:
{comment}
{comment} {analysis_str.replace(chr(10), f'{chr(10)}{comment} ')}
{comment} =======================================================\n\n"""

        docs_to_add.append(Document(page_content=header + cell_content, metadata=chunk_metadata))

    if docs_to_add:
        vector_store.add_documents(docs_to_add)
    print(f"✅ Ingested: {filename} ({num_cells} chunks)")
    return {"analysis": analysis_result, "chunks_created": num_cells}


# ============================================================================
# MODE: "markdown"  (deeponet)
#   split on "# %% [markdown]" -> analyze each code chunk -> one doc/chunk
# ============================================================================

def ingest_markdown(file_path, code_content, agent, vector_store):
    filename = os.path.basename(file_path)

    print("  - Splitting code by '%% [markdown]' delimiter...")
    split_pattern = r'(^# %%\s*\[markdown\].*?)(?=# %% \[markdown\]|$)'
    parts = re.split(split_pattern, code_content, flags=re.DOTALL | re.MULTILINE)

    cleaned_chunks = []
    if parts[0].strip():
        cleaned_chunks.append({"description": "File Prologue / Imports",
                               "code": parts[0].strip(), "is_prologue": True})
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            markdown_content = parts[i].strip()
            code_part = parts[i + 1].strip()
            description = re.sub(r'^# %%\s*\[markdown\]\s*', '', markdown_content)
            if code_part:
                cleaned_chunks.append({"description": description,
                                       "code": code_part, "is_prologue": False})

    num_chunks = len(cleaned_chunks)
    print(f"  - Found {num_chunks} chunks (prologue + function blocks).")

    base_metadata = {'source_file': file_path, 'filename': filename,
                     'type': 'code-notebook-layer-doc'}
    comment = '#'
    docs_to_add = []

    for i, chunk in enumerate(cleaned_chunks):
        part_num = i + 1
        code_part = chunk["code"]
        description_part = chunk["description"]
        print(f"  - Processing chunk {part_num}/{num_chunks}...")

        if chunk["is_prologue"]:
            print("    - Skipping agent for prologue.")
            analysis_result = {"function_name": "Prologue", "purpose": "Imports and Setup"}
        else:
            summary_string = ""
            try:
                print("    - Running Analyzer Agent on code chunk...")
                if hasattr(agent, "clear_history"):
                    agent.clear_history()
                summary_string = agent.run(code_part)['output'].strip()
                if summary_string.startswith("```json"):
                    summary_string = re.sub(r"^```json\n|```$", "", summary_string, flags=re.MULTILINE)
                analysis_result = json.loads(summary_string)
                print(f"    - Analysis: {analysis_result}")
            except Exception as e:
                print(f"    - Warning: Could not parse agent JSON summary '{summary_string}'. Error: {e}")
                analysis_result = {"function_name": "Unknown", "purpose": "Unknown"}

        chunk_metadata = base_metadata.copy()
        chunk_metadata['part'] = f"{part_num}/{num_chunks}"
        chunk_metadata['cell_description'] = description_part.split('\n', 1)[0].lstrip('# ').strip()
        chunk_metadata.update(analysis_result)

        header = f"""{comment} CODE ANALYSIS (PART {part_num} of {num_chunks})
{comment} FUNCTION: {analysis_result.get('function_name', 'N/A')}
{comment} PURPOSE: {analysis_result.get('purpose', 'N/A')}
{comment} =======================================================
{comment} =======================================================\n\n"""

        docs_to_add.append(Document(page_content=header + code_part, metadata=chunk_metadata))

    if docs_to_add:
        vector_store.add_documents(docs_to_add)
    print(f"✅ Ingested: {filename} ({num_chunks} chunks)")
    return {"chunks_created": num_chunks}


# ============================================================================
# COLLECTION MAINTENANCE
# ============================================================================

def clear_collection(db_path, collection_name):
    import chromadb
    print(f"Clearing collection '{collection_name}' at {db_path} ...")
    try:
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name=collection_name)
        all_ids = collection.get(include=[])['ids']
        if not all_ids:
            print(f"  Info: collection '{collection_name}' already empty.")
        else:
            collection.delete(ids=all_ids)
            print(f"  ✅ Deleted {len(all_ids)} elements from '{collection_name}'.")
    except Exception as e:
        print(f"  Info: nothing cleared ({e}).")


# ============================================================================
# MAIN
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="Ingest reference code into the ATHENA vector DB.")
    p.add_argument("--class", dest="cls", required=True, choices=list(CONFIGS.keys()),
                   help="Which knowledge base to ingest into.")
    p.add_argument("--path", help="Override target file/dir of code to ingest.")
    p.add_argument("--db-path", help="Override Chroma DB path.")
    p.add_argument("--collection", help="Override collection name.")
    p.add_argument("--model", default="gemini-3-flash-preview", help="Analyzer LLM model.")
    p.add_argument("--provider", default="google", help="Analyzer LLM provider.")
    p.add_argument("--embed-provider", default="google", help="Embeddings provider (API by default, no local ollama).")
    p.add_argument("--embed-model", default="models/gemini-embedding-001", help="Embeddings model.")
    p.add_argument("--clear", action="store_true", help="Clear the collection before ingesting.")
    p.add_argument("--clear-only", action="store_true", help="Only clear the collection, do not ingest.")
    args = p.parse_args()

    cfg = CONFIGS[args.cls]
    db_path = args.db_path or str((REPO_ROOT / cfg["db_path"]).resolve())
    collection = args.collection or cfg["collection"]
    target = Path(args.path) if args.path else (REPO_ROOT / cfg["target"])
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()

    print(f"Class       : {args.cls}")
    print(f"Target      : {target}")
    print(f"DB path     : {db_path}")
    print(f"Collection  : {collection}")
    print(f"Analyzer    : {args.model} ({args.provider})")
    print(f"Embeddings  : {args.embed_provider} / {args.embed_model or 'provider-default'}")

    if args.clear or args.clear_only:
        clear_collection(db_path, collection)
        if args.clear_only:
            return

    if not target.exists():
        sys.exit(f"ERROR: target path does not exist: {target}")

    vector_store = initialize_local_data(
        db_path=db_path,
        collection_name=collection,
        provider=args.embed_provider,
        model_name=args.embed_model,
    )

    agent = BasicAgent(model=args.model, model_provider=args.provider,
                       system_prompt=cfg["prompt"], verbose=False)

    if cfg["mode"] == "cells":
        def handler(fp, code, ag, vs): return ingest_cells(fp, code, ag, vs, schema=cfg["schema"])
    else:
        handler = ingest_markdown

    files = gather_files(target)
    if not files:
        sys.exit(f"ERROR: no .py/.jl files found under {target}")
    print(f"\nFound {len(files)} file(s) to ingest.\n")

    all_results = {}
    for fp in files:
        try:
            # BasicAgent is aliased to memoryAgent (stateful), so reset history
            # before each file/chunk batch to avoid accumulating prior context.
            if hasattr(agent, "clear_history"):
                agent.clear_history()
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                code_content = f.read()
            print(f"Processing: {os.path.basename(fp)} ({code_content.count(chr(10)) + 1} lines)...")
            all_results[fp] = handler(fp, code_content, agent, vector_store)
        except Exception as e:
            print(f"--- UNHANDLED ERROR analyzing {os.path.basename(fp)}: {e} ---")
            all_results[fp] = {"error": str(e)}

    out_file = REPO_ROOT / f"Ingestion/ingest_report_{args.cls}.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nAnalysis complete! ✅ Report saved to {out_file}")


if __name__ == "__main__":
    main()
