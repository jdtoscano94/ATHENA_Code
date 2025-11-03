import os
import sys
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.tools import tool
from typing import Optional,Dict, List, Tuple 
import subprocess
import shutil # 
import re
from IPython.display import display, Image
from datetime import datetime 
from langchain.agents import AgentExecutor
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser # Import the JSON parser
from langchain_core.documents import Document
import chromadb
from pathlib import Path # Import the Path object
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
#Utils
from crunch_llm.code.utils import *
# --- Corrected Team Function ---
def retrieval_team_PINN(
    user_question: str,
    retriever_agent: 'Retriever_Agent',
    max_iterations: int = 5
) -> Tuple[str, str, bool, str]:
    is_approval = False
    extraction_iterations = 0
    file_content = ''
    filename = ''
    retriever_output = '' # Will store the agent's final justification
    # A simple prompt for retrying
    retry_prompt = "That was not correct. Please try again. Analyze the request, call the retrieve tool, and provide the name of the best code template."

    while (is_approval == False and extraction_iterations < max_iterations):
        print(f"--- Retrieval Iteration: {extraction_iterations} ---")
        
        # --- 1. Run Retriever ---
        if extraction_iterations == 0:
            response = retriever_agent.run(user_question)
        else :
            response = retriever_agent.run(retry_prompt) # Use the simple retry
        
        print(f"Retriever Message:\n{response['output']}")
        target_filename = extract_filename(response['output'])
        
        if not target_filename:
            print("Warning: Retriever did not mention a filename. Retrying...")
            extraction_iterations += 1
            continue # Go to the next loop iteration
        # --- 2. Find Match and Validate ---
        file_content = ''
        filename = ''
        found_match = False
        
        for extracted_code in response['original_code']:
            my_document = extracted_code
            doc_filename = my_document.metadata.get('filename') # Use .get() for safety
            
            if doc_filename == target_filename:
                print(f"✅ Match found: Retriever successfully retrieved '{doc_filename}'")
                file_content = my_document.page_content
                filename = doc_filename
                is_approval = True    # This exits the outer `while` loop
                found_match = True    # This prevents the `if not found_match` block
                retriever_output = response['output'] # Save the successful justification
                break # Exit the inner `for` loop

        if not found_match:
             print(f"Warning: Retriever mentioned '{target_filename}', but it was NOT in the retrieved documents list. Retrying...")
             # The loop will retry using the generic retry_prompt

        extraction_iterations += 1

    if not is_approval:
        print(f"❌ Retrieval loop failed after {extraction_iterations} attempts.")
    else:
        print(f"✅ Retrieval loop approved file: {filename}")

    # Return the retriever's successful justification to be passed to the advisor
    return file_content, filename, is_approval, retriever_output

# Utils Team

def parse_planner_output(planner_output: str) -> Tuple[Dict[str, str], str]:
    tasks_dict: Dict[str, str] = {}
    dimensionality_note: str = ""
    block_regex = r'### (.*?)\n(.*?)(?=(?:### |\Z))'
    for match in re.finditer(block_regex, planner_output, re.DOTALL):
        header = match.group(1).strip()
        content = match.group(2).strip()
        
        if header.startswith("Dimensionality Check"):
            dimensionality_note = content
                
        elif header.startswith("Cell"):
            try:
                cell_id_match = re.search(r'^(Cell \d+)', header)
                if not cell_id_match:
                    continue #
                cell_id = cell_id_match.group(1).strip()
                
                tasks_dict[cell_id] = content 
            except Exception as e:
                print(f"Warning: Failed to parse block with header '{header}'. Error: {e}")
                
    return tasks_dict, dimensionality_note

def code_refinement_team_PINN(
    user_question: str,
    initial_code: str,
    filename: str,
    error_report: str,
    strategist_report:str,
    strategist_agent: 'aRAGAgent',
    planner_agent: 'aRAGAgent',
    patcher_agent: 'StructuredAgent',
    debug_approval:bool
    ):
    print("--- 1. Cleaning Code ---")
    clean_code = clean_code_comments(initial_code) 
    if debug_approval:
        print("--- 2. Calling Strategist Agent ---")
        prompt_strategist = (
            f"[USER_QUESTION]:\n{user_question}\n\n"
            f"[ADVISOR_REPORT]:\n{error_report}\n"
        )
         # The strategist is called *once* to get the high-level plan
        response_strategist = strategist_agent.run(prompt_strategist)
        strategist_report = response_strategist['output']
        print(f"--- Strategy Report Received ---\n{strategist_report}\n")

    print("--- 3. Starting Sequential, Stateful Refinement ---")
    split_code = split_code_into_cells(clean_code)
    refined_code_dict = {} # Use a new dict to store the output
    language = get_language_from_filename(filename)
    comment_char = get_comment_char(language)
    # Clear memory:
    planner_agent.clear_history()
    prompt_planner = (
        f"[MODE]:\n STUDY_PHASE\n\n" 
        f"[USER_REQUEST]:\n{user_question}\n\n"
        f"[FULL_CODE]:\n{clean_code}\n\n"
        f"[ERROR_REPORT]:\n{error_report}\n\n"
    )
    response=planner_agent.run(prompt_planner)
    print('Planner Study Phase:',response['output'])
    for cell_id in split_code.keys():
        print(f"\n--- Processing: {cell_id} ---")
        cell_content=split_code[cell_id]
        numbered_cell_code = add_line_numbers_to_code(cell_content, filename)
        prompt_planner = (
            f"[MODE]:\nREFACTORING PHASE\n\n" 
            f"[USER_REQUEST]:\n{user_question}\n\n"
            f"[OVERALL_STRATEGY_REPORT]:\n{strategist_report}\n\n"
            f"[CURRENT_CELL_CODE]:\n{numbered_cell_code}\n\n"
            f"[PREVIOUS_ERROR_REPORT]:\n{error_report}\n\n"
            f"[YOUR_TASK]:\n"
            "1. Analyze the [CURRENT_CELL_CODE] based on the [OVERALL_STRATEGY_REPORT].\n"
            "2. Review your chat history (your memory) to see what changes you've already made (e.g., variable renames, new functions). \n"
            "3. Proceed with the required modifications."
        )
        response_planner=planner_agent.run(prompt_planner)   
        planner_instructions = response_planner['output']
        print(f"  - [Planner]: Instructions received:\n{planner_instructions}\n")
        prompt_patcher = f"Instruction:\n{planner_instructions}\nCode:\n{numbered_cell_code}"
        response_patcher= patcher_agent.run(prompt_patcher)
        patched_numbered_code, patch_errors = apply_patch(
            numbered_cell_code.strip(), response_patcher
        )
        if patch_errors:
            print(f"  - ❌ ERROR applying patch to {cell_id}: {patch_errors}")
            refined_code_dict[cell_id] = cell_content
            continue

        print(f"  - [Patcher]: Patch applied successfully.")
        patched_code_no_lines = remove_line_number_tokens(patched_numbered_code)
        # Add modification markers for the *next* iteration's error report
        marked_code = (
            f"{patched_code_no_lines.strip()}\n"
        )
        refined_code_dict[cell_id] = marked_code
    print("\n--- 4. Reconstructing Code ---")
    refined_code_full = reconstruct_code_from_cells(refined_code_dict)
    return refined_code_full,strategist_report

def filing_team_PINN(user_question,filename,final_clean_code,naming_agent,requires_name=True,new_code_details='Test',base_results_dir='/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Results/PINN_codes/'):
    code_language=get_language_from_filename(filename)
    if requires_name:
        new_code_details = naming_agent.run('Original request:' + user_question)
        new_code_details['description']=user_question
    print(f"Proposed name:{new_code_details['filename']}\Question :{new_code_details['description']}\n")
    project_dir,final_filename,final_filepath=create_folder_and_save_code(new_code_details, 
                                                                          final_clean_code,
                                                                          code_language,
                                                                          base_results_dir=base_results_dir,
                                                                          add_description=False)
    return project_dir,final_filename,final_filepath,new_code_details


def debugging_team_PINN(
    user_question: str,
    final_filepath: str,
    log_file_name: str,  
    debugger_agent: 'StructuredAgent'
) -> Tuple[str, str, bool, str]:
    refined_code, found_code = read_files(final_filepath)
    # 1. Read BOTH logs
    log_content, found_log = read_files(log_file_name) 
    log_error_path = log_file_name[:-3] + 'err'
    log_error, found_error = read_files(log_error_path) 
    # 2. Smart Truncation Logic (Language-Aware)
    log_for_agent_err = ""
    if found_error:
        code_language = 'unknown'
        if final_filepath.endswith('.py'):
            code_language = 'python'
        elif final_filepath.endswith('.jl'):
            code_language = 'julia'
        traceback_header = ""
        if code_language == 'python':
            traceback_header = "Traceback (most recent call last):"
        elif code_language == 'julia':
            traceback_header = "ERROR:"
        traceback_index = -1
        if traceback_header:
            traceback_index = log_error.find(traceback_header)
        if traceback_index != -1:
            print(f"--- [Found {code_language} Error. Sending snippet to Debugger] ---")
            log_for_agent_err = log_error[traceback_index:]
        else:
            print(f"--- [No Traceback. Sending last 20 lines to Debugger] ---")
            log_lines = log_error.splitlines()
            log_for_agent_err = "\n".join(log_lines[-20:])
    full_log_for_agent = (f"--- STDOUT LOG (Full) ---\n{log_content}\n\n"
                          f"--- STDERR LOG (Truncated) ---\n{log_for_agent_err}\n")
    status = ""
    report = ""
    
    if found_code and (found_log or found_error):
        print("--- Running Debugging Team ---")
        debugger_input = (f"User Request:\n{user_question}\n\n"
                          f"Log File (Combined & Truncated):\n{full_log_for_agent}\n\n"
                          f"Final Code:\n{refined_code}")
        try:
            debugger_report = debugger_agent.run(debugger_input)
            status = debugger_report['status']
            report = debugger_report['analysis']
        except Exception as e:
            print(f"--- Debugger Agent Failed to Run or Parse: {e} ---")
            status = 'REJECTED_FATAL'
            report = (f"Debugger agent itself failed (e.g., bad JSON or API error): {e}. "
                      "This is a system-level fault, not a code bug.")
    else:
        status = 'REJECTED_FATAL'
        report = ('We could not read or load your files (code or logs). '
                  'This is a system-level error with the filing/reading process.')
    
    print(f"Debugger Status: {status}")
    print(f"Debugger Analysis: {report}")
    
    debug_approval = check_approval(status)
    return status, report, debug_approval, full_log_for_agent,log_content
def advisor_team_PINN(
    user_question: str,
    project_dir: str,
    log_content: str,
    advisor_agent: 'aRAGAgent_multimodal',
    advisor_parser: 'StructuredAgent'
) -> Tuple[str, bool]:
    print("--- Running Advisor Team (Fail-Fast Strategy) ---")

    # --- Step 1: Create human-readable report (for user's benefit) ---
    _, _ = create_consolidated_report_chronological(project_dir)
    print("Human-readable PDF report created.")

    # --- Step 2: Find all plots for the agent ---
    all_images = find_plots(project_dir, False)
    
    # --- Step 3: Find the summary plot path ---
    summary_plot_path = next(
        (path for path in all_images if os.path.basename(path) == 'summary_all.png'), 
        None
    )

    # --- Step 4: Main logic (if/else) based on finding the plot ---
    if summary_plot_path:
        # --- PATH A: 'summary_all.png' was FOUND ---
        print(f"✅ Found 'summary_all.png' at: {summary_plot_path}")
        print("Proceeding to multimodal analysis...")
        
        # 1. Encode the image (will raise FileNotFoundError or PIL errors)
        print(f"Encoding image from: {summary_plot_path}")
        image_data_url = encode_image_to_data_url(summary_plot_path)
        print(f"Image encoded. Base64 length: {len(image_data_url)}")

        # 2. Crop the log file to the last 20 lines
        log_lines = log_content.strip().split('\n')
        log_tail = '\n'.join(log_lines[-20:])

        # 3. Create the text prompt
        text_prompt_content = f"""
[USER REQUEST]:
{user_question}

[STDOUT LOG (TAIL)]:
Here are the final 20 lines of the log. Find the FINAL error here.
---
{log_tail}
---

[SUMMARY PLOT]:
This is the 'summary_all.png' plot for the run.

[YOUR TASK]:
As per your system instructions:
1. Analyze the [SUMMARY PLOT] for the qualitative story.
2. Find the exact 'Minimum achieved error:' in the [STDOUT LOG (TAIL)].
3. Provide your full 'Status', 'Analysis', and 'Recommendation' report.
"""
        # 4. Create the final multimodal input list
        multimodal_input = [
            {"type": "text", "text": text_prompt_content},
            {"type": "image_url", "image_url": {"url": image_data_url}}
        ]

        # 5. Run the multimodal agent (will raise API errors)
        print("Sending multimodal input to AdvisorAgent...")
        response_dict = advisor_agent.run(multimodal_input)
        markdown_report = response_dict.get("output")
        
        if not markdown_report:
            # Fail-fast if the agent returns an empty response
            raise ValueError("AdvisorAgent (multimodal) ran but returned no 'output'.")
            
        print("--- Advisor Markdown Report Received ---")
        print(markdown_report)

        # 6. Parse the agent's report (will raise parsing/KeyErrors)
        print("Parsing report with AdvisorParser...")
        json_response = advisor_parser.run(markdown_report)
        
        status = json_response.get("status", "REJECTED_FOR_REFINEMENT")
        analysis = json_response.get("analysis", "Parser failed to extract 'analysis'.")
        recommendation = json_response.get("recommendations_for_planner", "Parser failed to extract 'recommendations_for_planner'.")
        
        is_approved = (status.upper() == "APPROVED")
        error_report = f"Advisor Analysis:\n{analysis}\n\nRecommendation:\n{recommendation}"
            
    else:
        # --- PATH B: 'summary_all.png' was NOT FOUND ---
        # This is a planned failure, not a crash.
        print("❌ Advisor REJECTED: Required 'summary_all.png' not found.")
        
        analysis = ("REJECTED: Debugger approved execution, but the required summary plot 'summary_all.png' "
                    "was not found. Scientific validation is impossible without this plot.")
        
        recommendation = ("The code ran but did not produce 'summary_all.png'. This is a "
                          "fatal scientific error. Please modify the code (likely in the "
                          "final cells) to generate and save a single 'summary_all.png' "
                          "image. This image must contain: "
                          "1. (Top Row) Loss and L2 Error histories (log-scale). "
                          "2. (Bottom Row) Reference, Prediction, and Pointwise Error plots.")
        
        error_report = f"Advisor Analysis:\n{analysis}\n\nRecommendation:\n{recommendation}"
        is_approved = False

    # --- Step 5: Return the final status and report ---
    if is_approved:
        print("✅ Advisor APPROVED the scientific result.")
    else:
        print(f"❌ Advisor REJECTED. Sending report to Planner.")
        
    return error_report, is_approved

# Promps
naming_agent_system_prompt_PINN = """
You are an expert file naming agent for scientific computing projects. Your sole task is to read the provided **user request** and generate a single JSON object containing a descriptive and programmatic filename.

You must derive the filename *ONLY* from the user request.

---
**Output Format:**

Your output MUST be **only** the valid JSON object and nothing else. It must adhere to the following structure:

```json
{{
  "filename": "Descriptive_Filename_Here"
}}
```

---
**Decision Logic:**

You MUST analyze the user request to determine the problem type and build the name based on the following priority:

**1. Inverse Problems (Highest Priority)**
* If the request is to "find," "discover," "infer," or "learn" a parameter (e.g., "find the diffusion coefficient $D$"), the filename MUST start with `Inverse_`.
* **Format:** `Inverse_Equation_Parameter` (e.g., `Inverse_Burgers_Find_D`).
* If it's a hidden field (e.g., "infer $h(t)$", "discover the forcing $f(x,t)$"), use a similar format.
* **Format:** `Inverse_Equation_HiddenField_h_t` or `Inverse_Equation_HiddenForcing_f`.

**2. Forward Problems (Default Case)**
* If it's a standard forward solve (the default), build the name from these components in order:
* **`Equation`**: The name of the PDE (e.g., `AllenCahn`, `FisherKPP`).
* **`BCs`**: The boundary conditions (e.g., `Periodic`, `Dirichlet`).
* **`ICs`**: A brief summary of the initial condition (e.g., `Sin2pix_ICs`, `Gaussian_ICs`).

---
**Formatting Rules:**

* Use `PascalCase` or `Snake_Case` with underscores `_` separating components.
* Be descriptive but programmatic.
* DO NOT include a file extension (like `.py`).

---
**Example 1: Forward Problem**

*User Input (`{input}`):*
```
"Lets solve the Fisher-Kpp equation u_t=0.1u_xx+u(1-u)  with  IC=0.1sin(2pi x)+0.5 
the domain goes from 0 to 1 in time and from 0 to 2 in x; periodic boundary conditions"
```

*Your Required JSON Output:*
```json
{{
  "filename": "FisherKPP_Periodic_Sin2pix"
}}
```

---
**Example 2: Inverse Problem**

*User Input (`{input}`):*
```
"I have data for the Burgers equation. Can you help me find the viscosity parameter nu? The IC is a simple sine wave and BCs are periodic."
```

*Your Required JSON Output:*
```json
{{
  "filename": "Inverse_Burgers_Periodic_Find_nu"
}}
```

---
**Example 3: Inverse (Hidden Field)**

*User Input (`{input}`):*
```
"Let's solve the heat equation $u_t = u_xx + f(x,t)$. I have observations of $u$ and want to discover the hidden forcing term $f(x,t)$. Use Dirichlet BCs."
```

*Your Required JSON Output:*
```json
{{
  "filename": "Inverse_Heat_Dirichlet_HiddenForcing_f"
}}
```
"""


Retriever_system_prompt_PINN = """
You are an expert retriever for a scientific code library. Your task is to find the best code template for a user's request and report your findings.

**AGENT WORKFLOW:**

1.  **Analyze Request**: Read the user's request (e.g., "solve 1D Burgers' equation...").
2.  **Formulate Query**: Formulate a query to find the best methodological template.
    * For 1D/2D PINN problems (forward or inverse), your query MUST be: "PINN hybrid optimizer SSBroyden"
3.  **Call Tool**: Call the `retrieve` tool with your query.
4.  **Analyze Tool Output**: The tool will return documents and a summary (e.g., "Found 1 matching file: AllenCahn_PINN.py..."). You MUST get the filename from this summary.
5.  **CRITICAL: Report Result**: Your final response to the user MUST follow this format EXACTLY.

---
**CRITICAL OUTPUT FORMAT**

Your response MUST start with the filename on the very first line, inside square brackets.
Follow this with a brief justification.

**Example 1: Success**
```
[AllenCahn_PINN.py]

**Justification**
I've selected the `AllenCahn_PINN.py` template. This is our standard hybrid Adam+SSBroyden solver, which is the correct methodological choice for the user's 1D Burgers' equation problem.
```

**Example 2: Not Found**
```
[NOT_FOUND]

**Justification**
I could not find a suitable template. The query for "PINN hybrid optimizer SSBroyden" returned no results.
```
---

You MUST NOT add any other text or preamble before the `[filename.py]` or `[NOT_FOUND]` bracketed line. Your response must start with `[` on the first character of the first line.
"""
patcher_agent_system_prompt_PINN = """
You are an expert JAX and SciML programmer. Your task is to act as a "Patcher" for a single cell of a larger code template.

You will be called sequentially in a loop, one cell at a time. Your goal is to apply a specific modification to the cell you are given, while maintaining consistency with the patches you have made to *previous* cells.

**CRITICAL: YOUR INPUT**
Your input is NOT just a simple question. The user prompt will be a structured block of text containing:

1.  **CURRENT TASK:** Specific instructions for *this cell only* (e.G., "Replace the Allen-Cahn residual...")
    * **CRITICAL:** This input might *also* be the literal string `NO_CHANGE`.
2.  **ANNOTATED CODE:** The specific code cell you must modify, with `# LINE_N` comments.

**YOUR TASK & OUTPUT FORMAT**
Your **only** output must be a single, valid JSON object containing a list of changes. Do NOT add any other text, analysis, or explanations.

**Rule 1: The "NO_CHANGE" Signal (Highest Priority)**
* First, check the `CURRENT TASK`. If it contains **only** the exact string `NO_CHANGE`, this is a signal that no patch is required.
* Your output in this case **MUST** be an empty list of changes:
    ```json
    {{
      "changes": []
    }}
    ```

**Rule 2: Generating the Patch (Default Case)**
* If `CURRENT TASK` is *not* `NO_CHANGE`, you must:
* **Analyze:** Read the `CURRENT TASK` as your primary guide.
* **Use History:** You are stateful. Refer to your `chat_history` to remember the changes you've made to previous cells (e.g., "In Cell 5, I changed the BCs...") and ensure your new patch is consistent.
* **Patch:** Generate a JSON patch to apply the changes.

**CRITICAL CONSTRAINTS (DO NOT BREAK)**
1.  **DO NOT change variable names that connect cells.** The template relies on "global" variables passed between cells (e.g., `params`, `apply_fn`, `train_data`, `tc`, `xc`, `unflatten_func`). You must not rename them, as it will break downstream cells.
2.  **BE CONSISTENT:** If you defined a new function `Burgers_Residual` in a previous cell, you must use that exact name in a later cell.
3.  **USE LINE NUMBERS:** Your JSON patch *must* use the line numbers from the `# LINE_N` comments.
4.  **CLEAN CODE:** The `new_code_block` in your JSON must be clean Python. Do *not* include line number comments (`# LINE_N`) in it. Use `\n` for newlines.
5.  **IDENTATION:** When you write new code always pay attention to the identation level.
---
**JSON OUTPUT FORMAT**
```json
{{
  "changes": [
    {{
      "start_line": <int>,
      "end_line": <int>,
      "action": "<replace | delete | insert_after>",
      "new_code_block": "<string of new code>"
    }}
  ]
}}
```
---

**EXAMPLE 1: Modifying a Cell (No Error)**
```
CURRENT TASK:
* Action: Replace Allen-Cahn residual with the Fisher-KPP equation.
* Details: This is a forward problem. Create a new function `FisherKPP_Residual` that computes u, ut, and uxx. The function must return `ut - D * uxx - rho * u * (1.0 - u)`. Replace the final line to use this new function.

ANNOTATED CODE:
def Allen_Cahn(params, t, x, apply_fn): # LINE_95
    u = apply_fn(params, t, x) # LINE_96
    v_t = jnp.ones_like(t) # LINE_97
    v_x = jnp.ones_like(x) # LINE_98
    ut = jvp(lambda t_val: apply_fn(params, t_val, x), (t,), (v_t,))[1] # LINE_99
    uxx = hvp_fwdfwd(lambda x_val: apply_fn(params, t, x_val), (x,), (v_x,)) # LINE_100
    return ut - 0.0001 * uxx + 5.0 * u ** 3 - 5.0 * u # LINE_101
# LINE_102
PDE_residuals=Allen_Cahn # LINE_103
```

*Your Required JSON Output:*
```json
{{
  "changes": [
    {{
      "start_line": 95,
      "end_line": 101,
      "action": "replace",
      "new_code_block": "def FisherKPP_Residual(params, t, x, apply_fn):\n    u = apply_fn(params, t, x)\n    v_t = jnp.ones_like(t)\n    v_x = jnp.ones_like(x)\n\n    # Compute derivatives\n    ut = jvp(lambda t_val: apply_fn(params, t_val, x), (t,), (v_t,))[1]\n    uxx = hvp_fwdfwd(lambda x_val: apply_fn(params, t, x_val), (x,), (v_x,))\n\n    # Get parameters from GOAL\n    D = 0.1\n    rho = 1.0\n    return ut - D * uxx - rho * u * (1.0 - u)"
    }},
    {{
      "start_line": 103,
      "end_line": 103,
      "action": "replace",
      "new_code_block": "PDE_residuals=FisherKPP_Residual"
    }}
  ]
}}
```
---

**EXAMPLE 2: The "NO_CHANGE" Signal**

```
CURRENT TASK:
NO_CHANGE

ANNOTATED CODE:
import jax # LINE_1
import jax.numpy as jnp # LINE_2
import optax # LINE_3
import os # LINE_4
```

*Your Required JSON Output:*
```json
{{
  "changes": []
}}
```
"""



debugger_system_prompt_PINN = """
You are an expert Debugger for scientific computing, acting as a "gate" or "triage expert." Your sole job is to analyze a completed code execution and determine if it has passed or failed, and why.

You will be given three inputs:
1.  **User Request**: The original high-level requirements (e.g., "solve Burgers equation... data is in 'burgers.mat'").
2.  **Log File**: The stdout/stderr from the code's execution.
3.  **Final Code**: The complete multi-cell PINN script that was run (e.g., AllenCahn_PINN.py).

Your task is to analyze these inputs and return a single JSON object with two keys: "status" and "analysis".

---
**Analysis Workflow (Strict Priority):**

**Part 1: Check for Unrecoverable Environment/User Errors (Highest Priority)**
- First, analyze the **Log File** for errors that the **PlannerAgent CANNOT fix**.
- Examples: `ModuleNotFoundError`, `ImportError`, `PermissionError`, or a conda activation failure.
- If you find one of these, you must **STOP**. The loop cannot continue.
- In this case, your output MUST be:
    - "status": "STOP_UNRECOVERABLE"
    - "analysis": A clear message **for the human user** explaining what is wrong with their environment or input. (e.g., "The log shows a 'ModuleNotFoundError' for 'deepxde'. Please ensure this package is installed in your environment.")

---
**Part 1b: Special Triage for 'FileNotFoundError'**
- You will *only* perform this check if you find a `FileNotFoundError` in the **Log File**.
- This error is ambiguous. You must determine if it's a **User Error** (unfixable) or a **Planner Error** (fixable).

1.  **Extract Paths:**
    * From the **Log File**, find the exact file path that failed (e.g., `path_from_log`).
    * From the **User Request**, find the original file path the user provided (e.g., `path_from_user`).

2.  **Compare and Decide:**
    * **Case A (Paths Match):** `path_from_log` is identical to `path_from_user`.
        * This means the Planner *correctly* used the user's path, but the file is *actually* missing. This is a **User Error**.
        * **Output:**
            - "status": "STOP_UNRECOVERABLE"
            - "analysis": A message for the user: "The file specified in your request ('[path_from_user]') was not found. This is a user-side error. Please check that the file exists at this exact path."

    * **Case B (Paths DO NOT Match):** `path_from_log` is different from `path_from_user`.
        * This means the Planner made a mistake and *mis-copied* or *hallucinated* a path in the code. This is a **Planner Error**.
        * **Output:**
            - "status": "REJECTED_FATAL"
            - "analysis": A bug report for the Planner: "A 'FileNotFoundError' occurred. The Planner *incorrectly* used the path '[path_from_log]' in the code. The original User Request specified '[path_from_user]'. This is a fixable bug. The Planner must correct the file path in the code to *exactly* match the user's request."

---
**Part 2: Check for Other Fatal Runtime Errors (Fixable by Planner)**
- You will *only* perform this check if **Part 1 and 1b passed** (i.e., it's not an unrecoverable error).
- If you find any other *code logic* error (e.g., `exit_code != 0`, `TypeError:`, `AttributeError:`, `DimensionMismatch`), your output MUST be:
    - "status": "REJECTED_FATAL"
    - "analysis": A detailed **Bug Analysis** for the Planner. (e.g., "The log shows a 'TypeError' in Cell 11. This often means the function signature in Cell 7 does not match the arguments passed in Cell 11.")

---
**Part 3: Check for Cosmetic & Logical Mismatches**
- You will *only* perform this check if **Part 1, 1b, and 2 passed** (the log is clean, `exit_code == 0`).
- Now, compare the **User Request** against the **Final Code**.
- Check for mismatches:
    1.  **Explicit Mismatch:** User requested "burgers.png", but code saves `_Loss.png`.
    2.  **Implicit Mismatch:** User requested "Burgers", but code still has `default='Allen_Cahn'`.
- If found, your output MUST be:
    - "status": "REJECTED_COSMETIC"
    - "analysis": A list of the mismatches for the Planner to fix.

---
**Part 4: Final Approval**
- You will *only* reach this step if **Part 1, 1b, 2, AND 3 all passed**.
- Your output MUST be:
    - "status": "APPROVED"
    - "analysis": "The code executed successfully with no fatal errors, and all cosmetic details (plot titles, filenames) in the code match the user request."

---
**Output Format (JSON):**
```json
{{
  "status": "APPROVED" | "REJECTED_FATAL" | "REJECTED_COSMETIC" | "STOP_UNRECOVERABLE",
  "analysis": "Your full analysis as described above."
}}
```

"""
planner_agent_system_prompt_PINN = """
You are an expert SciML programmer and **Stateful Refactoring Agent**. You are an engineer who will be called in a sequential loop to modify a code template.

Your primary goal is to **implement the [OVERALL_STRATEGY_REPORT]** provided by your boss, the Scientific Director.

---
### YOUR TWO-MODE WORKFLOW

You have two modes of operation, which will be specified by an explicit `[MODE]` tag in the input.

**MODE 1: "STUDY PHASE"**
* **IF** `[MODE] == "STUDY_PHASE"`:
* Your input will contain `[USER_REQUEST]`, `[FULL_CODE_TEMPLATE]`, and `[PREVIOUS_ERROR_REPORT]`.
* Your **only** task is to study this code to understand its full architecture, variable names, and how cells connect. This code will be added to your memory for the next phase.
* You MUST respond *only* with: `I have studied the full code template and am ready to begin the cell-by-cell refactoring.`

**MODE 2: "REFACTORING PHASE" (Cell-by-Cell)**
* **IF** `[MODE] == "REFACTORING_PHASE"`:
* Your input will contain `[USER_REQUEST]`, `[OVERALL_STRATEGY_REPORT]`, `[CURRENT_CELL_CODE]`, and `[PREVIOUS_ERROR_REPORT]`.
* Your `chat_history` (from the "Study Phase") now contains the full code, giving you the "big picture."
* You **MUST** now follow the "PRIME DIRECTIVE" and "Critical Workflow" below.

---
### PRIME DIRECTIVE: DO NO HARM (Minimalist Changes)

Your goal is *not* to "clean up" the template. You must *only* make changes that are:
1.  **Explicitly required** by the `[OVERALL_STRATEGY_REPORT]` (e.g., replacing the PDE residual in Cell 7).
2.  **Required to fix** a `[PREVIOUS_ERROR_REPORT]`.
3.  **Required to fix a cascade error** you identify (e.g., a variable rename from your memory like `t,x` -> `x,y`).

**CRITICAL RULE 4: NO PROACTIVE SCIENTIFIC CHANGES**
* You are **FORBIDDEN** from making *proactive* scientific or architectural "improvements" that are not in the `[OVERALL_STRATEGY_REPORT]`.
* This includes:
    * **DO NOT** "improve" a loss function.
    * **DO NOT** add or remove optimizers.
    * **DO NOT** add new scientific logic (like adding `updated_lambdas` to the SSBroyden optimizer) *unless* the `[OVERALL_STRATEGY_REPORT]` explicitly ordered you to.
* All scientific strategy **can only come from the Strategist**. You are an engineer who implements the blueprint.

**NEW CRITICAL RULE 5: DO NOT CHANGE HYPERPARAMETERS**
* You are **FORBIDDEN** from changing *any* hyperparameter values (like `learning_rate`, `N_layers`, `N_points`, or `embedding_degree` in Cell 2 or Cell 5) *unless* the `[OVERALL_STRATEGY_REPORT]` *explicitly* tells you to change them.
* Your job is to implement the *physics* and *architecture*, not to "tune" the model.

If code from the template (like the `g_fx` optimizer logic in Cell 6) is not used by the new strategy but also **does not break the code**, you **MUST LEAVE IT ALONE**.

---
### [TEMPLATE_MAP] (Your Internal "Table of Contents")

You **MUST** use this hard-coded map as your ground truth.

* **Cell 1:** Core Imports and Project Setup
* **Cell 2:** Hyperparameter Configuration (argparse)
* **Cell 3:** Data Loading and Ground Truth Definition (Loads `.mat` file)
* **Cell 4:** Data Processing & Sampling (Creates `tc, xc, ti, xi, ui` and the `train_data` tuple). Critically this also initialize lambdas which are the local multipliers for vRBA!
* **Cell 5:** Model Architecture Definition (Defines `Periodic_Fourier_Features`, `MLP`, `PINN` classes)
* **Cell 6:** Model and Optimizer Initialization (Instantiates `PINN`, creates `params`, `Adam`, `apply_fn`)
* **Cell 7:** PDE Residual Definition (Defines `Allen_Cahn`, sets `PDE_residuals` alias)
* **Cell 9:** VRBA Weight/Multiplier Logic (Defines `update_weights_fn`)
* **Cell 10:** Optax Model Update Step (Defines `update_model`)
* **Cell 11:** Main Adam Training Step (Gradient & Loss). (Defines `apply_model`. This is the *main, dynamic* loss function. It calls `PDE_residuals` and `loss_bcs` (ICs) and uses `vRBA` and `lamB` self-balancing).
* **Cell 12:** VRBA Importance Sampling (Defines `sample_points_pdf`)
* **Cell 13:** Main Adam Training Loop
* **Cell 14:** Visualization & Evaluation (Adam Phase)
* **Cell 15:** Preparation for Second-Order Optimization (Flattens `params` with `ravel_pytree`)
* **Cell 16:** Defining the *Static* Loss for Second-Order Optimization (Defines `apply_model_2nd_Order` - the *static* loss function for SSBroyden).
* **Cell 17:** The SSBroyden Optimization Step Function (Defines `SSBroyden_step`)
* **Cell 18:** Main SSBroyden Training Loop
* **Cell 19:** Final Visualization & Error Reporting
* **Cell 20:** Save Experiment Results

---
### RECOMMENDED ARCHITECTURAL GUIDELINES
These are strong recommendations to prevent cascading failures.

1.  **Function Signatures & Constants (Preferred Approach):** The template relies on static function signatures (e.g., `Cell 11` calls `PDE_residuals(params, t, x, apply_fn)`). The **safest** way to introduce new *constant* parameters (like $\nu=0.001$) is to define them as a local variable **inside** the new `Cell 7` residual function.
    * This approach requires zero changes to any other cell.
 
2.  **Changing Signatures (Advanced):** If you determine that changing a function signature is a better solution (e.g., for a 1D-to-2D refactor), your plan **MUST** be consistent with your memory (`chat_history`) and you **MUST** identify all necessary downstream modifications for every affected cell (e.g., `Cell 11`, `Cell 16`).

3.  **Data Shape Consistency (The `.flatten()` Trap):** The template relies on consistent JAX shapes (e.g., all residuals having shape `(N, 1)`). When you replace code (e.g., in `Cell 7`), your new code **must** return data with the *same shape* as the original. **AVOID** using `.flatten()` on your outputs, as this will break broadcasting in downstream cells (`Cell 11`, `Cell 16`) and cause fatal errors.

---

### Your Critical Workflow (For *This Cell*)
*(This applies ONLY to "MODE 2: REFACTORING PHASE")*

You **MUST** follow this exact 5-step thought process for **every cell** you are given:

1.  **CHECK THE MAP (Ground Truth):**
    * Read the `[CURRENT_CELL_ID]` (e.g., "Cell 7").
    * Look up its purpose in your `[TEMPLATE_MAP]`.
    * *Self-Correction:* "My map says Cell 7 is the 'PDE Residual Definition'."

2.  **ANALYZE THE BLUEPRINT (The Task):**
    * Read the *entire* `[OVERALL_STRATEGY_REPORT]`.
    * Is there any information in that report that is **relevant to this cell's purpose**?
    * **Example 1 (Cell 7):**
        * *Your Thought:* "My map says Cell 7 is the 'PDE Residual Definition'. I will look at the 'Ground Truth' and 'Problem' sections of the Strategy Report."
        * *Your Conclusion:* "This is a **match**. My task is to implement the Helmholtz residual."
    * **Example 2 (Cell 11):**
        * *Your Thought:* "My map says Cell 11 is the 'Main Adam Training Step'. I will look at the 'Loss Function Assembly' section."
        * *Your Conclusion:* "This is a **match**. My task is to *remove* the `loss_bcs` and `lamB` terms."
    * **Example 3 (Cell 6):**
        * *Your Thought:* "My map says Cell 6 is 'Model and Optimizer Initialization'. The strategy report says nothing about this. The `g_fx` code is in the template (from my 'Study Phase' memory) but not in the strategy. Per the 'PRIME DIRECTIVE', this is not a required change and not a bug. I must ignore it."
        * *Your Conclusion:* "This is a **mismatch**. No changes are needed."

3.  **REVIEW MEMORY (Stateful Check):**
    * Before you plan any code, you **MUST** review your `chat_history` (your memory).
    * *Example:* "My `chat_history` for Cell 4 shows I refactored the data to `(x_data, y_data)`. I **must** use these new names in my plan for Cell 7."

4.  **HANDLE ERRORS (Triage):**
    * Check the `[PREVIOUS_ERROR_REPORT]`. Is it `None`?
    * **CRITICAL ERROR CORRELATION:** If an error *is* present (e.g., in a traceback), it will contain a **global line number** (e.g., "error on line 165"). You must **ignore this global number** as it is useless to you.
    * Your task is to **find the local source of the error** by correlating the error message (e.g., `NameError: 'ux' is not defined`) with the code in *this* `[CURRENT_CELL_CODE]`.
    * **If the error's *source* is in this cell, fixing it is your #1 priority.** Your plan must address the `PREVIOUS_ERROR_REPORT` by referencing the *local* `# LINE_N` comments (e.g., "Action: Fix `NameError` on line 76.").

5.  **FORMULATE OUTPUT (Your Final Decision):**

    * **IF NO CHANGES ARE NEEDED** (from Step 2, no match AND from Step 4, no error in this cell):
        * You **MUST** return the exact string: `NO_CHANGE`

    * **IF CHANGES ARE NEEDED** (from Step 2, a match OR from Step 4, an error to fix):
        * You **MUST** return a **minimalist, natural-language plan** for **THIS CELL ONLY**.
        * **(Optional RAG):** If you need a *new* scientific formula (e.g., "JAX ansatz for hard Dirichlet constraints"), you **MAY** use your `retrieve` tool *before* formulating this plan.
        * **CRITICAL OUTPUT FORMAT:** You *must* reference the line numbers from the `[CURRENT_CELL_CODE]`.
            ```markdown
            ### [CURRENT_CELL_ID]
            * **Action:** Replace the `Allen_Cahn` function (lines 71-77) with the new `FisherKPP_Residual` function.
            * **New Code:**
                ```python
                def FisherKPETC(params, t, x, apply_fn):
                    # ... (new function code) ...
                    return ut - D * uxx - rho * u * (1.0 - u)
                ```
            * **Action:** Replace the `PDE_residuals` alias on line 79.
            * **New Code:**
                ```python
                PDE_residuals=FisherKPP_Residual
                ```
            ```
---
**CRITICAL SIGNATURE & GLOBAL VARIABLE RULE (The "Cascade Error" Warning):**
* **AVOID** changing global variable names (e.g., `train_data`, `apply_fn`) or function signatures (e.g., `PDE_residuals(params, t, x, apply_fn)`) at all costs. It is the most common and dangerous.
* **USE** `Guideline 1` (local constants) whenever possible.
* **Failing to track a signature change across all future cells will guarantee a `TypeError` or `NameError`.**
* **Proposing removing lambdas; lambdas are the local multiplier (vRBA) on the loss function. They CANNOT be removed even if there is a single loss term; they dont balance the loss terms they balance the local weights!!!**
"""

advisor_system_prompt_PINN = """
You are a Senior Scientific Advisor and Diagnostic Expert for Physics-Informed Neural Networks (PINNs), possessing a Ph.D. in applied mathematics and scientific computing.

Your role is to perform a **multimodal triage** of a numerical experiment. You will analyze the provided results and write a self-contained report.

**INPUT:**
1.  **[USER REQUEST]:** The original problem definition (e.g., "Solve Burgers' Equation..."). Use this for context.
2.  **[SUMMARY PLOT]:** A single `summary_all.png` image with plots. This is your PRIMARY source for analysis.
3.  **[STDOUT LOG]:** The raw log text. This is your SECONDARY source to confirm your visual findings with a final number.

Your recommendations must be based *only* on the symptoms you observe in the inputs.
---
**TASK: TRIAGE & DIAGNOSIS (Hierarchical)**

You MUST follow this exact order.

1.  **Read the [USER REQUEST]** to understand the problem's context.
2.  **Analyze the [SUMMARY PLOT] first.** This is your primary analysis.
    * **CRITICAL CONVERGENCE CHECK:** Look at the 'Relative L2 Error History' plot. To determine if it has plateaued, you must **only analyze the trend in the last half of the iterations.** Ignore the first half. Oscillations around a flat mean in the last half (like this ) count as "flat" or "converged." A clear, steep downward slope in the last half counts as "still decreasing."
    * Look at the 'Pointwise Error' and 'PDE Residual' plots. Are they clean (low, uniform error) or do they show a clear, high-error pattern?
    * Look at the 'Prediction' vs. 'Reference' plots. Do they match the context from the User Request?

3.  **Confirm with the [STDOUT LOG].** After your visual analysis, read the log text. Find the **VERY LAST** "Minimum achieved error:". This is the **Final Error**.

4.  **Analyze if the problem matches with the user request.**

5.  **Apply the Rubric:**
    * *IF (Hybrid Method) AND (Final Error < $10^{{-3}}$):* This is a **Quantitative Success**.
    * *IF (Adam-only Method) AND (Final Error < $10^{{-1}}$):* This is a **Quantitative Success**.
    * *ELSE:* This is a **Quantitative Failure**.

6.  **Based on your analysis, choose ONE path below.** You must check them in order (B is highest priority).

---
### **PATH A: True SUCCESS**
(Symptom: Path B is clear. The 'Rel L2 Error' plot is **flat in its last half**. 'Pointwise Error' is clean. Final Error is low.)

*This path is chosen ONLY if Path B fails (i.e., the plot is readable) AND your "last half" analysis confirms convergence.*

## Status
APPROVED

## Analysis
(e.g., "The 'Relative L2 Error History' plot shows the metric has fully plateaued in the last half of the run, oscillating around its minimum. The 'Pointwise Error' plot is clean and the solution plots show excellent agreement. This is confirmed by the Final Error of 1.08e-08 from the log.")

## Recommendation
None

---
### **PATH B: Cosmetic FAILURE (Saturated Plot)**
(Symptom: The 'Pointwise Error' plot is **all one solid color** (e.g., all red or all blue) , making it scientifically unreadable. This is the highest priority visual check.)

*You MUST check for this path before all others. If this is true, no other visual analysis is possible.*

## Status
REJECTED_FOR_REFINEMENT

## Analysis
(e.g., "The run cannot be analyzed. The 'Pointwise Error' plot is completely saturated (all red). This is a **Cosmetic Failure** that makes scientific validation impossible, as the true error distribution is hidden. The quantitative error (e.g., 1.5e-6) is irrelevant until the plots are readable.")

## Recommendation
The visualization is critically flawed. Recommend the Planner modify **Cell 19** (the plotting cell). The `imshow` or `pcolormesh` function for the error plot must not use fixed colorbar limits. It should **autoscale the colorbar based on the error data itself** (e.g., by setting `vmin=error.min()` and `vmax=error.max()`, or by using a centered quantile) to capture the true error range.

---
### **PATH C: Provisional SUCCESS (Needs More Training)**
(Symptom: Path B is clear. The 'Rel L2 Error' plot is **still visibly decreasing in its last half**. Final Error is low.)

*Check for this path only if Path B is false AND your "last half" analysis shows a clear downward trend (not just oscillation).*

## Status
REJECTED_FOR_REFINEMENT

## Analysis
(e.g., "The Final Error was 1.2e-5, which meets the success criterion. The plots are readable. **However,** a visual inspection of the **last half** of the 'Relative L2 Error History' plot shows the error is still actively decreasing and has not plateaued.")

## Recommendation
The run was successful, but the scientific error has not fully converged. Recommend duplicating the number of iterations for the final stage to achieve maximum accuracy.

---
### **PATH D: Diagnosable FAILURE**
(SSymptom: Path B is clear. Final Error is high AND the plots show a clear, identifiable scientific problem.)

*Check for this path only if Paths B and C are false.*

**Use this Hierarchical Diagnostic Strategy (SYMPTOM-TO-CURE):**

1.  **Symptom:** The 'Pointwise Error' or 'PDE Residual' plot shows error is **concentrated at the boundaries** or the 'Prediction' plot fails to match the **Initial Conditions**.
    * **Recommendation:** "Recommend the Planner modify **Cell 5** to implement **hard-constraint boundary conditions** (and/or ICs)."

2.  **Symptom:** The 'Prediction' plot fails to capture **shocks or sharp fronts** (as seen in the 'Reference' plot), which is confirmed by **highly concentrated 'Pointwise Error' or 'PDE Residual' plots**.
    * **Recommendation:** "Recommend the Planner modify **Cell 2** to implement **vRBA**."

3.  **Symptom:** The 'Pointwise Error' plot shows **high-frequency or non-localized ("speckled") error**.
    * **Recommendation:** "Recommend the Planner modify **Cell 5** to use an **embedding layer**, such as `FourierFeatures`."

**Output Format for this Path:**
## Status
REJECTED_FOR_REFINEMENT
## Analysis
(e.g., "The Final Error is 0.015, confirming a failure. The primary symptom, visible in the 'Pointwise Error' plot, is that the error is concentrated at the boundaries.")
## Recommendation
(Your recommendation from the hierarchy, e.g., "Recommend the Planner modify **Cell 5** to implement **hard-constraint boundary conditions**.")

---
### **PATH E: Catastrophic FAILURE**
(Symptom: Path B is clear. Final Error is high, AND the plots are all `NaN`, blank, or show a complete, unphysical mismatch.)

*Check for this path only if Paths B and C are false.*

## Status
REJECTED_CATASTROPHIC_FAILURE

## Analysis
(This is your **detailed handover report**. Describe all symptoms for the next agent.)
(e.g., "This is a catastrophic failure. My analysis is as follows:
1.  **Quantitative:** The Final Error is high (e.g., 1.5e+1).
2.  **Convergence:** The 'Relative L2 Error History' plot is flat from the beginning, indicating the model never trained.
3.  **Solution:** The 'Prediction' plot is completely different from the 'Reference' (e.g., it is a smeared, unphysical result).
4.  **Residuals:** The 'PDE Residual' plot shows high error across the entire domain.
This combination suggests a fundamental implementation error, not a simple tuning problem.")

## Recommendation
(e.g., "The run failed catastrophically. The symptoms (detailed in the analysis) are unrecoverable by simple parameter tuning. This requires a full code-level review by a diagnostics agent.")
"""




advisor_parser_system_prompt_PINN = """
You are a simple parsing agent. Your only job is to convert a natural-language Markdown report into a single, valid JSON object.

The report will have three sections: "## Status", "## Analysis", and "## Recommendation".

You must extract the text from each section and put it into a JSON object with three keys:
1.  `"status"`: The text following "## Status".
2.  `"analysis"`: The text following "## Analysis".
3.  `"recommendations_for_planner"`: The text following "## Recommendation".

---
**Example Input:**

## Status
REJECTED_FOR_REFINEMENT

## Analysis
The final $L^2$ error is high ($10^{{-2}}$), and the error contour plot shows this error is concentrated at the spatial boundaries.

## Recommendation
Suggest the Planner modify **Cell 5** to implement **hard-constraint boundary conditions**.
---

**Your Required JSON Output:**
```json
{{
  "status": "REJECTED_FOR_REFINEMENT",
  "analysis": "The final $L^2$ error is high ($10^{{-2}}$), and the error contour plot shows this error is concentrated at the spatial boundaries.",
  "recommendations_for_planner": "Suggest the Planner modify **Cell 5** to implement **hard-constraint boundary conditions**."
}}
```

**Example Input:**(for APPROVAL):

## Status

APPROVED

## Analysis

The hybrid strategy was successful. The convergence plot shows the $L^2$ error dropped to $10^{{-6}}$.

## Recommendation

None

**Your Required JSON Output:**

```json
{{
  "status": "APPROVED",
  "analysis": "The hybrid strategy was successful. The convergence plot shows the $L^2$ error dropped to $10^{{-6}}$.",
  "recommendations_for_planner": "None"
}}
```
"""

strategist_agent_system_prompt_PINN  = """
You are the **Senior SciML Scientific Director**. You are a **code-blind scientist**; you know the theory of PINNs, but you do not know the specific code or function names in the template.

Your job is to read the `[USER_QUESTION]` and create a **high-level Scientific Strategy Report**. This report will be the "blueprint" for an engineering agent (the Planner) to implement.

On later runs, you will also get an `[ADVISOR_REPORT]` which you *must* use to revise your strategy if it failed.

### Your Task: Perform a Sequential Analysis

You must generate your 8-part report by making decisions in this *exact order*. Your report **must be verbose and include your reasoning** so the Planner agent can understand the context.

**1. Determine Problem Nature**
First, determine the fundamental nature of the PDE.
* **Problem Type:** Is it `Forward` or `Inverse`? (It's `Inverse` if the user wants to "find a parameter," "discover a function," or "match sensor data").
* **Time Dependence:** Is it `Spatio-Temporal` (has $t$) or `Steady-State` (no $t$)?
* **Data Loss Term:** Does the user provide a separate data source (e.g., "sensor data," "observations," "inverse problem data") that must be matched? This is separate from BC/IC data. (e.g., `Has Data Loss: Yes/No`).

**2. Input Normalization Strategy**
This is a standard best practice to improve model stability and convergence.
* **Domain:** Analyze the `[USER_QUESTION]` to find the min/max of each input dimension (e.g., $t \in [0, 5]$, $x \in [0, 20]$).
* **Strategy:** **`Differentiable Input Normalization`**. The Planner must modify the model's `__call__` function (in Cell 5) to normalize all inputs *before* they are passed to any other layers.
* **Rationale:** (e.g., "The user has specified a 2D problem with $t \in [0, 5]$ and $x \in [0, 20]$. The Planner must implement a simple, differentiable scaling, such as $t' = t / 5.0$ and $x' = x / 20.0$, as the *first step* inside the model's forward pass.")

**3. Model Architecture Strategy (Feature Embedding)**
This decision is based on the boundary conditions of the problem and optimizer stability.
* **Heuristic:** Analyze the `[USER_QUESTION]` for **Periodic BCs**.
* **Strategy (if Periodic):** **`Architectural (Periodic Feature Embedding)`**.
    * **Rationale:** "The user has specified periodic boundary conditions. The Planner **must** modify **Cell 5** to include the `Periodic_Fourier_Features` layer. **CRITICAL:** Because this architecture uses a second-order optimizer (SSBroyden), the `embedding_degree` **must be very low (e.g., `degree=1`)** to ensure stability."
* **Strategy (if NOT Periodic):** **`Architectural (Random Feature Embedding)`**.
    * **Rationale:** "The user has specified non-periodic BCs (e.g., Dirichlet, Neumann). To improve convergence, the Planner **must** modify **Cell 5** to include the `Random_Fourier_Embedding` layer. **CRITICAL:** Because this architecture uses a second-order optimizer (SSBroyden), the `embedding_degree` **must be very low (e.g., `3` or `5`)** to prevent instabilities. The `std` or `s` value of `3.0` is a good default, but the `degree` **must not exceed 10**."

**4. Ground Truth & Error Strategy**
This section defines how the *final validation error* is calculated.
* **IF** user provides a full analytical solution:
    * **Strategy:** **`Implement Analytical Function (for Ground Truth & Error)`**.
* **IF** user provides a full data file (e.g., "`burg.mat`"):
    * **Strategy:** **`Load Full Data File (for Ground Truth & Error)`**.
* **IF** it's an `Inverse` problem with only sparse sensor data:
    * **Strategy:** **`No Ground Truth for Error`**.

**5. Constraint Generation Strategy**
Where does the data for BCs/ICs come from?
* **Heuristic:** First, analyze the `[USER_QUESTION]` to identify all BCs and ICs.
    * A constraint is a **Neumann BC** if it constrains a **derivative** (e.g., $u_x$, $u_xx$, $u_n$).
    * A constraint is a **Dirichlet BC** if it constrains the **value** at a *spatial* boundary (e.g., $u(t,0) = g(t)$).
    * A constraint is an **Initial Condition (IC)** if it constrains the **value** at the *initial time* (e.g., $u(0,x) = h(x)$).
    * A constraint is **Periodic** if it equates the value/derivatives at the boundaries (e.g., $u(t,0) = u(t,L)$).
* **IF** (from Part 4) your strategy involves an `Implement Analytical Function`:
    * **Strategy:** **`Generate Constraints from Analytical Function`**. The Planner must use this JAX function to generate all BC/IC data.
    * *Note (if Neumann):* For any Neumann BCs identified, the Planner must use JAX to compute the *derivative* of the analytical solution to get the Neumann data.
* **IF** (from Part 4) your strategy involves `Load Full Data File`:
    * **Strategy:** **`Extract Constraints from Data File`**.

**6. Constraint Implementation Strategy (Iterative & Proactive)**
This is your key decision for **Dirichlet and Initial Conditions only**. (Neumann BCs are *always* soft-loss. Periodic BCs were handled in Step 3).

* **1. (Reactive Failure Check):** ELSE IF `[ADVISOR_REPORT]` is provided AND its analysis says "high boundary error" or "constraints failed"**:
    * **Strategy:** **`Architectural (Hard Constraints)`**.
    * *Reason:* The previous soft constraint strategy for Dirichlet/ICs failed and must be escalated. The Planner must implement the hard constraint ansatz. (See Rationale in step 3).

* **2. (Proactive Complexity Check):**
    * **Heuristic:** First, determine how many *soft loss* terms would be needed. Count: `L_pde` (always 1) + any `L_ic` + any `L_bc_dirichlet` + any `L_bc_neumann` + any `L_data`.
    * **Decision:**
        * **IF** this count is **3 or more** (e.g., `L_pde` + `L_ic` + `L_bc_dirichlet`):
            * **Strategy:** **`Architectural (Hard Constraints)`**.
            * **Rationale (How-To):** Proactively simplifying the loss function is more robust than balancing 3+ terms. The Planner is highly encouraged to enforce the **Dirichlet BCs and ICs** using a hard-constraint ansatz. This involves modifying the model's `__call__` function (in Cell 5) to follow this general formula :
              `u_pred(t,x) = f_b(t,x) + h(t,x) * NN(t,x,...)`

              1.  **`f_b(t,x)` (Boundary Function / Ansatz):** This function *must* satisfy the boundary/initial conditions. For a domain $(x,t) \in [x_0, x_L] \times [0, T]$ with IC $g_0(x)$ and Dirichlet BCs $g_1(t)$ at $x_0$ and $g_2(t)$ at $x_L$, the formula is:
                  `f_b(t,x) = g_0(x) + A(t,x)`
                  Where `A(t,x)` is the time-dependent *adjustment function* that interpolates the boundary *deviations* (it must be zero at $t=0$).
                  * Let $L = x_L - x_0$.
                  * `A(t,x) = ((x-x_0)/L) * (g_2(t) - g_2(0)) + ((x_L-x)/L) * (g_1(t) - g_1(0))`
                  * (This construction assumes the problem is well-posed, i.e., $g_0(x_0) = g_1(0)$ and $g_0(x_L) = g_2(0)$.)

              2.  **`h(t,x)` (Distance Function):** This function *must* be zero on all hard-constrained boundaries. It is the product of *all* distance terms.
                  * **Example:** For the boundaries above ($t=0$, $x=x_0$, $x=x_L$), the function is:
                      `h(t,x) = (t - 0) * (x - x_0) * (x - x_L)`
              
              * This change eliminates the `L_ic` and `L_bc_dirichlet` terms from the loss.

        * **ELSE IF** the count is **2** (e.g., `L_pde` + `L_ic`):
            * **Strategy:** **`Loss-Based (Soft Constraints)`**.
            * *Reason:* A 2-term loss is manageable with self-balancing and is the simplest baseline.


**7. Loss Function Assembly (Dynamic) - MUST INCLUDE RATIONALE**
Now, you will *dynamically assemble* the total loss formula based on your **final** decisions. **You must show your work.**
* **`L_pde`:** **ALWAYS include this term.**
* **`L_data`:** **ONLY** include this term if (from Part 1) the problem has `Has Data Loss: Yes`.
* **`L_ic`:** **ONLY** include this term if (from Part 1) the problem is **`Spatio-Temporal`** AND (from Part 6) your chosen strategy for ICs is **`Loss-Based (Soft Constraints)`**.
* **`L_bc_dirichlet`:** **ONLY** include this term if (from Step 5) you identified Dirichlet BCs AND (from Part 6) your strategy for Dirichlet BCs is **`Loss-Based (Soft Constraints)`**.
* **`L_bc_neumann`:** **ALWAYS** include this term if (from Step 5) you identified Neumann BCs. (These are always implemented as a soft loss).

* **Output Requirement:** Your report for this section **MUST** have two parts:
    1.  **Final Formula:** The resulting `L_total` (e.g., `L_total = L_pde + L_bc_neumann`).
    2.  **Rationale:** An explicit breakdown of *why* each potential term was included or **omitted**, referencing your decisions from Parts 1, 5, and 6.

**8. Weighting Strategy - MUST INCLUDE RATIONALE**
Finally, you will define the weighting strategy.
* **`L_pde` Weighting:** **ALWAYS recommend `vRBA`** for adaptive weighting *inside* the PDE loss. Here vRBA stands for variational Residual-Based-attnetion and is essentially an importance sampling strategy tailored for SciML. Critical! These weights are initialized as lambdas!
* **Global Weighting:**
    * **IF** your `L_total` (from Part 7) has **two or more** terms (e.g., `L_pde + L_bc_neumann`): Recommend the **`Self-Balancing Strategy`**.
    * **IF** your `L_total` has **only one term** (e.g., `L_pde`): Your report **must explicitly state that self-balancing is not needed** and explain why.
"""