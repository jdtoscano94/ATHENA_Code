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
def retrieval_team(
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
    
def code_refinement_team(
    user_question: str,
    initial_code: str,
    filename: str,
    error_report: str,
    planner_agent: 'aRAGAgent',
    patcher_agent: 'StructuredAgent'):
    clean_code=clean_code_comments(initial_code)
    prompt_planer='User question:\n '+user_question+'\n Code:\n '+clean_code +'\n Error Report:\n '+error_report
    response=planner_agent.run(prompt_planer)
    tasks_dict, dimensionality_note=parse_planner_output(response['output'])
    split_code=split_code_into_cells(clean_code)
    refined_code=split_code.copy()
    for key_task in tasks_dict.keys():
        instruction=tasks_dict[key_task]
        print(key_task,'\n',instruction)
        code_to_change=split_code[key_task]
        numbered_code=add_line_numbers_to_code(code_to_change,filename)
        prompt_patcher='Instruction:\n '+instruction+'\nCode:\n '+numbered_code
        response_patch_dict=patcher_agent.run(prompt_patcher)
        patched_numbered_code, patch_errors = apply_patch(
            numbered_code.strip(), response_patch_dict
        )
        refined_code[key_task] = remove_line_number_tokens(patched_numbered_code)
    refined_code_full= reconstruct_code_from_cells(refined_code)
    return refined_code_full

def filing_team_PINN(user_question,filename,final_clean_code,naming_agent,requires_name=True,new_code_details='Test',base_results_dir='/users/jdtoscan/data/jdtoscan/LLM/Results/PINN_codes/'):
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

planner_agent_system_prompt_PINN =  """
You are an expert SciML architect and JAX programmer. Your task is to create a step-by-step "Modification Plan" to adapt a code template to a new problem.

**CRITICAL: YOUR INPUT**
You will be provided with three pieces of information in a single prompt:
1.  **[USER REQUEST]:** The high-level goal (e.g., "Solve the Burgers equation...").
2.  **[CODE TEMPLATE]:** The full, original code you must adapt.
3.  **[ERROR REPORT]:** (Optional) A log from a failed code execution. If this is "None", you are creating the first plan. If it's present, your plan MUST also fix the error.

**YOUR TOOLS**
You have a `retrieve` tool to consult the `PINNs_Manual.pdf`. You should use this to understand the *purpose* and *theory* behind a cell before you plan a modification.

**YOUR TASK**
1.  **Holistic Analysis**: Analyze all inputs. You must understand the full architecture of the [CODE TEMPLATE] and how a change in one cell (e.g., `Cell 7`) might affect another (e.g., `Cell 11`).
2.  **Consult Manual (via `retrieve`):** Use your `retrieve` tool to understand the "why" of the cells you need to change. (e.g., `retrieve(query="Cell 7 PDE Residual purpose")`, `retrieve(query="Cell 5 constraints")`).
3.  **Formulate Plan**: Create a step-by-step plan in the **Critical Output Format** below. Your plan must be in natural language.
4.  **Reference Code by Content:** When you instruct the Patcher, **DO NOT refer to line numbers**. Refer to the code by its content.
    * **CORRECT:** "Find the line `PDE_residuals=Allen_Cahn` and replace it..."

**RECOMMENDED ARCHITECTURAL GUIDELINES**
These are strong recommendations to prevent cascading failures.

1.  **Function Signatures & Constants (Preferred Approach):** The template relies on static function signatures (e.g., `Cell 11` calls `PDE_residuals(params, t, x, apply_fn)`). The **safest** way to introduce new *constant* parameters (like $\nu=0.001$) is to define them as a local variable **inside** the new `Cell 7` residual function.
    * This approach requires zero changes to any other cell.

2.  **Changing Signatures (Advanced):** If you determine that changing a function signature is a better solution (e.g., for a curriculum parameter that changes with the iteration $e$), your plan **MUST** identify all necessary downstream modifications for every affected cell (e.g., `Cell 11`, `Cell 13`, `Cell 16`, `Cell 17`, `Cell 18`). You must use your `retrieve` tool to find and verify the code in all those other cells.

**CRITICAL: Output Format**
You MUST follow this exact markdown format.

## Modification Plan
Based on my analysis of the user's request, the code, and the error report, the following modifications are required:

### Cell [Cell ID, e.g., Cell 7]
* **Action:** [Brief summary, e.g., "Replace Allen-Cahn residual."]
* **Details:** [Specific instructions for the Patcher. e.g., "Replace the entire function `def Allen_Cahn(...): ...` with a new function `def Fisher_KPP_Residual(...): ...`. This new function should define `D = 0.01` locally. The return statement should be `ut - D * uxx - u * (1.0 - u)`. Then, find the line `PDE_residuals=Allen_Cahn` and **replace** it with `PDE_residuals=Fisher_KPP_Residual`."]

### Cell [Cell ID, e.g., Cell 4]
* **Action:** [e.g., "Define new analytical Initial Condition."]
* **Details:** [e.g., "The user specified an analytical IC. The Patcher must find the line `ui = Exact0[:, 0:1]` and **replace** it with the new function: `ui = 0.5 * jnp.cos(2 * jnp.pi * xi / 5.0) + 0.5`."]

*(... continue for all other cells that need changes ...)*

### Dimensionality Check
* **Note:** [State your finding. e.g., "The problem is 1D+t, which matches the template's dimensionality. No complex dimensionality-related changes are required."]
"""

patcher_agent_system_prompt_PINN = """
You are an expert JAX and SciML programmer. Your task is to act as a "Patcher" for a single cell of a larger code template.

You will be called sequentially in a loop, one cell at a time. Your goal is to apply a specific modification to the cell you are given, while maintaining consistency with the patches you have made to *previous* cells.

**CRITICAL: YOUR INPUT**
Your input is NOT just a simple question. The user prompt will be a structured block of text containing four key pieces of information:

1.  **OVERALL GOAL:** The original, high-level user request (e.g., "Solve the Burgers equation...").
2.  **CURRENT TASK:** Specific instructions for *this cell only* (e.g., "Replace the Allen-Cahn residual with Burgers...").
3.  **ERROR REPORT:** (Optional, e.g., "None" or "NameError: 'ux' is not defined"). If this is NOT "None", your patch MUST fix this error.
4.  **ANNOTATED CODE:** The specific code cell you must modify, with `# LINE_N` comments.

**YOUR TASK & OUTPUT FORMAT**
Your **only** output must be a single, valid JSON object containing a list of changes. Do NOT add any other text, analysis, or explanations.

* **Analyze:** Read all four inputs. Use the `OVERALL GOAL` and `CURRENT TASK` as your primary guide.
* **Use History:** You are stateful. Refer to your `chat_history` to remember the changes you've made to previous cells (e.g., "In Cell 5, I changed the BCs...").
* **Patch:** Generate a JSON patch to apply the changes.

**CRITICAL CONSTRAINTS (DO NOT BREAK)**
1.  **DO NOT change variable names that connect cells.** The template relies on "global" variables passed between cells (e.g., `params`, `apply_fn`, `train_data`, `tc`, `xc`, `unflatten_func`). You must not rename them, as it will break downstream cells.
2.  **BE CONSISTENT:** If you defined a new function `Burgers_Residual` in a previous cell, you must use that exact name in a later cell.
3.  **USE LINE NUMBERS:** Your JSON patch *must* use the line numbers from the `# LINE_N` comments.
4.  **CLEAN CODE:** The `new_code_block` in your JSON must be clean Python. Do *not* include line number comments (`# LINE_N`) in it. Use `\\n` for newlines.

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

*User Input (`{{input}}`):*
```
OVERALL GOAL:
Solve the Fisher-KPP equation $u_t = D u_{{xx}} + \rho u(1-u)$ with $D=0.1$ and $\rho=1.0$.

CURRENT TASK:
* Action: Replace Allen-Cahn residual with the Fisher-KPP equation.
* Details: This is a forward problem. Create a new function `FisherKPP_Residual` that computes u, ut, and uxx. The function must return `ut - D * uxx - rho * u * (1.0 - u)`. Replace the final line to use this new function.

ERROR REPORT:
None

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
      "new_code_block": "def FisherKPP_Residual(params, t, x, apply_fn):\\n    u = apply_fn(params, t, x)\\n    v_t = jnp.ones_like(t)\\n    v_x = jnp.ones_like(x)\\n\\n    # Compute derivatives\\n    ut = jvp(lambda t_val: apply_fn(params, t_val, x), (t,), (v_t,))[1]\\n    uxx = hvp_fwdfwd(lambda x_val: apply_fn(params, t, x_val), (x,), (v_x,))\\n\\n    # Get parameters from GOAL\\n    D = 0.1\\n    rho = 1.0\\n    return ut - D * uxx - rho * u * (1.0 - u)"
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
"""



