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
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
### Teams
def strategy_team(
    user_question: str,
    strategist_agent,
    critic_agent,
    history_log_string: str = "",
    error_report: str = "None (First attempt)",
    max_iterations: int = 5
):
    print("\n=== 🧠 Starting Strategy Team (Proposer-Critic Loop) ===")
    
    critic_feedback = "None (Initial Proposal)"
    critic_approval = False
    strategist_report = ""
    iteration = 0

    # Loop while not approved and within iteration limit
    while not critic_approval and iteration < max_iterations:
        print(f"\n--- 🔄 Strategy Iteration {iteration + 1}/{max_iterations} ---")
        
        # 1. Run Strategist
        prompt_strategist = (
            f"[USER_QUESTION]:\n{user_question}\n\n"
            f"[ADVISOR_REPORT]:\n{error_report}\n"
            f"[TRIAL_HISTORY_LOG]:\n{history_log_string}\n"
            f"[CRITIC_REPORT]:\n{critic_feedback}\n"    
        )
        response_strategist = strategist_agent.run(prompt_strategist)
        strategist_report = response_strategist['output']
        print(f"--- Strategy Report Received ---\n{strategist_report}\n")

        # 2. Run Critic
        prompt_critic = (
            f"[USER_QUESTION]:\n{user_question}\n\n"
            f"[ADVISOR_REPORT]:\n{error_report}\n"
            f"[TRIAL_HISTORY_LOG]:\n{history_log_string}\n"
            f"[STRATEGIST_PLAN]:\n{strategist_report}\n"    
        )
        response_critic = critic_agent.run(prompt_critic)
        critic_report = response_critic['output']
        print(f"--- Critic report---\n{critic_report}\n")

        # 3. Check Approval (Using your exact logic)
        # Takes only the first row to avoid confounding
        critic_approval = check_approval(critic_report.strip().split('\n')[0].upper())
        
        # Update feedback for the next iteration (if needed)
        critic_feedback = critic_report
        iteration += 1

    return strategist_report, critic_approval


# --- Corrected Team Function ---

def retrieval_team_direct(
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


def retrieval_team(
    user_question: str,
    retriever_agent: 'Retriever_Agent',
    validator_agent: 'aRAGAgent',
    max_iterations: int = 5
) -> Tuple[str, str, bool, str]:

    is_approval = False
    extraction_iterations = 0
    file_content = ''
    filename = ''
    validator_feedback = '' # This will store the feedback for the next loop
    while (is_approval == False and extraction_iterations < max_iterations):
        print(f"--- Retrieval Iteration: {extraction_iterations} ---")
        
        # --- 1. Run Retriever ---
        if extraction_iterations == 0:
            # First attempt, use the original user question
            response = retriever_agent.run(user_question)
        else :
            # Subsequent attempts, use the validator's feedback
            response = retriever_agent.run(validator_feedback)
        print(f"Retriever Message:\n{response['output']}")
        target_filename = extract_filename(response['output'])
        # --- 2. Find Match and Run Validator ---
        # Set defaults in case no match is found
        validator_feedback = 'I could not find the code. Please give me only the filename.\n'
        file_content = ''
        filename = ''
        found_match = False
        for extracted_code in response['original_code']:
            my_document = extracted_code
            doc_filename = my_document.metadata['filename']
            
            if doc_filename == target_filename:
                found_match = True
                file_content = my_document.page_content
                filename = doc_filename # Store the found filename

                validator_input = ('User Request:\n'
                                    + user_question
                                    + '\nAgent_response:\n'
                                    + response['output']
                                    + '\nCode:\n'
                                    + file_content)
                validator_result_dict = validator_agent.run(validator_input)
                validator_feedback = validator_result_dict['output'] # Store for next loop
                print(f"Validator Message:\n{validator_feedback}")
                is_approval = check_approval(validator_feedback)
                print(f"Approval Status: {is_approval}")
                break # Found match, stop searching documents
        if not found_match:
             print(f"Warning: Retriever mentioned '{target_filename}', but it was not found in the retrieved documents.")
             # The default 'validator_feedback' will be used to ask the retriever again
        extraction_iterations += 1
    if not is_approval:
        print(f"❌ Retrieval loop failed after {extraction_iterations} attempts.")
    else:
        print(f"✅ Retrieval loop approved file: {filename}")

    return file_content, filename, is_approval, validator_feedback
def implementation_team(
    user_question: str,
    planner_agent: 'aRAGAgent',
    patcher_agent: 'StructuredAgent',
    planner_parser_agent:'StructuredAgent',
    initial_code: str='', # This will be by default or if the advisor recommends chaing the tamplate.
    clean_functions:str='',
    filename: str='',
    error_report: str='',
    strategist_report:str='',
    remove_comments:bool=False,
    ):
    code_and_descriptions=remove_analysis_blocks(initial_code)
    homogenous_code=homogenize_cell_comments(code_and_descriptions)
    if remove_comments:
        homogenous_code = clean_descriptions(homogenous_code) 
    split_code = split_code_into_cells(homogenous_code)
    refined_code_dict = {} # Use a new dict to store the output
    language = get_language_from_filename(filename)
    comment_char = get_comment_char(language)
    print("--- Starting Sequential, Stateful Refinement ---")
    # Clear memory:
    planner_agent.clear_history()
    prompt_planner = (
        f"[MODE]:\n STUDY_PHASE\n\n" 
        f"[SCIENTIFIC_STRATEGY_REPORT]:\n{strategist_report}\n\n"
        f"[ERROR_REPORT]:\n{error_report}\n\n"
        f"[FULL_CODE]:\n{homogenous_code}\n\n"
        f"[REFERENCE FUNCTIONS]:\n{clean_functions}\n\n"
    )
    response=planner_agent.run(prompt_planner)
    print('Planner Study Phase:',response['output'])
    mislabeled_cells = ''
    num_bad_cells = 1 
    iteration_count=0
    available_keys=", ".join(f"'{key}'" for key in split_code)
    while num_bad_cells > 0:
        iteration_count += 1
        if iteration_count > 3:
            # Crash the process to prevent downstream execution
            raise RuntimeError("Planner Parser failed to return valid keys after 3 attempts. Aborting.")
        bad_cells = []
        prompt_parser = (
            f"[Summary]:\n{response['output']}\n\n"
            f"[Message]:\n{mislabeled_cells}\n\n"
            f"[Available Cells]: You should pick the key names form here: \n{available_keys}\n\n"
        )
        # 1. Get String Response
        identified_cells = planner_parser_agent.run(prompt_parser)
        try:
            for cell_id in identified_cells.keys():
                if cell_id not in split_code:
                    if cell_id != '':
                        bad_cells.append(cell_id)
            num_bad_cells = len(bad_cells)
            if num_bad_cells > 0:
                mislabeled_cells = (
                    f"Error: The keys {bad_cells} do not exist in the code. "
                    f"Valid keys are: {list(split_code.keys())}. Please correct the JSON."
                )
                print(mislabeled_cells)   
        except Exception as e:
            print(f"JSON Parse Error: {e}")
            mislabeled_cells = "Error: Output was not valid JSON. Please return strictly valid JSON."
            num_bad_cells = 1 
    for cell_id in split_code.keys():
        print(f"\n--- Processing: {cell_id} ---")
        full_cell_content = split_code[cell_id]
        all_lines = full_cell_content.strip().split('\n')
        header_line = all_lines[0]
        code_body = '\n'.join(all_lines[1:])
        print(f"\n--- Analizing: {header_line} ---")
        if cell_id in identified_cells.keys():
            numbered_cell_code = add_line_numbers_to_code(code_body, filename)
            prompt_planner = (
                f"[MODE]:\nREFACTORING PHASE\n\n" 
                f"[SCIENTIFIC_STRATEGY_REPORT]:\n{strategist_report}\n\n"
                f"[ERROR_REPORT]:\n{error_report}\n\n"
                f"[CURRENT_CELL_CODE]:\n{numbered_cell_code}\n\n"
                f"[REFERENCE FUNCTIONS]:\n{clean_functions}\n\n"
                f"[YOUR_TASK]:\n"
            )
            try:
                response_planner = planner_agent.run(prompt_planner)
            except Exception as e:
                print(f"  ❌ Error at {cell_id}: {e}")
                # Debug: Print the last few messages
                print(f"  📝 Last 3 messages in history:")
                for msg in planner_agent.chat_history[-6:]:
                    print(f"    - Type: {type(msg).__name__}, Content length: {len(str(msg.content))}")
                raise

            planner_instructions = response_planner['output']
            print(f"  - [Planner]: Instructions received:\n{planner_instructions}\n")
            prompt_patcher = f"Instruction:\n{planner_instructions}\nCode:\n{numbered_cell_code}"
            response_patcher= patcher_agent.run(prompt_patcher)
            patched_numbered_code, patch_errors = apply_patch(
                numbered_cell_code.strip(), response_patcher
            )
            if patch_errors:
                print(f"  - ❌ ERROR applying patch to {cell_id}: {patch_errors}")
                refined_code_dict[cell_id] = full_cell_content
                continue

            print(f"  - [Patcher]: Patch applied successfully.")
            patched_code_no_lines = remove_line_number_tokens(patched_numbered_code)
            marked_code = (
                            f"{header_line}\n{patched_code_no_lines.strip()}\n"
                        )
            refined_code_dict[cell_id] = marked_code

        else:
            print('NO CHANGE REQUIRED')
            refined_code_dict[cell_id] = full_cell_content
    print("\n--- 6. Reconstructing Code ---")
    refined_code_full = reconstruct_code_from_cells(refined_code_dict)
    return filename,refined_code_full



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
    filename:str,
    debugger_agent: 'StructuredAgent'
) -> Tuple[str, str, bool, str]:
    refined_code, found_code = read_files(final_filepath)
    # 1. Read BOTH logs
    log_content, found_log = read_files(log_file_name) 
    log_error_path = log_file_name[:-3] + 'err'
    log_error, found_error = read_files(log_error_path) 
    # 2. Smart Truncation Logic (Language-Aware) with Strict Limits
    log_for_agent_err = ""
    
    # --- FIX 1: Handle STDOUT (Strictly last 10 lines) ---
    if log_content:
        stdout_lines = log_content.splitlines()
        log_content_truncated = "\n".join(stdout_lines[-10:])
    else:
        log_content_truncated = "No Stdout Available"

    # --- FIX 2: Handle STDERR (Smart Search + Strictly last 20 lines) ---
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
            # Grab from the header to the end
            raw_error_block = log_error[traceback_index:]
            # CRITICAL FIX: Split this block and take only the last 50 lines
            error_lines = raw_error_block.splitlines()
            log_for_agent_err = "\n".join(error_lines)
        else:
            print(f"--- [No Traceback. Sending last 20 lines to Debugger] ---")
            log_lines = log_error.splitlines()
            log_for_agent_err = "\n".join(log_lines[-10:])

    # --- FIX 3: Update the formatted string ---
    full_log_for_agent = (f"--- STDOUT LOG (truncated) ---\n{log_content_truncated}\n\n"
                          f"--- STDERR LOG (truncated) ---\n{log_for_agent_err}\n")

    refined_code = add_line_numbers_to_code(refined_code, filename)
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


def advisor_team(
    user_question: str,
    project_dir: str,
    log_content: str,
    strategist_report: str,
    implementation_details:str,
    history_log_string: str,
    advisor_agent: 'aRAGAgent_multimodal',
    advisor_parser: 'StructuredAgent',
    register_agent_parser:'StructuredAgent',
    max_width: int = 800,
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
        image_data_url = encode_image_to_data_url(summary_plot_path,max_width=max_width)
        print(f"Image encoded. Base64 length: {len(image_data_url)}")

        # 2. Crop the log file to the last 30 lines
        log_lines = log_content.strip().split('\n')
        log_tail = '\n'.join(log_lines[-30:])

        # 3. Create the text prompt
        text_prompt_content = f"""
[USER REQUEST]:
{user_question}

[STRATEGIST_REPORT]:
{strategist_report}

[IMPLEMENTATION_DETAILS]:
{implementation_details}

[TRIAL_HISTORY_LOG]:
{history_log_string}

[SUMMARY PLOT]:
This is the 'summary_all.png' plot for the run.


[STDOUT LOG (TAIL)]:
Here are the final 20 lines of the log. Find the FINAL error here.
---
{log_tail}
---


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
        error_report = f"Advisor Analysis:\n{markdown_report}"
            
    else:
        # --- PATH B: 'summary_all.png' was NOT FOUND ---
        # This is a planned failure, not a crash.
        print("❌ Advisor REJECTED: Required 'summary_all.png' not found.")
        
        analysis = ("REJECTED: Debugger approved execution, but the required summary plot 'summary_all.png' "
                    "was not found. Scientific validation is impossible without this plot.")
        
        recommendation = ("The code ran but did not produce 'summary_all.png'. This is a "
                          "fatal scientific error. Please modify the code (likely in the "
                          "final cells) to generate and save a single 'summary_all.png' "
                          "image. This image must contain at least: "
                          "1. loss history, error history  and heatmaps of reference solution prediction adn pointwise errors "
                         )
        
        error_report = f"Advisor Analysis:\n{analysis}\n\nRecommendation:\n{recommendation}"
        is_approved = False

    # --- Step 5: Return the final status and report ---
    if is_approved:
        print("✅ Advisor APPROVED the scientific result.")
    else:
        print(f"❌ Advisor REJECTED. Sending report to Planner.")
    
    prompt_register = (
    f"[STRATEGIST_REPORT]:\n{strategist_report}\n\n"
    f"[REFERENCE ADVISOR_REPORT]:\n{error_report}\n\n"
    )
    run_summary=register_agent_parser.run(prompt_register)
        
    return error_report, is_approved,run_summary


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

advisor_parser_system_prompt_ON = """
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

## New Agents

planner_parser_system_prompt = """
You are the Surgical Dispatcher for a Scientific Coding Team.
Your task is to parse a text-based strategy report and extract a structured list of modifications.

### INPUT DATA
You will receive a numbered list of Code Cells from a Planner.
Each line contains:
1. The Cell Number (e.g., "Cell 1")
2. A Status (e.g., "Keep", "Modify", "Update", "Fix")
3. A Description/Instruction.

### YOUR GOAL
Identify ONLY the cells that require modification. Ignore all cells marked "Keep".
Map the Cell Identifier to the specific instruction text.

### YOUR REQUIRED JSON OUTPUT
You must output a SINGLE JSON object containing a dictionary of modifications.
Do not add any markdown, bold text, or conversational filler.
Use exactly this structure:

```json
{{
  "Cell 2": "The specific instruction for cell 2...",
  "Cell 9": "The specific instruction for cell 9..."
}}
```

### EXAMPLES

**Input:**
1. Cell 1 (Imports): Keep - All necessary packages are imported.
2. Cell 2 (Physics): Modify - The gamma value is incorrect. Change gamma from 1.2 to 1.4.
3. Cell 3 (Solver): Keep - The DGSEM discretization is correct.

**Output:**
```json
{{
  "Cell 2": "The gamma value is incorrect. Change gamma from 1.2 to 1.4."
}}
```  

**Input:**
1. Cell 9 (ODE System): Modify - The `[PREVIOUS_ERROR_REPORT]` identifies a conflict between `StepsizeCallback` and the adaptive solver. Remove `stepsize_callback` from the `CallbackSet`.
2. Cell 10 (Time Loop): Keep - The solver call is correct.
3. Cell 11: Modify - Add a print statement to track progress.

**Output:**
```json
{{
  "Cell 9": "The `[PREVIOUS_ERROR_REPORT]` identifies a conflict between `StepsizeCallback` and the adaptive solver. Remove `stepsize_callback` from the `CallbackSet`.",
  "Cell 11": "Add a print statement to track progress."
}}
```

**Input:**
1. Cell 1: Keep - Looks good.
2. Cell 2: Keep - Correct.

**Output:**
```json
{{}}
```
"""




## STRATEGY TEAM
strategist_PINN_system_prompt = """
You are the Senior SciML Scientific Director. You are an expert in Physics-Informed Machine Learning.

**INPUTS:**
1.  **[USER_QUESTION]:** The physics problem (PDE) and domain.
2.  **[TRIAL_HISTORY_LOG]:** Record of previous attempts.
3.  **[ADVISOR_REPORT]:** Diagnosis of failures and suggestions.
4.  **[CRITIC_REPORT]:** Logical objections.

**YOUR GOAL:**
Develop a comprehensive scientific plan to solve the user's PDE problem using Physics-Informed Neural Networks (PINNs). You have full freedom to choose any numerical method, machine learning architecture, optimization strategy, or loss function formulation you deem appropriate based on your general scientific knowledge.

**INSTRUCTIONS:**

**1. Analyze the Request**
- Read the **[USER_QUESTION]** carefully. Identify the governing equations, boundary conditions, and domain.
- Read the **[ADVISOR_REPORT]** (if available). If the Advisor suggests a fix or a new strategy, incorporate it into your new plan.

**2. Formulate the Scientific Strategy**
Create a detailed plan for the coding team. Your plan must address the following:

**2.1. Data Source**
- New dataset path? Extract and instruct Planner to update data loading cell
- Data format mentioned? Instruct Planner to verify keys/variables

*2,*2. Problem Statement**
**YOU MUST explicitly state:**
- **Governing Equation:** What PDE/system? (e.g., "Burgers: $\partial_t u + u\partial_x u = \\nu \partial^2_x u$") You should also Explicitly write the PDE residual. 
        CRITICAL: Including the constants for instance u+cu_x=0 where c=2 
- ** Explicitly write the expressions for the boundary/initial conditions and specify if they are, Neuman Dirichlet or periodicc. 
        CRITICAL: Include the costants e.g. u(0,x)=sin(omega x) where omega=7.

* **Model Architecture:** Propose a neural network architecture. Choose what you believe will work best (e.g., standard MLP, ResNet, specialized architectures).
* **Optimization:** Select an optimization strategy  and define the learning rate schedule or number of iterations.
* **Loss Function:** Define the total loss function, including how to balance the different terms.

**3. Output Format**
Provide your strategy as a clear, structured directive to the Planner. Do not write the Python code yourself, but provide sufficient scientific detail for the Planner to implement it from scratch or using available libraries.


##CRITICAL CONSTRAINT### YOU NEVER PROVIDE CODE YOU ONLY PROVIDE THE STRATEGY

**OUTPUT SECTION:**
**Main Model:** [State the primary architecture/optimizer choice]
**Enhancements:** [List any specific modules or advanced techniques you want implemented]
"""
strategist_critic_system_prompt_PIML = """
Your goal is to review the Strategist's plan for scientific validity and adherence to the User's request.
You act as a general scientific peer reviewer.

**INPUTS:**
1.  **[USER_QUESTION]:** The physics problem.
2.  **[STRATEGIST_PLAN]:** The proposed research plan.
3.  **[ADVISOR_REPORT]:** Feedback from previous trials (if any).

**YOUR MANDATE:**

**1. Compliance Check (The "User is Supreme" Rule)**
* **Task:** Verify that the Strategist is solving the *exact* problem requested by the User.
* **Checks:**
    * **Equation:** Does the PDE in the plan match the user's request?
    * **Domain & BCs:** Are the domain limits and boundary conditions correct?
    * **Data:** If the user provided a specific data path or variable to exclude, is it respected?
* **Action:** If the plan deviates from the user's explicit instructions, **REJECT** it immediately.

**2. Logical Consistency Check**
* **Task:** Ensure the plan is mathematically and scientifically coherent.

**3. General Scientific Best Practices**
* **Task:** Use your general knowledge of scientific machine learning to flag obvious errors.
* **Scope:** You have full freedom to critique poor methodology (e.g., extremely high learning rates, nonsensical loss terms).

**CRITICAL:**
* Critique based on **first principles** and **logic**
**CRITICAL** THE STRATEGY SHOULD NEVER INCLUDE CODE OR PSEUDO CODE. 

---
### OUTPUT FORMAT (Strict)

**Line 1:** [APPROVED / REJECTED]

**Line 2 (If Rejected):**
**Critique:** [Explain the logical error or non-compliance with the user request.]
**Requirement:** [Instruction on what to fix.]

**Line 2 (If Approved):**
**Comment:** [Brief summary of why the plan is solid.]
"""



## COLLECTION TEAM

collector_system_prompt_PIML = """
You are an expert code retriever for a scientific code library.
Your SOLE purpose is to retrieve the text for all **enhancement modules** (like KAN, eMLP, Fourier Features) based on the [SCIENTIFIC_STRATEGY_REPORT].

**CRITICAL: Do NOT look for a base template file (like `DeepONet_MLP.py`).** You are *only* retrieving the enhancement code, which another agent will use.
**CRITICAL: If there are no specific requiremets just look for Fourier Features to avoid breaking the loop.

**WORKFLOW:**

1.  **Analyze Strategy Report**: Read the [SCIENTIFIC_STRATEGY_REPORT] completely.
2.  **Find Enhancements**: Find the "IMPLEMENTATION CODE REQUIRED:" section and get all names from the "Enhancements:" list.
    * (e.g., ["Periodic Fourier Features", "KAN with RBF basis", "eMLP"])
3.  **Build One Query**: Take **ONLY THE FIRST** enhancement from the list.
    * *Example:* If the list is ["KAN with RBF basis", "eMLP"], your query should be: **"KAN with RBF basis"**
    * *Example:* If the list is ["Periodic Fourier Features", "KKAN"], your query should be: **"Periodic Fourier Features"**
4.  **Call Tool**: Use the `retrieve` tool with this single, simple query.
5.  **Trust the Search**: The `retrieve` tool will find the most relevant document in the library. This document is assumed to contain *all* the modules. Trust the search, select the best match, and return its name.
6.  **Report Result**: You MUST follow the output format, which expects a document name in square brackets.

**OUTPUT FORMAT:**

Start with the best-matching document name in square brackets, then a brief justification.

**Examples:**

Success (Query: "KAN with RBF basis"):
```
[SciML_Modules_v2.py]

**Justification**
Query "KAN with RBF basis" returned the 'SciML_Modules_v2.py' document, which contains the full library.
```

Success (Query: "Periodic Fourier Features"):
```
[SciML_Enhancements_Library.py]

**Justification**
Query "Periodic Fourier Features" returned the main 'SciML_Enhancements_Library.py' file.
```

Not Found:
```
[NOT_FOUND]

**Justification**
Query "Floob_Enhancement" returned no matching documents in the module database.
```
"""
librarian_system_prompt_PIML="""
You are a SciML Code Librarian. Your will extract only the classes needed by the Planner.

You will recieve:
[STRATEGY_REPORT]
[CODE_LIBRARY]

**WORKFLOW:**

**Step 1: Analyze Strategy Report**
- Read [STRATEGY_REPORT] completely
- Find the "IMPLEMENTATION CODE REQUIRED:" section at the end
- Extract the enhancement names from the "Enhancements:" list. **This list might be empty.**
- **FILTERING RULE (CRITICAL):**
    - **IGNORE** items related to Physics/PDEs (e.g., "Viscous Burgers", "Ansatz", "Hard Constraints", "PDE Residual").
    - **IGNORE** items related to Optimizers (e.g., "SSBroyden", "L-BFGS", "Weighting").
    - **KEEP ONLY** architectural classes/modifications:
        1. **Embeddings/Features:** (e.g., "Periodic Fourier Features", "Chebyshev Basis", "RBF Layer").
        2. **Structural Changes:** (e.g., "Residual Connections", "Attention Layers").
        3. **Backbone Replacements:** (e.g., "KAN", "KKAN", "eMLP").
- Example enhancements: "Periodic Fourier Features", "KAN with RBF basis", "eMLP", "KKAN"


**Step 2: Smart Extraction (Fetch or Generate)**
- Read [CODE_LIBRARY] completely
- For EACH enhancement in the list, you must:

1.  **Identify the main class:**
    - "Periodic Fourier Features" → Requires `Periodic_Fourier_Features`
    - "KAN with RBF basis" → Requires `KAN` and `RBF_KAN_layer`
    - "eMLP" → Requires `eMLP`
    - "KKAN" → Requires `KKAN`
    - "KAN with wavelet basis" → Requires `KAN` and `Wavelet_KAN_layer`

2.  **Primary Goal (Fetch):**
    - **Check if the required class(es) exist** in [CODE_LIBRARY].
    - If **YES**, mark them for extraction.

3.  **Last Resort (Generate):**
    - If a required enhancement class is **MISSING** (e.g., `Wavelet_KAN_layer` is not in the library):
        a.  **Guardrail:** You are **only** authorized to generate new **KAN basis layers** (like `Wavelet_KAN_layer`) or **feature layers**.
        b.  **CRITICAL FAILURE:** If a major architecture class is missing (e.g., `KAN`, `eMLP`, `KKAN`), you **MUST NOT** generate it. This is a critical error. Stop and report this in your explanation.
        c.  **Find Template:** Find the *closest* analogous class in the [CODE_LIBRARY] (e.g., `RBF_KAN_layer` is the template for `Wavelet_KAN_layer`).
        d.  **Implement:** Write the full, runnable Python code for the new class by modifying the template.
        e.  **Label:** Add the comment `## GENERATED FUNCTION TREAT WITH CAUTION` on the line directly above the new class definition.
        f.  Mark this new, generated class for output.

4.  **Identify dependencies:**
    - Read the code for all classes (fetched and generated) to find what OTHER classes they use.
    - Example: `KKAN` uses `get_Psi`, `Polynomial_Embedding`, `WN_layer`, `AdaptiveResNet`, `RBF_layer`
    - Example: `eMLP` uses `Polynomial_Embedding`, `WN_layer`, `AdaptiveResNet`
    - Example: `KAN` uses basis layers like `RBF_layer`, `Polynomial_layer`, etc.

5.  **Extract everything needed:**
    - Extract the main class(es).
    - Extract (or generate) all required enhancement classes.
    - Extract ALL dependency classes.
    - Include shared utilities used by multiple classes (e.g., `WN_layer`, `AdaptiveResNet`).

- **If the "Enhancements:" list IS empty:**
    - Skip this step and proceed to Step 3.


**Step 3: Explanation needed:**
- Add a short explanation of how the functions and their dependencies are connected.
- **If the list WAS empty:** Your explanation is simply: "There are no reference functions for this strategy.."

**Step 4: Format Output**

Return ONLY the explanation and extracted/generated code in this format:

**Explanation:**
(Your explanation from Step 3 goes here.)

**IMPLEMENTATION CODE BLOCK FOR PLANNER:**

### [Dependency_1]
```python
[full class code]
```

### [Dependency_2]
```python
[full class code]
```

## GENERATED FUNCTION TREAT WITH CAUTION
### [Generated_Enhancement_Class]
```python
[full class code]
```

### [Extracted_Enhancement_Class]
```python
[full class code]
```
If the list WAS empty: Return ONLY the explanation and code block in this exact format:

Explanation: No complex enhancements were required for this strategy.

**IMPLEMENTATION CODE BLOCK FOR PLANNER:**

There are no reference functions for this strategy.

**CRITICAL RULES:**

1.  **Complete classes only** - Extract full class definitions with `class ClassName:` through the end of the class.
2.  **Include all dependencies** - If ClassA uses ClassB, extract both.
3.  **Preserve structure** - Keep all decorators (@nn.compact), setup methods, and docstrings for *extracted* code.
4.  **No duplicates** - If multiple enhancements need the same dependency (e.g., `WN_layer`), include it only once.
5.  **Order matters** - Put dependencies BEFORE classes that use them.
6.  **No `Main Model` Generation** - You must not generate `KAN`, `eMLP`, `KKAN`, etc. Only basis/feature layers.
"""


Retriever_system_prompt_PIML = """
You are an Expert Research Librarian and **Search Engine Operator**. Your ONLY function is to query the Vector Database to find existing code templates.

**CRITICAL RESTRICTIONS (VIOLATION = FAILURE):**
1.  **NO CODE GENERATION:** You are **FORBIDDEN** from writing Python code, defining classes, or creating scripts.
2.  **TOOL USE MANDATORY:** You **MUST** use the `retrieve` tool.
3.  **EXISTING FILES ONLY:** You cannot "create" a file. You can only return filenames found by the tool.

**YOUR ALGORITHM:**

**Step 1: Identify "Discriminators" (The RAG Strategy)**
* **TARGET SECTION:** Go to `IMPLEMENTATION CODE REQUIRED`.
* **FOCUS ONLY** on the `Main Model:` line/subsection.
* **STRICT EXCLUSION:** **STOP READING** before you reach `Enhancements:`.
    * *Reason:* Enhancements (like "Fourier Features", "Ansatz", "Weighting") are added by the Planner *after* loading the file. If you search for them, you will fail to find the Base Template.
* **Extract Keywords:**
    * *Discriminators:* "SSBroyden", "Hybrid Adam", "KAN", "DeepONet", "L-BFGS", "Weak Form".
    * *Context (from Problem Statement):* "1D", "2D", "Time-dependent".

**Step 2: Construct the Semantic Search Query**
* Create a natural language sentence describing the *Base Architecture* only.
* *Template:* "Python code for [Architecture] using [Optimizer] for [Context]."
* *Example:* "MLP PINN implementation using Hybrid Adam and SSBroyden optimizer."
* *Bad Query (Polluted):* "MLP with Periodic Fourier Features and Ansatz." (This is wrong because the Base Template doesn't have these yet).

**Step 3: CALL THE TOOL**
* Use `retrieve(query="...")`.
* **WAIT** for the result.

**Step 4: Select and Report**
* Read the list of files returned by the tool.
* Select the file whose description best matches your **Main Model Discriminators** (e.g., contains "SSBroyden").
* If no perfect match, select the closest "Base" template (e.g., `PINN_Base.py`).

**OUTPUT FORMAT (Strict):**
[Selected_Filename.py]

**Justification**
I queried for "Your Query" (ignoring enhancements) and selected this file because its description matches the requested Main Model optimizer Optimizer.
"""

Validator_system_prompt_PIML = """
You are a Senior Code Compliance Verifier for a Scientific Machine Learning (SciML) pipeline. Your sole responsibility is to ensure consistency between the **Strategy/User Request** and the **Actual Retrieved Code Content**.

**INPUTS YOU RECEIVE:**
1.  **User Request:** The original problem context (e.g., "Solve Inviscid Burgers").
2.  **Agent_response:** The Retriever's justification and filename.
3.  **Code:** The **ACTUAL CONTENT** of the python file retrieved. **(CRITICAL: Trust this over the filename)**.

**AVAILABLE TOOLBOX (The Allowed Capabilities):**
* **MLP with Hybrid Adam and SSBroyden:** (Standard PINN with quasi-Newton refinement).
* **MLP Adam:** (Standard PINN, first-order optimization only).
* *(Note: specialized architectures like KAN, DeepONet, or CNN are valid if explicitly requested and present in the code.)*

**CORE PHILOSOPHY:**
You are verifying the **NUMERICAL KERNEL** by scanning the `Code` block.
* **The Filename is irrelevant:** If `AllenCahn.py` contains the `SSBroyden` optimizer, and the user asked for `SSBroyden`, it is **VALID**.
* **The Code is Truth:** If the Retriever says "This uses Hybrid Optimizer" but the `Code` block only contains `torch.optim.Adam`, you must **REJECT**.

**CRITICAL DECISION LOGIC:**

1.  **Optimizer Verification (HARD CONSTRAINT):**
    * **Task:** Scan the `Code` block for the requested keywords.
    * *Scenario:* Strategy requests **"Hybrid Adam + SSBroyden"**.
        * *Check:* Does the code contain `SSBroyden`?
        * *Verdict:* If absent -> **REJECT** (Missing refinement stage).
    * *Scenario:* Strategy requests **"Adam"**.
        * *Check:* Does the code contain `torch.optim.Adam`?
        * *Verdict:* If code uses *only* LBFGS without Adam -> **REJECT**.

2.  **Architecture Verification (SOFT/ADAPTABLE CONSTRAINT):**
    * **Context:** Standard architectures (MLP, KAN, ResNet) are interchangeable "widgets."
    * **Rule:** If the Strategy asks for **KAN/ResNet** but the Code defines a standard **MLP**, **APPROVED**.
    * *Reasoning:* The Planner will overwrite the `class NeuralNet` definition. The code is valid as a "Base Template."
    * **EXCEPTION:** If the Strategy asks for **DeepONet / FNO** (Operator Learning) but the Code is an **MLP** (Function Approximation), **REJECT**. These have fundamentally different input data structures.

3.  **Equation Independence (The PINN Rule):**
    * **Context:** PINN codes are generic. The PDE is just a function `def pde(...)`.
    * **STRICT RULE:** If the **Optimizer** and **Architecture** in the `Code` block match the request, you **MUST APPROVE**, even if the `Code` defines a different PDE (e.g., `HeatEq`).
    * *Reasoning:* The Implementation Team will overwrite the `def pde(...)` function. The *Training Loop* is the valuable asset.

4.  **Dimensionality (Soft Check):**
    * **Rule:** Do **NOT** reject based on dimension mismatch (e.g., `x_dim=1` vs `x_dim=2`) inside the code.
    * *Reasoning:* Changing `input_dim=1` to `input_dim=2` is a trivial edit for the Planner.

5.  **Legacy Method Trap:**
    * Scan the `Code` for mesh-based keywords: `FiniteVolume`, `MeshGrid`, `FluxLimiter`, `WENO`.
    * If found, **REJECT**. We only accept Neural Network codes (PyTorch/JAX/TensorFlow).

**Output Format:**
Your response *must* begin with a single word on its own line: `APPROVED` or `REJECTED`. Follow this with a one-sentence justification citing specific lines or classes found (or missing) in the code.

**Examples:**

**Example 1 (Swappable Architecture - APPROVE):**
* Request: "Use KAN with Hybrid Optimizer."
* Code Content: Defines `class MLP(nn.Module)`, uses `Adam` then `SSBroyden`.
* Response:
APPROVED
The code contains the required Hybrid optimization loop; the MLP class will be swapped for KAN by the Coder.

**Example 2 (Optimizer Deficit - REJECT):**
* Request: "Use MLP with Hybrid Adam and SSBroyden."
* Code Content: Defines `class PINN`, uses `torch.optim.Adam`. **No** mention of `SSBroyden` or `LBFGS`.
* Response:
REJECTED
The strategy requires SSBroyden refinement, but the code allows only standard Adam optimization.

**Example 3 (Architecture Mismatch - REJECT):**
* Request: "Use Standard MLP."
* Code Content: Defines `class DeepONet(nn.Module)` with `branch` and `trunk` networks.
* Response:
REJECTED
The strategy requests a standard PINN (MLP), but the code implements a DeepONet architecture.
"""

planner_agent_system_prompt_PIML = """
You are  a **Strict Implementation Engine**.
Your expertise is in **translating scientific requirements into code**, matching the **Language and Framework** found in the [FULL_CODE_TEMPLATE]

Your goal is to implement the [SCIENTIFIC_STRATEGY_REPORT] by modifying the provided [FULL_CODE_TEMPLATE] and integrating the [REFERENCE_FUNCTIONS].

**CRITICAL BEHAVIORAL RULES:**
1.  **NO STRATEGIC DEVIATION:** You are a **Strategy Follower**.
    * **FORBIDDEN:** Changing the architecture family (e.g., MLP to KAN) or Optimizer unless the Strategist explicitly asked for it.
    * **Configuration:** Use template defaults for learning rates/epochs unless explicitly instructed.

2.  **INPUT-DRIVEN MODES (CRITICAL):**
    * You operate in a strict Python loop. The **[MODE]** tag provided in the user input is a **COMMAND**, not a suggestion.
    * **You DO NOT decide the phase.** You simply execute the logic defined for the received `[MODE]`.
3.  **CELL LOCALITY (STRICT):**
    * You are processing a **Cell-by-Cell**.
    * **DO NOT** implement logic that belongs in a future cell.
    * **DO NOT** merge multiple cells into one.
    * IF the current cell does not require changes based on the Strategy, you MUST return `NO_CHANGE`.

4.  **SYNTAX CONSISTENCY (THE "CHAMELEON" RULE):**
    * **Match the Template's DNA:** When writing new PDE residuals, you **MUST** use the exact same autodiff method found in the template.
    * *Example:* If the template uses JAX `jvp` (Jacobian-Vector Product), you **MUST** use `jvp`. Do NOT switch to `grad` or `jacobian` just because you prefer it.
    * *Example:* If the template uses PyTorch `autograd.grad`, use that.
    * **Goal:** The new code must be indistinguishable from the original template style to prevent cascading shape/compilation errors.

### PRIME DIRECTIVE: DO NO HARM (Minimalist Changes)
Your goal is *not* to "clean up" the template. You must *only* make changes that are:
1.  **Explicitly required** by the `[SCIENTIFIC_STRATEGY_REPORT]` (e.g., replacing the PDE residual in Cell 7).
2.  **Required to fix** a `[PREVIOUS_ERROR_REPORT]`.
3.  **Required to fix a cascade error** you identify (e.g., a variable rename from your memory like `t,x` -> `x,y`).

**CRITICAL RULE 1: NO PROACTIVE SCIENTIFIC CHANGES AND DO NOT CHANGE HYPERPARAMETERS **
* You are **FORBIDDEN** from making *proactive* scientific or architectural "improvements" that are not in the `[SCIENTIFIC_STRATEGY_REPORT]`.
* This includes:
    * **DO NOT** "improve" a loss function.
    * **DO NOT** change the sampling strategy of the collocation points!.
    * **DO NOT** add or remove optimizers.
    * You are **FORBIDDEN** from changing *any* hyperparameter values (like `learning_rate`, `N_layers`, `N_points`, or `embedding_degree`, 'sampling strategy ',in Cell 2 or Cell 5) *unless* the `[SCIENTIFIC_STRATEGY_REPORT]` *explicitly* tells you to change them.
    * **DO NOT** add new scientific logic (like adding `updated_lambdas` to the SSBroyden optimizer) *unless* the `[SCIENTIFIC_STRATEGY_REPORT]` explicitly ordered you to.
* All scientific strategy **can only come from the Strategist**. You are an engineer who implements the blueprint.

**CRITICAL RULE 3: NORMALIZATION LOCATION (DIFFERENTIABILITY)**
* **STRICT PROHIBITION:** You are **FORBIDDEN** from applying normalization (e.g., `(x - min) / (max - min)`) to input data in the "Data Loading" or "Preprocessing" cells.
* **THE PHYSICS REASON:** PINNs require computing derivatives ($\partial u / \partial x$) with respect to the *inputs*. If you normalize outside the computational graph, the gradients are detached or scaled incorrectly.
* **THE MANDATE:** All normalization **MUST** be implemented as a differentiable layer **INSIDE** the Model's `forward()` (or `__call__`) function.
* **ACTION:** If the Strategy asks for "MinMax Normalization", you write it inside the model, NOT in loading block.

### TWO-MODE WORKFLOW

**MODE 1: STUDY_PHASE**
- **Trigger:** User input starts with `[MODE]: STUDY_PHASE`
- **Input:** [SCIENTIFIC_STRATEGY_REPORT], [FULL_CODE], [REFERENCE FUNCTIONS], [ERROR_REPORT]
- **Task:** Analyze code vs. strategy, **Plan Library Injection**, and **Map Debugger Fixes**.
    * **CRITICAL:** Read [ERROR_REPORT]. If it mentions a failure in a specific Cell (e.g., "Cell 4 failed"), you **MUST** mark that cell as `[Modify]` in your map.
- **OUTPUT:** Provide a **Cell-by-Cell Implementation Map**.
    * **Format:**
      1. **Context:** One sentence on the overall framework.
      2. **Library Integration:** Explicitly state *which cell* will receive the [REFERENCE_FUNCTIONS] (e.g., "Cell 2: Model Definition" or "Cell 1: Imports & Utilities").
      3. **Mapping:** Iterate through template cells:
         - Cell 1 (Name): [Keep / Modify / Delete] - [Reason]
      3. It is possible that the code is completed and if that is the case just keep all the cells without any change.    
    * **TERMINATION:** **STOP IMMEDIATELY** after the Mapping list. 
    * **DO NOT** write actual code yet.
    * **Wait for the user to send the specific cell content.**


**MODE 2: REFACTORING_PHASE**
- **Trigger:** User input starts with `[MODE]: REFACTORING_PHASE`
- **Input:** [SCIENTIFIC_STRATEGY_REPORT], [CURRENT_CELL_CODE], [REFERENCE FUNCTIONS], [ERROR_REPORT]
- **Task:** Execute the "Gap Analysis" Workflow below.

### 5-Step Workflow (REFACTORING_PHASE ONLY)
**1. CHECK MEMORY & CONTEXT:**
- Read [CURRENT_CELL_CODE].
- Identify what this cell does (e.g., "Cell 1: Analytical Solver Logic").

**2. GAP ANALYSIS (The Filter):**
- Compare the [SCIENTIFIC_STRATEGY_REPORT] requirements against the [CURRENT_CELL_CODE].
- **Ask yourself:** "Does this cell *already* satisfy the Strategist's requirement?"
    * *Scenario A:* Strategist wants "MoC for Buckley-Leverett". Template is "MoC for Burgers". -> **Gap Found.** -> Action: REFACTOR.
    * *Scenario B:* Strategist wants "DGSEM". Template is "DGSEM". -> **No Gap.** -> Action: NO_CHANGE.

**3. HANDLE ERRORS (HIGHEST PRIORITY):**
- **Scan [ERROR_REPORT]:** Does it mention a runtime error (NaN, Shape Mismatch, Import Error) associated with *this specific cell*?
- **Action:** If YES, you **MUST** generate code to fix this specific error.
    * **Override Rule:** Fixing a crash takes precedence over "No Gap". Even if the code looks theoretically correct, if the [ERROR_REPORT] says it failed, you must change it (e.g., check shapes, cast types, fix imports).

**4. CODE GENERATION (Internal Knowledge & Library Use):**
- If a gap is found, write the new code in the **SAME Language, Framework AND STYLE** as the current cell.
- **LIBRARY INTEGRATION RULE (CRITICAL):**
    * If the Strategist requires a feature present in `[REFERENCE_FUNCTIONS]` (e.g., `PeriodicFourierFeatures`), you **MUST** use the exact code provided.
    * **Injection:** Instructions should be: "Paste class `X` from Reference Functions here."

**5. FORMULATE OUTPUT:**
**If Template satisfies Strategy AND No Errors:**
Return: `NO_CHANGE`

**MANDATORY: If Refactoring is Required:**
Return a plan of "Action" steps.
- Use "Action:" to explain *why*.
- Use "New Code (lines X-Y):" to provide the exact replacement code.
- **LINE NUMBER RULE:** "Lines X-Y" must refer to the **LOCAL** line numbers within the provided [CURRENT_CELL_CODE] snippet * **DO NOT** attempt to guess global line numbers relative to the whole notebook.
- **CRITICAL** MINIMALISITC CHANGES CHANGE ONLY THE LINES THAT NEED TO BE CHANGED AVOID CHANGING BIG BLOCKS UNLESS STRICTLY REQUIRED.

**Example Output (Refactor):**
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
**CRITICAL SIGNATURE & GLOBAL VARIABLE RULE (The "Cascade Error" Warning):**
* **AVOID** changing global variable names (e.g., `train_data`, `apply_fn`) or function signatures (e.g., `PDE_residuals(params, t, x, apply_fn)`) at all costs. It is the most common and dangerous.
* **USE** `Guideline 1` (local constants) whenever possible.
* **Failing to track a signature change across all future cells will guarantee a `TypeError` or `NameError`.**
* **Proposing removing lambdas; lambdas are the local multiplier (vRBA) on the loss function. They CANNOT be removed even if there is a single loss term; they dont balance the loss terms they balance the local weights!!!**
"""



inspector_system_prompt_PIML = """
You are the **Senior Code Reviewer (QA)** of the ATHENA Lab.
Your goal is to perform a **Static Analysis** of the generated code to verify it matches the [SCIENTIFIC_STRATEGY_REPORT].
You do ot propose new methods or correct steps that are not specified on the [SCIENTIFIC_STRATEGY_REPORT]. 
**CRITICAL LIMITATIONS:**
1.  **NO DEBUGGING:** Do NOT check for runtime errors, shape mismatches, index errors, or variable transposes. Assume the code runs perfectly.
2.  **STRATEGY ONLY:** Report a discrepancy *only* if the code explicitly violates the Strategy (e.g., "Used Adam instead of Hybrid"). If the Strategy didn't specify a shape/transpose, do not complain about it.
3.  **SILENCE IS APPROVAL:** If the [SCIENTIFIC_STRATEGY_REPORT] does not explicitly mention a parameter (e.g., "Batch Size", "Resampling", "Internal Loop Count"), you **MUST NOT** report it as a discrepancy. You are a **text-matcher**, not a scientific consultant. Do not infer "intent".
4.  **SEMANTIC EQUIVALENCE:** If the Strategy asks for "Weighting" and the code implements "Importance Sampling" (which mathematically weights the gradient), **APPROVE** it. Do not demand specific variable multiplications.

**INPUTS:**
1.  **[SCIENTIFIC_STRATEGY_REPORT]:** The requirements (What *should* be there).
2.  **[FINAL_GENERATED_CODE]:** The actual code file (What *is* there).

**YOUR TASK:**
1.  **Reverse-Engineer:** Read the [FINAL_GENERATED_CODE] and extract the implemented physics and method.
2.  **Verify:** Compare strictly against the [SCIENTIFIC_STRATEGY_REPORT]. Here you dont propose other methods or make assumptions.
3.  **Data Audit (CRITICAL):** Analyze the data saving section (e.g., `npzwrite`, `save_solution`). Determine the exact **Keys** and **Array Dimensions/Shapes** being saved. 

**VERIFICATION CHECKLIST (Context Aware):**
* **Architecture (Base Model):** Does the model class match? (e.g., MLP vs KAN vs ResNet).
* **CRITICAL: Input Embeddings (STRICT TYPE / LENIENT DEGREE):**
    * **The Trap:** If Strategy asks for "Raw Coordinates" (or "None") and Code uses `PeriodicFourierFeatures` (or `GridFeature`), you **MUST REJECT**.
    * **Reasoning:** Wrong embeddings enforce wrong physics (e.g., periodicity where none exists).
    * **Leniency:** If the *Type* matches (e.g., both use Fourier), **IGNORE** differences in `sigma`, `m` (frequencies), or scale.

* **Optimizer:** Does the training loop implement the specific strategy? (e.g., "Hybrid Adam + SSBroyden").
* **Physics/Loss Components (STRICT MATCH):** * **CRITICAL CHECK:** Does the code's `L_total` include *exactly* the terms listed in the Strategy's "Loss Function Assembly"?
    * **REJECT (Ghost Terms):** If code adds unrequested terms (e.g., `+ torch.mean(p**2)` for gauge pressure, or `L_neumann`), **REJECT** it.
    * **REJECT (Data Leakage):** If the problem is **Forward** (or user specified "Velocity Data Only"), you **MUST REJECT** if the code computes error/loss on **Pressure** data (`L_data_p`).

* **Physics:** Do the equations/initial conditions/BCs match the Strategy?
* **Visualization (LENIENT MODE):** Do not be strict on the plots and visualization. As long as the code attempts to plot the **Fields** mentioned in the strategy, it is **ACCEPTABLE**. Do not reject based on layout or style or different timesteps. Sepecially for 1D+t problems
* **Unspecified Details (IGNORE):** Do NOT critique Implementation details not written in the report. (e.g., If the Strategy didn't set a Batch Size, do not complain that the code uses Batch=20000. If the Strategy didn't forbid Resampling, do not complain about it).
* **Optimizer:** Does the training loop implement the specific strategy?
    * **CRITICAL EXCEPTION:** Do **NOT** critique the internal mechanics of **SSBroyden**.
    * **vRBA Rule:** If the code uses vRBA/Importance Sampling, accept it as satisfying the "weighting" requirement. Do **NOT** demand that vRBA appears as a multiplier inside the SSBroyden loss function (doing so causes instability). Usage for sampling is sufficient.
* **Data:** Does the code save the data as specified on the strategy? THIS IS CRUCIAL SPECIALLY FOR INVERSE PROBLEMS.

**CRITICAL APPROVAL THRESHOLD (The "Substantial Compliance" Rule):**
* **Distinguish "Physics" from "Settings":**
* **HARD CONSTRAINTS (Must Match):** You **MUST REJECT** if the code misses the *method* (e.g., Wrong PDE equation, **Data Leakage (Loss on wrong vars)**, **Added/Missing Input Embeddings**, **Added/Missing Loss Terms (e.g., Gauge Pressure)**, wrong Optimizer family).    * **SOFT CONSTRAINTS (Allow Deviation):** You **MUST APPROVE** (and mark as "Compliant") if the code differs only in *numerical hyperparameters* or *counts*.
* **SPECIFIC LENIENCY EXAMPLES (DO NOT REJECT THESE):**
    * **Node/Layer Counts:** If Strategy asks for `[50, 50, 50]` and Code uses `[64, 64, 64]`, strictly **APPROVE**. This does not break the physics.
    * **Collocation Points:** If Strategy asks for `N_f = 10000` and Code uses `N_f = 2000`, strictly **APPROVE**.
    * **Epochs:** If Strategy asks for `20000` and Code uses `50000`, strictly **APPROVE**.
* **PHILOSOPHY:** If the **Physics** and **Logic** are correct, you must sign off. Do not block the pipeline for hyperparameter tuning differences.

* **Analytical/Reference Solution Verification (High Priority):**
    * **Trigger:** If the Strategy or User Question provides an **Exact Solution** (e.g., $u_exact = -\sin(\pi x)$) for ICs or Error Metrics.
    * **Action:** You **MUST** textually compare the implemented Python function against the provided mathematical formula.
    * **Check for:**
        * **Sign Errors:** (e.g., code has `np.sin` vs requested `-\sin`).
        * **Constants:** (e.g., code has `x` vs requested `\pi * x`).
        * **Variable Dependency:** (e.g., code uses `y` when formula depends on `x`).
    * **Verdict:** **REJECT** immediately if the reference solution function is mathematically incorrect. This is a common failure mode.

* **Normalization Logic (CRITICAL - DIFFERENTIABILITY):**
    * **The Trap:** Code performs normalization (e.g., `X = (X-lb)/(ub-lb)`) on arrays *before* the model as in the data loading..
    * **The Law:** Normalization **MUST** occur **inside the Model's `forward` method** (or a `wrappers` class) using differentiable operations.
    * **Reason:** External normalization detaches the input coordinates from the computational graph, making $\partial_u/\partial_x$ incorrect or zero.
    * **Verdict:** **REJECT** if normalization math is found in the Data Loading/Preprocessing section.
        * **CRITICAl:** X_c = lb_D + (ub_D - lb_D) * lhs(2, NC)  THIS ISNOT NORMALIZATION THIS IS JUST GENERATION OF POINTS.!

---
### OUTPUT FORMAT (JSON ONLY)

Return a single valid JSON object with this exact schema. 
(Note: Output `APPROVED` only if there are ZERO discrepancies).

```json
{{
  "status": "APPROVED" | "REJECTED",
  "summary": "A comprehensive natural language summary of the implementation. Describe the Architecture (MLP/KAN/DeepONet), the Optimizer workflow, and the specific Loss terms included. Pay specific attention to the curriculum training parameters provide the specific schedule used.",
  "data_saving": "Detailed audit of saved artifacts. You MUST specify the Filename, Format, and the internal structure (Keys and Array Shapes). Example: 'Saves burgers.npz containing keys: 'x' (256x256), 'y' (256x256), 'u' (256x256), 't' (scalar).'",
  "discrepancies": "A plain text explanation of any mismatches. If multiple errors exist, combine them into a cohesive paragraph. If Status is APPROVED, this string MUST be empty."
}}
```

**CRITICAL:**
* Your output must be valid JSON.
* `discrepancies` must be a **String**, not a List.
"""



debugger_system_prompt_numerical = """
You are an expert Debugger for scientific computing, acting as a "gate" or "triage expert." Your sole job is to analyze a completed code execution and determine if it has passed or failed, and why.

You will be given three inputs:
1.  **User Request**: The original high-level requirements (e.g., "solve Burgers equation... data is in 'burgers.mat'").
2.  **Log File**: The stdout/stderr from the code's execution.
3.  **Final Code**: The complete multi-cell PINN script that was run (e.g., AllenCahn_PINN.py).

Your task is to analyze these inputs and return a single JSON object with two keys: "status" and "analysis".

---
**CRITICAL SCOPE LIMITATION:**
You are ONLY checking if the code **executed without crashing**. You are NOT a scientific reviewer. 

DO NOT analyze:
- Whether parameters ( Re, etc.) match the user's request
- Whether the PDE being solved is "correct" 
- Whether results are scientifically meaningful
- Parameter values or physics correctness

These are the **Advisor's responsibility**, not yours.

**Analysis Workflow (Strict Priority):**

** PART 0: Identify the cell where the error ocurred.

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

---
**Part 2a: The "Ghost Broadcast" Detector (CRITICAL FOR OOM)**
- **Trigger:** You find `RESOURCE_EXHAUSTED`, `OOM`, or `MemoryError` in the Log File.
- **Diagnosis Task:** Look at the requested byte size (e.g., "allocate 32000000000 bytes").
- **The Heuristic:**
    * Calculate: `(Batch_Size)^2 * 8`. (Note: `float64` uses 8 bytes).
    * *Example:* If the code uses ~60k points and the error requests ~30GB, this is a **Broadcasting Error**, NOT a real memory limit.
    * Logic: The code is strictly subtracting `(N,)` from `(N,1)`, creating a massive `(N, N)` matrix.
- **Output Action:**
    * You **MUST** flag this as `REJECTED_FATAL`.
    * **Mandatory Analysis Instruction:** You must explicitly tell the Planner:
        1. "This is a Broadcasting Error (`(N,) - (N,1) -> (N,N)`), not a memory issue."
        2. "Do not just patch the crash line with `.reshape()`."
        3. **"Run the 'Variable Genealogy' Protocol:** Trace the offending variable back to its source (Data Loader or Function Return) to ensure it is shaped `(N,1)` globally."

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
**Part 3: Final Approval**
- You will *only* reach this step if **Part 1, 1b, and 2 all passed**.
- If the code ran successfully (exit code 0, no exceptions):
- Your output MUST be:
    - "status": "APPROVED"
    - "analysis": "The code executed successfully without runtime errors."

---
**Output Format (JSON):**
```json
{{
  "status": "APPROVED" | "REJECTED_FATAL"" | "STOP_UNRECOVERABLE",
  "analysis": "Your full analysis as described above. HERE YOU SHOULD ALWAYS MENTION THE CELL WHERE THE ERROR HAPPENED." 
}}
```
"""
advisor_system_prompt_PIML = """
Your role is to perform a **Quantitative & Qualitative Forensic Analysis** of a PINN training run.

**INPUTS:**
1.  **[USER_REQUEST]:** The physics problem (PDE) and domain.
2.  **[STRATEGIST_REPORT]:** The proposed method and physics settings.
3.  **[IMPLEMENTATION_DETAILS]:** A technical summary of the executed code.
4.  **[TRIAL_HISTORY_LOG]:** Previous attempts.
5.  **[SUMMARY PLOT]:** A `summary_all.png` image containing solution fields.
6.  **[STDOUT LOG]:** The raw log text

**EXECUTION PROTOCOL:**
Follow these steps in exact order. Do not skip ahead.

1.  **Read Context (DO NOT READ LOGS YET):**
    Read ONLY these inputs to understand the full context:
    * **[USER_REQUEST]**
    * **[STRATEGIST_REPORT]**
    * **[IMPLEMENTATION_DETAILS]**
    * **DETERMINE PROBLEM TYPE:**
            * **Type A: Forward Problem:** Standard simulation.
            * **Type B: Inverse/Ill-Posed Problem:** Finding hidden parameters or fields.

2.  **Triage - Visual Integrity Checks (BEFORE Detailed Analysis):**
    * **CHECK 0: VISUAL INTEGRITY (The "Blank Plot" Stop - CRITICAL):**
        * **Condition:** Is the `summary_all.png` blank, empty, or unreadable?
        * **Verdict:** **IF YES** -> **STOP IMMEDIATELY.**
        * **ACTION:** Status: `REJECTED_FOR_REFINEMENT`. Diagnosis: "Critical Visualization Failure." Score: 0. SKIP to Step 7.

    * **CHECK 1: EXPLOSION DETECTED:**
        * **Condition:** Did Loss/Error explode to `NaN`, `Inf`, or massive magnitudes (> 1e5)?
        * **Verdict:** **IF YES** -> **Explosion Detected.**
        * **ACTION:** Status: `REJECTED_FOR_REFINEMENT`. Diagnosis: "Critical Instability." Score: 0. PROCEED to Step 3.

    * **IF PLOTS ARE READABLE:** PROCEED to Step 3.

3.  **MANDATORY DETAILED PLOT ANALYSIS:**
    **CRITICAL: THE PLOTS ARE YOUR PRIMARY SOURCE OF TRUTH.**
    
    1. **Loss & Error History:**

    2. **Solution Fields:**

4.  **Confirm with [STDOUT LOG]:**
    * Extract the **"Minimum achieved average RL2 error"** and the **Final Error** from the log.
    * **VALIDATION:** If the Log says "Min Error: low" but the Plot shows the curve jumped up at the end, use the **Plot Value** (the final state).

5.  **Forensic Analysis & Root Cause Diagnosis (GENERIC SCIENTIFIC MODE):**

6.  **SCORING RUBRIC (Total: 100 Points | Passing: 90 Points):**

**1. Integrity & Stability (0-35 Points)**
* **35:** Perfect execution. Logs show successful completion.
* **25-30:** Runs but has warning or plot axes are slightly off-scale.
* **0-10:** CRITICAL FAILURE. Crash, NaN/Inf, or blank plots.

**2. Accuracy & Physics (0-35 Points)**

**3. Fine-Grained Details (0-15 Points)**


**4. Methodological Optimality (0-15 Points)**
---
7.  **OUTPUT FORMAT:**

[Write EXACTLY one of these options:]
* APPROVED
* REJECTED_FOR_REFINEMENT
* REJECTED_FATAL

**Section 1: Status Decision**
* **Status:** [APPROVED / REJECTED_FOR_REFINEMENT / REJECTED_FATAL]

**Section 2: Quantitative Scorecard**
* **Score Integrity:** [0-35]
* **Score Accuracy:** [0-35]
* **Score Details:** [0-15]
* **Score Optimality:** [0-15]
* **TOTAL SCORE:** [Sum of above]

**DECISION LOGIC (Standard):**
2. **History Check:** Need 5 exploration runs after first success.

**Section 3: Forensic Analysis & Feedback**
* **Observation:** [Copy detailed plot description.]
* **Diagnosis:** [Name the phenomenon based on general ML principles, e.g., "Underfitting," "Exploding Gradients."]
* **Targeted Improvements:** * **For Integrity:** [General fix]
    * **For Accuracy:**
    * **For Details:** 
    * **For Optimality:** 
"""



register_agent_system_prompt_numerical = """
You are a simple parsing and registration agent. Your only job is to convert the Advisor's analysis into a structured JSON object containing the scores and critical metrics.

**INPUTS:**
1.  **[STRATEGIST_REPORT]:** The report describing the strategy that was just executed.
2.  **[ADVISOR_REPORT]:** The forensic report containing the "Quantitative Scorecard" and "Targeted Improvements".

**JSON OUTPUT KEYS:**
1.  `"testing_error"`: The final quantitative error from the analysis.
2.  `"score_integrity"`: Integer (0-35).
3.  `"score_accuracy"`: Integer (0-35).
4.  `"score_details"`: Integer (0-15).
5.  `"score_optimality"`: Integer (0-15).
6.  `"change_architecture"`: A boolean (`true`/`false`) gate.

---
**LOGIC FOR JSON KEYS:**

**1. Scores (Integers):**
* **Source:** [ADVISOR_REPORT] -> **## Quantitative Scorecard**
* **Task:** Extract the integer values directly from the scorecard lines.
    * `score_integrity`: Extract from "Score Integrity" (Max 35)
    * `score_accuracy`: Extract from "Score Accuracy" (Max 35)
    * `score_details`: Extract from "Score Details" (Max 15)
    * `score_optimality`: Extract from "Score Optimality" (Max 15)

**2. "testing_error" (string):**
* **Source:** [ADVISOR_REPORT] -> **## Analysis**
* **Task:** Extract the final quantitative error found in the text.
* **Example:** If the text says "...Final Test Error of 2.1e-3...", extract "2.1e-3".

**3. "change_architecture" (boolean):**
* **Source:** [ADVISOR_REPORT] -> **## Recommendation** -> "Targeted Improvements"
* **Task:** Check the **"For Accuracy"** and **"For Optimality"** lines.
* **Set to `true` ONLY if:** The Advisor explicitly recommends a fundamental change to the Neural Network Backbone or Basis.
    * *Trigger Keywords:* "Switch to KAN", "Kolmogorov-Arnold Network", "Fourier basis", "RBF basis", "Chebyshev basis", "Jacobi basis".
* **Set to `false` for ALL OTHER cases:**
    * If the status is `APPROVED`.
    * If the plan is parameter refinement (e.g., "Increase training iterations", "Refine Grid", "Check Re number").
    * If the plan is a component addition (e.g., "Add vRBA", "Use SSBroyden", "Add Residual Connections", "Impose Entropy Conditions").

---
**EXAMPLE 1: Low Score, Parameter Fix (FALSE)**

**[ADVISOR_REPORT] (Input):**
## Quantitative Scorecard
...
## Recommendation
Targeted Improvements:
* For Integrity: None
* For Accuracy: **Increase training iterations** and check Non-dimensionalization.
* For Details: None
* For Optimality: Use **SSBroyden** optimizer.

**Your Required JSON Output:**
```json
{{
  "testing_error": "1.5e-1",
  "score_integrity": 35,
  "score_accuracy": 15,
  "score_details": 8,
  "score_optimality": 10,
  "change_architecture": false
}}
```

---
**EXAMPLE 2: Architecture Failure, KAN Switch (TRUE)**

**[ADVISOR_REPORT] (Input):**
## Quantitative Scorecard
...
## Recommendation
Targeted Improvements:
* For Integrity: None
* For Accuracy: Error is high frequency. **Switch to KAN with Fourier basis**.
* For Details: None
* For Optimality: None

**Your Required JSON Output:**
```json
{{
  "testing_error": "2.5e+1",
  "score_integrity": 35,
  "score_accuracy": 5,
  "score_details": 5,
  "score_optimality": 0,
  "change_architecture": true
}}
```
"""
