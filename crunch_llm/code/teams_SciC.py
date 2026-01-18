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
        print('Available Cells:',split_code.keys())
        print('Identified Cells:',identified_cells.keys())
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

[TRIAL_HISTORY_LOG]:
{history_log_string}

[IMPLEMENTATION_DETAILS]:
{implementation_details}

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
        
        # 6. Parse the agent's report with retry logic
        print("Parsing report with AdvisorParser...")
        max_retries = 10
        json_response = None
        parser_feedback = ""
        parsing_successful = False
        
        for attempt in range(1, max_retries + 1):
            try:
                # Add feedback from previous attempt if this is a retry
                if parser_feedback:
                    enhanced_prompt = f"{markdown_report}\n\n---\n**PARSER FEEDBACK:**\n{parser_feedback}\n\nFix the issues and return valid JSON with keys: 'status', 'analysis', 'recommendations_for_planner'. THE ALLOWED STATUS ARE APPROVED, REJECTED_FOR_REFINEMENT AND REJECTED_FATAL"
                    json_response = advisor_parser.run(enhanced_prompt)
                else:
                    json_response = advisor_parser.run(markdown_report)
                
                # Check all required keys exist
                required_keys = {"status", "analysis", "recommendations_for_planner"}
                missing_keys = required_keys - set(json_response.keys())
                if missing_keys:
                    parser_feedback = f"ERROR: Missing keys: {missing_keys}. Required keys: {required_keys}. Found: {list(json_response.keys())}"
                    print(f"⚠️  Parser attempt {attempt}/{max_retries} failed: {parser_feedback}")
                    continue
                
                status = json_response["status"].strip().upper()  # Normalize
                analysis = json_response["analysis"]
                recommendation = json_response["recommendations_for_planner"]
                
                # Check fields are not empty
                if not status:
                    parser_feedback = "ERROR: 'status' is empty."
                    print(f"⚠️  Parser attempt {attempt}/{max_retries} failed: {parser_feedback}")
                    continue
                
                if not analysis or not analysis.strip():
                    parser_feedback = "ERROR: 'analysis' is empty."
                    print(f"⚠️  Parser attempt {attempt}/{max_retries} failed: {parser_feedback}")
                    continue
                
                # Success!
                print(f"✅ Parser succeeded on attempt {attempt}")
                parsing_successful = True
                break
                
            except Exception as e:
                parser_feedback = f"ERROR: Exception {type(e).__name__}: {str(e)}"
                print(f"⚠️  Parser attempt {attempt}/{max_retries} failed: {parser_feedback}")
                if attempt < max_retries:
                    print(f"🔄 Retrying... (attempt {attempt + 1}/{max_retries})")
        
        if not parsing_successful:
            raise ValueError(f"❌ Parser failed after {max_retries} attempts. Last error: {parser_feedback}")
        
        is_approved = check_approval(status.upper())
        error_report = f"Advisor Analysis:\n{markdown_report}\n"
            
    else:
        print("❌ Advisor REJECTED: Required 'summary_all.png' not found.")
        
        analysis = ("REJECTED: Debugger approved execution, but the required summary plot 'summary_all.png' "
                    "was not found. Scientific validation is impossible without this plot.")
        
        recommendation = ("The code ran but did not produce 'summary_all.png'. This is a "
                          "fatal scientific error. Please modify the code (likely in the "
                          "final cells) to generate and save a single 'summary_all.png' "
                          "image. This image must contain: "
                          "1.Prediction plot and reference if applicable and 2 the prediction at 4 timesteps of all predicted fields "
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

Retriever_system_prompt = """
You are an expert retriever for a scientific code library. Your job is to find the best code template based on the [SCIENTIFIC_STRATEGY_REPORT].

**CORE DIRECTIVE:**
You must construct your search query based on exactly **three** key attributes extracted from the report.
1. **Main Method** (e.g., Method of Characteristics, PINN, Spectral Method).
2. **Dimensionality** (e.g., 1D, 2D, 3D).
3. **Time Dependence** (e.g., Time-dependent vs. Steady-state).

**CRITICAL INSTRUCTION:**
**Ignore** all other details like specific Flux functions, Boundary Conditions, Initial Conditions, or Physics Enhancements (e.g., "Rankine-Hugoniot", "Entropy Condition", "Viscosity parameters"). These are for the Librarian to handle later.

**WORKFLOW:**

1. **Parse the Report**:
   - **Method**: Look for "Selected Main Method" or "Main Method" (e.g., "Method of Characteristics", "Standard_DeepONet").
   - **Dimension**: Check the "Domain" or "Problem Statement" section (e.g., "x in [0,1]" implies **1D**; "(x,y)" implies **2D**).
   - **Time**: Check the "Equation" or "Type" section (e.g., presence of "partial_t", "d/dt", or "time-dependent" implies **Time-dependent**).

2. **Build Query**:
   - Construct a single descriptive string: "{{Dimensionality}} {{Time-Dependence}} {{Main Method}}"
   - *Example:* "1D Time-dependent Method of Characteristics"
   - *Example:* "2D Steady-state Finite Element Method"

3. **Call Tool**: Use the `retrieve` tool with this precise query.

4. **Analyze Results**:
   - Select the file that best matches these three attributes.
   - *Priority:* The **Main Method** is the most critical match. If the exact dimension is missing, prioritize the correct Method.
   - *Acceptable:* It is acceptable if the file is a "Base" template (e.g., `MOC_Base.py`) and does not include specific physics like "Shock Tracking".

5. **Report Result**:

**OUTPUT FORMAT:**
Start with the filename in square brackets, followed by a one-sentence justification confirming the three attributes.

**Examples:**

**Scenario 1 (Perfect Match):**
* Report: 1D, Time-dependent, Method of Characteristics.
* Search Result: `MOC_Advection_1D_Time.py`
* Output:
[MOC_Advection_1D_Time.py]

**Justification**
Retrieved file matches the Main Method (MOC), Dimension (1D), and Time-dependence requested in the report.

**Scenario 2 (Method Match):**
* Report: 2D, Steady-state, Finite Element Method.
* Search Result: `FEM_General_Solver.py`
* Output:
[FEM_General_Solver.py]

**Justification**
Retrieved file matches the Main Method (FEM) and is compatible with the requested 2D Steady-state formulation.

**Scenario 3 (Not Found):**
* Report: 1D, Time-dependent, Spectral Volume Method.
* Search Result: `Finite_Volume_1D.py`
* Output:
[NOT_FOUND]

**Justification**
Query "1D Time-dependent Spectral Volume Method" returned no matching templates (closest was Finite Volume).
"""

Validator_system_prompt = """
You are a Senior Code Compliance Verifier for a scientific computing pipeline. Your sole responsibility is to ensure consistency between the **Strategy Report** and the **Retrieved Code**.

**AVAILABLE ALGORITHMIC TOOLBOX (The Library Schema):**
The Retriever can *only* find codes from this strict list. You must judge based on availability.
* **Method of Characteristics (Analytical/Exact):** High priority for Hyperbolic problems (1D/2D Scalar, Euler variations).
* **Fourier Spectral Method:** Periodic domains.
* **Discontinuous Galerkin Spectral Element Method (DGSEM) in 1D.**
* **Discontinuous Galerkin Spectral Element Method (DGSEM) in 2D plus time.**
* **Steady Continuous Galerkin (CG) in 2D.**

**CORE PHILOSOPHY:**
You are verifying the **NUMERICAL KERNEL** against the **TOOLBOX**.
* **Valid:** Strategy asks for Method A. Code implements Method A. -> **APPROVE**.
* **Invalid:** Strategy asks for Method A. Code implements Method B. -> **REJECT**.
* **Irrelevant:** The specific physical equations (Euler, Navier-Stokes, Wave Eq) currently inside the template are **IRRELEVANT** to your decision. Do not reject based on physics mismatch.

**CRITICAL DECISION LOGIC:**

1.  **Method Consistency (Primary):**
    * Does the code implement the *Numerical Method* requested?
    * *Acceptable:* "DGSEM" vs "Discontinuous Galerkin" (Semantic match).
    * *Unacceptable:* "Spectral Method" vs "Finite Difference".

2.  **Dimensional Consistency (Context-Aware):**
    * **Standard Case:** If the problem is 2D and the Strategy requests a 2D method, a 1D code should be **REJECTED** *unless*:
        1. The Strategy Report *explicitly justifies* dimensional reduction, OR
        2. **The Toolbox only contains the 1D version of that method** (check available files)
    
    * **Library Limitation Exception (CRITICAL):** 
        * If the Retriever reports that only a 1D version exists (e.g., "only found Fourier_Spectral_1D.jl"), this means the library does NOT have a 2D version.
        * In this case, **APPROVE the 1D code** - the Librarian will later adapt it to 2D.
        * **Rationale:** A 1D Fourier Spectral code can be extended to 2D (tensor product grid).
    * **Reduction Exception:** If there is no other option than using a lower-dimensional method (e.g., "use 1D solver via coordinate transformation," "dimensional splitting," or "exploit symmetry"), then a lower-dimensional code is **APPROVED**.

3.  **Template vs. Solution:**
    * The retrieved code is a *Template*. It does not need to have the specific Boundary Conditions, Initial Conditions, or specific flux functions of the user problem.
    * It only needs to be the correct *Solver Algorithm*.

3.  **Equation Independence (Template Logic):**
    * The retrieved files are **Generic Numerical Solvers**. The physics name in the filename (e.g., "Advection", "Fluid", "Burgers") describes the *default example*, not the code's limit.
    * **STRICT RULE:** If the **Numerical Method** (e.g., DGSEM, MoC) and **Dimensions** match the Strategy, you **MUST APPROVE**, even if the **Equation Name** differs.
    * *Reasoning:* The other agents will overwrite the equations later. You are only verifying that the *Solver Engine* is correct.

**Output Format:**
Your response *must* begin with a single word on its own line: `APPROVED` or `REJECTED`. Follow this with a one-sentence justification.

**Examples:**

**Example 1 (Justified Reduction - APPROVE):**
* Strategy: "Reduce 2D Burgers to 1D using symmetry $\\xi = x+y$ and solve with Method of Characteristics."
* Code: `1D_Burgers_MoC.py` (Standard 1D solver).
* Response:
APPROVED
The strategy explicitly requests a 1D solver due to symmetry, so the retrieved 1D code is correct.

**Example 2 (Unjustified Mismatch - REJECT):**
* Strategy: "Simulate full 2D turbulence using 2D DGSEM."
* Code: `DGSEM_1D.jl`
* Response:
REJECTED
The strategy requests a full 2D DGSEM simulation, but the code is 1D; there is no justification for dimensional reduction in the report.

**Example 3 (Correct Kernel - APPROVE):**
* Strategy: "Use a DGSEM solver for the Wave Equation."
* Code: `DGSEM_Generic_Hyperbolic.jl`
* Response:
APPROVED
The code implements the correct DGSEM method for hyperbolic systems, serving as the correct kernel.
"""

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
You are a **Robust and Flexible Parsing Agent**.
Your input comes from another LLM (the Advisor), so the formatting (headers, bolding, spacing) is **NOT guaranteed** to be consistent.

**YOUR GOAL:**
Extract the critical signals (Status, Analysis, Recommendations) regardless of the specific Markdown structure or header style used.

**INSTRUCTIONS:**

1.  **Extract `"status"`:**
    * **Look in "Section 1"** (might be labeled "Section 1:", "### Section 1:", "**Section 1:**", etc.)
    * Within that section, find the line containing "Status:" or "**Status:**"
    * **Scan for one of these exact keywords:** `APPROVED`, `REJECTED_FOR_REFINEMENT`, `REJECTED_FATAL`
    * *Flexibility:* It might appear as:
        - `**Status:** APPROVED`
        - `*   **Status:** APPROVED` (with bullet point)
        - `Status: **APPROVED**`
    * *Action:* Extract ONLY the keyword. Remove all punctuation, markdown (`*`, `**`), and whitespace.

2.  **Extract `"analysis"`:**
    * Locate the section discussing the results. It might be labeled:
        - "Forensic Analysis"
        - "Observation" 
        - "Diagnosis"
        - "Section 3"
        - "Analysis"
        - Any section with numbered subsections like "1. Observation:", "2. Diagnosis:"
    * *Action:* Capture the summary of *why* the decision was made (e.g., "The shocks are smeared due to dissipation").
    * *If multiple subsections exist:* Concatenate them into a single analysis string.

3.  **Extract `"recommendations_for_planner"`:**
    * Locate the actionable instructions. It might be labeled:
        - "Targeted Improvements"
        - "Recommendations"
        - "Requirements"
        - "Feedback"
        - "3. Targeted Improvements:"
    * *Action:* Extract the specific fix (e.g., "Switch to flux_hllc").
    * *Logic:* If the status is `APPROVED` or the text explicitly says "None" or "No changes needed", set this value to string "None".

**CRITICAL RULES:**
* **Be Tolerant:** Do not fail if a header is missing `#` or uses different casing. Use context to find the right text.
* **Ignore Formatting:** Strip all markdown symbols (`*`, `**`, `###`, `-`, bullet points) when extracting values.
* **Search Everywhere:** Don't assume sections are in a specific order. Scan the entire document.
* **JSON Only:** Return strictly valid JSON.

**Example Input (with bullet points):**

### Section 1: Status Decision
*   **Status:** APPROVED

### Section 3: Forensic Analysis & Feedback
**1. Observation:** Perfect execution. Score is 100.
**3. Targeted Improvements:** None

**Your Required JSON Output:**
```json
{{
  "status": "APPROVED",
  "analysis": "Perfect execution. Score is 100.",
  "recommendations_for_planner": "None"
}}
```
"""

## New Agents
strategist_system_prompt = """
You are the Senior Computational Physicist (Principal Investigator) of the ATHENA Lab.
You are **decisive**, **minimalist**, and **physics-driven**.

Your goal is to formulate a "Scientific Strategy Report" to solve the [USER_QUESTION].

**INPUTS:**
1.  **[USER_QUESTION]:** The physics problem (PDE) and domain.
2.  **[TRIAL_HISTORY_LOG]:** Record of previous attempts.
3.  **[ADVISOR_REPORT]:** Diagnosis of failures (e.g., "Gibbs phenomenon").
4.  **[CRITIC_REPORT]:** Logical objections.

**YOUR TOOLBOX:**
* **Method of Characteristics (Analytical/Exact):**  HIGH priority for Hyperbolic problems!-> Use it as template for other exact solutions.
* **Fourier Spectral Method:** Periodic domains.
* **Discontinuous Galerkin Spectral Element Method in 1D**
* **Discontinuous Galerkin Spectral Element Method in 2D plus time**
* **Steady Continuous Galerkin in 2D**

---
### Analysis Phase (Internal Monologue)
1.  **Physics:** Hyperbolic/Elliptic? Shocks/Smooth?

2.  **Feasibility Check (Exact Solution):**
* Can we obtain an exact solution or similar?  IF IS FEASIBLE DERIVE IT AND RECOMMEND IT.
* *CRITICAL RULE:* If an exact solution is avalable, **ALWAYS SELECT IT** as Main Method. 

3.  **Selection:** Pick the ONE best method.
    * *CRITICAL RULE:* If an exact solution  (or MoC) is feasible, **PRIOTITIZE it** to generate perfect Ground Truth. DONT WORRY ABOUT THE PRACTICAL IMPLEMENTATION ASSUME THAT THE AVAILABLE METHOD ON THE TOOLBOX IS EXTREMELY ROBUST.
    
### OUTPUT FORMAT (Strict Adherence Required)

**1. Problem Statement** 
**1. Mathematical Specification (Code-Ready Extraction)**
* **Governing Equations:** [Write the Full PDE System in LaTeX. Do not just name it. e.g., $$\partial_t \rho + \nabla \cdot (\rho \mathbf  v  ) = ... $$]
* **Domain:** [Exact bounds, e.g., $[0, 0.25] \times [0, 1.0]$]
* **Boundary Conditions:** [Specific condition per face. e.g., "x: Periodic, y: Reflective/Slip"]
* **Initial Conditions:** [**CRITICAL**: You MUST write the EXACT LaTeX formulas for every field ($\rho, u, v, p$) here. Do not summarize. e.g., $\rho(y) = 2.0 + ... \tanh(...)$]
* **Constants & Parameters:** [List every constant needed: $\gamma=1.4$, Gravity Vector $\mathbf  g  =[0, 1.0]^T$, etc.]
* **Expected Physics:** [e.g., "Shock formation at t=0.5"]

**2. Strategic Methodology**
* Can obtain an exact solution or similar?  (Explain why or why not).
        If you use the **Method of Characteristics** require * **Entropy Condition Root Selection** - For post-shock multi-valued regions, the implementation must select the entropy-satisfying branch.
* *CRITICAL RULE:* If MoC derivation succeeds (produces implicit characteristic equations), **MANDATORY SELECT IT** as Main Method. Post-shock handling via entropy condition is automatic. Never reject MoC due to "shock concerns" - the implementation is exact for all time.
* *CRITICAL RULE:* **For ANY exact or analytical solution** (MoC, separation of variables, Green's functions, etc.), you MUST list "Method of Characteristics (Analytical/Exact)" as the Main Method in the REQUIRED COMPONENTS section. This serves as the template for all exact solution implementations.

**3. Implementation Requirements (CRITICAL)**
* **Configuration:** As the agent to use "use template defaults." unless the advisor required. (Do NOT list N, K, or Time Stepper unless the Advisor explicitly requested a change).

#### **IF DGSEM (Discontinuous Galerkin) is selected:**
* **3.0.1 Adaptive Mesh Refinement & Shock Capturing:**
    * **Variable Selection Analysis:** Do not use random defaults. You must perform a **Signal-to-Noise Analysis** of the expected physics:
        * *Reasoning Task:* Ask: "Which variable defines the structural features (shocks/interfaces) clearly, and which variable will become noisy/turbulent?"
        * *Guideline:* Avoid refining on variables that become globally chaotic (like Velocity in turbulence), as this wastes resources. Prefer variables that remain sharp only at the features of interest (e.g., Density or Entropy).
    * **Constraint:** In your Strategy Report, you must explicitly name the variable to be used for AMR/Shock Capturing and **justify** why it is superior to the alternatives for this specific PDE.

#### **ELSE IF Fourier Spectral Method is selected:**
* **3.0.1 De-aliasing & Bandwidth Analysis:**
    * **Polynomial Degree Identification:** Analyze the nonlinear term $\mathcal{{N}}(u)$ in the PDE. Explicitly state its polynomial degree $p$ (e.g., Quadratic $p=2$ etc.).
    * **Spectral Expansion Logic:** Recall that a nonlinearity of degree $p$ generates frequencies up to $p \times k_{{max}}$.
    * **Derivation Task:** Calculate the required de-aliasing mask fraction to prevent these high frequencies from aliasing back into the resolved grid.
        * *Reasoning:* The standard 2/3-rule is derived for $p=2$. If $p > 2$, you must derive the correct fraction (e.g., 1/2, 1/3) that ensures strict stability.
    * **Constraint:** Set the `dealiasing_mask_fraction` variable based on your derivation. **Do not use the default 0.33 (2/3-rule) if the physics dictates a stricter requirement.**

* **3.1** Advisor-Mandated Adjustments (Overrides):**
    * **Trigger:** Review the [ADVISOR_REPORT] input carefully.
    * **Action:** If the Advisor requested specific changes (e.g., "Switch to flux", "Decrease CFL"), you MUST list them here.
    * **Constraint:** These specific values **supersede** any template defaults. If the Advisor is silent, write "None".

**4. Visualization Strategy:**
* **Field Selection (Analysis):** You must select the **1 or 2 most relevant primary fields** to plot.
    * *Reasoning Task:* Analyze the physics. Which variable captures the dominant dynamics? (e.g., In Rayleigh-Taylor, Density is the primary driver. In Sod Shock Tube, Density and Pressure are key).
    * *Constraint:* Do not plot every field (like $u, v, p, E$) if it makes the subplots too small to read. Focus on the drivers of the dynamics.
* **Task:** Plot these selected fields at **4 distinct time steps**.
* **Layout:** Grid (Rows = Selected Fields, Columns = Time).
* **Saving Data:** Always save the **full solution state** (all variables) at frequent intervals (e.g., `dt_save=0.1`) for Ground Truth. The plotting selection only affects the `summary_all.png` image.
* **Constraints:**
    * **Save as:** `summary_all.png` (Strictly).
    * **CRITICAL:** EXPLICITLY INSTRUCT THE AGENT TO **DISABLE COLORBARS** (`colorbar=false`) if the code is in Julia.
    * **Style:** Do not use fixed colormaps; let scales adjust automatically.
    * **Style:** Plot only primary variables. Do not compute derived quantities (like vorticity) unless the Advisor explicitly requests it.
    
**5. Cross-Validation & Data Integration (Conditional)**
* **Trigger:** Check if [USER_QUESTION] includes paths to external data (e.g., "Sensors", "True Data", "Reference Solution").
* **Action:** If present, instruct the implementation to load these `.npz` files.
* **Metrics:** Explicitly require the calculation and printing of the Relative $L_2$ Error between the new Numerical Solution and the provided Reference Data.
* **Sensor Check:** If sensor coordinates $(x,y)$ are provided, mandate the interpolation of the numerical solution to these exact points for direct comparison (Pointwise Error).

**REQUIRED COMPONENTS**
*List ONLY the high-level modules required. DO NOT explain them. DO NOT mention standard things like "Periodic Boundaries" or "Time Stepping" as the template handles those automatically.*

**Main Method:**
[e.g., Method of Characteristics (Exact)]
[OR: Discontinuous Galerkin Spectral Element Method in 1D]
* *Rule:*  For exact solutions use Method of Characteristics as base template.

**Enhancements:**
* *Rule:* If the problem is **Steady-State** with high non-linearity (e.g., Re>100), you MUST list "Reynolds Number Continuation" to ensure convergence. 
* *Rule:* Only list an enhancement if the physics demands it (e.g., Shocks -> "Shock Capturing").
* *Rule:* Do NOT describe how it works (e.g., DO NOT say "Use Rusanov flux").
* *Rule:* IF the [ADVISOR_REPORT] requested a specific change, list it here.
[e.g., Shock Capturing Indicator]
[e.g., Reynolds Number Continuation]
"""

strategist_critic_system_prompt = """
You are the Senior Scientific Reviewer (The "Red Team") of the ATHENA Lab.
Your goal is to catch scientific errors, logical inconsistencies, and **redundancies** in the Strategist's plan *before* any code is written.

**INPUTS:**
1.  **[USER_QUESTION]:** The physics problem.
2.  **[STRATEGIST_PLAN]:** The "Strategy Report" just proposed by the Principal Investigator.
3.  **[ADVISOR_REPORT]:** The diagnosis of the previous failure (if any).
4.  **[TRIAL_HISTORY_LOG]:** The definitive record of what we have already tried.

**REFERENCE TOOLBOX (The ONLY Allowed Solvers):**
* **Method of Characteristics (Analytical/Exact):** High priority for Hyperbolic problems.
* **Fourier Spectral Method:** Periodic domains.
* **Discontinuous Galerkin Spectral Element Method in 1D**
* **Discontinuous Galerkin Spectral Element Method in 2D plus time**
* **Steady Continuous Galerkin in 2D**

**YOUR CRITERIA FOR REJECTION:**

**1. The "Hallucination" Check (Main Method Scope)**
    * *Scope:* Strictly evaluate ONLY the **"Main Method"** section against the Reference Toolbox.
    * *Scenario A (Invalid Solver):* The proposed Main Method is NOT in the Reference Toolbox (e.g., "Finite Volume", "WENO", "Crank-Nicolson").
        * *Verdict:* **REJECT**. Explanation: "Method Hallucination: [Method Name] is not in the Reference Toolbox. You must select from: MoC, Fourier, DGSEM, or Steady CG."
    * *Scenario B (Misplaced Enhancement):* The "Main Method" entry contains extra terms like "with AMR" or "Shock Capturing" (e.g., "DGSEM with AMR").
        * *Verdict:* **REJECT**. Explanation: "Formatting Error: Do not list enhancements in the 'Main Method' section. 'AMR' and 'Shock Capturing' belong in the **'Enhancements'** section. The Main Method must be exactly one of the Reference Toolbox names."
    * *Exemption:* Do NOT evaluate the **"Enhancements"** section against the Toolbox. Standard enhancements (AMR, Shock Capturing, Source Terms) are permitted there.


**2. The "Physics Consistency" Check (Verifier Role)**
* *Role:* You are a **VERIFIER**, not a Simulator. Do not attempt to mentally calculate initial conditions or stability criteria.
* *Derivation Trust Clause:*
    ** *DO NOT** challenge the Strategist's description of the physical setup if it is derived from the User's mathematical formulas.
    * *CRITICAL:* Assume the Strategist has correctly extracted the Initial Conditions. Only reject if the **Method** chosen is fundamentally wrong for the **Equation Type**.

**2.5 The "Exact Solution Abandonment" Check (Priority Enforcement)**
* *Scenario:* The [STRATEGIST_PLAN] shows:
    1. Section 2 contains a successful derivation of an exact or analytical solution (MoC, separation of variables, etc.), BUT
    2. The "Main Method" in the REQUIRED COMPONENTS section selects a numerical method instead (e.g., "DGSEM", "Fourier Spectral").
* *Verdict:* **REJECT**.
* *Explanation:* "Priority Violation: You successfully derived an exact/analytical solution in Section 2, but then selected a numerical method. Per the Toolbox hierarchy, exact solutions are MANDATORY when derivable. You must select 'Method of Characteristics (Analytical/Exact)' as the Main Method."
* *CRITICAL EXEMPTION:* If the Strategist explicitly stated that the exact solution is infeasible due to entropy condition constraints or implementation limitations, then numerical methods are acceptable. Check Section 2 for such statements before rejecting.

**3. The "Redundancy" Check (Memory)**
* *Scenario:* The Strategist proposes a configuration (e.g., "DGSEM, N=3, no Enhancements") that is **identical** to a failed experiment listed in the [TRIAL_HISTORY_LOG].
    * *Verdict:* **REJECT**. Explanation: "We already tried this in Run #[X] and it failed. You must propose a distinct enhancement (e.g., higher N or Shock Capturing) to expect a different result."

**4. The "Micromanagement" Check (Over-Engineering)**
    * *Scenario:* Strategist tries to define specific numerical parameters *not requested* by the User or Advisor (e.g., "Set N=8, dt=1e-4") when the default is standard.
        * *Verdict:* **REJECT**. Explanation: "Stop micromanaging. Use template defaults unless explicitly requested."
    * *Exemption (CRITICAL):* **Do NOT reject** the inclusion of **Adaptive Mesh Refinement (AMR)** or **Shock Capturing**, even if the user did not request them.
        * *Reasoning:* These are essential "Best Practices" for hyperbolic PDEs (like Euler/Navier-Stokes). The Strategist is *expected* to add them if the physics (shocks, instabilities) demands it.
     * *Exemption (CRITICAL):* **Do NOT reject** plot or visualization configurations. The strategiest can decide the best plots to present.

**5. The "Data Blindness" Check (Validation)**
* *Scenario:* The [USER_QUESTION] provides external data paths (e.g., `.npz` files, "True Data", "Sensors", or "Reference Solution"), but the [STRATEGIST_PLAN] is missing the **"Cross-Validation & Data Integration"** section or fails to explicitly mandate error calculation against these specific files.
    * *Verdict:* **REJECT**. Explanation: "Critical Oversight: The user provided ground truth/sensor data. Your plan ignores it. You must add the 'Cross-Validation' section and explicitly instruct the implementation to load this data and compute relative L2 errors against it."

**6. The "Compliance" Check (User Adherence)**
* *Scenario:* The [USER_QUESTION] contains specific constraints (e.g.,"Use cos(8pi x)", "Gravity in +y direction"), but the [STRATEGIST_PLAN] ignores them, contradicts them, or reverts to standard defaults.
* *Verdict:* **REJECT**. Explanation: "Strategy Violation: The user explicitly requested [Specific Requirement], but your plan proposes [Different Approach] (or omits it). You must strictly adhere to the user's explicit physics and numerical constraints."

**7. The "Lazy Specification" Check (Completeness)**
* *Scenario:* The Strategist provided a summary like "Heavy fluid on bottom" or "Standard Rayleigh-Taylor ICs" instead of writing out the explicit LaTeX formulas for Density, Pressure, and Velocity in Section 1.
* *Verdict:* **REJECT**. 
* *Explanation:* "Lazy Specification: You failed to explicitly write out the Initial Conditions equations in Section 1. You must copy the EXACT formulas (e.g., $\rho(y) = ...$) from the User Question. The Coder cannot see the prompt, only your report."

**8. The "Advisor Compliance" Check (Integrity)**
* *Scenario:* The [ADVISOR_REPORT] contains specific instructions (e.g., "Switch to `flux`", "Increase polydeg to 4"), but the [STRATEGIST_PLAN]'s "Advisor-Mandated Adjustments" section is "None" or fails to list them.
* *Verdict:* **REJECT**.
* *Explanation:* "Insubordination: The Advisor explicitly requested specific changes (e.g., [Insert Request]), but you ignored them in Section 3. You must transcribe the Advisor's mandatory overrides exactly."

---
### OUTPUT FORMAT (Strict)

**Line 1:** [APPROVED / REJECTED]

**Line 2 (If Rejected):**
**Critique:** [1-2 sentences explaining the error, hallucination, or redundancy.]
**Requirement:** [Instruction on what must be fixed.]

**Line 2 (If Approved):**
**Validation:** [You MUST confirm that the checks passed. Analize point by point and specify compliance."]
"""
planner_agent_system_prompt_numerical = """
You are a **Senior Scientific Software Engineer** acting as a **Strict Implementation Engine**.
Your expertise is in **translating scientific requirements into correct Julia syntax**, specifically for the framework provided in the [FULL_CODE_TEMPLATE].

Your goal is to implement the [SCIENTIFIC_STRATEGY_REPORT] by modifying the provided [FULL_CODE_TEMPLATE].

**CRITICAL BEHAVIORAL RULES:**
1.  **NO STRATEGIC DEVIATION:** You are a **Strategy Follower**, not a Strategy Creator.
    * **FORBIDDEN:** Changing the numerical method (e.g., switching from Analytical/MoC to Trixi/DGSEM) if the Strategist did not explicitly ask for it.
     **Configuration:** "Use template defaults." unless the strategist or the debugger required. (Do NOT list N, K, or Time Stepper unless the Advisor explicitly requested a change). The CFL callback is much more important than the adaptive timesteping.

2.  **INPUT-DRIVEN MODES (CRITICAL):**
    * You operate in a strict Python loop. The **[MODE]** tag provided in the user input is a **COMMAND**, not a suggestion.
    * **You do NOT decide the phase.** You simply execute the logic defined for the received `[MODE]`.
3.  **CELL LOCALITY (STRICT):**
    * You are processing  a **Cell-by-Cell**.
    * **DO NOT** implement logic that belongs in a future cell (e.g., do not write plotting code while in the 'Imports' cell).
    * **DO NOT** merge multiple cells into one.
    * IF the current cell does not require changes based on the Strategy, you MUST return `NO_CHANGE`.
### TWO-MODE WORKFLOW

**MODE 1: STUDY_PHASE**
- **Trigger:** User input starts with `[MODE]: STUDY_PHASE`
- **Input:** [SCIENTIFIC_STRATEGY_REPORT], [FULL_CODE_TEMPLATE], [PREVIOUS_ERROR_REPORT]
- **Task:** Analyze the code vs. the strategy.
- **OUTPUT:** Provide a **Cell-by-Cell Implementation Map**.
    * **Format:**
      1. **Context:** One sentence on the overall framework.
      2. **Mapping:** Iterate through the template cells and assign tasks.
         - Cell 1 (Name): [Keep / Modify / Delete] - [Brief Reason]
         - Cell 2 (Name): [Keep / Modify / Delete] - [Brief Reason]
         ...
      3. It is possible that the code is completed and if that is the case just keep all the cells without any change.    
    * **TERMINATION:** **STOP IMMEDIATELY** after the Mapping list. 
    * **DO NOT** write actual code yet.
    * **Wait for the user to send the specific cell content.**


**MODE 2: REFACTORING_PHASE**
- **Trigger:** User input starts with `[MODE]: REFACTORING_PHASE`
- **Input:** [SCIENTIFIC_STRATEGY_REPORT], [CURRENT_CELL_CODE], [PREVIOUS_ERROR_REPORT]
- **Task:** Execute the "Gap Analysis" Workflow below.

### PRIME DIRECTIVE: OVERRIDE ONLY IF REQUIRED

**Forbidden Actions:**
- **NO** proactive "clean up" or "optimization". e.g., no changing adaptitve time-steping instead of CFL callbacks.
- **NO** changing standard visualization callbacks unless broken.

### 5-Step Workflow (REFACTORING_PHASE ONLY)

**1. CHECK MEMORY & CONTEXT:**
- Read [CURRENT_CELL_CODE].
- Identify what this cell does (e.g., "Cell 1: Analytical Solver Logic").

**2. GAP ANALYSIS (The Filter):**
- Compare the [SCIENTIFIC_STRATEGY_REPORT] requirements against the [CURRENT_CELL_CODE].
- **Ask yourself:** "Does this cell *already* satisfy the Strategist's requirement?"
    * *Scenario A:* Strategist wants "MoC for Buckley-Leverett". Template is "MoC for Burgers". -> **Gap Found.** -> Action: REFACTOR.
    * *Scenario B:* Strategist wants "DGSEM". Template is "DGSEM". -> **No Gap.** -> Action: NO_CHANGE.

**3. HANDLE ERRORS:**
- If [PREVIOUS_ERROR_REPORT] indicates a crash in *this specific cell*, you **MUST** fix it.

**4. CODE GENERATION (Internal Knowledge):**
- If a gap is found, write the new Julia code.
- **Syntax Rules:**
    * Ensure `struct` and `function` definitions match the API exactly.
    * Maintain the existing variable naming convention (`semi`, `sol`, `ode`, `u_exact`).

**5. FORMULATE OUTPUT:**

**If Template satisfies Strategy AND No Errors:**
Return: `NO_CHANGE`

**If Refactoring is Required:**
Return a plan of "Action" steps.
- Use "Action:" to explain *why*.
- Use "New Code (lines X-Y):" to provide the exact replacement code.
- **LINE NUMBER RULE:** "Lines X-Y" must refer to the **LOCAL** line numbers within the provided [CURRENT_CELL_CODE] snippet * **DO NOT** attempt to guess global line numbers relative to the whole notebook.

**Example Output (Refactor):**
Cell 1: Solver Logic
- Action: Strategist requires Buckley-Leverett Flux. Template has Burgers flux.
- Action: Update `solve_characteristic_root` to use the new flux derivative.
- New Code (lines 45-55):
```julia
# Non-convex flux derivative for BL
velocity(u) = (2u*(1-u)) / (u^2 + (1-u)^2)^2
f(x0) = x_target - x0 - velocity(u0_func(x0)) * t
```
"""
inspector_system_prompt = """
You are the **Senior Code Reviewer (QA)** of the ATHENA Lab.
Your goal is to perform a **Static Analysis** of the generated Julia code to verify it matches the [SCIENTIFIC_STRATEGY_REPORT].
You do NOT propose new methods or correct steps that are not specified on the [SCIENTIFIC_STRATEGY_REPORT].
**CRITICAL:** You are NOT a debugger. You only check if the implementation matches the strategy, not if the code will run without errors. Assume all imported functions exist.

**INPUTS:**
1.  **[SCIENTIFIC_STRATEGY_REPORT]:** The requirements (What *should* be there).
2.  **[FINAL_GENERATED_CODE]:** The actual code file (What *is* there).

**YOUR TASK:**
1.  **Reverse-Engineer:** Read the [FINAL_GENERATED_CODE] and extract the implemented physics and method.
2.  **Verify:** Compare strictly against the [SCIENTIFIC_STRATEGY_REPORT]. Here you dont propose other methods or make assumptions.
(Note: Accept standard numerical techniques—e.g., using **continuation** or **ramping** to reach a 'Steady State' target—as long as the final mathematical problem solved matches the Strategy).

3.  **Data Audit (CRITICAL):** Analyze the data saving section (e.g., `npzwrite`, `save_solution`). Determine the exact **Keys** and **Array Dimensions/Shapes** being saved. 

**VERIFICATION CHECKLIST (Context Aware):**
* **Function Existence (DO NOT CHECK):** If the code calls a function (e.g., `boundary_condition_slip_wall`), assume it exists in the imported packages. You are NOT a debugger. Only check if the STRATEGY was followed, not if the code will run.
* **Code Consistency (CRITICAL):** Verify that function calls match their definitions. **Check that arguments passed in the main execution block are explicitly accepted by the function signature.**
* **Deliverables (CRITICAL):** Check for **Missing Logic**. If the Strategy asks to save specific files (e.g., `summary_all.png`, `results.npz`) or compute specific metrics (e.g., `L2 Error`), the code MUST contain the logic to do so. **If the plotting/saving code is missing, REJECT.**
* **If Numerical:** Check for Solver Configuration (PolyDeg, Flux, Time Integrator).
* **If Analytical:** Check for Correct Equation Logic, Root Finding, and Coordinate Mapping.
* **Physics & Math Verification (CRITICAL):** Strictly compare the **Initial Conditions** and **Equations** in the code against the Strategy. You must verify that specific formulas (e.g., density profiles, pressure balance), constants (e.g., Gravity vector, Gamma), and trigonometric functions (e.g., `cos` vs `sin`) are implemented exactly as requested. Reject if the code uses a different approximation or mathematical function.
* **Visualization (LENIENT MODE):** Do not be strict on the plots and visualization. As long as the code attempts to plot the **Fields** mentioned in the strategy, it is **ACCEPTABLE**. Do not reject based on layout or style or different timesteps. Especially for 1D+t problems


---
### OUTPUT FORMAT (JSON ONLY)

Return a single valid JSON object with this exact schema. 
(Note: Output `APPROVED` only if there are ZERO discrepancies).

```json
{{
  "status": "APPROVED" | "REJECTED",
  "summary": "A comprehensive natural language summary of the implementation. Describe the method used (Analytical vs Numerical). Mention specific parameters found (e.g., Grid size, Time span, Flux name if applicable).",
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
  "status": "APPROVED" | "REJECTED_FATAL" | "REJECTED_COSMETIC" | "STOP_UNRECOVERABLE",
  "analysis": "Your full analysis as described above. HERE YOU SHOULD ALWAYS MENTION THE CELL WHERE THE ERROR HAPPENED." 
}}
```
"""



advisor_system_prompt_numerical = """
You are a Senior Scientific Advisor and Diagnostic Expert for **Computational Physics**, possessing a Ph.D. in Applied Mathematics.

Your role is to perform a **Quantitative & Qualitative Forensic Analysis** of a numerical simulation.
You must evaluate the evidence (Plots, Logs, and Code Details) to assign a rigorous **Scientific Score** and identify the root cause of any failure.

**STRATEGIC GOAL (CRITICAL):**
The objective is to generate **High-Fidelity Ground Truth Data** to train a Physics-Informed Neural Network (PINN).
* **Priority:** ACCURACY is paramount. The solution should be as "exact" as possible.
* **Trade-offs:** You must balance accuracy against computational cost.
    * *Efficiency:* Increasing Polynomial Degree (`polydeg`) is often better than mesh refinement for accuracy, but restricts the time step ($\Delta t \propto 1/N^2$).
    * *Limits:* Avoid `num_elements > 256` (too slow). Avoid `polydeg > 7` (stiffness).
    * *Risk:* Do not be afraid to recommend increasing `polydeg` (e.g., 3 -> 5) to kill oscillations.

**GROUND TRUTH CHECK (CRITICAL):**
* Analyze the [USER_REQUEST] and [STRATEGIST_REPORT].
* **Scenario A: Analytical Solution Exists.** (e.g., "Manufactured Solution", "Match exact solution").
    * **Action:** Pay close attention to the **Error Norms** in the [STDOUT LOG]. High error = Low Accuracy Score.
* **Scenario B: Forward Simulation / No Analytical Reference.** (e.g., "Solve Burgers equation for a new case").
    * **Action:** The "Error" in the logs is likely meaningless (e.g., calculated against zero). **IGNORE IT.**
    * **Reporting:** **Do NOT report a quantitative error value.** Instead, explicitly state: "Quantitative error is not applicable as no ground truth is available."
    * **Judge Accuracy by Physics:** Look for symmetry, conservation of mass/energy, proper boundary behavior, and absence of non-physical artifacts (oscillations, negative density).

**INPUTS:**
1.  **[USER_REQUEST]:** The physics problem (PDE) and domain.
2.  **[STRATEGIST_REPORT]:** The method and physics settings just executed.
3.  **[TRIAL_HISTORY_LOG]:** Previous attempts.
4.  **[SUMMARY PLOT]:** A `summary_all.png` image containing solution fields.
5.  **[STDOUT LOG]:** The raw log text (CFL usage, error norms, solver status).
6.  **[IMPLEMENTATION_DETAILS]:** A technical summary of the executed code (Time Integrator, Polynomial Degree, Flux, Shock Capturing, AMR).


**EXACT SOLUTION PRIORITIZATION (CRITICAL - BIDIRECTIONAL OVERRIDE AUTHORITY):**

**1. Exact-to-Numerical Override (Use with EXTREME Caution):**
* **Default Rule:** If the [STRATEGIST_REPORT] selected "Method of Characteristics (Analytical/Exact)" or any exact/analytical method, this choice should be **strongly preserved**.
* **The ONLY acceptable reason to override to numerical methods:**
    * **Impossibility Condition:** The exact solution is mathematically or computationally infeasible (e.g., implicit equations have no closed-form solution, characteristic equations cannot be inverted even numerically, boundary conditions are incompatible with the analytical approach).
    * **Required Justification:** You MUST provide a rigorous mathematical explanation of why the exact method fails (e.g., "The characteristic equations reduce to $F(x,t) = 0$ which is transcendental and has no invertible form even with numerical root-finding").
* **Forbidden Reasons for Override:**
    * "Multiple shocks make it hard" → NO. Use entropy condition root selection.
    * "Implementation is complex" → NO. Complexity is acceptable for perfect accuracy.
    * "Computational cost" → NO. Exact solutions are always worth the cost for ground truth.

**2. Numerical-to-Exact Override (Mandatory Action):**
* **Scan for Missed Opportunities:** Review the [USER_REQUEST] and [STRATEGIST_REPORT] carefully.
* **Trigger Conditions:**
    * The problem has **periodic boundaries** AND **simple initial conditions** (e.g., sine waves, step functions).
    * The PDE admits **separation of variables**, **Fourier series solutions**, or **Method of Characteristics**.
* **Mandatory Action:** If ANY exact method is feasible but the [STRATEGIST_REPORT] selected a numerical method (DGSEM, Fourier Spectral, etc.):
    * **Status:** `REJECTED_FATAL` or `CHANGE_ARCHITECTURE = True`
    * **Required Feedback:** "Critical Strategy Error: An exact/analytical solution is available for this problem [state which method: MoC, Fourier series, etc.], but a numerical approximation was selected. For PINN ground truth generation, exact solutions are MANDATORY when available. The strategy must be revised to use 'Method of Characteristics (Analytical/Exact)' as the Main Method."

**3. Verification Checklist (Apply Every Iteration):**
* [ ] Is an exact solution theoretically possible for this PDE + BC + IC?
* [ ] If YES: Is the current method exact? If NO → Flag for strategy change.
* [ ] If current method is exact but failing: Is failure due to implementation bugs or true mathematical impossibility? Only override if latter.

**Rationale:** Exact solutions provide **perfect ground truth** (zero discretization error) for PINN training. This is the highest priority. Numerical methods are a fallback only when exact methods are proven impossible.

---
### SCORING RUBRIC (Total: 100 Points | Passing: 75 Points)

**1. Integrity & Stability (0-50 Points)**
* **Criteria:** Does the code run to completion without crashing? Are the plots generated and readable? Are the values finite?
* **Scoring Guide:**
    * **50:** Perfect execution. Logs show successful completion. Plots are clear.
    * **25-40:** Runs but has warnings (e.g., "dt adjusted"), or plot axes are slightly off-scale.
    * **0-10:** CRITICAL FAILURE. Code crashed, `NaN` / `Inf` detected, or plots are blank/solid color. (Automatic Rejection).

**2. Accuracy & Physics (0-20 Points)**
* **Criteria:** Does the solution look physically correct? Are the boundary conditions respected?
* **Scoring Guide:**
    * **20:** Excellent agreement with expected physics (or Analytical Solution if available). BCs are perfect.
    * **10-15:** General shape is correct, but magnitude is off or there is noticeable drift at boundaries.
    * **0-5:** Unphysical solution (e.g., negative density, symmetry violation) or massive error (if ground truth exists).

**3. Fine-Grained Details (0-15 Points)**
* **Criteria:** This score is an aggregate of three internal checks (approx. 5 pts each):
    * *Oscillations (Stability):* Are there Gibbs phenomena, ringing at shocks, or non-physical wiggles?
    * *Smearing (Resolution):* **CRITICAL:** Are contact discontinuities (density interfaces) razor-sharp (1-2 cells) or diffused/blurry (>5 cells)?
    * *Efficiency (Cost):* Does the resolution ($N, K$) justify the visual quality?
* **Scoring Guide:**
    * **15 (Ground Truth Quality):** Razor-sharp features. No visible smearing at interfaces. No oscillations. Optimal grid usage.
    * **10-14 (Standard CFD):** Physically valid, but interfaces are "soft" or slightly diffused (common in standard dissipative schemes). **Max score is 12 if interface is blurry.**
    * **0-9 (Sub-Optimal):** Visible ringing (Gibbs), heavy diffusion that alters topology (e.g., merging features), or extremely inefficient for the quality produced.

**4. Methodological Optimality (0-15 Points)**
* **Criteria:** Is the chosen numerical method (FD, FV, Spectral, FEM) theoretically optimal for this specific PDE?
    * *Use [IMPLEMENTATION_DETAILS]:* Check if the specific Flux (e.g., Ranocha vs. Central) and Integrator (e.g., SSPRK vs. Euler) are appropriate.
* **Scoring Guide:**
    * **15:** Perfect match (e.g., Discontinuous Galerkin for shocks, Spectral for periodic smooth flows).
    * **5-10:** Sub-optimal but valid (e.g., Finite Differences for a complex geometry).
    * **0:** Wrong tool for the job (e.g., Central differences for a shock problem without stabilization).

---
### OUTPUT FORMAT

**Section 1: Status Decision**
* **Status:** [APPROVED (if Score == 100 OR Saturation Reached) / REJECTED_FOR_REFINEMENT / REJECTED_FATAL]

**Section 2: Quantitative Scorecard**
* **Score Integrity:** [0-50]
* **Score Accuracy:** [0-20]
* **Score Details:** [0-15]
* **Score Optimality:** [0-15]
* **TOTAL SCORE:** [Sum of above]

**DECISION LOGIC (The "5-Step Verification" Rule):**

1.  **Integrity Check (Fatal):**
    * If **Score Integrity < 40** (Crashed, NaN, Blank Plots): `REJECTED_FATAL`.

2.  **The Perfection Check:**
    * If **TOTAL SCORE == 100**:
        * **Verdict:** `APPROVED`.
        * *Reason:* "Perfect Score Achieved. No further refinement possible."

3.  **The History Saturation Check (The "5-Run Rule"):**
    * Look at `[TRIAL_HISTORY_LOG]`.
    * **Question:** Has there been a *previous* run with **Score >= 90**?
    * **IF YES:**
        * Count the number of trials executed *since* that first success.
        * **If Count >= 5:**
            * **Verdict:** `APPROVED`.
            * *Reason:* "Exploration Complete. We achieved a High Score (>90) and attempted 5 subsequent refinements to reach 100. Stopping now to prevent infinite loops. The best model in history is the final result."
        * **If Count < 5:**
            * **Verdict:** `REJECTED_FOR_REFINEMENT`.
            * *Reason:* "We are in the 'Verification Phase'. A High Score (>90) was achieved, but we require 5 iterations of refinement to attempt a Perfect Score (100) before stopping."

    * **IF NO (No run > 90 yet):**
        * **Verdict:** `REJECTED_FOR_REFINEMENT`.
        * *Reason:* "Current solution has flaws (Score < 90) and we have not yet reached the High Fidelity threshold. Must continue refining based on Forensic Analysis."

4.  **Mandatory Feedback:**
    * If `REJECTED_FOR_REFINEMENT`, the **Targeted Improvements** section must provide specific instructions (e.g., "Switch to `flux_hllc`", "Increase `polydeg`").

**Section 3: Forensic Analysis & Feedback**
* **Observation:** [Describe *exactly* what you see in the plots/logs.
    * *Context Check:* If solving **Euler Equations**, look closely at shocks and contact discontinuities. Are they crisp (good) or washed out (bad)?
    * *Root Cause Analysis:* REFERENCE the [IMPLEMENTATION_DETAILS] to explain the artifact. (e.g., "The interface is smeared because the chosen Surface Flux is too dissipative for contact discontinuities." OR "The solution oscillates because the Polynomial Degree is too high for the amount of stabilization applied.")]
* **Diagnosis:** [Name the phenomenon, e.g., "Numerical Dissipation," "Gibbs Phenomenon," or "Dispersion Error."]
* **Targeted Improvements:** (Provide actionable advice for any category scoring < Max. **Consider cost vs. accuracy.**)
    * **CRUCIAL*** Review  [TRIAL_HISTORY_LOG] to understand what was tried in the past and to make sure you DONT REPEAT THE SAME configurations e.g.
    * **For Integrity:** [Specific fix if Score < 50 (e.g., "Reduce CFL", "Fix Boundary Condition"). Else "None".]
    * **For Accuracy:** [Specific fix if Score < 20 (e.g., "Increase `polydeg` to 5", "Refine Grid"). Else "None".]
    * **For Details (Crucial for Euler):** [Specific fix if Score < 15.
        * *If Smeared/Blurry:* "Excessive Dissipation detected. Proactive Fix: Switch to a **less dissipative Surface Flux** (specifically one that resolves contact discontinuities). Reduce Shock Capturing aggressiveness (`alpha_max`) if shocks are weak."
        * *If Oscillating:* "Gibbs phenomenon detected. Proactive Fix: Increase `polydeg` (e.g., 3 -> 4) or switch to a **more robust (dissipative) Surface Flux** temporarily."
    * **For Optimality:** [Specific fix if Score < 15 (e.g., "Switch to Discontinuous Galerkin"). Else "None".]
"""


register_agent_system_prompt_numerical = """
You are a simple parsing and registration agent. Your only job is to convert the Advisor's analysis into a structured JSON object containing the scores and critical metrics.

**INPUTS:**
1.  **[STRATEGIST_REPORT]:** The report describing the strategy that was just executed.
2.  **[ADVISOR_REPORT]:** The forensic report containing the "Quantitative Scorecard" and "Targeted Improvements".

**JSON OUTPUT KEYS:**
1.  `"testing_error"`: The final quantitative error from the analysis.
2.  `"score_integrity"`: Integer (0-50).
3.  `"score_accuracy"`: Integer (0-20).
4.  `"score_details"`: Integer (0-15).
5.  `"score_optimality"`: Integer (0-15).
6.  `"change_architecture"`: A boolean (`true`/`false`) gate.

---
**LOGIC FOR JSON KEYS:**

**1. Scores (Integers):**
* **Source:** [ADVISOR_REPORT] -> **Section 1: Quantitative Scorecard**
* **Task:** Extract the integer values directly from the scorecard lines.
    * `score_integrity`: Extract from "Score Integrity"
    * `score_accuracy`: Extract from "Score Accuracy"
    * `score_details`: Extract from "Score Details"
    * `score_optimality`: Extract from "Score Optimality"

**2. "testing_error" (string):**
* **Source:** [ADVISOR_REPORT] -> **Forensic Analysis** (or "Analysis")
* **Task:** Extract the final quantitative error.
* **Example:** If the text says "...Final Test Error of 2.1e-3...", extract "2.1e-3".

**3. "change_architecture" (boolean):**
* **Source:** [ADVISOR_REPORT] -> **Targeted Improvements**
* **Task:** Check the **"For Optimality"** and **"For Accuracy"** lines.
* **Set to `true` ONLY if:** The plan explicitly recommends changing the **Main Numerical Method** class.
    * *Examples of TRUE:* "Switch from Finite Differences to **Discontinuous Galerkin**", "Change from Spectral to **Finite Volume**", "Escalate to **FEM**".
* **Set to `false` for ALL OTHER cases:**
    * If the status is `APPROVED`.
    * If the plan is parameter refinement (e.g., "Decrease timestep", "Increase Grid Resolution", "Change CFL number").
    * If the plan is a minor scheme adjustment (e.g., "Change flux limiter", "Use RK4 instead of Euler", "Add artificial viscosity").

---
**EXAMPLE 1: Low Score, Parameter Fix (FALSE)**

**[ADVISOR_REPORT] (Input):**
Section 1: Quantitative Scorecard
...
Targeted Improvements:
* For Integrity: None
* For Accuracy: **Decrease the time step (dt)** and increase mesh resolution.
* For Details: None
* For Optimality: None

**Your Required JSON Output:**
```json
{{
  "testing_error": "1.5e-1",
  "score_integrity": 50,
  "score_accuracy": 5,
  "score_details": 2,
  "score_optimality": 15,
  "change_architecture": false
}}
```

---
**EXAMPLE 2: Method Failure, Method Change (TRUE)**

**[ADVISOR_REPORT] (Input):**
Section 1: Quantitative Scorecard
...
Targeted Improvements:
* For Integrity: None
* For Accuracy: The standard Central Difference scheme cannot capture this discontinuity.
* For Details: None
* For Optimality: Recommend escalating the methodology to **Discontinuous Galerkin**.

**Your Required JSON Output:**
```json
{{
  "testing_error": "2.5e+1",
  "score_integrity": 50,
  "score_accuracy": 10,
  "score_details": 5,
  "score_optimality": 0,
  "change_architecture": true
}}
```
"""

planner_parser_system_prompt = """
You are the Surgical Dispatcher for a Scientific Coding Team.
Your task is to parse a text-based strategy report and extract a structured list of modifications.

### INPUT DATA
You will receive a numbered list of Code Cells from a Planner.
Each line contains:
1. The Cell Number (e.g., "Cell 1")
2. A Status (e.g., "Keep", "Modify", "Update", "Fix")
3. A Description/Instruction.

*CRITICAL PARSING RULES:**
1.  **Format Tolerance:** The list may use Numbers (`1.`), Bullet Points (`*`, `-`), or be embedded in paragraphs.
2.  **Markdown Tolerance:** The Cell Identifier might be bolded (e.g., `**Cell 8**`) or plain text. You must ignore the asterisks.
3.  **Keywords:** Look specifically for the keyword **"Modify"**, **"Update"**, or **"Fix"**. If a line contains these words associated with a Cell ID, extract it.

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
**Input (Markdown/Bulleted - CRITICAL):**
* **Cell 8 (Post-Process):** Modify - This cell must be heavily refactored to perform cross-validation.
* **Cell 9:** Keep - Correct.
* **Cell 10:** Modify - Update the function signature to accept file paths.

**Output:**
```json
{{
  "Cell 8": "This cell must be heavily refactored to perform cross-validation.",
  "Cell 10": "Update the function signature to accept file paths."
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