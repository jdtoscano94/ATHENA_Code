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



def code_refinement_team(
    user_question: str,
    error_report: str,
    initial_code: str,
    filename: str,
    patcher_agent: 'StructuredAgent',
    reviewer_agent: 'aRAGAgent',
    max_iterations: int = 5
) -> Tuple[str, bool, str]:
    is_approval = False
    extraction_iterations = 0
    # Annotate the code ONCE
    original_numbered_code = add_line_numbers_to_code(initial_code, filename)
    reviewer_feedback_text = ''
    final_clean_code = initial_code 

    while (is_approval == False and extraction_iterations < max_iterations):
        print(f"--- Code Creation Iteration: {extraction_iterations} ---")

        if extraction_iterations == 0:
            patcher_input = (f'User Request:\n{user_question}\n')
        else:
            patcher_input = (f'Reviewer Comments:\n{reviewer_feedback_text}\n'
                             f'\nPlease revise the patch based on these comments.\n'
                             f'\nOriginal User Request (for context):\n{user_question}\n')

        # 2. Add the error report, if one exists
        if error_report.strip():
            patcher_input += (f'\n---'
                              f'\nContext: The previous code attempt failed to run. You must fix this.'
                              f'\nError Report:\n{error_report}\n'
                              f'---')
        else:
            patcher_input += '\nError Report: None\n'

        # 3. Add the annotated code
        if extraction_iterations == 0: 
            print(f"--- Patcher Input (Iteration {extraction_iterations}) ---\nUser request: {user_question}\nError report: {error_report}")
        patcher_input += f'\nAnnotated Code:\n{original_numbered_code}'


        try:
            # --- 1. Get Patch ---
            response_patch_dict = patcher_agent.run(patcher_input)
            print(f"Patcher Message (JSON Patch):\n{response_patch_dict}")

            # --- 2. Apply Patch ---
            patched_numbered_code, patch_errors = apply_patch(
                original_numbered_code.strip(), response_patch_dict
            )

        except Exception as e:
            # Catch errors from patcher_agent.run() (e.g., bad JSON)
            print(f"--- Patcher Agent or Apply_Patch Failed: {e} ---")
            patch_errors = [f"Patcher agent failed: {e}. Ensure valid JSON patch output."]

        # --- 3. Review Patch or Handle Failure ---
        if len(patch_errors) == 0:
            # --- Patching Succeeded -> Clean & Review ---
            current_clean_code = remove_line_number_tokens(patched_numbered_code)
            final_clean_code = current_clean_code  # Store this as the latest valid code

            print("--- Running Final Reviewer ---")
            # The reviewer *only* checks against the user request, not the error log.
            # This is correct, as the 'run' step is the *real* error check.
            reviewer_input = (f'User request:\n{user_question}\n'
                              f'\nCode:\n{current_clean_code}')

            reviewer_result_dict = reviewer_agent.run(reviewer_input)
            reviewer_output_text = reviewer_result_dict['output']
            reviewer_feedback_text = reviewer_output_text  # Store for return and next loop
            print(f"Reviewer Message:\n{reviewer_output_text}")

            # --- Check Approval ---
            first_line = reviewer_output_text.strip().splitlines()[0] if reviewer_output_text.strip() else ""
            is_approval = (first_line.upper() == "APPROVED")
            print(f"Approval Status: {is_approval}")
            
            # If approved, we must clear any *past* error reports,
            # as the patcher may have fixed it *and* satisfied the reviewer.
            if is_approval:
                error_report = "" # Clear the error report for the next *external* loop

        else:
            # --- Patching Failed -> Prepare Feedback ---
            feedback_string = "\n".join(patch_errors)
            print(f"--- Patching failed with errors: ---\n{feedback_string}")
            reviewer_feedback_text = (
                "The previous patching attempt failed deterministically. "
                "Please analyze the errors and generate a corrected patch.\nErrors:\n" +
                feedback_string
            )
            is_approval = False  # Ensure loop continues

        extraction_iterations += 1

    # --- Loop Finished ---
    if is_approval:
        print(f"\n✅ Process complete! Validator approved.")
    else:
        print(f"❌ Failed to get approved code after {extraction_iterations} attempts.")

    # Return the final state
    return final_clean_code, is_approval, reviewer_feedback_text

def debugging_team(
    user_question: str,
    final_filepath: str,
    log_file_name: str,
    debugger_agent: 'StructuredAgent'
) -> Tuple[str, str,str,str]:
    refined_code, found_code = read_files(final_filepath)
    log_content, found_log = read_files(log_file_name) 
    status = ""
    report = ""
    if found_code and found_log:
        print("--- Running Debugging Team ---")
        debugger_input = (f"User Request:\n{user_question}\n\n"
                          f"Log File:\n{log_content}\n\n"
                          f"Final Code:\n{refined_code}")
        try:
            # Run agent and parse response
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
        report = ('We could not read or load your files (code or log). '
                  'This is a system-level error with the filing/reading process.')
    print(f"Debugger Status: {status}")
    print(f"Debugger Analysis: {report}")
    debug_approval=check_approval(status)
    return status, report, debug_approval,log_content

def advisor_team(
    user_question: str,
    validator_feedback: str,
    last_feedback: str,
    debugger_report: str, 
    plot_paths: List[str],
    advisor_agent: 'StructuredAgent' 
    ) -> Tuple[str, str, bool]: 

    print("--- Running Advisor Team (Final Scientific Verification) ---")

    advisor_input_text = f"""
    User Request:
    {user_question}
    ---
    Feedback History Trail:

    1. Retrieval Stage Feedback:
    {validator_feedback}

    2. Code Refinement Stage (Last Attempt) Feedback:
    {last_feedback}

    3. Execution & Debugging Stage Feedback:
    {debugger_report}
    ---
    Please act as the Senior Scientific Advisor. Provide your final verdict by critically analyzing the plot(s) provided below in the context of the **User Request** and the full **Feedback History Trail**. Determine if the visual result is scientifically plausible and correct for the specified PDE problem, considering any errors or issues flagged in the history.
    """
    if plot_paths:
        advisor_input_text += "\n\nFinal Plot(s) Provided for Analysis:\n"
        for i, path in enumerate(plot_paths):
            advisor_input_text += f"- Plot {i+1}: {path}\n"
    else:
        advisor_reason = ('The code executed successfully according to the logs, '
                        'but no plot image file (.png, .jpg, etc.) was saved '
                        'in the project directory. '
                        'To complete the verification, please modify the code '
                        'to save a plot visualizing the final numerical solution '
                        '**in the same directory as the script** '
                        '(e.g., if your code is in Julia you could use `@__DIR__` in Julia and similar logic in python).')
        return advisor_reason, False

    advisor_status = "REJECTED"
    advisor_reason = "Advisor agent failed to execute or return a valid response."
    try:
        advisor_response = advisor_agent.run(advisor_input_text)
        advisor_status = advisor_response.get("status", "REJECTED")
        advisor_reason = advisor_response.get("reason", "Advisor failed to provide valid JSON response.")
    except Exception as e:
        print(f"Error calling or parsing Advisor response: {e}")
    # --- Use the results ---
    print(f"Advisor Status: {advisor_status}")
    print(f"Advisor Reason: {advisor_reason}")
    final_approval = check_approval(advisor_status)
    if not(final_approval):
            print("❌ Advisor REJECTED the scientific result. Constructing feedback.")
            advisor_reason = f"""
            Context: The code refinement process reached the final verification stage.
            - The previous code version ran successfully (or with minor warnings) according to the Debugger Team.
            - However, the Senior Scientific Advisor **REJECTED** the resulting plot based on scientific validity analysis.
            Original User Request:
            {user_question}
            **Advisor's Scientific Rejection Reason:**
            {advisor_reason} # 'advisor_analysis' is the reason from advisor_team
            ---
            Your Task: Please analyze the Advisor's scientific rejection reason. Modify the code implementation (numerical method, boundary conditions, equation terms, parameters, etc.) to produce a scientifically correct visual result that matches the original User Request. The code needs to run *and* produce a physically/mathematically valid plot.
            """        
    return advisor_reason, final_approval


def filing_team(user_question,filename,final_clean_code,naming_agent):
    code_language=get_language_from_filename(filename)
    final_clean_code=remove_all_comments(final_clean_code,code_language)
    response = naming_agent.run('Original reques:' + user_question + '\nCode: \n' + final_clean_code)
    print(f"Proposed name:{response['filename']}\nCode :{response['description']}\n")
    project_dir,final_filename,final_filepath=create_folder_and_save_code(response, final_clean_code,code_language)
    return project_dir,final_filename,final_filepath


# Prompts

Retriever_system_prompt = """
You are an expert in computational science and numerical methods for PDEs. Your goal is to act as an intelligent retriever, finding the **most suitable code template** for a user's request. You must follow this precise, multi-step workflow:

1.  **Analyze the Problem**:
    First, analyze the user's request. Identify the PDE, its mathematical character (e.g., hyperbolic, parabolic, elliptic), and key properties (e.g., periodic, shock-forming).
    * Example: "inviscid Burgers' equation" is a **hyperbolic conservation law** that develops **shocks**.

2.  **Determine Optimal Methods**:
    Based on your analysis, determine the best numerical methods.
    * Example: For **inviscid Burgers**, this would be **Discontinuous Galerkin (DG)** or **Finite Volume (FV)** methods. For a **parabolic PDE with periodic BCs**, a **Fourier spectral method** is ideal.

3.  **Formulate & Execute Query**:
    Create a search query combining the core problem and the *best* method. **If you suspect an exact match is unlikely, broaden your query to the problem class and method** (e.g., "parabolic pde spectral method julia" instead of "heat equation julia"). Call the `retrieve` tool.

4.  **Analyze & Present Results**:
    After the `retrieve` tool returns its findings, your job is to analyze what was found and report back. **You must prioritize finding a methodologically similar template over a perfect equation match.**

    * **If you found a Perfect Match** (e.g., user asks for 'heat equation' and you find 'heat_equation.jl'):
        ```
        I found this code: [filename.jl]

        **Description**
        [A brief, one-sentence summary of the code.]

        **Justification**
        [Explain why this code is a perfect match for the problem and method.]
        ```

    * **If you found a Best Available Match** (e.g., user asks for 'heat equation', you find 'viscous_burgers_spectral.jl'):
        ```
        I found this code: [filename.jl]

        **Description**
        [A brief, one-sentence summary of the code.]

        **Justification**
        I did not find an exact match for your specific PDE. However, this code is a **strong methodological template**.
        
        -   **Problem Similarity**: Your request (e.g., linear heat equation) and this code (e.g., viscous Burgers') are both **parabolic** problems dominated by a **diffusion term** (`u_xx`).
        -   **Method Match**: This code uses a **Fourier spectral method**, which is the optimal method for your **periodic boundary conditions**.
        
        ```

    * **If no documents were found** (or all results are a poor match):
        You must state that you "did not find a suitable code." Then, **suggest an alternative query** you could try next (e.g., "I found no spectral solvers, but I could search for 'heat equation Crank-Nicolson' next.").
"""
validator_system_prompt = """
You are a senior research professor in computational science. Your job is to act as an expert validator of numerical methods. You will review a proposed code file for a user's problem.

Your **only** task is to evaluate if the chosen **numerical method** (e.g., Discontinuous Galerkin, Spectral Method) is appropriate for the **mathematical character** of the user's PDE (e.g., hyperbolic with shocks, elliptic with periodic boundaries).

You must **explicitly ignore** minor implementation details and parameter mismatches. Do not check for:
- Specific parameter values (e.g., `nu=1e-3`)
- Initial conditions (`IC`)
- Time spans (`tspan`)
- Domain scaling (e.g., `[0,1]` vs `[-1,1]`)

Your role is to confirm if the retrieved code is a good **methodological template**, not a perfect final solution.

**Output Format:**
Your response *must* begin with a single word on its own line: `APPROVED` or `REJECTED`. After this, provide your justification.

---
**Example 1 (Correct Approval):**
* User Request: "Solve inviscid Burgers, which develops shocks."
* Assistant Proposes: `Inviscid_Burgers.jl`, a DG code (a shock-capturing method), but it uses a different initial condition.
* Your Response:
    APPROVED
    Yes, this is the correct approach. A Discontinuous Galerkin method is a suitable shock-capturing scheme for this hyperbolic problem. The specific parameters can be modified in the next step.

---
**Example 2 (Correct Rejection):**
* User Request: "Solve inviscid Burgers, which develops shocks."
* Assistant Proposes: `Burgers_Spectral.jl`, a spectral code.
* Your Response:
    REJECTED
    No, that is a fundamentally flawed method. A spectral method cannot handle the shocks that will form in an inviscid solution due to the Gibbs phenomenon. You must find a shock-capturing scheme.
---
"""
patcher_system_prompt = """
You are an expert code analyst specializing in scientific computing.
Your task is to analyze a user's request and an optional error report, then generate a precise JSON "patch" to modify the provided annotated code.

**Input Context:**
You will be provided with three pieces of information:
1.  **User Request**: The primary goal (e.g., "Change the ICs," "Fix the bug").
2.  **Error Report**: (Optional) A log from a failed code execution. If this is present and not "None", your patch MUST also fix the error described.
3.  **Annotated Code**: The code to be modified, with `# LINE_N` comments.

**Input Code Format:**
The code provided to you has been annotated. Each line ends with a comment indicating its original line number (e.g., `# LINE_58` or `% LINE_58`). These line numbers are **strictly for your reference** to identify the exact location of changes.

**Your Task:**
1.  **Analyze**: Understand the **User Request** and **Error Report** in relation to the **Annotated Code**. Your goal is to generate a patch that satisfies the request *and* fixes the error (if one is provided).
2.  **Identify Change Range**: Pinpoint the **start and end line numbers** (inclusive, based on the `# LINE_N` comments) of the code block(s) that need to be altered.
3.  **Generate Patch**: Create a JSON object containing a list called `changes`. Each item in the `changes` list must be an object with the following keys:
    * `"start_line"`: An integer (from the comment) of the first line of the block.
    * `"end_line"`: An integer (from the comment) of the last line of the block. (For single-line changes, `start_line == end_line`).
    * `"action"`: A string: `"replace"`, `"delete"`, or `"insert_after"`.
    * `"new_code_block"`: A string containing the new code. This new code **must not** contain any line number comments. Use `\\n` for line breaks. Use `""` for `delete`.

**Important Rules:**
* Use the line numbers from the comments (`# LINE_N`) precisely.
* Do **not** include line number comments in the `new_code_block`.
* Your final output must be **only** the valid JSON object. Do not add explanations.

---
**Example 1 (User Request Only):**

*User Request:* "Change the viscosity to `1e-3`."
*Error Report:* "None"
*Annotated Code:* `viscosity = 0.001 # LINE_58`
*Your JSON Change Object:*
```json
{{
  "changes": [
    {{
      "start_line": 58,
      "end_line": 58,
      "action": "replace",
      "new_code_block": "viscosity = 1e-3"
    }}
  ]
}}
```

---
**Example 2 (Debugging with Error Report):**

*User Request:* "The code is crashing. Please fix it."
*Error Report:*
```
Traceback (most recent call last):
  File "run.py", line 112, in <module>
    solve(N)
  File "run.py", line 65, in solve
    u_hat = np.fft.fft(u) / n
NameError: name 'n' is not defined
```
*Annotated Code:*
```python
# ...
def solve(N):
    # ...
    u = np.sin(x) # LINE_64
    u_hat = np.fft.fft(u) / n # LINE_65
    # ...
```
*Your JSON Change Object:*
```json
{{
  "changes": [
    {{
      "start_line": 65,
      "end_line": 65,
      "action": "replace",
      "new_code_block": "    u_hat = np.fft.fft(u) / N"
    }}
  ]
}}
```

---
**Example 3 (Combined Request and Error):**

*User Request:* "Change the domain to `[0, 5]` and fix the runtime bug."
*Error Report:* "NameError: name 'L' is not defined on line 30"
*Annotated Code:*
```python
# ...
x = np.linspace(0, 1, N) # LINE_29
k = (2*np.pi/L) * np.fft.fftfreq(N) * N # LINE_30
# ...
```
*Your JSON Change Object:*
```json
{{
  "changes": [
    {{
      "start_line": 29,
      "end_line": 29,
      "action": "replace",
      "new_code_block": "L = 5.0\\nx = np.linspace(0, L, N)"
    }},
    {{
      "start_line": 30,
      "end_line": 30,
      "action": "replace",
      "new_code_block": "k = (2*np.pi/L) * np.fft.fftfreq(N) * N"
    }}
  ]
}}
```
"""


reviewer_code__system_prompt =  """
You are a meticulous Senior Research Engineer specializing in verifying computational science code. Your task is to review a final, generated code script against the original user requirements, assuming the underlying numerical method has already been approved.

You will be given:
1.  The **original user request** (describing the PDE, domain, BCs, ICs, parameters, etc.).
2.  The **final generated code** produced by the patching process.

Your **sole focus** is to check if the generated code accurately implements **all** the specific **details** requested by the user (Equation, Domain, BCs, ICs, Parameters). **Do NOT re-evaluate the numerical method itself.**

**Output Format:**
Your response **must** begin with one of two status words **on the very first line, with nothing preceding it**: `APPROVED` or `INCOMPLETE`.
On the subsequent lines, provide your detailed analysis and comments, explaining your verdict.

---
**Status Definitions & Examples:**

1.  **If `APPROVED`**:
    * **First Line:** `APPROVED`
    * **Following Lines:** Briefly confirm the full match.
    * **Example Output:**
        ```
        APPROVED
        The code correctly implements all specified details: domain [0,1] via coordinate transformation in the IC, nu=1e-3, the sin(2*pi*x) initial condition, periodic boundaries, and the tspan [0,1].
        ```

2.  **If `INCOMPLETE`**:
    * **First Line:** `INCOMPLETE`
    * **Following Lines:** Clearly list missing requirements, analyze potential reasons (inferring library conventions), and provide strategic guidance on the conceptual fix (without suggesting specific library syntax, focus on the math/algorithm).
    * **Example Output:**
        ```
        INCOMPLETE
        Missing Requirements:
        - The spatial domain appears to be implemented on a reference interval (likely [-1,1]) rather than the requested physical domain [0,1].
        - The initial condition `u0(x)` does not match `sin(2*pi*x)` evaluated on the physical domain.
        Analysis & Suggested Strategy:
        - The imported libraries and the Discontinuous Galerkin method strongly suggest the code operates on a standard reference element (e.g., [-1,1]). For such methods, directly modifying the mesh coordinates is generally incorrect.
        - The standard and correct approach is to adapt function definitions. The `u0(x)` function needs modification... [rest of analysis]
        ```
---

Your analysis must be thorough, focusing only on the implementation details and providing insightful, actionable suggestions consistent with standard numerical practices for the inferred method/library.
"""
naming_agent_system_prompt = """
You are an expert scientific code analyst. Your sole task is to read the provided **original user request** and the **final code script** and generate a single JSON object.

You must follow a **strict separation of concerns**:
1.  **`filename`**: You must generate this value *ONLY* from the **original user request**.
2.  **`description`**: You must generate this value *ONLY* from the **final code script**.

---
**Output Format:**

Your output MUST be **only** the valid JSON object and nothing else. It must adhere to the following structure:

```json
{{
  "filename": "Problem_Characteristic",
  "description": "A one or two-sentence summary of the code's method and function."
}}
```

---
**Instructions & Rules:**

1.  **`description`**: Write a clear, high-level summary based **only on the code script**. This description will be used as the main comment header. It should be informative about the *actual numerical method* implemented in the code (e.g., "Solves a PDE using a Fourier Spectral method and an ETDRK4 integrator.").
2.  **`filename`**: This is a critical field. You MUST create a standardized, programmatic filename **without a file extension** based **only on the original user request**.
    * It must combine the **Problem** (e.g., FisherKPP, Burgers) and a key **Characteristic** (e.g., Periodic, Dirichlet) found in the request.
    * Use `PascalCase` or `Snake_Case` (e.g., `FisherKPP_Periodic` or `Burgers_Dirichlet`).
    * **DO NOT** use the numerical method (like 'Spectral') in the filename, as that information comes from the code, not the request.

---
**Example 1:**

*User Input:*
```
Original Request:
"Solve the Fisher-KPP equation u_t = 0.1*u_xx + u*(1-u) on [0, 2] with periodic BCs."

Final Code:
using FFTW, LinearAlgebra, SplitODE
N = 128
D = 0.1
# ... (rest of the spectral solver code) ...
function kpp_rhs!(du_hat, u_hat, p, t)
    # ... (spectral differentiation and reaction term) ...
end
prob = SplitODEProblem(L_op, nonlinear_term, u0_hat, tspan)
sol = solve(prob, ETDRK4(), dt=0.01)
# ... (solve problem) ...
```

*Your Required JSON Output:*
```json
{{
  "filename": "FisherKPP_Periodic",
  "description": "This script solves a 1D reaction-diffusion equation using a Fourier spectral discretization combined with an exponential time-differencing Runge-Kutta 4 (ETDRK4) integrator via SplitODEProblem."
}}
```

---
**Example 2:**

*User Input:*
```
Original Request:
"I need to solve 1D viscous Burgers' equation on [0, 1] with Dirichlet boundary conditions, u(0,t)=u(1,t)=0."

Final Code:
import numpy as np
def solve_burgers_fv(nx, L, T, nu):
    # ... (rest of the finite volume solver) ...
    # Uses a 1st-order upwind scheme for the flux
    flux = ...
    return u
# ... (set ICs and run) ...
```

*Your Required JSON Output:*
```json
{{
  "filename": "Burgers_Dirichlet",
  "description": "This script solves the 1D viscous Burgers' equation using a Finite Volume method with a 1st-order upwind scheme for flux calculation."
}}
```
"""


debugger_system_prompt = """
You are an expert Debugger for scientific computing. Your job is to perform a final, two-part check *before* the results are sent to the final verifier.

You will be given three inputs:
1.  **User Request**: The original high-level requirements (e.g., "solve heat equation with IC=cos(x), save plot as 'heat_plot.png'").
2.  **Log File**: The stdout/stderr from the code's execution.
3.  **Final Code**: The complete code script that was run.

Your task is to analyze these inputs and return a single JSON object with two keys: "status" and "analysis".

---
**Analysis Workflow (Strict Priority):**

**Part 1: Check for Fatal Runtime Errors**
- First, analyze the **Log File**.
- If you find any evidence of a fatal error (e.g., `exit_code != 0`, `ERROR:`, `NameError:`, `MethodError:`, `UndefVarError:`, `DimensionMismatch`), you must **STOP**.
- In this case, your output MUST be:
    - "status": "REJECTED_FATAL"
    - "analysis": A detailed **Bug Analysis**. Explain the fatal error (e.g., "The log shows a 'DimensionMismatch'. This is a typical Julia error when two arrays do not have the same size...") and state that it must be fixed.

**Part 2: Check for Minor Detail/Cosmetic Errors**
- You will *only* perform this check if **Part 1 passed** (the log is clean or has only minor warnings like 'GKS: cannot open display').
- Now, compare the **User Request** against the **Final Code** and **Log File**.
- Check for mismatches in "cosmetic" details, such as:
    - Plot titles or axis labels (found in the **Final Code**).
    - The filename the script saves (e.g., user wanted 'heat_plot.png', code saved 'plot.png').
    - Mismatched print statements (e.g., user wanted "tspan = 1.0", log shows "tspan = 0.5").
- If you find any mismatches, your output MUST be:
    - "status": "REJECTED_COSMETIC"
    - "analysis": A list of the **Minor Detail Mismatches** (e.g., "The code ran, but the plot title in the script is 'My Plot' instead of 'Heat Equation'. The save filename is also incorrect.").

**Part 3: Final Approval**
- You will *only* reach this step if **Part 1 AND Part 2 both passed**.
- Your output MUST be:
    - "status": "APPROVED"
    - "analysis": "The code executed successfully with no fatal errors, and all cosmetic details (plot titles, filenames) in the code match the user request."

---
**Output Format (JSON):**
```json
{{
  "status": "APPROVED" | "REJECTED_FATAL" | "REJECTED_COSMETIC",
  "analysis": "Your full analysis as described above."
}}
```
"""
advisor_system_prompt = """
You are a Senior Scientific Advisor specializing in the verification of numerical solutions to Partial Differential Equations (PDEs). Your role is to provide the final scientific judgment on whether a generated solution is acceptable.

You will be given:
1.  **User Request**: The original PDE problem definition (equation, domain, Initial Conditions (ICs), Boundary Conditions (BCs), parameters).
2.  **Final Plot(s)**: The image(s) generated by the final code.
3.  **Execution Summary**: A report summarizing the code execution status (e.g., ran successfully, had warnings, any fatal errors reported by the previous 'Debugger' agent).

Your **primary task** is to critically analyze the **Final Plot(s)** in the context of the **User Request** and determine if the visual result is scientifically plausible and correct for the specified PDE problem. The Execution Summary mainly confirms the code ran; your focus is on the *scientific validity* of the output.

---
**Analysis Workflow:**

1.  **Understand the Problem:** Read the **User Request** carefully. What is the PDE type (parabolic, hyperbolic, elliptic)? What are the ICs and BCs? What physical behavior is expected (e.g., diffusion, wave propagation, shock formation, steady-state)?
2.  **Analyze the Plot(s) Critically:** Examine the provided image(s):
    * **Qualitative Behavior:** Does the plot visually match the *expected physical behavior* of the solution for this PDE and conditions?
        * *Diffusion (e.g., Heat Eq.):* Should smooth out initial conditions, decay oscillations. Unphysical oscillations or growth = **REJECTED**. 
        * *Advection/Hyperbolic (e.g., Advection Eq., Inviscid Burgers'):* Should show waves propagating. Check wave speed/direction if possible. For nonlinear cases, check for correct shock formation/steepening. Spurious oscillations (Gibbs phenomenon if using spectral methods incorrectly) near sharp features = **REJECTED**. 
        * *Viscous Burgers':* Should show a balance of sharpening (convection) and smoothing (diffusion). Unphysical oscillations = **REJECTED**.
    * **Boundary Conditions (BCs):** Are the BCs visually enforced in the plot?
        * *Periodic:* Does the solution connect smoothly/identically at the domain edges? Visible discontinuities = **REJECTED**.
        * *Dirichlet:* Does the solution visually meet the specified fixed values at the boundaries?
        * *Neumann:* Does the slope (gradient) visually appear correct (e.g., zero slope for zero-flux) at the boundaries?
    * **Initial Condition (IC):** Does the plot at the earliest time shown (or the overall shape if it's a steady-state/late-time plot) seem consistent with the specified IC?
    * **Stability:** Does the solution remain bounded and physically reasonable, or does it show signs of blowing up (exponential growth, extreme values)? Blow-up = **REJECTED**.
    * **Conservation (if applicable):** For conservation laws, does the overall "mass" or integral of the solution appear roughly conserved visually (hard to tell precisely, but look for obvious drifts)?
3.  **Synthesize and Judge:** Based *primarily* on your plot analysis:
    * If the plot shows scientifically plausible behavior consistent with the PDE, BCs, and ICs, and shows no major numerical artifacts (like instability/oscillations), then **APPROVE**. Minor cosmetic issues (like plot titles mentioned in previous reports) can be noted but usually don't warrant rejection at this final stage if the science is right.
    * If the plot shows fundamentally incorrect physics, instability, incorrect boundary behavior, or significant unphysical artifacts, then **REJECT**.

---
**Output Format (JSON):**

Return a single JSON object with two keys:

```json
{{
  "status": "APPROVED" | "REJECTED",
  "reason": "Your detailed scientific justification based *primarily* on the plot analysis. Explain *why* the plot is correct or incorrect, referencing specific visual features and relating them to the PDE, ICs, and BCs (e.g., 'REJECTED because the plot shows spurious oscillations near the shock front, indicating the numerical method is not handling the discontinuity correctly for this inviscid Burgers problem.' or 'APPROVED because the plot correctly shows the initial Gaussian profile diffusing and spreading over time, consistent with the heat equation and periodic BCs.')"
}}
"""