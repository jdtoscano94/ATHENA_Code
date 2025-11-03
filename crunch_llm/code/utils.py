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

# Report
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.platypus import Image as ReportLabImage
from reportlab.lib import utils as reportlab_utils

# Encoding
import base64
import mimetypes
import io  
import PIL.Image


current_file_dir = Path(__file__).parent 
# Build the full path to the .env file in that *same directory*
dotenv_path = current_file_dir / '.env'
# Explicitly load that file
load_dotenv(dotenv_path=dotenv_path)


def initialize_local_data(
    db_path: str,
    collection_name: str = 'PINNs_Codes',
    provider: str = 'ollama',
    model_name: str =None
):
    embeddings: Embeddings = None
    model: str = model_name 
    # No pre-initialized function provided, use provider/model_name logic
    provider = provider.lower().strip() # Normalize the provider name
    try:
        if provider == 'ollama':
            # Use default local model if none specified
            model = model_name if model_name else 'bge-m3:567m'
            print(f"Initializing with Ollama. Model: {model}")
            embeddings = OllamaEmbeddings(model=model)

        elif provider == 'google':
            # Use default Gemini model if none specified
            model = model_name if model_name else 'models/text-embedding-004'
            print(f"Initializing with Google (Gemini). Model: {model}")
            
            if not os.getenv("GOOGLE_API_KEY"):
                print("Error: GOOGLE_API_KEY not found in environment variables.")
                return None
            embeddings = GoogleGenerativeAIEmbeddings(model=model)
        elif provider == 'openai':
            model = model_name if model_name else 'text-embedding-3-small'
            print(f"Initializing with OpenAI. Model: {model}")
            if not os.getenv("OPENAI_API_KEY"):
                print("Error: OPENAI_API_KEY not found in environment variables.")
                return None
            embeddings = OpenAIEmbeddings(model=model)
        else:
            print(f"Error: Unsupported provider '{provider}'. Must be 'ollama', 'google', or 'openai'.")
            return None

    except Exception as e:
        print(f"Error initializing embedding model '{model_name}' for {provider}: {e}")
        if provider == 'ollama':
             print("Please ensure Ollama is running.")
        return None
    try:
        client = chromadb.PersistentClient(path=db_path)
        
        # 2. Pass the client and embeddings to the LangChain wrapper
        vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings,
        )
        
        print(f"✅ Vector store initialized successfully from path: {db_path}")
        print(f"   Provider: {provider}, Model: {model}, Collection: {collection_name}")
        
        return vector_store

    except Exception as e:
        print(f"Error initializing ChromaDB client at {db_path}: {e}")
        return None


def extract_filename(response_text: str) -> Optional[str]:
    pattern = re.compile(
        r"[`']?([\w\.-]+\.(?:jl|py|m|cpp|c|f90|f))[`']?"
    )
    match = pattern.search(response_text)
    if match:
        return match.group(1)
    else:
        return 'You did not give me the right file name. Please look carefully again and only the filename of the code you suggest for this problem'


def check_approval(reviewer_message: str) -> bool:
    # --- Keyword Sets ---
    POSITIVE_WORDS = {
        'yes', 'proceed', 'good', 'correct', 'approved','aprove',
        'agreed', 'excellent',  'that\'s right'
    }
    NEGATIVE_WORDS = {
        'no', 'wrong', 'flawed','incomplete' ,'incorrect', 'rejected','reject'
        're-think', 're-analyze', 'however', 'but', 'fail'
    }
    text = reviewer_message.lower()
    text = re.sub(r'[^\w\s\']', '', text) 
    words = set(text.split())
    found_positive = not POSITIVE_WORDS.isdisjoint(words)
    found_negative = not NEGATIVE_WORDS.isdisjoint(words)
    if found_positive and not found_negative:
        return True
    else:
        return False


def _extract_line_number(old_code_block: str) -> Optional[int]:
    """Helper function to find the line number from an annotated string.
       Accepts '# LINE_N', '# line N', '% LINE_N', '% line N', etc."""
    match = re.search(r'(?:#|%|//|!)?\s*(?:LINE_|line\s+)(\d+)', old_code_block, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r'(\d+)\s*$', old_code_block)
    if match:
         print(f"  - WARNING: Found line number '{match.group(1)}' without standard '# line/LINE_' prefix in: '{old_code_block}'. Proceeding cautiously.")
         return int(match.group(1))
    return None


def apply_patch(original_code: str, patch_data: Dict) -> Tuple[str, List[str]]:
    print("--- Applying deterministic line range patch ---")
    errors: List[str] = []
    lines = original_code.splitlines()
    num_original_lines = len(lines)

    changes = patch_data.get('changes', [])

    if not changes:
        print("No changes found in the patch. Returning original code.")
        return original_code, errors

    # --- Process changes in REVERSE order of start_line to avoid index shifting ---
    # Sort by 'start_line' descending. Handle potential missing keys gracefully.
    try:
        sorted_changes = sorted(changes, key=lambda x: x.get('start_line', -1), reverse=True)
    except TypeError:
        errors.append("ERROR: Invalid 'start_line' found in changes list (must be numbers). Cannot apply patch.")
        print("  - ❌ ERROR: Invalid 'start_line' found. Aborting.")
        return original_code, errors # Return original code if sorting fails

    applied_change = False

    for i, change in enumerate(sorted_changes):
        # Validate change object add
        start_line = change.get('start_line')
        end_line = change.get('end_line')
        action = change.get('action')
        new_block = change.get('new_code_block') # Can be None for delete/replace if agent omits

        required_keys = ['start_line', 'end_line', 'action']
        if not all(k in change for k in required_keys) or new_block is None:
             error_msg = f"ERROR on change #{i+1} (original order might differ): Invalid change object format - Missing keys or 'new_code_block'. Change: {change}"
             print(f"  - ❌ {error_msg}")
             errors.append(error_msg)
             continue

        # Validate types and values
        if not all(isinstance(ln, int) for ln in [start_line, end_line]) or not isinstance(action, str) or not isinstance(new_block, str):
             error_msg = f"ERROR on change #{i+1}: Invalid data types in change object. Change: {change}"
             print(f"  - ❌ {error_msg}")
             errors.append(error_msg)
             continue

        if start_line <= 0 or end_line < start_line:
             error_msg = f"ERROR on change #{i+1}: Invalid line numbers (must be positive, end >= start). start={start_line}, end={end_line}"
             print(f"  - ❌ {error_msg}")
             errors.append(error_msg)
             continue

        # Convert to 0-based index
        start_idx = start_line - 1
        end_idx = end_line - 1 # Inclusive index for slicing end point

        # Bounds check (using original length for safety check, though list length changes)
        if start_idx >= num_original_lines or end_idx >= num_original_lines or start_idx < 0:
             error_msg = f"ERROR on change #{i+1}: Line numbers {start_line}-{end_line} are out of bounds for original code length {num_original_lines}."
             print(f"  - ❌ {error_msg}")
             errors.append(error_msg)
             continue

        # Prepare new lines (split new_block string)
        # Handle empty string case for deletion correctly
        new_lines = new_block.splitlines() if new_block else []

        # Apply the action
        print(f"  - Processing change for lines {start_line}-{end_line}: action='{action}'")
        try:
            if action == "replace":
                print(f"    - Replacing lines {start_line}-{end_line} with {len(new_lines)} new lines.")
                lines[start_idx : end_idx + 1] = new_lines
                applied_change = True
            elif action == "delete":
                print(f"    - Deleting lines {start_line}-{end_line}.")
                if new_block:
                     print(f"    - WARNING: 'new_code_block' provided for delete action will be ignored.")
                del lines[start_idx : end_idx + 1]
                applied_change = True
            elif action == "insert_after":
                print(f"    - Inserting {len(new_lines)} lines after line {end_line}.")
                # Insert *after* the end_idx
                lines[end_idx + 1 : end_idx + 1] = new_lines
                applied_change = True
            else:
                error_msg = f"ERROR on change #{i+1}: Unknown action '{action}'."
                print(f"  - ❌ {error_msg}")
                errors.append(error_msg)

        except IndexError as e:
            # Catch potential errors if indices become invalid despite reverse sorting (should be rare)
            error_msg = f"ERROR on change #{i+1}: IndexError during {action} for lines {start_line}-{end_line}. This might indicate conflicting changes. Details: {e}"
            print(f"  - ❌ {error_msg}")
            errors.append(error_msg)
        except Exception as e:
            # Catch any other unexpected errors during list manipulation
            error_msg = f"UNEXPECTED ERROR on change #{i+1} ({action} for lines {start_line}-{end_line}): {e}"
            print(f"  - ❌ {error_msg}")
            errors.append(error_msg)

    # Join the modified list of lines back into a single string
    modified_code = "\n".join(lines)

    # Reverse errors list to match original change order for reporting
    errors.reverse()

    return modified_code, errors


def remove_line_number_tokens(annotated_code: str) -> str:
    pattern = re.compile(r"(\s*(?:#|%|//|!)\s*(?:LINE_|line\s+)\d+\s*)$", re.IGNORECASE | re.MULTILINE)

    # Replace the matched comment pattern with an empty string
    cleaned_code = pattern.sub("", annotated_code)

    return cleaned_code


def add_line_numbers_to_code(code_content: str, filename: str) -> str:
    # Dictionary mapping language names to their comment syntax
    comment_styles = {
        'python': '#',
        'julia': '#',
        'matlab': '%',
        'c': '//',
        'cpp': '//',
        'fortran': '!'
    }
    
    # Dictionary mapping file extensions to language names
    extension_map = {
        '.py': 'python',
        '.jl': 'julia',
        '.m': 'matlab',
        '.c': 'c',
        '.cpp': 'cpp',
        '.f90': 'fortran',
        '.f': 'fortran'
    }

    # Extract the file extension
    file_extension = Path(filename).suffix.lower()
    
    # Determine the language based on the extension
    language = extension_map.get(file_extension, 'unknown') # Default to unknown
    
    # Get the correct comment character, defaulting to '#' if language is unknown
    comment = comment_styles.get(language, '#')
    
    # Split the original code into a list of lines
    lines = code_content.splitlines()
    
    # Use a list comprehension to create the new, annotated lines
    # using the specified format: # LINE_N
    numbered_lines = [f"{line} {comment} LINE_{i}" for i, line in enumerate(lines, start=1)]
    
    # Join the annotated lines back into a single string
    return "\n".join(numbered_lines)


def get_comment_char(language: str) -> str:
    """
    Returns the correct single-line comment character for a given language.
    """
    comment_styles = {
        'python': '#',
        'julia': '#',
        'matlab': '%',
        'c': '//',
        'cpp': '//',
        'fortran': '!'
    }
    # Default to '#' if language is unknown
    return comment_styles.get(language.lower(), '#')



def remove_all_comments(code_content: str, language: str) -> str:
    comment_patterns = {
        'python': [
            r'\"\"\"[\s\S]*?\"\"\"',  # Triple double-quote blocks
            r"\'\'\'[\s\S]*?\'\'\'",  # Triple single-quote blocks
            r"\s*#.*$"           # Single-line comments
        ],
        'julia': [
            r'#=[\s\S]*?=#',     # Julia block comments
            r"\s*#.*$"           # Single-line comments
        ],
        'matlab': [
            r'%\{[\s\S]*?%\}',   # MATLAB block comments
            r"\s*%.*$"           # Single-line comments
        ],
        'c': [
            r'/\*[\s\S]*?\*/',   # C-style block comments
            r"\s*//.*$"          # C-style single-line comments
        ],
        'cpp': [
            r'/\*[\s\S]*?\*/',   # C++-style block comments
            r"\s*//.*$"          # C++-style single-line comments
        ],
        'fortran': [
            r"\s*!.*$"           # Fortran single-line comments
        ],
    }
    
    patterns = comment_patterns.get(language.lower())
    
    cleaned_code = code_content
    
    if patterns:
        for pattern in patterns:
            if "$" in pattern:
                cleaned_code = re.sub(pattern, "", cleaned_code, flags=re.MULTILINE)
            else:
                cleaned_code = re.sub(pattern, "", cleaned_code)
        cleaned_code = re.sub(r"^\s*\n", "", cleaned_code, flags=re.MULTILINE)
        return cleaned_code.strip()
    else:
        print(f"Warning: No comment pattern defined for language '{language}'. Returning original code.")
        return code_content


def run_shell_script(script_path: str, 
                     script_args: List[str] = None, # <-- ADD THIS
                     log_path: str = "shell_run.log", 
                     shell_executable: str = None) -> bool:
    """
    Runs a shell script (.sh) using a subprocess and captures its output to a log file.

    Args:
        script_path (str): The path to the .sh script to run.
        script_args (List[str], optional): Arguments to pass to the shell script.
        log_path (str): The path to the file where stdout and stderr will be saved.
        shell_executable (str): The command or full path to the shell (e.g., 'bash', 'sh').
    """
    
    if not os.path.exists(script_path):
        print(f"Error: Shell script not found at '{script_path}'")
        return False

    if shell_executable is None:
        shell_executable = shutil.which('sh') 
        if shell_executable is None:
            shell_executable = shutil.which('bash')
            if shell_executable is None:
                print("Error: Could not find 'sh' or 'bash' executable in PATH.")
                return False
        print(f"Using shell found at: {shell_executable}")

    # --- MODIFIED COMMAND ---
    command = [shell_executable, script_path]
    if script_args:
        command.extend(script_args) # Pass arguments to the script
    # --- END MODIFICATION ---

    print(f"Running command: {' '.join(command)}")

    try:
        process = subprocess.run(
            command,
            capture_output=True, 
            text=True,           
            check=False          
        )

        with open(log_path, 'w', encoding='utf-8') as log_file:
            log_file.write("--- Standard Output ---\n")
            log_file.write(process.stdout + "\n")
            log_file.write("\n--- Standard Error ---\n")
            log_file.write(process.stderr + "\n")
            log_file.write(f"\n--- Exit Code: {process.returncode} ---\n")

        if process.returncode == 0:
            print(f"Shell script '{script_path}' executed successfully.")
            print(f"Output saved to '{log_path}'.")
            return True
        else:
            print(f"Error: Shell script '{script_path}' failed with exit code {process.returncode}.")
            print(f"Check '{log_path}' for details.")
            return False

    except FileNotFoundError:
        print(f"Error: Shell executable '{shell_executable}' not found at the specified path.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        try:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"\n--- Python Error during execution: {e} ---\n")
        except:
            pass 
        return False

def get_language_from_filename(filename: str) -> str:
    extension_map = {
        '.py': 'python',
        '.jl': 'julia',
        '.m': 'matlab',
        '.c': 'c',
        '.cpp': 'cpp',
        '.f90': 'fortran',
        '.f': 'fortran'
    }    
    try:
        file_extension = os.path.splitext(filename)[1].lower()
    except Exception as e:
        print(f"Error splitting filename '{filename}': {e}")
        return 'unknown'
    language = extension_map.get(file_extension, 'unknown')
    if language == 'unknown':
        print(f"Warning: Could not determine language for filename '{filename}' with extension '{file_extension}'.")
    return language

def create_folder_and_save_code(
    code_details: dict, 
    code_content: str, 
    language: str, 
    base_results_dir: str = "/users/jdtoscan/data/jdtoscan/LLM/Results/generated_codes",
    add_description: bool = True,
) -> (str, str, str):
    try:
        # --- 1. Get details from the dictionary ---
        filename_no_ext = code_details.get('filename')
        description = code_details.get('description')
        lang_lower = language.lower()
        if not filename_no_ext or not description:
            print(f"Error: 'filename' or 'description' missing from code_details dict.")
            return None, None, None
        lang_map = {
            'julia': ('.jl', '#'),
            'python': ('.py', '#'),
            'matlab': ('.m', '%'),
            'c': ('.c', '//'),
            'cpp': ('.cpp', '//'),
            'fortran': ('.f90', '!')
        }
        
        extension, comment_char = lang_map.get(lang_lower, ('.txt', '#'))
        
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M") 
        
        project_dir = f"{base_results_dir}/{filename_no_ext}/{timestamp}/"
        os.makedirs(project_dir, exist_ok=True)
        print(f"Created project directory: {project_dir}")
        
        final_filename = filename_no_ext + extension
        final_filepath = os.path.join(project_dir, final_filename)

        # --- 5. Format the final code with the description header ---
        if add_description:
            wrapped_description = f"\n{comment_char} ".join(description.splitlines())
            header = f"{comment_char} {wrapped_description}\n\n"
            final_code_to_save = header + code_content
        else:
            final_code_to_save= code_content

        # --- 6. Save the file ---
        with open(final_filepath, 'w', encoding='utf-8') as f:
            f.write(final_code_to_save)
        print(f"Successfully saved code to: {final_filepath}")

        # --- 7. Return the paths ---
        return project_dir, final_filename, final_filepath

    except Exception as e:
        print(f"Error during code organization: {e}")
        return None, None, None


def run_and_stream_shell_script(
    shell_script_path: str,
    script_args: List[str],
    log_path: str
) -> Tuple[bool, str]:
    
    cmd = ['/bin/bash', shell_script_path] + script_args
    
    log_content = []
    
    print(f"\n--- [Streaming Output silently to {log_path}] ---")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1 
        )

        with open(log_path, 'w', encoding='utf-8') as log_f:
            for line in iter(process.stdout.readline, ''):
                log_f.write(line)
                log_f.flush()
                log_content.append(line)
        process.stdout.close()
        return_code = process.wait()
        print(f"--- [End of Stream. Exit Code: {return_code}] ---")
        full_log_text = "".join(log_content)
        success = (return_code == 0)
        return success, full_log_text
    except Exception as e:
        print(f"\n--- [ERROR DURING SUBPROCESS EXECUTION] ---")
        print(str(e))
        print(f"--- [End of Stream] ---")
        error_message = f"Failed to execute script: {str(e)}\n\nPartial Log:\n{''.join(log_content)}"
        
        try:
            with open(log_path, 'a', encoding='utf-8') as log_f:
                log_f.write(f"\n\nPYTHON EXECUTION ERROR:\n{str(e)}")
        except:
            pass # Ignore if log writing also fails
            
        return False, error_message

def run_code(project_dir, final_filename, final_filepath):
    def get_language_from_filename(filename):
        if filename.endswith('.py'):
            return 'python'
        elif filename.endswith('.jl'):
            return 'julia'
        return 'unknown'

    code_language = get_language_from_filename(final_filename)
    shell_script_path = "" # Initialize
    
    if code_language == 'julia':
        shell_script_path = "/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Documents/Scripts/run_julia.sh"
        
    elif code_language == 'python':
        shell_script_path = "/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Documents/Scripts/run_python.sh"
    
    else:
        print(f"❌ Error: No run script defined for language '{code_language}'. Cannot run.")
        return False, "", "", "" # Return all 4 items on failure

    print(f"Attempting to run {final_filepath}...")
    log_file_name_out = os.path.join(project_dir, final_filename + '.out')
    log_file_name_err = os.path.join(project_dir, final_filename + '.err')

    script_args_list = [final_filepath, log_file_name_err]

    success, log_content = run_and_stream_shell_script(
        shell_script_path, 
        script_args=script_args_list,
        log_path=log_file_name_out # This is for stdout
    )

    if success:
        print(f"✅ Successfully ran {final_filename}.")
    else:
        print(f"❌ Error running {final_filename}.")
    return success, log_file_name_out


def read_files(log_file_name, verbose=False):
    log_content = ''
    try:
        with open(log_file_name, 'r') as f:
            log_content = f.read() 
        if verbose:
            print(log_content)
        return log_content, True
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_name}")
        return 'File not found', False
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return 'Could not read the file', False

def find_plots(project_dir: str, display_plots: bool = True) -> List[str]: # <-- MODIFIED SIGNATURE
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg')
    found_images = []
    if not os.path.isdir(project_dir):
        print(f"Warning: Project directory not found at {project_dir}")
        return []
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                full_path = os.path.join(root, file)
                print(f"--- Found plot: {full_path} ---")     
                if display_plots:
                    print("Displaying plot:")
                    try:
                        # This is the best way to display in a notebook
                        display(Image(filename=full_path, width=600))
                    except Exception as e:
                        print(f"Error displaying image {full_path}: {e}")             
                found_images.append(full_path)
    return found_images

def load_image_as_base64(filepath: str) -> str:
    try:
        with open(filepath, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error loading image {filepath}: {e}")
        return ""

### Utils PINNs

def clean_code_comments(full_code_content: str) -> str:
    
    pattern = re.compile(
        r'^\s*#\s*CODE ANALYSIS.*?(?:\n\s*#\s*={10,}.*?$)', 
        re.DOTALL | re.MULTILINE
    )
    cleaned_code = re.sub(pattern, '', full_code_content)    
    cleaned_code = re.sub(r'\n{3,}', '\n\n', cleaned_code)
    return cleaned_code.strip()

def split_code_into_cells(file_content: str) -> Dict[str, str]:
    pattern = r'(^\s*# ## Cell \d+:.*)'
    parts = re.split(pattern, file_content, flags=re.MULTILINE)
    cell_dict: Dict[str, str] = {}
    prologue = parts[0].strip()
    if prologue:
        cell_dict["Prologue"] = prologue
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            delimiter_line = parts[i]
            cell_content = parts[i+1]
            match = re.search(r'(Cell \d+)', delimiter_line)
            if match:
                cell_id = match.group(1) # e.g., "Cell 7"
                full_cell_code = delimiter_line + cell_content
                cell_dict[cell_id] = full_cell_code.strip()  
    return cell_dict

def reconstruct_code_from_cells(cell_dict: Dict[str, str]) -> str:
    def get_cell_number(key_str: str) -> int:
        """Helper function to extract the cell number for sorting."""
        match = re.search(r'Cell\s+(\d+)', key_str)
        if match:
            return int(match.group(1))
        return 0
    reconstructed_parts: List[str] = []
    if "prologue" in cell_dict:
        reconstructed_parts.append(cell_dict["prologue"])
    cell_keys = [key for key in cell_dict.keys() if key != "prologue"]    
    sorted_cell_keys = sorted(cell_keys, key=get_cell_number)
    for key in sorted_cell_keys:
        reconstructed_parts.append(cell_dict[key])
    return "\n\n".join(reconstructed_parts)


    # Create Report


def create_consolidated_report_chronological(
    project_dir: str, 
    output_pdf_path: str = "consolidated_report_chronological.pdf"
) -> tuple[bool, str | None]:
    """
    Creates a consolidated PDF report from all text and image files in a directory,
    sorting them CHRONOLOGICALLY by modification time.

    Returns:
        (bool, str | None): A tuple of (success_status, path_to_pdf_or_None)
    """
    print(f"Attempting to create CHRONOLOGICAL consolidated report: {output_pdf_path}")

    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg')
    text_extensions = ('.txt', '.log', '.out') 
    
    reportable_files = [] # This will store tuples of (mtime, full_path)
    
    if not os.path.isdir(project_dir):
        print(f"Warning: Project directory not found at {project_dir}")
        return False, None

    # --- 1. Find all reportable files and get their modification time ---
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.lower().endswith(image_extensions) or file.lower().endswith(text_extensions):
                full_path = os.path.join(root, file)
                try:
                    # Get modification time
                    mtime = os.path.getmtime(full_path)
                    reportable_files.append((mtime, full_path))
                except OSError as e:
                    print(f"Warning: Could not get mtime for {full_path}. Skipping. Error: {e}")

    if not reportable_files:
        print("No reportable files (.png, .jpg, .txt, .log, .out) found.")
        return False, None
        
    # --- 2. Sort files by modification time (the first element of the tuple) ---
    reportable_files.sort(key=lambda x: x[0])
    
    print(f"Found {len(reportable_files)} reportable files. Building PDF in chronological order...")
    
    # --- 3. Setup PDF Document ---
    full_output_path = os.path.join(project_dir, output_pdf_path)

    doc = SimpleDocTemplate(full_output_path, pagesize=A4,
                            rightMargin=inch, leftMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    
    styles = getSampleStyleSheet()
    filename_style = styles['Heading3']
    
    pre_style = ParagraphStyle(
        name='PreformattedText',
        parent=styles['Code'], # Use 'Code' as a base
        fontSize=8,            # Smaller font for logs
        leading=10,            # Line spacing
        alignment=TA_LEFT,
        wordWrap='CJK',        # Handle long lines without spaces
    )
    
    Story = [] # This is our list of PDF elements

    # --- 4. Build the PDF Story (now in chronological order) ---
    for mtime, file_path in reportable_files:
        filename = os.path.basename(file_path)
        
        try:
            p_filename = Paragraph(filename, filename_style)
            Story.append(p_filename)
            Story.append(Spacer(1, 0.15 * inch))

            if file_path.lower().endswith(image_extensions):
                img_reader = reportlab_utils.ImageReader(file_path)
                iw, ih = img_reader.getSize()
                if iw == 0:
                     print(f"Warning: Image {filename} has width 0. Skipping.")
                     continue
                aspect = ih / float(iw)
                new_height = (doc.width * aspect)
                img = ReportLabImage(file_path, width=doc.width, height=new_height)
                Story.append(img)

            elif file_path.lower().endswith(text_extensions):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text_content = f.read()
                
                pre = Preformatted(text_content, pre_style)
                Story.append(pre)
            
            Story.append(Spacer(1, 0.5 * inch))

        except Exception as e:
            print(f"Warning: Could not add {filename} to PDF. Error: {e}")

    # --- 5. Build (save) the PDF ---
    try:
        doc.build(Story)
        print(f"✅ Successfully created chronological report at: {full_output_path}")
        return True, full_output_path
    except Exception as e:
        print(f"Error building the final PDF: {e}")
        return False, None


def encode_image_to_data_url(filepath: str, max_width: int = 800) -> str:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at {filepath}")
        
    mime_type, _ = mimetypes.guess_type(filepath)
    if mime_type is None:
        mime_type = "image/png"
    
    try:
        img = PIL.Image.open(filepath)
        
        if img.width > max_width:
            width_percent = (max_width / float(img.width))
            new_height = int((float(img.height) * float(width_percent)))
            img = img.resize((max_width, new_height), PIL.Image.LANCZOS)
            print(f"DEBUG: Image resized to {max_width}x{new_height}")
        buffer = io.BytesIO()
        img_format = mime_type.split('/')[-1].upper()
        if img_format == 'JPEG':
            img = img.convert('RGB')
        img.save(buffer, format=img_format)
        image_bytes = buffer.getvalue()
    except Exception as e:
        print(f"Warning: Pillow processing failed ({e}). Falling back to original file.")
        with open(filepath, "rb") as image_file:
            image_bytes = image_file.read()
    base64_string = base64.b64encode(image_bytes).decode('utf-8')

    data_url = f"data:{mime_type};base64,{base64_string}"
    return data_url


