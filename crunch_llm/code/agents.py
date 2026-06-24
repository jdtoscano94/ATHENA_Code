# ============================================================================
# IMPORTS
# ============================================================================
import os
import json
import re
import select
import sys
import time
import chromadb
from dataclasses import dataclass
from typing import Union, Any, List, Dict, Optional
from dotenv import load_dotenv
import subprocess
from datetime import datetime
# LangChain Core
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.documents import Document
from langchain_core.tools import tool

# LangChain Agents 
from langchain.agents import create_agent

# LLM Providers
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_xai import ChatXAI

# Vector Store
from langchain_chroma import Chroma
from pathlib import Path 


# Load environment variables
current_file_dir = Path(__file__).parent 
dotenv_path = current_file_dir / '.env'

load_dotenv(dotenv_path=dotenv_path)



# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_message_content(content: Any) -> str:
    """
    Extract text content from different LLM response formats.
    
    Different LLM providers return responses in different formats:
    - Plain string (most common)
    - List of dicts with 'type' and 'text' keys (some models)
    - Other structured formats
    
    Args:
        content: The message content from an AI response
        
    Returns:
        Extracted text as a string
    """
    if isinstance(content, str):
        # Plain string response (most common)
        return content
    elif isinstance(content, list):
        # Structured response (e.g., from some models)
        # Extract text from list of content blocks
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                # Handle {'type': 'text', 'text': '...'} format
                if block.get('type') == 'text' and 'text' in block:
                    text_parts.append(block['text'])
                # Handle other dict formats that might have 'text' key
                elif 'text' in block:
                    text_parts.append(block['text'])
            elif isinstance(block, str):
                # Handle plain strings in the list
                text_parts.append(block)
        
        if text_parts:
            return '\n'.join(text_parts)
        else:
            # Fallback: convert the whole list to string
            return str(content)
    else:
        # Fallback for any other format
        return str(content)


# ============================================================================
# DEFAULT SYSTEM PROMPT
# ============================================================================

def _strip_ansi(text: str) -> str:
    ansi_escape = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\r")
    return ansi_escape.sub("", text)


@dataclass
class CodexRunResult:
    returncode: int
    output: str
    command: List[str]


class CodexDirectAgent:
    """
    Direct Codex exec wrapper.

    Usage:
    - pass a task prompt
    - set system_directory (Codex working root)
    - optionally pass a system_prompt to prefix instructions
    """

    def __init__(
        self,
        model: Optional[str] = None,
        timeout_seconds: int = 600,
        stream_output: bool = True,
        sanitize_output: bool = True,
        bypass_sandbox: bool = True,
    ):
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.stream_output = stream_output
        self.sanitize_output = sanitize_output
        self.bypass_sandbox = bypass_sandbox

    def run(self, prompt: str, system_directory: str, system_prompt: str = "") -> CodexRunResult:
        workdir = Path(system_directory).resolve()
        if not workdir.exists() or not workdir.is_dir():
            raise ValueError(f"system_directory must be an existing directory: {workdir}")

        final_prompt = prompt.strip()
        if system_prompt.strip():
            final_prompt = f"{system_prompt.strip()}\n\nTask:\n{final_prompt}"

        cmd = ["codex", "exec", "--color", "never", "-C", str(workdir)]
        if self.bypass_sandbox:
            cmd.insert(2, "--dangerously-bypass-approvals-and-sandbox")
        else:
            cmd.insert(2, "--full-auto")
            cmd.insert(3, "-s")
            cmd.insert(4, "workspace-write")

        if self.model:
            cmd.extend(["--model", self.model])

        cmd.append(final_prompt)

        process = subprocess.Popen(
            cmd,
            cwd=str(workdir),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        output_parts: List[str] = []
        deadline = time.time() + self.timeout_seconds

        try:
            while True:
                if time.time() > deadline:
                    process.kill()
                    raise TimeoutError(f"Codex exec timed out after {self.timeout_seconds}s")

                if process.stdout is None:
                    break

                ready, _, _ = select.select([process.stdout], [], [], 0.2)
                if ready:
                    chunk = process.stdout.readline()
                    if chunk:
                        output_parts.append(chunk)
                        if self.stream_output:
                            out = _strip_ansi(chunk) if self.sanitize_output else chunk
                            sys.stdout.write(out)
                            sys.stdout.flush()
                    elif process.poll() is not None:
                        break
                elif process.poll() is not None:
                    break

            if process.stdout is not None:
                tail = process.stdout.read() or ""
                if tail:
                    output_parts.append(tail)
                    if self.stream_output:
                        out = _strip_ansi(tail) if self.sanitize_output else tail
                        sys.stdout.write(out)
                        sys.stdout.flush()
        finally:
            process.wait(timeout=5)

        output = "".join(output_parts)
        if self.sanitize_output:
            output = _strip_ansi(output)

        return CodexRunResult(returncode=process.returncode or 0, output=output, command=cmd)

class CodingAgent:
    """
    Autonomous coding agent with tool-based execution and memory.
    
    The agent writes code, executes it using tools, and returns text explanations.
    No rigid requirements on output files - the agent is flexible and just explains
    what it did. We can read files later if needed.
    
    Memory: Maintains conversation history across iterations.
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic",
        system_prompt: str = 'You are a highly intelligent coding assistant. ',
        temperature: float = 0,
        python_executable: str = "python",
        work_dir: str = "./coding_workspace",
        verbose: bool = True
    ):
        """
        Initialize the coding agent.
        
        Args:
            model: Model name
            provider: LLM provider ("anthropic", "openai", "google", "ollama", "xai")
            system_prompt: System prompt defining agent behavior
            temperature: Sampling temperature
            python_executable: Path to Python interpreter
            work_dir: Working directory for code execution
            verbose: Print detailed logs
        """
        self.model = model
        self.provider = provider.lower()
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.python_executable = python_executable
        
        # FIXED: Resolve path properly, avoid duplication
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        self.verbose = verbose
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Create tools
        self.tools = self._create_tools()
        
        # Create agent with tools - FIXED IMPORT
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools
        )
        
        # Memory: conversation state (persists across run() calls)
        self.state = {"messages": [SystemMessage(content=self.system_prompt)]}
        
        if self.verbose:
            print(f"✅ CodingAgent initialized:")
            print(f"   Model: {self.model}")
            print(f"   Provider: {self.provider}")
            print(f"   Work directory: {self.work_dir}")
            print(f"   Available tools: {[t.name for t in self.tools]}")
    
    def _initialize_llm(self):
        """Initialize the LLM based on provider."""
        if self.provider == "anthropic":
            if "ANTHROPIC_API_KEY" not in os.environ:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            return ChatAnthropic(
                model=self.model,
                temperature=self.temperature
            )
        
        elif self.provider == "openai":
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OPENAI_API_KEY not found in environment")
            return ChatOpenAI(
                model=self.model,
                temperature=self.temperature
            )
        
        elif self.provider == "google":
            if "GOOGLE_API_KEY" not in os.environ:
                raise ValueError("GOOGLE_API_KEY not found in environment")
            return ChatGoogleGenerativeAI(
                model=self.model,
                temperature=self.temperature
            )
        
        elif self.provider == "ollama":
            return ChatOllama(
                model=self.model,
                temperature=self.temperature
            )
        
        elif self.provider in ["xai", "grok"]:
            if "XAI_API_KEY" not in os.environ:
                raise ValueError("XAI_API_KEY not found in environment")
            return ChatXAI(
                model=self.model,
                temperature=self.temperature,
                timeout=60,
                max_retries=2
            )
        
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _create_tools(self):
        """Create tools for the agent."""

        def _run_code(code_file) -> str:
            """Shared execution logic."""
            if self.verbose:
                print(f"\n{'='*80}")
                print("🔧 EXECUTING CODE")
                print(f"{'='*80}")
                print(f"Code file: {code_file}")
                print(f"Work dir: {self.work_dir}\n")
            result = subprocess.run(
                [self.python_executable, str(code_file)],
                capture_output=True, text=True, timeout=120,
                cwd=str(self.work_dir)
            )
            success = result.returncode == 0
            def _tail(text, n=20):
                if not text: return text
                lines = text.splitlines()
                if len(lines) > n:
                    return f"[... {len(lines)-n} lines truncated ...]\n" + "\n".join(lines[-n:])
                return text
            response = f"Execution {'successful' if success else 'failed'}.\n\n"
            if result.stdout:
                response += f"STDOUT (last 20 lines):\n{_tail(result.stdout)}\n\n"
            if result.stderr:
                response += f"STDERR (last 20 lines):\n{_tail(result.stderr)}\n\n"
            response += "✅ Code executed successfully." if success else f"❌ Execution failed with return code {result.returncode}"
            if self.verbose:
                print(response)
            return response, success

        @tool
        def execute_draft(code: str) -> str:
            """
            Execute a SHORT, THROWAWAY diagnostic script (column inspection, quick checks, etc).
            Writes to draft.py only — does NOT touch script.py or main.py.
            Use this for one-off exploration snippets, NOT for the main pipeline.
            """
            try:
                draft_file = (self.work_dir / "draft.py").resolve()
                with open(draft_file, 'w') as f:
                    f.write(code)
                response, _ = _run_code(draft_file)
                return response
            except subprocess.TimeoutExpired:
                return "❌ Execution timeout (120 seconds exceeded)"
            except Exception as e:
                return f"❌ Execution error: {str(e)}"

        @tool
        def execute_main_pipeline(code: str) -> str:
            """
            Execute the MAIN pipeline script (full exploration or preprocessing pipeline).
            Writes to script.py AND a timestamped archive (script_YYYYMMDD_HHMMSS.py).
            On success ONLY, also saves to main.py — the canonical last-good pipeline.
            Use this for the full exploration/preprocessing code, NOT for quick diagnostics.
            """
            try:
                import shutil as _shutil
                code_file = (self.work_dir / "script.py").resolve()
                with open(code_file, 'w') as f:
                    f.write(code)
                response, success = _run_code(code_file)
                if success:
                    # Promote to main.py — the canonical last-good pipeline
                    _shutil.copy(code_file, self.work_dir / "main.py")
                return response
            except subprocess.TimeoutExpired:
                return "❌ Execution timeout (120 seconds exceeded)"
            except Exception as e:
                return f"❌ Execution error: {str(e)}"

        @tool
        def list_files_in_workspace() -> str:
            """
            List all files in the working directory.
            Useful to check what files were created during execution.

            Returns:
                List of files in the working directory with sizes
            """
            try:
                files = [f for f in self.work_dir.iterdir() if f.is_file()]
                if not files:
                    return "No files in working directory."
                file_list = "Files in working directory:\n"
                for f in files:
                    size = f.stat().st_size / 1024  # KB
                    file_list += f"  - {f.name} ({size:.1f} KB)\n"
                if self.verbose:
                    print(f"\n📁 {file_list}")
                return file_list
            except Exception as e:
                return f"Error listing files: {str(e)}"

        return [execute_draft, execute_main_pipeline, list_files_in_workspace]

    
    def run(self, task: str) -> Dict[str, any]:
        """
        Execute a coding task.
        
        The agent writes and executes code, then returns a text explanation.
        No requirements on specific output files - completely flexible.
        
        Args:
            task: Description of what to do (can include data paths, requirements, etc.)
        
        Returns:
            Dictionary with:
                - 'response': str (agent's text explanation)
                - 'messages': list (full conversation history)
        """
        if self.verbose:
            print(f"\n{'#'*80}")
            print(f"# CODING TASK")
            print(f"{'#'*80}\n")
            task_preview = task[:300] + "..." if len(task) > 300 else task
            print(f"Task (preview): {task_preview}\n")
        
        # Add user message to state (memory preserved!)
        self.state["messages"].append(HumanMessage(content=task))
        
        # Invoke agent (it will use tools as needed)
        result = self.agent.invoke(self.state)
        
        # Update state with agent's response (memory!)
        self.state = result
        
        # Extract agent's final response
        response_text = self._get_last_ai_message()
        
        if self.verbose:
            print(f"\n{'='*80}")
            print("AGENT RESPONSE")
            print(f"{'='*80}\n")
            print(response_text)
        
        return {
            'response': response_text,
            'messages': self.state["messages"].copy()
        }
    
    def _get_last_ai_message(self) -> str:
        """Extract the last AI message from state, handling different response formats."""
        for msg in reversed(self.state["messages"]):
            if isinstance(msg, AIMessage):
                return extract_message_content(msg.content)
        return ""
    
    def clear_history(self):
        """Clear conversation history (reset memory)."""
        self.state = {"messages": [SystemMessage(content=self.system_prompt)]}
        if self.verbose:
            print("🧹 Memory cleared")
    
    def get_history(self) -> List:
        """Get current conversation history."""
        return self.state["messages"].copy()
    
    def set_system_prompt(self, prompt: str):
        """Update system prompt."""
        self.system_prompt = prompt
        # Update first message in state
        if self.state["messages"] and isinstance(self.state["messages"][0], SystemMessage):
            self.state["messages"][0] = SystemMessage(content=prompt)
        else:
            self.state["messages"].insert(0, SystemMessage(content=prompt))
        
        if self.verbose:
            print("✅ System prompt updated")

# ============================================================================
# MULTIMODAL RAG AGENT CLASS
# ============================================================================
class aRAGAgent_multimodal:
    def __init__(self, 
                 db_path: str,
                 model: str,
                 embeddings_model: str = 'bge-m3:567m',
                 model_provider: str = 'ollama',
                 embeddings_provider: str = 'ollama',
                 collection_name='documents',
                 system_prompt="You are a powerful AI assistant...",
                 temperature=0, 
                 verbose=False,
                 k=5):
        
        # Store system prompt for later use
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.k = k
        
        # --- 1. Initialize Embeddings Model ---
        embeddings = None
        if embeddings_provider.lower() == 'ollama':
            embeddings = OllamaEmbeddings(model=embeddings_model)
            
        elif embeddings_provider.lower() == 'google':
            if "GOOGLE_API_KEY" not in os.environ:
                raise ValueError("GOOGLE_API_KEY not found in environment.")
            embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model)
            
        elif embeddings_provider.lower() == 'openai':
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OPENAI_API_KEY not found in environment.")
            embeddings = OpenAIEmbeddings(model=embeddings_model)
        else:
            raise ValueError(f"Unknown embeddings_provider: '{embeddings_provider}'.")

        
        # --- 2. Initialize ChromaDB ---
        client = chromadb.PersistentClient(path=db_path)
        vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings
        )
        
        # Store vector_store as instance variable
        self.vector_store = vector_store
        
        # --- 3. Retrieval Tool ---     
        @tool
        def retrieve(query: str) -> str:
            """Retrieve information related to a query from the local code database."""
            if self.verbose:
                print(f"--- Retrieving documents for query: '{query}' ---")
            retrieved_docs = self.vector_store.similarity_search(query, k=self.k)
            
            if not retrieved_docs:
                return "No relevant documents found in the database."

            serialized = "\n\n".join(
                (f"Source File: {doc.metadata.get('filename', 'N/A')}\n"
                 f"Lines: {doc.metadata.get('lines', 'N/A')}\n"
                 f"Content:\n---\n{doc.page_content}\n---")
                for doc in retrieved_docs
            )
            return serialized

        self.retrieve_tool = retrieve
        
        # --- 4. Initialize Main LLM ---
        llm = None
        provider = model_provider.lower()
        
        if provider == 'ollama':
            llm = ChatOllama(model=model, temperature=temperature)
            
        elif provider == 'google':
            
            if "GOOGLE_API_KEY" not in os.environ:
                raise ValueError("GOOGLE_API_KEY not found in environment.")
            llm = ChatGoogleGenerativeAI(model=model, temperature=temperature) 
            
        elif provider == 'openai':
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OPENAI_API_KEY not found in environment.")
            llm = ChatOpenAI(model=model, temperature=temperature)
            
        elif provider in ['claude', 'anthropic']:
            if "ANTHROPIC_API_KEY" not in os.environ:
                raise ValueError("ANTHROPIC_API_KEY not found in environment.")
            llm = ChatAnthropic(model=model, temperature=temperature)
            
        elif provider in ['grok', 'xai']:
            if "XAI_API_KEY" not in os.environ:
                raise ValueError("XAI_API_KEY not found in environment.")
            llm = ChatXAI(model=model, temperature=temperature, timeout=60, max_retries=2)
            
        else:
            raise ValueError(f"Unknown model_provider: '{model_provider}'.")

        # Store LLM as instance variable for direct multimodal calls
        self.llm = llm
        
        # --- 5. Create Agent for Text-Only Inputs ---
        self.agent = create_agent(
            model=llm,
            tools=[retrieve],
            system_prompt=system_prompt,
            debug=verbose
        )
        
        # Initialize state
        self.state = {"messages": []}

    def _is_multimodal_input(self, user_input: Union[str, list, Any]) -> bool:
        """
        Check if the input contains multimodal content (images).
        """
        if not isinstance(user_input, list):
            return False
        
        # Check if any item in the list is an image
        for item in user_input:
            if isinstance(item, dict) and item.get("type") == "image_url":
                return True
        return False

    def _build_message_history(self) -> list:
        """
        Build the full message history including system prompt.
        """
        messages = []
        
        # Add system message
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        
        # Add chat history from state
        messages.extend(self.state["messages"])
        
        return messages

    def run(self, user_input: Union[str, list, Any]) -> dict:
        """
        Runs a single turn of the conversation.
        
        Automatically detects input type:
        - Text-only: Uses agent with tool calling capability
        - Multimodal (with images): Calls LLM directly (bypasses agent)
        
        Examples:
        - Text: "Solve the Burgers equation"
        - Multimodal: [
            {"type": "text", "text": "Analyze this plot"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
          ]
        """
        
        is_multimodal = self._is_multimodal_input(user_input)
        
        if self.verbose:
            print(f"DEBUG - Input type: {type(user_input)}")
            print(f"DEBUG - Is multimodal: {is_multimodal}")
            if isinstance(user_input, list):
                print(f"DEBUG - Input list length: {len(user_input)}")
        
        if is_multimodal:
            # --- MULTIMODAL PATH: Direct LLM invocation ---
            if self.verbose:
                print(">>> Using MULTIMODAL mode (direct LLM call)")
            
            # Build message history
            messages = self._build_message_history()
            
            # Add current multimodal input as HumanMessage
            messages.append(HumanMessage(content=user_input))
            
            # Invoke LLM directly
            ai_response = self.llm.invoke(messages)

            if isinstance(ai_response.content, list):
                # Extract text from list of content parts
                ai_message = ""
                for part in ai_response.content:
                    if isinstance(part, dict) and 'text' in part:
                        ai_message += part['text']
                    elif isinstance(part, str):
                        ai_message += part
            else:
                ai_message = ai_response.content

            # Update state
            self.state["messages"].extend([
                HumanMessage(content=user_input),
                AIMessage(content=ai_message),
            ])
            
            return {
                "input": user_input,
                "output": ai_message,
                "mode": "multimodal"
            }
        
        else:
            # --- TEXT-ONLY PATH: Use agent with tools ---
            if self.verbose:
                print(">>> Using TEXT-ONLY mode (agent with tools)")
            
            # Add user message to state
            self.state["messages"].append(HumanMessage(content=user_input))
            
            # Invoke agent
            result = self.agent.invoke(self.state)
            
            # Update state
            self.state = result
            
            # Extract AI response
            ai_message = extract_message_content(result["messages"][-1].content)

            return {
                "input": user_input,
                "output": ai_message
            }
    
    def clear_history(self):
        """Clear the conversation history."""
        self.state = {"messages": []}
        if self.verbose:
            print("Chat history cleared.")
    
    def get_history(self) -> list:
        """Get the current conversation history."""
        return self.state["messages"].copy()


# ============================================================================
# SINGLE-CHUNK RAG AGENT CLASS (memory + RAG, text-only)
# ============================================================================
class aRAGAgent_single_chunk:
    """Text-only RAG agent that retrieves code chunks from a Chroma collection
    to ground its responses. Used as the operator-learning (family C) strategist,
    which retrieves module/architecture examples to seed its proposals.

    Migrated from the legacy create_tool_calling_agent/AgentExecutor API to the
    langgraph `create_agent` API, mirroring aRAGAgent_multimodal's text-only path.
    The retrieval tool returns code content only (no metadata) — the original
    "single chunk" behavior.
    """
    def __init__(self,
                 db_path: str,
                 model: str,
                 embeddings_model: str = 'bge-m3:567m',
                 model_provider: str = 'ollama',
                 embeddings_provider: str = 'ollama',
                 collection_name='documents',
                 system_prompt="You are a powerful AI assistant...",
                 temperature=0,
                 verbose=False,
                 k=5):

        # Store config for later use
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.k = k

        # --- 1. Initialize Embeddings Model ---
        embeddings = None
        if embeddings_provider.lower() == 'ollama':
            embeddings = OllamaEmbeddings(model=embeddings_model)
        elif embeddings_provider.lower() == 'google':
            if "GOOGLE_API_KEY" not in os.environ:
                raise ValueError("GOOGLE_API_KEY not found in environment.")
            embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model)
        elif embeddings_provider.lower() == 'openai':
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OPENAI_API_KEY not found in environment.")
            embeddings = OpenAIEmbeddings(model=embeddings_model)
        else:
            raise ValueError(f"Unknown embeddings_provider: '{embeddings_provider}'.")

        # --- 2. Initialize ChromaDB ---
        client = chromadb.PersistentClient(path=db_path)
        vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings
        )
        self.vector_store = vector_store

        # --- 3. Retrieval Tool (returns code content only) ---
        @tool
        def retrieve(query: str) -> str:
            """Retrieve implementation code from the local code database."""
            if self.verbose:
                print(f"--- Retrieving documents for query: '{query}' ---")
            retrieved_docs = self.vector_store.similarity_search(query, k=self.k)
            if not retrieved_docs:
                return "No relevant documents found in the database."
            code_blocks = [doc.page_content for doc in retrieved_docs]
            return "\n\n---\n\n".join(code_blocks)

        self.retrieve_tool = retrieve

        # --- 4. Initialize Main LLM ---
        llm = None
        provider = model_provider.lower()
        if provider == 'ollama':
            llm = ChatOllama(model=model, temperature=temperature)
        elif provider == 'google':
            if "GOOGLE_API_KEY" not in os.environ:
                raise ValueError("GOOGLE_API_KEY not found in environment.")
            llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
        elif provider == 'openai':
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OPENAI_API_KEY not found in environment.")
            llm = ChatOpenAI(model=model, temperature=temperature)
        elif provider in ['claude', 'anthropic']:
            if "ANTHROPIC_API_KEY" not in os.environ:
                raise ValueError("ANTHROPIC_API_KEY not found in environment.")
            llm = ChatAnthropic(model=model, temperature=temperature)
        elif provider in ['grok', 'xai']:
            if "XAI_API_KEY" not in os.environ:
                raise ValueError("XAI_API_KEY not found in environment.")
            llm = ChatXAI(model=model, temperature=temperature, timeout=60, max_retries=2)
        else:
            raise ValueError(f"Unknown model_provider: '{model_provider}'.")
        self.llm = llm

        # --- 5. Create Agent (text-only, tool-calling) ---
        self.agent = create_agent(
            model=llm,
            tools=[retrieve],
            system_prompt=system_prompt,
            debug=verbose
        )

        # Initialize state
        self.state = {"messages": []}

    def run(self, user_question: str) -> dict:
        """Runs a single text-only RAG turn."""
        self.state["messages"].append(HumanMessage(content=user_question))
        result = self.agent.invoke(self.state)
        self.state = result
        ai_message = extract_message_content(result["messages"][-1].content)
        return {"input": user_question, "output": ai_message}

    def clear_history(self):
        """Clear the conversation history."""
        self.state = {"messages": []}
        if self.verbose:
            print("Chat history cleared.")

    def get_history(self) -> list:
        """Get the current conversation history."""
        return self.state["messages"].copy()


# ============================================================================
# MULTIMODAL AGENT CLASS (memory + vision, no RAG)
# ============================================================================
class Multimodal_agent:
    def __init__(self,
                 model: str,
                 model_provider: str = 'ollama',
                 system_prompt="You are a powerful AI assistant...",
                 temperature=0,
                 verbose=False):

        # Store system prompt for later use
        self.system_prompt = system_prompt
        self.verbose = verbose

        # --- 1. Initialize Main LLM ---
        llm = None
        provider = model_provider.lower()

        if provider == 'ollama':
            llm = ChatOllama(model=model, temperature=temperature)

        elif provider == 'google':
            if "GOOGLE_API_KEY" not in os.environ:
                raise ValueError("GOOGLE_API_KEY not found in environment.")
            llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)

        elif provider == 'openai':
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OPENAI_API_KEY not found in environment.")
            llm = ChatOpenAI(model=model, temperature=temperature)

        elif provider in ['claude', 'anthropic']:
            if "ANTHROPIC_API_KEY" not in os.environ:
                raise ValueError("ANTHROPIC_API_KEY not found in environment.")
            llm = ChatAnthropic(model=model, temperature=temperature)

        elif provider in ['grok', 'xai']:
            if "XAI_API_KEY" not in os.environ:
                raise ValueError("XAI_API_KEY not found in environment.")
            llm = ChatXAI(model=model, temperature=temperature, timeout=60, max_retries=2)

        else:
            raise ValueError(f"Unknown model_provider: '{model_provider}'.")

        # Store LLM as instance variable for direct multimodal calls
        self.llm = llm

        # Initialize state
        self.state = {"messages": []}

    def _is_multimodal_input(self, user_input: Union[str, list, Any]) -> bool:
        """
        Check if the input contains multimodal content (images).
        """
        if not isinstance(user_input, list):
            return False

        # Check if any item in the list is an image
        for item in user_input:
            if isinstance(item, dict) and item.get("type") == "image_url":
                return True
        return False

    def _build_message_history(self) -> list:
        """
        Build the full message history including system prompt.
        """
        messages = []

        # Add system message
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))

        # Add chat history from state
        messages.extend(self.state["messages"])

        return messages

    def run(self, user_input: Union[str, list, Any]) -> dict:
        """
        Runs a single turn of the conversation.

        Automatically detects input type:
        - Text-only: Calls LLM directly with full message history
        - Multimodal (with images): Calls LLM directly (same path)

        Examples:
        - Text: "Explain this concept"
        - Multimodal: [
            {"type": "text", "text": "Analyze this plot"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
          ]
        """

        is_multimodal = self._is_multimodal_input(user_input)

        if self.verbose:
            print(f"DEBUG - Input type: {type(user_input)}")
            print(f"DEBUG - Is multimodal: {is_multimodal}")
            if isinstance(user_input, list):
                print(f"DEBUG - Input list length: {len(user_input)}")

        if is_multimodal:
            # --- MULTIMODAL PATH: Direct LLM invocation ---
            if self.verbose:
                print(">>> Using MULTIMODAL mode (direct LLM call)")
        else:
            # --- TEXT-ONLY PATH: Direct LLM invocation ---
            if self.verbose:
                print(">>> Using TEXT-ONLY mode (direct LLM call)")

        # Build message history
        messages = self._build_message_history()

        # Add current input as HumanMessage
        messages.append(HumanMessage(content=user_input))

        # Invoke LLM directly
        ai_response = self.llm.invoke(messages)

        if isinstance(ai_response.content, list):
            # Extract text from list of content parts
            ai_message = ""
            for part in ai_response.content:
                if isinstance(part, dict) and 'text' in part:
                    ai_message += part['text']
                elif isinstance(part, str):
                    ai_message += part
        else:
            ai_message = ai_response.content

        # Update state
        self.state["messages"].extend([
            HumanMessage(content=user_input),
            AIMessage(content=ai_message),
        ])

        return {
            "input": user_input,
            "output": ai_message,
            "mode": "multimodal" if is_multimodal else "text"
        }

    def clear_history(self):
        """Clear the conversation history."""
        self.state = {"messages": []}
        if self.verbose:
            print("Chat history cleared.")

    def get_history(self) -> list:
        """Get the current conversation history."""
        return self.state["messages"].copy()


# ============================================================================
# RETRIEVER AGENT CLASS
# ============================================================================
class Retriever_Agent:
    def __init__(self,
                 db_path: str,
                 collection_name: str,
                 model='gpt-oss:120b',
                 model_provider: str = 'ollama',  # Added for consistency
                 embeddings_model='bge-m3:567m',
                 embeddings_provider: str = 'ollama',  # Added for consistency
                 system_prompt="Your prompt goes here",
                 temperature=0,
                 verbose=False,
                 k=3):
        """
        Initializes the agent with a "smart" retrieval tool that reconstructs documents.
        """
        self.verbose = verbose
        self.k = k
        
        # --- Database Initialization ---
        client = chromadb.PersistentClient(path=db_path)
        
        # --- Embedding Model Selection ---
        embeddings_provider = embeddings_provider.lower()
        
        if embeddings_provider == 'google' or "google" in embeddings_model.lower():
            if "GOOGLE_API_KEY" not in os.environ:
                raise ValueError("GOOGLE_API_KEY not found in environment.")
            self.embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model)
        elif embeddings_provider == 'openai' or "openai" in embeddings_model.lower():
            if "OPENAI_API_KEY" not in os.environ:
                 raise ValueError("OPENAI_API_KEY not found in environment.")
            self.embeddings = OpenAIEmbeddings(model=embeddings_model)
        else:
            # Default to Ollama
            self.embeddings = OllamaEmbeddings(model=embeddings_model)
            
        # Store vector_store as self so the tool can access it
        self.vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
        
        # A place to store the retrieved documents
        self.last_retrieved_docs = []

        # --- ✨ Define the tool INSIDE __init__ ---
        @tool
        def retrieve(query: str) -> str:
            """
            Retrieve information related to a query. If a retrieved document is part of a
            larger file, this tool automatically fetches all parts and reconstructs the
            full document.
            
            This tool returns a CONCISE SUMMARY of the found documents, not their full content.
            """
            # It can access 'self' from the outer scope
            if self.verbose:
                print(f"--- Initial search for query: '{query}' ---")
            initial_docs = self.vector_store.similarity_search(query, k=self.k)
            
            if not initial_docs:
                self.last_retrieved_docs = []
                return "No relevant documents found in the database."

            final_docs = []
            processed_files = set()

            for doc in initial_docs:
                source_file = doc.metadata.get('source_file')
                if source_file and source_file not in processed_files:
                    processed_files.add(source_file)
                    if 'part' in doc.metadata:
                        if self.verbose:
                            print(f"  → Found chunk from '{doc.metadata['filename']}'. Reconstructing full file...")
                        
                        all_chunks = self.vector_store.get(
                            where={'source_file': source_file}, 
                            include=["metadatas", "documents"]
                        )
                        
                        if not all_chunks['ids']:
                            if self.verbose:
                                print(f"  → Warning: No chunks found for source_file '{source_file}'.")
                            continue
                        
                        sorted_chunks = sorted(
                            zip(all_chunks['documents'], all_chunks['metadatas']), 
                            key=lambda item: int(item[1].get('part', '1/1').split('/')[0])
                        )
                        full_content = "\n".join([chunk[0] for chunk in sorted_chunks])
                        
                        # Use the metadata from the first chunk as the base
                        merged_metadata = sorted_chunks[0][1].copy()
                        merged_metadata.pop('part', None)
                        merged_metadata.pop('cell_description', None)
                        
                        reconstructed_doc = Document(page_content=full_content, metadata=merged_metadata)
                        final_docs.append(reconstructed_doc)
                    else:
                        # This is a single-chunk document
                        if self.verbose:
                            print(f"  → Found single document: '{doc.metadata.get('filename', 'N/A')}'")
                        final_docs.append(doc)
            
            self.last_retrieved_docs = final_docs
            
            # Create a concise SUMMARY for the LLM
            summary_lines = []
            for doc in final_docs:
                meta = doc.metadata
                summary = (
                    f"Source File: {meta.get('filename', 'N/A')}\n"
                    f"  Equation: {meta.get('equation', 'Unknown')}\n"
                    f"  Optimizer: {meta.get('optimizer_scheme', 'Unknown')}\n"
                    f"  Problem: {meta.get('problem_type', 'Unknown')}\n"
                    f"  Type: {meta.get('type', 'Unknown')}\n"
                )
                summary_lines.append(summary)
            
            serialized_summary = "\n---\n".join(summary_lines)
            
            if self.verbose:
                print(f"--- Returning summary to LLM ---\n{serialized_summary}\n")
            
            return serialized_summary
        
        # --- LLM Selection ---
        provider = model_provider.lower()
        llm = None

        if provider == 'ollama':
            llm = ChatOllama(model=model, temperature=temperature)
            
        elif provider == 'google':
            
            if "GOOGLE_API_KEY" not in os.environ:
                raise ValueError("GOOGLE_API_KEY not found in environment.")
            llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
            
        elif provider == 'openai':
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OPENAI_API_KEY not found in environment.")
            llm = ChatOpenAI(model=model, temperature=temperature)

        elif provider in ['claude', 'anthropic']:
            if "ANTHROPIC_API_KEY" not in os.environ:
                raise ValueError("ANTHROPIC_API_KEY not found in environment.")
            llm = ChatAnthropic(model=model, temperature=temperature)

        elif provider in ['grok', 'xai']:
            if "XAI_API_KEY" not in os.environ:
                raise ValueError("XAI_API_KEY not found in environment.")
            llm = ChatXAI(model=model, temperature=temperature, timeout=60, max_retries=2)

        else:
            raise ValueError(f"Unknown model_provider: '{model_provider}'.")
        
        # --- Modern Agent Setup ---
        self.agent = create_agent(
            model=llm,
            tools=[retrieve],
            system_prompt=system_prompt,
            debug=verbose
        )
        
        # Initialize state
        self.state = {"messages": []}

    def _extract_message_content(self, message) -> str:
        """Extract plain text from message content (handles new agent format)."""
        if hasattr(message, 'content'):
            content = message.content
            
            # Handle list of content blocks
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        text_parts.append(block.get('text', ''))
                    elif isinstance(block, str):
                        text_parts.append(block)
                return '\n'.join(text_parts) if text_parts else str(content)
            
            # Handle string content
            elif isinstance(content, str):
                return content
            
            # Fallback
            return str(content)
        
        return str(message)

    def run(self, user_question: str) -> dict:
        """
        Runs a single turn and appends the original retrieved code to the result.
        """
        # Clear last retrieved docs before new run
        self.last_retrieved_docs = [] 
        
        # Add user message to state
        self.state["messages"].append(HumanMessage(content=user_question))
        
        # Invoke agent
        result = self.agent.invoke(self.state)
        
        # Update state
        self.state = result
        
        # Extract AI response using robust helper
        ai_message = self._extract_message_content(result["messages"][-1])
        
        if self.verbose:
            print(f"AI Response: {ai_message}")
        
        # Return in compatible format with added retrieved docs
        return {
            "input": user_question,
            "output": ai_message,
            "original_code": self.last_retrieved_docs
        }

    def clear_history(self):
        """Clears the agent's chat history."""
        self.state = {"messages": []}
        if self.verbose:
            print("Chat history cleared.")

    def get_history(self) -> list:
        """Gets the agent's current chat history."""
        return self.state["messages"].copy()


# ============================================================================
# MEMORY AGENT CLASS
# ============================================================================
class memoryAgent:
    def __init__(self,
                 model: str,
                 model_provider: str = 'ollama',
                 system_prompt: str = "You are a powerful AI assistant...", 
                 temperature: float = 0, 
                 verbose: bool = False):
        """
        Initializes the agent with modern LangChain support.
        Supports: 'ollama', 'google', 'openai', 'claude', 'anthropic', 'grok', 'xai'.
        """
        
        self.verbose = verbose
        self.system_prompt = system_prompt
        llm = None 
        
        # Normalize provider string
        provider = model_provider.lower()

        # --- 1. OLLAMA ---
        if provider == 'ollama':
            llm = ChatOllama(model=model, temperature=temperature)
            
        # --- 2. GOOGLE ---
        elif provider == 'google':
            
            if "GOOGLE_API_KEY" not in os.environ:
                raise ValueError("GOOGLE_API_KEY not found in environment.")
            llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
            
        # --- 3. OPENAI (Standard) ---
        elif provider == 'openai':
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OPENAI_API_KEY not found in environment.")
            llm = ChatOpenAI(model=model, temperature=temperature)

        # --- 4. CLAUDE (Anthropic) ---
        elif provider in ['claude', 'anthropic']:
            if "ANTHROPIC_API_KEY" not in os.environ:
                raise ValueError("ANTHROPIC_API_KEY not found in environment.")
            llm = ChatAnthropic(model=model, temperature=temperature)

        # --- 5. GROK (xAI) - NATIVE IMPLEMENTATION! ---
        elif provider in ['grok', 'xai']:
            if "XAI_API_KEY" not in os.environ:
                raise ValueError("XAI_API_KEY not found in environment.")
            
            # Use native ChatXAI - much faster!
            llm = ChatXAI(
                model=model,
                temperature=temperature,
                timeout=60,
                max_retries=2
            )

        else:
            raise ValueError(
                f"Unknown model_provider: '{model_provider}'. "
                "Please use 'ollama', 'google', 'openai', 'claude', or 'grok'."
            )

        # --- Modern Agent Setup ---
        tools = []  # Add tools here if needed in the future
        
        # Create the agent using the modern API
        self.agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            debug=verbose
        )
        
        # Initialize state (replaces chat_history)
        self.state = {"messages": []}

    def run(self, user_question: str) -> dict:
        """
        Run a conversation turn.
        
        Returns dict with same structure as before: {"input": ..., "output": ...}
        """
        # Add user message to state
        self.state["messages"].append(HumanMessage(content=user_question))
        
        # Invoke agent
        result = self.agent.invoke(self.state)
        
        # Update state with full result
        self.state = result
        
        # Extract AI response (last message in the result)
        ai_message = extract_message_content(result["messages"][-1].content)
        
        if self.verbose:
            print(f"AI Response: {ai_message}")
        
        # Return in same format as before (backward compatible)
        return {
            "input": user_question,
            "output": ai_message
        }

    def clear_history(self):
        """Clears the agent's chat history."""
        self.state = {"messages": []}
        if self.verbose:
            print("Chat history cleared.")

    def get_history(self) -> list:
        """Get the current conversation history."""
        return self.state["messages"].copy()
# ============================================================================
# STRUCTURED AGENT CLASS
# ============================================================================
# Backward compatibility alias.
# Historically some code imported `BasicAgent` from this module.
# Keep that import path working by mapping it to the lightweight conversational agent.
BasicAgent = memoryAgent


class StructuredAgent:
    def __init__(self,
                 model: str, 
                 model_provider: str = 'ollama',
                 system_prompt: str = "You are a helpful assistant. Return output in JSON format.", 
                 temperature: float = 0, 
                 verbose: bool = False):
        
        self.verbose = verbose
        llm = None 
        
        # Normalize provider string
        provider = model_provider.lower()

        # --- 1. OLLAMA ---
        if provider == 'ollama':
            llm = ChatOllama(model=model, temperature=temperature)
            
        # --- 2. GOOGLE ---
        elif provider == 'google':
            
            if "GOOGLE_API_KEY" not in os.environ:
                raise ValueError("GOOGLE_API_KEY not found in environment.")
            llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
            
        # --- 3. OPENAI (Standard) ---
        elif provider == 'openai':
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OPENAI_API_KEY not found in environment.")
            llm = ChatOpenAI(model=model, temperature=temperature)

        # --- 4. CLAUDE (Anthropic) ---
        elif provider in ['claude', 'anthropic']:
            if "ANTHROPIC_API_KEY" not in os.environ:
                raise ValueError("ANTHROPIC_API_KEY not found in environment.")
            llm = ChatAnthropic(model=model, temperature=temperature)

        # --- 5. GROK (xAI) - NATIVE IMPLEMENTATION ---
        elif provider in ['grok', 'xai']:
            if "XAI_API_KEY" not in os.environ:
                raise ValueError("XAI_API_KEY not found in environment.")
            
            llm = ChatXAI(
                model=model,
                temperature=temperature,
                timeout=60,
                max_retries=2
            )

        else:
            raise ValueError(
                f"Unknown model_provider: '{model_provider}'. "
                "Please use 'ollama', 'google', 'openai', 'claude', or 'grok'."
            )

        # Store LLM for fallback parsing
        self.llm = llm
        
        # --- Chain Setup (LCEL - still works great for structured output!) ---
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"), 
        ])
        
        parser = JsonOutputParser()        
        self.chain = self.prompt | llm | parser

    def run(self, user_question: str, chat_history: List[BaseMessage] = None) -> Dict:
        """
        Stateless run with retry and fallback JSON parsing.
        
        Note: This agent uses LCEL chains, not the agent framework, because
        structured output is better handled with direct chain invocation.
        """
        
        history = chat_history or [] 
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                result = self.chain.invoke({
                    "input": user_question,
                    "chat_history": history 
                })
                
                if self.verbose:
                    print(f"Structured output: {result}")
                
                return result
                
            except OutputParserException as e:
                if self.verbose:
                    print(f"JSON parsing failed on attempt {attempt + 1}")
                
                if attempt == max_retries - 1:
                    # Fallback: get raw output and try manual JSON parsing
                    try:
                        if self.verbose:
                            print("Attempting fallback JSON parsing...")
                        
                        raw_output = (self.prompt | self.llm).invoke({
                            "input": user_question,
                            "chat_history": history
                        })
                        # Use the tolerant extractor: strips markdown fences and
                        # escapes stray backslashes (e.g. LaTeX like `t\in[0,3]`,
                        # which is invalid JSON and breaks bare json.loads).
                        result = extract_json_from_text(raw_output.content)
                        
                        if self.verbose:
                            print(f"Fallback parsing successful: {result}")
                        
                        return result
                    except Exception as fallback_error:
                        if self.verbose:
                            print(f"Fallback parsing also failed: {fallback_error}")
                        raise e  
                continue
            
            except Exception as e:
                if self.verbose:
                    print(f"Unexpected error: {e}")
                raise e


class Struct_memoryAgent:
    def __init__(self,
                 model: str,
                 model_provider: str = 'ollama',
                 system_prompt: str = "You are a powerful AI assistant...", 
                 temperature: float = 0, 
                 verbose: bool = False):
        """
        Initializes the agent with modern LangChain support.
        Supports: 'ollama', 'google', 'openai', 'claude', 'anthropic', 'grok', 'xai'.
        """
        
        self.verbose = verbose
        self.system_prompt = system_prompt
        llm = None 
        
        # Normalize provider string
        provider = model_provider.lower()

        # --- 1. OLLAMA ---
        if provider == 'ollama':
            llm = ChatOllama(model=model, temperature=temperature)
            
        # --- 2. GOOGLE ---
        elif provider == 'google':
            
            if "GOOGLE_API_KEY" not in os.environ:
                raise ValueError("GOOGLE_API_KEY not found in environment.")
            llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
            
        # --- 3. OPENAI (Standard) ---
        elif provider == 'openai':
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OPENAI_API_KEY not found in environment.")
            llm = ChatOpenAI(model=model, temperature=temperature)

        # --- 4. CLAUDE (Anthropic) ---
        elif provider in ['claude', 'anthropic']:
            if "ANTHROPIC_API_KEY" not in os.environ:
                raise ValueError("ANTHROPIC_API_KEY not found in environment.")
            llm = ChatAnthropic(model=model, temperature=temperature)

        # --- 5. GROK (xAI) - NATIVE IMPLEMENTATION! ---
        elif provider in ['grok', 'xai']:
            if "XAI_API_KEY" not in os.environ:
                raise ValueError("XAI_API_KEY not found in environment.")
            
            # Use native ChatXAI - much faster!
            llm = ChatXAI(
                model=model,
                temperature=temperature,
                timeout=60,
                max_retries=2
            )

        else:
            raise ValueError(
                f"Unknown model_provider: '{model_provider}'. "
                "Please use 'ollama', 'google', 'openai', 'claude', or 'grok'."
            )

        # --- Modern Agent Setup ---
        tools = []  # Add tools here if needed in the future
        
        # Create the agent using the modern API
        self.agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            debug=verbose
        )
        
        # Initialize state (replaces chat_history)
        self.state = {"messages": []}

    def run(self, user_question: str, max_retries: int = 3) -> dict:
        """
        Run a conversation turn with built-in safeguards:
          1. Retries the agent call up to `max_retries` times on failure.
          2. Automatically strips markdown fences and parses JSON from the response.
          3. Falls back to returning the raw string if JSON parsing is not possible.
        """
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                # Add user message (only on first attempt; state already has it on retry)
                if attempt == 1:
                    self.state["messages"].append(HumanMessage(content=user_question))

                # Invoke agent
                result = self.agent.invoke(self.state)

                # Update state with full result
                self.state = result

                # Extract AI response (last message in the result)
                ai_message = extract_message_content(result["messages"][-1].content)

                if self.verbose:
                    print(f"[Struct_memoryAgent] Response (attempt {attempt}): {ai_message[:200]}")

                # --- JSON extraction safeguard ---
                # Try to parse JSON (handles ```json ... ``` fences automatically)
                # If it fails, raise so the retry loop asks the agent again
                try:
                    parsed_output = extract_json_from_text(ai_message)
                    return parsed_output  # Return dict directly, same as StructuredAgent
                except Exception as json_err:
                    raise ValueError(
                        f"Agent response is not valid JSON (attempt {attempt}/{max_retries}). "
                        f"Error: {json_err}\nFull response:\n{ai_message}"
                    ) from json_err

            except Exception as e:
                last_error = e
                print(f"  ⚠️  [Struct_memoryAgent] Failed on attempt {attempt}/{max_retries}: {e}")
                if attempt < max_retries:
                    print(f"  ↺  Retrying (clearing history to avoid context overflow)...")
                    # Clear full history and re-add only the user question
                    # This prevents context from growing and truncating on retry
                    self.state = {"messages": []}
                    self.state["messages"].append(HumanMessage(content=user_question))
                else:
                    raise RuntimeError(
                        f"[Struct_memoryAgent] All {max_retries} attempts failed. "
                        f"Last error: {last_error}"
                    ) from last_error

    def clear_history(self):
        """Clears the agent's chat history."""
        self.state = {"messages": []}
        if self.verbose:
            print("Chat history cleared.")

    def get_history(self) -> list:
        """Get the current conversation history."""
        return self.state["messages"].copy()


## Auxiliary Functions
def extract_json_from_text(text: str) -> dict:
    """
    Robustly extracts a JSON object from a string, handling:
    - Markdown code fences (```json ... ``` or ``` ... ```)
    - Plain raw JSON strings
    Raises ValueError if no valid JSON dict can be parsed.
    """
    import json
    import re
    # 1. Try to strip a markdown code fence first
    fence_match = re.search(r'```(?:json)?\s*({.*})\s*```', text, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        # 2. Fall back to finding the first { ... } block in plain text
        brace_match = re.search(r'{.*}', text, re.DOTALL)
        if brace_match:
            cleaned = brace_match.group(0)
        else:
            raise ValueError(f"No JSON object found in text:\n{text}")
    # Unescape double-braces that some models emit (Python f-string style: {{ -> {)
    cleaned = cleaned.replace('{{', '{').replace('}}', '}')
    cleaned = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', cleaned)
    return json.loads(cleaned)

# CLI Agents
class gemini_cli_agent:
    def __init__(self, model='gemini-2.5-pro', approval_mode='yolo'):
        self.model = model
        self.approval_mode = approval_mode
        self.system_prefix = (
            "1) Explain in plain text your plan. \n"
            "2) As you make new modifications explain your reasons.\n"
            "3) If you are required to use an SKILL explicity mention that you found the SKILL.\n"
        )

    def run(self, prompt, project_dir):
        full_prompt = self.system_prefix + prompt
        shell_cmd = (
            f'cd {project_dir} && '
            f'gemini --model {self.model} '
            f'--approval-mode {self.approval_mode} '
            f'-p {repr(full_prompt)}'
        )
        process = subprocess.Popen(
            shell_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        output_lines = []
        for line in process.stdout:
            print(line, end='', flush=True)
            output_lines.append(line)
        stderr = process.stderr.read()
        process.wait()
        answer = ''.join(output_lines)
        if process.returncode != 0:
            print(f"[gemini_cli_agent] Error (code {process.returncode}):\n{stderr}")
        return {
            'output': answer,
            'error': stderr,
            'returncode': process.returncode,
        }