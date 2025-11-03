import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage,SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser # Import the JSON parser
from langchain_core.messages import BaseMessage # Needed for type hinting
from typing import Dict, List ,Union, Any
from langchain_core.documents import Document
import chromadb
from pathlib import Path # Import the Path object
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# Get the directory of the *current file* (e.g., /.../crunch_llm/code/)
current_file_dir = Path(__file__).parent 
# Build the full path to the .env file in that *same directory*
dotenv_path = current_file_dir / '.env'
# Explicitly load that file
load_dotenv(dotenv_path=dotenv_path)

class Retriever_Agent:
    def __init__(self,
                 db_path: str,
                 collection_name: str,
                 model='gpt-oss:120b',
                 embeddings_model='bge-m3:567m',
                 system_prompt="Your prompt goes here",
                 temperature=0,
                 verbose=False,
                 k=3):
        """
        Initializes the agent with a "smart" retrieval tool that reconstructs documents.
        """
        # --- Database Initialization ---
        client = chromadb.PersistentClient(path=db_path)
        
        # --- Embedding Model Selection (from aRAGAgent) ---
        # Store embeddings on self to be safe, though not strictly needed
        if "google" in embeddings_model.lower():
            if "GOOGLE_API_KEY" not in os.environ:
                raise ValueError("GOOGLE_API_KEY not found in .env file.")
            self.embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model)
        elif "openai" in embeddings_model.lower():
            if "OPENAI_API_KEY" not in os.environ:
                 raise ValueError("OPENAI_API_KEY not found in .env file.")
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

        # --- ✨ MODIFICATION: Define the tool INSIDE __init__ ---
        @tool
        def retrieve(query: str) -> str: # No 'self' argument
            """
            Retrieve information related to a query. If a retrieved document is part of a
            larger file, this tool automatically fetches all parts and reconstructs the
            full document.
            
            This tool returns a CONCISE SUMMARY of the found documents, not their full content.
            """
            # It can access 'self' from the outer scope
            if verbose:
                print(f"--- Initial search for query: '{query}' ---")
            initial_docs = self.vector_store.similarity_search(query, k=k)
            
            if not initial_docs:
                self.last_retrieved_docs = [] # Accesses self
                return "No relevant documents found in the database."

            final_docs = []
            processed_files = set()

            for doc in initial_docs:
                source_file = doc.metadata.get('source_file')
                if source_file and source_file not in processed_files:
                    processed_files.add(source_file)
                    if 'part' in doc.metadata:
                        if verbose:
                            print(f"  → Found chunk from '{doc.metadata['filename']}'. Reconstructing full file...")
                        # Accesses self.vector_store
                        all_chunks = self.vector_store.get(where={'source_file': source_file}, include=["metadatas", "documents"])
                        
                        if not all_chunks['ids']:
                             if verbose:
                                 print(f"  → Warning: No chunks found for source_file '{source_file}' despite initial hit.")
                             continue
                        
                        sorted_chunks = sorted(zip(all_chunks['documents'], all_chunks['metadatas']), key=lambda item: int(item[1].get('part', '1/1').split('/')[0]))
                        full_content = "\n".join([chunk[0] for chunk in sorted_chunks])
                        
                        # Use the metadata from the first chunk as the base
                        merged_metadata = sorted_chunks[0][1].copy()
                        merged_metadata.pop('part', None)
                        merged_metadata.pop('cell_description', None) # Remove cell-specific info
                        
                        reconstructed_doc = Document(page_content=full_content, metadata=merged_metadata)
                        final_docs.append(reconstructed_doc)
                    else:
                        # This is a single-chunk document (e.g., manual page)
                        if verbose:
                            print(f"  → Found single document: '{doc.metadata.get('filename', 'N/A')}'")
                        final_docs.append(doc)
            
            self.last_retrieved_docs = final_docs # Accesses self
            
            # --- MODIFICATION ---
            # Create a concise SUMMARY for the LLM, not the full content.
            # The LLM only needs this to *identify* the file.
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
            
            if verbose:
                print(f"--- Returning summary to LLM ---\n{serialized_summary}\n")
            
            # Return the summary, not the full content
            return serialized_summary
        
        # --- The rest of the agent setup ---
        llm = ChatOllama(model=model, temperature=temperature)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Pass the newly defined nested function 'retrieve'
        tools = [retrieve] 
        agent = create_tool_calling_agent(llm, tools, prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=verbose)
        self.chat_history = []

    # The 'retrieve' method is no longer here

    def run(self, user_question: str) -> dict:
        """
        Runs a single turn and appends the original retrieved code to the result.
        """
        # Clear last retrieved docs before new run
        self.last_retrieved_docs = [] 
        
        result = self.agent_executor.invoke({
            "input": user_question,
            "chat_history": self.chat_history
        })

        ai_message = result["output"]
        self.chat_history.extend([
            HumanMessage(content=user_question),
            AIMessage(content=ai_message),
        ])
        
        # Attach the FULL, RECONSTRUCTED code for the next agent to use
        result['original_code'] = self.last_retrieved_docs
        
        return result


class aRAGAgent:
    def __init__(self, 
                 db_path: str,
                 model: str,
                 embeddings_model: str ='bge-m3:567m',
                 model_provider: str = 'ollama',
                 embeddings_provider: str = 'ollama',
                 collection_name='documents',
                 system_prompt="You are a powerful AI assistant...",
                 temperature=0, 
                 verbose=False,
                 k=5):
        """
        Initializes a generalized RAG agent that can use ollama, google, or openai
        for both its main LLM and its embeddings model.
        """
        
        # --- 1. Initialize Embeddings Model (Generalized) ---
        embeddings = None
        if embeddings_provider.lower() == 'ollama':
            embeddings = OllamaEmbeddings(model=embeddings_model)
            
        elif embeddings_provider.lower() == 'google':
            if "GOOGLE_API_KEY" not in os.environ:
                raise ValueError("GOOGLE_API_KEY not found in .env file.")
            embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model)
            
        elif embeddings_provider.lower() == 'openai':
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OPENAI_API_KEY not found in .env file.")
            embeddings = OpenAIEmbeddings(model=embeddings_model)
        else:
            raise ValueError(f"Unknown embeddings_provider: '{embeddings_provider}'.")

        
        # --- 2. Initialize ChromaDB (Unchanged) ---
        client = chromadb.PersistentClient(path=db_path)
        vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings  # Now uses the generalized embeddings
        )
        
        # --- 3. Retrieval Tool (Unchanged) ---
        # The tool is defined inside __init__ so it can access 'vector_store'
        @tool
        def retrieve(query: str) -> str:
            """Retrieve information related to a query from the local code database."""
            if vector_store:
                print(f"--- Retrieving documents for query: '{query}' ---")
            retrieved_docs = vector_store.similarity_search(query, k=k)
            
            if not retrieved_docs:
                return "No relevant documents found in the database."

            serialized = "\n\n".join(
                (f"Source File: {doc.metadata.get('filename', 'N/A')}\n"
                 f"Lines: {doc.metadata.get('lines', 'N/A')}\n"
                 f"Content:\n---\n{doc.page_content}\n---")
                for doc in retrieved_docs
            )
            return serialized

        # --- 4. Initialize Main LLM (Generalized) ---
        llm = None
        if model_provider.lower() == 'ollama':
            llm = ChatOllama(model=model, temperature=temperature)
            
        elif model_provider.lower() == 'google':
            if "GOOGLE_API_KEY" not in os.environ:
                raise ValueError("GOOGLE_API_KEY not found in .env file.")
            llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
            
        elif model_provider.lower() == 'openai':
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OPENAI_API_KEY not found in .env file.")
            llm = ChatOpenAI(model=model, temperature=temperature)
        else:
            raise ValueError(f"Unknown model_provider: '{model_provider}'.")

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        tools = [retrieve]
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=verbose)
        self.chat_history = []

    def run(self, user_question: str) -> dict:
        """
        Runs a single turn of the conversation.
        """
        result = self.agent_executor.invoke({
            "input": user_question,
            "chat_history": self.chat_history
        })

        ai_message = result["output"]

        self.chat_history.extend([
            HumanMessage(content=user_question),
            AIMessage(content=ai_message),
        ])
        return result
    def clear_history(self):
        """Clears the agent's chat history."""
        self.chat_history = []
        # Access the verbose flag from the agent_executor
        if self.agent_executor.verbose:
            print("Chat history cleared.")

    def get_history(self) -> list:
        """Gets the agent's current chat history."""
        return self.chat_history


class aRAGAgent_multimodal:
    def __init__(self, 
                 db_path: str,
                 model: str,
                 embeddings_model: str ='bge-m3:567m',
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
        
        # --- 1. Initialize Embeddings Model (Generalized) ---
        embeddings = None
        if embeddings_provider.lower() == 'ollama':
            from langchain_community.embeddings import OllamaEmbeddings
            embeddings = OllamaEmbeddings(model=embeddings_model)
            
        elif embeddings_provider.lower() == 'google':
            if "GOOGLE_API_KEY" not in os.environ:
                raise ValueError("GOOGLE_API_KEY not found in .env file.")
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embeddings = GoogleGenerativeAIEmbeddings(model=embeddings_model)
            
        elif embeddings_provider.lower() == 'openai':
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OPENAI_API_KEY not found in .env file.")
            from langchain_openai import OpenAIEmbeddings
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
        
        # Store vector_store as instance variable for potential direct access
        self.vector_store = vector_store
        self.k = k
        
        # --- 3. Retrieval Tool ---     
        @tool
        def retrieve(query: str) -> str:
            """Retrieve information related to a query from the local code database."""
            if verbose:
                print(f"--- Retrieving documents for query: '{query}' ---")
            retrieved_docs = vector_store.similarity_search(query, k=k)
            
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
        
        # --- 4. Initialize Main LLM (Generalized) ---
        llm = None
        if model_provider.lower() == 'ollama':
            from langchain_community.chat_models import ChatOllama
            llm = ChatOllama(model=model, temperature=temperature)
            
        elif model_provider.lower() == 'google':
            if "GOOGLE_API_KEY" not in os.environ:
                raise ValueError("GOOGLE_API_KEY not found in .env file.")
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model=model, temperature=temperature) 
            
        elif model_provider.lower() == 'openai':
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OPENAI_API_KEY not found in .env file.")
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=model, temperature=temperature)
        else:
            raise ValueError(f"Unknown model_provider: '{model_provider}'.")

        # Store LLM as instance variable for direct multimodal calls
        self.llm = llm
        
        # --- 5. Create Agent for Text-Only Inputs ---
        from langchain.agents import create_tool_calling_agent, AgentExecutor
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        tools = [retrieve]
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=verbose)
        self.chat_history = []

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
        
        # Add chat history
        messages.extend(self.chat_history)
        
        return messages

    def run(self, user_input: Union[str, list, Any]) -> dict:
        """
        Runs a single turn of the conversation.
        
        Automatically detects input type:
        - Text-only: Uses agent executor with tool calling capability
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
            ai_message = ai_response.content
            
            # Update chat history
            self.chat_history.extend([
                HumanMessage(content=user_input),
                AIMessage(content=ai_message),
            ])
            
            return {
                "input": user_input,
                "output": ai_message,
                "mode": "multimodal"
            }
        
        else:
            if self.verbose:
                print(">>> Using TEXT-ONLY mode (agent with tools)")
            
            result = self.agent_executor.invoke({
                "input": user_input,
                "chat_history": self.chat_history
            })

            ai_message = result["output"]

            self.chat_history.extend([
                HumanMessage(content=user_input), 
                AIMessage(content=ai_message),
            ])

            return result
    
    def clear_history(self):
        """Clear the conversation history."""
        self.chat_history = []
        if self.verbose:
            print("Chat history cleared.")
    
    def get_history(self) -> list:
        """Get the current conversation history."""
        return self.chat_history.copy()


class memoryAgent:
    def __init__(self,
                  model='gpt-oss:120b', 
                 model_provider: str = 'ollama',
                 system_prompt: str = "You are a powerful AI assistant...", 
                 temperature: float = 0, 
                 verbose: bool = False):
        """
        Initializes the agent, its tools, and its memory.
        Can use 'ollama', 'google', or 'openai' providers.
        """

        llm = None # Initialize llm
        
        # --- NEW: Select LLM based on provider ---
        if model_provider.lower() == 'ollama':
            # This is the default, model='gpt-oss:20b'
            llm = ChatOllama(model=model, temperature=temperature)
            
        elif model_provider.lower() == 'google':
            if "GOOGLE_API_KEY" not in os.environ:
                raise ValueError("GOOGLE_API_KEY not found in .env file.")
            llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
            
        elif model_provider.lower() == 'openai':
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OPENAI_API_KEY not found in .env file.")
            llm = ChatOpenAI(model=model, temperature=temperature)
        else:
            raise ValueError(
                f"Unknown model_provider: '{model_provider}'. "
                "Please use 'ollama', 'google', or 'openai'."
            )
        # --- END NEW ---

        # The rest of the class is exactly the same
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        tools = []
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=verbose)
        self.chat_history = []

    def run(self, user_question: str) -> dict: # Changed return type hint to dict
        """
        Runs a single turn of the conversation.
        """
        result = self.agent_executor.invoke({
            "input": user_question,
            "chat_history": self.chat_history
        })

        ai_message = result["output"]

        # Update the internal chat history
        self.chat_history.append(HumanMessage(content=user_question))
        self.chat_history.append(AIMessage(content=ai_message))

        return result
class BasicAgent:
    def __init__(self,
                 model='gpt-oss:20b', 
                 model_provider: str = 'ollama', 
                 system_prompt="You are a powerful AI assistant...", 
                 temperature=0, 
                 verbose=False):

        llm = None # Initialize llm        
        if model_provider.lower() == 'ollama':
            llm = ChatOllama(model=model, temperature=temperature)
            
        elif model_provider.lower() == 'google':
            if "GOOGLE_API_KEY" not in os.environ:
                raise ValueError("GOOGLE_API_KEY not found in .env file.")
            llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
            
        elif model_provider.lower() == 'openai':
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OPENAI_API_KEY not found in .env file.")
            llm = ChatOpenAI(model=model, temperature=temperature)
        else:
            raise ValueError(
                f"Unknown model_provider: '{model_provider}'. "
                "Please use 'ollama', 'google', or 'openai'."
            )
        # --- END NEW ---

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        tools = []
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=verbose)

    def run(self, user_question: str) -> Dict: # <-- MODIFIED TYPE HINT
        result = self.agent_executor.invoke({
            "input": user_question,
        })
        return result

class BasicStructuredAgent:
    def __init__(self,
                 model='gpt-oss:120b', 
                 system_prompt="Your prompt goes here", 
                 temperature=0, 
                 verbose=False):
        
        # Create the standard LLM (no structured output wrapper)
        llm = ChatOllama(model=model, temperature=temperature)

        # Create the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}"),
        ])
        
        # Create the JSON parser
        parser = JsonOutputParser()

        # Create the chain: prompt -> llm -> parser
        self.chain = prompt | llm | parser

    def run(self, user_question: str) -> Dict:
        """
        Runs a single turn and returns a parsed dictionary.
        """
        result = self.chain.invoke({ "input": user_question })
        return result


class StructuredAgent:
    def __init__(self,
                 model: str, 
                 model_provider: str = 'ollama', # Now accepts 'ollama', 'google', or 'openai'
                 system_prompt: str = "Your prompt goes here", 
                 temperature: float = 0, 
                 verbose: bool = False):
        
        llm = None # Initialize llm
        
        # --- Select LLM based on provider ---
        if model_provider.lower() == 'ollama':
            # REVERTED: Rely *only* on the system prompt for JSON.
            llm = ChatOllama(model=model, temperature=temperature)
            
        elif model_provider.lower() == 'google':
            if "GOOGLE_API_KEY" not in os.environ:
                raise ValueError("GOOGLE_API_KEY not found in .env file.")
            
            # REVERTED: Removed the 'response_format' argument
            llm = ChatGoogleGenerativeAI(
                model=model, 
                temperature=temperature
            )
            
        elif model_provider.lower() == 'openai':
            if "OPENAI_API_KEY" not in os.environ:
                raise ValueError("OPENAI_API_KEY not found in .env file.")
                
            # REVERTED: Removed the 'model_kwargs' argument
            llm = ChatOpenAI(
                model=model, 
                temperature=temperature
            )
        else:
            raise ValueError(
                f"Unknown model_provider: '{model_provider}'. "
                "Please use 'ollama', 'google', or 'openai'."
            )

        # The rest of the class is identical
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"), 
        ])
        
        parser = JsonOutputParser()
        self.chain = self.prompt | llm | parser

    def run(self, user_question: str, chat_history: List[BaseMessage] = None) -> Dict:
        """
        Runs a single turn, considering chat history, and returns a parsed dictionary.
        """
        history = chat_history or [] 
        
        result = self.chain.invoke({
            "input": user_question,
            "chat_history": history 
        })
        return result
