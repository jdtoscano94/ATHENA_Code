import os
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser # Import the JSON parser
from langchain_core.messages import BaseMessage # Needed for type hinting
from langchain_core.documents import Document
from typing import Dict, List # Needed for type hintingfrom langchain_core.documents import Document
import chromadb


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
        embeddings = OllamaEmbeddings(model=embeddings_model)
        # Store vector_store as self so the tool can access it
        self.vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings
        )
        
        # A place to store the retrieved documents
        self.last_retrieved_docs = []

        # --- ✨ MODIFICATION: Define the tool INSIDE __init__ ---
        @tool
        def retrieve(query: str) -> str: # No 'self' argument
            """
            Retrieve information related to a query. If a retrieved document is part of a
            larger file, this tool automatically fetches all parts and reconstructs the
            full document before returning.
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
                        sorted_chunks = sorted(zip(all_chunks['documents'], all_chunks['metadatas']), key=lambda item: int(item[1].get('part', '1/1').split('/')[0]))
                        full_content = "\n".join([chunk[0] for chunk in sorted_chunks])
                        merged_metadata = sorted_chunks[0][1]
                        merged_metadata.pop('part', None)
                        merged_metadata.pop('lines', None)
                        reconstructed_doc = Document(page_content=full_content, metadata=merged_metadata)
                        final_docs.append(reconstructed_doc)
                    else:
                        final_docs.append(doc)
            
            self.last_retrieved_docs = final_docs # Accesses self
            
            serialized = "\n\n".join(
                (f"Source File: {doc.metadata.get('filename', 'N/A')}\n"
                 f"Content:\n---\n{doc.page_content}\n---")
                for doc in final_docs
            )
            return serialized
        
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
        result = self.agent_executor.invoke({
            "input": user_question,
            "chat_history": self.chat_history
        })

        ai_message = result["output"]
        self.chat_history.extend([
            HumanMessage(content=user_question),
            AIMessage(content=ai_message),
        ])
        
        result['original_code'] = self.last_retrieved_docs
        
        return result

class aRAGAgent:
    def __init__(self, 
                 db_path: str,
                 model='gpt-oss:20b', 
                 embeddings_model='bge-m3:567m',
                 collection_name='documents',
                 system_prompt="You are a powerful AI assistant designed to help a PhD student with their research in scientific machine learning. You must answer questions based on the context provided by your tools.",
                 temperature=0, 
                 verbose=False,
                 k=5):
        """
        Initializes the agent, its tools, and its local ChromaDB using an explicit client.
        """
        
        # 1. Create the persistent client that connects to your database folder.
        client = chromadb.PersistentClient(path=db_path)
        
        # 2. Initialize the embeddings model.
        embeddings = OllamaEmbeddings(model=embeddings_model)
        
        # 3. Pass the explicit client to the LangChain Chroma wrapper.
        #    We now use the 'client' argument instead of 'persist_directory'.
        vector_store = Chroma(
            client=client,
            collection_name=collection_name, # Ensure this matches your ingestion script
            embedding_function=embeddings
        )
        
        # --- The retrieval tool remains the same ---
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

        # --- The rest of the agent setup is unchanged ---
        llm = ChatOllama(model=model, temperature=temperature)

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


class memoryAgent:
    def __init__(self,
                       model='gpt-oss:20b', 
                       system_prompt="You are a powerful AI assistant designed to help a PhD student with their research in scientific machine learning.", 
                       temperature=0, 
                       verbose=False):
        """
        Initializes the agent, its tools, and its memory.
        """

        llm = ChatOllama(model=model, temperature=temperature)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        tools = []
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        # Store the agent_executor and chat_history as instance attributes
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=verbose)
        self.chat_history = []

    def run(self, user_question: str) -> str:
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
                       system_prompt="You are a powerful AI assistant designed to help a PhD student with their research in scientific machine learning. You must answer questions based on the context provided by your tools", 
                       temperature=0, 
                       verbose=False):
        """
        Initializes the agent, its tools, and its memory.
        """

        llm = ChatOllama(model=model, temperature=temperature)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            #MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        tools = []
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        # Store the agent_executor and chat_history as instance attributes
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=verbose)

    def run(self, user_question: str) -> str:
        """
        Runs a single turn of the conversation.
        """
        result = self.agent_executor.invoke({
            "input": user_question,
        })

        ai_message = result["output"]
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
                 model='gpt-oss:120b', 
                 system_prompt="Your prompt goes here", 
                 temperature=0, 
                 verbose=False): # Verbose isn't used here, but keeping it is fine
        
        llm = ChatOllama(model=model, temperature=temperature)

        # ✨ MODIFICATION 1: Add MessagesPlaceholder for chat history ✨
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"), # Add this line
            ("user", "{input}"), 
        ])
        
        parser = JsonOutputParser()
        self.chain = self.prompt | llm | parser

    # ✨ MODIFICATION 2: Update 'run' to accept and pass history ✨
    def run(self, user_question: str, chat_history: List[BaseMessage] = None) -> Dict:
        """
        Runs a single turn, considering chat history, and returns a parsed dictionary.
        
        Args:
            user_question (str): The current input/question for the agent.
            chat_history (List[BaseMessage], optional): A list of previous messages 
                                                        (HumanMessage, AIMessage). 
                                                        Defaults to None or an empty list.
        """
        # Ensure chat_history is a list, even if None is passed
        history = chat_history or [] 
        
        # Pass both input and history to the chain
        result = self.chain.invoke({
            "input": user_question,
            "chat_history": history 
        })
        return result
