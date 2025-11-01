import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import SupabaseVectorStore
from langchain import hub
from supabase.client import Client, create_client
from langchain_core.tools import tool
from langchain_core.output_parsers import JsonOutputParser # Import the JSON parser
from typing import Dict # The return type will now be a dictionary
class aRAGAgent:
    def __init__(self, model='gpt-oss:20b', 
                       embeddings_model='bge-m3:567m',
                       system_prompt="You are a powerful AI assistant designed to help a PhD student with their research in scientific machine learning. You must answer questions based on the context provided by your tools", 
                       temperature=0, 
                       verbose=False,
                       supabase=''):
                       
        """
        Initializes the agent, its tools, and its memory.
        """
        # The tool is defined inside __init__ 
        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Retrieve information related to a query."""
            from langchain_core.documents import Document
            
            # Generate embedding for the query
            embeddings = OllamaEmbeddings(model=embeddings_model)
            query_embedding = embeddings.embed_query(query)
            
            # Query Supabase directly using the RPC function
            response = supabase.rpc(
                'match_documents',
                {
                    'query_embedding': query_embedding,
                    'match_count': 2,
                    'filter': {}
                }
            ).execute()
            
            # Convert response to Document objects
            retrieved_docs = [
                Document(
                    page_content=doc['content'],
                    metadata=doc.get('metadata', {})
                )
                for doc in response.data
            ]
            
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs

        llm = ChatOllama(model=model, temperature=temperature)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        tools = [retrieve]
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
            MessagesPlaceholder(variable_name="chat_history"),
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