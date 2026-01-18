# %%
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
import sys
file_path = os.getcwd()
project_root = os.path.dirname(file_path)
print(project_root)
# Add the project root to the sys.path
if project_root not in sys.path:
    sys.path.append(project_root)
from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import tool
from typing import Dict, List, Union, Any 
from pathlib import Path
from IPython.display import display, Image
from typing import Optional,Dict, List, Tuple  
# Encoding
import base64
import mimetypes
import io  
import PIL.Image
import json 
import subprocess
import shutil # 
import re
from datetime import datetime 
import os
from pathlib import Path
# Our Libraries
from crunch_llm.code.agents import *
from crunch_llm.code.utils import *
from crunch_llm.code.teams_conceptualization import *

# %%
## Collection Team
coordinator_agent = memoryAgent(model='gpt-5.1', 
                     verbose=False,
                     system_prompt=coordinator_system_prompt,
                     model_provider='openai',     
                    )
gatekeeper_agent = StructuredAgent(
                    model='grok-4-1-fast-reasoning', 
                    model_provider='grok',
                    system_prompt=gatekeeper_system_prompt,
                    verbose=False
                )
# %%
# --- 2. Helper Functions ---

def is_approved(user_response):
    """
    Analyzes the user's input to determine if they approved the plan.
    Returns True for affirmative responses, False otherwise.
    """
    affirmative_keywords = [
        'yes', 'proceed', 'approved', 'ok', 'agree', 'looks good', 
        'correct', 'go ahead', 'fine', 'sure', 'perfect', 'done', 'agreed'
    ]
    
    # Normalize input
    response = user_response.strip().lower()
    
    # Check for direct matches or phrases starting with keywords
    # e.g., "Yes, looks good" or "Proceed with this"
    for word in affirmative_keywords:
        if response == word or response.startswith(word):
            return True
            
    return False

# %%
# --- 3. Interactive Chat Loop ---

def chat_with_coordinator(coordinator_agent):
    """
    Interactive chat loop with the Coordinator agent.
    - Detects when the agent proposes a plan.
    - Uses is_approved() to verify user confirmation.
    - Returns the plan text if approved, otherwise continues loop for refinement.
    """
    print("=" * 70)
    print("ATHENA Coordinator Agent")
    print("=" * 70)
    print("I'll help you formalize your computational research problem.")
    print("Type 'exit' to quit.\n")
    
    # State tracking
    pending_plan = None
    
    # Initial greeting
    initial_message = "Hello! I'm the Coordinator agent. What problem would you like to solve today?"
    print(f"Coordinator: {initial_message}\n")
    
    # Note: In a real conversation history, we'd append the initial message here.
    # For this loop, we rely on the agent's memory mechanism.
    
    while True:
        # 1. Get User Input
        if pending_plan:
            user_input = input("You (Approve/Modify): ").strip()
        else:
            user_input = input("You: ").strip()
        
        # 2. Check Exit Conditions
        if user_input.lower() in ['exit', 'quit']:
            print("\nExiting. Goodbye!")
            return None
            
        # 3. Handle Approval Logic (if a plan is pending)
        if pending_plan:
            if is_approved(user_input):
                print("\n" + "="*30)
                print(" PLAN APPROVED ")
                print("="*30)
                print("Forwarding to Gatekeeper...\n")
                return pending_plan
            else:
                print("\n[System]: Plan rejected. Sending feedback to Coordinator for refinement...\n")
                # The user's input (e.g., "No, change Re to 100") is passed to the agent below
                pending_plan = None

        # 4. Run Coordinator Agent
        # Pass the user's input (either new info or feedback on the rejected plan)
        result = coordinator_agent.run(user_input)
        
        # Extract response text
        if isinstance(result, dict) and 'output' in result:
            response = result['output']
        else:
            response = str(result)
            
        print(f"\nCoordinator: {response}\n")
        
        # 5. Check if Agent generated a new plan
        # We look for the specific trigger phrase defined in the system prompt
        if "reply with 'APPROVED', 'DONE', or 'AGREED'" in response:
            pending_plan = response
            print(">>> [System Alert]: A plan has been proposed. Please review above. <<<")

    return None


# %%
user_request = chat_with_coordinator(coordinator_agent)


if user_request:
    print("\n" + "=" * 70)
    print("ATHENA Gatekeeper Agent")
    print("=" * 70)
    print("Analyzing Approved Plan and Routing...\n")

    # Run the Gatekeeper
    result = gatekeeper_agent.run(user_request)
    
    # Extract output: Handle if agent returns Dict directly vs String
    try:
        routing_packet = None
        
        # Case 1: result is a dict containing 'output' which IS the structured dict
        if isinstance(result, dict) and 'output' in result:
            if isinstance(result['output'], dict):
                routing_packet = result['output']
            elif isinstance(result['output'], str):
                # Fallback: It's a string but clean JSON (as per user instruction)
                routing_packet = json.loads(result['output'])
        
        # Case 2: result is the structured dict itself
        elif isinstance(result, dict):
            routing_packet = result
            
        # Case 3: Fallback parsing if string
        else:
            routing_packet = json.loads(str(result))

        # Inject original request if missing
        if "user_request" not in routing_packet:
             routing_packet["user_request"] = user_request

        # --- UPDATED PRINT LOGIC FOR MULTI-STAGE PLANS ---
        print(f"Problem Name:    {routing_packet.get('problem_name', 'Unknown')}")
        print(f"Description:     {routing_packet.get('description', 'No description')}")
        print("-" * 30)
        print("EXECUTION PIPELINE:")
        
        pipeline = routing_packet.get('pipeline', [])
        if pipeline:
            for step in pipeline:
                stage_num = step.get('stage', '?')
                path_id = step.get('path', 'UNKNOWN')
                task = step.get('task', '')
                print(f"  [Stage {stage_num}] -> Team {path_id} : {task}")
        else:
            print("  [Error] No pipeline stages found in JSON.")
            
        # Save to Disk
        prob_name = routing_packet.get('problem_name', 'output').replace(" ", "_")
        filename = f"ATHENA_Request_{prob_name}.json"
        
        with open(filename, 'w') as f:
            json.dump(routing_packet, f, indent=4)
            
        print("-" * 30)
        print(f"\n[System]: Request saved successfully to: {os.path.abspath(filename)}")
        print("=" * 70)

    except Exception as e:
        print(f"Error processing Gatekeeper output: {e}")
        print("Raw Result:", result)

else:
    print("\n[System]: No approved user request found. Terminating.")