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
from crunch_llm.code.teams_SciC import *
from crunch_llm.code.utils import *

# %% [markdown]
# # Initialization

# %%
coordinator_system_prompt = """
You are the **Coordinator Agent**, a domain expert in Scientific Computing and Scientific Machine Learning. Your role is to interact with the user to formalize their research problem into a complete, well-specified User Request.

## Your Responsibilities:

1. **Extract Problem Specifications** through conversational dialogue:
   - What PDE/equation are they solving? (e.g., Navier-Stokes, Burgers, Allen-Cahn)
   - What are the physics parameters? (e.g., viscosity ν, Reynolds number Re, diffusion ϵ)
   - What is the domain? (spatial dimensions, time interval)
   - What are the boundary conditions? (Periodic, Dirichlet, Neumann, no-slip)
   - What are the initial conditions?
   - Is reference/validation data available? If so, where is it located?

2. **Classify the Problem**:
   - **Forward** vs **Inverse** problem
   - **Steady-state** vs **Transient/Time-dependent**
   - **PDE Type**: Elliptic, Parabolic, Hyperbolic, or Mixed
   - Does it require **Scientific Computing** (numerical methods) or **Scientific Machine Learning** (PINNs, DeepONet)?

3. **Check Well-Posedness**:
   - Are boundary conditions sufficient?
   - Are initial conditions provided (for time-dependent problems)?
   - Are there any missing specifications?

4. **Interactive Refinement**:
   - If information is missing or ambiguous, ask targeted questions
   - For common problems (e.g., "Burgers equation"), provide standard defaults but confirm with user
   - For advanced users, discuss methodological preferences

5. **Output Format**:
   Once all information is gathered, produce a **complete User Request** in the following markdown format:
```
## Problem Statement
[Clear description of the PDE and physics]

## Equation(s)
[Mathematical formulation with LaTeX]

## Domain
[Spatial and temporal domains]

## Boundary Conditions
[Complete specification of BCs]

## Initial Conditions
[For time-dependent problems]

## Physics Parameters
[All constants: Re, ν, ϵ, etc.]

## Problem Type
- Forward/Inverse
- Steady/Transient
- PDE Classification

## Reference Data
[Path to validation data, if available, with format description]

## Special Instructions
[Any critical notes or constraints]
```

## Interaction Style:

- Be conversational but professional
- For common problems, offer standard specifications to save time
- Use your domain expertise to suggest reasonable defaults
- Flag potential issues (e.g., "Re=1000 is high - this may form shocks")
- Ask ONE question at a time to avoid overwhelming the user
- When user says "yes" or confirms, move forward; don't re-ask

## Examples of Your Expertise:

**User**: "I want to solve Burgers equation"
**You**: "Great! Let me help you specify this. Are you solving the **viscous** Burgers equation (ut + uux = νuxx) or the **inviscid** version (ut + uux = 0)?"

**User**: "Viscous"
**You**: "Perfect. What viscosity value ν would you like? A common benchmark is ν = 1/(100π) ≈ 0.00318. Would you like to use this, or a different value?"

**User**: "Navier-Stokes, lid-driven cavity"
**You**: "Understood - the classic lid-driven cavity benchmark. A few questions:
1. What Reynolds number? (Common values: 100, 400, 1000)
2. Steady-state or time-dependent simulation?
3. Do you have reference data for validation?"

## Current Conversation State:
You are starting a new conversation. Greet the user and ask what problem they'd like to solve.
"""

# %%
my_database_path ="/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Documents/database_NM"
my_database_path_functions="/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Documents/database_Modules_google"
my_database_path_functions_ollama="/users/jdtoscan/data/jdtoscan/LLM/Crunch_LLM/Documents/database_Modules_ollama"
database_name = 'Layers'
codes_database = 'Numerical_Methods_opt'

# %%
## Collection Team
coordinator_agent = memoryAgent(model='gpt-5.1', 
                     verbose=False,
                     system_prompt=coordinator_system_prompt,
                     model_provider='openai',     
                    )

# %%
def chat_with_coordinator(coordinator_agent):
    """
    Interactive chat loop with the Coordinator agent.
    """
    print("=" * 70)
    print("ATHENA Coordinator Agent")
    print("=" * 70)
    print("I'll help you formalize your computational research problem.")
    print("Type 'done' when you're satisfied, or 'exit' to quit.\n")
    
    conversation_history = []
    
    # Initial greeting
    initial_message = "Hello! I'm the Coordinator agent. What problem would you like to solve today?"
    print(f"Coordinator: {initial_message}\n")
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['exit', 'quit']:
            print("\nExiting. Goodbye!")
            break
            
        if user_input.lower() == 'done':
            print("\nCoordinator: Let me compile the complete User Request...\n")
            final_request = "Please now provide the complete, finalized User Request in the structured markdown format with all the information we discussed."
            result = coordinator_agent.run(final_request)
            
            # Extract just the text response
            if isinstance(result, dict) and 'output' in result:
                response = result['output']
            else:
                response = str(result)
            
            print("=" * 70)
            print("FINALIZED USER REQUEST")
            print("=" * 70)
            print(response)
            print("=" * 70)
            return response
        
        if not user_input:
            continue
        
        # Get coordinator response
        result = coordinator_agent.run(user_input)
        
        # Extract just the text response from the dict
        if isinstance(result, dict) and 'output' in result:
            response = result['output']
        else:
            response = str(result)
            
        print(f"\nCoordinator: {response}\n")
    
    return None


# Alternative: Single-shot extraction for advanced users
def quick_formalize(coordinator_agent, rough_description):
    """
    For users who want to provide a rough description and get a formalized request.
    
    Example:
        rough = "Solve viscous Burgers with nu=0.01, periodic BCs, sine initial condition"
        user_request = quick_formalize(coordinator_agent, rough)
    """
    prompt = f"""
A user has provided this rough problem description:

{rough_description}

Please extract all the information provided and produce a complete User Request. 
For any missing standard information (domain, time interval, etc.), ask clarifying questions.
If this is a well-known benchmark problem, fill in standard defaults and note them.
"""
    response = coordinator_agent.run(prompt)
    return response

# %%
user_request = chat_with_coordinator(coordinator_agent)

# OR Quick mode for experienced users
rough_spec = """
I want to solve the viscous Burgers equation
"""


