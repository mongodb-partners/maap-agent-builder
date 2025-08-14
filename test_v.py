from dotenv import load_dotenv
load_dotenv()
import os
from agent_builder.yaml_loader import load_application
from pprint import pprint
import asyncio
import traceback


# Set up async event loop for proper handling of coroutines
try:
    # Load the agent application from the YAML configuration file
    application = load_application("./config/agents.yaml")

    # Extract the agent from the loaded application
    agent_instance = application.get("agent")
    
    print("Successfully loaded application and agent")
except Exception as e:
    print(f"Error loading application: {str(e)}")
    traceback.print_exc()

response = agent_instance.invoke(
    {"messages": [{"role": "user", "content": "In how many distribution centers is levocetrizine currently available? How many units are in stock?"}]}, 
    config={"thread_id": "ashwin", "recursion_limit": 10}
)

# Print the agent's response
pprint(response["messages"][-1].content)
