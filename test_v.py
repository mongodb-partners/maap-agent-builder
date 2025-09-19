import os
import traceback
from pprint import pprint
import pytest
from dotenv import load_dotenv

load_dotenv()

# Early skip: missing API keys OR missing optional heavy deps
_LLM_KEYS = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "FIREWORKS_API_KEY", "TOGETHER_API_KEY", "COHERE_API_KEY"]
if not any(os.getenv(k) for k in _LLM_KEYS):  # pragma: no cover - env dependent
    pytest.skip("Skipping integration test_v.py: no LLM API keys configured", allow_module_level=True)

try:  # Attempt to import heavy stack only after skip decision
    from agent_builder.yaml_loader import load_application
except ImportError as import_err:  # pragma: no cover
    pytest.skip(f"Skipping integration test_v.py due to import error: {import_err}", allow_module_level=True)


# Set up async event loop for proper handling of coroutines
try:
    # Load the agent application from the YAML configuration file
    application = load_application("./config/agents.yaml")

    # Extract the agent from the loaded application
    agent_instance = application.get("agent")
    
    print("Successfully loaded application and agent")
except (AssertionError, RuntimeError, ValueError) as e:  # Narrowed common error types
    print(f"Error loading application: {str(e)}")
    traceback.print_exc()
    agent_instance = None

if agent_instance is not None:  # pragma: no cover - integration path
    response = agent_instance.invoke(
        {"messages": [{"role": "user", "content": "In how many distribution centers is levocetrizine currently available? How many units are in stock?"}]}, 
        config={"thread_id": "ashwin", "recursion_limit": 10}
    )

    # Print the agent's response
    pprint(response["messages"][-1].content)
