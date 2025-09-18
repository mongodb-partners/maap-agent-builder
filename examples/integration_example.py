"""Integration Example (not part of automated pytest suite)

This script demonstrates loading the full agent application and invoking it.
It is intentionally excluded from pytest discovery to avoid requiring all
external API keys and cloud credentials during normal test runs.

Usage:
  python examples/integration_example.py

Environment:
  Requires at least one configured LLM API key (e.g. OPENAI_API_KEY) and
  any provider-specific credentials (AWS for Bedrock, etc.).
"""
import os
import traceback
from pprint import pprint
from dotenv import load_dotenv

load_dotenv()

_LLM_KEYS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "FIREWORKS_API_KEY",
    "TOGETHER_API_KEY",
    "COHERE_API_KEY",
]

if not any(os.getenv(k) for k in _LLM_KEYS):
    print("[integration_example] No LLM API keys set; skipping run.")
    raise SystemExit(0)

try:
    from agent_builder.yaml_loader import load_application
except ImportError as e:  # pragma: no cover
    print(f"Could not import application loader: {e}")
    raise SystemExit(0) from e

try:
    application = load_application("./config/agents.yaml")
    agent_instance = application.get("agent")
    print("Successfully loaded application and agent")
except Exception as e:  # pragma: no cover - integration specific
    print(f"Error loading application: {e}")
    traceback.print_exc()
    raise SystemExit(1) from e

query = (
    "In how many distribution centers is levocetrizine currently available? "
    "How many units are in stock?"
)

response = agent_instance.invoke(
    {"messages": [{"role": "user", "content": query}]},
    config={"thread_id": "integration_example", "recursion_limit": 10},
)

pprint(response["messages"][-1].content)
