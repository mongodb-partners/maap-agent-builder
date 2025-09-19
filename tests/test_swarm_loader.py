import os
import pytest

from agent_builder.swarm_loader import load_swarm_application
from agent_builder.yaml_loader import load_yaml

SWARM_YAML = """
llms:
  - name: dummy_llm
    provider: fireworks
    model_name: accounts/fireworks/models/llama4-maverick-instruct-basic
    temperature: 0.0

agents:
  - name: a1
    agent_type: react
    llm: dummy_llm
    system_prompt: "You are A1. Reply with A1 only."
  - name: a2
    agent_type: react
    llm: dummy_llm
    system_prompt: "You are A2. Reply with A2 only."

default_agent: a1

swarm:
  agents: [a1, a2]
  default_active_agent: a1
"""

@pytest.mark.skipif("FIREWORKS_API_KEY" not in os.environ, reason="Requires dummy FIREWORKS_API_KEY env var for LLM instantiation")
def test_swarm_loader_tmpfile(tmp_path):
    cfg_path = tmp_path / "swarm_agents.yaml"
    cfg_path.write_text(SWARM_YAML)

    components = load_swarm_application(str(cfg_path))
    assert "swarm" in components, "Swarm compiled app missing"
    assert components["swarm_config"]["default_active_agent"] == "a1"
    assert components["swarm_config"]["agents"] == ["a1", "a2"]

    # Basic invoke path (won't be meaningful without real model output but should not raise)
    app = components["swarm"]
    try:
        _ = app.invoke({"messages": [("user", "hi")]} , config={"configurable": {"thread_id": "t1"}})
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"Swarm invocation failed unexpectedly: {e}")
