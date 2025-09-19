import os
import sys
import tempfile
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:  # pragma: no cover - path setup
    sys.path.insert(0, PROJECT_ROOT)

from agent_builder.yaml_loader import load_application  # noqa: E402


def write_yaml(tmp_path, data):
    p = os.path.join(tmp_path, 'agents.yaml')
    with open(p, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f)
    return p


def minimal_llm(name="dummy_llm"):
    # Provide a minimal LLM config referencing a provider expected by project tests or a dummy provider handled in loaders.
    # Using fireworks provider as seen in sample YAML (adjust if loader requires actual API keys; here we rely on lazy instantiation tolerance).
    return {
        'name': name,
        'provider': 'fireworks',
        'model_name': 'accounts/fireworks/models/llama4-maverick-instruct-basic',
        'temperature': 0.0,
        'streaming': False,
    }


def test_multi_agents_basic():
    os.environ.setdefault('FIREWORKS_API_KEY', 'test-key')
    data = {
        'llms': [minimal_llm('llm_a'), minimal_llm('llm_b')],
        'agents': [
            {
                'name': 'agent_a',
                'agent_type': 'tool_call',
                'llm': 'llm_a'
            },
            {
                'name': 'agent_b',
                'agent_type': 'tool_call',
                'llm': 'llm_b'
            }
        ],
        'default_agent': 'agent_b'
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = write_yaml(tmp, data)
        components = load_application(path)
        assert 'agents' in components
        assert set(components['agents'].keys()) == {'agent_a', 'agent_b'}
        assert components['default_agent_name'] == 'agent_b'
        assert components['agent'] is components['agents']['agent_b']


def test_legacy_single_agent_still_supported():
    os.environ.setdefault('FIREWORKS_API_KEY', 'test-key')
    data = {
        'llms': [minimal_llm('llm_single')],
        'agent': {
            'name': 'legacy_agent',
            'agent_type': 'tool_call',
            'llm': 'llm_single'
        }
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = write_yaml(tmp, data)
        components = load_application(path)
        assert 'agent' in components
        assert 'agents' not in components  # legacy path
        assert components['default_agent_name'] == 'legacy_agent'
