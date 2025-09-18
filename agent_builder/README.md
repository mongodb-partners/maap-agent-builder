## MAAP Agent Builder (Package README)

For full project documentation see the root `README.md`.

### Async / Streaming Server

An async Quart-based server is available at `agent_builder/async_app.py` providing:

- `/chat` (async request/response)
- `/chat/stream` (SSE streaming)
- `/health`, `/reset`, `/threads`

Quick start:

```bash
python agent_builder/async_app.py --config agent_builder/agents.yaml --port 5000
```

Docker (async mode):

```bash
docker run -p 5000:5000 -e SERVER_MODE=async maap-agent-builder
```

Programmatic injection (skip YAML load):

```python
from agent_builder.async_app import AsyncAgentApp
app_wrapper = AsyncAgentApp(config_path="agent_builder/agents.yaml", agent=my_agent)
app = app_wrapper.app
```

If the agent supports `astream`, responses are streamed incrementally; otherwise a single `final` frame is emitted.

