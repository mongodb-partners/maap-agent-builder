import asyncio
import os
import sys
import pytest
from typing import Any, Dict, List

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent_builder.async_app import AsyncAgentApp


class DummyAsyncAgent:
    """A minimal async-capable agent interface for testing."""

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    async def ainvoke(self, input_data: Dict[str, Any], config: Dict[str, Any]):
        self.calls.append({"input": input_data, "config": config, "mode": "ainvoke"})
        # Mimic LangGraph style response
        messages = input_data.get("messages", []) + [("assistant", f"Echo: {messages_last_content(input_data)}")]  # type: ignore
        return {"messages": messages}

    async def astream(self, input_data: Dict[str, Any], config: Dict[str, Any]):
        # Stream three chunks
        user_content = messages_last_content(input_data)
        for part in ["Processing", "...", f"Echo: {user_content}"]:
            await asyncio.sleep(0)  # yield control
            yield {"messages": [("assistant", part)]}


def messages_last_content(input_data: Dict[str, Any]) -> str:
    messages = input_data.get("messages", [])
    if not messages:
        return ""
    last = messages[-1]
    if isinstance(last, tuple) and len(last) > 1:
        return last[1]
    if isinstance(last, dict):
        return last.get("content") or last.get("text") or ""
    return str(last)


def create_client():
    agent = DummyAsyncAgent()
    app_wrapper = AsyncAgentApp(config_path="/dev/null", agent=agent)  # skip loading
    return app_wrapper.app.test_client()


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_health():
    client = create_client()
    resp = await client.get("/health")
    data = await resp.get_json()
    assert resp.status_code == 200
    assert data["agent_loaded"] is True
    assert data["supports_async"] is True


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_chat_basic():
    payload = {"message": "Hello async"}
    client = create_client()
    resp = await client.post("/chat", json=payload)
    data = await resp.get_json()
    assert resp.status_code == 200
    assert data["response"].startswith("Echo: ")
    assert data["history"][-1][0] == "assistant"


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_chat_stream():
    payload = {"message": "stream please"}
    client = create_client()
    resp = await client.post("/chat/stream", json=payload)
    # Collect SSE stream
    text = b""
    async for chunk in resp.response:  # type: ignore[attr-defined]
        text += chunk
    decoded = text.decode()
    # Expect start, some message chunks, end markers
    assert '"type": "start"' in decoded
    assert ('Echo: stream please' in decoded) or ('Processing' in decoded)
    assert '"type": "end"' in decoded

