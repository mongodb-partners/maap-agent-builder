"""
Async Flask application for MAAP Agent Builder with streaming support.

This module provides an async web application for the MAAP Agent Builder,
handling agent initialization, request routing, chat history management,
and real-time streaming responses.
"""

import asyncio
import json
import os
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

from dotenv import load_dotenv
from quart import Quart, jsonify, request, Response
from quart_cors import cors

from agent_builder.utils.logging_config import get_logger

# Load environment variables from .env file if present
load_dotenv()

# Initialize logger
logger = get_logger(__name__)


class AsyncAgentApp:
    """
    Async Quart application for serving agents loaded from YAML configurations.
    Supports both regular and streaming responses.
    """

    def __init__(self, config_path: str, session_ttl: int = 3600, agent: Optional[Any] = None):
        """
        Initialize the async agent application with the specified YAML configuration.

        Args:
            config_path: Path to the YAML configuration file
            session_ttl: Time-to-live for session data in seconds (default: 1 hour)
        """
        # Ensure Flask sansio default config has expected key for Quart <-> Flask compatibility
        try:
            import flask.sansio.app as _flask_sansio_app  # type: ignore
            default_cfg = getattr(_flask_sansio_app, "DEFAULT_CONFIG", None)
            if isinstance(default_cfg, dict) and "PROVIDE_AUTOMATIC_OPTIONS" not in default_cfg:
                default_cfg["PROVIDE_AUTOMATIC_OPTIONS"] = True
        except Exception:  # pragma: no cover - best effort patch
            pass

        # Disable static route creation (not needed for API) to avoid config key issues during tests
        self.app = Quart(__name__, static_folder=None)
        self.app = cors(self.app, allow_origin="*")  # Enable CORS for all origins
        self.app.secret_key = os.environ.get("FLASK_SECRET_KEY", str(uuid.uuid4()))
        # Ensure compatibility with Flask config expectations in some versions
        if "PROVIDE_AUTOMATIC_OPTIONS" not in self.app.config:
            self.app.config["PROVIDE_AUTOMATIC_OPTIONS"] = True
        self.config_path = config_path
        self.session_ttl = session_ttl  # Store for potential future use
        self.components = None
        # Multi-agent containers
        self.agents = {}
        self.default_agent_name: Optional[str] = None
        # Allow directly providing an agent (useful for tests); multi-agent injection not yet supported externally
        self.agent = agent
        self.chat_histories = {}  # Store chat histories by thread_id

        # Register routes
        self.register_routes()

        # Load agent components unless an agent was injected
        if self.agent is None:
            self.load_components()

    def load_components(self):
        """Load agent and related components from the YAML configuration."""
        try:
            logger.info("Loading application components from %s", self.config_path)
            # Lazy import to avoid pulling in heavy deps when agent injected
            from agent_builder.yaml_loader import load_application  # type: ignore
            self.components = load_application(self.config_path)

            if "agents" in self.components:
                self.agents = self.components["agents"]
                self.default_agent_name = self.components.get("default_agent_name")
                self.agent = self.components.get("agent")  # default alias
            else:
                if "agent" not in self.components:
                    logger.error("No agent configured in the YAML file")
                    raise ValueError("No agent configured in the YAML file")
                self.agent = self.components.get("agent")
                self.default_agent_name = self.components.get("default_agent_name")
            if not self.agent:
                logger.error("Agent object not properly initialized")
                raise ValueError("Failed to initialize agent")

            logger.info("Agent successfully loaded from %s", self.config_path)
        except Exception as e:
            logger.error("Failed to load application components: %s", str(e))
            raise

    async def _invoke_agent_async(self, input_data: Dict[str, Any], config: Dict[str, Any]) -> Any:
        """
        Async wrapper for agent invocation.
        
        Args:
            input_data: Input data for the agent
            config: Configuration for the agent
            
        Returns:
            Agent response
        """
        try:
            # Check if agent has async methods
            if hasattr(self.agent, "ainvoke"):
                return await self.agent.ainvoke(input_data, config=config)
            elif hasattr(self.agent, "invoke"):
                # Run sync invoke in an executor to avoid blocking
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.agent.invoke, input_data, config)
            else:
                # Legacy agent format
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, self.agent, input_data.get("messages", [])[-1][1])
        except Exception as e:
            logger.exception("Error in async agent invocation: %s", str(e))
            raise

    async def _stream_agent_response(self, input_data: Dict[str, Any], config: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """
        Stream agent response in real-time.
        
        Args:
            input_data: Input data for the agent
            config: Configuration for the agent
            
        Yields:
            Streaming response chunks
        """
        try:
            # Check if agent supports streaming
            if hasattr(self.agent, "astream"):
                async for chunk in self.agent.astream(input_data, config=config):
                    if isinstance(chunk, dict):
                        if "messages" in chunk and chunk["messages"]:
                            last_message = chunk["messages"][-1]
                            if isinstance(last_message, tuple) and len(last_message) >= 2:
                                content = last_message[1]
                            elif hasattr(last_message, "content"):
                                content = last_message.content
                            else:
                                content = str(last_message)
                            
                            yield f"data: {json.dumps({'content': content, 'type': 'message'})}\n\n"
                    else:
                        yield f"data: {json.dumps({'content': str(chunk), 'type': 'chunk'})}\n\n"
            elif hasattr(self.agent, "stream"):
                # For sync streaming, we'll simulate by invoking normally
                # This is not ideal but provides compatibility
                result = await self._invoke_agent_async(input_data, config)
                
                # Simulate streaming by yielding the result
                if isinstance(result, dict) and "messages" in result:
                    agent_messages = result["messages"]
                    if agent_messages:
                        last_message = agent_messages[-1]
                        if isinstance(last_message, tuple) and len(last_message) >= 2:
                            content = last_message[1]
                        elif hasattr(last_message, "content"):
                            content = last_message.content
                        else:
                            content = str(last_message)
                        
                        yield f"data: {json.dumps({'content': content, 'type': 'final'})}\n\n"
                else:
                    yield f"data: {json.dumps({'content': str(result), 'type': 'final'})}\n\n"
            else:
                # Fallback: invoke normally and return as stream
                response = await self._invoke_agent_async(input_data, config)
                
                if isinstance(response, dict) and "messages" in response:
                    agent_messages = response["messages"]
                    if agent_messages:
                        last_message = agent_messages[-1]
                        if isinstance(last_message, tuple) and len(last_message) >= 2:
                            content = last_message[1]
                        elif hasattr(last_message, "content"):
                            content = last_message.content
                        else:
                            content = str(last_message)
                        
                        yield f"data: {json.dumps({'content': content, 'type': 'final'})}\n\n"
                else:
                    yield f"data: {json.dumps({'content': str(response), 'type': 'final'})}\n\n"
                    
        except Exception as e:
            logger.exception("Error in streaming agent response: %s", str(e))
            yield f"data: {json.dumps({'error': f'Error: {str(e)}', 'type': 'error'})}\n\n"

    def register_routes(self):
        """Register API routes for the Quart application."""

        @self.app.route("/health", methods=["GET"])
        async def health():
            """Health check endpoint."""
            if self.agent:
                return jsonify({
                    "status": "healthy",
                    "agent_loaded": True,
                    "supports_async": True,
                    "multi_agent": bool(self.agents),
                    "agents": list(self.agents.keys()) if self.agents else [self.default_agent_name] if self.default_agent_name else [],
                    "default_agent": self.default_agent_name,
                })
            return jsonify({"status": "unhealthy", "agent_loaded": False}), 503

        @self.app.route("/chat", methods=["POST"])
        async def chat():
            """Async chat endpoint to interact with the agent."""
            if not self.agent:
                return jsonify({"error": "Agent not loaded"}), 503

            try:
                data = await request.get_json()
                if not data or "message" not in data:
                    return jsonify({"error": "Missing required field: message"}), 400

                # Get configuration from request
                config = data.get("config", {})
                if "config" in data:
                    del data["config"]  # Remove config from the main data

                # Get thread_id from config or generate a new one
                thread_id = config.get("thread_id", str(uuid.uuid4()))
                config["thread_id"] = thread_id  # Ensure thread_id is in config

                user_message = data["message"]
                logger.info(
                    "Received async chat request with message: %s... for thread %s", user_message[:50], thread_id
                )

                # Get or initialize chat history for this thread
                chat_history = self.chat_histories.get(thread_id, [])

                # Agent selection (optional)
                requested_agent_name = (
                    config.get("agent_name")
                    or request.args.get("agent")
                    or None
                )
                active_agent = self.agent
                if requested_agent_name:
                    if requested_agent_name not in self.agents:
                        return jsonify({
                            "error": f"Agent '{requested_agent_name}' not found",
                            "available_agents": list(self.agents.keys())
                        }), 400
                    active_agent = self.agents[requested_agent_name]

                # Prepare input for the agent
                input_data = {"messages": chat_history + [("user", user_message)]}

                # Include any additional parameters from the request
                for key, value in data.items():
                    if key != "message" and key != "history":
                        input_data[key] = value

                try:
                    # Invoke the agent asynchronously
                    # Temporarily swap self.agent for invocation to reuse existing method
                    original_agent = self.agent
                    try:
                        self.agent = active_agent
                        response = await self._invoke_agent_async(input_data, config)
                    finally:
                        self.agent = original_agent

                    # Process agent response
                    if isinstance(response, dict) and "messages" in response:
                        # LangGraph agent response
                        agent_messages = response["messages"]
                        if agent_messages:
                            last_message = agent_messages[-1]
                            if (
                                isinstance(last_message, tuple)
                                and len(last_message) >= 2
                            ):
                                agent_response = last_message[1]
                            elif hasattr(last_message, "content"):
                                agent_response = last_message.content
                            else:
                                agent_response = str(last_message)
                        else:
                            agent_response = "No response from agent"
                    else:
                        agent_response = str(response)

                    # Update chat history
                    chat_history.append(("user", user_message))
                    chat_history.append(("assistant", agent_response))
                    self.chat_histories[thread_id] = chat_history

                    return jsonify(
                        {
                            "response": agent_response,
                            "history": chat_history,
                            "thread_id": thread_id,
                        }
                    )

                except Exception as agent_error:
                    logger.exception("Error in async agent invocation: %s", str(agent_error))
                    agent_response = f"Error: {str(agent_error)}"
                    
                    # Update chat history with error
                    chat_history.append(("user", user_message))
                    chat_history.append(("assistant", agent_response))
                    self.chat_histories[thread_id] = chat_history

                    return jsonify(
                        {
                            "response": agent_response,
                            "history": chat_history,
                            "thread_id": thread_id,
                        }
                    ), 500

            except Exception as e:
                logger.exception("Error processing async chat request: %s", str(e))
                return jsonify({"error": f"Failed to process request: {str(e)}"}), 500

        @self.app.route("/chat/stream", methods=["POST"])
        async def chat_stream():
            """Streaming chat endpoint for real-time responses."""
            if not self.agent:
                return jsonify({"error": "Agent not loaded"}), 503

            try:
                data = await request.get_json()
                if not data or "message" not in data:
                    return jsonify({"error": "Missing required field: message"}), 400

                # Get configuration from request
                config = data.get("config", {})
                if "config" in data:
                    del data["config"]  # Remove config from the main data

                # Get thread_id from config or generate a new one
                thread_id = config.get("thread_id", str(uuid.uuid4()))
                config["thread_id"] = thread_id  # Ensure thread_id is in config

                user_message = data["message"]
                logger.info(
                    "Received streaming chat request with message: %s... for thread %s", user_message[:50], thread_id
                )

                # Get or initialize chat history for this thread
                chat_history = self.chat_histories.get(thread_id, [])

                # Agent selection (optional)
                requested_agent_name = (
                    config.get("agent_name")
                    or request.args.get("agent")
                    or None
                )
                active_agent = self.agent
                if requested_agent_name:
                    if requested_agent_name not in self.agents:
                        return jsonify({
                            "error": f"Agent '{requested_agent_name}' not found",
                            "available_agents": list(self.agents.keys())
                        }), 400
                    active_agent = self.agents[requested_agent_name]

                # Prepare input for the agent
                input_data = {"messages": chat_history + [("user", user_message)]}

                # Include any additional parameters from the request
                for key, value in data.items():
                    if key != "message" and key != "history":
                        input_data[key] = value

                async def generate():
                    """Generate streaming response."""
                    yield f"data: {json.dumps({'type': 'start', 'thread_id': thread_id})}\n\n"
                    
                    full_response = ""
                    # Stream using selected agent (swap like above)
                    original_agent = self.agent
                    try:
                        self.agent = active_agent
                        async for chunk in self._stream_agent_response(input_data, config):
                            yield chunk
                    finally:
                        self.agent = original_agent
                        
                        # Extract content from chunk for history
                        try:
                            chunk_data = json.loads(chunk.replace("data: ", "").strip())
                            if chunk_data.get("type") != "error" and "content" in chunk_data:
                                full_response = chunk_data["content"]
                        except json.JSONDecodeError:
                            pass
                    
                    # Update chat history
                    chat_history.append(("user", user_message))
                    if full_response:
                        chat_history.append(("assistant", full_response))
                    self.chat_histories[thread_id] = chat_history
                    
                    yield f"data: {json.dumps({'type': 'end', 'thread_id': thread_id})}\n\n"

                return Response(
                    generate(),
                    mimetype="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "Content-Type",
                    },
                )

            except Exception as e:
                logger.exception("Error processing streaming chat request: %s", str(e))
                return jsonify({"error": f"Failed to process streaming request: {str(e)}"}), 500

        @self.app.route("/reset", methods=["POST"])
        async def reset():
            """Reset the chat history for a specific thread or all threads."""
            try:
                data = await request.get_json() or {}
                thread_id = data.get("thread_id")

                if thread_id:
                    # Reset specific thread
                    if thread_id in self.chat_histories:
                        self.chat_histories[thread_id] = []
                        logger.info("Reset chat history for thread %s", thread_id)
                        return jsonify(
                            {
                                "status": "success",
                                "message": f"Chat history reset for thread {thread_id}",
                            }
                        )
                    else:
                        logger.warning(
                            "Attempted to reset non-existent thread %s", thread_id
                        )
                        return jsonify(
                            {
                                "status": "warning",
                                "message": f"Thread {thread_id} not found",
                            }
                        )
                else:
                    # Reset all threads
                    self.chat_histories = {}
                    logger.info("Reset all chat histories")
                    return jsonify(
                        {"status": "success", "message": "All chat histories reset"}
                    )
            except Exception as e:
                logger.exception("Error resetting chat history: %s", str(e))
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": f"Failed to reset chat history: {str(e)}",
                        }
                    ),
                    500,
                )

        @self.app.route("/threads", methods=["GET"])
        async def list_threads():
            """List all active thread IDs."""
            try:
                threads = list(self.chat_histories.keys())
                return jsonify(
                    {"status": "success", "threads": threads, "count": len(threads)}
                )
            except Exception as e:
                logger.exception("Error listing threads: %s", str(e))
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": f"Failed to list threads: {str(e)}",
                        }
                    ),
                    500,
                )

    async def run_async(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
        """Run the async Quart application."""
        logger.info("Starting async agent server on %s:%s", host, port)
        await self.app.run_task(host=host, port=port, debug=debug)

    def run(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
        """Run the async Quart application (sync wrapper)."""
        logger.info("Starting async agent server on %s:%s", host, port)
        self.app.run(host=host, port=port, debug=debug)


async def create_async_app(config_path: str) -> Quart:
    """
    Factory function to create and configure an async Quart application.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Configured Quart application
    """
    agent_app = AsyncAgentApp(config_path)
    return agent_app.app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the MAAP Agent Builder Async Flask application"
    )
    parser.add_argument(
        "--config", "-c", required=True, help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to run the server on (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=5000,
        help="Port to run the server on (default: 5000)",
    )
    parser.add_argument("--debug", "-d", action="store_true", help="Run in debug mode")

    args = parser.parse_args()

    agent_app = AsyncAgentApp(args.config)
    agent_app.run(host=args.host, port=args.port, debug=args.debug)
