"""
Flask application for MAAP Agent Builder.

This module provides the main web application for the MAAP Agent Builder,
handling agent initialization, request routing, and chat history management.
"""

import os
import uuid
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, request, session

from agent_builder.utils.logging_config import get_logger
from agent_builder.yaml_loader import load_application

# Load environment variables from .env file if present
load_dotenv()

# Initialize logger
logger = get_logger(__name__)


class AgentApp:
    """
    Flask application for serving agents loaded from YAML configurations.
    """

    def __init__(self, config_path: str, session_ttl: int = 3600):
        """
        Initialize the agent application with the specified YAML configuration.

        Args:
            config_path: Path to the YAML configuration file
            session_ttl: Time-to-live for session data in seconds (default: 1 hour)
        """
        self.app = Flask(__name__)
        self.app.secret_key = os.environ.get("FLASK_SECRET_KEY", str(uuid.uuid4()))
        self.app.config["PERMANENT_SESSION_LIFETIME"] = session_ttl
        self.config_path = config_path
        self.components = None
        self.agent = None
        self.chat_histories = {}  # Store chat histories by thread_id

        # Register routes
        self.register_routes()

        # Load agent components
        self.load_components()

    def load_components(self):
        """Load agent and related components from the YAML configuration."""
        try:
            logger.info("Loading application components from %s", self.config_path)
            self.components = load_application(self.config_path)

            if "agent" not in self.components:
                logger.error("No agent configured in the YAML file")
                raise ValueError("No agent configured in the YAML file")

            self.agent = self.components.get("agent")
            if not self.agent:
                logger.error("Agent object not properly initialized")
                raise ValueError("Failed to initialize agent")

            logger.info("Agent successfully loaded from %s", self.config_path)
        except Exception as e:
            logger.error("Failed to load application components: %s", str(e))
            raise

    def register_routes(self):
        """Register API routes for the Flask application."""

        @self.app.route("/health", methods=["GET"])
        def health():
            """Health check endpoint."""
            if self.agent:
                return jsonify({"status": "healthy", "agent_loaded": True})
            return jsonify({"status": "unhealthy", "agent_loaded": False}), 503

        @self.app.route("/chat", methods=["POST"])
        def chat():
            """Chat endpoint to interact with the agent."""
            if not self.agent:
                return jsonify({"error": "Agent not loaded"}), 503

            try:
                data = request.json
                if not data or "message" not in data:
                    return jsonify({"error": "Missing required field: message"}), 400

                # Get configuration from request
                if "config" not in data:
                    config = {}
                    logger.warning("No config provided in request, using empty config")
                else:
                    config = data["config"]
                    del data["config"]  # Remove config from the main data

                # Get thread_id from config or generate a new one
                thread_id = config.get("thread_id", str(uuid.uuid4()))
                config["thread_id"] = thread_id  # Ensure thread_id is in config

                user_message = data["message"]
                logger.info(
                    f"Received chat request with message: {user_message[:50]}... for thread {thread_id}"
                )

                # Get or initialize chat history for this thread
                chat_history = self.chat_histories.get(thread_id, [])

                # Prepare input for the agent
                if hasattr(self.agent, "invoke"):
                    # For LangGraph agents
                    input_data = {"messages": chat_history + [("user", user_message)]}

                    # Include any additional parameters from the request
                    for key, value in data.items():
                        if key != "message" and key != "history":
                            input_data[key] = value

                    try:
                        # Invoke the agent
                        response = self.agent.invoke(input_data, config=config)

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
                    except Exception as agent_error:
                        logger.exception(
                            f"Error in agent invocation: {str(agent_error)}"
                        )
                        agent_response = f"Error: {str(agent_error)}"

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
                else:
                    # Legacy agent format fallback
                    logger.warning("Using legacy agent format")
                    try:
                        agent_response = str(self.agent(user_message))
                    except Exception as legacy_error:
                        logger.exception(
                            f"Error in legacy agent invocation: {str(legacy_error)}"
                        )
                        agent_response = f"Error: {str(legacy_error)}"

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

            except Exception as e:
                logger.exception(f"Error processing chat request: {str(e)}")
                return jsonify({"error": f"Failed to process request: {str(e)}"}), 500

        @self.app.route("/reset", methods=["POST"])
        def reset():
            """Reset the chat history for a specific thread or all threads."""
            try:
                data = request.json or {}
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
                            f"Attempted to reset non-existent thread {thread_id}"
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
                logger.exception(f"Error resetting chat history: {str(e)}")
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
        def list_threads():
            """List all active thread IDs."""
            try:
                threads = list(self.chat_histories.keys())
                return jsonify(
                    {"status": "success", "threads": threads, "count": len(threads)}
                )
            except Exception as e:
                logger.exception(f"Error listing threads: {str(e)}")
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": f"Failed to list threads: {str(e)}",
                        }
                    ),
                    500,
                )

    def run(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
        """Run the Flask application."""
        logger.info("Starting agent server on %s:%s", host, port)
        self.app.run(host=host, port=port, debug=debug)


def create_app(config_path: str) -> Flask:
    """
    Factory function to create and configure a Flask application.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Configured Flask application
    """
    agent_app = AgentApp(config_path)
    return agent_app.app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the MAAP Agent Builder Flask application"
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

    agent_app = AgentApp(args.config)
    agent_app.run(host=args.host, port=args.port, debug=args.debug)
