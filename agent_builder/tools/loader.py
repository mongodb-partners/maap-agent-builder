"""
Tool loader for MDB Agent Builder.

This module provides functionality to load different types of tools
including MongoDB tools, MCP tools, and other LangChain tools.
It handles tool configuration, validation, and instantiation.
"""

import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLLM
from langchain_core.tools import BaseTool

from agent_builder.tools.mcp import get_mcp_tools
from agent_builder.tools.mongodb import MongoDBTools
from agent_builder.utils.logging_config import get_logger

# Set up module logger
logger = get_logger(__name__)


class ToolType(str, Enum):
    """
    Enumeration of supported tool types in the MDB Agent Builder.

    These constants define the available tool types that can be configured and loaded.
    Each type corresponds to a specific functionality provided by the system.
    """

    VECTOR_SEARCH = "vector_search"  # Vector-based semantic search in MongoDB
    MONGODB_TOOLKIT = "mongodb_toolkit"  # Collection of MongoDB tools
    NL_TO_MQL = "nl_to_mql"  # Natural language to MongoDB Query Language
    MCP = "mcp"  # Model Context Protocol tools
    FULL_TEXT_SEARCH = "full_text_search"  # Text search in MongoDB


@dataclass
class ToolConfig:
    """
    Configuration class for setting up tools in the agent builder.

    This dataclass stores all necessary parameters for initializing
    different types of tools, with appropriate defaults for optional fields.

    Attributes:
        tool_type: The type of tool to initialize (see ToolType enum)
        name: The name identifier for the tool
        description: Optional description of the tool's purpose
        connection_str: MongoDB connection string (for MongoDB-related tools)
        namespace: MongoDB namespace in format "db.collection" (for MongoDB-related tools)
        embedding_model: Embedding model for vector-based tools
        llm: Language model for tools requiring LLM capabilities
        servers_config: Configuration for MCP servers
        additional_kwargs: Additional keyword arguments for tool initialization
    """

    tool_type: str
    name: str
    description: Optional[str] = None
    connection_str: Optional[str] = None
    namespace: Optional[str] = None
    embedding_model: Optional[Any] = None
    llm: Optional[BaseLLM] = None
    servers_config: Optional[Dict[str, Dict[str, Any]]] = None
    additional_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)


def load_tool(config: ToolConfig) -> BaseTool:
    """
    Load a tool based on the provided configuration.

    Args:
        config: ToolConfig containing tool type and other parameters

    Returns:
        An initialized LangChain tool instance

    Raises:
        ValueError: If the tool type is not supported or required configuration is missing
    """
    tool_type = config.tool_type.lower()
    tool_name = config.name or f"{tool_type}_tool"
    logger.info("Loading tool of type: %s, name: %s", tool_type, tool_name)

    if tool_type == ToolType.VECTOR_SEARCH:
        _check_required_fields(
            config, ["connection_str", "namespace", "embedding_model"], tool_type
        )

        # Create MongoDB tools instance
        mongodb_tools = MongoDBTools(
            name=tool_name,
            connection_str=config.connection_str,
            namespace=config.namespace,
            embedding_model=config.embedding_model,
            **config.additional_kwargs,
        )

        # Get vector retriever tool
        return mongodb_tools.get_vector_retriever_tool()

    elif tool_type == ToolType.MONGODB_TOOLKIT:
        _check_required_fields(
            config, ["connection_str", "namespace", "llm"], tool_type
        )

        # Create MongoDB tools instance
        mongodb_tools = MongoDBTools(
            name=tool_name,
            connection_str=config.connection_str,
            namespace=config.namespace,
            embedding_model=None,  # Not needed for MongoDB toolkit
            **config.additional_kwargs,
        )

        # Get MongoDB toolkit tools
        return mongodb_tools.get_mdb_toolkit(config.llm)

    elif tool_type == ToolType.NL_TO_MQL:
        _check_required_fields(
            config, ["connection_str", "namespace", "llm"], tool_type
        )

        # Create MongoDB tools instance
        mongodb_tools = MongoDBTools(
            name=tool_name,
            connection_str=config.connection_str,
            namespace=config.namespace,
            embedding_model=None,  # Not needed for NL to MQL
            **config.additional_kwargs,
        )

        # Get NL to MQL tool
        return mongodb_tools.get_nl_to_mql_tool(config.llm)

    elif tool_type == ToolType.MCP:
        _check_required_fields(config, ["servers_config"], tool_type)

        # Get MCP tools
        server_name = config.name
        return get_mcp_tools(config.servers_config, server_name)

    elif tool_type == ToolType.FULL_TEXT_SEARCH:
        _check_required_fields(config, ["connection_str", "namespace"], tool_type)

        # Create MongoDB tools instance
        mongodb_tools = MongoDBTools(
            name=tool_name,
            connection_str=config.connection_str,
            namespace=config.namespace,
            embedding_model=None,  # Not needed for full text search
            **config.additional_kwargs,
        )

        # Get full text search tool
        return mongodb_tools.get_full_text_search_tool()

    else:
        logger.error("Unsupported tool type: %s", tool_type)
        raise ValueError(f"Unsupported tool type: {tool_type}")


def _check_required_fields(config: ToolConfig, fields: List[str], tool_type: str):
    """
    Check if all required fields are present in the tool configuration.

    Args:
        config: The tool configuration to validate
        fields: List of field names that are required
        tool_type: The type of tool being validated (for error messages)

    Raises:
        ValueError: If any required field is missing
    """
    missing_fields = [field for field in fields if not getattr(config, field)]
    if missing_fields:
        error_msg = f"Missing required field(s) for {tool_type} tool: {', '.join(missing_fields)}"
        logger.error(error_msg)
        raise ValueError(error_msg)


def load_tools(configs: List[ToolConfig]) -> Dict[str, BaseTool]:
    """
    Load multiple tools based on the provided configurations.

    This function iterates through a list of tool configurations, initializes each tool,
    and handles any errors that might occur during tool initialization. It attempts to
    continue loading other tools even if some fail.

    Args:
        configs: List of ToolConfig objects describing tools to initialize

    Returns:
        Dictionary mapping tool names to initialized LangChain tool instances
    """
    logger.info("Loading %d tools", len(configs))
    tools = []

    for config in configs:
        try:
            # Attempt to load the tool based on its configuration
            tool = load_tool(config)

            if isinstance(tool, list):
                # Handle tools that return multiple tool instances (e.g., MongoDB toolkit)
                tools.extend(zip([config.name] * len(tool), tool))
                logger.info("Added %d tools from %s", len(tool), config.tool_type)
            else:
                # Handle single tool instance
                tools.append((config.name, tool))
                logger.info("Added tool: %s", config.name or config.tool_type)
        except RuntimeWarning as w:
            # Special handling for Runtime warnings like unawaited coroutines
            logger.warning(
                "RuntimeWarning loading tool %s: %s\n%s",
                config.name or config.tool_type,
                str(w),
                traceback.format_exc(),
            )
            # Continue loading other tools
        except Exception as e:  # pylint: disable=broad-except
            # Log errors but continue with other tools to maintain resilience
            logger.error(
                "Failed to load tool %s: %s\n%s",
                config.name or config.tool_type,
                str(e),
                traceback.format_exc(),
            )

    # Convert the list of (name, tool) pairs to a dictionary for easy access
    return dict(tools)
