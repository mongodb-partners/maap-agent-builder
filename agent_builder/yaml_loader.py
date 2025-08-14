import os
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import yaml
from pipe21 import Map, Pipe
from pydantic import BaseModel, create_model

from agent_builder.agents.loader import AgentConfig, load_agent
from agent_builder.embeddings.loader import (
    EmbeddingConfig,
    load_embedding_model,
    load_embedding_models,
)
from agent_builder.llms.loader import LLMConfig, load_llm, load_llms
from agent_builder.tools.loader import ToolConfig, load_tools
from agent_builder.utils.logger import logger
from agent_builder.utils.logging_config import get_logger

# Set up module logger
logger = get_logger(__name__)


def parse_response_model(response_dict: dict) -> BaseModel:
    """
    Parses a dictionary into a Pydantic model.

    Args:
        response_dict (dict): The dictionary to parse.

    Returns:
        BaseModel: A Pydantic model instance.
    """
    response_dict = (
        list(response_dict.items())
        | Map(lambda item: (item[0], tuple(item[1])))
        | Pipe(dict)
    )
    return create_model("ResponseModel", **response_dict)


def resolve_env_variables(data):
    """
    Recursively resolves environment variables in a dictionary or string.

    Handles both ${VAR_NAME} and ${VAR_NAME:-default_value} syntax.
    """
    if isinstance(data, dict):
        return {k: resolve_env_variables(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [resolve_env_variables(elem) for elem in data]
    elif isinstance(data, str):
        # Regex to find ${VAR_NAME} or ${VAR_NAME:-default_value}
        pattern = re.compile(r"\$\{(\w+)(:-([^}]*))?\}")

        def replace_match(match):
            var_name = match.group(1)
            default_value = match.group(3)

            # Get the environment variable, or use the default if provided
            value = os.environ.get(var_name, default_value)
            if value is None:
                logger.warning(
                    f"Environment variable '{var_name}' not set and no default provided."
                )
                raise ValueError(
                    f"Environment variable '{var_name}' not set and no default provided."
                )
            return value

        return pattern.sub(replace_match, data)
    return data


def load_yaml(file_path) -> dict:
    """
    Load and parse a YAML configuration file with environment variable resolution.

    Args:
        file_path: Path to the YAML file

    Returns:
        The parsed and resolved configuration dictionary

    Raises:
        FileNotFoundError: If the file doesn't exist
        YAMLError: If there's an error parsing the YAML
    """
    try:
        logger.info(f"Loading configuration from {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config:
            logger.warning(f"Empty or invalid YAML file: {file_path}")
            return {}

        logger.debug(f"Resolving environment variables in configuration")
        resolved_config = resolve_env_variables(config)
        return resolved_config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {file_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading YAML file {file_path}: {e}")
        raise


def load_application(config_path: str):
    """
    Load application components from a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        A dictionary containing the loaded application components
    """
    config = load_yaml(config_path)

    result = {}

    # Load embeddings
    if "embeddings" in config:
        logger.info("Loading embedding models")
        emb_configs = [EmbeddingConfig(**emb) for emb in config["embeddings"]]
        result["embeddings"] = load_embedding_models(emb_configs)

    # Load LLMs
    if "llms" in config:
        logger.info("Loading language models")
        llm_configs = [LLMConfig(**llm) for llm in config["llms"]]
        result["llms"] = load_llms(llm_configs)

    # Load tools with resolved references
    if "tools" in config:
        logger.info("Loading tools")
        tools_config = deepcopy(config["tools"])

        # Resolve embedding model references
        for tool in tools_config:
            if "embedding_model" in tool and isinstance(tool["embedding_model"], str):
                if tool["embedding_model"] not in result.get("embeddings", {}):
                    logger.error(
                        f"Referenced embedding model '{tool['embedding_model']}' not found"
                    )
                    raise ValueError(
                        f"Referenced embedding model '{tool['embedding_model']}' not found"
                    )
                tool["embedding_model"] = result["embeddings"][tool["embedding_model"]]

            if "llm" in tool and isinstance(tool["llm"], str):
                if tool["llm"] not in result.get("llms", {}):
                    logger.error(f"Referenced LLM '{tool['llm']}' not found")
                    raise ValueError(f"Referenced LLM '{tool['llm']}' not found")
                tool["llm"] = result["llms"][tool["llm"]]

        tool_configs = [ToolConfig(**tool) for tool in tools_config]
        result["tools"] = load_tools(tool_configs)

    # Load agent with resolved references
    if "agent" in config:
        logger.info("Loading agent")
        agent_config = deepcopy(config["agent"])

        # Resolve LLM reference
        if "llm" in agent_config and isinstance(agent_config["llm"], str):
            if agent_config["llm"] not in result.get("llms", {}):
                logger.error(f"Referenced LLM '{agent_config['llm']}' not found")
                raise ValueError(f"Referenced LLM '{agent_config['llm']}' not found")
            agent_config["llm"] = result["llms"][agent_config["llm"]]

        # Resolve tool references
        if "tools" in agent_config and isinstance(agent_config["tools"], list):
            resolved_tools = []
            for tool_name in agent_config["tools"]:
                if tool_name not in result.get("tools", {}):
                    logger.error(f"Referenced tool '{tool_name}' not found")
                    raise ValueError(f"Referenced tool '{tool_name}' not found")
                resolved_tools.append(result["tools"][tool_name])
            agent_config["tools"] = resolved_tools

        if "checkpointer" in config:
            agent_config["checkpointer_config"] = config["checkpointer"]

        # Load the agent
        agent_config_obj = AgentConfig(**agent_config)
        result["agent"] = load_agent(agent_config_obj)

    logger.info("Application components loaded successfully")
    return result
