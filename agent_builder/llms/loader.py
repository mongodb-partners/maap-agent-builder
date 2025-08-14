import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union

# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import ChatCohere
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_community.llms.sagemaker_endpoint import SagemakerEndpoint
from langchain_core.language_models import BaseLLM
from langchain_fireworks import ChatFireworks
from langchain_ollama.llms import OllamaLLM
from langchain_openai import AzureChatOpenAI
from langchain_together import ChatTogether

from agent_builder.utils.logger import logger

"""
LLM loader for various providers using LangChain.
This module provides functionality to load LLM models from different providers
including Bedrock, Fireworks, TogetherAI, Cohere, Anthropic, Azure, Google Gemini,
Ollama, and AWS SageMaker.
"""

# Set up module logger
# logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM models."""

    name: str
    provider: str
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    streaming: bool = False
    additional_kwargs: Optional[Dict[str, Any]] = None


# Provider configuration map
PROVIDER_CONFIG = {
    "bedrock": {
        "class": ChatBedrock,
        "env_var": None,
        "error_msg": None,
        "uses_api_key": False,
    },
    "fireworks": {
        "class": ChatFireworks,
        "env_var": "FIREWORKS_API_KEY",
        "error_msg": "API key required for Fireworks",
        "uses_api_key": False,
    },
    "together": {
        "class": ChatTogether,
        "env_var": "TOGETHER_API_KEY",
        "error_msg": "API key required for Together AI",
        "uses_api_key": True,
    },
    "cohere": {
        "class": ChatCohere,
        "env_var": "COHERE_API_KEY",
        "error_msg": "API key required for Cohere",
        "uses_api_key": False,
    },
    "anthropic": {
        "class": ChatAnthropic,
        "env_var": "ANTHROPIC_API_KEY",
        "error_msg": "API key required for Anthropic",
        "uses_api_key": False,
    },
    # "gemini": {
    #     "class": ChatGoogleGenerativeAI,
    #     "env_var": "GOOGLE_API_KEY",
    #     "error_msg": "API key required for Google Gemini",
    #     "uses_api_key": False,
    # },
    "azure": {
        "class": AzureChatOpenAI,
        "env_var": "AZURE_OPENAI_API_KEY",
        "error_msg": "API key required for Azure OpenAI",
        "uses_api_key": False,
        "extra_config": {
            "env_var": "AZURE_OPENAI_ENDPOINT",
            "config_key": "azure_endpoint",
            "error_msg": "Azure endpoint required for Azure OpenAI",
        },
    },
    "ollama": {
        "class": OllamaLLM,
        "env_var": None,
        "error_msg": None,
        "uses_api_key": False,
        "default_config": {"base_url": "http://localhost:11434"},
    },
    "sagemaker": {
        "class": SagemakerEndpoint,
        "env_var": None,
        "error_msg": None,
        "uses_api_key": False,
        "required_config": ["endpoint_name"],
        "default_config": {"region_name": "us-east-1"},
    },
}


def load_llm(config: LLMConfig) -> BaseLLM:
    """
    Load an LLM based on the provided configuration.

    Args:
        config: LLMConfig containing provider, model name, and other parameters

    Returns:
        An initialized LangChain LLM instance

    Raises:
        ValueError: If the provider is not supported or required configuration is missing
    """
    provider = config.provider.lower()
    logger.info(f"Loading LLM for provider: {provider}, model: {config.model_name}")

    # Check if provider is supported
    if provider not in PROVIDER_CONFIG:
        logger.error(f"Unsupported LLM provider: {provider}")
        raise ValueError(f"Unsupported LLM provider: {provider}")

    provider_info = PROVIDER_CONFIG[provider]

    # Build common kwargs
    kwargs = {
        "model": config.model_name,
        "temperature": config.temperature,
        "streaming": config.streaming,
    }

    # Add max_tokens if provided
    if config.max_tokens:
        kwargs["max_tokens"] = config.max_tokens

    # Check for required API keys
    if provider_info["env_var"] and provider_info["uses_api_key"]:
        # Handle both string and list env_var formats
        env_vars = (
            provider_info["env_var"]
            if isinstance(provider_info["env_var"], list)
            else [provider_info["env_var"]]
        )

        # Check if any of the environment variables are set
        env_var_set = any(os.environ.get(env_var) for env_var in env_vars)

        if not config.api_key and not env_var_set:
            logger.error(f"{provider_info['error_msg']}")
            raise ValueError(provider_info["error_msg"])

        # if provider_info["uses_api_key"]:
        #     # Use provided API key or find the first available environment variable
        #     if config.api_key:
        #         kwargs["api_key"] = config.api_key
        #     else:
        #         for env_var in env_vars:
        #             if os.environ.get(env_var):
        #                 kwargs["api_key"] = os.environ.get(env_var)
        #                 break

    # Handle extra configuration for specific providers
    if provider == "azure" and provider_info.get("extra_config"):
        extra = provider_info["extra_config"]
        azure_endpoint = (
            config.additional_kwargs.get("azure_endpoint")
            if config.additional_kwargs
            else None
        )

        if not azure_endpoint and not os.environ.get(extra["env_var"]):
            logger.error(f"{extra['error_msg']}")
            raise ValueError(extra["error_msg"])

        kwargs[extra["config_key"]] = azure_endpoint or os.environ.get(extra["env_var"])

    # Handle Ollama's special case
    if provider == "ollama":
        base_url = (config.additional_kwargs or {}).get(
            "base_url", provider_info["default_config"]["base_url"]
        )
        kwargs["base_url"] = base_url

    # Handle SageMaker's special case
    if provider == "sagemaker":
        if not config.additional_kwargs:
            logger.error("Additional kwargs required for SageMaker")
            raise ValueError("SageMaker requires additional configuration")

        for required_key in provider_info.get("required_config", []):
            if required_key not in config.additional_kwargs:
                logger.error(
                    f"Missing required configuration for SageMaker: {required_key}"
                )
                raise ValueError(
                    f"Missing required configuration for SageMaker: {required_key}"
                )

        # Apply defaults then override with user settings
        sagemaker_kwargs = provider_info["default_config"].copy()
        sagemaker_kwargs.update(config.additional_kwargs)

        return provider_info["class"](
            endpoint_name=sagemaker_kwargs["endpoint_name"],
            region_name=sagemaker_kwargs.get("region_name", "us-east-1"),
            model_kwargs=kwargs,
        )

    # Add any additional kwargs
    if config.additional_kwargs:
        kwargs.update(config.additional_kwargs)

    logger.debug(f"Initializing {provider} LLM with parameters: {kwargs}")
    return provider_info["class"](**kwargs)


def load_llms(configs: Union[LLMConfig, List[LLMConfig]]) -> Dict[str, BaseLLM]:
    """
    Load multiple LLMs based on the provided configurations.

    Args:
        configs: Either a single LLMConfig or a list of LLMConfigs

    Returns:
        A dictionary mapping LLM names to their initialized instances
    """
    if isinstance(configs, LLMConfig):
        configs = {configs.name: configs}

    llms = {}
    for config in configs:
        llms[config.name] = load_llm(config)

    return llms
