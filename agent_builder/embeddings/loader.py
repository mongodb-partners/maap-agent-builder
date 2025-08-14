import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

# Import supported embedding providers
from langchain_aws import BedrockEmbeddings
from langchain_community.embeddings import (
    CohereEmbeddings,
    OllamaEmbeddings,
    SagemakerEndpointEmbeddings,
    VertexAIEmbeddings,
)
from langchain_core.embeddings import Embeddings
from langchain_fireworks import FireworksEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_together import TogetherEmbeddings
from langchain_voyageai import VoyageAIEmbeddings

from agent_builder.utils.logger import logger
from agent_builder.utils.logging_config import get_logger

"""
Embeddings loader for various providers using LangChain.
This module provides functionality to load embedding models from different providers
including Bedrock, SageMaker, VertexAI, Azure, Together, Fireworks, Cohere, VoyageAI, and Ollama.
"""

# Set up module logger
logger = get_logger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""

    name: str
    provider: str
    model_name: str
    normalize: bool = False
    dimensions: Optional[int] = None
    additional_kwargs: Optional[Dict[str, Any]] = None
    api_key: Optional[str] = None


# Provider configuration map
PROVIDER_CONFIG = {
    "bedrock": {
        "class": BedrockEmbeddings,
        "env_var": None,
        "error_msg": None,
        "uses_api_key": False,
        "model_key": "model_id",
    },
    "sagemaker": {
        "class": SagemakerEndpointEmbeddings,
        "env_var": None,
        "error_msg": None,
        "uses_api_key": False,
        "required_config": ["endpoint_name"],
        "default_config": {"region_name": "us-east-1"},
        "model_key": None,
    },
    "vertexai": {
        "class": VertexAIEmbeddings,
        "env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "error_msg": "Google application credentials required for Vertex AI",
        "uses_api_key": False,
        "model_key": "model_name",
    },
    "azure": {
        "class": AzureOpenAIEmbeddings,
        "env_var": "AZURE_OPENAI_API_KEY",
        "error_msg": "API key required for Azure OpenAI",
        "uses_api_key": True,
        "model_key": "deployment",
        "extra_config": {
            "env_var": "AZURE_OPENAI_ENDPOINT",
            "config_key": "azure_endpoint",
            "error_msg": "Azure endpoint required for Azure OpenAI",
        },
    },
    "together": {
        "class": TogetherEmbeddings,
        "env_var": "TOGETHER_API_KEY",
        "error_msg": "API key required for Together AI",
        "uses_api_key": True,
        "model_key": "model_name",
    },
    "fireworks": {
        "class": FireworksEmbeddings,
        "env_var": "FIREWORKS_API_KEY",
        "error_msg": "API key required for Fireworks",
        "uses_api_key": True,
        "model_key": "model",
    },
    "cohere": {
        "class": CohereEmbeddings,
        "env_var": "COHERE_API_KEY",
        "error_msg": "API key required for Cohere",
        "uses_api_key": True,
        "model_key": "model",
    },
    "voyageai": {
        "class": VoyageAIEmbeddings,
        "env_var": "VOYAGE_API_KEY",
        "error_msg": "API key required for VoyageAI",
        "uses_api_key": True,
        "model_key": "model",
    },
    "ollama": {
        "class": OllamaEmbeddings,
        "env_var": None,
        "error_msg": None,
        "uses_api_key": False,
        "model_key": "model",
        "default_config": {"base_url": "http://localhost:11434"},
    },
    "huggingface": {
        "class": HuggingFaceEmbeddings,
        "env_var": None,
        "error_msg": None,
        "uses_api_key": False,
        "model_key": "model_name",
    },
}


def load_embedding_model(config: EmbeddingConfig) -> Embeddings:
    """
    Load an embedding model based on the provided configuration.

    Args:
        config: EmbeddingConfig containing provider, model name, and other parameters

    Returns:
        An initialized Embeddings instance

    Raises:
        ValueError: If the provider is not supported or required configuration is missing
    """
    provider = config.provider.lower()
    logger.info(
        f"Loading embedding model for provider: {provider}, model: {config.model_name}"
    )

    # Check if provider is supported
    if provider not in PROVIDER_CONFIG:
        available_providers = list(PROVIDER_CONFIG.keys())
        logger.error(
            f"Unsupported embedding provider: {provider}. Available providers: {available_providers}"
        )
        raise ValueError(
            f"Unsupported embedding provider: {provider}. Available providers: {available_providers}"
        )

    provider_info = PROVIDER_CONFIG[provider]

    # Build common kwargs
    kwargs = {}

    # Add model name with the appropriate key
    if provider_info["model_key"]:
        kwargs[provider_info["model_key"]] = config.model_name

    # Add dimensions if provided and supported by the provider
    if config.dimensions:
        # Currently, only these providers support explicit dimension specification
        dimension_supporting_providers = ["cohere", "huggingface"]
        if provider in dimension_supporting_providers:
            kwargs["dimensions"] = config.dimensions
        elif provider == "voyageai":
            kwargs["output_dimension"] = config.dimensions
        else:
            logger.warning(
                f"Provider {provider} doesn't support explicit dimension specification. Ignoring dimensions={config.dimensions}"
            )

    # Check for required API keys
    if provider_info["env_var"] and provider_info["uses_api_key"]:
        env_var = provider_info["env_var"]

        # Use API key from config or environment variable
        api_key = config.api_key or os.environ.get(env_var)

        if not api_key:
            logger.error(f"{provider_info['error_msg']}")
            raise ValueError(provider_info["error_msg"])

        if provider_info["uses_api_key"]:
            kwargs["api_key"] = api_key

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

        # SageMaker embeddings need special handling due to different constructor signature
        return provider_info["class"](
            endpoint_name=sagemaker_kwargs["endpoint_name"],
            region_name=sagemaker_kwargs.get("region_name", "us-east-1"),
            content_handler=sagemaker_kwargs.get("content_handler"),
            **kwargs,
        )

    # Some embedding models support normalization
    if hasattr(provider_info["class"], "normalize") and config.normalize:
        kwargs["normalize"] = config.normalize

    # Add any additional kwargs
    if config.additional_kwargs:
        for key, value in config.additional_kwargs.items():
            if key not in kwargs:
                kwargs[key] = value

    logger.debug(f"Initializing {provider} embedding model with parameters: {kwargs}")
    return provider_info["class"](**kwargs)


def load_embedding_models(
    configs: Union[EmbeddingConfig, List[EmbeddingConfig]],
) -> Dict[str, Embeddings]:
    """
    Load multiple embedding models based on the provided configurations.

    Args:
        configs: Either a single EmbeddingConfig or a list of EmbeddingConfigs

    Returns:
        A dictionary mapping embedding model names to their initialized instances
    """
    if isinstance(configs, EmbeddingConfig):
        configs = [configs]

    embeddings = {}
    for config in configs:
        embeddings[config.name] = load_embedding_model(config)

    return embeddings
