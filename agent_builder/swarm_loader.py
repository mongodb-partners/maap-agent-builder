"""Swarm loader for MAAP Agent Builder.

Allows defining a `swarm` section in a YAML config (or a dedicated swarm
config file) to assemble a LangGraph Swarm multi-agent workflow using
`langgraph-swarm`'s `create_swarm` helper.

Schema (embedded or standalone):

llms: [...]
embeddings: [...]
tools: [...]
agents: [...]
# optional: checkpointer: {...}

swarm:
  agents: [agent_name_1, agent_name_2, ...]   # required, >=2
  default_active_agent: agent_name_1          # optional (defaults to first in list)
  checkpointer:                               # optional (overrides top-level)
    type: mongodb | memory (default memory if omitted)
    connection_str: ...                       # required for mongodb
    db_name: ...
    collection_name: ...
    name: swarm_checkpointer
  use_global_checkpointer: true | false       # default true if top-level checkpointer exists
  store:                                      # (future) placeholder for long-term memory store
    type: memory

Returned keys (in addition to base loader output):
  swarm_workflow: the uncompiled workflow (StateGraph)
  swarm: compiled app (call .invoke / .astream)
  swarm_config: dict with parsed swarm config

Usage:
  from agent_builder.swarm_loader import load_swarm_application
  components = load_swarm_application("agents_with_swarm.yaml")
  swarm_app = components["swarm"]
  result = swarm_app.invoke({"messages": [("user", "Hello swarm!")]},
                             config={"configurable": {"thread_id": "t1"}})
"""
from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional

from agent_builder.yaml_loader import load_application, load_yaml
from agent_builder.utils.logging_config import get_logger
from agent_builder.utils.checkpointer import get_mongodb_checkpointer

logger = get_logger(__name__)

try:  # soft dependency
    from langgraph_swarm import create_swarm  # type: ignore
    _SWARM_AVAILABLE = True
except Exception:  # pragma: no cover
    _SWARM_AVAILABLE = False

try:
    from langgraph.checkpoint.memory import InMemorySaver  # type: ignore
except ImportError:  # pragma: no cover
    InMemorySaver = None  # type: ignore


def _build_checkpointer(cfg: Dict[str, Any], global_ckpt: Optional[Dict[str, Any]]):
    """Create a checkpointer instance based on swarm or global config.

    Precedence order:
      1. swarm.checkpointer
      2. (if use_global_checkpointer) top-level global checkpointer
      3. InMemorySaver fallback
    """
    # explicit swarm-level config
    if cfg.get("checkpointer"):
        ck = cfg["checkpointer"]
        if ck.get("type", "mongodb") == "mongodb":
            try:
                return get_mongodb_checkpointer(**ck)
            except Exception as e:  # pragma: no cover - failure path
                logger.warning("Failed to init MongoDB swarm checkpointer: %s", e)
        else:  # memory
            if InMemorySaver:
                return InMemorySaver()

    # global checkpointer reuse
    if cfg.get("use_global_checkpointer", True) and global_ckpt:
        try:
            return get_mongodb_checkpointer(**global_ckpt)
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to reuse global checkpointer: %s", e)

    # fallback memory
    if InMemorySaver:
        return InMemorySaver()
    logger.warning("No checkpointer available (dependency missing); proceeding without persistence")
    return None


def load_swarm_application(config_path: str) -> Dict[str, Any]:
    """Load application components and assemble a swarm workflow if configured.

    This wraps `load_application` then inspects for a `swarm` section.
    Returns dictionary including keys from base loader plus `swarm` artifacts when present.
    """
    raw_config = load_yaml(config_path)
    base = load_application(config_path)

    swarm_cfg = raw_config.get("swarm")
    if not swarm_cfg:
        logger.info("No 'swarm' section found in %s; returning base components only", config_path)
        return base

    if not _SWARM_AVAILABLE:
        raise ImportError(
            "Swarm configuration present but 'langgraph-swarm' is not installed. Install with: pip install langgraph-swarm"
        )

    agent_names: List[str] = swarm_cfg.get("agents", [])
    if not agent_names or len(agent_names) < 2:
        raise ValueError("swarm.agents must list at least two agent names")

    # Validate agent existence
    missing = [a for a in agent_names if a not in base.get("agents", {})]
    if missing:
        raise ValueError(f"swarm agents not found among loaded agents: {missing}")

    default_active = swarm_cfg.get("default_active_agent") or agent_names[0]
    if default_active not in agent_names:
        raise ValueError(
            f"default_active_agent '{default_active}' must be one of swarm.agents: {agent_names}"
        )

    # Build checkpointer precedence
    global_ckpt_cfg = raw_config.get("checkpointer")
    checkpointer = _build_checkpointer(swarm_cfg, global_ckpt_cfg)

    # Build workflow
    try:
        agents_list = [base["agents"][n] for n in agent_names]
        workflow = create_swarm(agents_list, default_active_agent=default_active)
        compiled = workflow.compile(checkpointer=checkpointer) if checkpointer else workflow.compile()
    except Exception as e:
        logger.error("Failed to create or compile swarm workflow: %s\n%s", e, traceback.format_exc())
        raise

    base["swarm_workflow"] = workflow
    base["swarm"] = compiled
    base["swarm_config"] = {
        "agents": agent_names,
        "default_active_agent": default_active,
        "has_checkpointer": bool(checkpointer),
    }
    logger.info("Swarm workflow created with agents=%s default=%s", agent_names, default_active)
    return base

__all__ = ["load_swarm_application"]
