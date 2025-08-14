# Config loader for MAAP Agent Builder
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import yaml
from pydantic import BaseModel


class CheckpointerConfig(BaseModel):
    connection_str: str
    db_name: Optional[str] = "langgraph"
    collection_name: Optional[str] = "checkpoints"
    name: Optional[str] = "mongodb_checkpointer"
