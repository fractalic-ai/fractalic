# Node
# - Node
# - NodeType
# - OperationType

import hashlib
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


    
class NodeType(Enum):
    HEADING = "heading"
    OPERATION = "operation"

class OperationType(Enum):
    REPLACE = "replace"
    PREPEND = "prepend"
    APPEND = "append"

@dataclass
class Node:
    type: NodeType
    name: str
    level: int
    params: Optional[Dict[str, Any]] = None  # Proper type annotation
    content: str = ""
    id: Optional[str] = None
    indent: int = 0
    source_path: Optional[str] = None
    source_block_id: Optional[str] = None
    target_path: Optional[str] = None
    target_block_id: Optional[str] = None
    key: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    prev: Optional['Node'] = None
    next: Optional['Node'] = None
    enabled: bool = True  # Persistent flag for run-once logic
    role: str = "user"  # Default role is "user", will be set to "assistant" for operation blocks and their results
    response_content: Optional[str] = None  # Store operation responses
    created_by: Optional[str] = None  # Store the key of the operation node that triggered this response
    # here we need to store the path of parent file file path for this run operation
    created_by_file : Optional[str] = None
    response_messages: Optional[list] = None  # Store full LLM/tool message trace
    is_system: bool = False  # Track if this is a system prompt block
    token_usage: Optional[Dict[str, Any]] = None  # Store token usage data for trace files

    @property
    def hash(self) -> str:
        return hashlib.md5(self.content.encode()).hexdigest()[:8]
    
    @classmethod
    def create_with_key(cls, preserve_key: str = None, **kwargs):
        """Create a Node with a specific key (for preserving attribution)"""
        node = cls(**kwargs)
        if preserve_key:
            node.key = preserve_key
        return node