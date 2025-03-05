
from enum import Enum

    
class Config:
    DEFAULT_OPERATION = "append"  # this operand is used when no operand is specified in the operation line

    LLM_PROVIDER = None
    API_KEY = None

    MODEL = None
    TEMPERATURE = None
    TOP_P = None
    TOP_K = None
    CONTEXT_SIZE = None
    SYSTEM_PROMPT = None  # TODO: not used now, implement later

    TOML_SETTINGS = None # it store raw file content of settings.toml
    #base_url = None  # Add base_url property

# Limit for @goto operation for one node in each run context
GOTO_LIMIT = 30




