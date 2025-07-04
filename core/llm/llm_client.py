import importlib
from core.config import Config
from core.utils import load_settings

class LLMClient:
    def __init__(self, model: str, provider: str = None):
        # Use provided provider or fallback to global config
        self.provider = provider or Config.LLM_PROVIDER or "openai"
        self.model    = model
        self.client   = self._initialize_client()

    def _initialize_client(self):
        api_key = Config.API_KEY
        cfg = Config.TOML_SETTINGS.get("settings", {}).get(self.provider, {})
        mcp_servers = Config.TOML_SETTINGS.get("mcp", {}).get("mcpServers", [])
        client_class = self._get_provider_client()
        # pass model into the client constructor
        return client_class(
            model=cfg.get("model"),
            api_key=api_key,
            settings=cfg,
            mcp_servers=mcp_servers
        )

    def _get_provider_client(self):
        """Return the LLM client class for the configured provider."""
        from core.llm.providers.openai_client import liteclient
        return liteclient
  
    def llm_call(self, prompt_text: str, messages: list = None, operation_params: dict = None) -> dict:
        """
        Call the LLM with either a text prompt or structured messages.
        
        Args:
            prompt_text: Traditional string prompt (may be empty if messages are used)
            messages: List of message dictionaries with role/content pairs (optional)
            operation_params: Additional parameters for the LLM call
        
        Returns:
            dict: { 'text': str, 'messages': list }
        """
        # Forward the call to the provider-specific implementation
        return self.client.llm_call(prompt_text, messages, operation_params)
