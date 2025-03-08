import importlib
from core.config import Config
from core.utils import load_settings

class LLMClient:
    def __init__(self, provider: str, model: str = None):
        self.provider = provider.lower()
        self.model = model
        self.client = self._initialize_client()

    def _initialize_client(self):
        module_name = f"core.llm.providers.{self.provider}_client"
        class_name = f"{self.provider}client"
        module = importlib.import_module(module_name)
        client_class = getattr(module, class_name)
        
        # Get provider settings from Config
        provider_settings = Config.TOML_SETTINGS.get('settings', {}).get(self.provider, {})
        
        # Override model in settings if specified in constructor
        if self.model:
            provider_settings = provider_settings.copy()  # Create a copy to avoid modifying original
            provider_settings['model'] = self.model
            
        api_key = provider_settings.get('apiKey')
        return client_class(api_key=api_key, settings=provider_settings)

    def llm_call(self, prompt_text: str, messages: list = None, operation_params: dict = None) -> str:
        """
        Call the LLM with either a text prompt or structured messages.
        
        Args:
            prompt_text: Traditional string prompt (may be empty if messages are used)
            messages: List of message dictionaries with role/content pairs (optional)
            operation_params: Additional parameters for the LLM call
        
        Returns:
            String response from the LLM
        """
        # Forward the call to the provider-specific implementation
        return self.client.llm_call(prompt_text, messages, operation_params)
