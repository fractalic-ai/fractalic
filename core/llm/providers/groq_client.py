from groq import Groq
from core.config import Config  # Import Config to access settings

class groqclient:
    def __init__(self, api_key: str, settings: dict = None):
        if not api_key:
            raise ValueError("Groq API key must be provided")
        self.client = Groq(api_key=api_key)
        self.settings = settings or {}

    def llm_call(self, prompt_text: str, operation_params: dict = None) -> str:
        # Use settings from Config, with default fallbacks
        model = self.settings.get('model', Config.MODEL or "llama-3.1-70b-versatile")
        temperature = operation_params.get('temperature', self.settings.get('temperature', Config.TEMPERATURE or 0.0))
        max_tokens = self.settings.get('max_tokens', Config.CONTEXT_SIZE or 4096)
        top_p = self.settings.get('top_p', Config.TOP_P or 1)
        system_prompt = self.settings.get('system_prompt', Config.SYSTEM_PROMPT or "You are a helpful assistant.")

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=False,
            stop=None,
        )
        return response.choices[0].message.content
