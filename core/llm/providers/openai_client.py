from openai import OpenAI
from core.config import Config  # Import Config to access settings
from core.utils import load_settings

class openaiclient:
    def __init__(self, api_key: str, settings: dict = None):
        self.settings = settings or {}
        base_url = self.settings.get('base_url', "")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def llm_call(self, prompt_text: str, messages: list = None, operation_params: dict = None, model: str = None) -> str:
        model = model or (self.settings.get('model') or "gpt-4")
        temperature = operation_params.get('temperature', self.settings.get('temperature', 0.0))
        max_completion_tokens = self.settings.get('contextSize', None)
        top_p = self.settings.get('topP', 1)
        
        system_prompt = self.settings.get('systemPrompt', "You are a helpful assistant.")
        
        # Use provided messages if available, otherwise construct from prompt_text
        if messages and len(messages) > 0:
            # If messages don't start with a system message, prepend one
            if not messages or messages[0].get('role') != 'system':
                api_messages = [{"role": "system", "content": system_prompt}] + messages
            else:
                api_messages = messages
        else:
            api_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ]

        response = self.client.chat.completions.create(
            model=model,
            messages=api_messages,
            max_completion_tokens=max_completion_tokens if max_completion_tokens else None,
            top_p=top_p,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        return response.choices[0].message.content
