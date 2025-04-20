import anthropic
from core.config import Config
import base64
from pathlib import Path
import mimetypes
from typing import Dict, Any
from rich.box import ASCII
import imghdr
from PIL import Image
import io
import sys  # for streaming print
from rich.console import Console
from rich.live import Live

SUPPORTED_MEDIA_TYPES = {
    'image/jpeg': ['jpeg', 'jpg'],
    'image/png': ['png'],
    'image/gif': ['gif'],
    'image/webp': ['webp']
}

MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB in bytes

class anthropicclient:
    def __init__(self, api_key: str, settings: dict = None):
        self.settings = settings or {}
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def _validate_image(self, image_path: Path) -> tuple[str, bytes]:
        """Validate image format and size"""
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Check file size
        file_size = image_path.stat().st_size
        if file_size > MAX_IMAGE_SIZE:
            raise ValueError(f"Image too large: {file_size/1024/1024:.1f}MB. Max size: 20MB")
            
        # Read image
        img_data = image_path.read_bytes()
        
        # Validate format
        img_format = imghdr.what(None, img_data)
        if not img_format:
            raise ValueError(f"Unable to determine image format for {image_path}")
            
        # Match format to mime type
        mime_type = None
        for mime, formats in SUPPORTED_MEDIA_TYPES.items():
            if img_format in formats:
                mime_type = mime
                break
                
        if not mime_type:
            raise ValueError(
                f"Unsupported image format: {img_format}. "
                f"Supported types: {', '.join(SUPPORTED_MEDIA_TYPES.keys())}"
            )
            
        return mime_type, img_data

    def _load_media(self, media_path: str) -> Dict[str, Any]:
        """Load image or document with validation and encode as base64."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.syntax import Syntax
        from rich.text import Text
        
        console = Console()
        width = console.width - 4  # Account for panel borders

        try:
            path = Path(media_path)
            if not path.exists():
                # Get operation context from current stack
                import inspect
                frame = inspect.currentframe()
                while frame:
                    if 'current_node' in frame.f_locals:
                        operation = frame.f_locals['current_node']
                        # Format operation content with proper wrapping
                        operation_text = operation.content.strip()


                        # Print formatted error message
                        console.print()
                        console.print("[bold red]✗ Error:[/bold red] Media not found")
                        console.print(f"[red]Path:[/red] {media_path}")
                        console.print(
                            Syntax(
                                operation_text,
                                "yaml",
                                line_numbers=True,
                                word_wrap=True,
                                theme="monokai",
                                background_color="grey15"
                            )
                        )
                        break
                    frame = frame.f_back

                raise FileNotFoundError(f"Media not found: {media_path}")

            # Rest of existing media loading code...
            if path.suffix.lower() == ".pdf":
                media_data = path.read_bytes()
                return {
                    "type": "document",
                    "source": {
                        "type": "base64", 
                        "media_type": "application/pdf",
                        "data": base64.b64encode(media_data).decode("utf-8")
                    }
                }
                
            # Otherwise, handle images (existing logic)
            file_size = path.stat().st_size
            if file_size > MAX_IMAGE_SIZE:
                raise ValueError(f"Image too large: {file_size/1024/1024:.1f}MB. Max size: 20MB")

            img_data = path.read_bytes()
            img_format = imghdr.what(None, img_data)
            if not img_format:
                raise ValueError(f"Unable to determine image format for {media_path}")

            mime_type = None
            for mime, formats in SUPPORTED_MEDIA_TYPES.items():
                if img_format in formats:
                    mime_type = mime
                    break
            if not mime_type:
                raise ValueError(f"Unsupported image format: {img_format}. "
                                 f"Supported types: {', '.join(SUPPORTED_MEDIA_TYPES.keys())}")

            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": base64.b64encode(img_data).decode('utf-8')
                }
            }
        except Exception as e:
            console.print(f"[bold red]✗ Error:[/bold red] {str(e)}")
            raise

    def llm_call(self, prompt_text: str, messages: list = None, operation_params: dict = None, model: str = None) -> str:
        model = model or self.settings.get('model', "claude-3-5-sonnet-20241022")
        max_tokens = self.settings.get('contextSize', 8192)
        temperature = operation_params.get('temperature', self.settings.get('temperature', 0.0)) if operation_params else self.settings.get('temperature', 0.0)
        system_prompt = self.settings.get('systemPrompt', "")
        # streaming flag (set to true always in llmop)
        stream = operation_params.get('stream', False) if operation_params else False
        
        # Prepare API call based on input type
        if messages and len(messages) > 0:
            # Anthropic uses a specific format for messages
            anthropic_messages = []
            
            # Look for a system message first
            system_msg = next((msg for msg in messages if msg.get('role') == 'system'), None)
            
            api_system_prompt = system_msg['content'] if system_msg else system_prompt
            
            # Convert other messages to Anthropic format
            for msg in messages:
                if msg.get('role') == 'system':
                    continue
                anthropic_messages.append({
                    "role": msg.get('role'),
                    "content": [{"type": "text", "text": msg.get('content')}]
                })
            
            # Add media if exists
            if operation_params and 'media' in operation_params:
                for i, media_path in enumerate(operation_params['media']):
                    for j, msg in enumerate(anthropic_messages):
                        if msg.get('role') == 'user':
                            anthropic_messages[j]['content'].insert(0, self._load_media(media_path))
                            break
            
            # call with or without streaming
            if stream:
                console = Console()
                response_text = ""
                stream_resp = self.client.messages.create(
                    model=model,
                    messages=anthropic_messages,
                    system=api_system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True
                )
                for event in stream_resp:
                    text = None
                    if hasattr(event, "type"):
                        if event.type == "content_block_delta":
                            if hasattr(event, "delta") and hasattr(event.delta, "text"):
                                text = event.delta.text
                        elif event.type == "completion":
                            text = getattr(event, "completion", None)
                    if text:
                        response_text += text
                        console.print(text, end="", highlight=False)
                console.print()  # newline after streaming
                return response_text
            else:
                response = self.client.messages.create(
                    model=model,
                    messages=anthropic_messages,
                    system=api_system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
        else:
            # Traditional approach with prompt_text
            content = []
            
            # Add media if exists
            if operation_params and 'media' in operation_params:
                for media_path in operation_params['media']:
                    content.append(self._load_media(media_path))
            
            # Add text prompt
            content.append({
                "type": "text",
                "text": prompt_text
            })
            # call with or without streaming
            if stream:
                console = Console()
                response_text = ""
                stream_resp = self.client.messages.create(
                    model=model,
                    system=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": content}],
                    stream=True
                )
                for event in stream_resp:
                    text = None
                    if hasattr(event, "type"):
                        if event.type == "content_block_delta":
                            if hasattr(event, "delta") and hasattr(event.delta, "text"):
                                text = event.delta.text
                        elif event.type == "completion":
                            text = getattr(event, "completion", None)
                    if text:
                        response_text += text
                        console.print(text, end="", highlight=False)
                console.print()
                return response_text
            else:
                response = self.client.messages.create(
                    model=model,
                    system=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": content}]
                )
        # non-streaming return
        return response.content[0].text

