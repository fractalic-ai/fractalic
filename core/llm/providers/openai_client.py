"""
lite_client.py  —  Unified multimodal client (LiteLLM)

✓  vision images (JPEG, PNG, GIF, WEBP)
✓  vision PDFs
      • OpenAI chat     → auto-upload + {"type":"file","file_id":…}
      • OpenAI responses→ inline {"type":"input_file","file_data":…}
      • Anthropic / others → inline base-64 data: URI
✓  stop_sequences + streaming trim
✓  toolkit-driven tool calls / schema
"""

# ================= stdlib / deps =================
import json, logging, os, base64, imghdr, re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from litellm import completion                          # chat-completions + Responses
import litellm                                          # for stream_chunk_builder utility
import openai                                           # raw SDK for file upload
import warnings
from core.plugins.tool_registry import ToolRegistry    # NEW import
from .rich_formatter import RichFormatter              # Rich functionality moved here
from core.config import Config                         # For context render mode configuration
from core.token_stats import token_stats               # Clean message queue-based token tracking

warnings.filterwarnings(
    "ignore",
    message="Valid config keys have changed in V2:.*'fields'",
    category=UserWarning,
    module="pydantic._internal._config"
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- media constants ----------------
SUPPORTED_MEDIA_TYPES = {
    "image/jpeg": ["jpeg", "jpg"],
    "image/png":  ["png"],
    "image/gif":  ["gif"],
    "image/webp": ["webp"],
}
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB

# ====================================================================
#  Console Manager - delegates Rich functionality to RichFormatter
# ====================================================================
class ConsoleManager:
    def __init__(self):
        self.formatter = RichFormatter()

    def show(self, role: str, content: str, end: str = "\n"):
        """Display content with role-based coloring"""
        self.formatter.show(role, content, end)

    def status(self, message: str):
        """Display status message"""
        self.formatter.status(message)
    
    def error(self, message: str):
        """Display error message"""
        self.formatter.error(message)

    def format_json_clean(self, json_str: str) -> str:
        """Format JSON string with proper indentation, no colors (for context)"""
        return self.formatter.format_json_clean(json_str)

    def format_json_colored(self, json_str: str) -> str:
        """Format JSON string with Rich syntax highlighting for terminal display"""
        return self.formatter.format_json_colored(json_str)

    def format_json(self, json_str: str, title: str = "JSON") -> str:
        """Format JSON string with proper indentation and nested JSON handling (clean version)"""
        return self.formatter.format_json(json_str, title)

# ====================================================================
#  Tool executor
# ====================================================================
class ToolExecutor:
    def __init__(self, tk: ToolRegistry, ui: ConsoleManager, tool_response_callback=None):
        self.tk, self.ui = tk, ui
        self.tool_response_callback = tool_response_callback

    def execute(self, fn: str, args_json: str) -> str:
        if fn not in self.tk:
            err = f"Tool '{fn}' not found."
            self.ui.error(err)
            return json.dumps({"error": err}, indent=2, ensure_ascii=False)
        try:
            res = self.tk[fn](**json.loads(args_json or "{}"))
            result = json.dumps(res, indent=2, ensure_ascii=False)
            
            # Call the callback to update Tool Loop AST after each tool execution
            if self.tool_response_callback:
                self.tool_response_callback(fn, args_json, result)
            
            return result
        except Exception as e:
            self.ui.error(f"Tool '{fn}' failed: {e}")
            return json.dumps({"error": str(e)}, indent=2, ensure_ascii=False)

# ====================================================================
#  Stream processor (trims stop-seq + handles tool calls)
# ====================================================================
class StreamProcessor:
    def __init__(self, ui: ConsoleManager, stop: Optional[List[str]]):
        self.ui, self.stop = ui, stop or []
        self.last_chunk = ""
        self.usage_info = None  # Track usage information

    def process(self, stream_iter):
        buf = ""
        try:
            for chunk in stream_iter:
                # Capture usage from chunk if available
                if hasattr(chunk, 'usage') and chunk.usage:
                    self.usage_info = chunk.usage
                elif isinstance(chunk, dict) and 'usage' in chunk and chunk['usage']:
                    self.usage_info = chunk['usage']
                
                delta = chunk["choices"][0]["delta"]
                txt = delta.get("content", "")
                if txt:  # Only process non-empty text
                    buf += txt
                    for s in self.stop:
                        if buf.endswith(s):
                            buf = buf[:-len(s)]
                            txt = txt[:-len(s)]
                    if txt:  # Only show non-empty text
                        # Only show new content since last chunk
                        if txt != self.last_chunk:
                            self.ui.show("", txt, end="")
                            self.last_chunk = txt
            # Add final newline after streaming is complete
            self.ui.show("", "")
        except Exception as e:
            # Ensure last_chunk is set for error reporting
            self.last_chunk = buf
            self.ui.error(f"Streaming error: {e}")
            raise
        return buf or "", self.usage_info  # Return tuple: (content, usage_info)


# ====================================================================
#  Tool Call Stream processor (handles both content and tool calls)
# ====================================================================
class ToolCallStreamProcessor:
    def __init__(self, ui: ConsoleManager, stop: Optional[List[str]]):
        self.ui = ui
        self.stop = stop or []
        self.chunks = []
        self.content_buffer = ""
        self.last_chunk = ""
        self.current_tool_calls = {}  # Track active tool calls by index
        self.displayed_tool_names = set()  # Track which tool names we've already shown
        self.usage_info = None  # Track usage information
        
    def process(self, stream_iter):
        import litellm
        
        try:
            for chunk in stream_iter:
                self.chunks.append(chunk)
                
                # Capture usage from chunk if available
                if hasattr(chunk, 'usage') and chunk.usage:
                    self.usage_info = chunk.usage
                elif isinstance(chunk, dict) and 'usage' in chunk and chunk['usage']:
                    self.usage_info = chunk['usage']
                
                # Handle finish_reason - stream is complete
                if chunk.get("choices") and chunk["choices"][0].get("finish_reason"):
                    break
                    
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                
                # Handle regular content streaming
                content = delta.get("content", "")
                if content:
                    self.content_buffer += content
                    # Apply stop sequence filtering
                    display_content = content
                    for s in self.stop:
                        if self.content_buffer.endswith(s):
                            self.content_buffer = self.content_buffer[:-len(s)]
                            display_content = display_content[:-len(s)]
                    
                    if display_content:
                        self.ui.show("", display_content, end="")
                        self.last_chunk = display_content
                
                # Handle tool call streaming
                tool_calls = delta.get("tool_calls", [])
                if tool_calls:
                    for tc_delta in tool_calls:
                        index = tc_delta.get("index", 0)
                        
                        # Initialize tool call tracking for this index
                        if index not in self.current_tool_calls:
                            self.current_tool_calls[index] = {
                                "id": None,
                                "name": None,
                                "arguments": ""
                            }
                        
                        # Get tool call ID and name (usually in first chunk for this tool)
                        if tc_delta.get("id"):
                            self.current_tool_calls[index]["id"] = tc_delta["id"]
                        
                        if tc_delta.get("function", {}).get("name"):
                            tool_name = tc_delta["function"]["name"]
                            self.current_tool_calls[index]["name"] = tool_name
                            
                            # Display tool call initiation (only once per tool)
                            if tool_name not in self.displayed_tool_names:
                                self.ui.show("", f"\n🔧 Calling tool: {tool_name}")
                                self.displayed_tool_names.add(tool_name)
                        
                        # Accumulate function arguments
                        if tc_delta.get("function", {}).get("arguments"):
                            args_chunk = tc_delta["function"]["arguments"]
                            self.current_tool_calls[index]["arguments"] += args_chunk
                            
                            # Optionally stream the arguments as they come in
                            # (You might want to disable this if the JSON becomes messy)
                            # self.ui.show("", args_chunk, end="")
                            
            # Add final newline if we were streaming content
            if self.content_buffer:
                self.ui.show("", "")
                
        except Exception as e:
            self.last_chunk = self.content_buffer
            self.ui.error(f"Tool call streaming error: {e}")
            raise
            
        # Reconstruct the complete message using LiteLLM's utility
        if self.chunks:
            try:
                final_message = litellm.stream_chunk_builder(self.chunks)
                # Capture usage from reconstructed response if available
                if hasattr(final_message, 'usage') and final_message.usage and not self.usage_info:
                    self.usage_info = final_message.usage
                return final_message, self.usage_info
            except Exception as e:
                # Check if this is a JSON parse error that we can provide better info for
                error_msg = str(e).lower()
                if "json" in error_msg and ("parse" in error_msg or "decode" in error_msg):
                    self.ui.error(f"JSON parsing error in stream reconstruction: {e}")
                    self.ui.show("", f"[yellow]Chunks received: {len(self.chunks)}[/yellow]")
                    # Log the problematic chunks for debugging
                    for i, chunk in enumerate(self.chunks[-3:]):  # Show last 3 chunks
                        self.ui.show("", f"[dim]Chunk {len(self.chunks)-3+i}: {str(chunk)[:100]}...[/dim]")
                else:
                    self.ui.error(f"Error reconstructing message: {e}")
                
                # Fallback: create basic message structure
                tool_calls_list = []
                for idx, tc_info in self.current_tool_calls.items():
                    if tc_info["id"] and tc_info["name"]:
                        tool_calls_list.append({
                            "id": tc_info["id"],
                            "type": "function",
                            "function": {
                                "name": tc_info["name"],
                                "arguments": tc_info["arguments"]
                            }
                        })
                
                # Create a mock response structure
                class MockChoice:
                    def __init__(self, content, tool_calls):
                        self.message = type('obj', (object,), {
                            'content': content,
                            'tool_calls': tool_calls if tool_calls else None
                        })()
                
                class MockResponse:
                    def __init__(self, content, tool_calls):
                        self.choices = [MockChoice(content, tool_calls)]
                
                return MockResponse(self.content_buffer, tool_calls_list), self.usage_info
        
        # If no chunks, return empty content
        empty_response = type('obj', (object,), {
            'choices': [type('obj', (object,), {
                'message': type('obj', (object,), {
                    'content': self.content_buffer,
                    'tool_calls': None
                })()
            })()]
        })()
        return empty_response, self.usage_info

# ====================================================================
#  Media helpers
# ====================================================================
def _validate_image(path: Path) -> tuple[str, bytes]:
    if path.stat().st_size > MAX_IMAGE_SIZE:
        raise ValueError("Image too large (>20 MB)")
    data = path.read_bytes()
    fmt = imghdr.what(None, data)
    if not fmt:
        raise ValueError("Unknown image format")
    mime = next((m for m, fmts in SUPPORTED_MEDIA_TYPES.items() if fmt in fmts), None)
    if not mime:
        raise ValueError(f"Unsupported format: {fmt}")
    return mime, data


def _upload_pdf_openai(pdf_path: Path, api_key: str) -> str:
    openai.api_key = api_key
    return openai.files.create(file=open(pdf_path, "rb"), purpose="vision").id


# ---------- Responses-API helper ------------------------------------
def _build_responses_blocks(prompt: str, media: List[str]) -> List[Dict[str, Any]]:
    blocks = [{"type": "input_text", "text": prompt}]
    for m in media:
        p = Path(m)
        data = base64.b64encode(p.read_bytes()).decode()
        blocks.insert(0, {
            "type": "input_file",
            "filename": p.name,
            "file_data": f"data:application/pdf;base64,{data}"
        })
    return [{"role": "user", "content": blocks}]

# ====================================================================
#  Embed media for Chat-Completions path
# ====================================================================
def _embed_media(item: str | dict, provider: str, api_key: str) -> Dict[str, Any]:
    # already-prepared dict (e.g. from previous call)
    if isinstance(item, dict) and "file_id" in item:
        return {"type": "file", "file": item}

    p = Path(item)
    ext = p.suffix.lower()
    if not p.exists():
        raise FileNotFoundError(item)

    # ---- PDF --------------------------------------------------------
    if (ext == ".pdf"):
        if provider == "openai":
            fid = _upload_pdf_openai(p, api_key)
            return {"type": "file", "file_id": fid,
                    "mime_type": "application/pdf"}
        data_b64 = base64.b64encode(p.read_bytes()).decode()
        return {"type": "image_url",
                "image_url": {"url": f"data:application/pdf;base64,{data_b64}"}}

    # ---- image ------------------------------------------------------
    mime, data = _validate_image(p)
    return {"type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode()}"}}

# ====================================================================
#  Token calculation helpers for detailed breakdown
# ====================================================================
def estimate_schema_tokens(tools: Optional[List[Dict[str, Any]]]) -> int:
    """Estimate tokens used by tool schemas in the request."""
    if not tools:
        return 0
    
    # Rough estimation: ~4 characters per token for JSON schema
    schema_text = json.dumps(tools, separators=(',', ':'))
    return len(schema_text) // 4

def calculate_detailed_tokens_enhanced(messages: List[Dict[str, Any]], 
                                      tools: Optional[List[Dict[str, Any]]] = None,
                                      calculated_prompt_tokens: int = 0,
                                      mcp_tool_tokens: int = 0,
                                      fractalic_tool_tokens: int = 0) -> tuple:
    """
    Calculate enhanced detailed token breakdown:
    - input_context_tokens: Message content, conversation history
    - input_mcp_schema_tokens: MCP tool schemas specifically  
    - input_fractalic_schema_tokens: Fractalic tool schemas specifically
    - total_input_tokens: Sum of context + all schema tokens
    """
    # If specific tool token counts not provided, estimate from tools
    if tools and (mcp_tool_tokens == 0 and fractalic_tool_tokens == 0):
        # Estimate tool schema tokens (rough approximation)
        tools_text = json.dumps(tools, separators=(',', ':'))
        estimated_tool_tokens = len(tools_text) // 4
        
        # For now, categorize all as Fractalic tools since we don't have MCP detection here
        # In a real implementation, you'd need to distinguish MCP vs Fractalic tools
        fractalic_tool_tokens = estimated_tool_tokens
        mcp_tool_tokens = 0
    
    total_schema_tokens = mcp_tool_tokens + fractalic_tool_tokens
    
    if calculated_prompt_tokens > 0:
        # Use provided calculation, subtract estimated schema tokens for context
        context_tokens = max(0, calculated_prompt_tokens - total_schema_tokens)
        total_input = calculated_prompt_tokens
    else:
        # Fallback: estimate context tokens from messages
        messages_text = json.dumps(messages, separators=(',', ':'))
        context_tokens = len(messages_text) // 4  # Rough estimation
        total_input = context_tokens + total_schema_tokens
    
    return context_tokens, mcp_tool_tokens, fractalic_tool_tokens, total_input

# ====================================================================
#  Main LiteLLM client
# ====================================================================
@dataclass
class liteclient:
    api_key: str
    model: str # = "openai/gpt-4o-mini"
    temperature: float = 0.1
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    system_prompt: str = "You are a helpful assistant."
    max_tool_turns: int = 300  # Increased from 30 to 300 for long-running workflows
    settings: Optional[Dict[str, Any]] = field(default=None, repr=False)
    tools_dir: str | Path = "tools"  # NEW
    mcp_servers: List[str] = field(default_factory=list)  # NEW
    registry: ToolRegistry = field(init=False)  # NEW

    def __post_init__(self):
        from core.config import Config
        # print(f"[DEBUG] Config.TOML_SETTINGS at liteclient init: {Config.TOML_SETTINGS}")
        # Use config value if mcp_servers is empty
        if not self.mcp_servers:
            mcp_from_config = Config.TOML_SETTINGS.get("mcp", {}).get("mcpServers", [])
            if mcp_from_config:
                # print(f"[DEBUG] Overriding mcp_servers with config value: {mcp_from_config}")
                self.mcp_servers = mcp_from_config
        if self.settings:
            s = self.settings
            self.model = s.get("model", self.model)
            self.temperature = s.get("temperature", self.temperature)
            self.top_p = s.get("top_p", s.get("topP", self.top_p))
            self.max_tokens = s.get("max_tokens",
                                     s.get("max_completion_tokens", self.max_tokens))
            self.system_prompt = s.get("system_prompt",
                                       s.get("systemPrompt", self.system_prompt))
            self.max_tool_turns = s.get("max_tool_turns", self.max_tool_turns)

        self.registry = ToolRegistry(self.tools_dir, self.mcp_servers)  # NEW
        # print(f"[DEBUG] ToolRegistry tools_dir: {self.registry.tools_dir.resolve()}")
        # print(f"[DEBUG] ToolRegistry MCP servers: {self.mcp_servers}")
        # Debug print: show discovered tools and schema
        # print("[DEBUG] Discovered tools:", list(self.registry.keys()))
        # print("[DEBUG] Tool schema:", json.dumps(self.registry.generate_schema(), indent=2))
        self.ui = ConsoleManager()
        self.exec = ToolExecutor(self.registry, self.ui, self._on_tool_response)  # registry replaces toolkit
        self.schema = self.registry.generate_schema()  # registry replaces toolkit
        self.tool_loop_ast = None  # Will be set by execution context

    def _on_tool_response(self, tool_name: str, args_json: str, result: str):
        """Callback to update Tool Loop AST after each tool response"""
        # print(f"[DEBUG] _on_tool_response called: tool={tool_name}, result_len={len(result)}")
        if self.tool_loop_ast is not None and hasattr(self, 'registry'):
            # Import here to avoid circular imports
            from core.operations.llm_op import process_tool_calls
            
            # Create a tool message from the response
            tool_message = {
                'role': 'tool',
                'content': result,
                'name': tool_name
            }
            
            # print(f"[DEBUG] Processing tool message for {tool_name}")
            # Process this single tool response
            new_tool_ast = process_tool_calls(None, [tool_message])
            if new_tool_ast and new_tool_ast.parser.nodes:
                # print(f"[DEBUG] New tool AST has {len(new_tool_ast.parser.nodes)} nodes")
                # Merge with existing Tool Loop AST
                if self.tool_loop_ast.parser.nodes:
                    # Combine existing and new tool content
                    combined_nodes = {}
                    combined_nodes.update(self.tool_loop_ast.parser.nodes)
                    combined_nodes.update(new_tool_ast.parser.nodes)
                    self.tool_loop_ast.parser.nodes = combined_nodes
                    
                    # Update head and tail
                    all_nodes = list(combined_nodes.values())
                    self.tool_loop_ast.parser.head = all_nodes[0] if all_nodes else None
                    self.tool_loop_ast.parser.tail = all_nodes[-1] if all_nodes else None
                    # print(f"[DEBUG] Merged Tool Loop AST now has {len(combined_nodes)} nodes")
                else:
                    self.tool_loop_ast = new_tool_ast
                    # print(f"[DEBUG] Set new Tool Loop AST with {len(new_tool_ast.parser.nodes)} nodes")
                
                # Update the registry with the updated Tool Loop AST
                if hasattr(self.registry, '_tool_loop_ast'):
                    self.registry._tool_loop_ast = self.tool_loop_ast
                    # print(f"[DEBUG] Updated registry Tool Loop AST")
                    # Print node IDs for debugging
                    for node_id, node in self.tool_loop_ast.parser.nodes.items():
                        # print(f"[DEBUG] Tool Loop AST node: {node_id} (id: {getattr(node, 'id', 'None')})")
                        pass
            else:
                # print(f"[DEBUG] No new tool AST created from {tool_name} response")
                pass
        else:
            # print(f"[DEBUG] Tool Loop AST not available or registry missing")
            pass

    # -----------------------------------------------------------------
    def _provider(self, op: Dict[str, Any]) -> str:
        return "openai"

    # -----------------------------------------------------------------
    def _record_llm_operation_unified(self, messages, tools, completion_tokens, prompt_tokens=0, operation_params=None, tool_calls=None):
        """
        UNIFIED method to handle ALL LLM token tracking.
        Replaces the 4+ different calculation paths with single consistent approach.
        
        Args:
            messages: List of conversation messages
            tools: List of available tool schemas
            completion_tokens: LLM response tokens
            prompt_tokens: Input tokens from LLM API (fallback)
            operation_params: Additional operation parameters
            tool_calls: Current tool calls (if any)
        
        Returns:
            str: Formatted display string for token usage
        """
        op = operation_params or {}
        
        # Step 1: Calculate tokens using ONE method (always enhanced calculation)
        # For token calculation, we need to exclude tool responses from messages to get accurate prompt tokens
        messages_for_calculation = []
        for msg in messages:
            # Skip tool response messages for prompt token calculation
            if msg.get("role") != "tool":
                # Create a clean copy of the message without tool_calls objects that might not be JSON serializable
                clean_msg = {
                    "role": msg.get("role"),
                    "content": msg.get("content")
                }
                # Only add tool_calls if they exist and are serializable
                if msg.get("tool_calls") and msg.get("role") == "assistant":
                    # Convert tool calls to a simple count for token calculation purposes
                    clean_msg["tool_calls"] = [{"function": {"name": "placeholder"}}] * len(msg.get("tool_calls", []))
                messages_for_calculation.append(clean_msg)
        
        context_tokens, mcp_tokens, fractalic_tokens, total_input = calculate_detailed_tokens_enhanced(
            messages=messages_for_calculation,
            tools=tools,
            calculated_prompt_tokens=prompt_tokens
        )
        
        # Step 2: Generate unique operation ID
        import time
        unique_id = f"llm_{int(time.time()*1000)}_unified_{op.get('model', self.model)}"
        
        # Step 3: Calculate total tokens
        total_tokens = total_input + completion_tokens
        
        # Step 4: Count tool calls from parameter (safer than parsing messages)
        tool_calls_count = 0
        if tool_calls:
            tool_calls_count = len(tool_calls) if isinstance(tool_calls, list) else 1
        
        # Step 5: Send to queue using ONE format (always same structure)
        token_stats.send_usage(
            operation_id=unique_id,
            model=op.get("model", self.model),
            input_tokens=total_input,
            input_context_tokens=context_tokens,
            input_schema_tokens=mcp_tokens + fractalic_tokens,
            input_mcp_schema_tokens=mcp_tokens,
            input_fractalic_schema_tokens=fractalic_tokens,
            output_tokens=completion_tokens,
            total_tokens=total_tokens,
            tool_calls_count=tool_calls_count,
            tools_available_count=len(tools),
            operation_type="llm_turn",
            source_file="openai_client.py"
        )
        
        # Step 6: Generate display using SAME values sent to queue
        session_turn = len(token_stats.messages)
        cumulative_total = token_stats.get_cumulative_total()
        
        display_text = (
            f"[TOKENS] Input: {total_input} "
            f"(Context: {context_tokens}, "
            f"MCP: {mcp_tokens}, "
            f"Fractalic: {fractalic_tokens}), "
            f"Output: {completion_tokens}, "
            f"Turn: {total_tokens}, "
            f"Cumulative: {cumulative_total}, "
            f"Tools: {len(tools)}, "
            f"Turn: {session_turn}"
        )
        
        # DEBUG: Show what we sent vs what we display (should always match)
        # print(f"[DEBUG UNIFIED] Operation: {unique_id}")
        # print(f"[DEBUG UNIFIED] Sent to queue: input={total_input}, output={completion_tokens}, total={total_tokens}")
        # print(f"[DEBUG UNIFIED] Display: {display_text}")
        # print(f"[DEBUG UNIFIED] Cumulative after: {token_stats.get_cumulative_total()}")
        
        return display_text
    
    # -----------------------------------------------------------------
    def _generate_operation_id(self, operation_type="llm"):
        """Generate unique operation ID for token tracking"""
        import time
        return f"{operation_type}_{int(time.time()*1000)}_{self.model.replace('/', '_')}"

    # -----------------------------------------------------------------
    class LLMCallException(Exception):
        def __init__(self, message, partial_result=None):
            super().__init__(message)
            self.partial_result = partial_result

    def llm_call(
        self,
        prompt_text: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        operation_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:

        op = operation_params or {}
        provider = self._provider(op)
        
        # Set OpenRouter-specific environment variables for custom app name display
        model_name = op.get("model", self.model)
        if "openrouter/" in model_name.lower():
            # Set OpenRouter app name so it shows as "Fractalic" instead of "litellm" in dashboard
            os.environ["OR_APP_NAME"] = "Fractalic"
            # Optionally set site URL for better identification
            os.environ["OR_SITE_URL"] = "https://fractalic.ai"

        # ── auto-upgrade to Responses API if any supplied media is a PDF ──
        wants_pdf = any(str(m).lower().endswith(".pdf") for m in op.get("media", []))
        use_resp = bool(op.get("use_responses_api") or
                        (provider == "openai" and wants_pdf))

        # ------------ Responses-API branch ----------------------------
        if use_resp and provider == "openai":
            from openai import OpenAI            # typed client
            client = OpenAI(api_key=self.api_key)

            #  1) bare model name (fixes "provider/model" prefixes)
            plain_model = op.get("model", self.model)
            if "/" in plain_model:
                plain_model = plain_model.split("/", 1)[1]

            #  2) correct payload
            payload = {
                "model": plain_model,
                "input": _build_responses_blocks(
                             prompt_text or "", op.get("media", [])),
                "text":   {"format": {"type": "text"}},   # <-- key line
                "temperature": op.get("temperature", self.temperature),
                "top_p":       op.get("top_p",        self.top_p),
                "max_output_tokens":
                    op.get("max_tokens", self.max_tokens) or 2048,
            }

            self.ui.status("[Responses API] sending request …")
            resp = client.responses.create(**payload)
            content = resp.output_text      # correct property per SDK documentation
            self.ui.show("", content)
            return {"text": content}             # ← early exit
        # ----------------------------------------------------------------

        # ------------ Chat-Completions branch -------------------------


        params = dict(
            model=op.get("model", self.model),
            temperature=op.get("temperature", self.temperature),
            top_p=op.get("top_p", self.top_p),
            max_tokens=op.get("max_tokens", self.max_tokens),
            stop=op.get("stop_sequences"),
            api_key= self.api_key
        )

        # Add stream_options only for providers that support it (OpenAI, Anthropic)
        # Vertex AI/Gemini doesn't support stream_options and will fail with JSON parsing errors
        model_name = op.get("model", self.model).lower()
        if not any(provider in model_name for provider in ["vertex", "gemini", "google"]):
            params["stream_options"] = {"include_usage": True}  # Enable usage tracking in streams

        # Remove or fix unsupported params for O-series models (e.g., o4-mini)
        model_name = op.get("model", self.model)
        if model_name in ["o4-mini", "openai/o4-mini"]:
            params.pop("top_p", None)
            # O-series only supports temperature=1
            if "temperature" in params and params["temperature"] != 1:
                params["temperature"] = 1

        # Handle tools parameter
        tools_param = op.get("tools", "none")
        has_tools = tools_param != "none"
        
        if tools_param == "none":
            # No tools available
            params["stream"] = True
        elif tools_param == "all":
            # Use all tools  
            params["tools"] = self.schema.copy()
            params["stream"] = True  # Enable streaming for tool calls too
        elif isinstance(tools_param, str) and tools_param.startswith("mcp/"):
            # Single MCP server filter
            mcp_server_name = tools_param[4:]  # Remove the mcp/ prefix
            filtered_schema = []
            
            for tool in self.schema:
                tool_name = tool["function"]["name"]
                
                # Check if this tool has MCP metadata and matches the server name
                if hasattr(self, 'registry') and self.registry:
                    for manifest in self.registry._manifests:
                        if (manifest.get("name") == tool_name and 
                            manifest.get("_service", "").lower() == mcp_server_name.lower()):
                            filtered_schema.append(tool)
                            break
            
            params["tools"] = filtered_schema
            params["stream"] = True  # Enable streaming for tool calls too
        elif isinstance(tools_param, list):
            # Filter tools based on the provided list
            filtered_schema = []
            
            for tool in self.schema:
                tool_name = tool["function"]["name"]
                should_include = False
                
                for filter_item in tools_param:
                    if isinstance(filter_item, str):
                        if filter_item.startswith("mcp/"):
                            # MCP server filter
                            mcp_server_name = filter_item[4:]  # Remove the mcp/ prefix
                            
                            # Check if this tool has MCP metadata and matches the server name
                            if hasattr(self, 'registry') and self.registry:
                                for manifest in self.registry._manifests:
                                    if (manifest.get("name") == tool_name and 
                                        manifest.get("_service", "").lower() == mcp_server_name.lower()):
                                        should_include = True
                                        break
                        else:
                            # Regular tool name filter
                            if tool_name == filter_item:
                                should_include = True
                
                if should_include:
                    filtered_schema.append(tool)
            params["tools"] = filtered_schema
            params["stream"] = True  # Enable streaming for tool calls too
        elif isinstance(tools_param, str):
            # Single tool name filter
            filtered_schema = [
                tool for tool in self.schema 
                if tool["function"]["name"] == tools_param
            ]
            params["tools"] = filtered_schema
            params["stream"] = True  # Enable streaming for tool calls too

        # ----- build messages -----
        if messages:
            hist = list(messages)
            if hist[0].get("role") != "system":
                hist.insert(0, {"role": "system", "content": self.system_prompt})
            if op.get("media"):
                for m in op["media"]:
                    for msg in hist:
                        if msg.get("role") == "user":
                            if isinstance(msg["content"], str):
                                msg["content"] = [{"type": "text",
                                                   "text": msg["content"]}]
                            msg["content"].insert(0,
                                                  _embed_media(m, provider, self.api_key))
                            break
            self.ui.show("user", f"[{len(hist)} msgs]")
        else:
            blocks = [_embed_media(m, provider, self.api_key)
                      for m in op.get("media", [])]
            blocks.append({"type": "text", "text": prompt_text})
            hist = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": blocks},
            ]
            self.ui.show("user", prompt_text or "[no prompt]")

        params["messages"] = hist

        # ----- conversation loop -----
        convo = []
        # Track cumulative usage across multiple turns
        cumulative_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "tool_calls_count": 0,
            "turns_count": 0
        }
        # Use operation-specific tools-turns-max if provided, otherwise use instance default
        max_turns = op.get("tools-turns-max", self.max_tool_turns)
        try:
            for turn_count in range(max_turns):
                # Always use streaming now, but with different processors
                try:
                    # Add timeout for streaming calls to prevent hanging
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError("Streaming call timed out")
                    
                    # Set a 300 second (5 minute) timeout for streaming
                    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(300)
                    
                    try:
                        rsp = completion(**params)
                        
                        # Use appropriate stream processor based on whether tools are available
                        usage_info = None
                        if has_tools:
                            # Use tool call stream processor for better tool call handling
                            tcsp = ToolCallStreamProcessor(self.ui, params["stop"])
                            stream_response, usage_info = tcsp.process(rsp)
                            
                            # Extract content and tool calls from the reconstructed message
                            if hasattr(stream_response, 'choices') and stream_response.choices:
                                msg = stream_response.choices[0].message
                                content = msg.content or ""
                                tool_calls = msg.tool_calls or []
                            else:
                                content = ""
                                tool_calls = []
                        else:
                            # Use simple stream processor for content-only responses
                            sp = StreamProcessor(self.ui, params["stop"])
                            content, usage_info = sp.process(rsp)
                            tool_calls = []
                            
                    finally:
                        signal.alarm(0)  # Cancel the alarm
                        signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
                        
                except TimeoutError as e:
                    self.ui.error("Streaming call timed out after 5 minutes")
                    error_partial = ""
                    if 'tcsp' in locals():
                        error_partial = tcsp.last_chunk
                    elif 'sp' in locals():
                        error_partial = sp.last_chunk
                    raise self.LLMCallException(f"Streaming timeout: {e}", partial_result=error_partial) from e
                except Exception as e:
                    # On streaming error, propagate buffer so far and exit immediately
                    self.ui.error(f"Streaming error: {e}")
                    error_partial = ""
                    if 'tcsp' in locals():
                        error_partial = tcsp.last_chunk
                    elif 'sp' in locals():
                        error_partial = sp.last_chunk
                    raise self.LLMCallException(f"Streaming error: {e}", partial_result=error_partial) from e

                # Don't print assistant content since it's already streamed
                convo.append(content or "")
                hist.append({"role": "assistant",
                             "content": content,
                             "tool_calls": tool_calls or None})

                # Accumulate usage data
                if usage_info:
                    cumulative_usage["prompt_tokens"] += usage_info.get("prompt_tokens", 0)
                    cumulative_usage["completion_tokens"] += usage_info.get("completion_tokens", 0)
                    cumulative_usage["total_tokens"] += usage_info.get("total_tokens", 0)
                    cumulative_usage["turns_count"] += 1
                    
                    # Token usage will be sent to queue later with calculated values
                
                if tool_calls:
                    cumulative_usage["tool_calls_count"] += len(tool_calls)

                if not tool_calls:
                    # LLM finished conversation naturally - show token usage for final turn
                    if usage_info:
                        prompt_tokens = usage_info.get('prompt_tokens', 0)
                        completion_tokens = usage_info.get('completion_tokens', 0)
                        
                        # UNIFIED APPROACH: Replace complex enhanced calculation with single method
                        display_text = self._record_llm_operation_unified(
                            messages=hist,  # Use actual conversation history
                            tools=params.get("tools", []),
                            completion_tokens=completion_tokens,
                            prompt_tokens=prompt_tokens,
                            operation_params=op,
                            tool_calls=None  # No tool calls when conversation finishes naturally
                        )
                        
                        # Display unified result
                        self.ui.show("", f"\n{display_text}")
                    break

                # ---- execute tool calls ----
                for tc in tool_calls:
                    args = tc["function"]["arguments"]
                    try:
                        parsed_args = json.loads(args) if args else {}
                        
                        # Format args for display and context
                        if args:
                            colored_args = self.ui.format_json_colored(args)
                            clean_args = self.ui.format_json_clean(args)
                        else:
                            colored_args = clean_args = "{}"
                        
                        # Display with colors
                        call_log_display = (f"> TOOL CALL, id: {tc['id']}\n"
                                          f"tool: {tc['function']['name']}\n"
                                          f"args:\n{colored_args}")
                        self.ui.show("", call_log_display)
                        
                        # Context with clean text
                        call_log_context = (f"> TOOL CALL, id: {tc['id']}\n"
                                          f"tool: {tc['function']['name']}\n"
                                          f"args:\n{clean_args}")
                        convo.append(call_log_context)

                        res = self.exec.execute(tc["function"]["name"], args)
                        
                        # Format response for display and context
                        if res and (res.strip().startswith(('{', '['))):
                            colored_response = self.ui.format_json_colored(res)
                            clean_response = self.ui.format_json_clean(res)
                        else:
                            colored_response = clean_response = res or ""
                        
                        # Display with colors
                        resp_log_display = (f"> TOOL RESPONSE, id: {tc['id']}\n"
                                          f"response:\n{colored_response}")
                        self.ui.show("", resp_log_display)
                        
                        # Context with clean text - special handling for fractalic_run
                        if tc["function"]["name"] == "fractalic_run" and res and res.strip().startswith('{'):
                            # Check if this is a fractalic_run response with return_content
                            try:
                                response_data = json.loads(res)
                                if isinstance(response_data, dict) and "return_content" in response_data:
                                    # Check context render mode to determine behavior
                                    context_render_mode = getattr(Config, 'CONTEXT_RENDER_MODE', 'direct')
                                    
                                    if context_render_mode == 'direct':
                                        # "direct" mode: Replace JSON with marker and render markdown directly
                                        resp_log_context = (f"> TOOL RESPONSE, id: {tc['id']}\n"
                                                          f"response:\n"
                                                          f'content: "_IN_CONTEXT_BELOW_"')
                                        convo.append(resp_log_context)
                                        
                                        # Also append the actual markdown content to context
                                        return_content = response_data["return_content"]
                                        # Handle escaped newlines in JSON strings
                                        if '\\n' in return_content:
                                            return_content = return_content.replace('\\n', '\n')
                                        if '\\r' in return_content:
                                            return_content = return_content.replace('\\r', '\r')
                                        if '\\t' in return_content:
                                            return_content = return_content.replace('\\t', '\t')
                                        convo.append(f"\n{return_content}\n")
                                    else:  # context_render_mode == 'json'
                                        # "json" mode: Keep actual JSON values, no direct markdown rendering
                                        resp_log_context = (f"> TOOL RESPONSE, id: {tc['id']}\n"
                                                          f"response:\n{clean_response}")
                                        convo.append(resp_log_context)
                                else:
                                    # Normal JSON response for fractalic_run without return_content
                                    resp_log_context = (f"> TOOL RESPONSE, id: {tc['id']}\n"
                                                      f"response:\n{clean_response}")
                                    convo.append(resp_log_context)
                            except json.JSONDecodeError:
                                # Fallback to normal response if JSON parsing fails
                                resp_log_context = (f"> TOOL RESPONSE, id: {tc['id']}\n"
                                                  f"response:\n{clean_response}")
                                convo.append(resp_log_context)
                        else:
                            # Normal tool response (not fractalic_run or not JSON)
                            resp_log_context = (f"> TOOL RESPONSE, id: {tc['id']}\n"
                                              f"response:\n{clean_response}")
                            convo.append(resp_log_context)

                        hist.append({"role": "tool",
                                     "tool_call_id": tc["id"],
                                     "name": tc["function"]["name"],
                                     "content": res})
                    except json.JSONDecodeError:
                        error_msg = f"Invalid JSON arguments for tool {tc['function']['name']}: {args}"
                        self.ui.error(error_msg)
                        self.ui.show("status", f"[red]Tool conversation terminated due to JSON parsing error[/red]")
                        convo.append(error_msg)
                        hist.append({"role": "tool",
                                     "tool_call_id": tc["id"],
                                     "name": tc["function"]["name"],
                                     "content": json.dumps({"error": error_msg}, indent=2)})
                        # Break the tool call loop and return current conversation
                        return {"text": "\n\n".join(convo), "messages": hist, "usage": cumulative_usage}
                
                # Display token usage after tool execution completes
                if usage_info:
                    prompt_tokens = usage_info.get('prompt_tokens', 0)
                    completion_tokens = usage_info.get('completion_tokens', 0)
                    total_tokens = usage_info.get('total_tokens', 0)
                    
                    # UNIFIED APPROACH: Replace streaming paths with single method
                    display_text = self._record_llm_operation_unified(
                        messages=hist,  # Use actual conversation history
                        tools=params.get("tools", []),
                        completion_tokens=completion_tokens,
                        prompt_tokens=prompt_tokens,
                        operation_params=op,
                        tool_calls=tool_calls  # Pass current tool calls
                    )
                    
                    # Display unified result
                    self.ui.show("", f"\n{display_text}")
                
                params["messages"] = hist
            
            # Check if we exited due to max turns limit
            if turn_count == max_turns - 1:
                self.ui.show("status", f"[yellow]Tool conversation reached maximum limit of {max_turns} turns[/yellow]")
                
        except self.LLMCallException as e:
            # Propagate with partial result
            raise
        except Exception as e:
            # Catch-all: propagate convo so far
            raise self.LLMCallException(f"Unexpected LLM error: {e}", partial_result="\n\n".join(convo)) from e
        
        return {"text": "\n\n".join(convo), "messages": hist, "usage": cumulative_usage}

# -------- legacy alias --------
openaiclient = liteclient
