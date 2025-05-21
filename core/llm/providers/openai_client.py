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
import json, ast, inspect, logging, os, base64, imghdr
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

from rich.console import Console
from litellm import completion                          # chat-completions + Responses
import openai                                           # raw SDK for file upload
import requests
import warnings
from core.plugins.tool_registry import ToolRegistry  # NEW import

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
#  Console
# ====================================================================
class ConsoleManager:
    def __init__(self):
        self.c = Console()

    def show(self, role: str, content: str, end: str = "\n"):
        colours = {"user": "cyan", "assistant": "green",
                   "error": "red", "status": "dim"}
        if role in colours:
            self.c.print(f"[{colours[role]}]{role.upper()}:[/] {content}", end=end)
        else:
            self.c.print(content, highlight=False, end=end)

    def status(self, m): self.show("status", m)
    def error(self, m):  self.show("error",  m)

# ====================================================================
#  Tool executor
# ====================================================================
class ToolExecutor:
    def __init__(self, tk: ToolRegistry, ui: ConsoleManager):
        self.tk, self.ui = tk, ui

    def execute(self, fn: str, args_json: str) -> str:
        if fn not in self.tk:
            err = f"Tool '{fn}' not found."
            self.ui.error(err)
            return json.dumps({"error": err}, indent=2, ensure_ascii=False)
        try:
            res = self.tk[fn](**json.loads(args_json or "{}"))
            return json.dumps(res, indent=2, ensure_ascii=False)
        except Exception as e:
            self.ui.error(f"Tool '{fn}' failed: {e}")
            return json.dumps({"error": str(e)}, indent=2, ensure_ascii=False)

# ====================================================================
#  Stream processor (trims stop-seq)
# ====================================================================
class StreamProcessor:
    def __init__(self, ui: ConsoleManager, stop: Optional[List[str]]):
        self.ui, self.stop = ui, stop or []
        self.last_chunk = ""

    def process(self, stream_iter):
        buf = ""
        for chunk in stream_iter:
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
        return buf or ""  # Ensure we always return a string, even if empty

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
    max_tool_turns: int = 30
    settings: Optional[Dict[str, Any]] = field(default=None, repr=False)
    tools_dir: str | Path = "tools"  # NEW
    mcp_servers: List[str] = field(default_factory=list)  # NEW
    registry: ToolRegistry = field(init=False)  # NEW

    def __post_init__(self):
        from core.config import Config
        print(f"[DEBUG] Config.TOML_SETTINGS at liteclient init: {Config.TOML_SETTINGS}")
        # Use config value if mcp_servers is empty
        if not self.mcp_servers:
            mcp_from_config = Config.TOML_SETTINGS.get("mcp", {}).get("mcpServers", [])
            if mcp_from_config:
                print(f"[DEBUG] Overriding mcp_servers with config value: {mcp_from_config}")
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
        print(f"[DEBUG] ToolRegistry tools_dir: {self.registry.tools_dir.resolve()}")
        print(f"[DEBUG] ToolRegistry MCP servers: {self.mcp_servers}")
        # Debug print: show discovered tools and schema
        print("[DEBUG] Discovered tools:", list(self.registry.keys()))
        print("[DEBUG] Tool schema:", json.dumps(self.registry.generate_schema(), indent=2))
        self.ui = ConsoleManager()
        self.exec = ToolExecutor(self.registry, self.ui)  # registry replaces toolkit
        self.schema = self.registry.generate_schema()  # registry replaces toolkit

    # -----------------------------------------------------------------
    def _provider(self, op: Dict[str, Any]) -> str:
        return "openai"

    # -----------------------------------------------------------------
    class LLMCallException(Exception):
        def __init__(self, message, partial_result=None):
            super().__init__(message)
            self.partial_result = partial_result

    def llm_call(
        self,
        prompt_text: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        operation_params: Optional[Dict[str, Any]] = None,
        *,
        stream: bool = False
    ) -> Dict[str, Any]:

        op = operation_params or {}
        provider = self._provider(op)

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
            stream=stream,
            api_key= self.api_key
        )

        # Handle tools parameter
        tools_param = op.get("tools", "none")
        if tools_param == "none":
            # No tools, ensure streaming is enabled
            params["stream"] = True
        elif tools_param == "all":
            # Use all tools - copy the entire schema
            params["tools"] = self.schema.copy()
        elif isinstance(tools_param, list):
            # Filter tools based on the provided list - create new list with matching tools
            filtered_schema = [
                tool for tool in self.schema 
                if tool["function"]["name"] in tools_param
            ]
            params["tools"] = filtered_schema

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
        try:
            for _ in range(self.max_tool_turns):
                if stream:
                    try:
                        sp = StreamProcessor(self.ui, params["stop"])
                        rsp = completion(stream=True, **params)
                        content = sp.process(rsp)
                    except Exception as e:
                        # On streaming error, propagate buffer so far
                        raise self.LLMCallException(f"Streaming error: {e}", partial_result=sp.last_chunk) from e
                    tool_calls = []
                else:
                    try:
                        rsp = completion(**params)
                        if hasattr(rsp, "choices"):  # Regular response
                            msg = rsp["choices"][0]["message"]
                            content = msg.get("content", "")
                            tool_calls = msg.get("tool_calls", [])
                        else:  # Streaming response
                            sp = StreamProcessor(self.ui, params["stop"])
                            content = sp.process(rsp)
                            tool_calls = []
                    except Exception as e:
                        # On error, propagate convo so far
                        raise self.LLMCallException(f"Completion error: {e}", partial_result="\n\n".join(convo)) from e

                self.ui.show("assistant", content or "[tool call]")
                convo.append(content or "")
                hist.append({"role": "assistant",
                             "content": content,
                             "tool_calls": tool_calls or None})

                if not tool_calls:
                    break

                # ---- execute tool calls ----
                for tc in tool_calls:
                    args = tc["function"]["arguments"]
                    try:
                        parsed_args = json.loads(args) if args else {}
                        call_log = (f"> TOOL CALL, id: {tc['id']}\n"
                                    f"tool: {tc['function']['name']}\n"
                                    f"args: {json.dumps(parsed_args, indent=2, ensure_ascii=False)}")
                        self.ui.show("", call_log)
                        convo.append(call_log or "")

                        res = self.exec.execute(tc["function"]["name"], args)
                        resp_log = (f"> TOOL RESPONSE, id: {tc['id']}\n"
                                    f"response: {res}")
                        self.ui.show("", resp_log)
                        convo.append(resp_log or "")

                        hist.append({"role": "tool",
                                     "tool_call_id": tc["id"],
                                     "name": tc["function"]["name"],
                                     "content": res})
                    except json.JSONDecodeError:
                        error_msg = f"Invalid JSON arguments for tool {tc['function']['name']}: {args}"
                        self.ui.error(error_msg)
                        convo.append(error_msg)
                        hist.append({"role": "tool",
                                     "tool_call_id": tc["id"],
                                     "name": tc["function"]["name"],
                                     "content": json.dumps({"error": error_msg}, indent=2)})
                        # Break the tool call loop and return current conversation
                        return {"text": "\n\n".join(convo), "messages": hist}
                params["messages"] = hist
        except self.LLMCallException as e:
            # Propagate with partial result
            raise
        except Exception as e:
            # Catch-all: propagate convo so far
            raise self.LLMCallException(f"Unexpected LLM error: {e}", partial_result="\n\n".join(convo)) from e
        return {"text": "\n\n".join(convo), "messages": hist}

# -------- legacy alias --------
openaiclient = liteclient
