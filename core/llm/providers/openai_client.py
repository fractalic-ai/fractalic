"""
lite_client.py  —  Unified multimodal client (LiteLLM)

✓ vision images (JPEG, PNG, GIF, WEBP)
✓ vision PDFs
   • OpenAI models → auto-upload then file_id reference
   • Anthropic / other → base-64 data: URI
✓ stop_sequences + streaming trim
✓ toolkit-driven tool calls / schema
"""

# ================= stdlib / deps =================
import json, ast, inspect, logging, time, base64, imghdr, os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

from rich.console import Console
from litellm import completion
import openai                          #  ← new
import requests

# Silence verbose http logging
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
#  Toolkit (unchanged)
# ====================================================================
class Toolkit(dict):
    def register(self, name: Optional[str] = None):
        def deco(fn: Callable):
            self[name or fn.__name__] = fn
            return fn
        return deco

    def generate_schema(self):
        out = []
        for n, fn in self.items():
            ps = inspect.signature(fn).parameters
            out.append({
                "type": "function",
                "function": {
                    "name": n,
                    "description": inspect.getdoc(fn) or f"Executes {n}.",
                    "parameters": {
                        "type": "object",
                        "properties": {p: {"type": "string"} for p in ps},
                        "required": [p for p, prm in ps.items()
                                     if prm.default is inspect.Parameter.empty],
                    },
                },
            })
        return out


toolkit = Toolkit()

# ====================================================================
#  Console
# ====================================================================
class ConsoleManager:
    def __init__(self): self.c = Console()
    def show(self, role: str, content: str):
        colours = {"user": "cyan", "error": "red", "status": "dim"}
        if role in colours:
            self.c.print(f"[{colours[role]}]{role.upper()}:[/] {content}")
        else:  # assistant / tool
            self.c.print(content, highlight=False)
    def status(self, m): self.show("status", m)
    def error (self, m): self.show("error",  m)

# ====================================================================
#  Tool executor
# ====================================================================
class ToolExecutor:
    def __init__(self, tk: Toolkit, ui: ConsoleManager):
        self.tk, self.ui = tk, ui
    def execute(self, fn: str, args_json: str) -> str:
        if fn not in self.tk:
            err = f"Tool '{fn}' not found."
            self.ui.error(err)
            return json.dumps({"error": err}, indent=2)
        try:
            res = self.tk[fn](**json.loads(args_json or "{}"))
            return json.dumps(res, indent=2)
        except Exception as e:
            self.ui.error(f"Tool '{fn}' failed: {e}")
            return json.dumps({"error": str(e)}, indent=2)

# ====================================================================
#  Stream processor (trims stop-seq)
# ====================================================================
class StreamProcessor:
    def __init__(self, ui: ConsoleManager, stop: Optional[List[str]]):
        self.ui, self.stop = ui, stop or []
    def process(self, stream_iter):
        buf = ""
        for chunk in stream_iter:
            txt = chunk["choices"][0]["delta"].get("content", "")
            buf += txt
            for s in self.stop:
                if buf.endswith(s):
                    buf = buf[:-len(s)]
                    txt = txt[:-len(s)]
            if txt:
                self.ui.show("", txt)
        return buf

# ====================================================================
#  Media helpers
# ====================================================================
def _validate_image(path: Path) -> tuple[str, bytes]:
    if path.stat().st_size > MAX_IMAGE_SIZE:
        raise ValueError("Image too large (>20 MB)")
    data = path.read_bytes()
    fmt  = imghdr.what(None, data)
    if not fmt:
        raise ValueError("Unknown image format")
    mime = next((m for m, fmts in SUPPORTED_MEDIA_TYPES.items() if fmt in fmts), None)
    if not mime:
        raise ValueError(f"Unsupported format: {fmt}")
    return mime, data

def _upload_pdf_openai(pdf_path: Path, api_key: str) -> str:
    import openai                      # already imported at top now
    openai.api_key = api_key           # set key for this call
    fid = openai.files.create(         # same as raw OpenAI call
        file=open(pdf_path, "rb"),
        purpose="vision"
    ).id
    return fid


def _embed_media(item: str | dict, provider: str, api_key: str) -> Dict[str, Any]:
    """
    item:   image/PDF path OR {"file_id": "...", "mime_type": "..."}
    provider: 'openai' | 'anthropic' | ...
    """
    # Pre-uploaded OpenAI file block
    if isinstance(item, dict) and "file_id" in item:
        return {"type": "file", "file": item}

    p = Path(item)
    if not p.exists():
        raise FileNotFoundError(item)

    if p.suffix.lower() == ".pdf":
        if provider == "openai":
            fid = _upload_pdf_openai(p, api_key)
            return {"type": "file",
                    "file": {"file_id": fid, "mime_type": "application/pdf"}}
        # Anthropic / Gemini – inline base64
        data_b64 = base64.b64encode(p.read_bytes()).decode()
        return {"type": "image_url",
                "image_url": {"url": f"data:application/pdf;base64,{data_b64}"}}

    # ---- image branch ----
    mime, data = _validate_image(p)
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime};base64,{base64.b64encode(data).decode()}",
        },
    }

# ====================================================================
#  Main LiteLLM client
# ====================================================================
@dataclass
class liteclient:
    api_key: str
    model: str = "openai/gpt-4o-mini"
    temperature: float = 0.1
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    system_prompt: str = "You are a helpful assistant."
    max_tool_turns: int = 5
    settings: Optional[Dict[str, Any]] = field(default=None, repr=False)
    toolkit: Toolkit = field(default_factory=lambda: toolkit)

    def __post_init__(self):
        if self.settings:
            s = self.settings
            self.model         = s.get("model",          self.model)
            self.temperature   = s.get("temperature",    self.temperature)
            self.top_p         = s.get("top_p", s.get("topP", self.top_p))
            self.max_tokens    = s.get("max_tokens", s.get("max_completion_tokens", self.max_tokens))
            self.system_prompt = s.get("system_prompt", s.get("systemPrompt", self.system_prompt))
            self.max_tool_turns= s.get("max_tool_turns", self.max_tool_turns)

        self.ui   = ConsoleManager()
        self.exec = ToolExecutor(self.toolkit, self.ui)
        self.schema = self.toolkit.generate_schema()

    # -----------------------------------------------------------------
    def _provider(self, op_params: Dict[str, Any]) -> str:
        if "provider" in op_params:
            return op_params["provider"]
        return self.model.split("/")[0]  # openai/…, anthropic/…

    # -----------------------------------------------------------------
    def llm_call(
        self,
        prompt_text: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        operation_params: Optional[Dict[str, Any]] = None,
        *,
        stream: bool = False
    ) -> str:

        op       = operation_params or {}
        provider = self._provider(op)

        # ------------- core params -------------
        params = {
            "model":      op.get("model", self.model),
            "temperature":op.get("temperature", self.temperature),
            "top_p":      op.get("top_p", self.top_p),
            "max_tokens": op.get("max_tokens", self.max_tokens),
            "stop":       op.get("stop_sequences"),
            "tools":      self.schema,
            "stream":     stream
        }
        # LiteLLM expects api_key via env or kwargs
        if provider == "openai":
            os.environ["OPENAI_API_KEY"] = self.api_key
        elif provider == "anthropic":
            os.environ["ANTHROPIC_API_KEY"] = self.api_key

        # ------------- build messages ----------
        if messages:
            hist = list(messages)
            if hist[0].get("role") != "system":
                hist.insert(0, {"role": "system", "content": self.system_prompt})
            self.ui.show("user", f"[{len(hist)} msgs]")
            if op.get("media"):
                for media in op["media"]:
                    for m in hist:
                        if m.get("role") == "user":
                            # normalise content
                            if isinstance(m["content"], str):
                                m["content"] = [{"type": "text", "text": m["content"]}]
                            elif isinstance(m["content"], dict):
                                m["content"] = [m["content"]]
                            m["content"].insert(0, _embed_media(media, provider, self.api_key))
                            break
        else:
            if prompt_text is None:
                raise ValueError("Need prompt_text or messages")
            blocks = []
            for media in op.get("media", []):
                blocks.append(_embed_media(media, provider, self.api_key))
            blocks.append({"type": "text", "text": prompt_text})
            hist = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": blocks},
            ]
            self.ui.show("user", prompt_text)

        params["messages"] = hist

        # ------------- conversation loop -------
        convo = []
        for _ in range(self.max_tool_turns):
            if stream:
                response = completion(stream=True, **params)
                content  = StreamProcessor(self.ui, params["stop"]).process(response)
                tool_calls = []
            else:
                resp = completion(**params)
                content   = resp["choices"][0]["message"].get("content", "")
                tool_calls= resp["choices"][0]["message"].get("tool_calls", [])

            self.ui.show("", content)
            convo.append(content)
            assistant_entry = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_entry["tool_calls"] = tool_calls
            hist.append(assistant_entry)

            if not tool_calls:
                break

            # ----- execute toolkit tools --------
            for tc in tool_calls:
                args_str = tc["function"]["arguments"]
                call_str = (
                    f"> TOOL CALL, id: {tc['id']}\n"
                    f"tool: {tc['function']['name']}\n"
                    f"args: {json.dumps(json.loads(args_str), indent=2)}"
                )
                self.ui.show("", call_str)
                convo.append(call_str)

                result = self.exec.execute(tc["function"]["name"], args_str)
                resp_str = f"> TOOL RESPONSE, id: {tc['id']}\nresponse: {result}"
                self.ui.show("", resp_str)
                convo.append(resp_str)

                hist.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "content": result,
                })

            params["messages"] = hist  # update for next turn

        return "\n\n".join(convo)


# ====================================================================
#  Example toolkit entries
# ====================================================================
@toolkit.register("calculate")
def calculate(expression: str):
    try:
        allowed = (
            ast.Expression, ast.Constant, ast.BinOp, ast.Add, ast.Sub,
            ast.Mult, ast.Div, ast.Pow, ast.USub
        )
        tree = ast.parse(expression, mode="eval")
        if all(isinstance(n, allowed) for n in ast.walk(tree)):
            return {"result": eval(compile(tree, "", "eval"))}
        return {"error": "Unsafe expression"}
    except Exception as e:
        return {"error": str(e)}


@toolkit.register("get_weather")
def get_weather(latitude: float, longitude: float):
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}&longitude={longitude}&current=temperature_2m"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}
    
# keep legacy import path – do NOT remove
openaiclient = liteclient

