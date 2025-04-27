# Import necessary libraries
import requests
import json
import re
import traceback
import ast
import time
import logging
from typing import List, Dict, Any, Tuple, Optional, Union, Callable, Sequence
from dataclasses import dataclass, field

# Rich library for enhanced console output
from rich.console import Console
from rich.syntax import Syntax
from rich.markdown import Markdown

# OpenAI library
from openai import OpenAI, APIError, RateLimitError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionChunk
)

# --- Setup Logging ---
# Configure logging for warnings (e.g., retries)
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants for Styles ---
USER_STYLE = "bold bright_blue"
ASSISTANT_STYLE = "bold bright_green"
TOOL_REQUEST_STYLE = "grey50"
TOOL_RESULT_STYLE = "grey50"
TOOL_RESULT_HEADER_STYLE = "bold grey50"
ERROR_STYLE = "bold red"
STATUS_STYLE = "dim"
STREAM_STYLE = "bright_green"
TURN_STYLE = "bold yellow"
WARN_STYLE = "yellow"

# --- ToolKit Definition and Registry ---

class ToolKit(dict):
    """A dictionary-like class to register and manage tools (functions)."""

    def register(self, name: Optional[str] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a function as a tool."""
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or func.__name__
            if tool_name in self:
                logger.warning(f"Tool '{tool_name}' is being redefined.")
            self[tool_name] = func
            # Attempt to preserve original function attributes if needed
            # setattr(func, '_is_tool', True)
            # setattr(func, '_tool_name', tool_name)
            return func
        return decorator

    def to_tools_schema(self) -> List[Dict[str, Any]]:
        """Generates the OpenAI tool schema list from registered functions."""
        schema_list: List[Dict[str, Any]] = []
        for name, func in self.items():
            try:
                # Introspect function signature for parameters and types
                sig = inspect.signature(func)
                parameters = sig.parameters
                annotations = {k: v.annotation for k, v in parameters.items() if v.annotation != inspect.Parameter.empty}
                # Remove 'return' annotation if present
                annotations.pop("return", None)

                properties = {}
                required = []
                for param_name, param_type in annotations.items():
                    json_type = self._map_type_to_json(param_type)
                    properties[param_name] = {"type": json_type}
                    # Assume parameters are required if they don't have a default value
                    if parameters[param_name].default == inspect.Parameter.empty:
                        required.append(param_name)

                # Use function docstring as description
                description = inspect.getdoc(func) or f"Executes the {name} tool."

                schema_list.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                })
            except Exception as e:
                logger.error(f"Failed to generate schema for tool '{name}': {e}", exc_info=True)
        return schema_list

    @staticmethod
    def _map_type_to_json(py_type: Any) -> str:
        """Maps Python types to JSON schema types."""
        if py_type == str: return "string"
        if py_type == int: return "integer"
        if py_type == float: return "number"
        if py_type == bool: return "boolean"
        if py_type == list or py_type == List: return "array"
        if py_type == dict or py_type == Dict: return "object"
        # Add more mappings as needed (e.g., for Union, Optional)
        # Default to string if type is unknown or complex
        return "string"

# --- Global Toolkit Instance ---
# Tools can be registered to this instance from anywhere before client initialization
# or passed via `available_functions` in settings.
import inspect # Needed for signature introspection in ToolKit
toolkit = ToolKit()

# --- Example Tool Implementations using the registry ---

@toolkit.register("calculate")
def safe_calculate(expression: str) -> Dict[str, Any]:
    """
    Safely evaluates a simple mathematical expression using AST parsing.
    Allows basic arithmetic operations (+, -, *, /, **).
    Args:
        expression: The mathematical expression string to evaluate.
    Returns:
        A dictionary containing the 'result' or an 'error' message.
    """
    # Allowed AST node types for safety
    allowed_nodes = {
        ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Call,
        ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
        # Be cautious adding more complex nodes like Name or Call without strict validation
    }
    # Allowed built-in functions (very restricted)
    allowed_builtins = {'abs': abs, 'pow': pow, 'round': round}

    if not isinstance(expression, str):
        return {"error": "Invalid input: Expression must be a string."}

    try:
        # Parse the expression into an Abstract Syntax Tree (AST)
        tree = ast.parse(expression, mode='eval')

        # Validate all nodes in the AST
        for node in ast.walk(tree):
            if type(node) not in allowed_nodes:
                 # Disallow specific potentially harmful nodes like Attribute, Subscript, etc.
                 if isinstance(node, (ast.Attribute, ast.Subscript, ast.Starred, ast.Lambda, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                      raise ValueError(f"Disallowed construct '{type(node).__name__}' found in expression.")
                 # Check for disallowed function calls
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in allowed_builtins:
                      raise ValueError(f"Disallowed function call '{node.func.id}' found in expression.")
                 # Check for disallowed names (variables)
                 if isinstance(node, ast.Name) and node.id not in allowed_builtins:
                      raise ValueError(f"Disallowed name/variable '{node.id}' found in expression.")
                 # Fallback for any other disallowed node type
                 if type(node) not in allowed_nodes:
                      raise ValueError(f"Disallowed node type '{type(node).__name__}' found in expression.")


        # Compile the validated AST
        code = compile(tree, filename='<safe_expression>', mode='eval')

        # Evaluate the compiled code with a restricted environment
        result = eval(code, {"__builtins__": allowed_builtins}, {}) # Only allow safe builtins

        # Validate result type
        if not isinstance(result, (int, float)):
            raise ValueError("Expression did not evaluate to a numerical result.")

        return {"result": result}

    except (SyntaxError, ValueError, TypeError, NameError, ZeroDivisionError) as e:
        return {"error": f"Failed to evaluate expression '{expression}': {e}"}
    except Exception as e: # Catch unexpected errors during parsing/evaluation
        logger.error(f"Unexpected error during calculation for '{expression}': {e}", exc_info=True)
        return {"error": f"An unexpected error occurred during calculation: {e}"}


@toolkit.register("get_weather")
def get_weather(latitude: float, longitude: float) -> Dict[str, Any]:
    """
    Gets the current weather for a specific location using the Open-Meteo API.
    Args:
        latitude: The latitude of the location.
        longitude: The longitude of the location.
    Returns:
        A dictionary containing the weather data or an 'error' message.
    """
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        return {"error": "Invalid input: Latitude and longitude must be numbers."}
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m"
    try:
        response = requests.get(url, timeout=10) # 10 second timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Weather API request timed out."}
    except requests.exceptions.HTTPError as http_err:
        return {"error": f"Weather API request failed with HTTP status {http_err.response.status_code}: {http_err}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Weather API request failed: {e}"}
    except json.JSONDecodeError:
        return {"error": "Failed to parse Weather API response (invalid JSON)."}
    except Exception as e:
        logger.error(f"Unexpected error in get_weather tool for lat={latitude}, lon={longitude}: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred in get_weather: {e}"}


# --- Helper: API Transport with Retry ---

# Using `slots=True` in the dataclass to reduce memory usage and improve attribute access speed.
@dataclass(slots=True)
class _Transport:
    """Handles raw OpenAI API calls with retry logic for rate limits."""
    api_key: str
    base_url: Optional[str] = None
    max_retries: int = 3 # Max number of retries on rate limit errors

    def _get_client(self) -> OpenAI:
        """Creates a new OpenAI client instance for each call to ensure thread safety if used concurrently."""
        # Consider optimizing if client creation becomes a bottleneck, but this is safer.
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat_completions_create(self, **kwargs) -> Any:
        """Calls the OpenAI chat completions endpoint with retry logic."""
        delay = 1.5 # Initial delay in seconds
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                client = self._get_client()
                # Pass all keyword arguments directly to the OpenAI client method
                response = client.chat.completions.create(**kwargs)
                return response
            except RateLimitError as e:
                last_exception = e
                if attempt == self.max_retries:
                    logger.error(f"Rate limit error persisted after {self.max_retries} retries.")
                    raise # Re-raise the last RateLimitError if all retries fail
                else:
                    logger.warning(f"Rate limit error encountered (attempt {attempt + 1}/{self.max_retries + 1}). Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= 2 # Exponential backoff
            except APIError as e:
                 # Handle other API errors (e.g., authentication, server errors) - potentially retry some?
                 logger.error(f"OpenAI API Error (Status {e.status_code}): {e.message}")
                 raise # Re-raise other API errors immediately
            except Exception as e:
                 # Catch unexpected errors during the API call
                 logger.error(f"Unexpected error during API call: {e}", exc_info=True)
                 raise # Re-raise unexpected errors

        # This point should ideally not be reached if RateLimitError is always raised on final attempt
        logger.error("Exited retry loop unexpectedly.")
        if last_exception:
            raise last_exception # Ensure the last known exception is raised
        else:
            raise RuntimeError("Failed to complete API call after retries without a specific exception.")


# --- Helper: Console Output Manager (from v2) ---

class _ConsoleManager:
    """Handles all console output using the Rich library."""
    def __init__(self, console_theme: Optional[str] = None):
        """Initializes the Rich Console."""
        self.console = Console(theme=console_theme)

    def display_message(self, role: str, content: Optional[str] = None, tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None, tool_call_id: Optional[str] = None, tool_name: Optional[str] = None):
        """Prints conversation steps (user, assistant, tool) to the console with appropriate styling."""
        self.console.print() # Blank line separator for clarity
        if role == "user":
            self.console.print("USER:", style=USER_STYLE)
            if content: self.console.print(content)
        elif role == "assistant":
            self.console.print("ASSISTANT:", style=ASSISTANT_STYLE)
            if content: self.console.print(Markdown(content)) # Render assistant content as Markdown for better formatting
            if tool_calls:
                self.console.print("-- Tool Calls Requested --", style=TOOL_REQUEST_STYLE)
                for tc in tool_calls:
                    # Display details of each requested tool call
                    self.console.print(f"  ID: {tc.id}\n  Func: {tc.function.name}\n  Args: {tc.function.arguments}", style=TOOL_REQUEST_STYLE)
        elif role == "tool":
            # Determine if the tool result content represents an error
            is_error = False
            parsed_content_obj = None
            error_message = content # Default error message is the raw content
            try:
                # Try parsing content as JSON to check for an 'error' key
                parsed_content_obj = json.loads(content or '{}')
                if isinstance(parsed_content_obj, dict) and 'error' in parsed_content_obj:
                    is_error = True
                    error_message = parsed_content_obj.get('error', error_message) # Get specific error message if available
            except json.JSONDecodeError:
                 # If not JSON, check common patterns in the string itself
                 is_error = isinstance(content, str) and ("error" in content.lower() or "fail" in content.lower())

            # Display header indicating tool result or error
            header = f"TOOL {'ERROR' if is_error else 'RESULT'} ({tool_name} ID: {tool_call_id}):"
            self.console.print(header, style=ERROR_STYLE if is_error else TOOL_RESULT_HEADER_STYLE)

            # Display the content: error message or formatted result
            if is_error:
                self.console.print(str(error_message), style=ERROR_STYLE)
            else:
                try: # Try to pretty-print the result as JSON if possible
                    # Ensure we handle the case where parsed_content_obj is None but content is valid JSON
                    obj_to_dump = parsed_content_obj if parsed_content_obj is not None else json.loads(content or '{}')
                    json_str = json.dumps(obj_to_dump, indent=2)
                    self.console.print(Syntax(json_str, "json", theme="monokai", line_numbers=False), style=TOOL_RESULT_STYLE)
                except (json.JSONDecodeError, TypeError): # Fallback to printing raw content if not valid JSON or not serializable
                    self.console.print(content or "[Empty content]", style=TOOL_RESULT_STYLE)

    def display_status(self, message: str):
        """Prints general status messages."""
        self.console.print(f"[{STATUS_STYLE}]{message}[/]")

    def display_warning(self, message: str):
        """Prints warning messages."""
        self.console.print(f"[{WARN_STYLE}]WARN:[/] {message}")

    def display_error(self, message: str, show_locals: bool = False):
        """Prints error messages, optionally with local variables from exception."""
        self.console.print(f"[{ERROR_STYLE}]ERROR:[/] {message}")
        if show_locals:
            self.console.print_exception(show_locals=True) # Useful for debugging internal errors

    def display_api_error(self, error: APIError):
        """Prints formatted details of an OpenAI APIError."""
        # This might be less used now that _Transport handles logging API errors
        self.console.print(f"\n[{ERROR_STYLE}]API ERROR:[/] Status={error.status_code}, Msg={error.message}, Type={error.type}")

    def display_stream_chunk(self, text_chunk: str):
        """Prints a chunk of text received during streaming without a newline."""
        self.console.print(text_chunk, end="", style=STREAM_STYLE, highlight=False)

    def display_stream_end(self, finish_reason: Optional[str]):
        """Prints the reason why the stream finished."""
        self.console.print(f"\n   [{STATUS_STYLE} i](Finish Reason: {finish_reason})[/]")

    def display_tool_execution_start(self, func_name: str, tool_call_id: str):
        """Prints a message indicating the start of a tool execution."""
        self.console.print(f"  [{STATUS_STYLE}]Executing Tool:[/] [{TOOL_RESULT_HEADER_STYLE}]{func_name}[/] (ID: {tool_call_id})")

    def display_tool_execution_details(self, label: str, value: Any):
        """Prints details during tool execution (e.g., arguments, results, errors)."""
        # Use error style if the label indicates an error
        style = ERROR_STYLE if "ERROR" in label.upper() else TOOL_RESULT_STYLE
        self.console.print(f"    [{style}]{label}: {value}[/]")

    def display_turn_start(self, turn_number: int):
        """Prints a separator indicating the start of a new conversation turn (for multi-turn tool use)."""
        self.console.print(f"\n--- Turn {turn_number} ---", style=TURN_STYLE)

    def display_max_turns_reached(self, max_turns: int):
        """Prints a message when the maximum number of tool interaction turns is reached."""
        self.console.print(f"[{ERROR_STYLE}]-- Reached max turns ({max_turns}). Exiting loop. --[/]")


# --- Helper: Stream Processor (from v2) ---

class _StreamProcessor:
    """Handles the logic for processing OpenAI API stream chunks and reconstructing messages."""

    def __init__(self, console: _ConsoleManager):
        """Initializes the stream processor with a console manager."""
        self.console = console
        # Initialize accumulators for the current stream
        self.response_content_accumulator = ""
        self.tool_calls_in_progress: Dict[int, Dict[str, Any]] = {}
        self._needs_final_newline = False # Track if content was printed without a newline

    def process_chunk(self, chunk: ChatCompletionChunk) -> Optional[str]:
        """
        Processes a single chunk from the OpenAI stream.
        Accumulates content and tool call information.
        Returns the finish_reason if the stream ends with this chunk, otherwise None.
        """
        finish_reason = chunk.choices[0].finish_reason if chunk.choices else None
        delta = chunk.choices[0].delta if chunk.choices and hasattr(chunk.choices[0], 'delta') else None

        if not delta:
            return finish_reason # End of stream or empty chunk

        # Process content delta
        if hasattr(delta, 'content') and delta.content:
            text_chunk = delta.content
            self.response_content_accumulator += text_chunk
            self.console.display_stream_chunk(text_chunk) # Display chunk immediately
            self._needs_final_newline = True # Mark that we need a newline at the end

        # Process tool call delta (accumulates parts of tool calls)
        if hasattr(delta, 'tool_calls') and delta.tool_calls:
            for tc_chunk in delta.tool_calls:
                # Use getattr for safer access to attributes like index
                index = getattr(tc_chunk, 'index', None)
                if index is None: continue # Skip if index is missing

                # Initialize tool call structure in our accumulator if it's the first time we see this index
                if index not in self.tool_calls_in_progress:
                    self.tool_calls_in_progress[index] = {"id": None, "function_name": "", "arguments": ""}
                    self.console.display_status(f"   [{TOOL_REQUEST_STYLE} i](Tool Call {index} triggered)[/]") # Notify user

                # Accumulate tool call details (id, function name, arguments)
                tool_id = getattr(tc_chunk, 'id', None)
                if tool_id:
                    self.tool_calls_in_progress[index]["id"] = tool_id # Store the tool call ID

                if hasattr(tc_chunk, 'function') and tc_chunk.function:
                    func = tc_chunk.function
                    func_name = getattr(func, 'name', None)
                    func_args = getattr(func, 'arguments', None)
                    if func_name:
                        self.tool_calls_in_progress[index]["function_name"] += func_name # Append function name parts
                    if func_args:
                        self.tool_calls_in_progress[index]["arguments"] += func_args # Append argument parts

        return finish_reason # Return finish reason if this chunk concludes the stream

    def finalize(self) -> Tuple[Optional[str], List[ChatCompletionMessageToolCall]]:
        """
        Finalizes processing after the stream ends.
        Adds a final newline if needed and reconstructs complete tool calls.
        Returns the accumulated content and the list of reconstructed tool calls.
        Resets internal state.
        """
        if self._needs_final_newline:
            self.console.console.print() # Add the final newline if content was printed

        final_content = self.response_content_accumulator or None
        reconstructed_calls = self._reconstruct_tool_calls()

        # Reset state for potential reuse
        self.response_content_accumulator = ""
        self.tool_calls_in_progress = {}
        self._needs_final_newline = False

        return final_content, reconstructed_calls

    def _reconstruct_tool_calls(self) -> List[ChatCompletionMessageToolCall]:
        """Rebuilds complete tool call objects from the accumulated chunks."""
        reconstructed_calls = []
        # Sort by index to ensure correct order
        for index, call_info in sorted(self.tool_calls_in_progress.items()):
            tool_id = call_info.get('id')
            func_name = call_info.get('function_name')
            func_args = call_info.get('arguments', '') # Default to empty string

            if tool_id and func_name: # Ensure we have the essential parts
                try:
                    # Validate arguments are valid JSON - crucial for downstream processing
                    json.loads(func_args or '{}') # Use '{}' if args string is empty
                    reconstructed_calls.append(ChatCompletionMessageToolCall(
                        id=tool_id,
                        function={'name': func_name, 'arguments': func_args},
                        type='function' # Assuming only function tools for now
                    ))
                except json.JSONDecodeError:
                    # Log a warning if arguments are not valid JSON
                    self.console.display_warning(f"Could not parse tool arguments for tool index {index} during stream reconstruction. Args: '{func_args}'")
                    # Still append the call but downstream execution will likely fail
                    reconstructed_calls.append(ChatCompletionMessageToolCall(
                        id=tool_id,
                        function={'name': func_name, 'arguments': func_args},
                        type='function'
                    ))
            else:
                 # Log if essential parts are missing
                 self.console.display_warning(f"Incomplete tool call data received for index {index} during stream reconstruction (Missing ID or Name).")

        return reconstructed_calls


# --- Helper: Tool Executor (Modified to use ToolKit) ---

class _ToolExecutor:
    """Handles the validation and execution of tools using a ToolKit."""

    def __init__(self, toolkit_instance: ToolKit, console: _ConsoleManager):
        """
        Initializes the ToolExecutor.
        Args:
            toolkit_instance: The ToolKit instance containing available functions.
            console: The ConsoleManager instance for logging execution details.
        """
        self.toolkit = toolkit_instance
        self.console = console

    def execute_tool(self, tool_call: ChatCompletionMessageToolCall) -> Dict[str, Any]:
        """
        Executes a single tool call requested by the LLM using the provided ToolKit.
        Handles argument parsing, function execution, error handling, and result serialization.
        Returns a dictionary formatted for the OpenAI API's tool result message.
        """
        function_name = tool_call.function.name
        tool_call_id = tool_call.id
        raw_arguments = tool_call.function.arguments # Arguments as a JSON string

        self.console.display_tool_execution_start(function_name, tool_call_id)

        result_content: Any = {"error": "Tool execution failed unexpectedly."} # Default error

        # Find the function in the toolkit
        function_to_call = self.toolkit.get(function_name)

        if function_to_call:
            # Found the function, attempt execution
            try:
                self.console.display_tool_execution_details("Args (raw)", raw_arguments)
                # Parse the JSON arguments string into a Python dictionary
                function_args = {} # Default to empty dict if no args
                if isinstance(raw_arguments, str) and raw_arguments.strip():
                    try:
                       function_args = json.loads(raw_arguments)
                    except json.JSONDecodeError as e:
                        # Raise a specific error if JSON parsing fails
                        raise ValueError(f"Invalid JSON arguments provided for '{function_name}': {e}") from e
                elif raw_arguments and not isinstance(raw_arguments, str):
                    # Handle unexpected argument types
                    raise ValueError(f"Expected string arguments for '{function_name}', got {type(raw_arguments).__name__}")

                self.console.display_tool_execution_details("Args (parsed)", function_args)

                # Ensure parsed arguments are a dictionary before calling the function
                if not isinstance(function_args, dict):
                     raise TypeError(f"Expected arguments for '{function_name}' to be a JSON object (dict), but received {type(function_args).__name__}")

                # --- Execute the actual function with keyword arguments ---
                function_response = function_to_call(**function_args)
                result_content = function_response # Store the direct result from the function
                self.console.display_tool_execution_details("Result", result_content)

            except json.JSONDecodeError as e: # Catch error from json.loads
                err_msg = f"Failed to decode JSON arguments for '{function_name}': {e}"
                result_content = {"error": err_msg}
                self.console.display_tool_execution_details("ARGUMENT PARSE ERROR", err_msg)
            except ValueError as ve: # Catch specific errors raised during argument validation/parsing
                err_msg = f"Argument validation/parsing failed for '{function_name}': {ve}"
                result_content = {"error": err_msg}
                self.console.display_tool_execution_details("ARGUMENT ERROR", err_msg)
            except TypeError as te: # Catch errors from calling the function (e.g., wrong args, missing args)
                err_msg = f"Argument mismatch calling function '{function_name}': {te}"
                result_content = {"error": err_msg}
                self.console.display_tool_execution_details("FUNCTION CALL ERROR", err_msg)
            except Exception as e: # Catch any other unexpected errors during the function's execution
                err_msg = f"Execution of function '{function_name}' failed unexpectedly: {e}"
                result_content = {"error": err_msg}
                # Log traceback for internal debugging
                logger.error(f"Unexpected error during execution of tool '{function_name}': {e}", exc_info=True)
                self.console.display_tool_execution_details("UNEXPECTED EXECUTION ERROR", f"{err_msg}") # Keep console output concise

        else: # Function name requested by LLM is not found in our toolkit
            error_msg = f"Unknown function requested by LLM: '{function_name}'. Not found in the configured toolkit."
            result_content = {"error": error_msg}
            self.console.display_tool_execution_details("UNKNOWN FUNCTION ERROR", error_msg)

        # --- Serialize the result back to a JSON string for the API ---
        try:
            content_str = json.dumps(result_content)
        except TypeError as e:
            # Handle cases where the function's return value is not JSON serializable
            err_msg = f"Tool result for '{function_name}' is not JSON serializable: {e}. Returning error message instead."
            logger.warning(f"Serialization error for tool '{function_name}': {e}")
            self.console.display_tool_execution_details("RESULT SERIALIZATION ERROR", err_msg)
            # Provide a fallback error structure
            content_str = json.dumps({
                "error": err_msg,
                "original_result_type": type(result_content).__name__,
                "original_result_str": str(result_content) # Convert original result to string
            })

        # Return the dictionary format required for a 'tool' role message
        return {
            "tool_call_id": tool_call_id,
            "role": "tool",
            "name": function_name,
            "content": content_str, # Content MUST be a string
        }


# --- Main Client Class (Dataclass, Lowercase) ---

@dataclass
class openaiclient:
    """
    Merged and refactored OpenAI client using helper components, ToolKit registry,
    dataclass configuration, and retry logic. Handles chat completions, tool usage,
    and streaming responses. Maintains legacy compatibility via `settings` dict.

    Attributes:
        api_key (str): The OpenAI API key (required).
        model (str): Default model name.
        temperature (float): Default sampling temperature.
        top_p (float): Default nucleus sampling parameter.
        max_tokens (Optional[int]): Default maximum tokens to generate.
        system_prompt (str): Default system prompt message.
        base_url (Optional[str]): Optional custom API base URL.
        max_tool_turns (int): Default maximum iterations for tool calls within a single llm_call.
        max_retries (int): Default maximum retries for rate limit errors.
        settings (Optional[dict]): Legacy settings dictionary for compatibility.
        # Internal helpers initialized in __post_init__
        _ui: _ConsoleManager = field(init=False, repr=False)
        _transport: _Transport = field(init=False, repr=False)
        _toolkit: ToolKit = field(init=False, repr=False)
        _tool_executor: _ToolExecutor = field(init=False, repr=False)
        _tools_schema: List[Dict[str, Any]] = field(init=False, repr=False)
    """
    api_key: str
    model: str = "gpt-4-turbo" # Updated default model
    temperature: float = 0.1
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    system_prompt: str = "You are a helpful assistant."
    base_url: Optional[str] = None
    max_tool_turns: int = 5
    max_retries: int = 3
    settings: Optional[Dict[str, Any]] = field(default=None, repr=False) # Store legacy settings if provided

    def __post_init__(self):
        """Initialize helper components and apply legacy settings."""
        if not self.api_key:
            raise ValueError("OpenAI API key is required.")

        # --- Apply legacy settings if provided ---
        # Settings dict values override dataclass defaults/provided values
        if self.settings:
            s = self.settings
            self.model = s.get("model", self.model)
            self.temperature = s.get("temperature", self.temperature)
            self.top_p = s.get("topP", s.get("top_p", self.top_p)) # Handle alternate key 'topP'
            # Handle alternate keys for max_tokens
            self.max_tokens = s.get("max_tokens", s.get("max_completion_tokens", s.get("contextSize", self.max_tokens)))
            self.system_prompt = s.get("systemPrompt", s.get("system_prompt", self.system_prompt)) # Handle alternate key
            self.base_url = s.get("base_url", self.base_url)
            self.max_tool_turns = s.get("max_tool_turns", self.max_tool_turns)
            self.max_retries = s.get("max_retries", self.max_retries)
            console_theme = s.get("console_theme")
        else:
            console_theme = None # Default theme if no settings

        # --- Initialize Helpers ---
        self._ui = _ConsoleManager(console_theme=console_theme)
        self._transport = _Transport(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=self.max_retries
        )

        # Initialize ToolKit: Start with global `toolkit`, then update from settings
        self._toolkit = ToolKit()
        self._toolkit.update(toolkit) # Copy globally registered tools first
        if self.settings:
            avail_funcs = self.settings.get("available_functions")
            if isinstance(avail_funcs, dict):
                self._ui.display_status(f"Updating toolkit with functions from settings: {list(avail_funcs.keys())}")
                self._toolkit.update(avail_funcs) # Add/overwrite with functions from settings

        # Generate the schema from the final toolkit
        self._tools_schema = self._toolkit.to_tools_schema()

        # Initialize Tool Executor with the final toolkit
        self._tool_executor = _ToolExecutor(toolkit_instance=self._toolkit, console=self._ui)

        self._ui.display_status(f"openaiclient initialized. Model: {self.model}, Tools: {list(self._toolkit.keys())}")


    def _prepare_api_request_params(
        self,
        model: str,
        messages: List[ChatCompletionMessageParam],
        temperature: float,
        max_tokens: Optional[int],
        top_p: float,
        stream: bool
    ) -> Dict[str, Any]:
        """Prepares the dictionary of parameters for the OpenAI API chat completions call."""
        request_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens, # Can be None
            "top_p": top_p,
            "stream": stream,
        }
        # Add tools schema if the toolkit is not empty
        if self._tools_schema:
            request_params["tools"] = self._tools_schema
            request_params["tool_choice"] = "auto" # Let the model decide when to use tools

        # Filter out parameters with None values before sending to API
        return {k: v for k, v in request_params.items() if v is not None}


    def _execute_api_call(self, request_params: Dict[str, Any]) -> Tuple[Optional[str], List[ChatCompletionMessageToolCall], Optional[ChatCompletionMessage]]:
        """
        Executes the API call using the _Transport helper (handles retries).
        Handles stream processing via _StreamProcessor if applicable.
        """
        stream = request_params.get("stream", False)
        final_content: Optional[str] = None
        requested_tool_calls: List[ChatCompletionMessageToolCall] = []
        completion_message: Optional[ChatCompletionMessage] = None

        try:
            response_iterator_or_obj = self._transport.chat_completions_create(**request_params)

            if stream:
                # --- Streaming API Call ---
                self._ui.display_status(">> Streaming response...")
                stream_processor = _StreamProcessor(self._ui) # New processor per stream
                finish_reason = None
                for chunk in response_iterator_or_obj: # Iterate through the stream chunks
                    finish_reason = stream_processor.process_chunk(chunk)
                    if finish_reason: break # Stop processing once stream ends

                self._ui.display_stream_end(finish_reason)
                final_content, requested_tool_calls = stream_processor.finalize()
                # Manually construct the message object from streamed parts
                completion_message = ChatCompletionMessage(
                    role="assistant",
                    content=final_content,
                    tool_calls=requested_tool_calls or None
                )
            else:
                # --- Non-Streaming API Call ---
                self._ui.display_status("<< Received non-streaming response.")
                response_obj = response_iterator_or_obj # The direct response object
                if response_obj.choices:
                    completion_message = response_obj.choices[0].message
                    final_content = completion_message.content
                    requested_tool_calls = list(completion_message.tool_calls or [])
                else:
                    self._ui.display_warning("No message choices received in API response.")
                    # Ensure defaults are returned
                    final_content = None
                    requested_tool_calls = []
                    completion_message = None

            return final_content, requested_tool_calls, completion_message

        except Exception as e:
            # Re-raise any unexpected exceptions to be handled by caller
            raise
        # Errors (including retried RateLimitError) are raised by _Transport
        # and caught by the llm_call method's main try/except block.
        # No need for APIError/Exception handling here as _Transport handles/re-raises.

    def _prepare_call_parameters(self, operation_params: Optional[Dict], model_override: Optional[str]) -> Dict[str, Any]:
        """
        Determines the final parameters for the LLM call, merging defaults,
        instance attributes (potentially set by legacy settings), and per-call overrides.
        """
        op_params = operation_params or {}

        # Precedence: per-call op_params > model_override > instance attributes > defaults (handled by dataclass)
        params = {
            'model': model_override or op_params.get('model', self.model),
            'temperature': op_params.get('temperature', self.temperature),
            'max_tokens': op_params.get('max_tokens', op_params.get('max_completion_tokens', self.max_tokens)), # Allow None
            'top_p': op_params.get('topP', op_params.get('top_p', self.top_p)),
            'stream': op_params.get('stream', False), # Default stream to False unless overridden
            'system_prompt': op_params.get('system_prompt', self.system_prompt),
            'max_tool_turns': op_params.get('max_tool_turns', self.max_tool_turns) # Allow overriding turns per call
        }
        self._ui.display_status(f"Call parameters: Model={params['model']}, Temp={params['temperature']}, Stream={params['stream']}")
        return params

    def _prepare_initial_messages(self, system_prompt: str, prompt_text: Optional[str], messages_input: Optional[List[Union[Dict, ChatCompletionMessageParam]]]) -> Tuple[List[Union[Dict, ChatCompletionMessageParam]], str]:
        """
        Prepares the initial list of messages for the API call based on
        either a prompt_text or a list of messages. Ensures a system prompt exists.
        Also returns the content to display for the initial user turn.
        """
        history_for_api: List[Union[Dict, ChatCompletionMessageParam]] = []
        user_display_content: str = "[No user input provided]" # Default display

        if messages_input:
            history_for_api = list(messages_input) # Create a copy
            has_system_prompt = False
            if history_for_api and isinstance(history_for_api[0], dict) and history_for_api[0].get('role') == 'system':
                has_system_prompt = True
            elif history_for_api and hasattr(history_for_api[0], 'role') and history_for_api[0].role == 'system':
                 has_system_prompt = True

            if not has_system_prompt:
                 history_for_api.insert(0, {"role": "system", "content": system_prompt})
                 self._ui.display_status("Prepended system prompt to provided messages.")
            user_display_content = f"[{len(messages_input)} messages provided as initial context]"

        elif prompt_text is not None:
            history_for_api = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ]
            user_display_content = prompt_text
        else:
            raise ValueError("Internal Error: _prepare_initial_messages called without prompt_text or messages_input.")

        # Basic validation of message structure before sending
        validated_history = []
        for msg in history_for_api:
             if isinstance(msg, dict) and 'role' in msg and ('content' in msg or 'tool_calls' in msg or msg.get('role') == 'tool'):
                 validated_history.append(msg)
             elif hasattr(msg, 'role') and (hasattr(msg, 'content') or hasattr(msg, 'tool_calls')): # Basic check for Pydantic models
                 validated_history.append(msg)
             else:
                 self._ui.display_warning(f"Skipping malformed message during initial preparation: {msg}")
        if not validated_history:
             raise ValueError("No valid messages found after preparation.")


        return validated_history, user_display_content


    # --- Main Public Method ---
    # Supports legacy positional args via *args and modern keyword args
    def llm_call(self, prompt_text: Optional[str] = None, messages: Optional[List[Union[Dict, ChatCompletionMessageParam]]] = None, operation_params: Optional[Dict] = None, model: Optional[str] = None, **kwargs) -> str:
        """
        Makes a call to the OpenAI LLM, orchestrating helper components.
        Handles initial input, API calls (with retries), tool execution loop, and console output.
        Supports legacy positional arguments for compatibility but prefers keyword arguments.

        Args:
            prompt_text (str, optional): The user's prompt text. Used if 'messages' is not provided. Defaults to None.
            messages (list, optional): A list of message dictionaries or ChatCompletionMessageParam objects
                                       representing the conversation history. Used if provided. Defaults to None.
            operation_params (dict, optional): Dictionary to override default/settings parameters for this specific call
                                              (e.g., 'temperature', 'stream', 'max_tokens', 'model', 'system_prompt', 'max_tool_turns'). Defaults to None.
            model (str, optional): Specific OpenAI model name override for this call (e.g., "gpt-4-turbo"). Defaults to None.
            **kwargs: Additional keyword arguments (currently ignored but captured for flexibility).

        Returns:
            str: A concatenated string log containing only the outputs generated *during* this call
                 (assistant messages and tool results), separated by double newlines.
                 Excludes the initial user input.

        Raises:
            ValueError: If neither 'prompt_text' nor 'messages' is provided.
            RuntimeError: If an API error (after retries) or other unhandled exception occurs.
        """
        # --- Argument Handling (Prioritize keywords, handle legacy dict) ---
        # Use keyword args directly if provided
        current_prompt_text = prompt_text
        current_messages = messages
        current_op_params = operation_params or {} # Ensure op_params is a dict
        current_model_override = model

        # --- Input Validation ---
        if current_prompt_text is None and current_messages is None:
            raise ValueError("Either 'prompt_text' or 'messages' must be provided to llm_call.")
        if current_prompt_text is not None and current_messages is not None:
             self._ui.display_warning("Both 'prompt_text' and 'messages' provided. Using 'messages' and ignoring 'prompt_text'.")
             current_prompt_text = None # Prioritize messages list

        # --- Parameter Preparation ---
        # Merges instance defaults, legacy settings (in __post_init__), and per-call overrides
        call_params = self._prepare_call_parameters(current_op_params, current_model_override)
        stream = call_params['stream'] # Get final stream setting
        max_turns = call_params['max_tool_turns'] # Get final max_turns setting

        # --- Prepare Initial Messages ---
        initial_api_history, initial_user_display = self._prepare_initial_messages(
            call_params['system_prompt'], current_prompt_text, current_messages
        )

        # --- Initialize Conversation State ---
        conversation_log_parts: List[str] = [] # Accumulates output log
        current_loop_history = list(initial_api_history) # Tracks messages for API

        # --- Display Initial User Input ---
        self._ui.display_message(role="user", content=initial_user_display)

        # --- Conversation Loop ---
        turn_counter = 0
        try:
            while turn_counter < max_turns:
                turn_counter += 1
                if turn_counter > 1: self._ui.display_turn_start(turn_counter)

                # --- Prepare API Messages for this Turn ---
                # Convert Pydantic models to dicts if necessary before sending
                api_messages: List[ChatCompletionMessageParam] = []
                for msg in current_loop_history:
                    if isinstance(msg, ChatCompletionMessage):
                        api_messages.append(msg.model_dump(exclude_unset=True, exclude_none=True)) # type: ignore
                    elif isinstance(msg, dict):
                        api_messages.append(msg) # type: ignore
                    else:
                         self._ui.display_warning(f"Skipping unknown history item type: {type(msg)}")

                # --- Execute API Call ---
                request_params = self._prepare_api_request_params(
                    model=call_params['model'], messages=api_messages,
                    temperature=call_params['temperature'], max_tokens=call_params['max_tokens'],
                    top_p=call_params['top_p'], stream=stream
                )
                assistant_content, requested_tool_calls, assistant_message_obj = self._execute_api_call(request_params)

                # --- Process Assistant Response ---
                if assistant_message_obj:
                    self._ui.display_message(role="assistant", content=assistant_content, tool_calls=requested_tool_calls)
                    # Add output to log string
                    log_entry = ""
                    if assistant_content: log_entry += f"{assistant_content}"
                    if requested_tool_calls:
                        calls_str = "\n".join([f"  - Func: {tc.function.name}, Args: {tc.function.arguments}" for tc in requested_tool_calls])
                        if assistant_content: log_entry += "\n" # Separator
                        log_entry += f"-- Tool Calls Requested --\n{calls_str}"
                    if log_entry.strip(): conversation_log_parts.append(log_entry.strip())

                    # Add assistant message to history
                    if not assistant_message_obj.tool_calls: assistant_message_obj.tool_calls = None # Ensure None if empty
                    current_loop_history.append(assistant_message_obj)
                else:
                    self._ui.display_warning("API call did not return a valid assistant message. Ending loop.")
                    break

                # --- Handle Tool Calls ---
                if requested_tool_calls:
                    if not self._toolkit:
                        self._ui.display_warning("LLM requested tools, but no tools are configured. Cannot proceed.")
                        break
                    tool_results_messages: List[Dict[str, Any]] = []
                    for tool_call in requested_tool_calls:
                        result_dict = self._tool_executor.execute_tool(tool_call)
                        tool_results_messages.append(result_dict)
                        self._ui.display_message(role="tool", tool_call_id=result_dict['tool_call_id'], tool_name=result_dict['name'], content=result_dict['content'])
                        # Add tool result to log string
                        conversation_log_parts.append(f"TOOL RESULT ({result_dict['name']}):\n{result_dict['content']}")
                    # Add results to history
                    current_loop_history.extend(tool_results_messages)
                    # Continue loop
                else:
                    # No tool calls, conversation turn complete
                    break # Exit loop normally

            # --- End of Loop ---
            if turn_counter >= max_turns:
                self._ui.display_max_turns_reached(max_turns)

        except (APIError, RuntimeError, ValueError) as e:
             # Catch errors raised during API calls, tool execution, or validation
             self._ui.display_error(f"Error during llm_call execution: {e}")
             # Re-raise the caught error to the caller
             raise
        except Exception as e:
             # Catch any other unexpected errors
             self._ui.display_error(f"Unexpected error in llm_call: {e}", show_locals=True)
             raise RuntimeError(f"Unexpected error in llm_call: {e}") from e

        # --- Return Concatenated Log String ---
        return "\n\n".join(part.strip() for part in conversation_log_parts if part.strip()).strip()

# Note: No __main__ section included as requested.
