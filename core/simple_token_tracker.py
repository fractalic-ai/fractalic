"""
Simple Token Tracking System for Fractalic

This module implements a straightforward token tracking mechanism that accumulates
actual API usage data from LiteLLM responses without complex calculations or estimations.
"""

from typing import Dict, Any, Optional
import litellm
from core.config import Config


class SimpleTokenTracker:
    """
    Tracks token usage across LLM API calls with simple accumulation.
    
    Maintains global totals and per-file breakdowns based on actual API billing data
    from LiteLLM responses.
    """
    
    def __init__(self):
        """Initialize the token tracker with zero counters."""
        self.global_input_tokens = 0
        self.global_output_tokens = 0
        self.global_cost = 0.0
        self.global_input_cost = 0.0
        self.global_output_cost = 0.0
        self.filestats: Dict[str, Dict[str, Any]] = {}
        self.file_run_counters: Dict[str, int] = {}  # Track run count per file
        self.last_call_data = None  # Store last call data for manual display
        self.processed_calls = set()  # Track processed call IDs to avoid duplicates
        self.current_model = None  # Track the current model being used
        self.debug_mode = False  # Enable for debugging token tracking issues
        self.turn_high_watermarks = {}  # Track highest token counts per turn
        
    def start_file(self, filename: str) -> None:
        """
        Initialize tracking for a new file execution.
        
        Args:
            filename: The name/path of the .md file being executed
        """
        # Increment run counter for this file
        if filename not in self.file_run_counters:
            self.file_run_counters[filename] = 0
        self.file_run_counters[filename] += 1
        
        # Create a unique key with filename and run number
        file_key = f"{filename} (run: {self.file_run_counters[filename]})"
        
        self.filestats[file_key] = {
            "file_input_tokens": 0,
            "file_output_tokens": 0,
            "file_cost": 0.0,
            "file_input_cost": 0.0,
            "file_output_cost": 0.0,
            "original_filename": filename,  # Keep track of original filename
            "run_number": self.file_run_counters[filename]
        }
    
    def record_llm_call(self, filename: str, input_tokens: int, output_tokens: int, turn_info: str = "") -> None:
        """
        Record tokens from an LLM API call (legacy method for backward compatibility).
        
        Args:
            filename: The name/path of the .md file being executed
            input_tokens: Actual prompt_tokens from API response (what we pay for input)
            output_tokens: Actual completion_tokens from API response (what we pay for output)
            turn_info: Optional information about tool turns (e.g., "turn 1/3")
        """
        self.record_llm_call_with_cost(filename, input_tokens, output_tokens, turn_info, 0.0)
        
    def record_llm_call_with_cost(self, filename: str, input_tokens: int, output_tokens: int, turn_info: str = "", actual_cost: float = 0.0) -> None:
        """
        Record tokens and cost from an LLM API call with immediate display.
        
        Args:
            filename: The name/path of the .md file being executed
            input_tokens: Actual prompt_tokens from API response (what we pay for input)
            output_tokens: Actual completion_tokens from API response (what we pay for output)
            turn_info: Optional information about tool turns (e.g., "turn 1/3")
            actual_cost: Actual cost from LiteLLM response
        """
        # Create unique call ID based on core parameters (excluding cost to handle multiple callbacks)
        base_call_id = f"{filename}:{input_tokens}:{output_tokens}:{turn_info}"
        
        # Debug logging for tracking issues
        if self.debug_mode:
            print(f"[DEBUG] Token call: {base_call_id}, cost={actual_cost}, already_processed={base_call_id in self.processed_calls}")
        
        # Check if we've already processed this exact call (by tokens/turn, not cost)
        if base_call_id in self.processed_calls:
            if self.debug_mode:
                print(f"[DEBUG] Skipping duplicate token call: {base_call_id}")
            return  # Skip duplicate
        
        # Mark as processed using base ID (without cost)
        self.processed_calls.add(base_call_id)
        
        self._record_call_data(filename, input_tokens, output_tokens, turn_info, actual_cost)
        self.print_status(filename, input_tokens, output_tokens, turn_info)
        
    def record_llm_call_with_cost_silent(self, filename: str, input_tokens: int, output_tokens: int, turn_info: str = "", actual_cost: float = 0.0) -> None:
        """
        Record tokens and cost from an LLM API call without immediate display.
        
        Args:
            filename: The name/path of the .md file being executed
            input_tokens: Actual prompt_tokens from API response (what we pay for input)
            output_tokens: Actual completion_tokens from API response (what we pay for output)
            turn_info: Optional information about tool turns (e.g., "turn 1/3")
            actual_cost: Actual cost from LiteLLM response
        """
        # Find the current active run key for this filename
        if filename in self.file_run_counters:
            run_number = self.file_run_counters[filename]
            current_file_key = f"{filename} (run: {run_number})"
        else:
            # File not initialized, skip recording (shouldn't happen)
            return
            
        if current_file_key not in self.filestats:
            # File not initialized properly, skip recording
            return
            
        # Create unique turn ID to track turn baselines
        turn_id = f"{filename}:{turn_info}"
        
        # Get baseline stats for this turn (where the turn started from)
        if turn_id not in self.turn_high_watermarks:
            # First callback for this turn - record the baseline
            self.turn_high_watermarks[turn_id] = {
                'baseline_input': self.filestats[current_file_key]["file_input_tokens"],
                'baseline_output': self.filestats[current_file_key]["file_output_tokens"],
                'latest_input': input_tokens,
                'latest_output': output_tokens
            }
        else:
            # Subsequent callback - update latest values
            self.turn_high_watermarks[turn_id]['latest_input'] = input_tokens
            self.turn_high_watermarks[turn_id]['latest_output'] = output_tokens
        
        # Calculate delta from the baseline for this turn
        baseline = self.turn_high_watermarks[turn_id]
        delta_input = max(0, baseline['latest_input'] - baseline['baseline_input'])
        delta_output = max(0, baseline['latest_output'] - baseline['baseline_output'])
        
        # Update file stats to reflect current cumulative values
        self.filestats[current_file_key]["file_input_tokens"] = baseline['baseline_input'] + delta_input
        self.filestats[current_file_key]["file_output_tokens"] = baseline['baseline_output'] + delta_output
        
        # Update global stats to reflect current cumulative values
        expected_global_input = baseline['baseline_input'] + delta_input
        expected_global_output = baseline['baseline_output'] + delta_output
        
        # Only update globals if they're different (to avoid double-counting)
        if self.global_input_tokens != expected_global_input or self.global_output_tokens != expected_global_output:
            # Adjust globals by the difference
            self.global_input_tokens = expected_global_input
            self.global_output_tokens = expected_global_output
            
        # Store last call data for manual display (only if we have actual new tokens)
        if delta_input > 0 or delta_output > 0:
            self.last_call_data = {
                'filename': filename,
                'input_tokens': delta_input,
                'output_tokens': delta_output,
                'turn_info': turn_info,
                'actual_cost': actual_cost
            }
        
    def _record_call_data(self, filename: str, input_tokens: int, output_tokens: int, turn_info: str, actual_cost: float) -> None:
        """Internal method to record call data without display."""
        # Find the current active run key for this filename
        current_file_key = None
        if filename in self.file_run_counters:
            run_number = self.file_run_counters[filename]
            current_file_key = f"{filename} (run: {run_number})"
        
        # Ensure file is initialized - if not, initialize it
        if current_file_key is None or current_file_key not in self.filestats:
            self.start_file(filename)
            # Re-get the file key after initialization
            run_number = self.file_run_counters[filename]
            current_file_key = f"{filename} (run: {run_number})"
            
        # Calculate separate input/output costs using typical pricing ratios
        # If we have total cost, estimate input vs output cost (input usually 2x cheaper than output)
        if actual_cost > 0:
            # GPT-4 style ratio: input ~30/1M, output ~60/1M tokens (1:2 ratio)
            total_tokens = input_tokens + output_tokens
            if total_tokens > 0:
                input_cost_ratio = 1.0 / 3.0  # Input is 1/3 of combined cost
                output_cost_ratio = 2.0 / 3.0  # Output is 2/3 of combined cost
                
                # Calculate proportional costs
                input_portion = (input_tokens / total_tokens) if total_tokens > 0 else 0
                output_portion = (output_tokens / total_tokens) if total_tokens > 0 else 0
                
                estimated_input_cost = actual_cost * input_portion * input_cost_ratio / (input_portion * input_cost_ratio + output_portion * output_cost_ratio)
                estimated_output_cost = actual_cost - estimated_input_cost
            else:
                estimated_input_cost = 0.0
                estimated_output_cost = 0.0
        else:
            estimated_input_cost = 0.0
            estimated_output_cost = 0.0
            
        # Update file stats using the current run's key
        self.filestats[current_file_key]["file_input_tokens"] += input_tokens
        self.filestats[current_file_key]["file_output_tokens"] += output_tokens
        self.filestats[current_file_key]["file_cost"] += actual_cost
        self.filestats[current_file_key]["file_input_cost"] += estimated_input_cost
        self.filestats[current_file_key]["file_output_cost"] += estimated_output_cost
        
        # Update global stats
        old_global_input = self.global_input_tokens
        old_global_output = self.global_output_tokens
        
        self.global_input_tokens += input_tokens
        self.global_output_tokens += output_tokens
        self.global_cost += actual_cost
        self.global_input_cost += estimated_input_cost
        self.global_output_cost += estimated_output_cost
        
        # Debug massive token jumps (only if debug mode enabled)
        if self.debug_mode and (input_tokens > 50000 or (old_global_input > 0 and input_tokens > old_global_input)):
            print(f"\n[TOKEN ANOMALY DETECTED]")
            print(f"  Previous global: {old_global_input}")
            print(f"  Adding: {input_tokens}")
            print(f"  New global: {self.global_input_tokens}")
            print(f"  File: {filename}, Turn: {turn_info}")
            print(f"  Call stack:")
            import traceback
            traceback.print_stack(limit=10)
    
    def get_model_context_size(self, model: str) -> Optional[int]:
        """
        Get the context size for a given model using LiteLLM.
        
        Args:
            model: The model name (e.g., "openai/gpt-4o-mini", "anthropic/claude-3-sonnet")
            
        Returns:
            The maximum context size in tokens, or None if not available
        """
        try:
            model_info = litellm.get_model_info(model)
            
            # Handle both dict and object responses
            if isinstance(model_info, dict):
                # Use max_input_tokens for context monitoring, fallback to max_tokens
                max_tokens = model_info.get('max_input_tokens') or model_info.get('max_tokens')
            else:
                # Use max_input_tokens for context monitoring, fallback to max_tokens
                max_tokens = getattr(model_info, 'max_input_tokens', None) or getattr(model_info, 'max_tokens', None)
            
            return max_tokens
        except Exception:
            # If model info is not available, return None
            return None
    
    def get_default_model(self) -> str:
        """
        Get the default model from settings.
        
        Returns:
            The default model name from TOML_SETTINGS
        """
        if Config.TOML_SETTINGS:
            return Config.TOML_SETTINGS.get("defaultProvider", "unknown")
        return "unknown"
    
    def print_status(self, filename: str, llm_input: int, llm_output: int, turn_info: str = "") -> None:
        """
        Print compact token usage after each LLM call.
        
        Args:
            filename: The name/path of the .md file being executed
            llm_input: Input tokens from this specific call
            llm_output: Output tokens from this specific call
            turn_info: Optional information about tool turns
        """
        # Find the current active run key for this filename
        current_file_key = None
        if filename in self.file_run_counters:
            run_number = self.file_run_counters[filename]
            current_file_key = f"{filename} (run: {run_number})"
        
        # If we can't find the file key, initialize and try again
        if current_file_key is None or current_file_key not in self.filestats:
            self.start_file(filename)
            run_number = self.file_run_counters[filename]
            current_file_key = f"{filename} (run: {run_number})"
        
        file_input = self.filestats[current_file_key]["file_input_tokens"]
        file_output = self.filestats[current_file_key]["file_output_tokens"]
        
        # Get context fill information - current conversation size (input + output for this call)
        context_fill_count = llm_input + llm_output
        
        # Use current model if available, otherwise fall back to default
        model_to_use = self.current_model or self.get_default_model()
        context_size = self.get_model_context_size(model_to_use)
        
        if context_size:
            context_fill_percent = int((context_fill_count / context_size) * 100)
            context_info = f" | context: {context_fill_count} / {context_size} ({context_fill_percent}% fill)"
        else:
            context_info = f" | context: {context_fill_count} / unknown (unknown% fill)"
        
        # Compact gray display with context information
        print(f"\033[90mTokens result in/out: +{llm_input}/+{llm_output} (total in/out: {file_input}/{file_output} (file); {self.global_input_tokens}/{self.global_output_tokens} (session)){context_info}\033[0m")
    
    def record_llm_call_direct(self, filename: str, input_tokens: int, output_tokens: int, turn_info: str = "") -> None:
        """
        Display tokens without updating counters (callback already handled the counting).
        
        Args:
            filename: The name/path of the .md file being executed
            input_tokens: Actual prompt_tokens from API response 
            output_tokens: Actual completion_tokens from API response
            turn_info: Optional information about tool turns
        """
        # Just display the tokens, don't update counters (callback already did)
        self.print_status(filename, input_tokens, output_tokens, turn_info)
    
    def print_last_call_status(self) -> None:
        """Print status for the last recorded call (for manual display after tool execution)."""
        if self.last_call_data:
            data = self.last_call_data
            # Add newline before token display for proper spacing
            print()
            self.print_status(data['filename'], data['input_tokens'], data['output_tokens'], data['turn_info'])
            self.last_call_data = None  # Clear after display
    
    def print_pre_call_info(self, filename: str, has_tools: bool, tool_count: int = 0, turn_number: int = 1, max_turns: int = 1, 
                           messages=None, model=None, tools=None) -> None:
        """
        Print information before making an LLM call, including LiteLLM token estimation.
        
        Args:
            filename: The name/path of the .md file being executed
            has_tools: Whether tools are enabled for this call
            tool_count: Number of tools available
            turn_number: Current turn in tool conversation
            max_turns: Maximum turns allowed
            messages: Messages to be sent to LLM (for token estimation)
            model: Model name for token estimation
            tools: Tools schema for token estimation
        """
        # Remove verbose pre-call display - just continue with existing Fractalic output
    
    def get_file_stats(self, filename: str) -> Dict[str, int]:
        """
        Get token statistics for a specific file (current run).
        
        Args:
            filename: The name/path of the .md file
            
        Returns:
            Dictionary with file_input_tokens and file_output_tokens for current run
        """
        # Find the current active run key for this filename
        if filename in self.file_run_counters:
            run_number = self.file_run_counters[filename]
            current_file_key = f"{filename} (run: {run_number})"
            if current_file_key in self.filestats:
                stats = self.filestats[current_file_key].copy()
                # Return simplified format for backwards compatibility
                return {
                    "file_input_tokens": stats["file_input_tokens"],
                    "file_output_tokens": stats["file_output_tokens"]
                }
        return {"file_input_tokens": 0, "file_output_tokens": 0}
    
    def get_global_stats(self) -> Dict[str, int]:
        """
        Get global token statistics.
        
        Returns:
            Dictionary with global_input_tokens and global_output_tokens
        """
        return {
            "global_input_tokens": self.global_input_tokens,
            "global_output_tokens": self.global_output_tokens
        }
    
    def get_all_file_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get token statistics for all files.
        
        Returns:
            Dictionary mapping filenames to their token statistics
        """
        return self.filestats.copy()
    
    def print_session_summary(self) -> None:
        """
        Print detailed session summary at the end of execution.
        """
        print("\n" + "="*60)
        print("TOKEN USAGE SUMMARY")
        print("="*60)
        
        # Per-file breakdown
        if self.filestats:
            print("Per-file breakdown:")
            for filename, stats in self.filestats.items():
                file_input = stats["file_input_tokens"]
                file_output = stats["file_output_tokens"]
                file_cost = stats.get("file_cost", 0.0)
                file_input_cost = stats.get("file_input_cost", 0.0)
                file_output_cost = stats.get("file_output_cost", 0.0)
                file_total = file_input + file_output
                print(f"  📄 {filename}")
                print(f"     Input: {file_input:,} tokens | Output: {file_output:,} tokens | Total: {file_total:,} tokens")
                if file_cost > 0:
                    print(f"     Cost: ${file_input_cost:.6f} (input) + ${file_output_cost:.6f} (output) = ${file_cost:.6f}")
                else:
                    print(f"     Cost: N/A")
        
        # Session totals
        global_total = self.global_input_tokens + self.global_output_tokens
        print(f"\nSession totals:")
        print(f"  📊 Input: {self.global_input_tokens:,} tokens")
        print(f"  📊 Output: {self.global_output_tokens:,} tokens") 
        print(f"  📊 Total: {global_total:,} tokens")
        
        # Actual cost from LiteLLM
        if hasattr(self, 'global_cost') and self.global_cost > 0:
            print(f"\nActual cost (from LiteLLM):")
            if hasattr(self, 'global_input_cost') and hasattr(self, 'global_output_cost'):
                print(f"  💰 Input cost: ${self.global_input_cost:.6f}")
                print(f"  💰 Output cost: ${self.global_output_cost:.6f}")
                print(f"  💰 Total cost: ${self.global_cost:.6f}")
            else:
                print(f"  💰 Total cost: ${self.global_cost:.6f}")
        else:
            print(f"\nCost tracking:")
            print(f"  💰 N/A (no pricing data available)")
        print("="*60)


# Global instance for use throughout the application
token_tracker = SimpleTokenTracker()