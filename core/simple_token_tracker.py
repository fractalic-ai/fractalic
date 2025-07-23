"""
Simple Token Tracking System for Fractalic

This module implements a straightforward token tracking mechanism that accumulates
actual API usage data from LiteLLM responses without complex calculations or estimations.
"""

from typing import Dict, Any


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
        self.last_call_data = None  # Store last call data for manual display
        self.processed_calls = set()  # Track processed call IDs to avoid duplicates
        
    def start_file(self, filename: str) -> None:
        """
        Initialize tracking for a new file execution.
        
        Args:
            filename: The name/path of the .md file being executed
        """
        self.filestats[filename] = {
            "file_input_tokens": 0,
            "file_output_tokens": 0,
            "file_cost": 0.0,
            "file_input_cost": 0.0,
            "file_output_cost": 0.0
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
        # Create unique call ID to avoid double-counting from streaming callbacks
        call_id = f"{filename}:{input_tokens}:{output_tokens}:{turn_info}:{actual_cost}"
        if call_id in self.processed_calls:
            return  # Skip duplicate
        self.processed_calls.add(call_id)
        
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
        # Create unique call ID to avoid double-counting from streaming callbacks
        call_id = f"{filename}:{input_tokens}:{output_tokens}:{turn_info}:{actual_cost}"
        if call_id in self.processed_calls:
            return  # Skip duplicate
        self.processed_calls.add(call_id)
        
        self._record_call_data(filename, input_tokens, output_tokens, turn_info, actual_cost)
        # Store last call data for manual display later
        self.last_call_data = {
            'filename': filename,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'turn_info': turn_info,
            'actual_cost': actual_cost
        }
        
    def _record_call_data(self, filename: str, input_tokens: int, output_tokens: int, turn_info: str, actual_cost: float) -> None:
        """Internal method to record call data without display."""
        # Ensure file is initialized
        if filename not in self.filestats:
            self.start_file(filename)
            
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
            
        # Update file stats
        self.filestats[filename]["file_input_tokens"] += input_tokens
        self.filestats[filename]["file_output_tokens"] += output_tokens
        self.filestats[filename]["file_cost"] += actual_cost
        self.filestats[filename]["file_input_cost"] += estimated_input_cost
        self.filestats[filename]["file_output_cost"] += estimated_output_cost
        
        # Update global stats
        self.global_input_tokens += input_tokens
        self.global_output_tokens += output_tokens
        self.global_cost += actual_cost
        self.global_input_cost += estimated_input_cost
        self.global_output_cost += estimated_output_cost
    
    def print_status(self, filename: str, llm_input: int, llm_output: int, turn_info: str = "") -> None:
        """
        Print compact token usage after each LLM call.
        
        Args:
            filename: The name/path of the .md file being executed
            llm_input: Input tokens from this specific call
            llm_output: Output tokens from this specific call
            turn_info: Optional information about tool turns
        """
        file_input = self.filestats[filename]["file_input_tokens"]
        file_output = self.filestats[filename]["file_output_tokens"]
        
        # Compact gray display
        print(f"\033[90mTokens result in/out: +{llm_input}/+{llm_output} (total in/out: {file_input}/{file_output} (file); {self.global_input_tokens}/{self.global_output_tokens} (session))\033[0m")
    
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
        Get token statistics for a specific file.
        
        Args:
            filename: The name/path of the .md file
            
        Returns:
            Dictionary with file_input_tokens and file_output_tokens
        """
        if filename not in self.filestats:
            return {"file_input_tokens": 0, "file_output_tokens": 0}
        return self.filestats[filename].copy()
    
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