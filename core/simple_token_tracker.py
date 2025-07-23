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
        self.filestats: Dict[str, Dict[str, int]] = {}
        
    def start_file(self, filename: str) -> None:
        """
        Initialize tracking for a new file execution.
        
        Args:
            filename: The name/path of the .md file being executed
        """
        self.filestats[filename] = {
            "file_input_tokens": 0,
            "file_output_tokens": 0
        }
    
    def record_llm_call(self, filename: str, input_tokens: int, output_tokens: int, turn_info: str = "") -> None:
        """
        Record tokens from an LLM API call.
        
        Args:
            filename: The name/path of the .md file being executed
            input_tokens: Actual prompt_tokens from API response (what we pay for input)
            output_tokens: Actual completion_tokens from API response (what we pay for output)
            turn_info: Optional information about tool turns (e.g., "turn 1/3")
        """
        # Ensure file is initialized
        if filename not in self.filestats:
            self.start_file(filename)
            
        # Update file stats
        self.filestats[filename]["file_input_tokens"] += input_tokens
        self.filestats[filename]["file_output_tokens"] += output_tokens
        
        # Update global stats
        self.global_input_tokens += input_tokens
        self.global_output_tokens += output_tokens
        
        # Print current status
        self.print_status(filename, input_tokens, output_tokens, turn_info)
    
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
                file_total = file_input + file_output
                print(f"  📄 {filename}")
                print(f"     Input: {file_input:,} tokens | Output: {file_output:,} tokens | Total: {file_total:,} tokens")
        
        # Session totals
        global_total = self.global_input_tokens + self.global_output_tokens
        print(f"\nSession totals:")
        print(f"  📊 Input: {self.global_input_tokens:,} tokens")
        print(f"  📊 Output: {self.global_output_tokens:,} tokens") 
        print(f"  📊 Total: {global_total:,} tokens")
        
        # Cost estimation (approximate)
        # GPT-4 pricing: ~$30/1M input tokens, ~$60/1M output tokens
        input_cost = (self.global_input_tokens / 1_000_000) * 30
        output_cost = (self.global_output_tokens / 1_000_000) * 60
        total_cost = input_cost + output_cost
        print(f"\nApproximate cost (GPT-4 rates):")
        print(f"  💰 Input cost: ${input_cost:.4f}")
        print(f"  💰 Output cost: ${output_cost:.4f}")
        print(f"  💰 Total cost: ${total_cost:.4f}")
        print("="*60)


# Global instance for use throughout the application
token_tracker = SimpleTokenTracker()