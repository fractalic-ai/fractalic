"""
Token usage tracking module for fractalic.

Provides session-level aggregation and reporting of LLM token usage
across all operations and sources.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class UsageRecord:
    """Individual operation usage record."""
    operation_id: str
    timestamp: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tool_calls_count: int = 0
    turns_count: int = 1
    operation_type: str = "llm_call"
    source_file: Optional[str] = None


class TokenTracker:
    """
    Singleton class for tracking token usage across all LLM operations.
    
    Aggregates usage data from all sources (.md files, agents, operations)
    and provides session-level reporting and export capabilities.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.session_usage: List[UsageRecord] = []
        self.session_start = datetime.now()
        self._initialized = True
    
    def log_operation(self, usage_data: dict, operation_id: str = None, 
                     model: str = None, operation_type: str = "llm_call",
                     source_file: str = None):
        """
        Log a single LLM operation's token usage.
        
        Args:
            usage_data: Dictionary containing token usage counts
            operation_id: Unique identifier for the operation
            model: Model name used for the operation
            operation_type: Type of operation (llm_call, tool_call, etc.)
            source_file: Source file or context for the operation
        """
        record = UsageRecord(
            operation_id=operation_id or f"op_{len(self.session_usage)}",
            timestamp=datetime.now().isoformat(),
            model=model or "unknown",
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
            tool_calls_count=usage_data.get("tool_calls_count", 0),
            turns_count=usage_data.get("turns_count", 1),
            operation_type=operation_type,
            source_file=source_file
        )
        self.session_usage.append(record)
    
    def get_session_total(self) -> dict:
        """
        Get aggregated usage totals for the entire session.
        
        Returns:
            Dictionary with total usage counts and metadata
        """
        total = {
            "prompt_tokens": sum(r.prompt_tokens for r in self.session_usage),
            "completion_tokens": sum(r.completion_tokens for r in self.session_usage),
            "total_tokens": sum(r.total_tokens for r in self.session_usage),
            "tool_calls_count": sum(r.tool_calls_count for r in self.session_usage),
            "operations_count": len(self.session_usage),
            "session_duration": str(datetime.now() - self.session_start)
        }
        return total
    
    def print_session_summary(self):
        """Print a comprehensive session usage summary to console."""
        total = self.get_session_total()
        print(f"\n=== SESSION TOKEN USAGE SUMMARY ===")
        print(f"Total Prompt Tokens: {total['prompt_tokens']:,}")
        print(f"Total Completion Tokens: {total['completion_tokens']:,}")
        print(f"Total Tokens: {total['total_tokens']:,}")
        print(f"Total Tool Calls: {total['tool_calls_count']}")
        print(f"Total LLM Operations: {total['operations_count']}")
        print(f"Session Duration: {total['session_duration']}")
        
        # Breakdown by source file
        by_file = {}
        for record in self.session_usage:
            file = record.source_file or "unknown"
            if file not in by_file:
                by_file[file] = {"tokens": 0, "operations": 0}
            by_file[file]["tokens"] += record.total_tokens
            by_file[file]["operations"] += 1
        
        if len(by_file) > 1:
            print(f"\nBreakdown by source:")
            for file, stats in by_file.items():
                print(f"  {file}: {stats['tokens']:,} tokens, {stats['operations']} operations")
        
        # Breakdown by model
        by_model = {}
        for record in self.session_usage:
            model = record.model
            if model not in by_model:
                by_model[model] = {"tokens": 0, "operations": 0}
            by_model[model]["tokens"] += record.total_tokens
            by_model[model]["operations"] += 1
        
        if len(by_model) > 1:
            print(f"\nBreakdown by model:")
            for model, stats in by_model.items():
                print(f"  {model}: {stats['tokens']:,} tokens, {stats['operations']} operations")
    
    def export_usage_report(self, format="json") -> str:
        """
        Export usage data in the specified format.
        
        Args:
            format: Export format ("json" or "csv")
            
        Returns:
            Formatted string containing usage data
        """
        if format == "json":
            return json.dumps([asdict(r) for r in self.session_usage], indent=2)
        elif format == "csv":
            if not self.session_usage:
                return "No usage data to export"
            
            # CSV header
            header = "operation_id,timestamp,model,prompt_tokens,completion_tokens,total_tokens,tool_calls_count,turns_count,operation_type,source_file\n"
            
            # CSV rows
            rows = []
            for record in self.session_usage:
                row = f"{record.operation_id},{record.timestamp},{record.model},{record.prompt_tokens},{record.completion_tokens},{record.total_tokens},{record.tool_calls_count},{record.turns_count},{record.operation_type},{record.source_file or ''}"
                rows.append(row)
            
            return header + "\n".join(rows)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_usage_by_file(self) -> Dict[str, dict]:
        """Get usage breakdown by source file."""
        by_file = {}
        for record in self.session_usage:
            file = record.source_file or "unknown"
            if file not in by_file:
                by_file[file] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "tool_calls_count": 0,
                    "operations_count": 0
                }
            by_file[file]["prompt_tokens"] += record.prompt_tokens
            by_file[file]["completion_tokens"] += record.completion_tokens
            by_file[file]["total_tokens"] += record.total_tokens
            by_file[file]["tool_calls_count"] += record.tool_calls_count
            by_file[file]["operations_count"] += 1
        return by_file
    
    def get_usage_by_model(self) -> Dict[str, dict]:
        """Get usage breakdown by model."""
        by_model = {}
        for record in self.session_usage:
            model = record.model
            if model not in by_model:
                by_model[model] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "tool_calls_count": 0,
                    "operations_count": 0
                }
            by_model[model]["prompt_tokens"] += record.prompt_tokens
            by_model[model]["completion_tokens"] += record.completion_tokens
            by_model[model]["total_tokens"] += record.total_tokens
            by_model[model]["tool_calls_count"] += record.tool_calls_count
            by_model[model]["operations_count"] += 1
        return by_model
    
    def reset_session(self):
        """Reset the session tracking data."""
        self.session_usage.clear()
        self.session_start = datetime.now()


# Global instance for easy access
token_tracker = TokenTracker()
