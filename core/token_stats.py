"""
Clean message queue-based token tracking system for Fractalic.

Provides real-time token usage tracking through a simple message queue
with on-demand aggreg        print("=== SESSION TOKEN USAGE SUMMARY ===")
        print(f"Total Prompt Tokens: {summary['total_prompt_tokens']:,}")
        print(f"Total Completion Tokens: {summary['total_completion_tokens']:,}")
        print(f"Total Tokens: {summary['total_tokens']:,}")
        print(f"Total Operations: {summary['operations_count']}")
        print(f"Session Duration: {summary['session_duration']:.2f}s")
        
        # Show file breakdown using get_stats_by_file
        file_stats = self.get_stats_by_file()
        print(f"[DEBUG] File stats: {file_stats}")
        if file_stats:
            print("\n=== FILES BREAKDOWN ===")
            for file_path, stats in file_stats.items():
                print(f"{file_path}: {stats['total_tokens']:,} tokens ({stats['operations_count']} ops)")
        else:
            print("\n[DEBUG] No file stats found")
        print()and flexible reporting.
"""

import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class TokenMessage:
    """Individual token usage message with detailed token breakdown."""
    operation_id: str
    timestamp: float
    model: str
    # Input tokens (what we send to model)
    input_tokens: int  # Total input tokens
    input_context_tokens: int  # Messages, text content
    input_schema_tokens: int  # Tool schemas (MCP + Fractalic tools)
    input_mcp_schema_tokens: int  # MCP tool schemas specifically
    input_fractalic_schema_tokens: int  # Fractalic tool schemas specifically
    # Output tokens (what model generates)
    output_tokens: int  # Completion tokens from model
    total_tokens: int  # Total tokens for this operation
    # Tool information
    tool_calls_count: int  # Number of tool calls made
    tools_available_count: int  # Number of tools available/sent in schema
    operation_type: str  # "llm_turn", "llm_final", "@run", "tool_call", etc.
    source_file: str
    metadata: Optional[Dict[str, Any]] = None


class TokenStatsQueue:
    """
    Message queue-based token tracking system.
    
    Simple interface: send messages, get aggregates on-demand.
    No complex state management or singletons.
    """
    
    def __init__(self):
        self.messages: List[TokenMessage] = []
        self.session_start = time.time()
    
    def send(self, message: TokenMessage):
        """Send a token usage message to the queue."""
        self.messages.append(message)
    
    def send_usage(self, 
                   operation_id: str,
                   model: str,
                   input_tokens: int,
                   input_context_tokens: int,
                   input_schema_tokens: int,
                   input_mcp_schema_tokens: int,
                   input_fractalic_schema_tokens: int,
                   output_tokens: int,
                   total_tokens: int,
                   tool_calls_count: int,
                   tools_available_count: int,
                   operation_type: str,
                   source_file: str,
                   metadata: Optional[Dict[str, Any]] = None):
        """Convenience method to send detailed token usage without creating TokenMessage."""
        message = TokenMessage(
            operation_id=operation_id,
            timestamp=time.time(),
            model=model,
            input_tokens=input_tokens,
            input_context_tokens=input_context_tokens,
            input_schema_tokens=input_schema_tokens,
            input_mcp_schema_tokens=input_mcp_schema_tokens,
            input_fractalic_schema_tokens=input_fractalic_schema_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            tool_calls_count=tool_calls_count,
            tools_available_count=tools_available_count,
            operation_type=operation_type,
            source_file=source_file,
            metadata=metadata
        )
        self.send(message)
    
    def send_usage_legacy(self, 
                         operation_id: str,
                         model: str,
                         prompt_tokens: int,
                         completion_tokens: int,
                         total_tokens: int,
                         operation_type: str,
                         source_file: str,
                         metadata: Optional[Dict[str, Any]] = None):
        """Legacy method for backward compatibility - assumes all prompt tokens are context."""
        self.send_usage(
            operation_id=operation_id,
            model=model,
            input_tokens=prompt_tokens,
            input_context_tokens=prompt_tokens,  # Assume all prompt tokens are context
            input_schema_tokens=0,  # No schema breakdown available
            input_mcp_schema_tokens=0,  # No MCP breakdown available
            input_fractalic_schema_tokens=0,  # No Fractalic breakdown available
            output_tokens=completion_tokens,
            total_tokens=total_tokens,
            tool_calls_count=0,  # Legacy calls don't track tool usage
            tools_available_count=0,  # Legacy calls don't track available tools
            operation_type=operation_type,
            source_file=source_file,
            metadata=metadata
        )
    
    def get_cumulative_total(self) -> int:
        """Get total tokens used across all operations."""
        return sum(msg.total_tokens for msg in self.messages)
    
    def get_cumulative_by_type(self, operation_type: str) -> int:
        """Get total tokens for specific operation type."""
        return sum(msg.total_tokens for msg in self.messages 
                  if msg.operation_type == operation_type)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session statistics with detailed token breakdown."""
        if not self.messages:
            return {
                "total_input_tokens": 0,
                "total_input_context_tokens": 0,
                "total_input_schema_tokens": 0,
                "total_input_mcp_schema_tokens": 0,
                "total_input_fractalic_schema_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "total_tool_calls": 0,
                "total_tools_available": 0,
                "operations_count": 0,
                "session_duration": time.time() - self.session_start,
                "models_used": [],
                "operation_types": []
            }
        
        return {
            "total_input_tokens": sum(msg.input_tokens for msg in self.messages),
            "total_input_context_tokens": sum(msg.input_context_tokens for msg in self.messages),
            "total_input_schema_tokens": sum(msg.input_schema_tokens for msg in self.messages),
            "total_input_mcp_schema_tokens": sum(msg.input_mcp_schema_tokens for msg in self.messages),
            "total_input_fractalic_schema_tokens": sum(msg.input_fractalic_schema_tokens for msg in self.messages),
            "total_output_tokens": sum(msg.output_tokens for msg in self.messages),
            "total_tokens": sum(msg.total_tokens for msg in self.messages),
            "total_tool_calls": sum(msg.tool_calls_count for msg in self.messages),
            "total_tools_available": sum(msg.tools_available_count for msg in self.messages),
            "operations_count": len(self.messages),
            "session_duration": time.time() - self.session_start,
            "models_used": list(set(msg.model for msg in self.messages)),
            "operation_types": list(set(msg.operation_type for msg in self.messages))
        }
    
    def get_stats_by_file(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed token usage breakdown by source file."""
        stats = {}
        for msg in self.messages:
            file = msg.source_file
            if file not in stats:
                stats[file] = {
                    "input_tokens": 0,
                    "input_context_tokens": 0,
                    "input_schema_tokens": 0,
                    "input_mcp_schema_tokens": 0,
                    "input_fractalic_schema_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "tool_calls_count": 0,
                    "tools_available_count": 0,
                    "operations_count": 0,
                    "models": set(),
                    "operation_types": set()
                }
            
            stats[file]["input_tokens"] += msg.input_tokens
            stats[file]["input_context_tokens"] += msg.input_context_tokens
            stats[file]["input_schema_tokens"] += msg.input_schema_tokens
            stats[file]["input_mcp_schema_tokens"] += msg.input_mcp_schema_tokens
            stats[file]["input_fractalic_schema_tokens"] += msg.input_fractalic_schema_tokens
            stats[file]["output_tokens"] += msg.output_tokens
            stats[file]["total_tokens"] += msg.total_tokens
            stats[file]["tool_calls_count"] += msg.tool_calls_count
            stats[file]["tools_available_count"] += msg.tools_available_count
            stats[file]["operations_count"] += 1
            stats[file]["models"].add(msg.model)
            stats[file]["operation_types"].add(msg.operation_type)
        
        # Convert sets to lists for JSON serialization
        for file_stats in stats.values():
            file_stats["models"] = list(file_stats["models"])
            file_stats["operation_types"] = list(file_stats["operation_types"])
        
        return stats
    
    def get_stats_by_model(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed token usage breakdown by model."""
        stats = {}
        for msg in self.messages:
            model = msg.model
            if model not in stats:
                stats[model] = {
                    "input_tokens": 0,
                    "input_context_tokens": 0,
                    "input_schema_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "operations_count": 0,
                    "files": set(),
                    "operation_types": set()
                }
            
            stats[model]["input_tokens"] += msg.input_tokens
            stats[model]["input_context_tokens"] += msg.input_context_tokens
            stats[model]["input_schema_tokens"] += msg.input_schema_tokens
            stats[model]["output_tokens"] += msg.output_tokens
            stats[model]["total_tokens"] += msg.total_tokens
            stats[model]["operations_count"] += 1
            stats[model]["files"].add(msg.source_file)
            stats[model]["operation_types"].add(msg.operation_type)
        
        # Convert sets to lists for JSON serialization
        for model_stats in stats.values():
            model_stats["files"] = list(model_stats["files"])
            model_stats["operation_types"] = list(model_stats["operation_types"])
        
        return stats
    
    def print_session_summary(self):
        """Print comprehensive session usage summary with detailed token breakdown."""
        summary = self.get_session_summary()
        
        print(f"\n=== SESSION TOKEN USAGE SUMMARY ===")
        print(f"Total Input Tokens: {summary['total_input_tokens']:,}")
        print(f"  - Context Tokens: {summary['total_input_context_tokens']:,}")
        print(f"  - Schema Tokens: {summary['total_input_schema_tokens']:,}")
        print(f"    • MCP Tools: {summary['total_input_mcp_schema_tokens']:,}")
        print(f"    • Fractalic Tools: {summary['total_input_fractalic_schema_tokens']:,}")
        print(f"Total Output Tokens: {summary['total_output_tokens']:,}")
        print(f"Total Tokens: {summary['total_tokens']:,}")
        print(f"Total Tool Calls: {summary['total_tool_calls']}")
        print(f"Total Tools Available: {summary['total_tools_available']}")
        print(f"Total Operations: {summary['operations_count']}")
        print(f"Session Duration: {summary['session_duration']:.2f}s")
        
        # Breakdown by file (always show, even for single file)
        file_stats = self.get_stats_by_file()
        if file_stats:
            print(f"\nBreakdown by source:")
            for file, stats in file_stats.items():
                schema_breakdown = f"(Context: {stats['input_context_tokens']:,}, MCP: {stats['input_mcp_schema_tokens']:,}, Fractalic: {stats['input_fractalic_schema_tokens']:,})"
                print(f"  {file}: {stats['total_tokens']:,} tokens, {stats['operations_count']} operations")
                print(f"    Input: {stats['input_tokens']:,} {schema_breakdown}, Output: {stats['output_tokens']:,}")
                print(f"    Tool Calls: {stats['tool_calls_count']}, Tools Available: {stats['tools_available_count']}")
        
        # Breakdown by model
        model_stats = self.get_stats_by_model()
        if len(model_stats) > 1:
            print(f"\nBreakdown by model:")
            for model, stats in model_stats.items():
                print(f"  {model}: {stats['total_tokens']:,} tokens, {stats['operations_count']} operations")
    
    def export_messages(self, format: str = "json") -> str:
        """Export all messages in specified format with detailed token breakdown."""
        if format == "json":
            import json
            return json.dumps([asdict(msg) for msg in self.messages], indent=2)
        elif format == "csv":
            if not self.messages:
                return "No messages to export"
            
            # CSV header with detailed token columns
            header = "operation_id,timestamp,model,input_tokens,input_context_tokens,input_schema_tokens,output_tokens,total_tokens,operation_type,source_file\n"
            
            # CSV rows
            rows = []
            for msg in self.messages:
                timestamp_str = datetime.fromtimestamp(msg.timestamp).isoformat()
                row = f"{msg.operation_id},{timestamp_str},{msg.model},{msg.input_tokens},{msg.input_context_tokens},{msg.input_schema_tokens},{msg.output_tokens},{msg.total_tokens},{msg.operation_type},{msg.source_file}"
                rows.append(row)
            
            return header + "\n".join(rows)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def reset(self):
        """Reset the queue (clear all messages)."""
        self.messages.clear()
        self.session_start = time.time()


# Global instance for easy access throughout the application
token_stats = TokenStatsQueue()
