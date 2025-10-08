"""
Simplified Token Tracking System for Fractalic

Replaces previous complex incremental/deduplicating logic with a straightforward
approach:

- Every LiteLLM completion call immediately prints its +in/+out (delta) tokens
- Aggregates are maintained for the session and per file run
- Each @run (or fractalic_run tool) execution of the same file is tracked
  distinctly via a (run: N) suffix
- Per-file call history retained (calls list) for potential future introspection
- Handles providers (LiteLLM) that return cumulative usage by converting to deltas

Backward compatibility:
- start_file(...) retained
- get_file_stats / get_all_file_stats / get_global_stats retained
- print_session_summary retained (simplified)
- Legacy methods record_llm_call_with_cost_silent / print_last_call_status kept
  as no-ops or thin wrappers to new logic to avoid import/runtime errors elsewhere
"""

from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class CallRecord:
    call_index: int
    model: str
    input_tokens: int  # delta for this call
    output_tokens: int  # delta for this call
    total_input_seen: int  # cumulative as reported by provider after this call (per file)
    total_output_seen: int
    timestamp: str
    schema_adjustment: int = 0  # extra prompt tokens added for tool schema not billed/reported
    # Cost tracking (from LiteLLM)
    response_cost: float = 0.0  # total cost in USD
    input_cost: float = 0.0  # input/prompt cost in USD
    output_cost: float = 0.0  # output/completion cost in USD
    tool_usage_cost: float = 0.0  # tool usage cost in USD


class SimpleTokenTracker:
    def __init__(self):
        self.session_start = datetime.utcnow()
        self.global_input_tokens = 0
        self.global_output_tokens = 0
        self.global_cost = 0.0  # Total session cost in USD
        self.filestats: Dict[str, Dict[str, Any]] = {}
        self.file_run_counters: Dict[str, int] = {}
        self.current_model = None  # Updated by caller (openai client)

    # -------- core public API --------
    def start_file(self, filename: str) -> None:
        if filename not in self.file_run_counters:
            self.file_run_counters[filename] = 0
        self.file_run_counters[filename] += 1
        file_key = self._current_file_key(filename)
        self.filestats[file_key] = {
            "file_input_tokens": 0,            # aggregated deltas
            "file_output_tokens": 0,           # aggregated deltas
            "file_cost": 0.0,                  # aggregated cost in USD
            "last_seen_input": 0,              # last cumulative reported
            "last_seen_output": 0,             # last cumulative reported
            "calls": []  # list[CallRecord]
        }

    def record_call(
        self,
        filename: str,
        model: str,
        reported_input_tokens: int,
        reported_output_tokens: int,
        *,
        schema_adjustment: int = 0,
        response_cost: float = 0.0,
        input_cost: float = 0.0,
        output_cost: float = 0.0,
        tool_usage_cost: float = 0.0
    ) -> None:
        """Record an LLM API call.

        reported_* values may be per-call or cumulative; we convert to delta
        relative to last seen per file run.
        schema_adjustment: additional prompt tokens attributable to tool/function schema
                            that the provider did not report but we include for realistic context size.
        response_cost: total cost in USD (from LiteLLM)
        input_cost: input/prompt cost in USD (from LiteLLM cost_breakdown)
        output_cost: output/completion cost in USD (from LiteLLM cost_breakdown)
        tool_usage_cost: tool usage cost in USD (from LiteLLM cost_breakdown)
        """
        if filename not in self.file_run_counters:
            self.start_file(filename)
        file_key = self._current_file_key(filename)
        if file_key not in self.filestats:
            self.start_file(filename)
            file_key = self._current_file_key(filename)

        stats = self.filestats[file_key]
        last_in = stats["last_seen_input"]
        last_out = stats["last_seen_output"]

        # Compute deltas (protect against negative or zero regressions)
        delta_in = reported_input_tokens - last_in
        delta_out = reported_output_tokens - last_out
        if delta_in < 0 or reported_input_tokens < last_in:
            # Reset scenario; treat as fresh values
            delta_in = reported_input_tokens
        if delta_out < 0 or reported_output_tokens < last_out:
            delta_out = reported_output_tokens

        # If provider already sends per-call deltas, last_seen will be 0; works fine
        # Update last seen to the reported cumulative values
        stats["last_seen_input"] = reported_input_tokens
        stats["last_seen_output"] = reported_output_tokens

        # Update aggregates with deltas only
        if delta_in > 0:
            stats["file_input_tokens"] += delta_in
            self.global_input_tokens += delta_in
        if schema_adjustment > 0:
            # Count schema adjustment toward input totals
            stats["file_input_tokens"] += schema_adjustment
            self.global_input_tokens += schema_adjustment
        if delta_out > 0:
            stats["file_output_tokens"] += delta_out
            self.global_output_tokens += delta_out

        # Update cost aggregates
        if response_cost > 0:
            stats["file_cost"] += response_cost
            self.global_cost += response_cost

        call_index = len(stats["calls"]) + 1
        stats["calls"].append(
            CallRecord(
                call_index=call_index,
                model=model,
                input_tokens=delta_in + schema_adjustment,
                output_tokens=delta_out,
                total_input_seen=reported_input_tokens,
                total_output_seen=reported_output_tokens,
                timestamp=datetime.utcnow().isoformat(),
                schema_adjustment=schema_adjustment,
                response_cost=response_cost,
                input_cost=input_cost,
                output_cost=output_cost,
                tool_usage_cost=tool_usage_cost
            )
        )

        # Immediate compact print (show deltas with cost)
        fin = stats["file_input_tokens"]
        fout = stats["file_output_tokens"]
        fcost = stats["file_cost"]
        schema_part = f" +schema {schema_adjustment}" if schema_adjustment > 0 else ""

        # Format cost display with improved readability
        if response_cost > 0:
            # Format individual token costs
            in_cost_str = f"${input_cost:.6f}" if input_cost > 0 else "$0"
            out_cost_str = f"${output_cost:.6f}" if output_cost > 0 else "$0"

            # Build token display with inline costs
            token_display = f"+{delta_in}{schema_part} ({in_cost_str}) / +{delta_out} ({out_cost_str})"

            # Add tool cost if present
            if tool_usage_cost > 0:
                token_display += f" [tool: ${tool_usage_cost:.6f}]"

            file_cost_display = f" | ${fcost:.6f}"
            session_cost_display = f" | ${self.global_cost:.6f}"
        else:
            token_display = f"+{delta_in}{schema_part}/+{delta_out}"
            file_cost_display = ""
            session_cost_display = ""

        print(
            f"\033[90mTOKENS +in/+out: {token_display} | "
            f"file total: {fin}/{fout}{file_cost_display} | "
            f"session total: {self.global_input_tokens}/{self.global_output_tokens}{session_cost_display} | "
            f"file: {file_key} | model: {model}\033[0m"
        )

    # -------- summaries / getters --------
    def get_file_stats(self, filename: str) -> Dict[str, Any]:
        file_key = self._current_file_key(filename)
        stats = self.filestats.get(file_key)
        if not stats:
            return {"file_input_tokens": 0, "file_output_tokens": 0, "file_cost": 0.0}
        return {
            "file_input_tokens": stats["file_input_tokens"],
            "file_output_tokens": stats["file_output_tokens"],
            "file_cost": stats.get("file_cost", 0.0),
        }

    def get_all_file_stats(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for k, v in self.filestats.items():
            out[k] = {
                "file_input_tokens": v["file_input_tokens"],
                "file_output_tokens": v["file_output_tokens"],
                "calls": [asdict(c) for c in v["calls"]],
            }
        return out

    def get_global_stats(self) -> Dict[str, Any]:
        return {
            "global_input_tokens": self.global_input_tokens,
            "global_output_tokens": self.global_output_tokens,
            "global_cost": self.global_cost,
        }

    def print_session_summary(self) -> None:
        print("\n" + "=" * 60)
        print("TOKEN USAGE SUMMARY")
        print("=" * 60)
        for file_key, stats in self.filestats.items():
            fin = stats["file_input_tokens"]
            fout = stats["file_output_tokens"]
            total = fin + fout
            fcost = stats.get("file_cost", 0.0)

            # Format cost display
            cost_str = f" | cost ${fcost:.6f}" if fcost > 0 else ""
            print(f"  📄 {file_key}: in {fin:,} | out {fout:,} | total {total:,}{cost_str} | calls {len(stats['calls'])}")

        print("\nSession totals:")
        sin = self.global_input_tokens
        sout = self.global_output_tokens
        print(f"  📊 Input:  {sin:,}")
        print(f"  📊 Output: {sout:,}")
        print(f"  📊 Total:  {sin + sout:,}")

        # Add session cost if available
        if self.global_cost > 0:
            print(f"  💰 Cost:   ${self.global_cost:.6f}")

        print("=" * 60)

    # -------- legacy compatibility (thin wrappers) --------
    def record_llm_call_with_cost_silent(self, filename: str, input_tokens: int, output_tokens: int, turn_info: str = "", actual_cost: float = 0.0):  # noqa: ARG002
        self.record_call(filename, self.current_model or "unknown", input_tokens, output_tokens)

    def record_llm_call(self, filename: str, input_tokens: int, output_tokens: int, turn_info: str = ""):  # noqa: ARG002
        self.record_call(filename, self.current_model or "unknown", input_tokens, output_tokens)

    def record_llm_call_with_cost(self, filename: str, input_tokens: int, output_tokens: int, turn_info: str = "", actual_cost: float = 0.0):  # noqa: ARG002
        self.record_call(filename, self.current_model or "unknown", input_tokens, output_tokens)

    def record_llm_call_direct(self, filename: str, input_tokens: int, output_tokens: int, turn_info: str = ""):  # noqa: ARG002
        self.record_call(filename, self.current_model or "unknown", input_tokens, output_tokens)

    def print_last_call_status(self):
        # Each call prints immediately now
        pass

    def get_last_call_stats(self, filename: str = None) -> dict | None:
        """Get statistics for the last LLM call.

        Returns dict with: model, input_tokens, output_tokens, total_input, total_output,
                          response_cost, input_cost, output_cost, tool_usage_cost
        or None if no calls recorded.
        """
        if filename:
            file_key = self._current_file_key(filename)
            if file_key in self.filestats and self.filestats[file_key]["calls"]:
                last_call = self.filestats[file_key]["calls"][-1]
                return {
                    "model": last_call.model,
                    "input_tokens": last_call.input_tokens,
                    "output_tokens": last_call.output_tokens,
                    "total_input": last_call.total_input_seen,
                    "total_output": last_call.total_output_seen,
                    "timestamp": last_call.timestamp,
                    "response_cost": last_call.response_cost,
                    "input_cost": last_call.input_cost,
                    "output_cost": last_call.output_cost,
                    "tool_usage_cost": last_call.tool_usage_cost
                }

        # If no filename, try to get the most recent call from any file
        all_calls = []
        for stats in self.filestats.values():
            all_calls.extend(stats["calls"])

        if all_calls:
            last_call = all_calls[-1]
            return {
                "model": last_call.model,
                "input_tokens": last_call.input_tokens,
                "output_tokens": last_call.output_tokens,
                "total_input": last_call.total_input_seen,
                "total_output": last_call.total_output_seen,
                "timestamp": last_call.timestamp,
                "response_cost": last_call.response_cost,
                "input_cost": last_call.input_cost,
                "output_cost": last_call.output_cost,
                "tool_usage_cost": last_call.tool_usage_cost
            }

        return None

    def print_pre_call_info(self, *args, **kwargs):  # noqa: D401, ANN001
        return None

    # -------- helpers --------
    def _current_file_key(self, filename: str) -> str:
        if filename not in self.file_run_counters:
            return f"{filename} (run: 0)"  # not started yet
        return f"{filename} (run: {self.file_run_counters[filename]})"


# Global instance
token_tracker = SimpleTokenTracker()