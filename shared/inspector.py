"""Bounded inspection loop (brief B06): the model may request read-only
tools; each call is validated and run locally; the loop ends on a final
answer, cancellation, the deadline, or an exhausted budget. When it ends
without an answer, the model is asked to answer from what it has and say
what is missing, instead of inventing evidence."""

import json
import time
from dataclasses import dataclass, field
from typing import Callable

from shared.project_context import ContextSource
from shared.project_tools import ProjectTools

MAX_ROUNDS = 6
DEADLINE_S = 30.0
MAX_TOOL_CHARS = 60_000  # total tool output allowed per turn

COMPLETE = "COMPLETE"
INCOMPLETE = "INCOMPLETE"
CANCELLED = "CANCELLED"


@dataclass
class InspectionResult:
    reply: str | None
    status: str
    evidence: list[ContextSource] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    reason: str = ""


def run_inspection(nim, messages: list[dict], tools: ProjectTools, *,
                   max_rounds: int = MAX_ROUNDS, deadline_s: float = DEADLINE_S,
                   is_current: Callable[[], bool] = lambda: True,
                   on_call: Callable[[dict], None] | None = None,
                   max_tokens: int = 900, require_tool: bool = False,
                   final_thinking: bool = False, final_model: str | None = None,
                   final_max_tokens: int = 2000) -> InspectionResult:
    messages = list(messages)
    started = time.monotonic()
    evidence: list[ContextSource] = []
    calls: list[dict] = []
    used_chars = 0
    reason = ""

    for round_no in range(max_rounds):
        if not is_current():
            return InspectionResult(None, CANCELLED, evidence, calls)
        if time.monotonic() - started > deadline_s:
            reason = f"inspection deadline of {deadline_s:.0f}s reached"
            break
        if used_chars > MAX_TOOL_CHARS:
            reason = "inspection context budget exhausted"
            break

        # First hop of an investigative turn must inspect something: the model
        # otherwise answers about code it has not read.
        choice = "required" if (require_tool and round_no == 0) else "auto"
        msg = nim.chat_with_tools(messages, tools.schemas, max_tokens=max_tokens,
                                  thinking=False, tool_choice=choice)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            if not is_current():
                return InspectionResult(None, CANCELLED, evidence, calls)
            content = msg.get("content")
            if content and (final_thinking or final_model):
                # Tool use stays on the fast path; the answer itself is rewritten
                # with thinking on and/or a larger model. Keep the draft if that fails.
                kwargs = {"model": final_model} if final_model else {}
                better = nim.chat_with_tools(messages, None, max_tokens=final_max_tokens,
                                             thinking=final_thinking, **kwargs)
                content = better.get("content") or content
                if not is_current():
                    return InspectionResult(None, CANCELLED, evidence, calls)
            if content:
                return InspectionResult(content, COMPLETE, evidence, calls)
            reason = "model returned an empty answer"
            break

        messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": tool_calls})
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = None  # malformed arguments become an ERROR result, not a crash
            result = tools.call(name, args)
            evidence.extend(result.sources)
            used_chars += len(result.output)
            record = {"tool": name, "args": args, "status": result.status,
                      "truncated": result.truncated, "chars": len(result.output)}
            calls.append(record)
            if on_call:
                on_call(record)
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                             "content": result.output})
    else:
        reason = f"used all {max_rounds} inspection rounds"

    if not is_current():
        return InspectionResult(None, CANCELLED, evidence, calls)
    messages.append({"role": "user", "content": (
        f"Inspection stopped: {reason}. Answer now using only the evidence "
        "already gathered, and list in MISSING what you could not check.")})
    msg = nim.chat_with_tools(messages, None, max_tokens=max_tokens, thinking=False)
    return InspectionResult(msg.get("content") or None, INCOMPLETE, evidence, calls, reason)
