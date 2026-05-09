import json
from api.tools.base import ToolResult, FAILURE_EMPTY, FAILURE_MALFORMED
from api.context import SharedContext, AgentID


async def run(ctx: SharedContext, requesting_agent: str) -> ToolResult:
    if not requesting_agent or not isinstance(requesting_agent, str):
        return ToolResult(
            success=False,
            failure_mode=FAILURE_MALFORMED,
            error_message="requesting_agent must be a non-empty string",
        )

    previous_output = ctx.agent_outputs.get(requesting_agent)
    if not previous_output:
        return ToolResult(
            success=False,
            failure_mode=FAILURE_EMPTY,
            error_message=f"no previous output found for agent: {requesting_agent}",
        )

    # collect all outputs for cross-agent contradiction check
    all_outputs = {
        agent_id: output.output
        for agent_id, output in ctx.agent_outputs.items()
    }

    contradictions = _find_contradictions(
        requesting_agent,
        previous_output.output,
        all_outputs,
    )

    return ToolResult(
        success=True,
        data={
            "previous_output": previous_output.output,
            "token_count": previous_output.token_count,
            "contradictions_found": contradictions,
        },
    )


def _find_contradictions(
    agent_id: str,
    own_output: str,
    all_outputs: dict[str, str],
) -> list[dict]:
    contradictions = []
    own_lower = own_output.lower()

    for other_agent, other_output in all_outputs.items():
        if other_agent == agent_id:
            continue
        other_lower = other_output.lower()

        # simple heuristic: flag if negation patterns appear near shared keywords
        own_words = set(own_lower.split())
        other_words = set(other_lower.split())
        shared = own_words & other_words

        negation_markers = ["not", "no", "never", "false", "incorrect", "wrong"]
        for marker in negation_markers:
            if marker in other_words and shared:
                contradictions.append({
                    "with_agent": other_agent,
                    "marker": marker,
                    "shared_context": list(shared)[:5],
                })
                break

    return contradictions