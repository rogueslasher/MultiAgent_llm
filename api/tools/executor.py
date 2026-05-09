import time
import json
import hashlib
from db import get_pool
from api.tools.base import ToolResult
from api.tools import web_search, code_sandbox, data_lookup, self_reflection
from api.context import SharedContext

MAX_RETRIES = 2

TOOL_MAP = {
    "web_search": web_search.run,
    "code_sandbox": code_sandbox.run,
    "data_lookup": data_lookup.run,
    "self_reflection": self_reflection.run,
}

# how orchestrator handles each failure mode
FAILURE_HANDLERS = {
    "timeout": "retry",
    "empty_results": "retry_with_modified_input",
    "malformed_input": "abort",
    "query_error": "abort",
    "execution_error": "log_and_continue",
}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


async def _log_tool_call(
    pool, job_id: str, agent_id: str, tool_name: str,
    input_data: dict, result: ToolResult,
    latency_ms: int, retry_number: int,
):
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO tool_calls
                (job_id, agent_id, tool_name, input, output,
                 latency_ms, accepted, retry_number, failure_mode)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
            job_id,
            agent_id,
            tool_name,
            json.dumps(input_data),
            json.dumps(result.data) if result.data else None,
            latency_ms,
            result.success,
            retry_number,
            result.failure_mode,
        )


async def call(
    tool_name: str,
    input_data: dict,
    job_id: str,
    agent_id: str,
    ctx: SharedContext = None,
) -> ToolResult:
    if tool_name not in TOOL_MAP:
        return ToolResult(
            success=False,
            failure_mode="malformed_input",
            error_message=f"unknown tool: {tool_name}",
        )

    pool = await get_pool()
    tool_fn = TOOL_MAP[tool_name]
    result = None

    for attempt in range(MAX_RETRIES + 1):
        t0 = time.time()

        if tool_name == "self_reflection":
            result = await tool_fn(ctx=ctx, **input_data)
        else:
            result = await tool_fn(**input_data)

        latency_ms = int((time.time() - t0) * 1000)
        await _log_tool_call(
            pool, job_id, agent_id, tool_name,
            input_data, result, latency_ms, attempt,
        )

        if result.success:
            break

        handler = FAILURE_HANDLERS.get(result.failure_mode, "abort")

        if handler == "abort":
            break
        elif handler == "retry" and attempt < MAX_RETRIES:
            continue
        elif handler == "retry_with_modified_input" and attempt < MAX_RETRIES:
            # widen the query on empty results
            if "query" in input_data:
                input_data["query"] = input_data["query"] + " related information"
            continue
        else:
            break

    return result