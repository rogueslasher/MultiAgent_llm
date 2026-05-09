import time
import json
import asyncio
from api.llm import get_client, get_model
from db import get_pool
from api.tools.base import ToolResult, FAILURE_TIMEOUT, FAILURE_EMPTY, FAILURE_MALFORMED

TIMEOUT_SECONDS = 10

NL_TO_SQL_PROMPT = """Convert this natural language query to a SQL SELECT statement.

Query: {query}

Available tables and columns:
- jobs(id, query, status, created_at, updated_at)
- eval_runs(id, triggered_by, total_cases, passed, failed, scores, created_at)
- eval_cases(id, eval_run_id, case_id, category, query, expected_answer, actual_answer, score_correctness, score_citation, score_contradiction, score_tool_efficiency, score_budget_compliance, score_critique_agreement, passed, created_at)
- prompt_rewrites(id, eval_run_id, agent_id, dimension, status, created_at)

Rules:
- Only generate SELECT statements
- No INSERT, UPDATE, DELETE, DROP
- Return only the SQL, no explanation

SQL:"""


async def run(natural_language_query: str) -> ToolResult:
    if not natural_language_query or not isinstance(natural_language_query, str):
        return ToolResult(
            success=False,
            failure_mode=FAILURE_MALFORMED,
            error_message="query must be a non-empty string",
        )

    try:
        result = await asyncio.wait_for(
            _execute(natural_language_query),
            timeout=TIMEOUT_SECONDS,
        )
        return result
    except asyncio.TimeoutError:
        return ToolResult(
            success=False,
            failure_mode=FAILURE_TIMEOUT,
            error_message=f"data lookup timed out after {TIMEOUT_SECONDS}s",
        )


async def _execute(natural_language_query: str) -> ToolResult:
    client = get_client()
    model = get_model()

    prompt = NL_TO_SQL_PROMPT.format(query=natural_language_query)
    response = await client.chat.completions.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    sql = response.choices[0].message.content.strip()

    # safety check — only allow SELECT
    if not sql.lower().startswith("select"):
        return ToolResult(
            success=False,
            failure_mode=FAILURE_MALFORMED,
            error_message="generated SQL is not a SELECT statement",
        )

    pool = await get_pool()
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql)
    except Exception as e:
        return ToolResult(
            success=False,
            failure_mode="query_error",
            error_message=str(e),
        )

    if not rows:
        return ToolResult(
            success=False,
            failure_mode=FAILURE_EMPTY,
            error_message="query returned no results",
        )

    return ToolResult(
        success=True,
        data={
            "sql": sql,
            "rows": [dict(r) for r in rows],
            "row_count": len(rows),
        },
    )