import hashlib
import time
import json
from api.llm import get_client, get_model
from db import get_pool
from api.context import (
    SharedContext,
    AgentOutput,
    AgentID,
    SubTask,
    TaskStatus,
)


DECOMPOSITION_PROMPT = """You are a decomposition agent. Your job is to break a query into typed subtasks with explicit dependencies.

Query: {query}

Rules:
- Each subtask must have a type: one of [factual, analytical, retrieval, computational]
- Dependencies must reference subtask ids that exist in the same list
- A subtask cannot execute until all its dependencies are resolved
- Be precise, do not create unnecessary subtasks

Respond with a JSON object:
{{
    "subtasks": [
        {{
            "id": "<short unique id>",
            "type": "<factual|analytical|retrieval|computational>",
            "query": "<specific question this subtask answers>",
            "dependencies": ["<id>", ...]
        }}
    ]
}}"""


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


async def _log(pool, job_id: str, event_type: str, payload: dict,
               input_text: str = "", output_text: str = "",
               latency_ms: int = 0, token_count: int = 0,
               policy_violation: bool = False):
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO agent_logs
                (job_id, agent_id, event_type, input_hash, output_hash,
                 latency_ms, token_count, policy_violation, payload)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
            job_id,
            AgentID.DECOMPOSITION.value,
            event_type,
            _hash(input_text) if input_text else None,
            _hash(output_text) if output_text else None,
            latency_ms,
            token_count,
            policy_violation,
            json.dumps(payload),
        )


def _resolve_execution_order(subtasks: list[SubTask]) -> list[list[SubTask]]:
    id_map = {st.id: st for st in subtasks}
    waves = []
    resolved = set()
    remaining = list(subtasks)

    while remaining:
        wave = [
            st for st in remaining
            if all(dep in resolved for dep in st.dependencies)
        ]
        if not wave:
            for st in remaining:
                st.status = TaskStatus.FAILED
            break
        for st in wave:
            resolved.add(st.id)
        waves.append(wave)
        remaining = [st for st in remaining if st.id not in resolved]

    return waves


async def run(ctx: SharedContext) -> AgentOutput:
    pool = await get_pool()
    client = get_client()
    model = get_model()
    budget = ctx.token_budgets.get(AgentID.DECOMPOSITION.value, 2048)
    prompt = DECOMPOSITION_PROMPT.format(query=ctx.query)

    estimated_input_tokens = len(prompt.split()) + 50
    used = ctx.token_usage.get(AgentID.DECOMPOSITION.value, 0)
    if used + estimated_input_tokens > budget:
        violation_msg = f"decomposition exceeded budget: {used + estimated_input_tokens} > {budget}"
        ctx.policy_violations.append(violation_msg)
        await _log(pool, ctx.job_id, "budget_violation",
                   {"message": violation_msg}, policy_violation=True)

    t0 = time.time()
    response = await client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    latency_ms = int((time.time() - t0) * 1000)
    token_count = response.usage.prompt_tokens + response.usage.completion_tokens
    raw = response.choices[0].message.content.strip()

    ctx.token_usage[AgentID.DECOMPOSITION.value] = (
        ctx.token_usage.get(AgentID.DECOMPOSITION.value, 0) + token_count
    )

    try:
        data = json.loads(raw)
        subtasks = [SubTask(**st) for st in data["subtasks"]]
    except (json.JSONDecodeError, KeyError, TypeError):
        await _log(pool, ctx.job_id, "parse_error",
                   {"raw": raw}, prompt, raw, latency_ms, token_count)
        subtasks = []

    waves = _resolve_execution_order(subtasks)
    for i, wave in enumerate(waves):
        for st in wave:
            st.status = TaskStatus.DONE if i == 0 else TaskStatus.PENDING

    output = AgentOutput(
        agent_id=AgentID.DECOMPOSITION,
        output=json.dumps([st.model_dump() for st in subtasks]),
        subtasks=subtasks,
        token_count=token_count,
    )

    ctx.agent_outputs[AgentID.DECOMPOSITION.value] = output

    await _log(
        pool, ctx.job_id, "agent_complete",
        {"subtask_count": len(subtasks), "waves": len(waves)},
        prompt, raw, latency_ms, token_count,
    )

    return output