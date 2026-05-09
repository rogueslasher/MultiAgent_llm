import hashlib
import time
from api.llm import get_client, get_model
from db import get_pool
from api.context import (
    SharedContext,
    AgentID,
    RoutingDecision,
)
import os
import json


ROUTING_PROMPT = """You are the orchestrator of a multi-agent pipeline.

Given the current state of the job, decide which agent to invoke next.

Agents available:
- decomposition: breaks the query into typed subtasks with dependencies
- rag: retrieves relevant chunks and performs multi-hop reasoning
- critique: reviews all agent outputs, scores claims, flags specific spans
- synthesis: merges all outputs, resolves contradictions, produces final answer

Rules:
- decomposition must run before rag
- critique must run after at least one of decomposition or rag
- synthesis must run last after critique
- do not repeat an agent unless there is a specific reason
- if all agents have run and synthesis is done, return "done"

Current state:
{state}

Respond with a JSON object:
{{
    "next_agent": "<agent_name or done>",
    "justification": "<why this agent next>",
    "context_budget": <max tokens to give this agent as integer>
}}"""


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


async def _log(pool, job_id: str, event_type: str, payload: dict,
               input_text: str = "", output_text: str = "",
               latency_ms: int = 0, token_count: int = 0):
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO agent_logs
                (job_id, agent_id, event_type, input_hash, output_hash,
                 latency_ms, token_count, payload)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
            job_id,
            AgentID.ORCHESTRATOR.value,
            event_type,
            _hash(input_text) if input_text else None,
            _hash(output_text) if output_text else None,
            latency_ms,
            token_count,
            json.dumps(payload),
        )


def _build_state_summary(ctx: SharedContext) -> str:
    completed = list(ctx.agent_outputs.keys())
    routing = [
        {"agent": r.next_agent, "justification": r.justification}
        for r in ctx.routing_history
    ]
    return json.dumps({
        "query": ctx.query,
        "completed_agents": completed,
        "routing_history": routing,
        "policy_violations": ctx.policy_violations,
    })


async def route(ctx: SharedContext) -> RoutingDecision | None:
    pool = await get_pool()
    client = get_client()
    model = get_model()
    state_summary = _build_state_summary(ctx)
    prompt = ROUTING_PROMPT.format(state=state_summary)

    t0 = time.time()
    response = await client.chat.completions.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    latency_ms = int((time.time() - t0) * 1000)
    token_count = response.usage.prompt_tokens + response.usage.completion_tokens
    raw = response.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        await _log(pool, ctx.job_id, "routing_parse_error",
                   {"raw": raw}, prompt, raw, latency_ms, token_count)
        return None

    if data.get("next_agent") == "done":
        await _log(pool, ctx.job_id, "routing_done",
                   data, prompt, raw, latency_ms, token_count)
        return None

    decision = RoutingDecision(
        next_agent=AgentID(data["next_agent"]),
        justification=data["justification"],
        context_budget=data["context_budget"],
    )
    ctx.routing_history.append(decision)
    ctx.token_budgets[decision.next_agent.value] = decision.context_budget

    await _log(pool, ctx.job_id, "routing_decision",
               data, prompt, raw, latency_ms, token_count)

    return decision