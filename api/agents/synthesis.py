import hashlib
import time
import json
from api.context import BudgetManager, compress, count_tokens
from api.llm import get_client, get_model
from db import get_pool
from api.context import (
    SharedContext,
    AgentOutput,
    AgentID,
    ProvenanceEntry,
)

SYNTHESIS_PROMPT = """You are a synthesis agent. Your job is to merge outputs from all agents, resolve any contradictions, and produce a final answer.

Original query: {query}

Agent outputs:
{agent_outputs}

Flagged contradictions from critique agent:
{flagged_claims}

Rules:
- Merge all relevant information into a single coherent answer
- For every flagged contradiction you must explicitly resolve it, do not leave contradictions in the final answer
- For every sentence in your answer record which agent and which chunk it came from
- If a sentence comes from multiple agents pick the most confident source
- Do not introduce new information not present in the agent outputs

Respond with a JSON object:
{{
    "answer": "<full final answer>",
    "contradiction_resolutions": [
        {{
            "contradiction": "<the flagged claim>",
            "resolution": "<how you resolved it and why>"
        }}
    ],
    "provenance": [
        {{
            "sentence": "<exact sentence from your answer>",
            "source_agent": "<agent_id>",
            "source_chunk_id": "<chunk_id or null>"
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
            AgentID.SYNTHESIS.value,
            event_type,
            _hash(input_text) if input_text else None,
            _hash(output_text) if output_text else None,
            latency_ms,
            token_count,
            policy_violation,
            json.dumps(payload),
        )


def _format_agent_outputs(ctx: SharedContext) -> str:
    parts = []
    for agent_id, output in ctx.agent_outputs.items():
        if agent_id == AgentID.CRITIQUE.value:
            continue
        parts.append(f"[{agent_id}]\n{output.output}")
    return "\n\n".join(parts)


def _format_flagged_claims(ctx: SharedContext) -> str:
    critique = ctx.agent_outputs.get(AgentID.CRITIQUE.value)
    if not critique:
        return "None"
    flagged = [c for c in critique.claims if c.flagged]
    if not flagged:
        return "None"
    return "\n".join([
        f"- \"{c.text}\" — {c.flag_reason}"
        for c in flagged
    ])

async def run(ctx: SharedContext) -> AgentOutput:
    pool = await get_pool()
    client = get_client()
    model = get_model()

    budget_manager = BudgetManager(ctx)
    budget_manager.declare(AgentID.SYNTHESIS.value, ctx.token_budgets.get(AgentID.SYNTHESIS.value, 4096))

    agent_outputs_text = _format_agent_outputs(ctx)
    flagged_claims_text = _format_flagged_claims(ctx)

    prompt = SYNTHESIS_PROMPT.format(
        query=ctx.query,
        agent_outputs=agent_outputs_text,
        flagged_claims=flagged_claims_text,
    )

    if budget_manager.needs_compression(AgentID.SYNTHESIS.value, prompt):
        prompt = await compress(prompt, count_tokens(prompt), ctx.job_id)

    budget_manager.consume(AgentID.SYNTHESIS.value, prompt)

    t0 = time.time()
    response = await client.chat.completions.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    latency_ms = int((time.time() - t0) * 1000)
    token_count = response.usage.prompt_tokens + response.usage.completion_tokens
    raw = response.choices[0].message.content.strip()

    budget_manager.consume(AgentID.SYNTHESIS.value, raw)

    try:
        data = json.loads(raw)
        answer = data.get("answer", "")
        resolutions = data.get("contradiction_resolutions", [])
        provenance = [
            ProvenanceEntry(
                sentence=p["sentence"],
                source_agent=AgentID(p["source_agent"]),
                source_chunk_id=p.get("source_chunk_id"),
            )
            for p in data.get("provenance", [])
        ]
    except (json.JSONDecodeError, KeyError, TypeError):
        await _log(pool, ctx.job_id, "parse_error",
                   {"raw": raw}, prompt, raw, latency_ms, token_count)
        answer = ""
        resolutions = []
        provenance = []

    ctx.final_answer = answer

    output = AgentOutput(
        agent_id=AgentID.SYNTHESIS,
        output=answer,
        provenance=provenance,
        token_count=token_count,
    )

    ctx.agent_outputs[AgentID.SYNTHESIS.value] = output

    await _log(
        pool, ctx.job_id, "agent_complete",
        {
            "answer_length": len(answer),
            "contradictions_resolved": len(resolutions),
            "provenance_entries": len(provenance),
        },
        prompt, raw, latency_ms, token_count,
    )

    return output