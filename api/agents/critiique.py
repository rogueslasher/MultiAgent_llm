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
    Claim,
)

CRITIQUE_PROMPT = """You are a critique agent. Your job is to review the output of other agents and score each claim individually.

Original query: {query}

Agent output to review:
Agent: {agent_id}
Output: {output}

Rules:
- Break the output into individual claims
- For each claim assign a confidence score between 0 and 1
- If you disagree with a claim flag it with the exact span of text you disagree with and why
- Do not flag the output as a whole, only specific claims
- Be precise and objective

Respond with a JSON object:
{{
    "claims": [
        {{
            "text": "<exact claim text>",
            "confidence": <float between 0 and 1>,
            "flagged": <true|false>,
            "flag_reason": "<why you disagree, or null if not flagged>"
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
            AgentID.CRITIQUE.value,
            event_type,
            _hash(input_text) if input_text else None,
            _hash(output_text) if output_text else None,
            latency_ms,
            token_count,
            policy_violation,
            json.dumps(payload),
        )


async def run(ctx: SharedContext) -> AgentOutput:
    pool = await get_pool()
    client = get_client()
    model = get_model()
    budget = ctx.token_budgets.get(AgentID.CRITIQUE.value, 4096)

    # collect all agent outputs to review
    agents_to_review = [
        AgentID.DECOMPOSITION.value,
        AgentID.RAG.value,
    ]

    all_claims = []
    total_tokens = 0
    critique_summary = {}

    for agent_id in agents_to_review:
        agent_output = ctx.agent_outputs.get(agent_id)
        if not agent_output:
            continue

        prompt = CRITIQUE_PROMPT.format(
            query=ctx.query,
            agent_id=agent_id,
            output=agent_output.output,
        )

        # budget check
        estimated = len(prompt.split()) + 50
        used = ctx.token_usage.get(AgentID.CRITIQUE.value, 0)
        if used + estimated > budget:
            violation_msg = f"critique exceeded budget reviewing {agent_id}: {used + estimated} > {budget}"
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
        total_tokens += token_count
        raw = response.choices[0].message.content.strip()

        ctx.token_usage[AgentID.CRITIQUE.value] = (
            ctx.token_usage.get(AgentID.CRITIQUE.value, 0) + token_count
        )

        try:
            data = json.loads(raw)
            claims = [Claim(**c) for c in data.get("claims", [])]
        except (json.JSONDecodeError, KeyError, TypeError):
            await _log(pool, ctx.job_id, "parse_error",
                       {"raw": raw, "agent_reviewed": agent_id},
                       prompt, raw, latency_ms, token_count)
            claims = []

        all_claims.extend(claims)
        flagged = [c for c in claims if c.flagged]
        critique_summary[agent_id] = {
            "total_claims": len(claims),
            "flagged": len(flagged),
            "avg_confidence": (
                sum(c.confidence for c in claims) / len(claims)
                if claims else 0
            ),
        }

        await _log(
            pool, ctx.job_id, "critique_complete",
            {
                "agent_reviewed": agent_id,
                "claims": len(claims),
                "flagged": len(flagged),
            },
            prompt, raw, latency_ms, token_count,
        )

    output = AgentOutput(
        agent_id=AgentID.CRITIQUE,
        output=json.dumps(critique_summary),
        claims=all_claims,
        token_count=total_tokens,
    )

    ctx.agent_outputs[AgentID.CRITIQUE.value] = output

    return output