import json
import uuid
import difflib
from api.llm import get_client, get_model
from db import get_pool

META_PROMPT = """You are a meta-agent responsible for improving the performance of a multi-agent pipeline.

You have just run an evaluation and these are the worst performing cases:

Failed cases:
{failed_cases}

Scores by dimension:
{scores_by_dimension}

The worst performing dimension is: {worst_dimension}
The worst performing agent prompt is: {worst_agent}

Current prompt for {worst_agent}:
{current_prompt}

Your job:
- Identify exactly why this prompt is causing failures in the {worst_dimension} dimension
- Propose a rewritten version of the prompt that fixes the identified issues
- Be specific, do not make vague improvements

Respond with a JSON object:
{{
    "diagnosis": "<exactly why this prompt is failing>",
    "proposed_prompt": "<full rewritten prompt>",
    "justification": "<what you changed and why it will improve {worst_dimension}>"
}}"""

# current prompts per agent — imported here so meta agent can read and propose rewrites
AGENT_PROMPTS = {
    "decomposition": "api/agents/decomposition.py",
    "rag": "api/agents/rag.py",
    "critique": "api/agents/critique.py",
    "synthesis": "api/agents/synthesis.py",
}

DIMENSION_TO_AGENT = {
    "correctness": "synthesis",
    "citation_accuracy": "rag",
    "contradiction_resolution": "synthesis",
    "tool_efficiency": "orchestrator",
    "budget_compliance": "decomposition",
    "critique_agreement": "critique",
}


def _read_prompt(agent_id: str) -> str:
    prompt_var = {
        "decomposition": "DECOMPOSITION_PROMPT",
        "rag": "RETRIEVAL_PROMPT",
        "critique": "CRITIQUE_PROMPT",
        "synthesis": "SYNTHESIS_PROMPT",
    }
    filepath = AGENT_PROMPTS.get(agent_id)
    if not filepath:
        return ""
    try:
        with open(filepath) as f:
            content = f.read()
        var_name = prompt_var.get(agent_id, "")
        start = content.find(f'{var_name} = """')
        if start == -1:
            return ""
        start += len(f'{var_name} = """')
        end = content.find('"""', start)
        return content[start:end].strip()
    except FileNotFoundError:
        return ""


def _make_diff(original: str, proposed: str) -> str:
    diff = difflib.unified_diff(
        original.splitlines(),
        proposed.splitlines(),
        lineterm="",
        fromfile="original",
        tofile="proposed",
    )
    return "\n".join(diff)


def _find_worst_dimension(scores: dict) -> tuple[str, str]:
    """Returns (worst_dimension, worst_agent)"""
    overall = scores.get("overall", {})
    if not overall:
        return "correctness", "synthesis"

    worst_dim = min(overall.keys(), key=lambda d: overall[d].get("mean", 1.0))
    worst_agent = DIMENSION_TO_AGENT.get(worst_dim, "synthesis")
    return worst_dim, worst_agent


async def propose_rewrite(eval_run_id: str) -> str | None:
    pool = await get_pool()
    client = get_client()
    model = get_model()

    async with pool.acquire() as conn:
        run = await conn.fetchrow(
            "SELECT * FROM eval_runs WHERE id = $1", eval_run_id
        )
        if not run:
            return None

        failed_cases = await conn.fetch("""
            SELECT case_id, category, query, actual_answer,
                   score_correctness, score_citation,
                   score_contradiction, score_tool_efficiency,
                   score_budget_compliance, score_critique_agreement,
                   justifications
            FROM eval_cases
            WHERE eval_run_id = $1 AND passed = false
        """, eval_run_id)

    if not failed_cases:
        return None

    scores = json.loads(run["scores"])
    worst_dim, worst_agent = _find_worst_dimension(scores)
    current_prompt = _read_prompt(worst_agent)

    failed_summary = json.dumps([
        {
            "case_id": r["case_id"],
            "category": r["category"],
            "query": r["query"],
            "actual_answer": r["actual_answer"],
            "justifications": json.loads(r["justifications"]) if r["justifications"] else {},
        }
        for r in failed_cases
    ], indent=2)

    scores_summary = json.dumps(scores.get("overall", {}), indent=2)

    prompt = META_PROMPT.format(
        failed_cases=failed_summary,
        scores_by_dimension=scores_summary,
        worst_dimension=worst_dim,
        worst_agent=worst_agent,
        current_prompt=current_prompt,
    )

    response = await client.chat.completions.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
        proposed_prompt = data.get("proposed_prompt", "")
        justification = data.get("justification", "")
        diagnosis = data.get("diagnosis", "")
    except (json.JSONDecodeError, KeyError):
        return None

    diff = _make_diff(current_prompt, proposed_prompt)
    rewrite_id = str(uuid.uuid4())

    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO prompt_rewrites (
                id, eval_run_id, agent_id, dimension,
                original_prompt, proposed_prompt, diff, justification, status
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'pending')
        """,
            rewrite_id,
            eval_run_id,
            worst_agent,
            worst_dim,
            current_prompt,
            proposed_prompt,
            diff,
            f"{diagnosis}\n\n{justification}",
        )

    return rewrite_id