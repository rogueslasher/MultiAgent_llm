import json
import uuid
from datetime import datetime
from db import get_pool
from api.context.schema import SharedContext
from api.pipeline import run_pipeline
from api.eval.cases import EVAL_CASES, EvalCase
from api.eval.runner import run_eval, compute_delta
from api.eval.scorer import score_case, CaseScore


async def run_eval(
    triggered_by: str = "manual",
    prompt_rewrite_id: str = None,
    case_ids: list[str] = None,
) -> str:
    pool = await get_pool()
    eval_run_id = str(uuid.uuid4())

    cases = EVAL_CASES
    if case_ids:
        cases = [c for c in EVAL_CASES if c.case_id in case_ids]

    results: list[CaseScore] = []

    for case in cases:
        ctx = SharedContext(
            job_id=str(uuid.uuid4()),
            query=case.query,
        )

        # run pipeline, consume all events
        async for _ in run_pipeline(ctx):
            pass

        case_score = score_case(case, ctx)
        results.append(case_score)

        # store job
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO jobs (id, query, status)
                VALUES ($1, $2, 'done')
                ON CONFLICT (id) DO NOTHING
            """, ctx.job_id, case.query)

        # store eval case result
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO eval_cases (
                    id, eval_run_id, case_id, category, query,
                    expected_answer, actual_answer,
                    score_correctness, score_citation,
                    score_contradiction, score_tool_efficiency,
                    score_budget_compliance, score_critique_agreement,
                    justifications, passed
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7,
                    $8, $9, $10, $11, $12, $13, $14, $15
                )
            """,
                str(uuid.uuid4()),
                eval_run_id,
                case.case_id,
                case.category,
                case.query,
                case.expected_answer,
                ctx.final_answer,
                case_score.correctness.score,
                case_score.citation_accuracy.score,
                case_score.contradiction_resolution.score,
                case_score.tool_efficiency.score,
                case_score.budget_compliance.score,
                case_score.critique_agreement.score,
                json.dumps({
                    "correctness": case_score.correctness.justification,
                    "citation_accuracy": case_score.citation_accuracy.justification,
                    "contradiction_resolution": case_score.contradiction_resolution.justification,
                    "tool_efficiency": case_score.tool_efficiency.justification,
                    "budget_compliance": case_score.budget_compliance.justification,
                    "critique_agreement": case_score.critique_agreement.justification,
                }),
                case_score.passed,
            )

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    # aggregate scores by category and dimension
    summary = _build_summary(results)

    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO eval_runs (
                id, triggered_by, prompt_rewrite_id,
                total_cases, passed, failed, scores
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
            eval_run_id,
            triggered_by,
            prompt_rewrite_id,
            len(results),
            passed,
            failed,
            json.dumps(summary),
        )

    return eval_run_id


def _build_summary(results: list[CaseScore]) -> dict:
    categories = ["baseline", "ambiguous", "adversarial"]
    dimensions = [
        "correctness", "citation_accuracy", "contradiction_resolution",
        "tool_efficiency", "budget_compliance", "critique_agreement",
    ]

    summary = {}
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        if not cat_results:
            continue
        summary[cat] = {}
        for dim in dimensions:
            scores = [getattr(r, dim).score for r in cat_results]
            summary[cat][dim] = {
                "mean": round(sum(scores) / len(scores), 3),
                "min": round(min(scores), 3),
                "max": round(max(scores), 3),
            }

    summary["overall"] = {
        dim: {
            "mean": round(
                sum(getattr(r, dim).score for r in results) / len(results), 3
            )
        }
        for dim in dimensions
    }

    return summary

async def compute_delta(
    original_eval_run_id: str,
    new_eval_run_id: str,
) -> dict:
    pool = await get_pool()

    async with pool.acquire() as conn:
        original_cases = await conn.fetch("""
            SELECT case_id, score_correctness, score_citation,
                   score_contradiction, score_tool_efficiency,
                   score_budget_compliance, score_critique_agreement
            FROM eval_cases
            WHERE eval_run_id = $1
        """, original_eval_run_id)

        new_cases = await conn.fetch("""
            SELECT case_id, score_correctness, score_citation,
                   score_contradiction, score_tool_efficiency,
                   score_budget_compliance, score_critique_agreement
            FROM eval_cases
            WHERE eval_run_id = $1
        """, new_eval_run_id)

    original_map = {r["case_id"]: dict(r) for r in original_cases}
    new_map = {r["case_id"]: dict(r) for r in new_cases}

    dimensions = [
        "score_correctness", "score_citation", "score_contradiction",
        "score_tool_efficiency", "score_budget_compliance",
        "score_critique_agreement",
    ]

    delta = {}
    for case_id, new_scores in new_map.items():
        if case_id not in original_map:
            continue
        original_scores = original_map[case_id]
        delta[case_id] = {
            dim: round(
                (new_scores[dim] or 0) - (original_scores[dim] or 0), 3
            )
            for dim in dimensions
        }

    overall_delta = {}
    for dim in dimensions:
        changes = [
            delta[case_id][dim]
            for case_id in delta
        ]
        overall_delta[dim] = round(
            sum(changes) / len(changes) if changes else 0, 3
        )

    return {
        "per_case": delta,
        "overall": overall_delta,
    }