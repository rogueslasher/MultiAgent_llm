import json
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from db import get_pool
from api.eval.runner import run_eval, compute_delta

router = APIRouter()


class ReviewRequest(BaseModel):
    rewrite_id: str
    decision: str  # approve | reject


@router.post("/rewrites/review")
async def review_rewrite(request: ReviewRequest, background_tasks: BackgroundTasks):
    if request.decision not in ("approve", "reject"):
        return {
            "error_code": "INVALID_DECISION",
            "message": "decision must be 'approve' or 'reject'",
            "job_id": None,
        }

    pool = await get_pool()

    async with pool.acquire() as conn:
        rewrite = await conn.fetchrow("""
            SELECT * FROM prompt_rewrites WHERE id = $1
        """, request.rewrite_id)

        if not rewrite:
            return {
                "error_code": "REWRITE_NOT_FOUND",
                "message": f"no rewrite found with id {request.rewrite_id}",
                "job_id": None,
            }

        if rewrite["status"] != "pending":
            return {
                "error_code": "ALREADY_REVIEWED",
                "message": f"rewrite already has status: {rewrite['status']}",
                "job_id": None,
            }

        await conn.execute("""
            UPDATE prompt_rewrites
            SET status = $1, reviewed_at = NOW()
            WHERE id = $2
        """, request.decision + "d", request.rewrite_id)

    if request.decision == "approve":
        async with pool.acquire() as conn:
            failed_cases = await conn.fetch("""
                SELECT case_id FROM eval_cases
                WHERE eval_run_id = $1 AND passed = false
            """, rewrite["eval_run_id"])

        case_ids = [r["case_id"] for r in failed_cases]
        original_eval_run_id = str(rewrite["eval_run_id"])
        rewrite_id = request.rewrite_id

        async def run_and_delta():
            new_eval_run_id = await run_eval(
                triggered_by="prompt_rewrite",
                prompt_rewrite_id=rewrite_id,
                case_ids=case_ids,
            )
            delta = await compute_delta(
                original_eval_run_id,
                new_eval_run_id,
            )
            async with pool.acquire() as conn:
                await conn.execute("""
                    UPDATE prompt_rewrites
                    SET justification = justification || $1
                    WHERE id = $2
                """,
                    f"\n\nPERFORMANCE DELTA:\n{json.dumps(delta, indent=2)}",
                    rewrite_id,
                )

        background_tasks.add_task(run_and_delta)

        return {
            "message": "rewrite approved, re-eval triggered on failed cases",
            "rewrite_id": request.rewrite_id,
            "case_ids": case_ids,
        }

    return {
        "message": "rewrite rejected",
        "rewrite_id": request.rewrite_id,
    }


@router.get("/rewrites/pending")
async def get_pending_rewrites():
    pool = await get_pool()
    async with pool.acquire() as conn:
        rewrites = await conn.fetch("""
            SELECT id, eval_run_id, agent_id, dimension,
                   diff, justification, created_at
            FROM prompt_rewrites
            WHERE status = 'pending'
            ORDER BY created_at DESC
        """)

    return {
        "pending": [
            {
                "rewrite_id": str(r["id"]),
                "eval_run_id": str(r["eval_run_id"]),
                "agent_id": r["agent_id"],
                "dimension": r["dimension"],
                "diff": r["diff"],
                "justification": r["justification"],
                "created_at": r["created_at"].isoformat(),
            }
            for r in rewrites
        ]
    }