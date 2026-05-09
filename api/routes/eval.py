import json
from fastapi import APIRouter, BackgroundTasks
from db import get_pool
from api.eval.runner import run_eval

router = APIRouter()


@router.get("/eval/latest")
async def get_latest_eval():
    pool = await get_pool()
    async with pool.acquire() as conn:
        run = await conn.fetchrow("""
            SELECT * FROM eval_runs
            ORDER BY created_at DESC
            LIMIT 1
        """)

    if not run:
        return {
            "error_code": "NO_EVAL_RUN",
            "message": "no eval runs found",
            "job_id": None,
        }

    return {
        "eval_run_id": str(run["id"]),
        "triggered_by": run["triggered_by"],
        "total_cases": run["total_cases"],
        "passed": run["passed"],
        "failed": run["failed"],
        "scores": json.loads(run["scores"]),
        "created_at": run["created_at"].isoformat(),
    }


@router.post("/eval/rerun")
async def rerun_failed(background_tasks: BackgroundTasks):
    pool = await get_pool()

    # get failed case ids from latest run
    async with pool.acquire() as conn:
        latest = await conn.fetchrow("""
            SELECT id FROM eval_runs
            ORDER BY created_at DESC
            LIMIT 1
        """)
        if not latest:
            return {
                "error_code": "NO_EVAL_RUN",
                "message": "no eval runs to rerun",
                "job_id": None,
            }

        failed_cases = await conn.fetch("""
            SELECT case_id FROM eval_cases
            WHERE eval_run_id = $1 AND passed = false
        """, latest["id"])

    case_ids = [r["case_id"] for r in failed_cases]
    if not case_ids:
        return {"message": "no failed cases to rerun"}

    background_tasks.add_task(
        run_eval,
        triggered_by="rerun",
        case_ids=case_ids,
    )

    return {
        "message": f"rerun triggered for {len(case_ids)} failed cases",
        "case_ids": case_ids,
    }