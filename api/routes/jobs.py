import json
import uuid
import redis.asyncio as aioredis
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from db import get_pool
from api.config import config
from api.context.schema import SharedContext
from api.pipeline import run_pipeline

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


async def _get_redis():
    return aioredis.from_url(config.redis_url)


@router.post("/query")
async def submit_query(request: QueryRequest):
    job_id = str(uuid.uuid4())
    pool = await get_pool()

    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO jobs (id, query, status)
            VALUES ($1, $2, 'pending')
        """, job_id, request.query)

    # push to worker queue for async persistence
    redis = await _get_redis()
    await redis.rpush(
        "mega:jobs",
        json.dumps({"job_id": job_id, "query": request.query}),
    )
    await redis.aclose()

    async def event_stream():
        ctx = SharedContext(job_id=job_id, query=request.query)
        try:
            async for event in run_pipeline(ctx):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'job_id': job_id})}\n\n"
        finally:
            async with pool.acquire() as conn:
                await conn.execute("""
                    UPDATE jobs SET status = 'done', updated_at = NOW()
                    WHERE id = $1
                """, job_id)
            yield f"data: {json.dumps({'type': 'done', 'job_id': job_id})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/trace/{job_id}")
async def get_trace(job_id: str):
    pool = await get_pool()

    async with pool.acquire() as conn:
        job = await conn.fetchrow(
            "SELECT * FROM jobs WHERE id = $1", job_id
        )
        if not job:
            return {
                "error_code": "JOB_NOT_FOUND",
                "message": f"no job found with id {job_id}",
                "job_id": job_id,
            }

        logs = await conn.fetch("""
            SELECT * FROM agent_logs
            WHERE job_id = $1
            ORDER BY created_at ASC
        """, job_id)

        tool_calls = await conn.fetch("""
            SELECT * FROM tool_calls
            WHERE job_id = $1
            ORDER BY created_at ASC
        """, job_id)

    events = []

    for log in logs:
        events.append({
            "type": "agent_event",
            "timestamp": log["created_at"].isoformat(),
            "agent_id": log["agent_id"],
            "event_type": log["event_type"],
            "latency_ms": log["latency_ms"],
            "token_count": log["token_count"],
            "policy_violation": log["policy_violation"],
            "payload": json.loads(log["payload"]) if log["payload"] else None,
        })

    for call in tool_calls:
        events.append({
            "type": "tool_call",
            "timestamp": call["created_at"].isoformat(),
            "agent_id": call["agent_id"],
            "tool_name": call["tool_name"],
            "input": json.loads(call["input"]) if call["input"] else None,
            "output": json.loads(call["output"]) if call["output"] else None,
            "latency_ms": call["latency_ms"],
            "accepted": call["accepted"],
            "retry_number": call["retry_number"],
            "failure_mode": call["failure_mode"],
        })

    events.sort(key=lambda e: e["timestamp"])

    return {
        "job_id": job_id,
        "query": job["query"],
        "status": job["status"],
        "created_at": job["created_at"].isoformat(),
        "trace": events,
    }