import asyncio
import json
import os
import signal
import redis.asyncio as aioredis
from api.config import config
from api.context.schema import SharedContext
from api.pipeline import run_pipeline
from db import get_pool, close_pool


QUEUE_KEY = "mega:jobs"


async def process_job(job_data: dict):
    pool = await get_pool()
    job_id = job_data["job_id"]
    query = job_data["query"]

    ctx = SharedContext(job_id=job_id, query=query)

    try:
        async with pool.acquire() as conn:
            await conn.execute("""
                UPDATE jobs SET status = 'running', updated_at = NOW()
                WHERE id = $1
            """, job_id)

        async for _ in run_pipeline(ctx):
            pass

        async with pool.acquire() as conn:
            await conn.execute("""
                UPDATE jobs SET status = 'done', updated_at = NOW()
                WHERE id = $1
            """, job_id)

    except Exception as e:
        async with pool.acquire() as conn:
            await conn.execute("""
                UPDATE jobs SET status = 'failed', updated_at = NOW()
                WHERE id = $1
            """, job_id)
        print(f"job {job_id} failed: {e}")


async def main():
    print("worker started, listening for jobs...")
    redis = aioredis.from_url(config.redis_url)

    loop = asyncio.get_event_loop()
    stop = asyncio.Event()

    def shutdown():
        print("worker shutting down...")
        stop.set()

    loop.add_signal_handler(signal.SIGTERM, shutdown)
    loop.add_signal_handler(signal.SIGINT, shutdown)

    while not stop.is_set():
        try:
            item = await redis.blpop(QUEUE_KEY, timeout=2)
            if item:
                _, raw = item
                job_data = json.loads(raw)
                await process_job(job_data)
        except Exception as e:
            print(f"worker error: {e}")
            await asyncio.sleep(1)

    await redis.aclose()
    await close_pool()
    print("worker stopped")


if __name__ == "__main__":
    asyncio.run(main())