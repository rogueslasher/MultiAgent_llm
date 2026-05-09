from fastapi import FastAPI
from contextlib import asynccontextmanager
from db import get_pool, close_pool
from api.routes.jobs import router as jobs_router
from api.routes.eval import router as eval_router
from api.routes.rewrites import router as rewrites_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = await get_pool()
    migration = open("db/migrations/001_initial.sql").read()
    async with pool.acquire() as conn:
        await conn.execute(migration)
    yield
    await close_pool()


app = FastAPI(title="mega-ai", lifespan=lifespan)

app.include_router(jobs_router)
app.include_router(eval_router)
app.include_router(rewrites_router)


@app.get("/health")
async def health():
    return {"status": "ok"}