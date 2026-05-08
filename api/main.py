from fastapi import FastAPI
from contextlib import asynccontextmanager
from db import get_pool, close_pool
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = await get_pool()
    migration = open("db/migrations/001_initial.sql").read()
    async with pool.acquire() as conn:
        await conn.execute(migration)
    yield
    await close_pool()

app = FastAPI(title="mega-ai", lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok"}