import hashlib
import time
import json
from api.llm import get_client, get_model
from db import get_pool
from api.context import (
    SharedContext,
    AgentOutput,
    AgentID,
    Chunk,
    Citation,
)
from api.context import BudgetManager, compress, count_tokens
from api.tools.executor import call as tool_call

RETRIEVAL_PROMPT = """You are a retrieval agent. Given a query and a set of text chunks, identify the most relevant chunks for answering the query.

Query: {query}

Chunks:
{chunks}

Rules:
- Select at least 2 chunks that are relevant
- Assign a relevance score between 0 and 1 to each selected chunk
- You must use multi-hop reasoning: the answer requires combining information from multiple chunks

Respond with a JSON object:
{{
    "selected_chunks": [
        {{
            "chunk_id": "<id>",
            "relevance_score": <float>,
            "reason": "<why this chunk is relevant>"
        }}
    ]
}}"""

REASONING_PROMPT = """You are a retrieval-augmented reasoning agent. Answer the query using only the provided chunks.

Query: {query}

Chunks:
{chunks}

Rules:
- You must reason across at least 2 chunks before forming your answer
- For every claim in your answer cite which chunk it came from
- Do not use knowledge outside the provided chunks

Respond with a JSON object:
{{
    "answer": "<full answer>",
    "citations": [
        {{
            "claim": "<specific claim from your answer>",
            "chunk_id": "<id of chunk that supports this claim>",
            "source": "<source url or name>"
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
            AgentID.RAG.value,
            event_type,
            _hash(input_text) if input_text else None,
            _hash(output_text) if output_text else None,
            latency_ms,
            token_count,
            policy_violation,
            json.dumps(payload),
        )


def _format_chunks(chunks: list[Chunk]) -> str:
    return "\n\n".join([
        f"[{c.chunk_id}] (source: {c.source})\n{c.content}"
        for c in chunks
    ])


def _get_mock_chunks(query: str) -> list[Chunk]:
    return [
        Chunk(
            chunk_id="chunk_001",
            source="knowledge_base/doc_1.txt",
            content=f"Relevant background information about: {query[:60]}. "
                    "This chunk contains foundational context that partially addresses the query.",
            relevance_score=0.85,
        ),
        Chunk(
            chunk_id="chunk_002",
            source="knowledge_base/doc_2.txt",
            content=f"Additional details and evidence related to: {query[:60]}. "
                    "This chunk provides supporting facts that complement chunk_001.",
            relevance_score=0.78,
        ),
        Chunk(
            chunk_id="chunk_003",
            source="knowledge_base/doc_3.txt",
            content=f"Contrasting perspective or edge case regarding: {query[:60]}. "
                    "This chunk introduces nuance that requires multi-hop reasoning.",
            relevance_score=0.65,
        ),
    ]


async def run(ctx: SharedContext) -> AgentOutput:
    pool = await get_pool()
    client = get_client()
    model = get_model()

    budget_manager = BudgetManager(ctx)
    budget_manager.declare(AgentID.RAG.value, ctx.token_budgets.get(AgentID.RAG.value, 4096))

    # get subtasks from decomposition if available
    decomp_output = ctx.agent_outputs.get(AgentID.DECOMPOSITION.value)
    if decomp_output and decomp_output.subtasks:
        queries = [
            st.query for st in decomp_output.subtasks
            if st.type == "retrieval"
        ] or [ctx.query]
    else:
        queries = [ctx.query]

    all_chunks = []
    total_tokens = 0

    for query in queries:
        # web search tool call to augment retrieval
        search_result = await tool_call(
            tool_name="web_search",
            input_data={"query": query},
            job_id=ctx.job_id,
            agent_id=AgentID.RAG.value,
            ctx=ctx,
        )

        web_chunks = []
        if search_result.success:
            web_chunks = [
                Chunk(
                    chunk_id=f"web_{j}",
                    source=r["url"],
                    content=r["snippet"],
                    relevance_score=r["relevance_score"],
                )
                for j, r in enumerate(
                    search_result.data.get("results", [])[:2]
                )
            ]

        raw_chunks = _get_mock_chunks(query) + web_chunks
        chunks_text = _format_chunks(raw_chunks)
        prompt = RETRIEVAL_PROMPT.format(query=query, chunks=chunks_text)

        if budget_manager.needs_compression(AgentID.RAG.value, prompt):
            prompt = await compress(prompt, count_tokens(prompt), ctx.job_id)

        budget_manager.consume(AgentID.RAG.value, prompt)

        t0 = time.time()
        response = await client.chat.completions.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        latency_ms = int((time.time() - t0) * 1000)
        token_count = response.usage.prompt_tokens + response.usage.completion_tokens
        total_tokens += token_count
        raw = response.choices[0].message.content.strip()

        budget_manager.consume(AgentID.RAG.value, raw)

        try:
            data = json.loads(raw)
            selected_ids = {
                sc["chunk_id"] for sc in data.get("selected_chunks", [])
            }
            selected = [c for c in raw_chunks if c.chunk_id in selected_ids]
            if len(selected) < 2:
                selected = raw_chunks[:2]
            all_chunks.extend(selected)
        except (json.JSONDecodeError, KeyError):
            await _log(pool, ctx.job_id, "retrieval_parse_error",
                       {"raw": raw}, prompt, raw, latency_ms, token_count)
            all_chunks.extend(raw_chunks[:2])

        await _log(pool, ctx.job_id, "retrieval_complete",
                   {"query": query, "chunks_selected": len(selected)},
                   prompt, raw, latency_ms, token_count)

    # deduplicate chunks
    seen = set()
    unique_chunks = []
    for c in all_chunks:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            unique_chunks.append(c)

    # multi-hop reasoning over selected chunks
    chunks_text = _format_chunks(unique_chunks)
    reasoning_prompt = REASONING_PROMPT.format(
        query=ctx.query,
        chunks=chunks_text,
    )

    if budget_manager.needs_compression(AgentID.RAG.value, reasoning_prompt):
        reasoning_prompt = await compress(
            reasoning_prompt, count_tokens(reasoning_prompt), ctx.job_id
        )

    budget_manager.consume(AgentID.RAG.value, reasoning_prompt)

    t0 = time.time()
    response = await client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": reasoning_prompt}],
        response_format={"type": "json_object"},
    )
    latency_ms = int((time.time() - t0) * 1000)
    token_count = response.usage.prompt_tokens + response.usage.completion_tokens
    total_tokens += token_count
    raw = response.choices[0].message.content.strip()

    budget_manager.consume(AgentID.RAG.value, raw)

    try:
        data = json.loads(raw)
        answer = data.get("answer", "")
        citations = [
            Citation(
                claim=c["claim"],
                chunk_id=c["chunk_id"],
                source=c["source"],
            )
            for c in data.get("citations", [])
        ]
    except (json.JSONDecodeError, KeyError):
        await _log(pool, ctx.job_id, "reasoning_parse_error",
                   {"raw": raw}, reasoning_prompt, raw, latency_ms, token_count)
        answer = ""
        citations = []

    output = AgentOutput(
        agent_id=AgentID.RAG,
        output=answer,
        chunks=unique_chunks,
        citations=citations,
        token_count=total_tokens,
    )

    ctx.agent_outputs[AgentID.RAG.value] = output

    await _log(
        pool, ctx.job_id, "agent_complete",
        {
            "chunks_used": len(unique_chunks),
            "citations": len(citations),
            "hops": len(queries),
        },
        reasoning_prompt, raw, latency_ms, token_count,
    )

    return output