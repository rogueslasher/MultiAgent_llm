import json
from typing import AsyncGenerator
from api.context.schema import SharedContext, AgentID
from api.context.budget import BudgetManager
from api.agents import route
from api.agents import decomposition, rag, critique, synthesis


AGENT_MAP = {
    AgentID.DECOMPOSITION: decomposition.run,
    AgentID.RAG: rag.run,
    AgentID.CRITIQUE: critique.run,
    AgentID.SYNTHESIS: synthesis.run,
}

MAX_TURNS = 10


async def run_pipeline(ctx: SharedContext) -> AsyncGenerator[dict, None]:
    budget_manager = BudgetManager(ctx)

    yield {
        "type": "pipeline_start",
        "job_id": ctx.job_id,
        "query": ctx.query,
    }

    for turn in range(MAX_TURNS):
        # orchestrator decides next agent
        yield {
            "type": "routing",
            "turn": turn,
            "completed_agents": list(ctx.agent_outputs.keys()),
        }

        decision = await route(ctx)

        if decision is None:
            break

        yield {
            "type": "agent_start",
            "agent_id": decision.next_agent.value,
            "justification": decision.justification,
            "context_budget": decision.context_budget,
            "budget_remaining": budget_manager.remaining(decision.next_agent.value),
        }

        agent_fn = AGENT_MAP.get(decision.next_agent)
        if not agent_fn:
            yield {
                "type": "error",
                "message": f"unknown agent: {decision.next_agent}",
            }
            break

        output = await agent_fn(ctx)

        yield {
            "type": "agent_complete",
            "agent_id": decision.next_agent.value,
            "token_count": output.token_count,
            "budget_remaining": budget_manager.remaining(decision.next_agent.value),
            "policy_violations": ctx.policy_violations,
            "output_preview": output.output[:200],
        }

        # stream output tokens
        words = output.output.split()
        chunk_size = 5
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            yield {
                "type": "token_stream",
                "agent_id": decision.next_agent.value,
                "content": chunk,
            }

        if decision.next_agent == AgentID.SYNTHESIS:
            break

    yield {
        "type": "pipeline_complete",
        "job_id": ctx.job_id,
        "final_answer": ctx.final_answer,
        "policy_violations": ctx.policy_violations,
    }