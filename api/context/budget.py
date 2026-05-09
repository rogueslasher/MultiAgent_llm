import tiktoken
from api.context.schema import SharedContext, AgentID

ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


class BudgetManager:
    def __init__(self, ctx: SharedContext):
        self.ctx = ctx

    def declare(self, agent_id: str, budget: int):
        self.ctx.token_budgets[agent_id] = budget
        if agent_id not in self.ctx.token_usage:
            self.ctx.token_usage[agent_id] = 0

    def remaining(self, agent_id: str) -> int:
        budget = self.ctx.token_budgets.get(agent_id, 0)
        used = self.ctx.token_usage.get(agent_id, 0)
        return max(0, budget - used)

    def consume(self, agent_id: str, text: str) -> bool:
        """
        Returns True if within budget, False if violation.
        Logs violation to context but does not truncate.
        """
        tokens = count_tokens(text)
        remaining = self.remaining(agent_id)

        if tokens > remaining:
            violation = (
                f"{agent_id} policy violation: "
                f"attempted to consume {tokens} tokens, "
                f"only {remaining} remaining"
            )
            self.ctx.policy_violations.append(violation)
            self.ctx.token_usage[agent_id] = (
                self.ctx.token_usage.get(agent_id, 0) + tokens
            )
            return False

        self.ctx.token_usage[agent_id] = (
            self.ctx.token_usage.get(agent_id, 0) + tokens
        )
        return True

    def needs_compression(self, agent_id: str, text: str) -> bool:
        tokens = count_tokens(text)
        return tokens > self.remaining(agent_id)