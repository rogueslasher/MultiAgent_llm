from pydantic import BaseModel
from typing import Any, Optional


class ToolResult(BaseModel):
    success: bool
    data: Optional[dict[str, Any]] = None
    failure_mode: Optional[str] = None
    error_message: Optional[str] = None


FAILURE_TIMEOUT = "timeout"
FAILURE_EMPTY = "empty_results"
FAILURE_MALFORMED = "malformed_input"