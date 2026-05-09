import asyncio
import sys
from io import StringIO
from api.tools.base import ToolResult, FAILURE_TIMEOUT, FAILURE_EMPTY, FAILURE_MALFORMED

TIMEOUT_SECONDS = 10
BLOCKED = ["import os", "import sys", "import subprocess", "open(", "__import__"]


async def run(code: str) -> ToolResult:
    if not code or not isinstance(code, str):
        return ToolResult(
            success=False,
            failure_mode=FAILURE_MALFORMED,
            error_message="code must be a non-empty string",
        )

    for blocked in BLOCKED:
        if blocked in code:
            return ToolResult(
                success=False,
                failure_mode=FAILURE_MALFORMED,
                error_message=f"blocked statement detected: {blocked}",
            )

    try:
        result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _execute, code),
            timeout=TIMEOUT_SECONDS,
        )
        return result
    except asyncio.TimeoutError:
        return ToolResult(
            success=False,
            failure_mode=FAILURE_TIMEOUT,
            error_message=f"code execution timed out after {TIMEOUT_SECONDS}s",
        )


def _execute(code: str) -> ToolResult:
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    exit_code = 0

    try:
        import contextlib
        with contextlib.redirect_stdout(stdout_capture), \
             contextlib.redirect_stderr(stderr_capture):
            exec(compile(code, "<sandbox>", "exec"), {})
    except Exception as e:
        stderr_capture.write(str(e))
        exit_code = 1

    stdout = stdout_capture.getvalue()
    stderr = stderr_capture.getvalue()

    if not stdout and not stderr and exit_code == 0:
        return ToolResult(
            success=True,
            data={"stdout": "", "stderr": "", "exit_code": 0},
        )

    return ToolResult(
        success=exit_code == 0,
        data={
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
        },
        failure_mode=None if exit_code == 0 else "execution_error",
        error_message=stderr if exit_code != 0 else None,
    )