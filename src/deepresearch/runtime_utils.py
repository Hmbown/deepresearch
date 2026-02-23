"""Runnable invocation helpers shared across graph modules."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig

_logger = logging.getLogger(__name__)


def runnable_supports_config(callable_obj: Any) -> bool:
    """Return whether a runnable call target accepts a `config` argument."""
    try:
        parameters = inspect.signature(callable_obj).parameters.values()
    except (TypeError, ValueError):
        return True
    has_var_kwargs = any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters)
    if has_var_kwargs:
        return True
    return any(param.name == "config" for param in parameters)


async def invoke_runnable_with_config(runnable: Any, payload: Any, config: RunnableConfig | None) -> Any:
    """Invoke a runnable while passing RunnableConfig when supported."""
    if hasattr(runnable, "ainvoke"):
        ainvoke = runnable.ainvoke
        if config is not None and runnable_supports_config(ainvoke):
            return await ainvoke(payload, config=config)
        return await ainvoke(payload)
    invoke = runnable.invoke
    if config is not None and runnable_supports_config(invoke):
        return invoke(payload, config=config)
    return invoke(payload)


async def invoke_structured_with_retries(
    structured_model: Any,
    prompt_content: str,
    config: RunnableConfig | None,
    schema_name: str,
    *,
    max_retries: int = 1,
) -> Any | None:
    """Invoke structured-output models with bounded retry and deterministic fallback."""
    for attempt in range(max_retries):
        try:
            return await invoke_runnable_with_config(
                structured_model,
                [HumanMessage(content=prompt_content)],
                config,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _logger.warning(
                "Structured output failure for %s (attempt %s/%s): %s",
                schema_name,
                attempt + 1,
                max_retries,
                exc,
            )
            if attempt + 1 >= max_retries:
                return None
