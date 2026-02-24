"""EvaluatorCallbackHandler that posts LLM-as-judge scores after each trace."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler
from langsmith import Client

from .evaluators import eval_composite

logger = logging.getLogger(__name__)

_READ_RETRIES = 3
_READ_RETRY_DELAYS_SECONDS = (0.25, 0.5, 1.0)


class OnlineEvalCallbackHandler(BaseCallbackHandler):
    """Callback handler that runs composite evaluation after a trace completes.

    Designed to run in a background thread so evaluation latency does not
    affect the user-facing response time.

    Usage::

        handler = OnlineEvalCallbackHandler()
        result = await app.ainvoke(inputs, config={"callbacks": [handler]})
    """

    def __init__(self, client: Client | None = None):
        self._client = client or Client()

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID | str | None = None,
        **kwargs: Any,
    ) -> None:
        """Fire after root chain completes — schedule eval in background thread."""
        parent_run_id = kwargs.get("parent_run_id")
        if parent_run_id is not None:
            return

        if run_id is None:
            return

        thread = threading.Thread(
            target=self._run_eval_sync,
            args=(str(run_id),),
            daemon=True,
        )
        thread.start()

    def _run_eval_sync(self, run_id: str) -> None:
        """Run composite eval and post feedback scores to LangSmith."""
        run = None
        for attempt in range(1, _READ_RETRIES + 1):
            try:
                run = self._client.read_run(run_id)
                break
            except Exception:
                if attempt >= _READ_RETRIES:
                    logger.warning(
                        "Online eval: failed to read run %s after %s attempts",
                        run_id,
                        _READ_RETRIES,
                        exc_info=True,
                    )
                    return
                time.sleep(_READ_RETRY_DELAYS_SECONDS[min(attempt - 1, len(_READ_RETRY_DELAYS_SECONDS) - 1)])

        if run is None:
            return

        try:
            result = eval_composite(run, self._client)
        except Exception:
            logger.warning("Online eval: composite eval failed for run %s", run_id, exc_info=True)
            return

        self._post_feedback(run_id, result)

    def _post_feedback(self, run_id: str, result: dict[str, Any]) -> None:
        """Post all three feedback scores to LangSmith."""
        answer_result = result.get("answer_result", {})
        process_result = result.get("process_result", {})

        for feedback in [answer_result, process_result, result]:
            key = feedback.get("key")
            score = feedback.get("score")
            comment = feedback.get("comment", "")
            if key and score is not None:
                try:
                    self._client.create_feedback(
                        run_id=run_id,
                        key=key,
                        score=score,
                        comment=comment[:4000],
                    )
                except Exception:
                    logger.warning("Online eval: failed to post feedback %s for run %s", key, run_id, exc_info=True)


def build_eval_callback(client: Client | None = None) -> OnlineEvalCallbackHandler:
    """Create an OnlineEvalCallbackHandler ready to attach to invoke config."""
    return OnlineEvalCallbackHandler(client=client)


def attach_online_eval_callback(
    config: dict[str, Any] | None = None,
    *,
    client: Client | None = None,
) -> dict[str, Any]:
    """Return invoke config with one OnlineEvalCallbackHandler attached.

    Use this for API/Studio-style invocations where callbacks are not injected
    by the CLI wrapper.
    """
    resolved = dict(config or {})
    callbacks = list(resolved.get("callbacks", []))
    if not any(isinstance(callback, OnlineEvalCallbackHandler) for callback in callbacks):
        callbacks.append(build_eval_callback(client=client))
    resolved["callbacks"] = callbacks
    return resolved
