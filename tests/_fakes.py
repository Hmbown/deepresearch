from __future__ import annotations

from langchain_core.messages import AIMessage


class FakeStructuredRunner:
    def __init__(self, owner, schema_name: str):
        self._owner = owner
        self._schema_name = schema_name

    async def ainvoke(self, messages, config=None):
        del config
        self._owner.structured_calls.append((self._schema_name, messages))
        queue = self._owner.structured_responses.get(self._schema_name, [])
        if not queue:
            raise AssertionError(f"No fake response configured for schema {self._schema_name}")
        response = queue.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeLLM:
    def __init__(self, *, structured_responses=None, freeform_responses=None):
        self.structured_responses = {name: list(values) for name, values in (structured_responses or {}).items()}
        self.freeform_responses = list(freeform_responses or [])
        self.structured_calls = []
        self.freeform_calls = []

    def with_structured_output(self, schema):
        return FakeStructuredRunner(self, schema.__name__)

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages, config=None):
        del config
        self.freeform_calls.append(messages)
        if not self.freeform_responses:
            return AIMessage(content="fallback freeform output")
        response = self.freeform_responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeSupervisorGraph:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def ainvoke(self, payload, config=None):
        del payload, config
        self.calls += 1
        if not self._responses:
            return {}
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response
