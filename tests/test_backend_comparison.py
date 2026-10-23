import asyncio

from friday import backend_comparison as comparison
from friday.brain import LLMEvent, ToolCall, configured_transport


class FakeTransport:
    def __init__(self, response="", calls=()):
        self.response, self.calls = response, calls

    async def stream(self, request):
        yield LLMEvent(kind="text", text=self.response)
        yield LLMEvent(kind="turn_end", tool_calls=self.calls)


def test_fixed_suite_scores_quality_and_tool_routing():
    arithmetic = comparison.TASKS[0]
    result = asyncio.run(comparison.observe(
        FakeTransport('{"net":220,"vat":41.8,"gross":261.8}'), arithmetic
    ))
    assert result["quality_pass"] is True
    assert result["routing_pass"] is None

    route = comparison.TASKS[2]
    result = asyncio.run(comparison.observe(
        FakeTransport(calls=(ToolCall("1", "list_processes", {}),)), route
    ))
    assert result["routing_pass"] is True
    assert result["quality_pass"] is None


def test_backend_selection_is_opt_in_and_validated(monkeypatch):
    from friday import local_transformers

    class StubTransformersTransport:
        def __init__(self, model):
            self.model = model

    monkeypatch.setattr(local_transformers, "TransformersTransport", StubTransformersTransport)
    monkeypatch.delenv("FRIDAY_LLM_BACKEND", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert configured_transport().__class__.__name__ == "AnthropicTransport"
    monkeypatch.setenv("FRIDAY_LLM_BACKEND", "transformers")
    monkeypatch.setenv("FRIDAY_LOCAL_MODEL", "test/model")
    assert configured_transport().model == "test/model"
