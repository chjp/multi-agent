from __future__ import annotations

from typing import Any

import pytest
from openai.types.responses import ResponseCompletedEvent, ResponseOutputMessage, ResponseOutputText

from agents import ManualModel, ModelSettings, ModelTracing
from agents.handoffs import Handoff
from agents.tool import FunctionTool


@pytest.mark.asyncio
async def test_get_response_manual_model() -> None:
    printed: list[str] = []
    responses = ["Manual reply"]

    async def response_provider() -> str:
        return responses.pop(0)

    async def noop_tool(context: Any, arguments: str) -> str:
        return "ok"

    async def noop_handoff(context: Any, arguments: str):
        raise AssertionError("handoff should not run in test")

    tool = FunctionTool(
        name="test_tool",
        description="A simple helper.",
        params_json_schema={},
        on_invoke_tool=noop_tool,
    )
    handoff = Handoff(
        tool_name="transfer_to_helper",
        tool_description="Transfer to helper agent.",
        input_json_schema={},
        on_invoke_handoff=noop_handoff,
        agent_name="Helper",
    )

    model = ManualModel(print_fn=printed.append, response_provider=response_provider)

    response = await model.get_response(
        system_instructions="Be concise.",
        input="Hello",
        model_settings=ModelSettings(),
        tools=[tool],
        output_schema=None,
        handoffs=[handoff],
        tracing=ModelTracing.DISABLED,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    )

    assert len(response.output) == 1
    message = response.output[0]
    assert isinstance(message, ResponseOutputMessage)
    assert len(message.content) == 1
    first_content = message.content[0]
    assert isinstance(first_content, ResponseOutputText)
    assert first_content.text == "Manual reply"
    assert response.response_id == "manual-response"
    assert response.usage.requests == 0
    assert response.usage.total_tokens == 0

    assert printed[0] == "=== Manual model interaction ==="
    assert any("Hello" in line for line in printed)
    assert any("test_tool" in line for line in printed)
    assert any("Helper" in line for line in printed)


@pytest.mark.asyncio
async def test_stream_response_manual_model() -> None:
    printed: list[str] = []
    responses = ["Streamed manual response"]

    async def response_provider() -> str:
        return responses.pop(0)

    model = ManualModel(print_fn=printed.append, response_provider=response_provider)

    events = [
        event
        async for event in model.stream_response(
            system_instructions=None,
            input=[{"role": "user", "content": "Ping"}],
            model_settings=ModelSettings(),
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
        )
    ]

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, ResponseCompletedEvent)
    assert event.response.id == "manual-response"
    assert event.response.usage is not None
    assert event.response.usage.input_tokens == 0
    assert event.response.usage.output_tokens == 0
    assert len(event.response.output) == 1
    streamed_message = event.response.output[0]
    assert isinstance(streamed_message, ResponseOutputMessage)
    assert streamed_message.content
    first_content = streamed_message.content[0]
    assert isinstance(first_content, ResponseOutputText)
    assert first_content.text == "Streamed manual response"

    assert printed[0] == "=== Manual model interaction ==="
    assert any("Ping" in line for line in printed)
    assert any("Available tools: none" == line for line in printed)
    assert any("Available handoffs: none" == line for line in printed)
