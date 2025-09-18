from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from pprint import pformat
from typing import Any

from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
)
from openai.types.responses.response_prompt_param import ResponsePromptParam
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from ..agent_output import AgentOutputSchemaBase
from ..handoffs import Handoff
from ..items import ItemHelpers, ModelResponse, TResponseInputItem, TResponseStreamEvent
from ..model_settings import ModelSettings
from ..tool import Tool
from ..usage import Usage
from .interface import Model, ModelTracing

ManualResponseProvider = Callable[[], Awaitable[str]]
ManualPrintFn = Callable[[str], Any]


_RESPONSE_ID = "manual-response"
_MESSAGE_ID = "manual-message"


def _tool_name(tool: Tool) -> str:
    return getattr(tool, "name", tool.__class__.__name__)


def _tool_description(tool: Tool) -> str | None:
    return getattr(tool, "description", None) or getattr(tool, "tool_description", None)


def _format_tools(tools: Sequence[Tool]) -> list[str]:
    formatted: list[str] = []
    for tool in tools:
        name = _tool_name(tool)
        description = _tool_description(tool)
        if description:
            formatted.append(f"  - {name}: {description}")
        else:
            formatted.append(f"  - {name}")
    return formatted


def _format_handoffs(handoffs: Sequence[Handoff]) -> list[str]:
    formatted: list[str] = []
    for handoff in handoffs:
        description = handoff.tool_description
        if description:
            formatted.append(f"  - {handoff.agent_name} via `{handoff.tool_name}`: {description}")
        else:
            formatted.append(f"  - {handoff.agent_name} via `{handoff.tool_name}`")
    return formatted


def _default_response_provider() -> Awaitable[str]:
    return asyncio.to_thread(input, "Manual response: ")


async def manual_prompt_interaction(
    *,
    system_instructions: str | None,
    input: str | list[TResponseInputItem],
    tools: Sequence[Tool],
    handoffs: Sequence[Handoff],
    prompt: ResponsePromptParam | None,
    print_fn: ManualPrintFn = print,
    response_provider: ManualResponseProvider | None = None,
) -> str:
    """Prompt for a manual response by printing context to stdout."""

    normalized_input = ItemHelpers.input_to_new_input_list(input)

    print_fn("=== Manual model interaction ===")
    if system_instructions:
        print_fn("System instructions:")
        print_fn(system_instructions)
    else:
        print_fn("System instructions: <none>")

    print_fn("Conversation items:")
    print_fn(pformat(normalized_input, sort_dicts=False))

    if prompt is not None:
        print_fn("Prompt configuration:")
        print_fn(pformat(prompt, sort_dicts=False))

    if tools:
        print_fn("Available tools:")
        for line in _format_tools(tools):
            print_fn(line)
    else:
        print_fn("Available tools: none")

    if handoffs:
        print_fn("Available handoffs:")
        for line in _format_handoffs(handoffs):
            print_fn(line)
    else:
        print_fn("Available handoffs: none")

    print_fn("Provide the next assistant message.")

    provider = response_provider or _default_response_provider
    return await provider()


def _build_output_message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=_MESSAGE_ID,
        content=[
            ResponseOutputText(text=text, type="output_text", annotations=[]),
        ],
        role="assistant",
        status="completed",
        type="message",
    )


def _usage_to_response_usage(usage: Usage) -> ResponseUsage:
    return ResponseUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        input_tokens_details=InputTokensDetails(
            cached_tokens=usage.input_tokens_details.cached_tokens
        ),
        output_tokens_details=OutputTokensDetails(
            reasoning_tokens=usage.output_tokens_details.reasoning_tokens
        ),
    )


def _build_completed_event(message: ResponseOutputMessage, usage: Usage) -> ResponseCompletedEvent:
    response = Response(
        id=_RESPONSE_ID,
        created_at=0,
        model="manual",
        object="response",
        output=[message],
        tool_choice="none",
        tools=[],
        top_p=None,
        usage=_usage_to_response_usage(usage),
        parallel_tool_calls=False,
    )
    return ResponseCompletedEvent(
        type="response.completed",
        response=response,
        sequence_number=0,
    )


class ManualModel(Model):
    """A simple model implementation that prompts a human for responses."""

    def __init__(
        self,
        *,
        print_fn: ManualPrintFn = print,
        response_provider: ManualResponseProvider | None = None,
    ) -> None:
        self._print_fn = print_fn
        self._response_provider = response_provider

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> ModelResponse:
        del model_settings, output_schema, tracing, previous_response_id, conversation_id

        response_text = await manual_prompt_interaction(
            system_instructions=system_instructions,
            input=input,
            tools=tools,
            handoffs=handoffs,
            prompt=prompt,
            print_fn=self._print_fn,
            response_provider=self._response_provider,
        )

        message = _build_output_message(response_text)
        usage = Usage()
        return ModelResponse(output=[message], usage=usage, response_id=_RESPONSE_ID)

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        del model_settings, output_schema, tracing, previous_response_id, conversation_id

        response_text = await manual_prompt_interaction(
            system_instructions=system_instructions,
            input=input,
            tools=tools,
            handoffs=handoffs,
            prompt=prompt,
            print_fn=self._print_fn,
            response_provider=self._response_provider,
        )

        message = _build_output_message(response_text)
        usage = Usage()
        yield _build_completed_event(message, usage)
