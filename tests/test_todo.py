"""Regression tests for todo creation input handling."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from agents.tool_context import ToolContext

from strix.tools.todo import tools
from strix.tools.todo.tools import create_todo


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.fixture(autouse=True)
async def _isolate_store() -> AsyncIterator[None]:
    tools._todos_storage.clear()
    yield
    tools._todos_storage.clear()


async def _create(todos: list[Any], agent_id: str = "root") -> dict[str, Any]:
    ctx = ToolContext(
        context={"agent_id": agent_id},
        tool_name="create_todo",
        tool_call_id="call-1",
        tool_arguments="{}",
    )
    raw = await create_todo.on_invoke_tool(ctx, json.dumps({"todos": json.dumps(todos)}))
    return json.loads(raw)  # type: ignore[no-any-return]


@pytest.mark.asyncio
async def test_an_unknown_priority_no_longer_discards_the_batch() -> None:
    """A model-chosen label like "medium" must not lose the whole plan.

    ``_normalize_priority`` raised on any value outside the four valid labels,
    and ``create_todo`` turned that into a failed batch: every todo in the same
    call was dropped, including the ones with acceptable priorities.
    """
    result = await _create(
        [
            {"title": "Recon", "priority": "medium"},
            {"title": "Probe /admin", "priority": "high"},
            {"title": "Report"},
        ],
    )

    assert result["success"] is True
    assert result["created_count"] == 3
    by_title = {created["title"]: created["priority"] for created in result["created"]}
    assert by_title == {"Recon": "normal", "Probe /admin": "high", "Report": "normal"}


@pytest.mark.asyncio
async def test_a_non_string_priority_no_longer_discards_the_batch() -> None:
    """A numeric priority used to raise AttributeError past the error handler."""
    result = await _create(
        [
            {"title": "Recon", "priority": 2},
            {"title": "Report"},
        ],
    )

    assert result["success"] is True
    assert result["created_count"] == 2
    assert {created["priority"] for created in result["created"]} == {"normal"}
