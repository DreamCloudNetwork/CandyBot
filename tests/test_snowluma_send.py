"""SnowlumaClient.send_group_msg：消息段透传、message_id 解析与失败暴露。"""

from __future__ import annotations

import pytest

from candybot.snowluma import SnowlumaClient, SnowlumaSettings, SnowlumaError

SETTINGS = SnowlumaSettings(
    mcp_command="npx",
    mcp_args=["-y", "@snowluma/mcp"],
    endpoint="http://10.0.0.5:3000/",
    api_key="t",
    mode="write",
    timeout_ms=30000,
    allow_private_endpoint=True,
)


class RecordingClient(SnowlumaClient):
    """替掉传输层：记录 invoke_action 的 params 并按脚本返回响应。"""

    def __init__(self, payload: object, is_error: bool = False):
        super().__init__(SETTINGS)
        self.payload = payload
        self.is_error = is_error
        self.calls: list[tuple[str, dict]] = []

    async def _call_tool_json(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        return self.is_error, self.payload


async def test_send_text_wraps_into_text_segment_and_returns_id():
    client = RecordingClient({"status": "ok", "retcode": 0, "data": {"message_id": 123}})
    mid = await client.send_group_msg(42, "你好呀")
    assert mid == 123
    name, args = client.calls[0]
    assert name == "invoke_action"
    assert args["action"] == "send_group_msg"
    assert args["params"] == {
        "group_id": 42,
        "message": [{"type": "text", "data": {"text": "你好呀"}}],
    }


async def test_send_accepts_segment_array_verbatim():
    segments = [
        {"type": "reply", "data": {"id": "123"}},
        {"type": "text", "data": {"text": "＊正确词"}},
    ]
    client = RecordingClient({"retcode": 0, "data": {"message_id": 456}})
    mid = await client.send_group_msg(42, segments)
    assert mid == 456
    assert client.calls[0][1]["params"]["message"] == segments


async def test_missing_message_id_returns_none():
    client = RecordingClient({"retcode": 0, "data": {}})
    assert await client.send_group_msg(42, "嗨") is None


async def test_failed_retcode_raises_even_without_mcp_error():
    """SnowLuma 对 retcode≠0 不报 MCP 错误而是当数据返回：必须识别为失败。"""
    client = RecordingClient(
        {"status": "failed", "retcode": 1, "wording": "internal error", "data": None}
    )
    with pytest.raises(SnowlumaError, match="retcode=1"):
        await client.send_group_msg(42, "你好")


async def test_mcp_level_error_still_raises():
    client = RecordingClient("工具调用炸了", is_error=True)
    with pytest.raises(SnowlumaError):
        await client.send_group_msg(42, "你好")
