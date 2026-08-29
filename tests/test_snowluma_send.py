"""SnowlumaClient：OneBot v11 兼容 HTTP API 的调用封装。

RecordingClient 替掉 call_action 测业务语义（消息段透传、message_id 解析、
retcode 失败暴露）；HTTP 传输层用本地 aiohttp 测试服务端真跑一遍
（URL 拼接、Bearer 鉴权头、JSON 请求体与信封解析、非 2xx 报错）。
"""

from __future__ import annotations

import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from candybot.models import SnowlumaSettings
from candybot.snowluma import SnowlumaClient, SnowlumaError

SETTINGS = SnowlumaSettings(
    endpoint="http://10.0.0.5:3000/",
    api_key="t",
    timeout_ms=30000,
    allow_private_endpoint=True,
)


class RecordingClient(SnowlumaClient):
    """替掉传输层：记录 action 与 params 并按脚本返回响应信封。"""

    def __init__(self, payload: object, error: Exception | None = None):
        super().__init__(SETTINGS)
        self.payload = payload
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    async def call_action(self, action: str, **params) -> dict:
        self.calls.append((action, params))
        if self.error is not None:
            raise self.error
        return self.payload  # type: ignore[return-value]


# ------------------------------------------------------------ 业务语义


async def test_send_text_wraps_into_text_segment_and_returns_id():
    client = RecordingClient({"status": "ok", "retcode": 0, "data": {"message_id": 123}})
    mid = await client.send_group_msg(42, "你好呀")
    assert mid == 123
    action, params = client.calls[0]
    assert action == "send_group_msg"
    assert params == {
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
    assert client.calls[0][1]["message"] == segments


async def test_missing_message_id_returns_none():
    client = RecordingClient({"retcode": 0, "data": {}})
    assert await client.send_group_msg(42, "嗨") is None


async def test_failed_retcode_raises_even_with_http_200():
    """OneBot 约定 retcode≠0 也返回 HTTP 200：必须按信封识别为失败。"""
    client = RecordingClient(
        {"status": "failed", "retcode": 1, "wording": "internal error", "data": None}
    )
    with pytest.raises(SnowlumaError, match="retcode=1"):
        await client.send_group_msg(42, "你好")


async def test_transport_error_propagates():
    client = RecordingClient(None, error=SnowlumaError("send_group_msg 请求失败：boom"))
    with pytest.raises(SnowlumaError, match="boom"):
        await client.send_group_msg(42, "你好")


async def test_query_login_info_returns_data_or_none():
    ok = RecordingClient({"retcode": 0, "data": {"user_id": 99, "nickname": "糖糖"}})
    assert (await ok.query_login_info())["user_id"] == 99
    failed = RecordingClient({"retcode": 401, "status": "failed"})
    assert await failed.query_login_info() is None
    broken = RecordingClient(None, error=SnowlumaError("连接拒绝"))
    assert await broken.query_login_info() is None


async def test_call_before_start_raises():
    client = SnowlumaClient(SETTINGS)
    with pytest.raises(SnowlumaError, match="尚未启动"):
        await client.send_group_msg(42, "你好")


# ------------------------------------------------------------ HTTP 传输层


@pytest.fixture
async def onebot_server():
    """假的 SnowLuma HTTP API：记录请求并按 action 路径返回不同响应。"""
    requests: list[dict] = []

    async def handle(request: web.Request) -> web.Response:
        body = json.loads(await request.text())
        requests.append(
            {
                "path": request.path,
                "method": request.method,
                "auth": request.headers.get("Authorization"),
                "content_type": request.headers.get("Content-Type"),
                "body": body,
            }
        )
        action = request.path.lstrip("/")
        if action == "send_group_msg":
            return web.json_response(
                {"status": "ok", "retcode": 0, "data": {"message_id": 777}}
            )
        if action == "get_version_info":
            return web.json_response(
                {"status": "ok", "retcode": 0, "data": {"version": "1.0.0-test"}}
            )
        if action == "forbidden":
            return web.Response(status=403, text="forbidden")
        if action == "not_json":
            return web.Response(text="<html>nope</html>")
        return web.json_response(
            {
                "status": "failed",
                "retcode": 1,
                "wording": "unknown action",
                "data": None,
            }
        )

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handle)
    server = TestServer(app)
    await server.start_server()
    server.requests = requests  # type: ignore[attr-defined]
    try:
        yield server
    finally:
        await server.close()


def _settings_for(server: TestServer, api_key: str = "t") -> SnowlumaSettings:
    return SnowlumaSettings(
        endpoint=f"http://127.0.0.1:{server.port}",  # 故意不带尾部斜杠，验证拼接
        api_key=api_key,
        timeout_ms=5000,
        allow_private_endpoint=True,
    )


async def test_http_roundtrip(onebot_server):
    client = SnowlumaClient(_settings_for(onebot_server))
    await client.start()
    try:
        await client.probe()
        mid = await client.send_group_msg(42, "你好呀")
        assert mid == 777
    finally:
        await client.stop()
    send = onebot_server.requests[-1]
    assert send["method"] == "POST"
    assert send["path"] == "/send_group_msg"
    assert send["auth"] == "Bearer t"
    assert send["content_type"] == "application/json"
    assert send["body"] == {
        "group_id": 42,
        "message": [{"type": "text", "data": {"text": "你好呀"}}],
    }


async def test_no_auth_header_when_api_key_empty(onebot_server):
    client = SnowlumaClient(_settings_for(onebot_server, api_key=""))
    await client.start()
    try:
        await client.send_group_msg(42, "嗨")
    finally:
        await client.stop()
    assert onebot_server.requests[-1]["auth"] is None


async def test_failed_retcode_exposed_over_http(onebot_server):
    """未知 action：HTTP 200 但信封 retcode=1 → call_action 返回、上层拒收。"""
    client = SnowlumaClient(_settings_for(onebot_server))
    await client.start()
    try:
        envelope = await client.call_action("get_group_list")
        assert envelope["retcode"] == 1
    finally:
        await client.stop()


async def test_probe_rejects_failed_retcode():
    client = RecordingClient({"retcode": 1, "status": "failed", "wording": "auth"})
    with pytest.raises(SnowlumaError, match="探活失败"):
        await client.probe()


async def test_http_error_status_and_bad_json_raise(onebot_server):
    client = SnowlumaClient(_settings_for(onebot_server))
    await client.start()
    try:
        with pytest.raises(SnowlumaError, match="HTTP 403"):
            await client.call_action("forbidden")
        with pytest.raises(SnowlumaError, match="不是 JSON"):
            await client.call_action("not_json")
    finally:
        await client.stop()
