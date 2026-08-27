from __future__ import annotations

import asyncio
import hashlib
import hmac
import json

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from candybot.events_server import EventsServer, verify_signature


def run(coro):
    return asyncio.run(coro)


def test_verify_signature():
    body = b'{"x":1}'
    secret = "s3cret"
    good = hmac.new(secret.encode(), body, hashlib.sha1).hexdigest()
    assert verify_signature(secret, body, good)
    assert not verify_signature(secret, body, "deadbeef")
    assert not verify_signature(secret, body, None)
    # 未配置 secret 时全部放行
    assert verify_signature(None, body, None)


def _make_app_client(handler, secret=None):
    server = EventsServer(handler, host="127.0.0.1", port=0, secret=secret)
    app = server._make_app()
    return TestClient(TestServer(app))


async def _post_event(client: TestClient, payload, headers=None):
    return await client.post(
        "/onebot/event",
        data=json.dumps(payload).encode() if isinstance(payload, dict) else payload,
        headers=headers or {},
    )


def test_post_event_dispatch_and_204():
    """EventsServer 只做传输：所有 JSON 事件都按序回调，类型过滤在 bot 层。"""
    received = []

    async def handler(event):
        received.append(event)

    async def flow():
        client = await _make_app_client(handler).__aenter__()
        resp = await _post_event(client, {"post_type": "meta_event"})
        assert resp.status == 204
        resp = await _post_event(client, {"post_type": "message", "message_id": 5})
        assert resp.status == 204
        await client.close()

    run(flow())
    assert [e["post_type"] for e in received] == ["meta_event", "message"]
    assert received[1]["message_id"] == 5


def test_bad_json_400():
    async def handler(event):  # pragma: no cover
        raise AssertionError("不应回调")

    async def flow():
        client = await _make_app_client(handler).__aenter__()
        resp = await client.post("/onebot/event", data=b"not-json")
        assert resp.status == 400
        await client.close()

    run(flow())


def test_signature_enforced_when_configured():
    got = []

    async def handler(event):
        got.append(event)

    secret = "top"
    body = json.dumps({"post_type": "message"}).encode()
    sign = hmac.new(secret.encode(), body, hashlib.sha1).hexdigest()

    async def flow():
        client = await _make_app_client(handler, secret=secret).__aenter__()
        bad = await client.post("/onebot/event", data=body, headers={"X-Signature": "no"})
        assert bad.status == 403
        ok = await client.post("/onebot/event", data=body, headers={"X-Signature": sign})
        assert ok.status == 204
        await client.close()

    run(flow())
    assert len(got) == 1


def test_handler_exception_still_204():
    async def handler(event):
        raise RuntimeError("boom")

    async def flow():
        client = await _make_app_client(handler).__aenter__()
        resp = await _post_event(client, {"post_type": "message"})
        assert resp.status == 204  # 不让 OneBot 端重推风暴
        await client.close()

    run(flow())
