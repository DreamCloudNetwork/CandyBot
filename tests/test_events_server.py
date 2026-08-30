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


# ------------------------------------------------- 表情包只读供图路由（http 模式）

_SHA = "a" * 64


def _make_sticker_client(tmp_path):
    """挂上表情包目录的事件服务测试客户端；返回 (client, 根目录)。"""
    root = tmp_path / "stickers"
    (root / "42").mkdir(parents=True)
    (root / "42" / f"{_SHA}.png").write_bytes(b"\x89PNG fake")

    async def handler(event):  # pragma: no cover
        raise AssertionError("供图不应触发事件回调")

    server = EventsServer(
        handler, host="127.0.0.1", port=0, stickers_dir=root
    )
    return TestClient(TestServer(server._make_app())), root


def test_sticker_route_serves_file(tmp_path):
    async def flow():
        client, _ = _make_sticker_client(tmp_path)
        await client.__aenter__()
        resp = await client.get(f"/stickers/42/{_SHA}.png")
        assert resp.status == 200
        assert resp.headers["Content-Type"] == "image/png"
        # 文件名即内容指纹：允许长期缓存，外链被反复取图也不重传
        assert "immutable" in resp.headers["Cache-Control"]
        assert await resp.read() == b"\x89PNG fake"
        await client.close()

    run(flow())


def test_sticker_route_rejects_non_fingerprint_names(tmp_path):
    """只认收藏命名规则（纯数字群号 + 64 位小写十六进制指纹 + 已知图片后缀）。"""

    async def flow():
        client, _ = _make_sticker_client(tmp_path)
        await client.__aenter__()
        for bad in (
            f"/stickers/42/{_SHA.upper()}.png",  # 大写十六进制
            f"/stickers/42/{'b' * 63}.png",  # 指纹长度不足
            f"/stickers/42/{_SHA}.exe",  # 后缀不认识
            "/stickers/4a/" + _SHA + ".png",  # 群号非纯数字
            f"/stickers/99/{_SHA}.png",  # 合法命名但该图不存在
        ):
            resp = await client.get(bad)
            assert resp.status == 404, bad
        await client.close()

    run(flow())


def test_sticker_route_not_registered_without_dir(tmp_path):
    """没配表情包目录时压根没有这条路由（POST /onebot/event 不受影响）。"""

    async def handler(event):
        pass

    async def flow():
        client = await _make_app_client(handler).__aenter__()
        resp = await client.get(f"/stickers/42/{_SHA}.png")
        assert resp.status == 404
        ok = await _post_event(client, {"post_type": "message"})
        assert ok.status == 204
        await client.close()

    run(flow())


def test_sticker_route_blocks_traversal(tmp_path):
    """越界写法拿不到表情包目录之外的文件。"""

    async def flow():
        client, root = _make_sticker_client(tmp_path)
        await client.__aenter__()
        (tmp_path / "secret.txt").write_text("secret")
        # 裸 `../` 会被 URL 规范化成路由之外的路径；编码后的 `%2F` 才会进到
        # 路由里，由命名校验拦下
        for bad in (
            "/stickers/42/../../secret.txt",
            "/stickers/42/..%2F..%2Fsecret.txt",
            "/stickers/42/%2e%2e%2fsecret.txt",
            "/stickers/42/secret.txt",
        ):
            resp = await client.get(bad)
            assert resp.status == 404, bad
        assert (root / "42" / f"{_SHA}.png").exists()
        await client.close()

    run(flow())
