"""接收 SnowLuma/OneBot HTTP POST 事件上报的轻量 aiohttp 服务。"""

from __future__ import annotations

import hashlib
import hmac
import logging
from collections.abc import Awaitable, Callable

from aiohttp import web

logger = logging.getLogger(__name__)

# 事件上报请求体默认上限（bot.max_event_body_bytes 未配置时的内置值）：
# direct 模式的大图事件正文可能超过它，超限返回 413。
MAX_BODY_BYTES = 1 * 1024 * 1024


def verify_signature(secret: str | None, body: bytes, header_value: str | None) -> bool:
    """OneBot v11 标准：X-Signature = hex(hmac_sha1(secret, body))。"""
    if not secret:
        return True
    if not header_value:
        return False
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha1).hexdigest()
    return hmac.compare_digest(expected, header_value.strip())


class EventsServer:
    """POST /onebot/event → 校验签名 → 交给 async 回调处理。"""

    def __init__(
        self,
        handler: Callable[[dict], Awaitable[None]],
        *,
        host: str,
        port: int,
        secret: str | None = None,
        max_body_bytes: int = MAX_BODY_BYTES,
    ):
        self._handler = handler
        self._host = host
        self._port = port
        self._secret = secret
        self._max_body_bytes = int(max_body_bytes)
        self._runner: web.AppRunner | None = None

    def _make_app(self) -> web.Application:
        app = web.Application(client_max_size=self._max_body_bytes)
        app.router.add_post("/onebot/event", self._on_event)
        return app

    async def _on_event(self, request: web.Request) -> web.Response:
        body = await request.read()
        logger.debug("收到事件，body: %s",body)
        if not verify_signature(self._secret, body, request.headers.get("X-Signature")):
            logger.warning("事件上报签名校验失败，来源 %s", request.remote)
            return web.Response(status=403)
        try:
            import json

            event = json.loads(body)
        except ValueError:
            logger.warning("事件上报不是合法 JSON：%r", body[:200])
            return web.Response(status=400)
        if not isinstance(event, dict):
            return web.Response(status=400)
        # 无论处理结果如何都回 2xx，避免 OneBot 端反复重推积压
        try:
            await self._handler(event)
        except Exception:
            logger.exception("处理事件时出错（已忽略）")
        return web.Response(status=204)

    async def start(self) -> None:
        self._runner = web.AppRunner(self._make_app(), access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        logger.info("事件服务已启动：http://%s:%d/onebot/event", self._host, self._port)

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            logger.info("事件服务已停止")
