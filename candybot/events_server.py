"""接收 SnowLuma/OneBot HTTP POST 事件上报的轻量 aiohttp 服务。

顺带在配置了表情包目录时挂一条只读 GET /stickers/<群号>/<文件> 路由，
供 stickers.send_mode=http 的跟发外链取图（见 stickers.py）。
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

from aiohttp import web

from .stickers import STICKER_URL_PREFIX, resolve_sticker_file, sticker_content_type

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
    """POST /onebot/event → 校验签名 → 交给 async 回调处理。

    stickers_dir 非空时额外挂表情包只读供图路由（send_mode=http 用）。
    """

    def __init__(
        self,
        handler: Callable[[dict], Awaitable[None]],
        *,
        host: str,
        port: int,
        secret: str | None = None,
        max_body_bytes: int = MAX_BODY_BYTES,
        stickers_dir: Path | None = None,
    ):
        self._handler = handler
        self._host = host
        self._port = port
        self._secret = secret
        self._max_body_bytes = int(max_body_bytes)
        self._stickers_dir = stickers_dir
        self._runner: web.AppRunner | None = None

    def _make_app(self) -> web.Application:
        app = web.Application(client_max_size=self._max_body_bytes)
        app.router.add_post("/onebot/event", self._on_event)
        if self._stickers_dir is not None:
            app.router.add_get(
                f"{STICKER_URL_PREFIX}/{{group_id}}/{{filename}}", self._serve_sticker
            )
        return app

    async def _serve_sticker(self, request: web.Request) -> web.Response:
        """按收藏命名规则回表情包文件；不合规则一律 404，不透露原因。"""
        if self._stickers_dir is None:  # 路由本就只在配了目录时挂载
            return web.Response(status=404)
        path = resolve_sticker_file(
            self._stickers_dir,
            request.match_info["group_id"],
            request.match_info["filename"],
        )
        if path is None or not path.is_file():
            return web.Response(status=404)
        content_type = sticker_content_type(path.name)
        # 文件名即内容指纹，内容不会变：允许长期缓存，外链被反复取图也不重传
        return web.FileResponse(
            path,
            headers={
                "Content-Type": content_type or "application/octet-stream",
                "Cache-Control": "public, max-age=31536000, immutable",
            },
        )

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
        if self._stickers_dir is not None:
            logger.info(
                "表情包供图路由已挂载：%s/<群号>/<指纹>.<ext>（根目录 %s）",
                STICKER_URL_PREFIX,
                self._stickers_dir,
            )

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            logger.info("事件服务已停止")
