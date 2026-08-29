"""SnowLuma HTTP 客户端封装。

直接调用 SnowLuma 的 OneBot v11 兼容 HTTP API（文档：
https://snowluma.github.io/api/index.html ）：每个 action 通过
``POST {endpoint}/{action}`` 调用，请求体为 JSON 参数，响应为标准
OneBot 信封；api_key 非空时以 ``Authorization: Bearer`` 头携带。
send_group_msg 支持纯文本与 OneBot v11 消息段数组（如 reply 引用段），
并从响应中解出 message_id 供引用更正使用。
"""

from __future__ import annotations

import json
import logging

import aiohttp

from .models import SnowlumaSettings

logger = logging.getLogger(__name__)


def _text_segments(message: str | list[dict]) -> list[dict]:
    """纯文本包成单个 text 段；段数组原样透传（OneBot v11 segment[]）。"""
    if isinstance(message, str):
        return [{"type": "text", "data": {"text": message}}]
    return message


def _raise_for_failed_response(envelope: object) -> None:
    """按 OneBot 信封判定逻辑失败：retcode≠0 时 SnowLuma 仍返回 HTTP 200，
    不检查会把发送失败当成功，后续拿不到 message_id 的引用更正也会静默丢失。"""
    if (
        isinstance(envelope, dict)
        and "retcode" in envelope
        and envelope.get("retcode", 0) != 0
    ):
        raise SnowlumaError(
            "发送群消息失败："
            f"retcode={envelope.get('retcode')!r} "
            f"wording={envelope.get('wording')!r} "
            f"status={envelope.get('status')!r}"
        )


def _extract_message_id(payload: object) -> int | None:
    """从 OneBot 响应 {retcode, data:{message_id}} 里取发出的消息 id。"""
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    source = data if isinstance(data, dict) else payload
    try:
        return int(source["message_id"])
    except (KeyError, TypeError, ValueError):
        return None


class SnowlumaError(RuntimeError):
    """SnowLuma 调用层错误（HTTP 失败或业务 retcode≠0）。"""


class SnowlumaClient:
    """管理到 SnowLuma HTTP API 的 aiohttp 会话的完整生命周期。"""

    def __init__(self, settings: SnowlumaSettings):
        self._settings = settings
        # endpoint 补尾部 '/'，让 action 以相对路径拼在其后（含路径前缀的部署也能对上）
        endpoint = settings.endpoint
        self._base_url = endpoint if endpoint.endswith("/") else endpoint + "/"
        self._session: aiohttp.ClientSession | None = None

    # ------------------------------------------------------------ 生命周期

    async def start(self) -> None:
        s = self._settings
        timeout = aiohttp.ClientTimeout(total=max(s.timeout_ms / 1000.0, 5.0))
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if s.api_key:
            headers["Authorization"] = f"Bearer {s.api_key}"
        self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        logger.info("SnowLuma HTTP 客户端已启动 → %s", s.endpoint)

    async def stop(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            try:
                await session.close()
            except Exception as exc:
                logger.warning("关闭 HTTP 会话出错（忽略）：%s", exc)

    # ------------------------------------------------------------ action 调用

    async def call_action(self, action: str, **params) -> dict:
        """POST {endpoint}/{action}，请求体为 params，返回解析后的 OneBot 信封。

        HTTP 层失败（连接错误、非 2xx）与响应不是 JSON 对象都抛
        SnowlumaError；retcode 判定留给调用方（见 _raise_for_failed_response）。
        """
        if self._session is None:
            raise SnowlumaError("SnowLuma HTTP 客户端尚未启动")
        url = self._base_url + action
        try:
            async with self._session.post(url, json=params) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    raise SnowlumaError(
                        f"{action} 返回 HTTP {resp.status}：{text[:200]!r}"
                    )
        except SnowlumaError:
            raise
        except aiohttp.ClientError as exc:
            raise SnowlumaError(f"{action} 请求失败：{exc}") from exc
        try:
            envelope = json.loads(text)
        except ValueError:
            raise SnowlumaError(f"{action} 响应不是 JSON：{text[:200]!r}") from None
        if not isinstance(envelope, dict):
            raise SnowlumaError(f"{action} 响应不是 JSON 对象：{envelope!r}")
        return envelope

    async def probe(self) -> None:
        """探活：确认 HTTP 端点可达、鉴权通过（get_version_info 返回 retcode=0）。"""
        envelope = await self.call_action("get_version_info")
        retcode = envelope.get("retcode", 0)
        if retcode != 0:
            raise SnowlumaError(
                f"SnowLuma 探活失败：retcode={retcode!r} "
                f"wording={envelope.get('wording')!r}"
            )
        data = envelope.get("data")
        version = data.get("version") if isinstance(data, dict) else None
        logger.info("SnowLuma HTTP 连接正常（version=%r）", version)

    async def send_group_msg(
        self, group_id: int, message: str | list[dict]
    ) -> int | None:
        """发送群消息，返回 OneBot 给出的 message_id（响应未携带时为 None）。

        message 可以是纯文本（包成单个 text 段，内容按字面发送、不解析 CQ
        码），也可以是 OneBot v11 消息段数组（如 [{"type": "reply", ...}]
        引用段）。
        """
        segments = _text_segments(message)
        envelope = await self.call_action(
            "send_group_msg", group_id=group_id, message=segments
        )
        _raise_for_failed_response(envelope)
        logger.info("已发送群消息到 %d，响应：%r", group_id, envelope)
        return _extract_message_id(envelope)

    async def query_login_info(self) -> dict | None:
        """只读查询登录账号；失败返回 None 不影响启动。"""
        try:
            envelope = await self.call_action("get_login_info")
        except Exception as exc:
            logger.debug("get_login_info 失败：%s", exc)
            return None
        if envelope.get("retcode", 0) != 0:
            return None
        data = envelope.get("data")
        return data if isinstance(data, dict) else None
