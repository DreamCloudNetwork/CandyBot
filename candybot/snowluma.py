"""SnowLuma MCP 客户端封装。

通过 stdio 启动 ``@snowluma/mcp`` 子进程并保持长会话；发消息走 write 模式
的 ``invoke_action`` 工具（封装 OneBot v11 的 send_group_msg）。
send_group_msg 支持纯文本与 OneBot v11 消息段数组（如 reply 引用段），
并从响应中解出 message_id 供引用更正使用。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .models import SnowlumaSettings

logger = logging.getLogger(__name__)

_SEND_GROUP_MSG_RE = re.compile(r"send[_-]?group[_-]?msg|group.*send.*message", re.I)


def _text_segments(message: str | list[dict]) -> list[dict]:
    """纯文本包成单个 text 段；段数组原样透传（OneBot v11 segment[]）。"""
    if isinstance(message, str):
        return [{"type": "text", "data": {"text": message}}]
    return message


def _send_params(action: str, group_id: int, segments: list[dict]) -> dict:
    return {
        "action": action,
        "params": {"group_id": group_id, "message": segments},
    }


def _raise_for_failed_response(payload: object) -> None:
    """按 OneBot 信封判定逻辑失败：SnowLuma MCP 对 retcode≠0 不报 MCP 错误，
    而是把完整响应当数据返回。不检查 retcode 会把发送失败当成功，
    后续拿不到 message_id 的引用更正也会静默丢失。"""
    if isinstance(payload, dict):
        envelope = payload.get("data") if "retcode" not in payload else payload
        if isinstance(envelope, dict) and envelope.get("retcode", 0) != 0:
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
    """SnowLuma 调用层错误（业务失败或工具缺失）。"""


class SnowlumaClient:
    """管理一个 stdio MCP 会话的完整生命周期。"""

    def __init__(self, settings: SnowlumaSettings):
        self._settings = settings
        self._transport = None
        self._session: ClientSession | None = None
        self._send_tool_lock = asyncio.Lock()
        self._resolved_send_action: str | None = None

    # ------------------------------------------------------------ 生命周期

    async def start(self) -> None:
        s = self._settings
        env: dict[str, str] = {
            "SNOWLUMA_MCP_ENDPOINT": s.endpoint,
            "SNOWLUMA_MCP_MODE": s.mode,
            "SNOWLUMA_MCP_TIMEOUT_MS": str(s.timeout_ms),
        }
        if s.api_key:
            env["SNOWLUMA_MCP_TOKEN"] = s.api_key
        params = StdioServerParameters(
            command=s.mcp_command, args=list(s.mcp_args), env=env
        )
        logger.info("正在启动 SnowLuma MCP：%s %s → %s", s.mcp_command, " ".join(s.mcp_args), s.endpoint)
        self._transport = stdio_client(params)
        try:
            read_stream, write_stream = await self._transport.__aenter__()
        except Exception:
            self._transport = None
            raise
        try:
            self._session = ClientSession(read_stream, write_stream)
            await self._session.__aenter__()
            await self._session.initialize()
        except Exception:
            await self.stop()
            raise
        logger.info("SnowLuma MCP 会话已建立")

    async def stop(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            try:
                await session.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning("关闭 MCP 会话出错（忽略）：%s", exc)
        transport, self._transport = self._transport, None
        if transport is not None:
            try:
                await transport.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning("关闭 MCP 传输出错（忽略）：%s", exc)

    # ------------------------------------------------------------ 工具调用

    async def _list_tool_names(self) -> list[str]:
        assert self._session is not None
        result = await self._session.list_tools()
        return [tool.name for tool in result.tools]

    async def _call_tool_json(self, name: str, arguments: dict) -> tuple[bool, object]:
        """调用工具，返回 (is_error, 解析后的 JSON 或原始文本)。"""
        assert self._session is not None
        timeout = max(self._settings.timeout_ms / 1000.0, 5.0)
        result = await self._session.call_tool(name, arguments, read_timeout_seconds=timeout)
        texts: list[str] = []
        for block in result.content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                texts.append(text)
        payload: object
        if result.structured_content is not None:
            payload = result.structured_content
        elif len(texts) == 1:
            try:
                payload = json.loads(texts[0])
            except ValueError:
                payload = texts[0]
        else:
            payload = "\n".join(texts)
        return bool(result.is_error), payload

    async def probe(self) -> None:
        """探活：确认工具目录可读、invoke_action 可用且处于 write 模式。"""
        names = await self._list_tool_names()
        logger.info("SnowLuma MCP 提供工具：%s", ", ".join(names))
        if "invoke_action" not in names:
            raise SnowlumaError(
                f"MCP 工具列表中缺少 invoke_action（当前：{names}）；"
                '请把 snowluma.mode 设为 "write"'
            )

    async def send_group_msg(
        self, group_id: int, message: str | list[dict]
    ) -> int | None:
        """发送群消息，返回 OneBot 给出的 message_id（响应未携带时为 None）。

        message 可以是纯文本（包成单个 text 段，内容按字面发送、不解析 CQ
        码），也可以是 OneBot v11 消息段数组（如 [{"type": "reply", ...}]
        引用段）；unknown action 时自动从目录模糊匹配一次。
        """
        segments = _text_segments(message)
        action = self._resolved_send_action or "send_group_msg"
        params = _send_params(action, group_id, segments)
        try:
            is_error, payload = await self._call_tool_json("invoke_action", params)
        except Exception as exc:
            detail = str(exc).lower()
            if not ("unknown" in detail and "action" in detail):
                raise
            is_error, payload = await self._retry_with_fuzzy_action(group_id, segments)
        _raise_for_failed_response(payload)
        if is_error:
            raise SnowlumaError(f"发送群消息被拒绝：{payload!r}")
        logger.info("已发送群消息到 %d，响应：%r", group_id, payload)
        return _extract_message_id(payload)

    async def _retry_with_fuzzy_action(
        self, group_id: int, segments: list[dict]
    ) -> tuple[bool, object]:
        async with self._send_tool_lock:
            if self._resolved_send_action:
                params = _send_params(
                    self._resolved_send_action, group_id, segments
                )
                return await self._call_tool_json("invoke_action", params)
            names = await self._list_tool_names()
            matches = [n for n in names if _SEND_GROUP_MSG_RE.search(n)]
            if not matches:
                raise SnowlumaError(f"目录中找不到任何发消息 action（工具：{names}）")
            chosen = sorted(matches)[0]
            logger.warning("send_group_msg 不存在，改用目录匹配到的 %r", chosen)
            self._resolved_send_action = chosen
            params = _send_params(chosen, group_id, segments)
            return await self._call_tool_json("invoke_action", params)

    async def query_login_info(self) -> dict | None:
        """只读查询登录账号（read 模式也可用）；失败返回 None 不影响启动。"""
        for action in ("get_login_info",):
            try:
                is_error, payload = await self._call_tool_json(
                    "query_action", {"action": action}
                )
            except Exception as exc:
                logger.debug("query_action(%s) 失败：%s", action, exc)
                continue
            if not is_error and isinstance(payload, dict):
                data = payload.get("data", payload)
                if isinstance(data, dict):
                    return data
        return None
