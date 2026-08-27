"""领域模型与配置校验。

这里定义运行时消息记录（ChatRecord）、归一化结果（NormalizedMessage），
以及从 config.ConfigClass 解析出来的强类型配置（Settings 系列）。
endpoint 的 SSRF 校验也在本模块。
"""

from __future__ import annotations

import ipaddress
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

MULTIMODAL_MODES = ("direct", "describe", "placeholder")


# ---------------------------------------------------------------- 运行时模型


@dataclass(slots=True)
class ChatRecord:
    """一条群聊消息（含机器人自己发出的）。

    持久化到 JSONL 时只保留小字段；images（base64 大块）不落盘也不进历史。
    """

    message_id: int
    group_id: int
    user_id: int
    nickname: str
    text: str
    ts: float
    is_self: bool = False
    images: tuple[str, ...] = field(default=(), repr=False)

    def to_json(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "group_id": self.group_id,
            "user_id": self.user_id,
            "nickname": self.nickname,
            "text": self.text,
            "ts": self.ts,
            "is_self": self.is_self,
        }

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> ChatRecord:
        return cls(
            message_id=int(obj["message_id"]),
            group_id=int(obj["group_id"]),
            user_id=int(obj["user_id"]),
            nickname=str(obj.get("nickname", "")),
            text=str(obj.get("text", "")),
            ts=float(obj.get("ts", time.time())),
            is_self=bool(obj.get("is_self", False)),
        )


@dataclass(slots=True)
class NormalizedMessage:
    """normalize 后的群消息事件。"""

    record: ChatRecord
    mentioned_me: bool


@dataclass(slots=True)
class Decision:
    """一次发言决策的结果。"""

    should_reply: bool
    forced: bool = False
    score: int | None = None
    reason: str = ""


# ---------------------------------------------------------------- 配置模型


@dataclass(frozen=True)
class BotSettings:
    self_qq: int
    listen_host: str
    listen_port: int
    event_secret: str | None
    data_dir: str
    log_level: str  # 大写级别名，main.py 据此设置根 logger


@dataclass(frozen=True)
class GroupProfile:
    group_id: int | None  # None 表示 groups_default 兜底
    enabled: bool
    persona: str
    proactivity_threshold: int
    cooldown_seconds: int
    context_size: int


@dataclass(frozen=True)
class AISettings:
    base_url: str
    api_key: str


@dataclass(frozen=True)
class ModelSettings:
    judge: str
    reply: str
    vision: str | None


@dataclass(frozen=True)
class GenerationSettings:
    reply_max_tokens: int
    temperature: float
    max_context_chars: int
    timeout_seconds: float
    emoji_chance: float = 0.25   # 每条回复允许保留 emoji 的概率，0~1
    emoji_max: int = 2           # 允许保留时的最大 emoji 个数


@dataclass(frozen=True)
class MultimodalSettings:
    mode: str
    download_media: bool


@dataclass(frozen=True)
class RateLimitSettings:
    global_daily_limit: int | None


@dataclass(frozen=True)
class SnowlumaSettings:
    mcp_command: str
    mcp_args: list[str]
    endpoint: str
    api_key: str
    mode: str
    timeout_ms: int
    allow_private_endpoint: bool


@dataclass(frozen=True)
class Settings:
    bot: BotSettings
    groups: dict[int, GroupProfile]
    groups_default: GroupProfile
    ai_backend: AISettings
    models: ModelSettings
    generation: GenerationSettings
    multimodal: MultimodalSettings
    rate_limit: RateLimitSettings
    snowluma: SnowlumaSettings

    def profile_for(self, group_id: int) -> GroupProfile | None:
        """严格白名单语义。

        - ``groups`` 非空时：只有键中列出的群被服务；单条目可用自身字段覆盖
          默认值（哨兵值 -1 / "" 表示继承 ``groups_default``）；
          条目的 ``enabled: false`` 单独禁用该群。
        - ``groups`` 为空时：``groups_default.enabled == true`` 则服务所有群
          （全量兜底模式）；否则拒绝一切群。
        """
        profile = self.groups.get(group_id)
        if profile is not None:
            if not profile.enabled:
                return None
            return self._merge_profile(profile)
        # 未列入白名单：仅当根本没写任何白名单、且默认开启时才全量放行
        if not self.groups and self.groups_default.enabled:
            return self._merge_profile(
                GroupProfile(
                    group_id=None,
                    enabled=True,
                    persona="",
                    proactivity_threshold=-1,
                    cooldown_seconds=-1,
                    context_size=-1,
                )
            )
        return None

    def _merge_profile(self, profile: GroupProfile) -> GroupProfile:
        """把单群条目的哨兵值替换为 groups_default 的对应值。"""
        base = self.groups_default
        return GroupProfile(
            group_id=profile.group_id,
            enabled=profile.enabled,
            persona=profile.persona or base.persona,
            proactivity_threshold=(
                profile.proactivity_threshold
                if profile.proactivity_threshold >= 0
                else base.proactivity_threshold
            ),
            cooldown_seconds=(
                profile.cooldown_seconds
                if profile.cooldown_seconds >= 0
                else base.cooldown_seconds
            ),
            context_size=(
                profile.context_size if profile.context_size > 0 else base.context_size
            ),
        )


# ---------------------------------------------------------------- 配置解析


def _require_section(cfg: Any, name: str) -> dict[str, Any]:
    """按段名取配置段。兼容真实 ConfigClass 的属性访问与测试用映射访问。"""
    try:
        section = getattr(cfg, name)
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"config.json 缺少必需的配置段 `{name}`") from exc
    if not isinstance(section, dict):
        raise ValueError(f"config.json 中 `{name}` 应为对象")
    return section


def _get(section: dict[str, Any], key: str, default: Any) -> Any:
    value = section.get(key, default)
    return default if value is None else value


def _parse_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"配置项 `{key}` 应为布尔值")


def _parse_int(section: dict[str, Any], key: str, default: int) -> int:
    value = _get(section, key, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"配置项 `{key}` 应为整数，实际是 {value!r}") from exc


def _parse_optional_int(section: dict[str, Any], key: str) -> int | None:
    value = section.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"配置项 `{key}` 应为整数或 null，实际是 {value!r}") from exc


def _parse_str(section: dict[str, Any], key: str, default: str) -> str:
    value = _get(section, key, default)
    if not isinstance(value, str):
        raise ValueError(f"配置项 `{key}` 应为字符串")
    return value


def _parse_optional_str(section: dict[str, Any], key: str) -> str | None:
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"配置项 `{key}` 应为字符串或 null")
    return value


def load_settings(cfg: Any) -> Settings:
    """从 ConfigClass 单例（按段名取属性、返回 dict）解析并校验全部配置。"""
    bot_cfg = _require_section(cfg, "bot")
    self_qq = _parse_int(bot_cfg, "self_qq", 0)
    if self_qq <= 0:
        raise ValueError("config.json → bot.self_qq 必须配置为机器人的 QQ 号")
    listen_host = _parse_str(bot_cfg, "listen_host", "127.0.0.1")
    listen_port = _parse_int(bot_cfg, "listen_port", 5700)
    log_level = _parse_str(bot_cfg, "log_level", "INFO").upper()
    if log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError(
            "config.json → bot.log_level 只能是 DEBUG/INFO/WARNING/ERROR/CRITICAL，"
            f"实际是 {log_level!r}"
        )

    # 白名单：groups 里的每个 key 都是一个群号
    groups_raw = _require_section(cfg, "groups")
    groups: dict[int, GroupProfile] = {}
    for key, raw in groups_raw.items():
        gid = _coerce_group_id(key)
        groups[gid] = _parse_group_profile(raw, f"groups.{key}", gid)

    default_profile = _parse_group_profile(
        _require_section(cfg, "groups_default"), "groups_default", None
    )
    if not default_profile.persona:
        raise ValueError("groups_default.persona 不能为空")

    ai_cfg = _require_section(cfg, "ai_backend")
    ai_settings = AISettings(
        base_url=_parse_str(ai_cfg, "base_url", ""),
        api_key=_parse_str(ai_cfg, "api_key", ""),
    )

    models_cfg = _require_section(cfg, "models")
    model_settings = ModelSettings(
        judge=_parse_str(models_cfg, "judge", ""),
        reply=_parse_str(models_cfg, "reply", ""),
        vision=_parse_optional_str(models_cfg, "vision"),
    )
    if not model_settings.judge or not model_settings.reply:
        raise ValueError("config.json → models.judge / models.reply 必须指定模型名")

    gen_cfg = _require_section(cfg, "generation")
    emoji_chance = float(_get(gen_cfg, "emoji_chance", 0.25))
    if not 0 <= emoji_chance <= 1:
        raise ValueError(
            f"配置项 `generation.emoji_chance` 应在 0~1 之间，实际是 {emoji_chance!r}"
        )
    emoji_max = _parse_int(gen_cfg, "emoji_max", 2)
    if emoji_max < 0:
        raise ValueError(
            f"配置项 `generation.emoji_max` 不能为负数，实际是 {emoji_max!r}"
        )
    generation_settings = GenerationSettings(
        reply_max_tokens=_parse_int(gen_cfg, "reply_max_tokens", 500),
        temperature=float(_get(gen_cfg, "temperature", 0.8)),
        max_context_chars=_parse_int(gen_cfg, "max_context_chars", 8000),
        timeout_seconds=float(_get(gen_cfg, "timeout_seconds", 60)),
        emoji_chance=emoji_chance,
        emoji_max=emoji_max,
    )

    mm_cfg = _require_section(cfg, "multimodal")
    mm_mode = _parse_str(mm_cfg, "mode", "placeholder")
    if mm_mode not in MULTIMODAL_MODES:
        raise ValueError(
            f"multimodal.mode 只能是 {'/'.join(MULTIMODAL_MODES)}，实际是 {mm_mode!r}"
        )
    multimodal_settings = MultimodalSettings(
        mode=mm_mode,
        download_media=_parse_bool(mm_cfg.get("download_media", True), "download_media"),
    )

    rate_cfg = _require_section(cfg, "rate_limit")
    rate_limit_settings = RateLimitSettings(
        global_daily_limit=_parse_optional_int(rate_cfg, "global_daily_limit")
    )

    snow_cfg = _require_section(cfg, "snowluma")
    endpoint = _parse_str(snow_cfg, "endpoint", "")
    allow_private = _parse_bool(
        snow_cfg.get("allow_private_endpoint", False), "allow_private_endpoint"
    )
    validate_endpoint_url(endpoint, allow_private=allow_private)
    snowluma_settings = SnowlumaSettings(
        mcp_command=_parse_str(snow_cfg, "mcp_command", "npx"),
        mcp_args=list(_get(snow_cfg, "mcp_args", ["-y", "@snowluma/mcp"])),
        endpoint=endpoint,
        api_key=_parse_str(snow_cfg, "api_key", ""),
        mode=_parse_str(snow_cfg, "mode", "read"),
        timeout_ms=_parse_int(snow_cfg, "timeout_ms", 30000),
        allow_private_endpoint=allow_private,
    )
    if snowluma_settings.mode != "write":
        raise ValueError(
            'snowluma.mode 必须是 "write"，否则机器人无法调用 send_group_msg 发言'
        )

    return Settings(
        bot=BotSettings(
            self_qq=self_qq,
            listen_host=listen_host,
            listen_port=listen_port,
            event_secret=_parse_optional_str(bot_cfg, "event_secret"),
            data_dir=_parse_str(bot_cfg, "data_dir", "data"),
            log_level=log_level,
        ),
        groups=groups,
        groups_default=default_profile,
        ai_backend=ai_settings,
        models=model_settings,
        generation=generation_settings,
        multimodal=multimodal_settings,
        rate_limit=rate_limit_settings,
        snowluma=snowluma_settings,
    )


def _coerce_group_id(key: str) -> int:
    try:
        return int(key)
    except ValueError as exc:
        raise ValueError(f"groups 中的群号必须是整数字符串，实际是 {key!r}") from exc


def _parse_group_profile(raw: Any, label: str, group_id: int | None) -> GroupProfile:
    if not isinstance(raw, dict):
        raise ValueError(f"`{label}` 应为对象")
    return GroupProfile(
        group_id=group_id,
        enabled=_parse_bool(raw.get("enabled", True), f"{label}.enabled"),
        persona=_parse_str(raw, "persona", ""),
        proactivity_threshold=_parse_int(raw, "proactivity_threshold", -1),
        cooldown_seconds=_parse_int(raw, "cooldown_seconds", -1),
        context_size=_parse_int(raw, "context_size", -1),
    )


# ---------------------------------------------------------------- SSRF 校验


def is_private_or_reserved_host(host: str) -> bool:
    """判断 host 是否为环回/私有/保留地址。

    IP 字面量直接判定；域名先尝试 DNS 解析，任一解析结果落在本地/保留
    网段即视为私有。注意本函数做同步 socket 调用，仅在启动阶段使用。
    """
    host = host.strip("[]").lower()
    if not host:
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return _addr_is_local(addr)

    import socket

    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        return True
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return True  # 解析不了的域名按不可达处理，直接拒绝
    for info in infos:
        try:
            addr = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if _addr_is_local(addr):
            return True
    return False


def _addr_is_local(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        addr.is_loopback
        or addr.is_private
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def validate_request_url(url: str) -> None:
    """对将要发起服务端 HTTP(S) 请求的 URL 做安全校验。

    仅允许 http/https，且 host 必须是公网地址（拒绝 localhost、环回、
    私有与保留网段）。用于下载图片等媒体资源。
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"仅允许 http/https URL，实际 scheme 是 {parsed.scheme!r}")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"URL 缺少 host：{url!r}")
    if is_private_or_reserved_host(hostname):
        raise ValueError(f"拒绝访问本地/私有/保留地址：{hostname}")


def validate_endpoint_url(url: str, *, allow_private: bool) -> None:
    """校验 SnowLuma OneBot endpoint。

    与 validate_request_url 相同的规则，但 allow_private=True 时放行
    内网地址（本工具默认连接本机/局域网内的 SnowLuma 实例）。
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"snowluma.endpoint 仅允许 http/https，实际 scheme 是 {parsed.scheme!r}"
        )
    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"snowluma.endpoint 缺少 host：{url!r}")
    if allow_private:
        return
    if is_private_or_reserved_host(hostname):
        raise ValueError(
            f"snowluma.endpoint 指向本地/私有地址 {hostname}；"
            "如确需连接内网实例，请把 snowluma.allow_private_endpoint 设为 true"
        )
