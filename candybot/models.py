"""领域模型与配置校验。

这里定义运行时消息记录（ChatRecord）、归一化结果（NormalizedMessage），
以及从 config.ConfigClass 解析出来的强类型配置（Settings 系列）。
endpoint 的 SSRF 校验也在本模块。
"""

from __future__ import annotations

import ipaddress
import math
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

MULTIMODAL_MODES = ("direct", "describe", "placeholder")

# 单张图片在后续对话中的展示形态：
# show        继续把原图 base64 作为内容块传给回复模型；
# summarized  只传总结文字「[图片：…]」；
# placeholder 只展示 [图片] 占位符。
IMAGE_STATE_SHOW = "show"
IMAGE_STATE_SUMMARIZED = "summarized"
IMAGE_STATE_PLACEHOLDER = "placeholder"
IMAGE_STATES = (IMAGE_STATE_SHOW, IMAGE_STATE_SUMMARIZED, IMAGE_STATE_PLACEHOLDER)


def _aligned_image_states(count: int, states: Any) -> tuple[str, ...]:
    """把任意外部输入整理成长度恰为 count 的合法状态表，缺省补 show。"""
    if not isinstance(states, (list, tuple)):
        states = ()
    aligned = [s if s in IMAGE_STATES else IMAGE_STATE_SHOW for s in states]
    aligned += [IMAGE_STATE_SHOW] * (count - len(aligned))
    return tuple(aligned[:count])


# ---------------------------------------------------------------- 运行时模型


@dataclass(slots=True)
class ChatRecord:
    """一条群聊消息（含机器人自己发出的）。

    images 是 base64 data URL 元组，无论 multimodal 模式如何都会入库
    （撤回时随记录一起消亡；超过保留期的原图由图片回收清空数据）；
    image_states 与 images 对齐，描述每张图在后续对话中的展示形态；
    image_summaries 保存视觉模型给的总结，供展示与降级时复用。
    image_states 缺省视为全部 show。
    槽位语义：images 中的空串表示该图原图已按保留期回收，此时状态必为
    summarized/placeholder（加载层保证不会以 show 出现），总结仍保留。
    """

    message_id: int
    group_id: int
    user_id: int
    nickname: str
    text: str
    ts: float
    is_self: bool = False
    images: tuple[str, ...] = field(default=(), repr=False)
    image_states: tuple[str, ...] = field(default=(), repr=False)
    image_summaries: dict[int, str] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # 防御性对齐：外部构造长度不一致时以 images 为准
        self.image_states = _aligned_image_states(len(self.images), self.image_states)
        clean: dict[int, str] = {}
        for key, value in (self.image_summaries or {}).items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, str):
                clean[idx] = value
        self.image_summaries = clean

    # ------------------------------------------------------------ 图片形态

    def state_of(self, index: int) -> str:
        if 0 <= index < len(self.image_states):
            return self.image_states[index]
        return IMAGE_STATE_SHOW

    def summary_of(self, index: int) -> str | None:
        summary = (self.image_summaries or {}).get(index)
        return summary or None

    def set_image_state(self, index: int, state: str, *, summary: str | None = None) -> None:
        """更新第 index 张图的展示形态；给 summary 时同步保存总结文本。"""
        if state not in IMAGE_STATES:
            raise ValueError(f"非法的图片状态：{state!r}")
        states = list(
            self.image_states
            if len(self.image_states) == len(self.images)
            else _aligned_image_states(len(self.images), self.image_states)
        )
        states[index] = state
        self.image_states = tuple(states)
        if summary:
            sums = dict(self.image_summaries or {})
            sums[index] = summary
            self.image_summaries = sums


@dataclass(slots=True)
class NormalizedMessage:
    """normalize 后的群消息事件。"""

    record: ChatRecord
    mentioned_me: bool


@dataclass(slots=True)
class Decision:
    """一次发言决策的结果。

    forced = @必答；engaged = 判定消息在延续与机器人的对话——两者都不受
    冷却与护栏限制，区别在于 engaged 仍消耗每日主动发言配额、且不刷新冷却。
    """

    should_reply: bool
    forced: bool = False
    engaged: bool = False
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
    # 结构性反插嘴护栏：-1/缺省 = 内置默认，0 = 关闭
    min_gap_messages: int = -1
    busy_rate_per_min: int = -1


# 护栏内置默认：发言后至少隔这么多条他人消息才再评估；近一分钟消息达到
# 该条数即整体静默（只答 @）。0 均可显式关闭。
MIN_GAP_MESSAGES_DEFAULT = 3
BUSY_RATE_PER_MIN_DEFAULT = 6
# 历史条数内置默认：groups_default 未写 context_size（哨兵 -1）时兜底。
# 不允许 ≤0 生效——memory.tail 对非正数返回空列表，等于静默失忆。
CONTEXT_SIZE_DEFAULT = 20


def _apply_builtin_defaults(profile: GroupProfile) -> GroupProfile:
    """把 groups_default 里缺省(-1)的字段替换为内置默认值。"""
    if (
        profile.min_gap_messages >= 0
        and profile.busy_rate_per_min >= 0
        and profile.context_size > 0
    ):
        return profile
    return GroupProfile(
        group_id=profile.group_id,
        enabled=profile.enabled,
        persona=profile.persona,
        proactivity_threshold=profile.proactivity_threshold,
        cooldown_seconds=profile.cooldown_seconds,
        context_size=(
            profile.context_size
            if profile.context_size > 0
            else CONTEXT_SIZE_DEFAULT
        ),
        min_gap_messages=(
            profile.min_gap_messages
            if profile.min_gap_messages >= 0
            else MIN_GAP_MESSAGES_DEFAULT
        ),
        busy_rate_per_min=(
            profile.busy_rate_per_min
            if profile.busy_rate_per_min >= 0
            else BUSY_RATE_PER_MIN_DEFAULT
        ),
    )


@dataclass(frozen=True)
class AISettings:
    """全局默认提供商：models 里未写 base_url / api_key 的模型继承这里。"""

    base_url: str
    api_key: str


@dataclass(frozen=True)
class ModelConfig:
    """单个模型角色的生效配置（base_url / api_key 已按继承规则解析完毕）。

    context_window 为该模型的上下文窗口（token），运行时据此约束送入的
    历史长度；max_output_tokens 为该模型单次调用的输出上限（token），
    未配置时各角色回落到自己的内置/全局默认值。
    tool_use=False 表示该模型不支持（或不希望使用）工具调用：该角色改走
    纯文本协议（judge/评估在正文输出 JSON，reply 用 <drop_img>/<recall_img>
    标记），请求不携带 tools 参数，提示词里的输出契约随之一致——保证
    提示词永远只约定模型能力范围内的回答方式。运行中若端点报工具相关
    错误或忽略 tools 参数，该角色也会自动降级为纯文本协议。
    forced_tool_choice=False 表示该模型不支持强制指定工具（tool_choice 的
    required/object 形式，思考（thinking）模式的模型普遍如此，如 qwen3
    系列会直接报 400）：请求仍携带 tools 但改用 tool_choice="auto"，由
    提示词引导模型主动调用；模型没调用时同样自动降级为纯文本协议。
    """

    model: str
    base_url: str
    api_key: str  # 可为空：本地服务等无密钥端点（运行时回退环境变量/占位符）
    context_window: int | None
    max_output_tokens: int | None
    tool_use: bool = True
    forced_tool_choice: bool = True


@dataclass(frozen=True)
class ModelSettings:
    judge: ModelConfig
    reply: ModelConfig
    vision: ModelConfig | None


@dataclass(frozen=True)
class GenerationSettings:
    reply_max_tokens: int
    temperature: float
    max_context_chars: int
    timeout_seconds: float
    emoji_chance: float = 0.25   # 每条回复允许保留 emoji 的概率，0~1
    emoji_max: int = 2           # 允许保留时的最大 emoji 个数
    # 门槛复核（复评）：首评分数严格高于 recheck_min_score 又低于本群门槛时，
    # 把真实门槛告知 judge 再裁一次；enabled=False 时直接采信首评。
    recheck_enabled: bool = True
    recheck_min_score: int = 5
    # direct 模式下整个历史层最多同时传入的原图张数，超出从最旧的开始摘除
    max_history_images: int = 8


@dataclass(frozen=True)
class MultimodalSettings:
    mode: str
    download_media: bool


@dataclass(frozen=True)
class StorageSettings:
    """聊天存储（candy.db）策略。

    image_retention_days：聊天原图的保留天数；超过后原图数据被回收，
    记录降级为总结/占位符但文本与总结永久保留。
    """

    image_retention_days: int


@dataclass(frozen=True)
class RateLimitSettings:
    global_daily_limit: int | None


# 敷衍池内置默认：回复过长或拆条后为空时随机抽取其一代替发送。
LAZY_REPLIES_DEFAULT = ("呃呃", "不晓得", "懒得说", "不知道", "emm")

# 多音字（如「银行」的行）的错字策略：word_reading 取词典词内读音照常替换；
# skip 则多音字整体跳过、只替换单一读音的字。两种策略下读音无法确定的
# 多音字（单独成词等）都绝不替换，避免产出读音对不上的「假同音」错字。
TYPO_POLYPHONE_MODES = ("word_reading", "skip")
TYPO_POLYPHONE_MODE_DEFAULT = "word_reading"


@dataclass(frozen=True)
class ResponsePostProcessSettings:
    """输出层拟人化后处理配置（拆条 / 打字延迟 / 错别字 / 敷衍兜底）。

    enabled=False 时发送链路与未引入后处理前完全一致（整条单发、零延迟、
    不触发连发被打断后的重想）。打字延迟 = 逐字估时 × typing_speed，
    单条封顶 60 秒；错别字三率的语义：
    typo_error_rate 为单字被同音替换的概率，typo_tone_error_rate 为替换时
    选用错误声调拼音的概率，typo_word_replace_rate 为整词替换的概率；
    产生错字后以 typo_correction_probability 的概率追加一条「＊正确词」更正。
    keep_strong_punctuation 控制拆条时句末的 ! ?（含全半角）是否保留，
    其余句末标点总去掉；typo_polyphone_mode 控制多音字错字策略
    （见 TYPO_POLYPHONE_MODES 注释）；max_length 按显示字数计，
    emoji/颜文字序列整体算 1 个字。
    更正经 OneBot v11 reply 消息段引用最后一条正文发送（见 bot.py 与
    snowluma.py）。
    """

    enabled: bool = True
    typing_speed: float = 1.0
    max_split: int = 3
    max_length: int = 120
    keep_strong_punctuation: bool = True
    typo_error_rate: float = 0.05
    typo_tone_error_rate: float = 0.3
    typo_word_replace_rate: float = 0.2
    typo_correction_probability: float = 0.5
    typo_polyphone_mode: str = TYPO_POLYPHONE_MODE_DEFAULT
    lazy_replies: tuple[str, ...] = LAZY_REPLIES_DEFAULT


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
    storage: StorageSettings
    rate_limit: RateLimitSettings
    snowluma: SnowlumaSettings
    response_post_process: ResponsePostProcessSettings

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
            min_gap_messages=(
                profile.min_gap_messages
                if profile.min_gap_messages >= 0
                else base.min_gap_messages
            ),
            busy_rate_per_min=(
                profile.busy_rate_per_min
                if profile.busy_rate_per_min >= 0
                else base.busy_rate_per_min
            ),
        )


# ---------------------------------------------------------------- 配置解析


def _require_section(cfg: Any, name: str) -> dict[str, Any]:
    """按段名取配置段。兼容真实 ConfigClass 的属性访问与测试用映射访问。"""
    try:
        section = getattr(cfg, name)
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"config.json5 缺少必需的配置段 `{name}`") from exc
    if not isinstance(section, dict):
        raise ValueError(f"config.json5 中 `{name}` 应为对象")
    return section


def _optional_section(cfg: Any, name: str) -> dict[str, Any]:
    """同 _require_section，但段整体缺省时返回空对象（条目级校验另行兜底）。"""
    try:
        section = getattr(cfg, name)
    except (AttributeError, KeyError):
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"config.json5 中 `{name}` 应为对象")
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


def _parse_float(section: dict[str, Any], key: str, default: float) -> float:
    value = _get(section, key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"配置项 `{key}` 应为数字，实际是 {value!r}") from exc


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
        raise ValueError("config.json5 → bot.self_qq 必须配置为机器人的 QQ 号")
    listen_host = _parse_str(bot_cfg, "listen_host", "127.0.0.1")
    listen_port = _parse_int(bot_cfg, "listen_port", 5700)
    log_level = _parse_str(bot_cfg, "log_level", "INFO").upper()
    if log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError(
            "config.json5 → bot.log_level 只能是 DEBUG/INFO/WARNING/ERROR/CRITICAL，"
            f"实际是 {log_level!r}"
        )

    # 白名单：groups 里的每个 key 都是一个群号
    groups_raw = _require_section(cfg, "groups")
    groups: dict[int, GroupProfile] = {}
    for key, raw in groups_raw.items():
        gid = _coerce_group_id(key)
        groups[gid] = _parse_group_profile(raw, f"groups.{key}", gid)

    default_profile = _apply_builtin_defaults(
        _parse_group_profile(
            _require_section(cfg, "groups_default"), "groups_default", None
        )
    )
    if not default_profile.persona:
        raise ValueError("groups_default.persona 不能为空")

    ai_cfg = _optional_section(cfg, "ai_backend")
    ai_settings = AISettings(
        base_url=_parse_str(ai_cfg, "base_url", ""),
        api_key=_parse_str(ai_cfg, "api_key", ""),
    )

    models_cfg = _require_section(cfg, "models")
    model_settings = ModelSettings(
        judge=_parse_model_config(models_cfg.get("judge"), "models.judge", ai_settings),
        reply=_parse_model_config(models_cfg.get("reply"), "models.reply", ai_settings),
        vision=(
            _parse_model_config(models_cfg["vision"], "models.vision", ai_settings)
            if models_cfg.get("vision") is not None
            else None
        ),
    )

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
    recheck_min_score = _parse_int(gen_cfg, "recheck_min_score", 5)
    if not 0 <= recheck_min_score <= 10:
        raise ValueError(
            f"配置项 `generation.recheck_min_score` 应在 0~10 之间，"
            f"实际是 {recheck_min_score!r}"
        )
    max_history_images = _parse_int(gen_cfg, "max_history_images", 8)
    if max_history_images < 0:
        raise ValueError(
            f"配置项 `generation.max_history_images` 不能为负数，实际是 {max_history_images!r}"
        )
    generation_settings = GenerationSettings(
        reply_max_tokens=_parse_int(gen_cfg, "reply_max_tokens", 500),
        temperature=float(_get(gen_cfg, "temperature", 0.8)),
        max_context_chars=_parse_int(gen_cfg, "max_context_chars", 8000),
        timeout_seconds=float(_get(gen_cfg, "timeout_seconds", 60)),
        emoji_chance=emoji_chance,
        emoji_max=emoji_max,
        recheck_enabled=_parse_bool(
            gen_cfg.get("recheck_enabled", True), "recheck_enabled"
        ),
        recheck_min_score=recheck_min_score,
        max_history_images=max_history_images,
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

    # storage 段可整体省略（全部走默认值）
    storage_cfg = _optional_section(cfg, "storage")
    image_retention_days = _parse_int(storage_cfg, "image_retention_days", 7)
    if image_retention_days < 1:
        raise ValueError(
            f"配置项 `storage.image_retention_days` 不能小于 1，实际是 "
            f"{image_retention_days!r}"
        )
    storage_settings = StorageSettings(image_retention_days=image_retention_days)

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

    # response_post_process 段可整体省略（全部走默认值）
    pp_cfg = _optional_section(cfg, "response_post_process")

    def _parse_probability(key: str, default: float) -> float:
        value = _parse_float(pp_cfg, key, default)
        if not 0 <= value <= 1:
            raise ValueError(
                f"配置项 `response_post_process.{key}` 应在 0~1 之间，实际是 {value!r}"
            )
        return value

    typing_speed = _parse_float(pp_cfg, "typing_speed", 1.0)
    # json5 支持 Infinity / NaN 字面量且 float() 一律照收：inf 会让连发的
    # asyncio.sleep 永久挂起该群队列，nan 会静默关闭延迟，都必须拒收。
    if not math.isfinite(typing_speed) or typing_speed < 0:
        raise ValueError(
            "配置项 `response_post_process.typing_speed` 应为非负有限数字，"
            f"实际是 {typing_speed!r}"
        )
    polyphone_mode = _parse_str(pp_cfg, "typo_polyphone_mode", TYPO_POLYPHONE_MODE_DEFAULT)
    if polyphone_mode not in TYPO_POLYPHONE_MODES:
        raise ValueError(
            "配置项 `response_post_process.typo_polyphone_mode` 应为 "
            f"{' / '.join(map(repr, TYPO_POLYPHONE_MODES))} 之一，"
            f"实际是 {polyphone_mode!r}"
        )
    max_split = _parse_int(pp_cfg, "max_split", 3)
    if max_split < 1:
        raise ValueError(
            f"配置项 `response_post_process.max_split` 不能小于 1，实际是 {max_split!r}"
        )
    max_length = _parse_int(pp_cfg, "max_length", 120)
    if max_length < 1:
        raise ValueError(
            f"配置项 `response_post_process.max_length` 不能小于 1，实际是 {max_length!r}"
        )
    lazy_raw = _get(pp_cfg, "lazy_replies", list(LAZY_REPLIES_DEFAULT))
    if (
        not isinstance(lazy_raw, (list, tuple))
        or not lazy_raw
        or not all(isinstance(item, str) and item.strip() for item in lazy_raw)
    ):
        raise ValueError(
            "配置项 `response_post_process.lazy_replies` 应为非空的字符串列表"
        )
    post_process_settings = ResponsePostProcessSettings(
        enabled=_parse_bool(pp_cfg.get("enabled", True), "response_post_process.enabled"),
        typing_speed=typing_speed,
        max_split=max_split,
        max_length=max_length,
        keep_strong_punctuation=_parse_bool(
            pp_cfg.get("keep_strong_punctuation", True),
            "response_post_process.keep_strong_punctuation",
        ),
        typo_error_rate=_parse_probability("typo_error_rate", 0.05),
        typo_tone_error_rate=_parse_probability("typo_tone_error_rate", 0.3),
        typo_word_replace_rate=_parse_probability("typo_word_replace_rate", 0.2),
        typo_correction_probability=_parse_probability(
            "typo_correction_probability", 0.5
        ),
        typo_polyphone_mode=polyphone_mode,
        lazy_replies=tuple(str(item) for item in lazy_raw),
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
        storage=storage_settings,
        rate_limit=rate_limit_settings,
        snowluma=snowluma_settings,
        response_post_process=post_process_settings,
    )


def _coerce_group_id(key: str) -> int:
    try:
        return int(key)
    except ValueError as exc:
        raise ValueError(f"groups 中的群号必须是整数字符串，实际是 {key!r}") from exc


def _parse_model_config(raw: Any, label: str, defaults: AISettings) -> ModelConfig:
    """解析 models 里的单个角色条目，缺省项继承全局默认提供商。

    raw 允许两种写法：模型名字符串（等价于只写 model 字段），或对象
    （可覆盖 base_url / api_key 并配置 context_window / max_output_tokens）。
    """
    if raw is None:
        raise ValueError(f"config.json5 → `{label}` 必须指定模型名")
    if isinstance(raw, str):
        raw = {"model": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"`{label}` 应为模型名字符串或配置对象")
    model = _parse_str(raw, "model", "")
    if not model:
        raise ValueError(f"config.json5 → `{label}.model` 必须指定模型名")
    context_window = _parse_optional_int(raw, "context_window")
    if context_window is not None and context_window <= 0:
        raise ValueError(
            f"配置项 `{label}.context_window` 必须为正整数，实际是 {context_window!r}"
        )
    max_output = _parse_optional_int(raw, "max_output_tokens")
    if max_output is not None and max_output <= 0:
        raise ValueError(
            f"配置项 `{label}.max_output_tokens` 必须为正整数，实际是 {max_output!r}"
        )
    if context_window is not None and max_output is not None and max_output >= context_window:
        raise ValueError(
            f"配置项 `{label}.max_output_tokens`（{max_output}）必须小于 "
            f"`{label}.context_window`（{context_window}），否则模型装不下任何输入"
        )
    base_url = _parse_str(raw, "base_url", "") or defaults.base_url
    api_key = _parse_str(raw, "api_key", "") or defaults.api_key
    if not base_url:
        raise ValueError(
            f"`{label}` 未配置 base_url，且 ai_backend.base_url 为空："
            "两者至少要有一处给出 OpenAI 兼容 API 地址"
        )
    return ModelConfig(
        model=model,
        base_url=base_url,
        api_key=api_key,
        context_window=context_window,
        max_output_tokens=max_output,
        tool_use=_parse_bool(raw.get("tool_use", True), f"{label}.tool_use"),
        forced_tool_choice=_parse_bool(
            raw.get("forced_tool_choice", True), f"{label}.forced_tool_choice"
        ),
    )


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
        min_gap_messages=_parse_int(raw, "min_gap_messages", -1),
        busy_rate_per_min=_parse_int(raw, "busy_rate_per_min", -1),
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
