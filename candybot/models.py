"""领域模型与配置校验。

这里定义运行时消息记录（ChatRecord）、归一化结果（NormalizedMessage），
以及从 config.ConfigClass 解析出来的强类型配置（Settings 系列）。
endpoint 的 SSRF 校验也在本模块。
"""

from __future__ import annotations

import ipaddress
import math
import re
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
    is_command 标记命令插件产生的消息（用户发送的被判定为命令的消息、
    机器人以自语身份发出的命令回复），照常入库（审计与印象统计可用），
    plugins.include_commands_in_history=false 时据此把两者过滤出模型的
    历史上下文。
    """

    message_id: int
    group_id: int
    user_id: int
    nickname: str
    text: str
    ts: float
    is_self: bool = False
    is_command: bool = False
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
    """normalize 后的群消息事件。

    sticker_flags 与 record.images 下标对齐，标记每张图是否像「表情包类」
    （识别来源按 multimodal 模式而异，见 normalize.py 与 stickers.py）；
    只在收图当次事件处理里使用，不入库。
    """

    record: ChatRecord
    mentioned_me: bool
    sticker_flags: tuple[bool, ...] = ()


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
    # 机器人自己的昵称：自发言写回记忆的昵称、@/回复占位文本
    # （normalize）。此前固定在代码里，默认值保持原样「糖糖」。
    self_nickname: str = "糖糖"
    # 事件上报请求体上限（字节）。direct 模式下事件正文可能带较大的图片
    # 数据，超限时可按需调大。烘进 aiohttp 服务的构造参数，改动需重启。
    max_event_body_bytes: int = 1024 * 1024


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
    # learning 为可选的第四角色：群印象总结、表达/黑话学习等后台学习任务
    # 用它；未配置时继承 judge（便宜快速）的配置（见 AIClient）。
    learning: ModelConfig | None = None


# 临时随机回复风格的内置默认示例（generation.multiple_reply_style）：
# 打破固定腔调，每条回复按概率抽一条注入 L4，风格贴合群聊场景。
MULTIPLE_REPLY_STYLE_DEFAULT = (
    "用 1-2 个字进行回复",
    "只回一个语气词，比如「嗯」「哦」「哈」",
    "只说半句话，像打了一半就顺手发了出去",
    "用反问或吐槽接一句，别一本正经地回答",
    "像赶时间一样，能省的字全省掉",
)

# 「太像 AI」检测的内置默认正则规则（generation.ai_flavor_rules）：
# 按 re.MULTILINE 编译，^ 类规则因此能命中任意一行行首。
AI_FLAVOR_RULES_DEFAULT = (
    r"作为(一)?(个|位)?(AI|ai|人工智能|语言模型|大模型)",
    r"很(高兴|乐意|荣幸)(为|帮)您",
    r"^\s*以下是",
    r"^\s*#{1,6}\s",          # markdown 标题残留
    r"\*\*[^*\n]+\*\*",       # markdown 加粗残留
    r"^\s*[-*]\s+\S",         # 行首 markdown 列表残留
)


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
    # ---- 决策层三项增强（各自关闭时行为与引入前完全一致）----
    # freshness_check_enabled：发送前新鲜度检查——回复生成期间有明确指向
    #   bot 的新消息（@我/回复我）时，并入最新上下文重生成一次（至多一次）。
    freshness_check_enabled: bool = True
    # observe_band / observe_delay_seconds：观望——终评分落在
    #   [门槛 - observe_band, 门槛) 且未被护栏直接终止的消息，延迟
    #   observe_delay_seconds 秒后取届时最新上下文重判一次（每条消息至多
    #   一次，护栏与配额路径不变）；observe_band=0 关闭。
    observe_band: int = 2
    observe_delay_seconds: float = 45.0
    # repetition_guard_enabled：重复抑制——目标消息之后已有自己的发言且
    #   对方没再开口时（判定规则见 bot._already_replied_to），在 L4 注入
    #   「不要和之前的发言重复」的提醒。
    repetition_guard_enabled: bool = True
    # ---- 风格多样性与内容拦截（关闭时行为与引入前完全一致）----
    # 任务 A 临时随机风格：每条回复独立按 multiple_probability 掷点，命中时
    #   从 multiple_reply_style 随机抽一条注入 L4；概率 0 或列表为空即关闭。
    multiple_reply_style: tuple[str, ...] = MULTIPLE_REPLY_STYLE_DEFAULT
    multiple_probability: float = 0.0
    # 任务 B AI 味拦截：回复清洗完成后过一轮 ai_flavor_rules 正则检测，
    #   命中则把被拦截回复与原因附进 L4 重生成一次；ai_flavor_retries 为
    #   最大重试次数（0 关闭整个环节），重试后仍命中则放行并记 warning。
    ai_flavor_rules: tuple[str, ...] = AI_FLAVOR_RULES_DEFAULT
    ai_flavor_retries: int = 1
    # ---- 原写死在 ai.py / bot.py 里的调用与重试参数（默认值=原字面量）----
    # 各角色采样温度：reply 用上面的 temperature；judge/vision/learning
    # 此前固定在 ai.py 的各调用处。
    judge_temperature: float = 0.2
    describe_temperature: float = 0.3    # describe 模式图片转述
    assess_temperature: float = 0.2      # direct 模式入库评估
    learning_temperature: float = 0.3    # 后台学习类一次性调用
    # 生成回复的网络重试（bot._generate_with_retry）：总尝试次数与首次退避
    # 秒数（其后每次 ×2）。
    generate_max_attempts: int = 2
    generate_retry_base_delay: float = 2.0
    # 连发被打断后每轮最多几次「重想」（bot._decide_and_reply 的 reconsider
    # 预算），预算用尽后剩余腹稿按原计划发完。
    max_reconsider_per_burst: int = 2


@dataclass(frozen=True)
class MultimodalSettings:
    mode: str
    download_media: bool
    # 原写死在 normalize.py 的下载参数（默认值=原字面量）：
    download_timeout_seconds: float = 15.0   # 单张图片下载超时
    max_image_bytes: int = 8 * 1024 * 1024   # 超过该字节数放弃下载
    max_images_per_message: int = 4          # 单条消息至多下载几张图


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


@dataclass(frozen=True)
class LearningSettings:
    """记忆与学习机制（每日群印象、表达学习、黑话学习）的配置。

    总开关 enabled 关掉全部后台学习任务；impression_enabled /
    expression_enabled / jargon_enabled 分别控制单项。学习任务在后台
    asyncio 任务中执行，失败只记 warning 日志并跳过本次，不影响主链路。

    impression_days：注入 L2 的最近印象天数（天内字节级稳定）；
    impression_max_chars：单日印象的字数上限（提示词与入库裁剪同用它）；
    expression_batch_size：同群被热缓存淘汰的消息攒够这么多条触发一次学习；
    expression_max_inject：单次回复最多注入的表达条数（L4，加权随机）；
    expression_self_review：是否让 AI 自审过滤低质量表达条目；
    jargon_max_entries：每群黑话条目上限，超限淘汰最久未命中的；
    jargon_max_inject：单次回复最多注入的命中黑话条数（L4，机械匹配）。
    """

    enabled: bool = True
    impression_enabled: bool = True
    impression_days: int = 3
    impression_max_chars: int = 300
    expression_enabled: bool = True
    expression_batch_size: int = 10
    expression_max_inject: int = 3
    expression_self_review: bool = True
    jargon_enabled: bool = True
    jargon_max_entries: int = 50
    jargon_max_inject: int = 5
    # ---- 原写死在 learning.py 的调度与限流参数（默认值=原字面量）----
    # 被淘汰消息缓冲的容量倍数（批大小 × 该系数），溢出丢最旧的；
    pending_buffer_factor: int = 3
    # 每日印象总结送入的聊天文本字符预算（从当天最新一条向前截取）；
    impression_text_budget: int = 6000
    # 每批黑话学习实际做双路推断的候选数上限（一个候选至少 3 次 LLM 调用）；
    jargon_candidates_per_batch: int = 5
    # 黑话含义入库长度上限（防模型长篇大论撑爆 L4）。
    jargon_meaning_max_chars: int = 200


# 敷衍池内置默认：回复过长或拆条后为空时随机抽取其一代替发送。
LAZY_REPLIES_DEFAULT = ("呃呃", "不晓得", "懒得说", "不知道", "emm")

# describe 模式表情包识别的内置默认关键词（stickers.summary_keywords）：
# 视觉模型总结里出现任一词（忽略大小写）即视为表情包类。
STICKER_SUMMARY_KEYWORDS_DEFAULT = ("表情包", "梗图", "斗图", "动图", "meme")

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
    # ---- 打字延迟的单字耗时模型（原写死在 postprocess.py，默认值=原字面量）----
    # 估算秒数 =（各类耗时×数量）× typing_speed 倍率，单条另有封顶。
    typing_cjk_seconds: float = 0.3        # 中文/全角字符，每字
    typing_latin_seconds: float = 0.15     # 英文数字等半角字符，每字
    typing_special_seconds: float = 1.0    # emoji 序列/颜文字，每块
    typing_single_multiplier: float = 3.0  # 无任何特殊块的单字回复加倍
    max_typing_delay_seconds: float = 60.0  # 单条打字延迟封顶秒数


@dataclass(frozen=True)
class StickerSettings:
    """表情包最小版（收集 + 小概率跟发，见 stickers.py）。

    enabled：总开关（关掉既不收集也不跟发）；
    send_probability：成功发送一条文字回复后跟发一张表情包的概率；
    max_count：全局收藏上限（跨群合计），超限替换最久未使用的条目
      （删除记录并删除图片文件）。
    """

    enabled: bool = True
    send_probability: float = 0.05
    max_count: int = 64
    # 识别启发式参数（原写死在 stickers.py，默认值=原字面量）：
    # placeholder 模式「尺寸小」启发式的边长上限（像素），较长边不超过才收集；
    max_side_px: int = 512
    # describe 模式总结文本的命中关键词（忽略大小写，任一命中即视为表情包类）。
    summary_keywords: tuple[str, ...] = STICKER_SUMMARY_KEYWORDS_DEFAULT


@dataclass(frozen=True)
class PluginSettings:
    """命令插件系统（见 plugin_api.py / commandline.py）。

    enabled：总开关。关掉后 / 开头的消息不再被拦截，与大模型自主回复
      引入命令功能之前完全一致（现取现读，热重载即时生效）；
    dir：插件目录（相对工作目录），启动时逐个导入其中的 .py 文件；
    timeout_seconds：异步 handler 的执行超时，超时按失败回复。
    include_commands_in_history：插件产生的消息（用户发送的被判定为命令
      的消息、机器人以自语身份发出的命令回复）是否送入模型的历史上下文。
      true（默认）与引入该配置前一致；false 时两者照常入库并打上
      ChatRecord.is_command 标记（审计、每日印象统计仍然可见），只是
      judge/reply/重想的上下文组装时过滤掉、不再占用 context_size 名额。
      现取现读，改完对之后的每次模型请求生效；存量记录的标记由
      migrations.py 的手动迁移按当前注册表回填（见该模块 docstring）。
    注册表在构建期装载：新增/修改插件文件需重启机器人生效。
    """

    enabled: bool = True
    dir: str = "plugins"
    timeout_seconds: float = 30.0
    include_commands_in_history: bool = True


@dataclass(frozen=True)
class SnowlumaSettings:
    """SnowLuma OneBot HTTP API 连接参数（见 snowluma.py）。"""

    endpoint: str
    api_key: str
    timeout_ms: int
    allow_private_endpoint: bool
    # ---- 发送重试（bot._send_with_retry，原写死；默认值=原字面量）----
    # 总尝试次数与首次退避秒数（其后每次 ×2）；现取现读，热重载即时生效。
    send_max_attempts: int = 3
    send_retry_delay_seconds: float = 1.5


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
    # learning 段可整体省略（全部走默认值），故带默认值
    learning: LearningSettings = LearningSettings()
    # stickers 段同样可整体省略
    stickers: StickerSettings = StickerSettings()
    # plugins 段同样可整体省略（缺省即启用命令插件，目录为 plugins/）
    plugins: PluginSettings = PluginSettings()

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
    self_nickname = _parse_str(bot_cfg, "self_nickname", "糖糖").strip()
    if not self_nickname:
        raise ValueError("config.json5 → bot.self_nickname 不能为空")
    max_event_body_bytes = _parse_int(bot_cfg, "max_event_body_bytes", 1024 * 1024)
    if max_event_body_bytes < 1024:
        raise ValueError(
            f"配置项 `bot.max_event_body_bytes` 不能小于 1024，"
            f"实际是 {max_event_body_bytes!r}"
        )
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
        learning=(
            _parse_model_config(
                models_cfg["learning"], "models.learning", ai_settings
            )
            if models_cfg.get("learning") is not None
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
    observe_band = _parse_int(gen_cfg, "observe_band", 2)
    if not 0 <= observe_band <= 10:
        raise ValueError(
            f"配置项 `generation.observe_band` 应在 0~10 之间（0 关闭观望），"
            f"实际是 {observe_band!r}"
        )
    multiple_probability = _parse_float(gen_cfg, "multiple_probability", 0.0)
    if not 0 <= multiple_probability <= 1:
        raise ValueError(
            f"配置项 `generation.multiple_probability` 应在 0~1 之间，"
            f"实际是 {multiple_probability!r}"
        )

    def _parse_str_tuple(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
        raw = _get(gen_cfg, key, list(default))
        if (
            not isinstance(raw, (list, tuple))
            or not all(isinstance(item, str) and item.strip() for item in raw)
        ):
            raise ValueError(f"配置项 `generation.{key}` 应为非空字符串的列表")
        return tuple(str(item).strip() for item in raw)

    multiple_reply_style = _parse_str_tuple(
        "multiple_reply_style", MULTIPLE_REPLY_STYLE_DEFAULT
    )
    ai_flavor_rules = _parse_str_tuple("ai_flavor_rules", AI_FLAVOR_RULES_DEFAULT)
    for pattern in ai_flavor_rules:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(
                f"配置项 `generation.ai_flavor_rules` 含非法正则 {pattern!r}：{exc}"
            ) from exc
    ai_flavor_retries = _parse_int(gen_cfg, "ai_flavor_retries", 1)
    if ai_flavor_retries < 0:
        raise ValueError(
            f"配置项 `generation.ai_flavor_retries` 不能为负数（0 表示关闭），"
            f"实际是 {ai_flavor_retries!r}"
        )
    observe_delay_seconds = _parse_float(gen_cfg, "observe_delay_seconds", 45.0)
    # 观望延时直接喂给 asyncio.sleep：inf/nan 会让任务永久挂起或行为未定，
    # 与 typing_speed 同口径拒收
    if not math.isfinite(observe_delay_seconds) or observe_delay_seconds < 0:
        raise ValueError(
            "配置项 `generation.observe_delay_seconds` 应为非负有限数字，"
            f"实际是 {observe_delay_seconds!r}"
        )

    def _parse_temperature(key: str, default: float) -> float:
        value = _parse_float(gen_cfg, key, default)
        if not 0 <= value <= 2:
            raise ValueError(
                f"配置项 `generation.{key}` 应在 0~2 之间，实际是 {value!r}"
            )
        return value

    def _parse_backoff(key: str, default: float) -> float:
        # 退避延时直接喂给 asyncio.sleep：与 observe_delay_seconds 同口径拒收
        value = _parse_float(gen_cfg, key, default)
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"配置项 `generation.{key}` 应为非负有限数字，实际是 {value!r}"
            )
        return value

    judge_temperature = _parse_temperature("judge_temperature", 0.2)
    describe_temperature = _parse_temperature("describe_temperature", 0.3)
    assess_temperature = _parse_temperature("assess_temperature", 0.2)
    learning_temperature = _parse_temperature("learning_temperature", 0.3)
    generate_max_attempts = _parse_int(gen_cfg, "generate_max_attempts", 2)
    if generate_max_attempts < 1:
        raise ValueError(
            f"配置项 `generation.generate_max_attempts` 不能小于 1，"
            f"实际是 {generate_max_attempts!r}"
        )
    generate_retry_base_delay = _parse_backoff("generate_retry_base_delay", 2.0)
    max_reconsider_per_burst = _parse_int(gen_cfg, "max_reconsider_per_burst", 2)
    if max_reconsider_per_burst < 0:
        raise ValueError(
            f"配置项 `generation.max_reconsider_per_burst` 不能为负数（0 关闭重想），"
            f"实际是 {max_reconsider_per_burst!r}"
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
        freshness_check_enabled=_parse_bool(
            gen_cfg.get("freshness_check_enabled", True),
            "generation.freshness_check_enabled",
        ),
        observe_band=observe_band,
        observe_delay_seconds=observe_delay_seconds,
        repetition_guard_enabled=_parse_bool(
            gen_cfg.get("repetition_guard_enabled", True),
            "generation.repetition_guard_enabled",
        ),
        multiple_reply_style=multiple_reply_style,
        multiple_probability=multiple_probability,
        ai_flavor_rules=ai_flavor_rules,
        ai_flavor_retries=ai_flavor_retries,
        judge_temperature=judge_temperature,
        describe_temperature=describe_temperature,
        assess_temperature=assess_temperature,
        learning_temperature=learning_temperature,
        generate_max_attempts=generate_max_attempts,
        generate_retry_base_delay=generate_retry_base_delay,
        max_reconsider_per_burst=max_reconsider_per_burst,
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
        download_timeout_seconds=_parse_float(mm_cfg, "download_timeout_seconds", 15.0),
        max_image_bytes=_parse_int(mm_cfg, "max_image_bytes", 8 * 1024 * 1024),
        max_images_per_message=_parse_int(mm_cfg, "max_images_per_message", 4),
    )
    if not math.isfinite(multimodal_settings.download_timeout_seconds) or (
        multimodal_settings.download_timeout_seconds <= 0
    ):
        raise ValueError(
            "配置项 `multimodal.download_timeout_seconds` 应为正有限数字，"
            f"实际是 {multimodal_settings.download_timeout_seconds!r}"
        )
    for key, value in (
        ("max_image_bytes", multimodal_settings.max_image_bytes),
        ("max_images_per_message", multimodal_settings.max_images_per_message),
    ):
        if value < 1:
            raise ValueError(f"配置项 `multimodal.{key}` 不能小于 1，实际是 {value!r}")

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
    send_max_attempts = _parse_int(snow_cfg, "send_max_attempts", 3)
    if send_max_attempts < 1:
        raise ValueError(
            f"配置项 `snowluma.send_max_attempts` 不能小于 1，"
            f"实际是 {send_max_attempts!r}"
        )
    send_retry_delay_seconds = _parse_float(snow_cfg, "send_retry_delay_seconds", 1.5)
    if not math.isfinite(send_retry_delay_seconds) or send_retry_delay_seconds < 0:
        raise ValueError(
            "配置项 `snowluma.send_retry_delay_seconds` 应为非负有限数字，"
            f"实际是 {send_retry_delay_seconds!r}"
        )
    snowluma_settings = SnowlumaSettings(
        endpoint=endpoint,
        api_key=_parse_str(snow_cfg, "api_key", ""),
        timeout_ms=_parse_int(snow_cfg, "timeout_ms", 30000),
        allow_private_endpoint=allow_private,
        send_max_attempts=send_max_attempts,
        send_retry_delay_seconds=send_retry_delay_seconds,
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
    def _parse_typing_seconds(key: str, default: float) -> float:
        value = _parse_float(pp_cfg, key, default)
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"配置项 `response_post_process.{key}` 应为非负有限数字，"
                f"实际是 {value!r}"
            )
        return value

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
        typing_cjk_seconds=_parse_typing_seconds("typing_cjk_seconds", 0.3),
        typing_latin_seconds=_parse_typing_seconds("typing_latin_seconds", 0.15),
        typing_special_seconds=_parse_typing_seconds("typing_special_seconds", 1.0),
        typing_single_multiplier=_parse_typing_seconds("typing_single_multiplier", 3.0),
        max_typing_delay_seconds=_parse_typing_seconds("max_typing_delay_seconds", 60.0),
    )

    # learning 段可整体省略（全部走默认值）
    learn_cfg = _optional_section(cfg, "learning")

    def _parse_positive_int(key: str, default: int) -> int:
        value = _parse_int(learn_cfg, key, default)
        if value < 1:
            raise ValueError(
                f"配置项 `learning.{key}` 不能小于 1，实际是 {value!r}"
            )
        return value

    learning_settings = LearningSettings(
        enabled=_parse_bool(learn_cfg.get("enabled", True), "learning.enabled"),
        impression_enabled=_parse_bool(
            learn_cfg.get("impression_enabled", True), "learning.impression_enabled"
        ),
        impression_days=_parse_positive_int("impression_days", 3),
        impression_max_chars=_parse_positive_int("impression_max_chars", 300),
        expression_enabled=_parse_bool(
            learn_cfg.get("expression_enabled", True), "learning.expression_enabled"
        ),
        expression_batch_size=_parse_positive_int("expression_batch_size", 10),
        expression_max_inject=_parse_positive_int("expression_max_inject", 3),
        expression_self_review=_parse_bool(
            learn_cfg.get("expression_self_review", True),
            "learning.expression_self_review",
        ),
        jargon_enabled=_parse_bool(
            learn_cfg.get("jargon_enabled", True), "learning.jargon_enabled"
        ),
        jargon_max_entries=_parse_positive_int("jargon_max_entries", 50),
        jargon_max_inject=_parse_positive_int("jargon_max_inject", 5),
        pending_buffer_factor=_parse_positive_int("pending_buffer_factor", 3),
        impression_text_budget=_parse_positive_int("impression_text_budget", 6000),
        jargon_candidates_per_batch=_parse_positive_int(
            "jargon_candidates_per_batch", 5
        ),
        jargon_meaning_max_chars=_parse_positive_int("jargon_meaning_max_chars", 200),
    )

    # stickers 段可整体省略（全部走默认值）
    sticker_cfg = _optional_section(cfg, "stickers")
    send_probability = _parse_float(sticker_cfg, "send_probability", 0.05)
    if not 0 <= send_probability <= 1:
        raise ValueError(
            f"配置项 `stickers.send_probability` 应在 0~1 之间，"
            f"实际是 {send_probability!r}"
        )
    sticker_max_count = _parse_int(sticker_cfg, "max_count", 64)
    if sticker_max_count < 1:
        raise ValueError(
            f"配置项 `stickers.max_count` 不能小于 1，实际是 {sticker_max_count!r}"
        )
    max_side_px = _parse_int(sticker_cfg, "max_side_px", 512)
    if max_side_px < 1:
        raise ValueError(
            f"配置项 `stickers.max_side_px` 不能小于 1，实际是 {max_side_px!r}"
        )
    keywords_raw = _get(sticker_cfg, "summary_keywords", list(STICKER_SUMMARY_KEYWORDS_DEFAULT))
    if (
        not isinstance(keywords_raw, (list, tuple))
        or not all(isinstance(item, str) and item.strip() for item in keywords_raw)
    ):
        raise ValueError(
            "配置项 `stickers.summary_keywords` 应为非空字符串的列表"
            "（显式 [] 表示 describe 模式的关键词识别永不命中）"
        )
    sticker_settings = StickerSettings(
        enabled=_parse_bool(sticker_cfg.get("enabled", True), "stickers.enabled"),
        send_probability=send_probability,
        max_count=sticker_max_count,
        max_side_px=max_side_px,
        summary_keywords=tuple(str(item).strip() for item in keywords_raw),
    )

    # plugins 段可整体省略（全部走默认值）
    plugin_cfg = _optional_section(cfg, "plugins")
    plugin_timeout = _parse_float(plugin_cfg, "timeout_seconds", 30.0)
    if plugin_timeout < 1:
        raise ValueError(
            f"配置项 `plugins.timeout_seconds` 不能小于 1，实际是 {plugin_timeout!r}"
        )
    plugin_dir = _parse_str(plugin_cfg, "dir", "plugins")
    if not plugin_dir.strip():
        raise ValueError("配置项 `plugins.dir` 不能为空")
    plugin_settings = PluginSettings(
        enabled=_parse_bool(plugin_cfg.get("enabled", True), "plugins.enabled"),
        include_commands_in_history=_parse_bool(
            plugin_cfg.get("include_commands_in_history", True),
            "plugins.include_commands_in_history",
        ),
        dir=plugin_dir.strip(),
        timeout_seconds=plugin_timeout,
    )

    return Settings(
        bot=BotSettings(
            self_qq=self_qq,
            listen_host=listen_host,
            listen_port=listen_port,
            event_secret=_parse_optional_str(bot_cfg, "event_secret"),
            data_dir=_parse_str(bot_cfg, "data_dir", "data"),
            log_level=log_level,
            self_nickname=self_nickname,
            max_event_body_bytes=max_event_body_bytes,
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
        learning=learning_settings,
        stickers=sticker_settings,
        plugins=plugin_settings,
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
