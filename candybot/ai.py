"""OpenAI 兼容 API 的调用角色：judge（打分）、reply（回复）、vision（转述）、
learning（群印象/表达/黑话/人物事实等后台学习任务，未配置时继承 judge 的模型）、
embedding（表达语义检索的文本向量化，可选角色，只走 /embeddings 接口）。

judge 角色另兼任主动发言的空闲评估（evaluate_proactive，任务 4）：同一端点
能力、同一工具协议降级开关，输出是二元 speak + intent 而非评分。

每个角色可以来自不同提供商（各自的 base_url / api_key），并可单独配置
上下文窗口与输出上限（见 models.ModelConfig）。learning 角色的后台学习任务
一律走一次性纯文本调用（按提示词契约输出 JSON、从正文解析），不携带 tools、
也不参与 judge/reply 的分层前缀缓存；唯一例外是 smart 跟发选图（pick_sticker），
它同步等待在决策链路里，按「强制工具调用 + 自动降级纯文本」的通用模式走。
embedding 角色只在配置了 models.embedding 时可用（AIClient.embed）；未配置
即抛错，调用方（learning.py）按失败记 WARNING 并降级，绝不静默掩盖。

判定、回复、图片入库评估与表情包审核/选图默认通过强制工具调用（tool_choice
指定函数）获得结构化结果；models.<role>.forced_tool_choice=false 的角色改用 tool_choice=
"auto"（思考模式不支持 required 强制指定的模型，如 qwen3 系列），仍走工具
协议。models.<role>.tool_use=false 的角色改走纯文本协议——提示词
要求在正文输出 JSON / 末尾标记，请求不携带 tools 参数。运行中若端点报
工具相关错误，或接受了 tools 却不返回工具调用，该角色自动降级为纯文本
协议（本进程内生效），被拒的当次请求也会立即换纯文本契约补发，不让这
条消息因此丢失；这保证提示词永远只约定模型能力范围内的回答方式，端点
不支持而只回正文时也回退按正文解析。
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from .aiflavor import detect_ai_flavor
from .models import ChatRecord, GenerationSettings, ModelConfig, ModelSettings
from .postprocess import EMOJI_RE as _EMOJI_RE
from .prompts import (
    Message,
    ai_flavor_retry_block,
    expression_evaluation_prompt,
    expression_learning_prompt,
    final_user_prompt_judge,
    final_user_prompt_judge_recheck,
    final_user_prompt_proactive_judge,
    final_user_prompt_proactive_reply,
    final_user_prompt_reconsider,
    final_user_prompt_reply,
    history_to_turns,
    impression_summary_prompt,
    jargon_compare_inference_prompt,
    jargon_extraction_prompt,
    jargon_inference_alone_prompt,
    jargon_inference_with_context_prompt,
    person_fact_evaluation_prompt,
    person_learning_prompt,
    reply_history_turns,
    sticker_pick_prompt,
)

logger = logging.getLogger(__name__)

_SCORE_RE = re.compile(r"\{\s*\"score\"[\s\S]*?}")

# 收图入库评估输出的扁平 JSON 对象（summary + keep）
_ASSESS_RE = re.compile(r"\{[^{}]*\}")

# 回复末尾的图片生命周期标记：<drop_img 消息编号> / <recall_img 消息编号>
_IMAGE_OP_VALID_RE = re.compile(r"<(drop_img|recall_img)\s+(\d+)\s*>")
# 剥除一切形如标签的残留（含模型写歪的未闭合片段），防止泄进群里
_IMAGE_OP_ANY_RE = re.compile(r"</?(?:drop_img|recall_img)\b[^<>]*>")

# 思考段：闭合的 <think>…</think> 整块删除；未闭合（如被 max_tokens 截断）时
# 从 <think> 起全部删除；孤立的 </think> 一并清理。思考内容绝不能参与判定或回复。
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>|<think>[\s\S]*$|</think>")

# emoji 序列的识别规则与输出层后处理共用，定义见 postprocess.EMOJI_RE。

# 概率掷点统一用加密安全随机源：emoji 掷点本身没有安全属性，但安全扫描
# 会把 random.random() 按「不安全随机数」告警，走 SystemRandom 消除噪音。
_RNG = random.SystemRandom()


@dataclass(frozen=True)
class JudgeVerdict:
    score: int
    reason: str
    to_me: bool = False  # 这条消息是否在对我说、延续与我相关的对话


@dataclass(frozen=True)
class ProactiveVerdict:
    """空闲评估（任务 4）的输出：要不要主动说话、想表达什么。

    这是与逐条打分不同的另一个决策问题，故不照搬 0-10 分：二元 + 一句话
    理由（intent 供生成层使用）。speak 只认显式布尔 true，解析歧义一律
    静默（宁可不说话）。"""

    speak: bool
    intent: str = ""


@dataclass(frozen=True)
class StickerAssessment:
    """表情包收藏的 vision 审核输出（任务 2）：是否适合收藏 + 描述 + 情绪标签。

    acceptable=false 时 description 由提示词约定为拒绝理由（DEBUG 日志带出），
    emotion 可能为空串。描述与情绪供 sticker_meta 表入库、smart 模式选图。"""

    acceptable: bool
    description: str
    emotion: str


@dataclass(frozen=True)
class ImageAssessment:
    """收图入库时视觉模型的判定：总结文本 + 是否继续向模型展示原图 +
    是否为表情包类（供 stickers 收集，见 normalize / stickers）。

    sticker_assessment 为表情包收藏审核结论（描述 + 情绪 + 是否可收藏），
    只在 assess_image 被要求合并产出（want_sticker_meta=True，即
    stickers.moderation_enabled 开启的 direct 模式收图）且模型给出了
    可用结论时非 None；与图片记忆管理（summary/keep）各自开关独立。"""

    summary: str | None
    keep_raw: bool
    is_sticker: bool = False
    sticker_assessment: StickerAssessment | None = None


@dataclass(frozen=True)
class ImageOp:
    """回复模型给出的图片生命周期操作（动作名, 目标消息编号）。"""

    action: str
    message_id: int


@dataclass(frozen=True)
class ReplyDraft:
    """回复模型的一次产出：群聊正文 + 随附的图片生命周期操作。"""

    text: str
    ops: tuple[ImageOp, ...] = ()


# 判定/回复/入库评估的结构化输出工具。只用于承载模型结论（服务端不执行），
# tool_choice 强制指定函数名，正文内容一律不作为答案来源。
_JUDGE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_judgment",
        "description": "提交你对最新消息的回复判定结论。",
        "parameters": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 10,
                    "description": "0~10 的回复意愿评分",
                },
                "to_me": {
                    "type": "boolean",
                    "description": "这条消息是否在对你说、或延续与你有关的对话",
                },
                "reason": {"type": "string", "description": "一句话理由"},
            },
            "required": ["score", "to_me", "reason"],
        },
    },
}

# 主动发言空闲评估（任务 4）的工具：二元 speak + 一句话 intent，不照搬评分。
_PROACTIVE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_proactive",
        "description": "提交你在潜水时刻是否主动发言的决定。",
        "parameters": {
            "type": "object",
            "properties": {
                "speak": {
                    "type": "boolean",
                    "description": "是否要主动发言；不确定或没有明显值得说的就填 false",
                },
                "intent": {
                    "type": "string",
                    "description": "一句话说明你想主动表达什么；不说话时留空",
                },
            },
            "required": ["speak", "intent"],
        },
    },
}

_REPLY_TOOL = {
    "type": "function",
    "function": {
        "name": "send_reply",
        "description": "提交你要发送到群里的回复及图片生命周期操作。",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "要发送的群聊消息正文，像真人打字，不加引号或舞台指示",
                },
                "drop_img": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "以后用不到原图的历史消息编号，可省略",
                },
                "recall_img": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "想重新查看原图的历史消息编号，可省略",
                },
            },
            "required": ["text"],
        },
    },
}

_ASSESS_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_assessment",
        "description": "提交图片入库评估：一句话总结、是否保留原图、是否表情包类。",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "不超过40字的内容总结"},
                "keep": {"type": "boolean", "description": "后续对话是否还需要查看原图"},
                "sticker": {
                    "type": "boolean",
                    "description": "是否为表情包/梗图这类主要用于斗图的图",
                },
            },
            "required": ["summary", "keep", "sticker"],
        },
    },
}

# 收藏审核开启（stickers.moderation_enabled）时，direct 模式收图的入库评估
# 与表情包审核合并为一次 vision 调用：submit_assessment 追加三个表情包字段
# （非必填——只有判为表情包类才需要给出），避免对同一张图重复请求 vision。
_STICKER_MODERATION_FIELDS = {
    "acceptable": {
        "type": "boolean",
        "description": "该表情包是否适合收藏作斗图（审核标准见提示词）",
    },
    "sticker_description": {
        "type": "string",
        "description": "不超过40字中立具体地描述图里在干什么；acceptable=false 时写拒绝理由",
    },
    "emotion": {
        "type": "string",
        "description": "情绪标签，如 得意/无语/狂喜/阴阳怪气/卖萌",
    },
}


def _assess_tool(want_sticker_meta: bool) -> dict:
    """按是否合并表情包审核选 submit_assessment 的参数表。"""
    if not want_sticker_meta:
        return _ASSESS_TOOL
    tool = copy.deepcopy(_ASSESS_TOOL)
    tool["function"]["parameters"]["properties"].update(_STICKER_MODERATION_FIELDS)
    return tool


# 表情包审核标准与描述要求（assess_sticker 与合并调用的提示词共用）。
_STICKER_MODERATION_RULES = (
    "表情包不得含色情、暴力、政治敏感内容；不得是真人照片、游戏/网页截图、"
    "二维码或广告图；画面文字过多（超过 5 个汉字的大段文字图）不算表情包。"
    "描述须中立具体（如「柴犬歪头疑惑」，不要「一张可爱的图」），"
    "不超过 40 字、说明图里在干什么；acceptable=false 时改写拒绝理由；"
    "情绪标签如 得意/无语/狂喜/阴阳怪气/卖萌，不限于这些例子。"
)

# 独立审核调用（placeholder 等模式下 stickers.py 收集路径）用的工具。
_STICKER_ASSESS_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_sticker_assessment",
        "description": "提交表情包收藏审核：是否适合收藏、内容描述、情绪标签。",
        "parameters": {
            "type": "object",
            "properties": {
                "acceptable": {
                    "type": "boolean",
                    "description": "是否适合收藏作斗图（审核标准见提示词）",
                },
                "description": {
                    "type": "string",
                    "description": "不超过40字中立具体地描述图里在干什么；acceptable=false 时写拒绝理由",
                },
                "emotion": {
                    "type": "string",
                    "description": "情绪标签，如 得意/无语/狂喜/阴阳怪气/卖萌",
                },
            },
            "required": ["acceptable", "description", "emotion"],
        },
    },
}

# smart 跟发选图（learning 角色）用的工具：pick=0 表示模型选择不发。
_STICKER_PICK_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_sticker_pick",
        "description": "提交表情包选图结论：候选编号（0＝不发）与一句话理由。",
        "parameters": {
            "type": "object",
            "properties": {
                "pick": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "选中的候选编号；0 表示本次不发表情包",
                },
                "reason": {"type": "string", "description": "一句话理由"},
            },
            "required": ["pick", "reason"],
        },
    },
}


_IMAGE_HEAD = 48  # debug 日志中图片 data URL 只保留头部字符数

# judge 需要输出思考段，上限太小会把结论截掉；models.judge.max_output_tokens 可覆盖
_JUDGE_MAX_TOKENS = 1000
# 视觉两个调用各自的输出上限；models.vision.max_output_tokens 可统一覆盖
_DESCRIBE_MAX_TOKENS = 80
_ASSESS_MAX_TOKENS = 400
# 学习类调用的内置输出上限；models.learning.max_output_tokens 可统一覆盖
_IMPRESSION_MAX_TOKENS = 800
_LEARN_MAX_TOKENS = 2000
_REVIEW_MAX_TOKENS = 300  # 表达自审 / 黑话双路比对这类一句话判定
# 上下文窗口预算中为消息 role 标记等固定格式开销预留的 token 数
_CONTEXT_OVERHEAD_TOKENS = 128


def format_messages_for_log(messages: list[Message]) -> str:
    """把每次请求的消息数组转成可读文本（debug 用）。

    多模态内容块中的 base64 图片只显示长度和头部片段，避免日志爆炸。
    """
    parts: list[str] = []
    for index, message in enumerate(messages):
        content = message["content"]
        if isinstance(content, str):
            body = content
        else:  # OpenAI 内容块数组（direct 多模态）
            rendered: list[str] = []
            for block in content:
                if block.get("type") == "text":
                    rendered.append(str(block.get("text", "")))
                elif block.get("type") == "image_url":
                    url = str(block.get("image_url", {}).get("url", ""))
                    head = url[:_IMAGE_HEAD] + ("…" if len(url) > _IMAGE_HEAD else "")
                    rendered.append(f"[图片 data URL 共 {len(url)} 字符：{head}]")
                else:
                    rendered.append(f"[未知块类型 {block.get('type')!r}]")
            body = "\n".join(rendered)
        parts.append(f"──[{index}] {message['role']}──\n{body}")
    return "\n".join(parts)


def _turn_to_message(turn) -> Message:
    """HistoryTurn → OpenAI 消息；带附图的回合转成 text + image_url 块。"""
    if turn.images:
        return {
            "role": turn.role,
            "content": [
                {"type": "text", "text": turn.content},
                *[
                    {"type": "image_url", "image_url": {"url": data_url}}
                    for data_url in turn.images
                ],
            ],
        }
    return {"role": turn.role, "content": turn.content}


def split_image_ops(text: str) -> tuple[str, list[ImageOp]]:
    """剥除回复里的图片生命周期标记，返回（干净正文, 按序操作列表）。

    接受 <drop_img 编号> / <recall_img 编号> 两种闭合写法；无论是否合法，
    一切形似的标签片段都会被剥除，保证不会发进群里。操作允许出现在正文
    任意位置，重复操作交由记忆层幂等处理。
    """
    ops = [
        ImageOp(action=action, message_id=int(num))
        for action, num in _IMAGE_OP_VALID_RE.findall(text)
    ]
    clean = _IMAGE_OP_ANY_RE.sub("", text)
    lines = [ln for ln in clean.splitlines() if ln.strip()]
    return "\n".join(lines).strip(), ops


def _forced_tool_choice(tool: dict) -> dict:
    """强制模型调用指定函数的 tool_choice 参数。"""
    return {"type": "function", "function": {"name": tool["function"]["name"]}}


def _tool_request_kwargs(tool: dict, *, force: bool = True) -> dict:
    """工具调用模式的 create 附加参数：携带工具定义并（默认）强制调用。

    force=False 时改用 tool_choice="auto"：思考（thinking）模式的模型普遍
    不支持 required/object 的强制指定（如 qwen3 系列会报 400），但接受
    auto，由提示词引导模型主动调用。
    """
    choice = _forced_tool_choice(tool) if force else "auto"
    return {"tools": [tool], "tool_choice": choice}


def _looks_like_tools_rejection(exc: Exception) -> bool:
    """端点把 tools/tool_choice 参数当错误时，报错文本几乎都带这些字样。

    只认报错关键词，不看状态码：400 也可能是上下文超限等无关错误，
    误降级会让可用的工具协议被白白放弃。
    """
    text = str(exc).lower()
    return "tool" in text or "function" in text or "工具" in text


def _returned_tool_call(response) -> bool:
    """响应是否携带工具调用；False 说明端点接受了 tools 却没用（多半忽略）。"""
    try:
        return bool(response.choices[0].message.tool_calls)
    except (AttributeError, IndexError):
        return False


def _tool_arguments(response, name: str) -> dict | None:
    """取响应中首个可用工具调用的参数对象；没有工具调用时返回 None。

    优先取名为 name 的调用，取不到时退而求其次接受第一个能解析出参数
    字典的调用（部分兼容端点会改写函数名）。参数不是合法 JSON 视为没有。
    """
    try:
        calls = response.choices[0].message.tool_calls or []
    except (AttributeError, IndexError):
        return None
    fallback: dict | None = None
    for call in calls:
        fn = getattr(call, "function", None)
        if fn is None:
            continue
        raw = fn.arguments or ""
        try:
            args = json.loads(raw)
        except ValueError:
            logger.warning("工具调用 %s 的参数不是合法 JSON：%r", name, raw[:200])
            continue
        if not isinstance(args, dict):
            continue
        if fn.name == name:
            return args
        if fallback is None:
            fallback = args
    return fallback


def _ops_from_reply_args(args: dict) -> list[ImageOp]:
    """send_reply 工具参数里的图片生命周期操作；无法转换的脏值直接忽略。"""
    ops: list[ImageOp] = []
    for action in ("drop_img", "recall_img"):
        values = args.get(action)
        if values is None:
            continue
        if not isinstance(values, list):
            values = [values]
        for value in values:
            try:
                ops.append(ImageOp(action=action, message_id=int(value)))
            except (TypeError, ValueError):
                continue
    return ops


class AIClient:
    """封装三种 LLM 调用。所有方法都可能抛出 openai 库异常，由上层决定重试。

    每个角色一个 ModelConfig：可指向不同提供商并携带各自的窗口/输出限额；
    相同 (base_url, api_key) 的角色共享同一个 AsyncOpenAI 连接池。
    """

    def __init__(
        self,
        *,
        models: ModelSettings,
        generation: GenerationSettings,
        multimodal_mode: str = "placeholder",
    ):
        self._judge = models.judge
        self._reply = models.reply
        self._vision = models.vision
        # learning 角色未配置时继承 judge（便宜快速）：学习任务与判定共用模型
        self._learning = models.learning or models.judge
        # embedding 为可选角色（不继承任何模型）：只用于表达语义检索的向量化
        self._embedding = models.embedding
        self._generation = generation
        # 只有 direct 模式才允许任何图片（历史层或当前消息）进入请求
        self._multimodal_mode = multimodal_mode
        self._clients: dict[tuple[str, str], AsyncOpenAI] = {}
        # 每个角色当前是否走工具调用协议：初值来自模型配置，运行中遇端点
        # 不支持（报错或忽略 tools）时降级为 False，本进程内不再回升。
        # sticker_pick 是借用 learning 模型配置（_cfg_for）的独立协议开关：
        # smart 选图走工具调用，其余学习任务照旧纯文本，降级互不影响。
        self._tools_on: dict[str, bool] = {
            "judge": models.judge.tool_use,
            "reply": models.reply.tool_use,
            "vision": models.vision.tool_use if models.vision else False,
            "sticker_pick": self._learning.tool_use,
        }

    @property
    def judge_tool_use(self) -> bool:
        """judge 角色当前是否走工具调用协议。"""
        return self._tools_on["judge"]

    @property
    def reply_tool_use(self) -> bool:
        """reply 角色当前是否走工具调用协议；bot 据此选择 L1 守则措辞。"""
        return self._tools_on["reply"]

    def _disable_tools(self, role: str, why: str) -> None:
        """该角色降级为纯文本协议；此后提示词契约与请求都不再涉及 tools。"""
        if not self._tools_on[role]:
            return
        self._tools_on[role] = False
        logger.warning(
            "%s 角色模型 %s 疑似不支持工具调用（%s），已改用纯文本协议；"
            "如需固定行为，请在 config.json5 中为该角色设置 tool_use: false",
            role,
            self._cfg_for(role).model,
            why,
        )

    def _cfg_for(self, role: str) -> ModelConfig:
        return {
            "judge": self._judge,
            "reply": self._reply,
            "vision": self._vision,
            "learning": self._learning,
            # smart 选图借用 learning 角色的模型配置（协议开关独立）
            "sticker_pick": self._learning,
        }[role]

    def _client_for(self, cfg: ModelConfig) -> AsyncOpenAI:
        key = (cfg.base_url, cfg.api_key)
        client = self._clients.get(key)
        if client is None:
            # api_key 留空表示端点无需密钥：先取环境变量，再退本地服务通用的占位符
            api_key = cfg.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
            client = AsyncOpenAI(base_url=cfg.base_url or None, api_key=api_key)
            self._clients[key] = client
        return client

    async def _create_chat(
        self,
        role: str,
        request: dict,
        text_messages: Callable[[], list[Message]],
    ) -> tuple[Any, bool]:
        """发送请求；端点拒绝 tools 时降级为纯文本协议并立即补发本次请求。

        工具请求报错时若只标记降级再抛出，当次调用会被上层当失败丢弃
        （judge 失败＝这条消息不发言）。这里换上 text_messages 重建的纯
        文本消息并去掉 tools/tool_choice 补发一次，让当次调用也走通；
        重试仍失败则原样抛出。返回（响应, 响应是否来自工具调用请求）。
        """
        cfg = self._cfg_for(role)
        try:
            response = await self._client_for(cfg).chat.completions.create(**request)
        except Exception as exc:
            if "tools" not in request or not _looks_like_tools_rejection(exc):
                raise
            self._disable_tools(role, str(exc))
            retry = {**request, "messages": text_messages()}
            retry.pop("tools", None)
            retry.pop("tool_choice", None)
            response = await self._client_for(cfg).chat.completions.create(**retry)
            return response, False
        return response, "tools" in request

    def _history_chars(self, cfg: ModelConfig, prompt_chars: int) -> int:
        """结合模型上下文窗口推算历史层的字符预算。

        窗口按 token 计而历史裁剪按字符计，中文场景保守按 1 token ≈ 1 char
        折算，并预留本次输出与固定格式开销；未配置窗口时沿用全局字符上限。
        direct 模式原图的 token 开销因提供商而异，不参与估算。
        """
        if cfg.context_window is None:
            return self._generation.max_context_chars
        reserved = (cfg.max_output_tokens or 0) + prompt_chars + _CONTEXT_OVERHEAD_TOKENS
        return max(0, min(self._generation.max_context_chars, cfg.context_window - reserved))

    # ---------------------------------------------------------------- embedding

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """embedding 角色（OpenAI 兼容 /embeddings）批量向量化。

        返回与入参顺序对齐的向量列表：兼容端点未必按请求顺序返回 data，
        统一按其 index 排序。未配置 models.embedding、端点报错或返回条数
        与请求不符都如实上抛——启动配置校验保证 vector 模式有该角色，
        运行期失败由调用方（learning.py）记 WARNING 并降级处理。
        """
        if self._embedding is None:
            raise RuntimeError("models.embedding 未配置，无法计算文本向量")
        response = await self._client_for(self._embedding).embeddings.create(
            model=self._embedding.model,
            input=list(texts),
            timeout=self._generation.timeout_seconds,
        )
        items = sorted(response.data, key=lambda d: getattr(d, "index", 0) or 0)
        vectors = [list(item.embedding) for item in items]
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"embedding 返回 {len(vectors)} 条向量，与本次请求的 {len(texts)} 条文本不符"
            )
        return vectors

    # ---------------------------------------------------------------- judge

    async def judge_interest(
        self,
        static_system: str,
        runtime_system: str,
        recent_records: list[ChatRecord],
        current_message: ChatRecord,
        now_text: str,
        *,
        threshold: int | None = None,
        prev_verdict: JudgeVerdict | None = None,
        min_score: int | None = None,
    ) -> JudgeVerdict:
        """判断模型评估是否回复该消息；解析失败按 0 分处理并记日志。

        prev_verdict 传入首评的判定时进入复核模式：本次调用会把本群真实
        门槛与复核下限 min_score 告知模型，请其针对「高于下限却未达门槛」
        的首评结论重新裁定。首评与复核共用相同的 L1-L3 前缀，KVCache 可
        直接复用。输出契约随 judge 角色的工具调用状态选择。
        """
        use_tools = self._tools_on["judge"]
        if prev_verdict is not None:
            assert threshold is not None, "复核模式必须提供 threshold 以告知门槛"
            assert min_score is not None, "复核模式必须提供 min_score 以告知触发下限"
            tag = "[judge·复核]"
        else:
            tag = "[judge]"

        def build_messages(via_tool: bool) -> list[Message]:
            # 历史层不含当前消息本身（它单独出现在指令层，保证历史层前缀稳定）
            # 预算需扣除 L4 文本本身，故先构造指令层再裁历史
            if prev_verdict is not None:
                final_user = final_user_prompt_judge_recheck(
                    now_text,
                    current_message,
                    prev_score=prev_verdict.score,
                    prev_reason=prev_verdict.reason,
                    threshold=threshold,
                    min_score=min_score,
                    via_tool=via_tool,
                )
            else:
                final_user = final_user_prompt_judge(
                    now_text, current_message, via_tool=via_tool
                )
            prompt_chars = len(static_system) + len(runtime_system) + len(final_user)
            turns, _ = history_to_turns(
                recent_records[:-1], self._history_chars(self._judge, prompt_chars)
            )
            return [
                {"role": "system", "content": static_system},
                {"role": "system", "content": runtime_system},
                *[{"role": t.role, "content": t.content} for t in turns],
                {"role": "user", "content": final_user},
            ]

        chat = build_messages(use_tools)
        logger.debug(
            "%s model=%s 消息数=%d\n%s",
            tag,
            self._judge.model,
            len(chat),
            format_messages_for_log(chat),
        )
        request: dict = dict(
            model=self._judge.model,
            messages=chat,
            temperature=self._generation.judge_temperature,
            # 推理型 judge 模型的思考段会占掉不少 token，上限太小会把结论截掉
            max_tokens=self._judge.max_output_tokens or _JUDGE_MAX_TOKENS,
            timeout=self._generation.timeout_seconds,
        )
        if use_tools:
            request.update(
                _tool_request_kwargs(_JUDGE_TOOL, force=self._judge.forced_tool_choice)
            )
        response, via_tools = await self._create_chat(
            "judge", request, lambda: build_messages(False)
        )
        if via_tools:
            if not _returned_tool_call(response):
                # 端点没报错但也没调用工具：多半整个忽略了 tools 参数
                self._disable_tools("judge", "响应中无工具调用")
            else:
                args = _tool_arguments(response, _JUDGE_TOOL["function"]["name"])
                if args is not None:
                    return self._verdict_from_args(args)
        # 回退/纯文本协议：按旧约定从正文解析
        raw = (response.choices[0].message.content or "").strip()
        return self._parse_verdict(raw)

    async def evaluate_proactive(
        self,
        static_system: str,
        runtime_system: str,
        recent_records: list[ChatRecord],
        now_text: str,
    ) -> ProactiveVerdict:
        """主动发言空闲评估（任务 4）：潜水时刻看一眼，要不要自己开口。

        与逐条打分的 judge 是两个不同的决策问题，不复用 0-10 分：输出
        二元 speak + 一句话 intent（供生成层照着说）。走 judge 角色的工具/
        文本双协议与自动降级（同一端点能力，共用 `_tools_on["judge"]`）。
        调用异常原样上抛（bot 层记 WARNING 本轮作罢、绝不重试轰炸）；输出
        解析不出或 speak 非显式布尔 true 一律返回静默（宁可不说话）。
        """
        use_tools = self._tools_on["judge"]

        def build_messages(via_tool: bool) -> list[Message]:
            final_user = final_user_prompt_proactive_judge(now_text, via_tool=via_tool)
            prompt_chars = len(static_system) + len(runtime_system) + len(final_user)
            turns, _ = history_to_turns(
                recent_records, self._history_chars(self._judge, prompt_chars)
            )
            return [
                {"role": "system", "content": static_system},
                {"role": "system", "content": runtime_system},
                *[{"role": t.role, "content": t.content} for t in turns],
                {"role": "user", "content": final_user},
            ]

        chat = build_messages(use_tools)
        logger.debug(
            "[proactive] model=%s 消息数=%d\n%s",
            self._judge.model,
            len(chat),
            format_messages_for_log(chat),
        )
        request: dict = dict(
            model=self._judge.model,
            messages=chat,
            temperature=self._generation.judge_temperature,
            max_tokens=self._judge.max_output_tokens or _JUDGE_MAX_TOKENS,
            timeout=self._generation.timeout_seconds,
        )
        if use_tools:
            request.update(
                _tool_request_kwargs(
                    _PROACTIVE_TOOL, force=self._judge.forced_tool_choice
                )
            )
        response, via_tools = await self._create_chat(
            "judge", request, lambda: build_messages(False)
        )
        if via_tools:
            if not _returned_tool_call(response):
                self._disable_tools("judge", "响应中无工具调用")
            else:
                args = _tool_arguments(response, _PROACTIVE_TOOL["function"]["name"])
                if args is not None:
                    return self._proactive_from_args(args)
        # 回退/纯文本协议：按契约从正文解析 JSON
        raw = (response.choices[0].message.content or "").strip()
        return self._parse_proactive(raw)

    @staticmethod
    def _proactive_from_args(args: dict) -> ProactiveVerdict:
        # speak 只认布尔 true：脏数据、缺失一律静默（绝不误冒泡）
        speak = args.get("speak") is True
        intent = str(args.get("intent") or "").strip()[:100]
        return ProactiveVerdict(speak=speak, intent=intent)

    @staticmethod
    def _parse_proactive(raw: str) -> ProactiveVerdict:
        visible = _strip_think(raw)
        obj = _extract_json(visible)
        if isinstance(obj, dict):
            return AIClient._proactive_from_args(obj)
        logger.warning("主动发言评估输出无法解析，按静默处理：%r", raw[:200])
        return ProactiveVerdict(speak=False)

    @staticmethod
    def _verdict_from_args(args: dict) -> JudgeVerdict:
        try:
            score = int(args["score"])
        except (KeyError, TypeError, ValueError):
            logger.warning("judge 工具调用缺少合法 score：%r", args)
            return JudgeVerdict(score=0, reason="工具调用参数缺失")
        return JudgeVerdict(
            score=max(0, min(10, score)),
            # to_me 只认布尔 true：脏数据一律视为 False，绝不误判「在和我说话」
            to_me=args.get("to_me") is True,
            reason=str(args.get("reason", ""))[:100],
        )

    @staticmethod
    def _parse_verdict(raw: str) -> JudgeVerdict:
        # 判定只看思考段之外的内容：推理文本里的编号、锚点分数都不能当结果
        visible = _strip_think(raw)
        if "<think>" in raw and "</think>" not in raw:
            logger.warning("judge 输出含未闭合的 <think>，疑似被 max_tokens 截断")
            return JudgeVerdict(score=0, reason="思考被截断，输出解析失败")
        obj = _verdict_from_json(visible)
        if obj is not None:
            return obj
        num = re.search(r"\b(\d|10)\b", visible)
        if num:
            logger.warning("解析到非结构化输出，原文: %s", visible[:200])
            return JudgeVerdict(score=int(num.group(1)), reason="从非结构化输出中提取")
        logger.warning("judge 输出无法解析：%r", raw[:200])
        return JudgeVerdict(score=0, reason="输出解析失败")

    # ---------------------------------------------------------------- reply

    async def generate_reply(
        self,
        static_system: str,
        runtime_system: str,
        recent_records: list[ChatRecord],
        current_message: ChatRecord | None,
        now_text: str,
        *,
        forced: bool,
        engaged: bool = False,
        score: int | None = None,
        reason: str = "",
        expression_hints: Sequence[tuple[str, str]] = (),
        jargon_hints: Sequence[tuple[str, str]] = (),
        repetition_warning: bool = False,
        person_hints: Sequence[tuple[str, Sequence[str]]] = (),
        proactive_intent: str | None = None,
    ) -> ReplyDraft | None:
        """回复模型生成一句群聊回应；direct 模式下历史与当前消息可携带图片块。

        proactive_intent（任务 4 自发言）非 None 时走自发言变体：L4 换成
        final_user_prompt_proactive_reply（没人问话、按 intent 主动开口），
        历史层为传入的全部 recent_records（没有「当前消息」可剥离），
        current_message 忽略。AI 味拦截、临时风格与 emoji 清洗照常。None
        （常规回复）时输出与引入本参数之前逐字节一致。

        工具调用模式下经 send_reply 的强制调用拿到（正文, 图片操作）；纯文本
        协议下按旧约定解析正文（标记写在末尾）。返回 None 表示无话可说。

        expression_hints / jargon_hints 为学习机制产出的注入条目（见
        prompts.final_user_prompt_reply），属于每次回复都可变的易变信息，
        只进 L4 指令层。repetition_warning 同理：bot 层判定很可能在重复
        刚才的发言时为 True，L4 注入重复提醒。person_hints 为人物长期记忆
        画像（(昵称, 事实列表)，learning.pick_person_profiles 产出），同为
        易变信息只进 L4；为空时输出与引入人物记忆之前字节级一致。

        风格多样性与内容拦截两层附加处理（注入都只进 L4，reconsider_reply
        均不走这两层）：

        - 任务 A：每次调用独立掷点抽一条临时风格（见 _pick_temporary_style），
          命中注入 L4；发生下面的 AI 味重生成时沿用同一条风格，不再重掷。
        - 任务 B：生成与清洗完成后过一轮 AI 味正则检测（aiflavor），命中时
          把被拦截回复与原因附进 L4 重新生成，至多 ai_flavor_retries 次
          （0 关闭整个环节）；重试后仍命中则放行并记 warning——宁可留着
          这句稍假的话，也绝不死循环卡住决策队列。这是内容级重试，与 bot
          层 _generate_with_retry 的网络重试相互独立；重试调用自身抛出的
          网络异常照常上抛，交网络重试处理。
        """
        temporary_style = self._pick_temporary_style()
        retry_notes: list[str] = []
        # 自发言没有「当前消息」可剥离：整段 recent 都是历史层
        history_records = recent_records if proactive_intent is not None else recent_records[:-1]
        current_for_call = None if proactive_intent is not None else current_message

        def build_text_part(via_tool: bool) -> str:
            if proactive_intent is not None:
                text = final_user_prompt_proactive_reply(
                    now_text,
                    proactive_intent,
                    via_tool=via_tool,
                    expression_hints=expression_hints,
                    jargon_hints=jargon_hints,
                    temporary_style=temporary_style,
                    person_hints=person_hints,
                )
            else:
                text = final_user_prompt_reply(
                    now_text,
                    current_message,
                    forced=forced,
                    engaged=engaged,
                    score=score,
                    reason=reason,
                    via_tool=via_tool,
                    expression_hints=expression_hints,
                    jargon_hints=jargon_hints,
                    repetition_warning=repetition_warning,
                    temporary_style=temporary_style,
                    person_hints=person_hints,
                )
            return "\n\n".join([text, *retry_notes]) if retry_notes else text

        draft = await self._reply_call(
            static_system,
            runtime_system,
            history_records,
            current_for_call,
            build_text_part,
        )
        gen = self._generation
        if draft is not None and draft.text and gen.ai_flavor_rules and gen.ai_flavor_retries > 0:
            violation = detect_ai_flavor(draft.text, gen.ai_flavor_rules)
            attempts = 0
            while violation is not None and attempts < gen.ai_flavor_retries:
                attempts += 1
                logger.info(
                    "[reply] AI 味拦截（第 %d/%d 次重生成）：%s，被拦截回复：%r",
                    attempts,
                    gen.ai_flavor_retries,
                    violation,
                    draft.text[:120],
                )
                retry_notes.append(ai_flavor_retry_block(draft.text, violation))
                draft = await self._reply_call(
                    static_system,
                    runtime_system,
                    history_records,
                    current_for_call,
                    build_text_part,
                )
                violation = (
                    detect_ai_flavor(draft.text, gen.ai_flavor_rules)
                    if draft is not None and draft.text
                    else None
                )
            if violation is not None:
                logger.warning(
                    "[reply] AI 味重生成 %d 次后仍命中，放行：%s（回复：%r）",
                    attempts,
                    violation,
                    (draft.text if draft else "")[:120],
                )
        return draft

    def _pick_temporary_style(self) -> str | None:
        """任务 A：每条回复独立掷点抽一条临时风格；未命中或关闭返回 None。

        概率为 0 或池为空时直接短路、不消耗随机数；掷点与抽取统一走加密
        安全的 _RNG（与 emoji 掷点同一约定）。
        """
        gen = self._generation
        if gen.multiple_probability <= 0 or not gen.multiple_reply_style:
            return None
        if _RNG.random() >= gen.multiple_probability:
            return None
        style = str(_RNG.choice(gen.multiple_reply_style)).strip()
        if not style:
            return None
        logger.debug("[reply] 注入临时风格：%r", style)
        return style

    async def reconsider_reply(
        self,
        static_system: str,
        runtime_system: str,
        recent_records: list[ChatRecord],
        now_text: str,
        *,
        sent_segments: Sequence[str],
        pending_segments: Sequence[str],
    ) -> ReplyDraft:
        """连发期间被他人插话后，重新裁定腹稿里还没发出去的部分。

        与 generate_reply 的区别：历史层直接用截至当下的完整尾部（插来的
        消息与自己已发出的连发片段都已如实入库），不再剥离「当前消息」；
        指令层转述未发送的腹稿请模型取舍。返回的 ReplyDraft 可能正文为空，
        空即「模型决定不发了」；调用失败照常抛异常，由上层重试后按原计划
        继续——两种「没有内容」必须可区分，故这里不把空正文折成 None。
        """
        draft = await self._reply_call(
            static_system,
            runtime_system,
            list(recent_records),
            None,
            lambda via_tool: final_user_prompt_reconsider(
                now_text, sent_segments, pending_segments, via_tool=via_tool
            ),
        )
        return draft or ReplyDraft("")

    async def _reply_call(
        self,
        static_system: str,
        runtime_system: str,
        history_records: list[ChatRecord],
        current_message: ChatRecord | None,
        build_text_part: Callable[[bool], str],
    ) -> ReplyDraft | None:
        """generate_reply / reconsider_reply 共用的 reply 模型调用。

        history_records 即历史层内容（「当前消息」的剥离位置由调用方按其
        约定完成）；build_text_part 按协议生成本次 L4 指令层，端点拒绝
        tools 降级补发时以 via_tool=False 重建。正文为空返回 None。
        """
        use_tools = self._tools_on["reply"]
        text_part = build_text_part(use_tools)
        # 预算需扣除 L4 文本本身，故先构造指令层再裁历史；原图 token 开销不参与估算
        prompt_chars = len(static_system) + len(runtime_system) + len(text_part)
        budget = self._history_chars(self._reply, prompt_chars)
        if self._multimodal_mode == "direct":
            turns, _ = reply_history_turns(
                history_records,
                budget,
                self._generation.max_history_images,
            )
            history_messages = [_turn_to_message(t) for t in turns]
        else:
            turns, _ = history_to_turns(history_records, budget)
            history_messages = [{"role": t.role, "content": t.content} for t in turns]
        final_user: str | list[dict]
        if (
            self._multimodal_mode == "direct"
            and current_message is not None
            and current_message.images
        ):
            blocks: list[dict] = [{"type": "text", "text": text_part}]
            for data_url in current_message.images:
                blocks.append({"type": "image_url", "image_url": {"url": data_url}})
            final_user = blocks
        else:
            final_user = text_part

        chat: list[Message] = [
            {"role": "system", "content": static_system},
            {"role": "system", "content": runtime_system},
            *history_messages,
            {"role": "user", "content": final_user},
        ]
        logger.debug(
            "[reply] model=%s 消息数=%d\n%s",
            self._reply.model,
            len(chat),
            format_messages_for_log(chat),
        )
        request: dict = dict(
            model=self._reply.model,
            messages=chat,
            temperature=self._generation.temperature,
            max_tokens=self._reply.max_output_tokens or self._generation.reply_max_tokens,
            timeout=self._generation.timeout_seconds,
        )
        if use_tools:
            request.update(
                _tool_request_kwargs(_REPLY_TOOL, force=self._reply.forced_tool_choice)
            )

        def text_messages() -> list[Message]:
            # 降级补发：仅把 L4 指令换成纯文本契约（direct 模式保留图片块）。
            # L1 守则由调用方按当时的角色状态生成，此处无法重建、仍带工具措辞，
            # 但 L4 是最后的明确指令（只输出正文），输出契约以它为准。
            retry_text = build_text_part(False)
            last = chat[-1]["content"]
            content = (
                [{"type": "text", "text": retry_text}, *last[1:]]
                if isinstance(last, list)
                else retry_text
            )
            retried = [*chat]
            retried[-1] = {"role": "user", "content": content}
            return retried

        response, via_tools = await self._create_chat("reply", request, text_messages)
        if via_tools and _returned_tool_call(response):
            args = _tool_arguments(response, _REPLY_TOOL["function"]["name"])
            if args is not None:
                # 工具正文仍过一遍标记剥除：模型偶尔会沿用旧习惯把标记写进 text
                text, tag_ops = split_image_ops(str(args.get("text") or ""))
                ops = (*_ops_from_reply_args(args), *tag_ops)
            else:
                text, ops = "", ()
        else:
            if via_tools:
                # 端点没报错但也没调用工具：多半整个忽略了 tools 参数
                self._disable_tools("reply", "响应中无工具调用")
            # 回退/纯文本协议：按旧约定从正文解析（标记写在末尾）
            text, ops = split_image_ops(response.choices[0].message.content or "")
        text = _strip_noise(text)
        roll_ok = _RNG.random() < self._generation.emoji_chance
        text = _cap_emojis(text, self._generation.emoji_max if roll_ok else 0)
        if not text:
            return None
        return ReplyDraft(text, tuple(ops))

    # --------------------------------------------------------------- vision

    async def describe_image(self, data_url: str) -> str | None:
        """视觉模型把图片转成一句话描述；未配置 vision 模型时返回 None。"""
        if not self._vision:
            return None
        logger.debug(
            "[vision] model=%s 图片 %d 字符",
            self._vision.model,
            len(data_url),
        )
        response = await self._client_for(self._vision).chat.completions.create(
            model=self._vision.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请用不超过40个字描述这张图片的内容要点。"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            temperature=self._generation.describe_temperature,
            max_tokens=self._vision.max_output_tokens or _DESCRIBE_MAX_TOKENS,
            timeout=self._generation.timeout_seconds,
        )
        desc = (response.choices[0].message.content or "").strip()
        return desc or None

    async def assess_image(
        self, data_url: str, *, want_sticker_meta: bool = False
    ) -> ImageAssessment:
        """视觉模型一次性完成「总结 + 是否值得长期保留原图」的入库判定。

        未配置 vision 模型或调用/解析失败时安全侧返回保留原图：宁可
        多花些 token 也不凭空丢信息，后续仍可通过 <drop_img> 标记降级。

        want_sticker_meta=True 时在同一次调用里追加表情包收藏审核输出
        （acceptable/description/emotion，仅当判为表情包类才需要给出），
        供 direct 模式收集免二次请求 vision；结论放在 sticker_assessment 字段。
        """
        if not self._vision:
            return ImageAssessment(summary=None, keep_raw=True)
        use_tools = self._tools_on["vision"]
        tool = _assess_tool(want_sticker_meta)
        logger.debug(
            "[vision·assess] model=%s 图片 %d 字符%s",
            self._vision.model,
            len(data_url),
            "（含表情包审核）" if want_sticker_meta else "",
        )

        def messages(via_tool: bool) -> list[Message]:
            text = (
                "这是一张群聊里发来的图片。请先用不超过40个字总结它的内容要点，"
                "再判断后续对话是否还需要继续查看这张原图：只有当图片包含未来"
                "可能被反复引用的具体信息（文字截图、代码、表格、关键画面细节等）"
                "才值得保留原图；表情包、梗图之类的总结即可。最后判断它是不是"
                "「表情包类」图片：以玩梗、表达情绪为目的的斗图表情、梗图、"
                "搞笑动图都算；截图、照片、信息图不算。\n"
            )
            if want_sticker_meta:
                text += (
                    "若判定为表情包类（sticker=true），还要给出三个字段："
                    "acceptable 表示该图是否适合收藏作斗图，"
                    "sticker_description 用不超过40字中立具体地描述图里在干什么"
                    "（acceptable=false 时改写拒绝理由），emotion 给一个情绪标签。"
                    "审核标准与描述要求：" + _STICKER_MODERATION_RULES + "\n"
                )
            if via_tool:
                text += (
                    "完成后调用 submit_assessment 工具提交：summary 为一句话总结，"
                    "keep 表示后续对话是否还需要查看原图，sticker 表示是否为表情包类。"
                )
                if want_sticker_meta:
                    text += (
                        "sticker=true 时另附 acceptable、sticker_description 与 emotion。"
                    )
            else:
                text += '只输出一个 JSON 对象：{"summary": "一句话总结",'
                text += ' "keep": true 或 false, "sticker": true 或 false'
                if want_sticker_meta:
                    text += (
                        '，sticker 为 true 时另附 "acceptable": true 或 false,'
                        ' "sticker_description": "…", "emotion": "…"'
                    )
                text += "}"
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ]

        request: dict = dict(
            model=self._vision.model,
            messages=messages(use_tools),
            temperature=self._generation.assess_temperature,
            max_tokens=self._vision.max_output_tokens or _ASSESS_MAX_TOKENS,
            timeout=self._generation.timeout_seconds,
        )
        if use_tools:
            request.update(
                _tool_request_kwargs(tool, force=self._vision.forced_tool_choice)
            )
        response, via_tools = await self._create_chat(
            "vision", request, lambda: messages(False)
        )
        if via_tools:
            if not _returned_tool_call(response):
                # 端点没报错但也没调用工具：多半整个忽略了 tools 参数
                self._disable_tools("vision", "响应中无工具调用")
            else:
                args = _tool_arguments(response, tool["function"]["name"])
                if args is not None:
                    return self._assessment_from_args(args, want_sticker_meta)
        # 回退/纯文本协议：按旧约定从正文解析
        raw = (response.choices[0].message.content or "").strip()
        return self._parse_assessment(raw, want_sticker_meta)

    @staticmethod
    def _assessment_from_args(
        args: dict, want_sticker_meta: bool = False
    ) -> ImageAssessment:
        summary = args.get("summary")
        summary = str(summary).strip()[:200] if isinstance(summary, str) else None
        is_sticker = args.get("sticker") is True
        # keep 只认显式 false 为放弃展示：解析歧义一律保守保留原图；
        # sticker 反过来只认显式 true：歧义一律不收集表情包
        return ImageAssessment(
            summary=summary or None,
            keep_raw=args.get("keep") is not False,
            is_sticker=is_sticker,
            sticker_assessment=(
                _sticker_assessment_from_args(args, "sticker_description")
                if want_sticker_meta and is_sticker
                else None
            ),
        )

    @staticmethod
    def _parse_assessment(raw: str, want_sticker_meta: bool = False) -> ImageAssessment:
        visible = _strip_think(raw)
        match = _ASSESS_RE.search(visible)
        if match:
            try:
                obj = json.loads(match.group(0))
            except (ValueError, TypeError):
                obj = None
            if isinstance(obj, dict):
                summary = obj.get("summary")
                summary = str(summary).strip()[:200] if isinstance(summary, str) else None
                is_sticker = obj.get("sticker") is True
                # keep 只认显式 false 为放弃展示：解析歧义一律保守保留原图；
                # sticker 只认显式 true：歧义一律不收集
                return ImageAssessment(
                    summary=summary or None,
                    keep_raw=obj.get("keep") is not False,
                    is_sticker=is_sticker,
                    sticker_assessment=(
                        _sticker_assessment_from_args(obj, "sticker_description")
                        if want_sticker_meta and is_sticker
                        else None
                    ),
                )
        logger.warning("图片入库评估输出无法解析：%r", raw[:200])
        return ImageAssessment(summary=None, keep_raw=True)

    async def assess_sticker(self, data_url: str) -> StickerAssessment | None:
        """表情包收藏审核（任务 2）：一次 vision 调用产出「可否收藏 + 描述 + 情绪」。

        用于收集路径上没有现成入库评估可合并的场合（placeholder/describe
        模式）。未配置 vision 模型返回 None（收集侧维持现状、无 meta）；
        网络/端点异常原样上抛，由收集链路记 WARNING 后同样退回无 meta 收藏；
        输出无法解析时记 WARNING 返回 None。
        """
        if not self._vision:
            return None
        use_tools = self._tools_on["vision"]
        logger.debug(
            "[vision·sticker-assess] model=%s 图片 %d 字符",
            self._vision.model,
            len(data_url),
        )

        def messages(via_tool: bool) -> list[Message]:
            text = (
                "这是一张群聊里的表情包候选图。请审核它是否适合收藏作斗图用途，"
                "并给出内容与情绪描述。审核标准与描述要求："
                + _STICKER_MODERATION_RULES
                + "\n"
            )
            if via_tool:
                text += "完成后调用 submit_sticker_assessment 工具提交。"
            else:
                text += (
                    '只输出一个 JSON 对象：{"acceptable": true 或 false,'
                    ' "description": "…", "emotion": "…"}'
                )
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ]

        request: dict = dict(
            model=self._vision.model,
            messages=messages(use_tools),
            temperature=self._generation.assess_temperature,
            max_tokens=self._vision.max_output_tokens or _ASSESS_MAX_TOKENS,
            timeout=self._generation.timeout_seconds,
        )
        if use_tools:
            request.update(
                _tool_request_kwargs(
                    _STICKER_ASSESS_TOOL, force=self._vision.forced_tool_choice
                )
            )
        response, via_tools = await self._create_chat(
            "vision", request, lambda: messages(False)
        )
        if via_tools:
            if not _returned_tool_call(response):
                # 端点没报错但也没调用工具：多半整个忽略了 tools 参数
                self._disable_tools("vision", "响应中无工具调用")
            else:
                args = _tool_arguments(response, _STICKER_ASSESS_TOOL["function"]["name"])
                if args is not None:
                    assessment = _sticker_assessment_from_args(args, "description")
                    if assessment is not None:
                        return assessment
                    logger.warning("表情包审核输出缺少可用结论：%r", args)
                    return None
        # 回退/纯文本协议：按旧约定从正文解析
        raw = (response.choices[0].message.content or "").strip()
        match = _ASSESS_RE.search(_strip_think(raw))
        obj = None
        if match:
            try:
                parsed = json.loads(match.group(0))
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, dict):
                obj = _sticker_assessment_from_args(parsed, "description")
        if obj is None:
            logger.warning("表情包审核输出无法解析：%r", raw[:200])
        return obj

    async def pick_sticker(
        self, context_text: str, candidates: Sequence[tuple[str, str]]
    ) -> tuple[int | None, str]:
        """smart 跟发选图（learning 角色）：按当前语境从候选里挑一张，可以不发。

        candidates 为 (描述, 情绪) 列表，提示词里从 1 编号；返回
        （0 基候选下标, 一句话理由），下标 None＝模型明确选择不发。
        调用异常、输出解析不出、编号越界一律抛 ValueError——调用方
        （stickers.StickerStore）统一按「选图失败」退回随机抽选，必须与
        模型显式作罢区分开，绝不能把解析歧义当成「不发」。
        """
        use_tools = self._tools_on["sticker_pick"]
        cfg = self._learning
        entries = "\n".join(
            f"{number}. {description}【{emotion}】" if emotion else f"{number}. {description}"
            for number, (description, emotion) in enumerate(candidates, start=1)
        )
        logger.debug(
            "[sticker-pick] model=%s 候选 %d 张", cfg.model, len(candidates)
        )

        def messages(via_tool: bool) -> list[Message]:
            return [
                {
                    "role": "user",
                    "content": sticker_pick_prompt(
                        context_text, entries, via_tool=via_tool
                    ),
                }
            ]

        request: dict = dict(
            model=cfg.model,
            messages=messages(use_tools),
            temperature=self._generation.learning_temperature,
            max_tokens=cfg.max_output_tokens or _REVIEW_MAX_TOKENS,
            timeout=self._generation.timeout_seconds,
        )
        if use_tools:
            request.update(_tool_request_kwargs(_STICKER_PICK_TOOL, force=cfg.forced_tool_choice))
        response, via_tools = await self._create_chat(
            "sticker_pick", request, lambda: messages(False)
        )
        if via_tools:
            if not _returned_tool_call(response):
                # 端点没报错但也没调用工具：多半整个忽略了 tools 参数
                self._disable_tools("sticker_pick", "响应中无工具调用")
            else:
                args = _tool_arguments(response, _STICKER_PICK_TOOL["function"]["name"])
                if args is not None:
                    return self._parse_sticker_pick(args, len(candidates))
        # 回退/纯文本协议：从正文解析 JSON 契约
        raw = _strip_think((response.choices[0].message.content or "").strip())
        parsed = _extract_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"选图输出无法解析：{raw[:200]!r}")
        return self._parse_sticker_pick(parsed, len(candidates))

    @staticmethod
    def _parse_sticker_pick(args: dict, num_candidates: int) -> tuple[int | None, str]:
        raw_pick = args.get("pick")
        reason = str(args.get("reason") or "").strip()[:100]
        if raw_pick is None or isinstance(raw_pick, bool):
            raise ValueError(f"选图输出缺少合法 pick：{args!r}")
        try:
            pick = int(raw_pick)
        except (TypeError, ValueError):
            raise ValueError(f"选图输出 pick 不是编号：{raw_pick!r}") from None
        if pick == 0:
            return None, reason
        if not 1 <= pick <= num_candidates:
            raise ValueError(f"选图编号越界：{pick}（候选 {num_candidates} 张）")
        return pick - 1, reason

    # -------------------------------------------------------------- learning

    async def _learning_call(self, prompt: str, *, default_max_tokens: int) -> str:
        """一次性纯文本学习调用（不带 tools）。网络/端点异常原样上抛，
        由上层学习任务记 warning 并跳过本次——后台任务容忍失败。"""
        cfg = self._learning
        response = await self._client_for(cfg).chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._generation.learning_temperature,
            max_tokens=cfg.max_output_tokens or default_max_tokens,
            timeout=self._generation.timeout_seconds,
        )
        return _strip_think((response.choices[0].message.content or "").strip())

    async def summarize_impression(
        self, day: str, chat_text: str, max_chars: int
    ) -> str | None:
        """任务 A：把某群某天的聊天记录总结成一段「今日群聊印象」。"""
        text = await self._learning_call(
            impression_summary_prompt(day, chat_text, max_chars),
            default_max_tokens=_IMPRESSION_MAX_TOKENS,
        )
        return text or None

    async def learn_expressions(self, chat_text: str) -> list[tuple[str, str]]:
        """任务 B：从被淘汰的聊天记录中提取「(情境, 风格)」表达规律。

        情境与风格各裁剪到 20 字（提示词已要求，此处兜底），最多取 10 条。
        """
        text = await self._learning_call(
            expression_learning_prompt(chat_text), default_max_tokens=_LEARN_MAX_TOKENS
        )
        parsed = _extract_json(text)
        if not isinstance(parsed, list):
            if text:
                logger.warning("表达学习输出无法解析：%r", text[:200])
            return []
        out: list[tuple[str, str]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            situation = str(item.get("situation") or "").strip()
            style = str(item.get("style") or "").strip()
            if not situation or not style:
                continue
            out.append((situation[:20], style[:20]))
        return out[:10]

    async def review_expression(self, situation: str, style: str) -> bool:
        """任务 B 可选自审：只有明确 suitable=true 才通过（歧义一律拒收）。"""
        text = await self._learning_call(
            expression_evaluation_prompt(situation, style),
            default_max_tokens=_REVIEW_MAX_TOKENS,
        )
        parsed = _extract_json(text)
        if isinstance(parsed, dict):
            return parsed.get("suitable") is True
        logger.warning("表达自审输出无法解析：%r", text[:200])
        return False

    async def learn_person_facts(self, chat_text: str) -> list[tuple[str, str]]:
        """任务 3：从被淘汰的聊天记录中抽取关于群友的稳定事实。

        返回 (该事实关于谁的昵称, fact) 列表。提示词硬性要求宁缺勿错；
        解析层再兜一道：昵称与事实都非空才收，**超长的整条丢弃**——
        fact 截半会产出误导画像的残句，昵称截断还可能撞上另一个人的
        30 字前缀造成误挂（宁缺勿错），故一律不裁剪（上限均取 30 字，
        fact 提示词已要求，昵称按 QQ 群名片上限量级）。最多取 10 条。
        学不到（输出不可解析/空数组）返回空列表。
        """
        text = await self._learning_call(
            person_learning_prompt(chat_text), default_max_tokens=_LEARN_MAX_TOKENS
        )
        parsed = _extract_json(text)
        if not isinstance(parsed, list):
            if text:
                logger.warning("人物事实学习输出无法解析：%r", text[:200])
            return []
        out: list[tuple[str, str]] = []
        dropped = 0
        for item in parsed:
            if not isinstance(item, dict):
                continue
            nickname = str(item.get("nickname") or "").strip()
            fact = str(item.get("fact") or "").strip()
            if not nickname or not fact:
                continue
            # 超长整条丢弃，不裁剪（见本方法 docstring：宁缺勿错）
            if len(nickname) > 30 or len(fact) > 30:
                dropped += 1
                continue
            out.append((nickname, fact))
        if dropped:
            logger.debug("人物事实学习：丢弃超长候选 %d 条", dropped)
        return out[:10]

    async def review_person_fact(self, nickname: str, fact: str) -> bool:
        """任务 3 可选自审（person_self_review，默认关）：只有明确
        suitable=true 才通过（歧义一律拒收，与表达自审同口径）。"""
        text = await self._learning_call(
            person_fact_evaluation_prompt(nickname, fact),
            default_max_tokens=_REVIEW_MAX_TOKENS,
        )
        parsed = _extract_json(text)
        if isinstance(parsed, dict):
            return parsed.get("suitable") is True
        logger.warning("人物事实自审输出无法解析：%r", text[:200])
        return False

    async def extract_jargon_terms(self, chat_text: str) -> list[str]:
        """任务 C 第一步：从聊天流中提取黑话候选词条（去重、限长、最多 10 个）。"""
        text = await self._learning_call(
            jargon_extraction_prompt(chat_text), default_max_tokens=_LEARN_MAX_TOKENS
        )
        parsed = _extract_json(text)
        if not isinstance(parsed, list):
            if text:
                logger.warning("黑话提取输出无法解析：%r", text[:200])
            return []
        terms: list[str] = []
        seen: set[str] = set()
        for item in parsed:
            if not isinstance(item, dict):
                continue
            term = str(item.get("content") or "").strip()
            if not term or len(term) > 16 or term in seen:
                continue
            terms.append(term)
            seen.add(term)
        return terms[:10]

    async def infer_jargon_with_context(
        self, term: str, context_text: str
    ) -> tuple[str | None, bool]:
        """黑话双路推断·第一路（带上下文）。返回（含义, 信息不足标记）。"""
        text = await self._learning_call(
            jargon_inference_with_context_prompt(term, context_text),
            default_max_tokens=_LEARN_MAX_TOKENS,
        )
        parsed = _extract_json(text)
        if not isinstance(parsed, dict):
            logger.warning("黑话 %r 带上下文推断输出无法解析：%r", term, text[:200])
            return None, True
        meaning = str(parsed.get("meaning") or "").strip()
        return (meaning or None), parsed.get("no_info") is True

    async def infer_jargon_alone(self, term: str) -> str | None:
        """黑话双路推断·第二路（仅词条本身）。"""
        text = await self._learning_call(
            jargon_inference_alone_prompt(term), default_max_tokens=_LEARN_MAX_TOKENS
        )
        parsed = _extract_json(text)
        if not isinstance(parsed, dict):
            logger.warning("黑话 %r 仅词条推断输出无法解析：%r", term, text[:200])
            return None
        meaning = str(parsed.get("meaning") or "").strip()
        return meaning or None

    async def compare_jargon_inference(
        self, meaning_with_context: str, meaning_alone: str
    ) -> bool:
        """黑话双路一致性判定：两次推断一致才认为「真的理解」（歧义按不一致）。"""
        text = await self._learning_call(
            jargon_compare_inference_prompt(meaning_with_context, meaning_alone),
            default_max_tokens=_REVIEW_MAX_TOKENS,
        )
        parsed = _extract_json(text)
        if isinstance(parsed, dict):
            return parsed.get("is_similar") is True
        logger.warning("黑话双路比对输出无法解析：%r", text[:200])
        return False


def _sticker_assessment_from_args(
    args: dict, description_key: str
) -> StickerAssessment | None:
    """从审核类调用的参数对象解析 StickerAssessment；无可用结论返回 None。

    acceptable 只认显式布尔；审核通过（true）却给不出非空描述时视为无结论
    （smart 候选以描述为生命线，宁缺毋滥），退回收集端按无 meta 处理。
    acceptable=false 的拒绝结论即使描述为空也有意义（不收藏），照常返回。
    """
    acceptable = args.get("acceptable")
    if not isinstance(acceptable, bool):
        return None
    description = str(args.get(description_key) or "").strip()[:40]
    emotion = str(args.get("emotion") or "").strip()[:20]
    if acceptable and not description:
        return None
    return StickerAssessment(
        acceptable=acceptable, description=description, emotion=emotion
    )


def _extract_json(text: str) -> Any | None:
    """从学习类调用的正文里解析 JSON 对象/数组。

    容忍模型在 JSON 前后附带解释文字或 markdown 代码块围栏；解析不出
    返回 None（调用方按「本次学习无产出」处理，绝不抛业务异常）。
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*[ \t]*\n?", "", stripped)
        stripped = re.sub(r"\n?```\s*$", "", stripped)
    try:
        return json.loads(stripped)
    except ValueError:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = stripped.find(open_ch)
        end = stripped.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except ValueError:
                continue
    return None


def _strip_think(text: str) -> str:
    """删除 <think> 思考段（含未闭合截断的），返回剩余正文。"""
    return _THINK_RE.sub("", text).strip()


def _verdict_from_json(text: str) -> JudgeVerdict | None:
    """从整段文本或其中嵌入的 {"score": …} 对象解析判定，失败返回 None。"""
    candidates = [text]
    match = _SCORE_RE.search(text)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and "score" in obj:
            # to_me 只认 JSON 布尔：字符串/数字等脏数据一律视为 False，
            # 绝不因解析歧义误判「在和我说话」
            to_me = obj.get("to_me") is True
            return JudgeVerdict(
                score=max(0, min(10, int(obj["score"]))),
                to_me=to_me,
                reason=str(obj.get("reason", ""))[:100],
            )
    return None


def _strip_noise(text: str) -> str:
    """去掉模型偶尔加的引号包裹和 <think> 段落。"""
    text = _strip_think(text)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'“”":
        text = text[1:-1].strip()
    return text


def _cap_emojis(text: str, limit: int) -> str:
    """把 emoji 序列的数量压到 limit 以内，保留最先出现的那些。

    超额的序列连同紧随其后的空白一起删除，避免中文里残留双空格；
    limit <= 0 时全部剔除。删完为空由调用方按"无话可说"处理。
    """
    dropped: list[tuple[int, int]] = []
    kept = 0
    for match in _EMOJI_RE.finditer(text):
        if kept < limit:
            kept += 1
            continue
        start, end = match.span()
        # 键帽序列里的数字本身不是 emoji，只删修饰部分
        if match.group()[0] in "123456789#*":
            start += 1
        while end < len(text) and text[end].isspace():
            end += 1
        dropped.append((start, end))
    if not dropped:
        return text
    pieces: list[str] = []
    cursor = 0
    for start, end in dropped:
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces).strip()
