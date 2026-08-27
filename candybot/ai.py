"""OpenAI 兼容 API 的三个调用角色：judge（打分）、reply（回复）、vision（转述）。"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass

from openai import AsyncOpenAI

from .models import ChatRecord, GenerationSettings
from .prompts import (
    Message,
    final_user_prompt_judge,
    final_user_prompt_judge_recheck,
    final_user_prompt_reply,
    history_to_turns,
    reply_history_turns,
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

# 基础 emoji 字符范围。刻意不含几何符号（U+25A0~25FF）等普通文本符号区，
# 颜文字 (=^ω^=)、(・ω・) 与箭头 ↑ 不会误伤。
_EMOJI_CHARS = (
    "\U0001F000-\U0001FAFF"  # 象形符号/表情/补充表情各区
    "\u2600-\u27BF"          # 杂项符号与装饰符（☀ ✅ ❤）
    "\u2B00-\u2BFF"          # 星形与常见聊天箭头（⭐ ⭕ ⬛）
)
# emoji 序列：国旗按成对区域指示符算一个，键帽含数字本身，
# 其余为单个基础字符 + 变体选择符/肤色修饰 + ZWJ 组合的完整序列。
_EMOJI_RE = re.compile(
    "[\U0001F1E6-\U0001F1FF]{2}"
    "|[1-9#*]\uFE0F?\u20E3"
    f"|[{_EMOJI_CHARS}](?:[\uFE0F\U0001F3FB-\U0001F3FF]|\u200D[{_EMOJI_CHARS}])*"
)


@dataclass(frozen=True)
class JudgeVerdict:
    score: int
    reason: str
    to_me: bool = False  # 这条消息是否在对我说、延续与我相关的对话


@dataclass(frozen=True)
class ImageAssessment:
    """收图入库时视觉模型的判定：总结文本 + 是否继续向模型展示原图。"""

    summary: str | None
    keep_raw: bool


@dataclass(frozen=True)
class ImageOp:
    """回复模型给出的图片生命周期操作（动作名, 目标消息编号）。"""

    action: str
    message_id: int


_IMAGE_HEAD = 48  # debug 日志中图片 data URL 只保留头部字符数


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


class AIClient:
    """封装三种 LLM 调用。所有方法都可能抛出 openai 库异常，由上层决定重试。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        judge_model: str,
        reply_model: str,
        vision_model: str | None,
        generation: GenerationSettings,
        multimodal_mode: str = "placeholder",
    ):
        self._client = AsyncOpenAI(base_url=base_url or None, api_key=api_key)
        self._judge_model = judge_model
        self._reply_model = reply_model
        self._vision_model = vision_model
        self._generation = generation
        # 只有 direct 模式才允许任何图片（历史层或当前消息）进入请求
        self._multimodal_mode = multimodal_mode

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
        直接复用。
        """
        # 历史层不含当前消息本身（它单独出现在指令层，保证历史层前缀稳定）
        turns, _ = history_to_turns(recent_records[:-1], self._generation.max_context_chars)
        history_messages = [{"role": t.role, "content": t.content} for t in turns]
        if prev_verdict is not None:
            assert threshold is not None, "复核模式必须提供 threshold 以告知门槛"
            assert min_score is not None, "复核模式必须提供 min_score 以告知触发下限"
            final_user = final_user_prompt_judge_recheck(
                now_text,
                current_message,
                prev_score=prev_verdict.score,
                prev_reason=prev_verdict.reason,
                threshold=threshold,
                min_score=min_score,
            )
            tag = "[judge·复核]"
        else:
            final_user = final_user_prompt_judge(now_text, current_message)
            tag = "[judge]"
        chat: list[Message] = [
            {"role": "system", "content": static_system},
            {"role": "system", "content": runtime_system},
            *history_messages,
            {"role": "user", "content": final_user},
        ]
        logger.debug(
            "%s model=%s 消息数=%d\n%s",
            tag,
            self._judge_model,
            len(chat),
            format_messages_for_log(chat),
        )
        response = await self._client.chat.completions.create(
            model=self._judge_model,
            messages=chat,
            temperature=0.2,
            # 推理型 judge 模型的思考段会占掉不少 token，上限太小会把结论截掉
            max_tokens=1000,
            timeout=self._generation.timeout_seconds,
        )
        raw = (response.choices[0].message.content or "").strip()
        return self._parse_verdict(raw)

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
        current_message: ChatRecord,
        now_text: str,
        *,
        forced: bool,
        engaged: bool = False,
        score: int | None = None,
        reason: str = "",
    ) -> str | None:
        """回复模型生成一句群聊回应；direct 模式下历史与当前消息可携带图片块。"""
        if self._multimodal_mode == "direct":
            turns, _ = reply_history_turns(
                recent_records[:-1],
                self._generation.max_context_chars,
                self._generation.max_history_images,
            )
            history_messages = [_turn_to_message(t) for t in turns]
        else:
            turns, _ = history_to_turns(recent_records[:-1], self._generation.max_context_chars)
            history_messages = [{"role": t.role, "content": t.content} for t in turns]
        text_part = final_user_prompt_reply(
            now_text, current_message, forced=forced, engaged=engaged, score=score, reason=reason
        )
        final_user: str | list[dict]
        if self._multimodal_mode == "direct" and current_message.images:
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
            self._reply_model,
            len(chat),
            format_messages_for_log(chat),
        )
        response = await self._client.chat.completions.create(
            model=self._reply_model,
            messages=chat,
            temperature=self._generation.temperature,
            max_tokens=self._generation.reply_max_tokens,
            timeout=self._generation.timeout_seconds,
        )
        reply = (response.choices[0].message.content or "").strip()
        reply = _strip_noise(reply)
        roll_ok = random.random() < self._generation.emoji_chance
        reply = _cap_emojis(reply, self._generation.emoji_max if roll_ok else 0)
        return reply or None

    # --------------------------------------------------------------- vision

    async def describe_image(self, data_url: str) -> str | None:
        """视觉模型把图片转成一句话描述；未配置 vision 模型时返回 None。"""
        if not self._vision_model:
            return None
        logger.debug(
            "[vision] model=%s 图片 %d 字符",
            self._vision_model,
            len(data_url),
        )
        response = await self._client.chat.completions.create(
            model=self._vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请用不超过40个字描述这张图片的内容要点。"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            temperature=0.3,
            max_tokens=80,
            timeout=self._generation.timeout_seconds,
        )
        desc = (response.choices[0].message.content or "").strip()
        return desc or None

    async def assess_image(self, data_url: str) -> ImageAssessment:
        """视觉模型一次性完成「总结 + 是否值得长期保留原图」的入库判定。

        未配置 vision 模型或调用/解析失败时安全侧返回保留原图：宁可
        多花些 token 也不凭空丢信息，后续仍可通过 <drop_img> 标记降级。
        """
        if not self._vision_model:
            return ImageAssessment(summary=None, keep_raw=True)
        logger.debug(
            "[vision·assess] model=%s 图片 %d 字符",
            self._vision_model,
            len(data_url),
        )
        response = await self._client.chat.completions.create(
            model=self._vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "这是一张群聊里发来的图片。请先用不超过40个字总结它的内容要点，"
                                "再判断后续对话是否还需要继续查看这张原图：只有当图片包含未来"
                                "可能被反复引用的具体信息（文字截图、代码、表格、关键画面细节等）"
                                "才值得保留原图；表情包、梗图之类的总结即可。\n"
                                '只输出一个 JSON 对象：{"summary": "一句话总结",'
                                ' "keep": true 或 false}'
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            temperature=0.2,
            max_tokens=400,
            timeout=self._generation.timeout_seconds,
        )
        raw = (response.choices[0].message.content or "").strip()
        return self._parse_assessment(raw)

    @staticmethod
    def _parse_assessment(raw: str) -> ImageAssessment:
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
                # keep 只认显式 false 为放弃展示：解析歧义一律保守保留原图
                return ImageAssessment(
                    summary=summary or None, keep_raw=obj.get("keep") is not False
                )
        logger.warning("图片入库评估输出无法解析：%r", raw[:200])
        return ImageAssessment(summary=None, keep_raw=True)


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
