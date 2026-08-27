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
    final_user_prompt_reply,
    history_to_turns,
)

logger = logging.getLogger(__name__)

_SCORE_RE = re.compile(r"\{\s*\"score\"[\s\S]*?\}")

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
    ):
        self._client = AsyncOpenAI(base_url=base_url or None, api_key=api_key)
        self._judge_model = judge_model
        self._reply_model = reply_model
        self._vision_model = vision_model
        self._generation = generation

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
    ) -> JudgeVerdict:
        """判断模型评估是否回复该消息；解析失败按 0 分处理并记日志。"""
        # 历史层不含当前消息本身（它单独出现在指令层，保证历史层前缀稳定）
        turns, _ = history_to_turns(recent_records[:-1], self._generation.max_context_chars)
        history_messages = [{"role": t.role, "content": t.content} for t in turns]
        chat: list[Message] = [
            {"role": "system", "content": static_system},
            {"role": "system", "content": runtime_system},
            *history_messages,
            {
                "role": "user",
                "content": final_user_prompt_judge(now_text, current_message, threshold),
            },
        ]
        logger.debug(
            "[judge] model=%s 消息数=%d\n%s",
            self._judge_model,
            len(chat),
            format_messages_for_log(chat),
        )
        response = await self._client.chat.completions.create(
            model=self._judge_model,
            messages=chat,
            temperature=0.2,
            max_tokens=200,
            timeout=self._generation.timeout_seconds,
        )
        raw = (response.choices[0].message.content or "").strip()
        return self._parse_verdict(raw)

    @staticmethod
    def _parse_verdict(raw: str) -> JudgeVerdict:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "score" in obj:
                return JudgeVerdict(
                    score=max(0, min(10, int(obj["score"]))),
                    reason=str(obj.get("reason", ""))[:100],
                )
        except (ValueError, TypeError):
            pass
        match = _SCORE_RE.search(raw)
        if match:
            try:
                obj = json.loads(match.group(0))
                return JudgeVerdict(
                    score=max(0, min(10, int(obj["score"]))),
                    reason=str(obj.get("reason", ""))[:100],
                )
            except (ValueError, TypeError):
                pass
        num = re.search(r"\b(\d|10)\b", raw)
        if num:
            logger.warning("解析到非结构化输出，原文: %s", raw)
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
        score: int | None = None,
        reason: str = "",
    ) -> str | None:
        """回复模型生成一句群聊回应；direct 模式下附带图片内容块。"""
        turns, _ = history_to_turns(recent_records[:-1], self._generation.max_context_chars)
        text_part = final_user_prompt_reply(
            now_text, current_message, forced=forced, score=score, reason=reason
        )
        final_user: str | list[dict]
        if current_message.images:
            blocks: list[dict] = [{"type": "text", "text": text_part}]
            for data_url in current_message.images:
                blocks.append({"type": "image_url", "image_url": {"url": data_url}})
            final_user = blocks
        else:
            final_user = text_part

        history_messages = [{"role": t.role, "content": t.content} for t in turns]
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


def _strip_noise(text: str) -> str:
    """去掉模型偶尔加的引号包裹和 <think> 段落。"""
    if "<think>" in text:
        text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
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
