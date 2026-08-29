"""任务 A（临时随机风格）与任务 B（AI 味拦截重生成）的 AIClient 行为单测。

LLM 端点全部 mock（纯文本协议 + 脚本化响应），随机源替换为固定种子的
确定性实现（tests.deterministic_rng），验证概率行为与重试链路可复现。
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

import candybot.ai as ai_mod
from candybot.ai import AIClient
from candybot.aiflavor import detect_ai_flavor
from candybot.models import (
    AI_FLAVOR_RULES_DEFAULT,
    MULTIPLE_REPLY_STYLE_DEFAULT,
    ChatRecord,
    GenerationSettings,
    ModelConfig,
    ModelSettings,
)
from tests.deterministic_rng import SeededRng


def _gen(**over) -> GenerationSettings:
    values = dict(
        reply_max_tokens=500,
        temperature=0.8,
        max_context_chars=8000,
        timeout_seconds=60,
    )
    values.update(over)
    return GenerationSettings(**values)


def _models() -> ModelSettings:
    # reply 走纯文本协议：脚本化端点只回正文，请求不带 tools，便于逐次记录
    return ModelSettings(
        judge=ModelConfig("j", "https://judge.example.com/v1", "kj", None, None),
        reply=ModelConfig(
            "r", "https://reply.example.com/v1", "kr", None, None, tool_use=False
        ),
        vision=None,
    )


def _record(text: str = "在吗") -> ChatRecord:
    return ChatRecord(
        message_id=1, group_id=2, user_id=3, nickname="群友", text=text, ts=0
    )


def _install_scripted(monkeypatch, replies: list[str]) -> list[dict]:
    """按脚本逐条返回正文响应；记录每次 create 的请求参数。"""
    calls: list[dict] = []

    class _Scripted:
        def __init__(self, *, base_url=None, api_key=None):
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            calls.append(kwargs)
            text = replies[min(len(calls) - 1, len(replies) - 1)]
            message = SimpleNamespace(content=text, tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(ai_mod, "AsyncOpenAI", _Scripted)
    return calls


async def _generate(ai: AIClient) -> object:
    current = _record()
    return await ai.generate_reply(
        "L1", "L2", [current], current, "2026-08-29 10:00:00", forced=True
    )


# ---------------------------------------------------------------- AI 味检测规则


@pytest.mark.parametrize(
    "text",
    [
        "作为一个AI，我想说",
        "作为AI语言模型，我没有感受",
        "很乐意帮您解决这个问题",
        "以下是我的分析：",
        "先说结论\n- 第一条要点",
        "**重点**在这里",
        "开头正常\n## 然后冒出个 markdown 标题",
    ],
)
def test_default_rules_hit(text: str):
    reason = detect_ai_flavor(text, AI_FLAVOR_RULES_DEFAULT)
    assert reason is not None and "命中规则" in reason


@pytest.mark.parametrize(
    "text",
    [
        "哈哈确实",
        "这个 bug 修好了",
        "我作为路人看了一眼",  # 「作为」≠「作为AI」
        "用 - 连接的两个词",  # 连字符不在行首
        "下面说两句",
        "",
    ],
)
def test_default_rules_pass(text: str):
    assert detect_ai_flavor(text, AI_FLAVOR_RULES_DEFAULT) is None


def test_empty_rules_never_hit():
    assert detect_ai_flavor("作为AI，很高兴为您服务", ()) is None


# ---------------------------------------------------------------- 任务 A：临时风格


class _BoomRng:
    """关闭路径不得消耗随机数：碰一下掷点即失败。"""

    def random(self) -> float:
        raise AssertionError("关闭时不应掷点")

    def choice(self, seq):
        raise AssertionError("关闭时不应抽风格")


def test_pick_style_disabled_short_circuits(monkeypatch):
    monkeypatch.setattr(ai_mod, "_RNG", _BoomRng())
    # 概率 0（默认）与池为空：都不掷点，直接 None
    assert AIClient(models=_models(), generation=_gen())._pick_temporary_style() is None
    ai = AIClient(
        models=_models(),
        generation=_gen(multiple_probability=1.0, multiple_reply_style=()),
    )
    assert ai._pick_temporary_style() is None


def test_pick_style_seeded_probability_behaviour(monkeypatch):
    """seed 0 的首个 random() ≈ 0.0782：0.05 不中、0.1 命中；同种子结果确定。"""
    monkeypatch.setattr(ai_mod, "_RNG", SeededRng(0))
    miss = AIClient(models=_models(), generation=_gen(multiple_probability=0.05))
    assert miss._pick_temporary_style() is None

    monkeypatch.setattr(ai_mod, "_RNG", SeededRng(0))
    hit = AIClient(models=_models(), generation=_gen(multiple_probability=0.1))
    assert hit._pick_temporary_style() in MULTIPLE_REPLY_STYLE_DEFAULT

    styles = []
    for _ in range(2):
        monkeypatch.setattr(ai_mod, "_RNG", SeededRng(42))
        ai = AIClient(models=_models(), generation=_gen(multiple_probability=1.0))
        styles.append(ai._pick_temporary_style())
    assert styles[0] is not None and styles[0] == styles[1]


async def test_generate_reply_injects_style_into_l4(monkeypatch):
    monkeypatch.setattr(ai_mod, "_RNG", SeededRng(1))
    calls = _install_scripted(monkeypatch, ["好嘞"])
    ai = AIClient(models=_models(), generation=_gen(multiple_probability=1.0))
    draft = await _generate(ai)
    assert draft is not None and draft.text == "好嘞"
    content = calls[0]["messages"][-1]["content"]
    assert "【临时风格】本次回复请遵循这个额外风格：" in content
    assert any(style in content for style in MULTIPLE_REPLY_STYLE_DEFAULT)


async def test_generate_reply_no_style_by_default(monkeypatch):
    """multiple_probability 默认 0：L4 不出现临时风格块（向后兼容）。"""
    calls = _install_scripted(monkeypatch, ["好嘞"])
    ai = AIClient(models=_models(), generation=_gen())
    await _generate(ai)
    assert "临时风格" not in calls[0]["messages"][-1]["content"]


# ---------------------------------------------------------------- 任务 B：拦截重生成


async def test_retry_rewrites_with_constraint(monkeypatch):
    """命中 → L4 附加拦截原因重生成一次，产出重写后的回复。"""
    calls = _install_scripted(monkeypatch, ["作为AI，我很乐意帮您", "好嘞"])
    ai = AIClient(models=_models(), generation=_gen())
    draft = await _generate(ai)
    assert draft is not None and draft.text == "好嘞"
    assert len(calls) == 2
    first = calls[0]["messages"][-1]["content"]
    second = calls[1]["messages"][-1]["content"]
    assert "被拦截" not in first
    assert "被拦截" in second
    assert "作为AI，我很乐意帮您" in second  # 被拦截回复原文转述进约束
    assert "命中规则" in second and "更口语" in second


async def test_retry_still_hits_passes_with_warning(monkeypatch, caplog):
    """重试仍命中：放行并记 warning，绝不死循环。"""
    with caplog.at_level(logging.INFO, logger="candybot.ai"):
        calls = _install_scripted(
            monkeypatch, ["作为AI，我很乐意帮您", "以下是重写结果"]
        )
        ai = AIClient(models=_models(), generation=_gen())
        draft = await _generate(ai)
    assert draft is not None and draft.text == "以下是重写结果"
    assert len(calls) == 2  # 只重试一次（默认 ai_flavor_retries=1）
    assert "AI 味拦截" in caplog.text  # INFO：拦截原因与重试动作
    assert "仍命中" in caplog.text and "放行" in caplog.text  # WARNING


async def test_retries_zero_skips_detection(monkeypatch):
    calls = _install_scripted(monkeypatch, ["作为AI，我很乐意帮您"])
    ai = AIClient(models=_models(), generation=_gen(ai_flavor_retries=0))
    draft = await _generate(ai)
    assert draft.text == "作为AI，我很乐意帮您"
    assert len(calls) == 1


async def test_rules_empty_skips_detection(monkeypatch):
    calls = _install_scripted(monkeypatch, ["很高兴为您服务"])
    ai = AIClient(models=_models(), generation=_gen(ai_flavor_rules=()))
    draft = await _generate(ai)
    assert draft.text == "很高兴为您服务"
    assert len(calls) == 1


async def test_retry_empty_result_returns_none(monkeypatch):
    """重写后模型决定不说了（空正文）：按「无话可说」返回 None。"""
    calls = _install_scripted(monkeypatch, ["作为AI，我很乐意帮您", ""])
    ai = AIClient(models=_models(), generation=_gen())
    assert await _generate(ai) is None
    assert len(calls) == 2


async def test_clean_reply_makes_single_call(monkeypatch):
    """默认规则下正常口语回复不触发任何重试。"""
    calls = _install_scripted(monkeypatch, ["哈哈确实"])
    ai = AIClient(models=_models(), generation=_gen())
    draft = await _generate(ai)
    assert draft.text == "哈哈确实"
    assert len(calls) == 1


async def test_reconsider_reply_not_content_retried(monkeypatch):
    """AI 味重试只覆盖普通回复生成；连发重想不走这一环节。"""
    calls = _install_scripted(monkeypatch, ["以下是还要发的话"])
    ai = AIClient(models=_models(), generation=_gen())
    draft = await ai.reconsider_reply(
        "L1",
        "L2",
        [_record()],
        "now",
        sent_segments=[],
        pending_segments=["腹稿"],
    )
    assert draft.text == "以下是还要发的话"
    assert len(calls) == 1
