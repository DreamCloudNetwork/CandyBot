"""输出层拟人化后处理单测：拆条、打字时长、错别字与敷衍兜底。"""

from __future__ import annotations

import pytest

from candybot.models import LAZY_REPLIES_DEFAULT, ResponsePostProcessSettings
from candybot.postprocess import (
    TypoGenerator,
    display_len,
    estimate_typing_time,
    process_reply,
    split_reply_segments,
)
from tests.deterministic_rng import SeededRng

# ---------------------------------------------------------------- 拆条


def test_split_basic_punctuation():
    """按句末标点拆条；默认去掉句末标点，! ?（含全角 ！ ？）按配置保留。"""
    text = "你好呀。今天天气不错！改天一起玩。"
    assert split_reply_segments(text, 3, keep_strong_punctuation=False) == [
        "你好呀",
        "今天天气不错",
        "改天一起玩",
    ]
    assert split_reply_segments(text, 3, keep_strong_punctuation=True) == [
        "你好呀",
        "今天天气不错！",
        "改天一起玩",
    ]
    # 全角 ？ 与 ! ！ ? 一样受开关控制，不再被无条件丢掉
    assert split_reply_segments("你好吗？走！嗯。", 3, keep_strong_punctuation=True) == [
        "你好吗？",
        "走！",
        "嗯",
    ]
    assert split_reply_segments("你好吗？走！", 3, keep_strong_punctuation=False) == [
        "你好吗",
        "走",
    ]


def test_split_removes_bracket_narration():
    """成对的（）/() 旁白被删除；不配对的原样保留。"""
    assert split_reply_segments("好的（内心OS），明天见", 3) == ["好的", "明天见"]
    assert split_reply_segments("半路（括号 不删", 3) == ["半路（括号 不删"]


def test_split_nested_bracket_narration_leaves_no_orphan():
    """嵌套括号旁白内层剥完后剩下的空括号对要一并清掉，
    绝不能把「（）」这类残段原样发到群里。"""
    assert split_reply_segments("嗯嗯（（笑））好", 3) == ["嗯嗯好"]
    assert split_reply_segments("（（哭））", 3) == []
    assert split_reply_segments("哈哈（（（晕）））走了", 3) == ["哈哈走了"]
    # 真内容不受额外影响：仍与旧行为一致，只失去最内一层
    assert split_reply_segments("测试a(b(c)d)e结束", 3) == ["测试a(bd)e结束"]


def test_split_kaomoji_never_broken():
    """颜文字（含半角括号形态）绝不被拆断，也不被括号旁白删除误伤。"""
    segments = split_reply_segments("(=^ω^=)哈哈，(・ω・)要走了", 3)
    assert segments == ["(=^ω^=)哈哈", "(・ω・)要走了"]
    # 整体只有一个颜文字时原样保留
    assert split_reply_segments("(・ω・)", 3) == ["(・ω・)"]


def test_split_emoji_sequence_never_broken():
    """emoji 序列（肤色修饰、ZWJ 组合）作为整体保留，不产生半截代理对。"""
    family = "\U0001F468‍\U0001F469‍\U0001F467"
    segments = split_reply_segments(f"好的👌🏼，走起{family}🤣👉", 3)
    assert segments == ["好的👌🏼", f"走起{family}🤣👉"]


def test_split_merges_overflow_into_last():
    """超过 max_split 时剩余句子并入最后一条（句中标点保留，句末按策略处理）。"""
    segments = split_reply_segments("一。二。三。四。五。", 3, keep_strong_punctuation=False)
    assert segments == ["一", "二", "三。四。五"]
    segments = split_reply_segments("一。二。三。四。五。", 1)
    assert segments == ["一。二。三。四。五"]


def test_split_empty_after_narration_removal():
    assert split_reply_segments("（笑）(叹气)", 3) == []


def test_split_single_word_no_delimiter():
    assert split_reply_segments("嗯", 3) == ["嗯"]


# ---------------------------------------------------------------- 打字时长


def test_typing_time_char_classes():
    assert estimate_typing_time("你好") == pytest.approx(0.6)        # 中文 0.3/字
    assert estimate_typing_time("ok1") == pytest.approx(0.45)        # 英文数字 0.15/字
    assert estimate_typing_time("好的，收到。") == pytest.approx(1.8)  # 全角标点按 0.3


def test_typing_time_single_char_triple():
    """单字符回复按 3 倍计：「嗯」不是秒回的。"""
    assert estimate_typing_time("嗯") == pytest.approx(0.9)
    assert estimate_typing_time("a") == pytest.approx(0.45)
    assert estimate_typing_time("嗯嗯") == pytest.approx(0.6)


def test_typing_time_emoji_and_kaomoji_fixed_one_second():
    assert estimate_typing_time("😊") == pytest.approx(1.0)
    assert estimate_typing_time("(=^ω^=)") == pytest.approx(1.0)
    assert estimate_typing_time("😊😊") == pytest.approx(2.0)
    assert estimate_typing_time("你好😊") == pytest.approx(1.6)


def test_typing_time_speed_multiplier():
    assert estimate_typing_time("你好", 2.0) == pytest.approx(1.2)
    assert estimate_typing_time("你好", 0.0) == 0.0
    assert estimate_typing_time("你好", -1.0) == 0.0


def test_typing_time_capped_and_nonfinite_speed():
    """单条封顶 60 秒，防止高倍率/超长文本把所在群的串行队列堵死；
    异常倍率（NaN 漏网时）按「关闭延迟」处理，绝不喂给 asyncio.sleep。"""
    assert estimate_typing_time("字" * 1000, 100.0) == pytest.approx(60.0)
    assert estimate_typing_time("你好", float("nan")) == 0.0
    assert estimate_typing_time("你好", float("inf")) == pytest.approx(60.0)


# ---------------------------------------------------------------- 显示字数


def test_display_len_counts_each_token_as_one_char():
    """emoji 序列（ZWJ 组合）与颜文字整体按 1 字计，半角字符逐字计。"""
    family = "\U0001F468‍\U0001F469‍\U0001F467"
    assert len(family) == 5 and display_len(family) == 1
    assert display_len(family + "好") == 2
    assert display_len("(=^ω^=)哈哈") == 3


def test_process_reply_max_length_uses_display_len():
    """emoji 密集的回复码点数超上限但显示字数不超：仍发原文不降级。"""
    family = "\U0001F468‍\U0001F469‍\U0001F467"
    text = f"走起{family}哈哈"  # 码点 9 > 5，显示字数 = 2+1+2 = 5
    settings = settings_with(
        max_length=5, typo_error_rate=0.0, typo_word_replace_rate=0.0
    )
    assert process_reply(text, settings, rng=SeededRng(1)).messages == [text]
    # 真的超过显示字数时才走敷衍兜底
    long_settings = settings_with(
        max_length=4, typo_error_rate=0.0, typo_word_replace_rate=0.0
    )
    processed = process_reply(text, long_settings, rng=SeededRng(1))
    assert processed.messages[0] in LAZY_REPLIES_DEFAULT


# ---------------------------------------------------------------- 错别字


def test_typo_disabled_when_rates_zero():
    generator = TypoGenerator(error_rate=0.0, word_replace_rate=0.0)
    assert generator.active is False
    assert generator.create_typo("今天天气不错") == ("今天天气不错", [])


def test_typo_deterministic_with_seed():
    """固定随机种子输出可复现；被替换的字/词都会进入更正列表且来自原文。"""
    text = "这个方案确实不错，哈哈哈"
    a = TypoGenerator(error_rate=1.0, word_replace_rate=1.0, rng=SeededRng(42)).create_typo(text)
    b = TypoGenerator(error_rate=1.0, word_replace_rate=1.0, rng=SeededRng(42)).create_typo(text)
    assert a == b
    typoed, corrections = a
    assert typoed != text  # error_rate=1.0 下几乎必有替换
    assert corrections and all(word in text for word in corrections)


def test_typo_word_replacement():
    """整词替换命中同音真词：一副 → 一服（词库中同拼音的另一条）。"""
    generator = TypoGenerator(error_rate=0.0, word_replace_rate=1.0, rng=SeededRng(1))
    typoed, corrections = generator.create_typo("戴上一副手套")
    assert "一服" in typoed
    assert corrections == ["一副"]


def test_typo_never_invents_missing_chars():
    """非汉字（标点、字母、颜文字占位）绝不参与替换。"""
    generator = TypoGenerator(error_rate=1.0, tone_error_rate=1.0, rng=SeededRng(5))
    typoed, _ = generator.create_typo("(=^ω^=)abc123，。!")
    assert typoed == "(=^ω^=)abc123，。!"


# ---------------------------------------------------------------- 多音字


def test_typo_polyphone_word_reading_uses_context_reading():
    """word_reading：「银行」的行按词内读音 háng 找同音替换，不再产出 xíng 系假同音。"""
    generator = TypoGenerator(
        error_rate=1.0, tone_error_rate=0.0, word_replace_rate=0.0,
        polyphone_mode="word_reading", rng=SeededRng(3),
    )
    typoed, corrections = generator.create_typo("去银行存钱")
    assert "行" not in typoed  # 行 被 háng 的同音字替换
    assert "行" in corrections


def test_typo_polyphone_skip_keeps_ambiguous_chars():
    """skip：多音字整体不动，单读音字照常替换。"""
    generator = TypoGenerator(
        error_rate=1.0, tone_error_rate=0.0, word_replace_rate=0.0,
        polyphone_mode="skip", rng=SeededRng(3),
    )
    typoed, corrections = generator.create_typo("去银行存钱")
    assert "行" in typoed and "行" not in corrections
    assert "钱" not in typoed  # 钱 非多音字，不受 skip 影响


def test_typo_polyphone_standalone_always_skipped():
    """多音字单独成词、读音无从确定：两种模式都绝不替换。"""
    for mode in ("word_reading", "skip"):
        generator = TypoGenerator(
            error_rate=1.0, tone_error_rate=1.0, word_replace_rate=1.0,
            polyphone_mode=mode, rng=SeededRng(7),
        )
        typoed, corrections = generator.create_typo("行")
        assert typoed == "行", mode
        assert corrections == [], mode


def test_process_reply_respects_polyphone_mode():
    """配置贯通：process_reply 把 typo_polyphone_mode 传给错字生成器。"""
    settings = settings_with(
        typo_error_rate=1.0, typo_tone_error_rate=0.0, typo_word_replace_rate=0.0,
        typo_correction_probability=0.0, typo_polyphone_mode="skip",
    )
    processed = process_reply("去银行存钱", settings, rng=SeededRng(3))
    assert "行" in processed.messages[0]


# ---------------------------------------------------------------- process_reply


def settings_with(**over) -> ResponsePostProcessSettings:
    return ResponsePostProcessSettings(**over)


def test_process_reply_disabled_matches_legacy():
    settings = settings_with(enabled=False)
    processed = process_reply("你好。再见。", settings, rng=SeededRng(0))
    assert processed.messages == ["你好。再见。"]
    assert processed.memory_text == "你好。再见。"


def test_process_reply_lazy_fallback_for_long_text():
    """单条超过 max_length 不硬拆，改为从敷衍池随机抽一条。"""
    settings = settings_with(max_length=20, typo_error_rate=0.0, typo_word_replace_rate=0.0)
    processed = process_reply("好" * 30, settings, rng=SeededRng(7))
    assert len(processed.messages) == 1
    assert processed.messages[0] in settings.lazy_replies
    assert processed.memory_text == processed.messages[0]
    # 空池防御：未配置时走内置默认池
    default_settings = settings_with(max_length=1)
    assert process_reply("太" * 200, default_settings).messages[0] in LAZY_REPLIES_DEFAULT


def test_process_reply_lazy_when_only_narration():
    processed = process_reply("（叹气）", settings_with(), rng=SeededRng(1))
    assert len(processed.messages) == 1
    assert processed.messages == [processed.memory_text]
    assert processed.messages[0] in LAZY_REPLIES_DEFAULT


def test_process_reply_memory_text_is_clean_original():
    """错别字与更正不进记忆：memory_segments 是与发送逐条对齐的无错字原文。"""
    settings = settings_with(
        typo_error_rate=1.0, typo_word_replace_rate=0.5, typo_correction_probability=1.0
    )
    processed = process_reply(
        "这个方案确实不错。明天再讨论吧。", settings, rng=SeededRng(3)
    )
    assert processed.memory_text == "这个方案确实不错\n明天再讨论吧"
    # 逐条对齐：与 messages 一一对应、每条不含换行（发送链路按此逐条写回）
    assert processed.memory_segments == ["这个方案确实不错", "明天再讨论吧"]
    assert len(processed.memory_segments) == len(processed.messages)
    assert all("\n" not in seg for seg in processed.memory_segments)
    assert processed.correction is not None and processed.correction.startswith("＊")
    assert processed.correction[1:] in processed.memory_text  # 更正词来自原文


def test_process_reply_correction_probability_zero():
    settings = settings_with(
        typo_error_rate=1.0, typo_word_replace_rate=0.0, typo_correction_probability=0.0
    )
    processed = process_reply("这个方案确实不错。", settings, rng=SeededRng(9))
    assert processed.correction is None


def test_process_reply_deterministic_with_seed():
    settings = settings_with(typo_error_rate=0.3, typo_correction_probability=0.8)
    text = "今天天气真好，想出去玩。你要不要一起？"
    first = process_reply(text, settings, rng=SeededRng(2026))
    second = process_reply(text, settings, rng=SeededRng(2026))
    assert first == second
