"""_cap_emojis 的行为：数量上限、组合序列计数与误伤防护。"""

from __future__ import annotations

from candybot.ai import _cap_emojis


def test_remove_all():
    assert _cap_emojis("在啊，怎么了？🍬", 0) == "在啊，怎么了？"


def test_keeps_first_n_and_drops_rest():
    assert _cap_emojis("好🎉耶🎊哇🎈", 1) == "好🎉耶哇"
    assert _cap_emojis("好🎉耶🎊哇🎈", 2) == "好🎉耶🎊哇"
    assert _cap_emojis("好🎉耶🎊哇🎈", 3) == "好🎉耶🎊哇🎈"


def test_within_limit_unchanged():
    text = "今天心情不错😊"
    assert _cap_emojis(text, 2) == text


def test_no_double_space_after_removal():
    assert _cap_emojis("今天 😂 真好玩", 0) == "今天 真好玩"
    assert _cap_emojis("🍬开头带一个", 0) == "开头带一个"


def test_empty_result_after_removal():
    assert _cap_emojis("🍬 ", 0) == ""


def test_kaomoji_and_plain_symbols_untouched():
    text = "(=^ω^=) 来啦～ (・ω・)ノ ↑↑ ●▽● 23333"
    assert _cap_emojis(text, 0) == text


def test_zwj_family_counts_as_one():
    family = "\U0001F468\u200D\U0001F469\u200D\U0001F467\u200D\U0001F466"
    assert _cap_emojis(f"全家福{family}哈哈", 0) == "全家福哈哈"
    assert _cap_emojis(f"a{family}b{family}", 1) == f"a{family}b"


def test_flag_counts_as_one_sequence():
    cn = "\U0001F1E8\U0001F1F3"  # 🇨🇳
    assert _cap_emojis(f"中国{cn}加油{cn}", 1) == f"中国{cn}加油"
    assert _cap_emojis(f"中国{cn}", 0) == "中国"


def test_keycap_keeps_digit():
    assert _cap_emojis("第1️⃣名", 0) == "第1名"


def test_skin_tone_modifier_stays_in_sequence():
    wave = "\U0001F44B\U0001F3FB"  # 👋🏻
    assert _cap_emojis(f"你好呀{wave}", 0) == "你好呀"


def test_variation_selector_and_zwj_chain():
    fire = "\u2764\uFE0F\u200D\U0001F525"  # ❤️‍🔥
    assert _cap_emojis(f"太酷了{fire}", 0) == "太酷了"
