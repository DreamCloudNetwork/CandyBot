"""_parse_verdict 与 _strip_noise 对 <think> 思考段的处理。"""

from __future__ import annotations

from candybot.ai import AIClient, _strip_noise

parse = AIClient._parse_verdict


def test_plain_json():
    v = parse('{"score": 8, "reason": "对方在等我"}')
    assert v.score == 8
    assert v.reason == "对方在等我"
    assert v.to_me is False                # 缺省 to_me 视为非对话延续


def test_to_me_flag_parsing():
    v = parse('{"score": 9, "to_me": true, "reason": "在追问刚才的话"}')
    assert v.score == 9
    assert v.to_me is True
    # 非布尔值按假处理，绝不因脏数据误判为"在和我说话"
    assert parse('{"score": 9, "to_me": "yes"}').to_me is False
    assert parse('{"score": 4, "to_me": false, "reason": "路人话题"}').to_me is False


def test_json_after_closed_think():
    raw = '<think>按锚点该给 10 分？不，先看 3 分的档……还是回吧</think>{"score": 7, "reason": "该接话"}'
    v = parse(raw)
    assert v.score == 7
    assert v.reason == "该接话"


def test_think_only_numbers_never_extracted():
    raw = "<think>1. 有人@我时应当给高分\n2. 结论：给 9 分</think>"
    assert parse(raw).score == 0


def test_truncated_unclosed_think_scores_zero():
    # 日志中的实际案例：max_tokens 太小，思考没写完就被截断，且含有编号列表
    raw = '<think>让我分析一下这条消息：\n\n"糖糖？"\n\n1. 有人@我时应当给高分 - 这是明确的'
    v = parse(raw)
    assert v.score == 0
    assert v.reason != ""


def test_visible_bare_number_after_think():
    raw = "<think>纠结中，感觉分数不低</think>我认为应该回复，8 分。"
    v = parse(raw)
    assert v.score == 8


def test_embedded_json_in_visible_text():
    raw = '好的，结论如下：{"score": 5, "reason": "可回可不回"} 以上。'
    v = parse(raw)
    assert v.score == 5


def test_stray_closing_tag_ignored():
    v = parse('</think>{"score": 6, "reason": "可接话"}')
    assert v.score == 6


def test_score_clamped():
    assert parse('{"score": 99}').score == 10
    assert parse('{"score": -3}').score == 0


def test_reply_strips_unclosed_think_entirely():
    assert _strip_noise("<think>让我想想怎么回比较自然") == ""
    assert _strip_noise('<think>嗯</think>"在啊～"') == "在啊～"
