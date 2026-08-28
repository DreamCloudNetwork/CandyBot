from __future__ import annotations

import time

from candybot.models import ChatRecord
from candybot.prompts import (
    build_messages,
    final_user_prompt_judge,
    final_user_prompt_judge_recheck,
    final_user_prompt_reply,
    history_to_turns,
    nickname_list_from_history,
    record_to_turn,
    runtime_system_prompt,
    static_system_prompt,
)


def rec(mid: int, uid: int, nick: str, text: str, is_self=False) -> ChatRecord:
    return ChatRecord(mid, 42, uid, nick, text, time.time() + mid, is_self=is_self)


PERSONA = "你叫测试君。"


def test_static_layer_is_byte_stable_across_calls():
    a = static_system_prompt(PERSONA, "reply")
    b = static_system_prompt(PERSONA, "reply")
    assert a == b
    j1 = static_system_prompt(PERSONA, "judge")
    j2 = static_system_prompt(PERSONA, "judge")
    assert j1 == j2
    assert PERSONA in a and PERSONA in j1


def test_runtime_layer_stable_within_day_and_order_independent_nicks():
    nicks1 = ["甲", "乙"]
    nicks2 = ["乙", "甲"]  # 顺序变化被视为不同 → 放 L2 是取舍后的位置（低频）
    a = runtime_system_prompt(42, "2026-08-27", nicks1)
    b = runtime_system_prompt(42, "2026-08-27", nicks1)
    assert a == b
    c = runtime_system_prompt(42, "2026-09-01", nicks2)
    assert a != c  # 日期变了必须失效其后缓存——语义正确


def test_history_only_append_truncates_from_head():
    records = [rec(i, 100 + i, f"u{i}", f"msg{i}") for i in range(10)]
    turns_full, trunc1 = history_to_turns(records, max_chars=10**9)
    assert len(turns_full) == 10 and not trunc1

    limit = 15
    turns_small, truncated = history_to_turns(records, max_chars=limit)
    assert truncated
    # 从头整块淘汰，剩余是尾部连续段
    joined = [t.content for t in turns_small]
    assert all("msg" in j for j in joined)
    total_chars = sum(len(t.content) for t in turns_small)
    # 单条不超过上限时，整块淘汰后总长必然 <= limit（至少保留最后一条）
    tail_single = len(record_to_turn(records[-1]).content)
    assert total_chars <= max(limit, tail_single)


def test_history_never_returns_empty_when_over_limit():
    records = [rec(i, 100, "u", "x" * 50) for i in range(5)]
    turns, _ = history_to_turns(records, max_chars=10)
    assert len(turns) >= 1  # 至少保留最后一条


def test_record_role_mapping_and_sender_label():
    other = record_to_turn(rec(1, 5, "小明", "hi"))
    mine = record_to_turn(rec(2, 99, "糖糖", "hello", is_self=True))
    assert other.role == "user" and other.content == "小明(5)：hi"
    assert mine.role == "assistant" and mine.content == "hello"

    # 指令层同样暴露完整发送者信息
    prompt = final_user_prompt_judge("2026-08-27 10:00:00", rec(3, 6, "小红", "在吗"))
    assert "来自 小红(6)：" in prompt

    # 昵称为空时退化为纯 QQ 标签
    anon = rec(4, 7, "", "无昵称")
    assert record_to_turn(anon).content == "QQ7：无昵称"


def test_build_messages_layer_order():
    static = static_system_prompt(PERSONA, "reply")
    runtime = runtime_system_prompt(42, "2026-08-27", [])
    hist = [record_to_turn(rec(i, 3, "nick", f"m{i}")) for i in range(3)]
    messages = build_messages(static, runtime, hist, "最终指令")
    roles = [m["role"] for m in messages]
    assert roles == ["system", "system", "user", "user", "user", "user"]
    assert messages[0]["content"] == static
    assert messages[1]["content"] == runtime
    assert [m["content"] for m in messages[2:5]] == [t.content for t in hist]
    assert messages[-1]["content"] == "最终指令"


def test_prefix_property_between_two_calls():
    """KV Cache 核心性质：同一会话相邻两次调用，前者消息数组是后者的严格前缀增量。

    即：前 n 条完全一致（L1/L2 不变、历史只追加），只有最后一层不同。
    """
    static = static_system_prompt(PERSONA, "reply")
    runtime = runtime_system_prompt(42, "2026-08-27", ["甲"])
    records_v1 = [rec(i, 3, "甲", f"早{i}") for i in range(3)]
    records_v2 = records_v1 + [rec(3, 4, "乙", "晚")]

    msg_v1 = build_messages(
        static,
        runtime,
        [record_to_turn(r) for r in records_v1],
        final_user_prompt_judge("2026-08-27 10:00:00", records_v1[-1]),
    )
    msg_v2 = build_messages(
        static,
        runtime,
        [record_to_turn(r) for r in records_v2],
        final_user_prompt_judge("2026-08-27 10:00:05", records_v2[-1]),
    )
    # v1 的 system+历史部分与 v2 的对应部分逐字节相同
    assert msg_v1[:-1] == msg_v2[:-1][:-1]
    # 最后的指令层各自不同
    assert msg_v1[-1] != msg_v2[-1]


def test_judge_prompt_has_anchors_and_never_reveals_threshold():
    """judge 静态层必须有评分锚点与一致性要求；reply 静态层则不需要。"""
    judge_static = static_system_prompt(PERSONA, "judge")
    reply_static = static_system_prompt(PERSONA, "reply")
    assert "评分锚点" in judge_static
    assert "一致性要求" in judge_static
    assert "评分锚点" not in reply_static

    # 首评指令层绝不能透露发言门槛，否则模型会围绕门槛打分
    prompt = final_user_prompt_judge("2026-08-27 10:00:00", rec(1, 5, "小明", "在吗"))
    assert "发言门槛" not in prompt


def test_recheck_prompt_reveals_threshold_and_quotes_first_verdict():
    """复核指令层告知真实门槛与触发下限并引用首评结论，同时强调禁止凑分。"""
    prompt = final_user_prompt_judge_recheck(
        "2026-08-27 10:00:00",
        rec(1, 5, "小明", "在吗"),
        prev_score=7,
        prev_reason="话题开放能接话",
        threshold=8,
        min_score=4,
    )
    assert "判定是 7 分" in prompt and "话题开放能接话" in prompt
    assert "高于 4 分" in prompt
    assert "发言门槛" in prompt and "8 分（满分 10）" in prompt
    assert "不低于 8" in prompt and "低于 8" in prompt
    assert "凑" in prompt  # 明确提醒不要为过线抬分

    # 关键语义：主动参与开放话题本身就是正当理由，不能暗示「没人在等就不能说」
    assert "开口的理由并非只有「有人在等我回应」" in prompt
    assert "话题向整个群开放" in prompt


def test_nickname_list_from_history():
    """L2 成员表使用与历史层一致的完整标签（含 QQ 号）。"""
    hist = [record_to_turn(rec(i, 3, f"n{i%2}", "x")) for i in range(6)]
    hist.append(record_to_turn(rec(9, 99, "糖糖", "me", is_self=True)))
    names = nickname_list_from_history(hist)
    assert names == ["n0(3)", "n1(3)"]  # assistant 不计入；保序去重；带 QQ 号


def test_output_contract_follows_via_tool_flag():
    """提示词输出契约必须与请求是否携带 tools 严格配套，不容错配。"""
    msg = rec(1, 5, "小明", "在吗")

    # reply：L1 守则与 L4 指令随 via_tool 同步切换
    tool_static = static_system_prompt(PERSONA, "reply")
    text_static = static_system_prompt(PERSONA, "reply", via_tool=False)
    assert "send_reply" in tool_static and "send_reply" not in text_static
    assert "<drop_img" not in tool_static and "<drop_img" in text_static
    assert "send_reply" in final_user_prompt_reply("t", msg, forced=True)
    assert (
        "send_reply"
        not in final_user_prompt_reply("t", msg, forced=True, via_tool=False)
    )

    # judge：L1 守则通用，L4 的输出契约随 via_tool 切换
    judge_static = static_system_prompt(PERSONA, "judge")
    assert "send_reply" not in judge_static  # judge 不说话，两种模式共用
    assert "submit_judgment" in final_user_prompt_judge("t", msg)
    assert "submit_judgment" not in final_user_prompt_judge("t", msg, via_tool=False)
    assert "JSON" in final_user_prompt_judge("t", msg, via_tool=False)

    recheck = final_user_prompt_judge_recheck(
        "t", msg, prev_score=7, prev_reason="r", threshold=8, min_score=4, via_tool=False
    )
    assert "submit_judgment" not in recheck and "JSON" in recheck
