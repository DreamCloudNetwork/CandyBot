"""bot 发送链路接入后处理的编排测试：逐条发送、打字延迟、失败中断与记忆写回。"""

from __future__ import annotations

import asyncio
import logging

from candybot import bot as bot_module
from candybot.ai import ReplyDraft
from candybot.postprocess import estimate_typing_time, process_reply
from tests.deterministic_rng import SeededRng
from tests.test_integration import (
    FakeAI,
    FakeSnowluma,
    JudgeVerdict,
    group_event,
    make_settings,
    wait_until,
)

REPLY_TEXT = "第一条消息。第二条内容！第三条先说。第四条被合并进最后一条。"
EXPECTED_SEGMENTS = ["第一条消息", "第二条内容！", "第三条先说。第四条被合并进最后一条"]


def _self_texts(memory) -> list[str]:
    """热缓存中自己的发言文本（按记忆顺序），用于等待与断言逐条写回。"""
    return [r.text for r in memory.tail(20) if r.is_self]

# wait_until 依赖真实 sleep 让出事件循环；打桩 sleep 之后必须仍交还控制权
_REAL_SLEEP = asyncio.sleep
# wait_until 的轮询步长：全局 sleep 打桩会被它一并记录，断言时需过滤
_POLL_STEP = 0.02


def _typing_events(events: list) -> list:
    """剔除 wait_until 轮询产生的 sleep 记录，只留业务事件序列。"""
    return [(kind, p) for kind, p in events if not (kind == "sleep" and p == _POLL_STEP)]


class ScriptedAI(FakeAI):
    """固定返回一条指定文本的回复。"""

    def __init__(self, text: str):
        super().__init__(JudgeVerdict(9, "测试"))
        self.text = text

    async def generate_reply(self, *args, **kwargs) -> ReplyDraft:
        return ReplyDraft(self.text)


class ReconsiderAI(ScriptedAI):
    """固定重想产出的 AI：空正文＝放弃剩余腹稿，非空＝改写后继续。"""

    def __init__(self, text: str, reconsider_text: str):
        super().__init__(text)
        self.reconsider_text = reconsider_text

    async def reconsider_reply(self, static_system, runtime_system, recent,
                               now_text, *, sent_segments, pending_segments):
        self.reconsider_calls.append({
            "recent_len": len(recent),
            "sent": tuple(sent_segments),
            "pending": tuple(pending_segments),
        })
        return ReplyDraft(self.reconsider_text)


class InterruptDuringGenAI(ReconsiderAI):
    """生成期间（第一条发出之前）就有他人消息插进来。

    注入的消息 id 固定，第二次生成调用时会被 dedup 丢弃，不会再算作新的插话。
    """

    def __init__(self, bot, text: str, reconsider_text: str):
        super().__init__(text, reconsider_text)
        self.bot = bot

    async def generate_reply(self, *args, **kwargs):
        await self.bot._on_event(group_event(2, "我先插一句", uid=1001))
        return await super().generate_reply(*args, **kwargs)


def _arm_interrupt(
    bot, monkeypatch, plan: dict[int, tuple[int, str, int]], *, at_me: bool = False
) -> dict:
    """在第 n 次群发消息发送时先注入一条他人消息（模拟连发期间被插话）。

    plan 的值为 (message_id, 文本, user_id)；返回计数字典便于断言发送次数。
    at_me=True 让插话必然被接话（forced 绕过反插话护栏），用于断言插话的
    后续完整轨迹；否则插话是否换来回复取决于护栏配置。
    """
    chained_send = bot._snowluma.send_group_msg
    counter = {"n": 0}

    async def send_with_interrupt(group_id, message):
        counter["n"] += 1
        entry = plan.get(counter["n"])
        if entry is not None:
            await bot._on_event(
                group_event(entry[0], entry[1], uid=entry[2], at_me=at_me)
            )
        return await chained_send(group_id, message)

    monkeypatch.setattr(bot._snowluma, "send_group_msg", send_with_interrupt)
    return counter


SPLIT_PP = {
    "enabled": True,
    "typing_speed": 1.0,
    "max_split": 3,
    "typo_error_rate": 0.0,
    "typo_word_replace_rate": 0.0,
}


class FailingSnowluma(FakeSnowluma):
    """对指定文本的消息始终发送失败。"""

    def __init__(self, fail_on: str):
        super().__init__()
        self.fail_on = fail_on

    async def send_group_msg(self, group_id: int, message) -> int:
        if isinstance(message, str) and message == self.fail_on:
            raise RuntimeError("网络中断")
        return await super().send_group_msg(group_id, message)


async def build_postprocess_bot(
    tmp_path,
    monkeypatch,
    *,
    post_process: dict,
    reply_text=REPLY_TEXT,
    ai=None,
    generation: dict | None = None,
):
    """构造开启后处理的机器人；asyncio.sleep 记录到 events 并立即返回。"""
    settings = make_settings(tmp_path, generation, post_process=post_process)
    bot = bot_module.CandyBot(settings)
    bot._snowluma = FakeSnowluma()
    bot._ai = ai if ai is not None else ScriptedAI(reply_text)
    events: list = []

    async def fake_sleep(delay):
        events.append(("sleep", delay))
        await _REAL_SLEEP(0)

    async def record_send(group_id, message):
        events.append(("send", message))
        return await FakeSnowluma.send_group_msg(bot._snowluma, group_id, message)

    monkeypatch.setattr(bot_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(bot._snowluma, "send_group_msg", record_send)
    return bot, events


async def test_multi_segment_send_with_typing_delay(tmp_path, monkeypatch, caplog):
    """拆条逐发：第一条零延迟（生成耗时即自然延迟），其后每条先按下一条
    估算打字时长 sleep；DEBUG 日志可见拆条与打字预估。"""
    bot, events = await build_postprocess_bot(
        tmp_path,
        monkeypatch,
        post_process={
            "enabled": True,
            "typing_speed": 1.0,
            "max_split": 3,
            "typo_error_rate": 0.0,
            "typo_word_replace_rate": 0.0,
        },
    )
    try:
        memory = await bot._memory.get(42)
        with caplog.at_level(logging.DEBUG, logger="candybot"):
            await bot._on_event(group_event(1, "聊聊", at_me=True))
            await wait_until(lambda: len(_self_texts(memory)) == 3)

        expected = EXPECTED_SEGMENTS
        assert [text for _, text in bot._snowluma.sent] == expected
        # 每条正文发送成功后各自写回记忆：三条独立的自发言记录
        assert _self_texts(memory) == expected
        # 事件顺序：send → sleep → send → sleep → send，第一条前无 sleep
        typing = _typing_events(events)
        assert [kind for kind, _ in typing] == ["send", "sleep", "send", "sleep", "send"]
        assert typing[1] == ("sleep", estimate_typing_time(expected[1], 1.0))
        assert typing[3] == ("sleep", estimate_typing_time(expected[2], 1.0))
        assert any("拆为 3 条" in record.message for record in caplog.records)
        assert any("预计打字" in record.message for record in caplog.records)
    finally:
        await bot.stop()


async def test_interrupted_multi_send_keeps_real_order(tmp_path, monkeypatch):
    """连发中途有人插话：每条正文发送成功即单独写回记忆，穿插进来的消息
    落在真实的时间位置，自己的多条发言不会合并成一段挤在插话之后。"""
    bot, _events = await build_postprocess_bot(
        tmp_path,
        monkeypatch,
        post_process={
            "enabled": True,
            "typing_speed": 1.0,
            "max_split": 3,
            "typo_error_rate": 0.0,
            "typo_word_replace_rate": 0.0,
        },
    )
    memory = await bot._memory.get(42)
    chained_send = bot._snowluma.send_group_msg
    send_count = 0

    async def send_with_interrupt(group_id, message):
        nonlocal send_count
        send_count += 1
        if send_count == 2:
            # 第 1 条已发送并写回、第 2 条还在打字：此时有人插话
            await bot._on_event(group_event(2, "不是", uid=1001))
        return await chained_send(group_id, message)

    monkeypatch.setattr(bot._snowluma, "send_group_msg", send_with_interrupt)
    try:
        await bot._on_event(group_event(1, "聊聊", at_me=True))
        # 插话消息之后也会被回复（又三条自发言），用 >= 等待首轮连发完成
        await wait_until(lambda: len(_self_texts(memory)) >= 3)
        seq = [
            ("self" if r.is_self else "user", r.text) for r in memory.tail(20)
        ]
        assert seq[:5] == [
            ("user", "@糖糖\n聊聊"),
            ("self", "第一条消息"),
            ("user", "不是"),
            ("self", "第二条内容！"),
            ("self", "第三条先说。第四条被合并进最后一条"),
        ]
    finally:
        await bot.stop()


async def test_reconsider_lets_ai_abort_remaining(tmp_path, monkeypatch):
    """被打断后重想决定不说了：已发出的保留，腹稿既不发也不进记忆。

    插话在第 2 条发送期间进入（模拟打字间隙）：重想发生在第 3 条发出前，
    放弃后第 3 条不再发送，其更正也随之作废。
    """
    ai = ReconsiderAI(REPLY_TEXT, "")
    bot, _ = await build_postprocess_bot(tmp_path, monkeypatch, post_process=SPLIT_PP, ai=ai)
    memory = await bot._memory.get(42)
    # 插话 @机器人：放弃后它必然接话，可据完整发送轨迹验证旧腹稿没发出去
    _arm_interrupt(bot, monkeypatch, {2: (2, "不是", 1001)}, at_me=True)
    try:
        await bot._on_event(group_event(1, "聊聊", at_me=True))
        # 首轮止于 2 条；插话「不是」随后被正常判定，另起 3 条新连发
        await wait_until(lambda: len(_self_texts(memory)) == 5)

        assert len(ai.reconsider_calls) == 1
        assert ai.reconsider_calls[0]["sent"] == ("第一条消息", "第二条内容！")
        assert ai.reconsider_calls[0]["pending"] == ("第三条先说。第四条被合并进最后一条",)
        # 第 3 次发送已是针对插话的新回复开头：旧腹稿从未发出
        assert [text for _, text in bot._snowluma.sent][:3] == [
            "第一条消息",  # 首轮
            "第二条内容！",  # 首轮（重想前已按原计划发出）
            "第一条消息",  # 对「不是」的新回复（REPLY_TEXT 重放）
        ]
    finally:
        await bot.stop()


async def test_reconsider_can_rewrite_remaining(tmp_path, monkeypatch):
    """重想可顺着插话改写剩余内容：改后的话接着逐条发送并各自入记忆。"""
    ai = ReconsiderAI(REPLY_TEXT, "算了当我没说")
    bot, _ = await build_postprocess_bot(tmp_path, monkeypatch, post_process=SPLIT_PP, ai=ai)
    memory = await bot._memory.get(42)
    _arm_interrupt(bot, monkeypatch, {2: (2, "不是", 1001)})
    try:
        await bot._on_event(group_event(1, "聊聊", at_me=True))
        await wait_until(lambda: len(_self_texts(memory)) >= 3)

        assert len(ai.reconsider_calls) == 1
        assert ai.reconsider_calls[0]["sent"] == ("第一条消息", "第二条内容！")
        assert ai.reconsider_calls[0]["pending"] == ("第三条先说。第四条被合并进最后一条",)
        seq = [
            ("self" if r.is_self else "user", r.text) for r in memory.tail(20)
        ]
        assert seq[:5] == [
            ("user", "@糖糖\n聊聊"),
            ("self", "第一条消息"),
            ("user", "不是"),
            ("self", "第二条内容！"),
            ("self", "算了当我没说"),
        ]
        assert [text for _, text in bot._snowluma.sent[:3]] == [
            "第一条消息",
            "第二条内容！",
            "算了当我没说",
        ]
    finally:
        await bot.stop()


async def test_reconsider_budget_exhausted_sends_as_planned(tmp_path, monkeypatch):
    """重想预算用尽：之后的插话不再触发调用，剩余腹稿按原计划发完。"""
    ai = ScriptedAI(REPLY_TEXT)  # FakeAI 默认重想＝一字不改，bot 沿用原计划
    bot, _ = await build_postprocess_bot(
        tmp_path,
        monkeypatch,
        post_process=SPLIT_PP,
        ai=ai,
        generation={"max_reconsider_per_burst": 1},  # 预算从 2 收紧到 1
    )
    memory = await bot._memory.get(42)
    _arm_interrupt(bot, monkeypatch, {1: (2, "不是", 1001), 2: (3, "哈？", 1002)})
    try:
        await bot._on_event(group_event(1, "聊聊", at_me=True))
        await wait_until(lambda: len(_self_texts(memory)) >= 3)

        # 两次插话都检测到，但只有第一次换来重想调用；模型答「照原样」
        # 后原计划三条完整发出
        assert len(ai.reconsider_calls) == 1
        assert [text for _, text in bot._snowluma.sent[:3]] == EXPECTED_SEGMENTS
    finally:
        await bot.stop()


async def test_typing_speed_zero_disables_delay(tmp_path, monkeypatch):
    bot, events = await build_postprocess_bot(
        tmp_path,
        monkeypatch,
        post_process={
            "enabled": True,
            "typing_speed": 0.0,
            "typo_error_rate": 0.0,
            "typo_word_replace_rate": 0.0,
        },
    )
    try:
        memory = await bot._memory.get(42)
        await bot._on_event(group_event(1, "聊聊", at_me=True))
        await wait_until(lambda: len(_self_texts(memory)) == 3)
        # 0.0 秒的延迟不再 sleep（estimate 为 0 时跳过 await）
        assert [kind for kind, _ in _typing_events(events)] == ["send", "send", "send"]
    finally:
        await bot.stop()


async def test_mid_failure_aborts_remaining(tmp_path, monkeypatch, caplog):
    """第二条重试用尽仍失败：放弃剩余条目、按现有错误路径记日志。
    写回跟着发送走：已发出的第一条进记忆，失败的条目不进。"""
    settings = make_settings(
        tmp_path,
        post_process={
            "enabled": True,
            "max_split": 3,
            "typo_error_rate": 0.0,
            "typo_word_replace_rate": 0.0,
        },
    )
    bot = bot_module.CandyBot(settings)
    bot._snowluma = FailingSnowluma("第二条内容！")
    bot._ai = ScriptedAI(REPLY_TEXT)
    sleeps: list[float] = []

    async def fake_sleep(delay):
        sleeps.append(delay)
        await _REAL_SLEEP(0)

    monkeypatch.setattr(bot_module.asyncio, "sleep", fake_sleep)
    try:
        with caplog.at_level(logging.DEBUG, logger="candybot"):
            await bot._on_event(group_event(1, "聊聊", at_me=True))
            await wait_until(lambda: "发送失败" in caplog.text)

        # 只有第一条成功发出；第二条的 3 次重试与 1.5/3.0 秒退避、
        # 第三条的发送尝试都不应出现
        assert [text for _, text in bot._snowluma.sent] == ["第一条消息"]
        assert 1.5 in sleeps and 3.0 in sleeps
        memory = await bot._memory.get(42)
        assert _self_texts(memory) == ["第一条消息"]
    finally:
        await bot.stop()


async def test_sent_record_uses_clean_typo_free_text(tmp_path, monkeypatch):
    """写回记忆的必须是拆条合并后的无错字原文（错别字与更正不进 L3）。"""
    post_process = {
        "enabled": True,
        "typing_speed": 0.0,
        "max_split": 3,
        "typo_error_rate": 0.3,
        "typo_correction_probability": 1.0,
    }
    settings = make_settings(tmp_path, post_process=post_process)
    bot = bot_module.CandyBot(settings)
    bot._pp_rng = SeededRng(2026)  # 固定种子：与下面复算的期望完全同轨迹
    bot._snowluma = FakeSnowluma()
    bot._ai = ScriptedAI(REPLY_TEXT)

    async def fake_sleep(delay):
        await _REAL_SLEEP(0)

    monkeypatch.setattr(bot_module.asyncio, "sleep", fake_sleep)
    try:
        memory = await bot._memory.get(42)
        await bot._on_event(group_event(1, "聊聊", at_me=True))
        # 每条正文发送成功即写回；等三条全部落地且更正（段数组）也发出
        await wait_until(
            lambda: len(_self_texts(memory)) == 3
            and isinstance(bot._snowluma.sent[-1][1], list)
        )

        expected = process_reply(
            REPLY_TEXT, settings.response_post_process, rng=SeededRng(2026)
        )
        self_records = [r for r in memory.tail(20) if r.is_self]
        # 逐条写回的是无错字原文：条数、顺序、内容都与拆条对齐
        assert [r.text for r in self_records] == expected.memory_segments
        assert all("\n" not in r.text for r in self_records)
        assert [text for _, text in bot._snowluma.sent[:3]] == expected.messages
        # 确实走到了错字+更正路径，且更正不进记忆
        assert expected.correction and expected.correction.startswith("＊")
        assert all(expected.correction not in r.text for r in self_records)
        # 更正以 OneBot reply 段引用最后一条正文发送
        correction_message = bot._snowluma.sent[-1][1]
        assert isinstance(correction_message, list)
        assert correction_message[0]["type"] == "reply"
        assert correction_message[0]["data"]["id"] == str(bot._snowluma._next_id - 1)
        assert correction_message[1] == {
            "type": "text",
            "data": {"text": expected.correction},
        }
    finally:
        await bot.stop()


async def test_correction_falls_back_to_plain_text_without_message_id(tmp_path, monkeypatch, caplog):
    """发送响应拿不到 message_id 时：记警告，更正退回无引用的纯文本段。"""
    post_process = {
        "enabled": True,
        "typing_speed": 0.0,
        "max_split": 3,
        "typo_error_rate": 1.0,
        "typo_correction_probability": 1.0,
    }
    settings = make_settings(tmp_path, post_process=post_process)
    bot = bot_module.CandyBot(settings)
    bot._pp_rng = SeededRng(2026)
    bot._snowluma = FakeSnowluma()
    bot._ai = ScriptedAI(REPLY_TEXT)

    async def fake_send(group_id, message):
        bot._snowluma.sent.append((group_id, message))
        return None  # 模拟响应不带 message_id

    async def fake_sleep(delay):
        await _REAL_SLEEP(0)

    monkeypatch.setattr(bot_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(bot._snowluma, "send_group_msg", fake_send)
    try:
        with caplog.at_level(logging.WARNING, logger="candybot"):
            memory = await bot._memory.get(42)
            await bot._on_event(group_event(1, "聊聊", at_me=True))
            await wait_until(
                lambda: len(_self_texts(memory)) == 3
                and isinstance(bot._snowluma.sent[-1][1], list)
            )
        expected = process_reply(
            REPLY_TEXT, settings.response_post_process, rng=SeededRng(2026)
        )
        assert expected.correction  # 该种子下确实产生更正
        correction_message = bot._snowluma.sent[-1][1]
        assert isinstance(correction_message, list)
        assert [seg["type"] for seg in correction_message] == ["text"]
        assert any("未返回 message_id" in r.message for r in caplog.records)
    finally:
        await bot.stop()


async def test_correction_references_message_id_zero(tmp_path, monkeypatch, caplog):
    """message_id 0 是合法 id：更正仍带引用，不得按「拿不到 id」降级并误报警告。"""
    settings = make_settings(
        tmp_path,
        post_process={
            "enabled": True,
            "typing_speed": 0.0,
            "max_split": 3,
            "typo_error_rate": 1.0,
            "typo_correction_probability": 1.0,
        },
    )
    bot = bot_module.CandyBot(settings)
    bot._pp_rng = SeededRng(2026)
    bot._snowluma = FakeSnowluma()
    bot._ai = ScriptedAI(REPLY_TEXT)

    async def fake_send(group_id, message):
        bot._snowluma.sent.append((group_id, message))
        return 0  # 全部实现都返回 id 0（falsy，但不代表缺失）

    async def fake_sleep(delay):
        await _REAL_SLEEP(0)

    monkeypatch.setattr(bot_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(bot._snowluma, "send_group_msg", fake_send)
    try:
        memory = await bot._memory.get(42)
        with caplog.at_level(logging.WARNING, logger="candybot"):
            await bot._on_event(group_event(1, "聊聊", at_me=True))
            await wait_until(
                lambda: len(_self_texts(memory)) == 3
                and isinstance(bot._snowluma.sent[-1][1], list)
            )
        correction_message = bot._snowluma.sent[-1][1]
        assert isinstance(correction_message, list)
        assert correction_message[0] == {"type": "reply", "data": {"id": "0"}}
        assert not any("未返回 message_id" in r.message for r in caplog.records)
    finally:
        await bot.stop()


class StubServer:
    async def start(self):
        pass

    async def stop(self):
        pass


async def _start_bot_with_stubs(bot):
    bot._server = StubServer()
    await bot.start()
    await bot.stop()


async def test_start_preheats_typo_indexes(tmp_path, monkeypatch):
    """启用错字率时，start() 在后台线程预建拼音反查表（避免首条回复卡事件循环）。"""
    calls: list[bool] = []
    monkeypatch.setattr(bot_module, "ensure_indexes", lambda: calls.append(True))
    settings = make_settings(tmp_path, post_process={"enabled": True, "typing_speed": 0.0})
    bot = bot_module.CandyBot(settings)
    bot._snowluma = FakeSnowluma()
    await _start_bot_with_stubs(bot)
    assert calls == [True]


async def test_start_skips_typo_index_warmup_when_unneeded(tmp_path, monkeypatch):
    """后处理关闭或错字率为 0：没有热路径可言，不预热。"""
    calls: list[bool] = []
    monkeypatch.setattr(bot_module, "ensure_indexes", lambda: calls.append(True))
    for post_process in (
        {"enabled": False},
        {"enabled": True, "typo_error_rate": 0.0, "typo_word_replace_rate": 0.0},
    ):
        calls.clear()
        settings = make_settings(tmp_path, post_process=post_process)
        bot = bot_module.CandyBot(settings)
        bot._snowluma = FakeSnowluma()
        await _start_bot_with_stubs(bot)
        assert calls == [], post_process


# ------------------------------------------------ 放弃/失败时的护栏与配额记账


async def _build_guard_bot(tmp_path, monkeypatch, *, snowluma, ai, post_process):
    """按主动插话路径驱动的记账测试脚手架：sleep 打桩、不记录事件序列。"""
    settings = make_settings(tmp_path, post_process=post_process)
    bot = bot_module.CandyBot(settings)
    bot._snowluma = snowluma
    bot._ai = ai

    async def fake_sleep(delay):
        await _REAL_SLEEP(0)

    monkeypatch.setattr(bot_module.asyncio, "sleep", fake_sleep)
    return bot


async def test_abort_all_before_first_send_refunds_quota_and_keeps_guards(
    tmp_path, monkeypatch
):
    """生成期间被插话、重想发生在首条发出前并全部放弃：一条也没说，
    日配额退还、不刷新冷却与间隔——插话随后照常过判断并得到回应，
    不会被刚刷新的冷却挡在门外（prompts 里承诺的行为）。"""
    bot = await _build_guard_bot(
        tmp_path,
        monkeypatch,
        snowluma=FakeSnowluma(),
        ai=None,
        post_process=SPLIT_PP,
    )
    ai = InterruptDuringGenAI(bot, "腹稿一。腹稿二", "")
    bot._ai = ai
    try:
        memory = await bot._memory.get(42)
        runtime = bot._runtimes[42]
        await bot._on_event(group_event(1, "聊聊", uid=1000))  # 主动插话触发
        # 第二轮（对插话「我先插一句」的回应）完整发出两条
        await wait_until(lambda: len(_self_texts(memory)) == 2)

        assert len(ai.reconsider_calls) == 1
        assert ai.reconsider_calls[0]["sent"] == ()  # 放弃发生在开口之前
        # 第一轮腹稿从未发出；群里的两条都是回应插话的新回复
        assert [text for _, text in bot._snowluma.sent] == ["腹稿一", "腹稿二"]
        # 配额消耗 2 次、放弃的那次退还：净计 1 次
        assert bot._daily_replies == 1
        assert runtime.last_proactive_ts > 0  # 冷却只从真正发言后起算
    finally:
        await bot.stop()


async def test_first_segment_failure_refunds_quota_and_keeps_guards(
    tmp_path, monkeypatch, caplog
):
    """首条重试 3 次仍发送失败：与「没说过话」等价，退还配额、不刷新护栏。"""
    bot = await _build_guard_bot(
        tmp_path,
        monkeypatch,
        snowluma=FailingSnowluma("第一条消息"),
        ai=ScriptedAI(REPLY_TEXT),
        post_process=SPLIT_PP,
    )
    try:
        runtime = bot._runtimes[42]
        with caplog.at_level(logging.ERROR, logger="candybot"):
            await bot._on_event(group_event(1, "聊聊", uid=1000))
            await wait_until(lambda: "发送失败" in caplog.text)
        assert bot._daily_replies == 0
        assert runtime.last_proactive_ts == 0.0
        assert runtime.msgs_since_reply == 10**9 + 1  # 未清零：从未发言
        memory = await bot._memory.get(42)
        assert _self_texts(memory) == []
    finally:
        await bot.stop()


async def test_partial_send_failure_still_updates_guards(tmp_path, monkeypatch, caplog):
    """第 1 条已发出、第 2 条重试用尽：已经开口就要记账，配额也不退还。"""
    bot = await _build_guard_bot(
        tmp_path,
        monkeypatch,
        snowluma=FailingSnowluma("第二条内容！"),
        ai=ScriptedAI(REPLY_TEXT),
        post_process=SPLIT_PP,
    )
    try:
        runtime = bot._runtimes[42]
        with caplog.at_level(logging.ERROR, logger="candybot"):
            await bot._on_event(group_event(1, "聊聊", uid=1000))
            await wait_until(lambda: "发送失败" in caplog.text)
        await wait_until(lambda: runtime.msgs_since_reply == 0)
        assert runtime.last_proactive_ts > 0
        assert bot._daily_replies == 1
        memory = await bot._memory.get(42)
        assert _self_texts(memory) == ["第一条消息"]
    finally:
        await bot.stop()


class CorrectionFailingSnowluma(FakeSnowluma):
    """正文可发出，但段数组形态的更正消息始终发送失败。"""

    async def send_group_msg(self, group_id: int, message) -> int:
        if isinstance(message, list):
            raise RuntimeError("更正发送失败")
        return await super().send_group_msg(group_id, message)


async def test_correction_failure_keeps_guard_bookkeeping(
    tmp_path, monkeypatch, caplog
):
    """全部正文已发出、仅更正这条表层噪音失败：记账不能跟着丢失。"""
    post_process = {
        "enabled": True,
        "typing_speed": 0.0,
        "max_split": 3,
        "typo_error_rate": 1.0,
        "typo_correction_probability": 1.0,
    }
    bot = await _build_guard_bot(
        tmp_path, monkeypatch, snowluma=CorrectionFailingSnowluma(), ai=None,
        post_process=post_process,
    )
    bot._pp_rng = SeededRng(2026)
    bot._ai = ScriptedAI(REPLY_TEXT)
    expected = process_reply(
        REPLY_TEXT,
        make_settings(tmp_path, post_process=post_process).response_post_process,
        rng=SeededRng(2026),
    )
    assert expected.correction  # 该种子下确实会发更正
    try:
        runtime = bot._runtimes[42]
        with caplog.at_level(logging.ERROR, logger="candybot"):
            await bot._on_event(group_event(1, "聊聊", uid=1000))
            await wait_until(lambda: "发送失败" in caplog.text)
        await wait_until(lambda: runtime.msgs_since_reply == 0)
        assert runtime.last_proactive_ts > 0
        assert bot._daily_replies == 1
        memory = await bot._memory.get(42)
        assert _self_texts(memory) == expected.memory_segments
    finally:
        await bot.stop()


async def test_reconsider_skipped_when_postprocess_disabled(tmp_path, monkeypatch):
    """总开关关闭：插话不触发重想、更不放弃回复，行为回到未引入后处理前。"""
    bot = await _build_guard_bot(
        tmp_path,
        monkeypatch,
        snowluma=FakeSnowluma(),
        ai=None,
        post_process={"enabled": False},
    )
    ai = InterruptDuringGenAI(bot, "完整一条回复", "")
    bot._ai = ai
    try:
        memory = await bot._memory.get(42)
        await bot._on_event(group_event(1, "聊聊", uid=1000))
        await wait_until(lambda: ai.judge_calls >= 2)  # 插话也过了 judge
        assert ai.reconsider_calls == []
        # 整条照发，一条不少（旧行为没有「放弃」这个选项）
        assert [text for _, text in bot._snowluma.sent] == ["完整一条回复"]
    finally:
        await bot.stop()


async def test_verbatim_echo_with_whitespace_keeps_original_plan(tmp_path, monkeypatch):
    """重想逐字复读但带尾部空白/空行：应识别为照原样继续，沿用原计划，
    不得被误判成改写而把合并句重新拆开发送。"""
    ai = ReconsiderAI(REPLY_TEXT, "第三条先说。第四条被合并进最后一条\n \n")
    bot, _ = await build_postprocess_bot(tmp_path, monkeypatch, post_process=SPLIT_PP, ai=ai)
    memory = await bot._memory.get(42)
    _arm_interrupt(bot, monkeypatch, {2: (2, "不是", 1001)}, at_me=True)
    try:
        await bot._on_event(group_event(1, "聊聊", at_me=True))
        await wait_until(lambda: len(_self_texts(memory)) >= 3)

        assert len(ai.reconsider_calls) == 1
        # 第 3 条按原计划整体发出（改写路径会把这句再拆成两条）
        assert [text for _, text in bot._snowluma.sent[:3]] == EXPECTED_SEGMENTS
        assert _self_texts(memory)[:3] == EXPECTED_SEGMENTS
    finally:
        await bot.stop()
