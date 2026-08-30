"""主动发言心跳（任务 4）：调度、退避、护栏、竞态检查与发送记账全链路。

调度层用假时钟 + 固定种子随机源直接步进 HeartbeatScheduler.tick()；
执行层用真实 CandyBot 编排 + 假 SnowLuma/AI（复用 test_integration 基建），
LLM 调用全部 mock。
"""

from __future__ import annotations

import asyncio
import random
import time
from types import SimpleNamespace

import pytest

from candybot.ai import AIClient, JudgeVerdict, ProactiveVerdict, ReplyDraft
from candybot.bot import CandyBot
from candybot.heartbeat import HeartbeatScheduler, IdleEvaluation
from candybot.models import ProactiveSettings
from candybot.prompts import (
    final_user_prompt_proactive_judge,
    final_user_prompt_proactive_reply,
    final_user_prompt_reply,
)
from candybot.models import ChatRecord
from tests.test_integration import (
    FakeAI,
    FakeSnowluma,
    drain_tick,
    group_event,
    make_settings,
    wait_until,
)


# ------------------------------------------------------------ 调度层基建

class FakeClock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def pro_settings(**overrides) -> ProactiveSettings:
    base = dict(
        enabled=True,
        idle_min_seconds=600.0,
        idle_max_seconds=1800.0,
        only_active_today=False,
        min_today_messages=5,
        max_per_group_per_day=2,
        respond_reset_minutes=10.0,
        context_messages=-1,
    )
    base.update(overrides)
    return ProactiveSettings(**base)


def make_sched(**proactive_overrides):
    """假时钟调度器 + 固定种子窗口掷点 + 收集入队条目的回调。"""
    clock = FakeClock()
    proactive = pro_settings(**proactive_overrides)
    box = SimpleNamespace(proactive=proactive)
    items: list[tuple[int, IdleEvaluation]] = []

    async def enqueue(group_id: int, item: IdleEvaluation) -> None:
        items.append((group_id, item))

    sched = HeartbeatScheduler(lambda: box, enqueue, clock=clock)
    sched._rng = random.Random(7)
    return sched, clock, items


async def fire_and_complete(sched, clock, items, *, spoke: bool) -> None:
    """模拟一轮完整心跳：到点入队 → 处理完（可选实际发出）→ finish 重排。"""
    state = sched._states[42]
    clock.t = state.next_due
    await sched.tick()
    assert items, "到点后应投递空闲评估"
    items.clear()
    if spoke:
        sched.note_spoken(42)
    sched.finish(42)


# ------------------------------------------------------------ 调度：排程与入队


async def test_idle_due_enqueues_once_and_window_in_range():
    """空闲到点投递一次（不叠发）；窗口均匀随机落在 [min, max]。"""
    sched, clock, items = make_sched()
    sched.note_inbound(42)
    await sched.tick()  # 排程
    expected = random.Random(7).uniform(600.0, 1800.0)
    state = sched._states[42]
    assert state.next_due == pytest.approx(clock.t + expected)
    assert 600.0 <= state.next_due - clock.t <= 1800.0

    clock.t = state.next_due
    await sched.tick()
    assert [(gid, item.seq) for gid, item in items] == [(42, 1)]
    assert state.pending is True
    await sched.tick()  # 处理完之前绝不投递第二条
    assert len(items) == 1


async def test_inbound_before_due_reschedules():
    """到点前来了新消息：空闲重新起算，不会在旧窗口时刻冒泡。"""
    sched, clock, items = make_sched()
    sched.note_inbound(42)
    await sched.tick()
    old_due = sched._states[42].next_due
    clock.t = old_due - 100
    sched.note_inbound(42)  # 群友又说话了
    assert sched._states[42].next_due is None
    await sched.tick()
    new_due = sched._states[42].next_due
    assert new_due > old_due  # 从新时刻重新掷窗口
    clock.t = old_due  # 旧时刻已过，新窗口未到期
    await sched.tick()
    assert items == []


async def test_only_active_today_gate():
    """only_active_today：当天他人消息不足门槛的群不参与心跳。"""
    sched, clock, items = make_sched(only_active_today=True, min_today_messages=3)
    sched.note_inbound(42)
    sched.note_inbound(42)
    await sched.tick()
    assert sched._states[42].next_due is None  # 才 2 条，不排程
    sched.note_inbound(42)  # 第 3 条达到门槛
    await sched.tick()
    assert sched._states[42].next_due is not None


async def test_disabled_gate_blocks_tick_and_sync_creates_no_task():
    """enabled=false：tick 不投递、sync 不创建任何任务（关闭=零差异）。"""
    clock = FakeClock()
    box = SimpleNamespace(proactive=pro_settings(enabled=False))
    items: list[tuple[int, IdleEvaluation]] = []

    async def enqueue(gid, item):
        items.append((gid, item))

    sched = HeartbeatScheduler(lambda: box, enqueue, clock=clock)
    sched.sync()
    assert sched._task is None
    sched.note_inbound(42)
    await sched.tick()
    state = sched._states[42]
    state.next_due = clock.t - 1  # 强行造一个「早已到期」的窗口
    await sched.tick()
    assert items == []


# ------------------------------------------------------------ 调度：退避与观察窗


async def test_backoff_doubles_and_caps_at_8():
    """没人接话：每轮观察窗到点后窗口 ×2，封顶 ×8。"""
    sched, clock, items = make_sched(respond_reset_minutes=10)
    expect = 2
    for _round in range(4):
        sched.note_inbound(42)
        await sched.tick()  # 排程
        assert sched._states[42].next_due is not None
        await fire_and_complete(sched, clock, items, spoke=True)
        # 观察窗未结算：不提前掷下一轮窗口
        await sched.tick()
        assert sched._states[42].next_due is None
        # 无人接话，观察窗到点：退避翻倍
        clock.advance(10 * 60 + 1)
        await sched.tick()
        assert sched._states[42].multiplier == expect
        # 新窗口从本轮收尾时刻起算、乘当前倍数
        base = sched._states[42].base_override
        assert base is None, "排程后应消费排程基准"
        expect = min(expect * 2, 8)


async def test_response_within_watch_resets_backoff():
    """发言后观察窗内有人接话（@/回复，directed）：退避当场重置为 ×1。"""
    sched, clock, items = make_sched(respond_reset_minutes=10)
    sched.note_inbound(42)
    await sched.tick()
    await fire_and_complete(sched, clock, items, spoke=True)
    assert sched._states[42].watch_since is not None
    clock.advance(60)  # 观察窗内（<10 分钟）
    sched.note_inbound(42, directed=True)  # 有人接话了（@机器人）
    assert sched._states[42].multiplier == 1
    assert sched._states[42].watch_since is None


async def test_undirected_chat_does_not_reset_backoff():
    """群友互聊不算接话：观察窗不重置、到点照常翻倍退避；只有 @/回复才重置。"""
    sched, clock, items = make_sched(respond_reset_minutes=10)
    for expect in (2, 4):
        sched.note_inbound(42)
        await sched.tick()  # 排程
        await fire_and_complete(sched, clock, items, spoke=True)
        clock.advance(60)
        sched.note_inbound(42)  # 观察窗内的普通闲聊：watch 与倍数都不动
        assert sched._states[42].watch_since is not None
        assert sched._states[42].multiplier == expect // 2
        # 观察窗到点仍没人「接话」：照常退避翻倍
        clock.advance(10 * 60)
        await sched.tick()
        assert sched._states[42].multiplier == expect
        assert sched._states[42].watch_since is None


# ------------------------------------------------------------ bot 层基建


class ProAI(FakeAI):
    """judge 一律低分静默（被动链路不动）；空闲评估与生成按预置序列出。"""

    def __init__(self, proactive_verdicts=None, drafts=None):
        super().__init__(JudgeVerdict(1, "不关我事"))
        self.proactive_verdicts = list(proactive_verdicts or [])
        self.drafts = list(drafts or [])
        self.proactive_calls = 0
        self.proactive_recent_lens: list[int] = []

    async def evaluate_proactive(self, static_system, runtime_system, recent, now_text):
        self.proactive_calls += 1
        self.proactive_recent_lens.append(len(recent))
        verdict = (
            self.proactive_verdicts.pop(0)
            if self.proactive_verdicts
            else ProactiveVerdict(False)
        )
        if isinstance(verdict, Exception):
            raise verdict
        return verdict

    async def generate_reply(
        self,
        static_system,
        runtime_system,
        recent,
        current_message,
        now_text,
        *,
        forced,
        engaged=False,
        score=None,
        reason="",
        expression_hints=(),
        jargon_hints=(),
        repetition_warning=False,
        person_hints=(),
        proactive_intent=None,
    ):
        self.reply_calls.append(
            {
                "recent_len": len(recent),
                "forced": forced,
                "current": current_message,
                "proactive_intent": proactive_intent,
                "expression_hints": list(expression_hints),
            }
        )
        text = self.drafts.pop(0) if self.drafts else "想起来一句"
        return ReplyDraft(text)


async def build(tmp_path, *, proactive=None, rate_limit=None, post_process=None):
    settings = make_settings(
        tmp_path,
        post_process=post_process,
        rate_limit=rate_limit,
        proactive=proactive
        or {"enabled": True, "idle_min_seconds": 5, "idle_max_seconds": 10},
    )
    bot = CandyBot(settings)
    bot._snowluma = FakeSnowluma()
    return bot


async def chat(bot, *pairs) -> None:
    """灌入若干条普通消息（judge 低分静默），给空闲评估攒上下文。"""
    for mid, text in pairs:
        await bot._on_event(group_event(mid, text))
    await drain_tick()


def idle_item(bot, group_id: int = 42) -> IdleEvaluation:
    """造一条与当前计数对齐的空闲评估条目（竞态检查会放行）。"""
    return IdleEvaluation(enqueued_at=time.time(), seq=bot._heartbeat.seq_of(group_id))


# ------------------------------------------------------------ 执行：发言与记账


async def test_idle_speaks_and_books(tmp_path):
    """说 → 生成（自发言变体）→ 发送 → 记账：冷却/间隔/配额/当日上限全对齐。"""
    bot = await build(tmp_path)
    try:
        await chat(bot, (1, "周末去哪玩好"), (2, "我准备去爬山"))
        ai = ProAI([ProactiveVerdict(True, "关心一下爬山安排")], ["爬山记得多带水"])
        bot._ai = ai
        await bot._run_idle_evaluation(42, idle_item(bot))

        assert bot._snowluma.sent == [(42, "爬山记得多带水")]
        call = ai.reply_calls[0]
        assert call["proactive_intent"] == "关心一下爬山安排"
        assert call["forced"] is False and call["current"] is None
        # 与被动回复同一本账
        assert bot._daily_replies == 1
        runtime = bot._runtimes[42]
        assert runtime.last_proactive_ts > 0
        assert runtime.msgs_since_reply == 0
        assert bot._heartbeat.spoken_today(42) == 1
        # 写回记忆（is_self）
        memory = await bot._memory.get(42)
        assert memory.last().is_self
        assert memory.last().text == "爬山记得多带水"
    finally:
        await bot.stop()


async def test_idle_with_postprocess_split(tmp_path):
    """自发言照常过后处理（拆条/打字延迟）；逐条写回记忆。"""
    # 关闭错别字注入：写回记忆的本就允许与发出的（带错字）不同，这里断言的
    # 是「拆条 + 逐条写回」结构对齐，不该被随机错字干扰
    bot = await build(
        tmp_path,
        post_process={
            "enabled": True,
            "typing_speed": 0.0,
            "typo_error_rate": 0.0,
            "typo_word_replace_rate": 0.0,
        },
    )
    try:
        await chat(bot, (1, "聊聊周末"))
        bot._ai = ProAI([ProactiveVerdict(True, "接话头")], ["先说一句。再来一句。最后补一句。"])
        await bot._run_idle_evaluation(42, idle_item(bot))
        texts = [msg for _, msg in bot._snowluma.sent]
        assert len(texts) >= 2  # 被拆条
        memory = await bot._memory.get(42)
        self_texts = [r.text for r in memory.tail(10) if r.is_self]
        assert self_texts == texts  # 逐条写回与发出对齐
    finally:
        await bot.stop()


async def test_idle_burst_reconsiders_on_interrupt(tmp_path, monkeypatch):
    """主动连发中途被插话：与被动同源触发重想，空正文＝放弃剩余腹稿。

    以决策时刻（生成前）为基线，第一条正文发出后插进来的他人消息算插话：
    下一条发出前问模型——放弃则剩余不发且不进记忆，已发出的 ≥1 条照常
    记账（当日上限/冷却），与被动链路的行为完全一致。
    """
    bot = await build(
        tmp_path,
        post_process={
            "enabled": True,
            "typing_speed": 0.0,
            "max_split": 3,
            "typo_error_rate": 0.0,
            "typo_word_replace_rate": 0.0,
        },
    )
    try:
        await chat(bot, (1, "聊聊周末"))

        class IdleReconsiderAI(ProAI):
            async def reconsider_reply(self, static_system, runtime_system, recent,
                                       now_text, *, sent_segments, pending_segments):
                self.reconsider_calls.append({
                    "sent": tuple(sent_segments),
                    "pending": tuple(pending_segments),
                })
                return ReplyDraft("")  # 空正文＝剩余腹稿不再发

        ai = IdleReconsiderAI(
            [ProactiveVerdict(True, "接话头")], ["先说一句。再来一句。最后补一句。"]
        )
        bot._ai = ai
        chained = bot._snowluma.send_group_msg

        async def send_then_interrupt(group_id, message):
            result = await chained(group_id, message)
            if len(bot._snowluma.sent) == 1:
                # 第一条正文刚发出：群友插一句普通消息（judge 低分不会回）
                await bot._on_event(group_event(7, "我先插一句", uid=1001))
            return result

        monkeypatch.setattr(bot._snowluma, "send_group_msg", send_then_interrupt)
        await bot._run_idle_evaluation(42, idle_item(bot))

        assert ai.reconsider_calls, "主动连发被插话应触发重想"
        assert len(ai.reconsider_calls[0]["pending"]) == 2  # 剩余两条腹稿
        # 放弃生效：只发出第一条，剩余从未发出、也从未写回记忆
        assert [msg for _, msg in bot._snowluma.sent] == list(
            ai.reconsider_calls[0]["sent"]
        )
        memory = await bot._memory.get(42)
        self_texts = [r.text for r in memory.tail(10) if r.is_self]
        assert self_texts == list(ai.reconsider_calls[0]["sent"])
        # 至少发出了 1 条正文：当日上限与冷却照常记账
        assert bot._heartbeat.spoken_today(42) == 1
        assert bot._runtimes[42].last_proactive_ts > 0
    finally:
        await bot.stop()


async def test_directed_message_through_event_resets_backoff(tmp_path):
    """_on_event 接线：@我/回复我的消息以 directed=True 记进心跳；普通消息不算接话。"""
    bot = await build(tmp_path)
    try:
        await chat(bot, (1, "随便聊聊"))
        bot._ai = ProAI([])  # at_me 会走 @必答生成，换假 AI 避免真实端点调用
        hb = bot._heartbeat
        hb.note_spoken(42)  # 模拟刚发出一次主动发言：观察窗开启
        hb._states[42].multiplier = 2
        await bot._on_event(group_event(9, "闲聊", uid=1001))
        assert hb._states[42].multiplier == 2  # 普通闲聊不重置退避
        assert hb._states[42].watch_since is not None
        await bot._on_event(group_event(10, "你说的对", uid=1001, at_me=True))
        assert hb._states[42].multiplier == 1  # @它＝有人接话，当场重置
        assert hb._states[42].watch_since is None
    finally:
        await bot.stop()


async def test_context_messages_param(tmp_path):
    """context_messages：-1 沿用群 context_size，正数覆盖评估看到的条数。"""
    bot = await build(
        tmp_path, proactive={"enabled": True, "idle_min_seconds": 5, "idle_max_seconds": 10, "context_messages": 2}
    )
    try:
        await chat(bot, (1, "一"), (2, "二"), (3, "三"), (4, "四"))
        ai = ProAI([ProactiveVerdict(False)])
        bot._ai = ai
        await bot._run_idle_evaluation(42, idle_item(bot))
        assert ai.proactive_recent_lens == [2]
    finally:
        await bot.stop()


# ------------------------------------------------------------ 执行：护栏与静默


async def test_daily_cap_blocks_before_llm(tmp_path):
    """每群当日上限不过 → 完全不调模型（省钱）。"""
    bot = await build(tmp_path)
    try:
        ai = ProAI([ProactiveVerdict(True, "不该被调用")])
        bot._ai = ai
        bot._heartbeat.note_spoken(42)
        bot._heartbeat.note_spoken(42)  # 达到默认上限 2
        await bot._run_idle_evaluation(42, idle_item(bot))
        assert ai.proactive_calls == 0
        assert bot._snowluma.sent == []
    finally:
        await bot.stop()


async def test_global_quota_blocks_before_llm(tmp_path):
    """全局日配额满 → 同样不调模型。"""
    bot = await build(tmp_path, rate_limit={"global_daily_limit": 0})
    try:
        ai = ProAI([ProactiveVerdict(True, "不该被调用")])
        bot._ai = ai
        await bot._run_idle_evaluation(42, idle_item(bot))
        assert ai.proactive_calls == 0
    finally:
        await bot.stop()


async def test_cooldown_blocks_after_llm(tmp_path):
    """LLM 说 speak 之后再过冷却护栏（双保险）：拦下则不生成不发送。"""
    bot = await build(tmp_path)
    try:
        runtime = bot._runtimes[42]
        runtime.last_proactive_ts = time.time()  # 群 42 默认冷却 60 秒
        ai = ProAI([ProactiveVerdict(True, "想说")], ["不该生成的稿"])
        bot._ai = ai
        await bot._run_idle_evaluation(42, idle_item(bot))
        assert ai.proactive_calls == 1
        assert ai.reply_calls == []
        assert bot._snowluma.sent == []
        assert bot._heartbeat.spoken_today(42) == 0
    finally:
        await bot.stop()


async def test_race_new_message_after_enqueue_abandons(tmp_path):
    """入队之后又有他人消息入库：直接放弃本轮，不拿旧上下文说话、不调模型。"""
    bot = await build(tmp_path)
    try:
        ai = ProAI([ProactiveVerdict(True, "不该说话")])
        bot._ai = ai
        bot._heartbeat.note_inbound(42)
        item = IdleEvaluation(enqueued_at=time.time(), seq=bot._heartbeat.seq_of(42))
        bot._heartbeat.note_inbound(42)  # 入队后群友又说话了
        await bot._run_idle_evaluation(42, item)
        assert ai.proactive_calls == 0
        assert bot._snowluma.sent == []
        assert bot._heartbeat._states[42].pending is False  # finally 里重新排程
    finally:
        await bot.stop()


async def test_evaluate_failure_is_silent_no_retry(tmp_path):
    """评估调用失败：记 WARNING 本轮作罢，绝不重试轰炸。"""
    bot = await build(tmp_path)
    try:
        ai = ProAI([RuntimeError("端点炸了")])
        bot._ai = ai
        await bot._run_idle_evaluation(42, idle_item(bot))
        assert ai.proactive_calls == 1  # 就一次，不重试
        assert bot._snowluma.sent == []
    finally:
        await bot.stop()


async def test_speak_false_keeps_silent(tmp_path):
    """模型选沉默：静默重排程、不生成不记账。"""
    bot = await build(tmp_path)
    try:
        ai = ProAI([ProactiveVerdict(False)])
        bot._ai = ai
        await bot._run_idle_evaluation(42, idle_item(bot))
        assert ai.proactive_calls == 1
        assert ai.reply_calls == []
        assert bot._snowluma.sent == []
        assert bot._heartbeat.spoken_today(42) == 0
    finally:
        await bot.stop()


async def test_disabled_switch_voids_queued_item(tmp_path):
    """出队时开关已关（热重载改 false）：条目作废。"""
    bot = await build(tmp_path)
    try:
        from dataclasses import replace as dc_replace

        bot._settings = dc_replace(
            bot._settings, proactive=dc_replace(bot._settings.proactive, enabled=False)
        )
        ai = ProAI([ProactiveVerdict(True, "不该说话")])
        bot._ai = ai
        await bot._run_idle_evaluation(42, idle_item(bot))
        assert ai.proactive_calls == 0
    finally:
        await bot.stop()


# ------------------------------------------------------------ 执行：发送失败


async def test_all_send_failed_not_counted(tmp_path, monkeypatch):
    """全部发送失败：退日配额、不计每群上限、不刷新冷却——与被动规则对齐。"""
    bot = await build(tmp_path)
    try:
        async def boom(group_id, message):
            raise RuntimeError("发送炸了")

        monkeypatch.setattr(bot, "_send_with_retry", boom)
        ai = ProAI([ProactiveVerdict(True, "想说")], ["发不出去的话"])
        bot._ai = ai
        await bot._run_idle_evaluation(42, idle_item(bot))
        assert bot._daily_replies == 0  # 扣了又退
        assert bot._heartbeat.spoken_today(42) == 0
        assert bot._runtimes[42].last_proactive_ts == 0.0
        assert bot._snowluma.sent == []
    finally:
        await bot.stop()


# ------------------------------------------------------------ 队列集成


async def test_idle_and_real_message_serialized(tmp_path):
    """空闲评估与真实消息同队列互斥不乱序：评估生成期间 @必答照常入队，
    但排在评估之后处理，两条发言按队列顺序先后出现、绝不并发交叠。"""
    bot = await build(tmp_path)
    try:
        gate = asyncio.Event()
        ai = ProAI([ProactiveVerdict(True, "先说的")], ["主动的话", "被动回复"])
        inner_generate = ai.generate_reply

        async def gated(*args, **kwargs):
            await gate.wait()
            return await inner_generate(*args, **kwargs)

        ai.generate_reply = gated
        bot._ai = ai

        await bot._enqueue_idle(42, idle_item(bot))  # worker 消费，卡在生成里
        await wait_until(lambda: ai.proactive_calls == 1, timeout=3)
        # 生成执行中：@必答照常入队，但串行队列保证它排在评估之后
        await bot._on_event(group_event(5, "糖糖在吗", at_me=True))
        await drain_tick()
        assert len(bot._snowluma.sent) == 0  # 闸门未开，两条都还没发出
        gate.set()
        await wait_until(lambda: len(bot._snowluma.sent) == 2, timeout=3)
        assert [msg for _, msg in bot._snowluma.sent] == ["主动的话", "被动回复"]
    finally:
        await bot.stop()


# ------------------------------------------------------------ 生命周期


async def test_enabled_toggle_via_reload(tmp_path):
    """默认关闭不创建任务；热重载开→拉起循环任务，关→立即取消。"""
    off = make_settings(tmp_path)  # proactive 段缺省 = 关闭
    on = make_settings(
        tmp_path, proactive={"enabled": True, "idle_min_seconds": 5, "idle_max_seconds": 10}
    )
    holder = {"s": on}
    bot = CandyBot(off, settings_loader=lambda: holder["s"])
    assert bot._heartbeat._task is None  # 关闭态构造：不创建任何任务

    holder["s"] = on
    assert bot.reload_settings()
    task = bot._heartbeat._task
    assert task is not None and not task.done()

    holder["s"] = off
    assert bot.reload_settings()
    assert bot._heartbeat._task is None
    await asyncio.sleep(0.05)  # cancel 请求要过一轮事件循环才落到任务上
    assert task.cancelled()
    await bot.stop()


async def test_note_inbound_cheap_when_disabled(tmp_path):
    """关闭态下消息入库处只留纯记账：seq/last_inbound 更新，但永不排程。"""
    bot = await build(tmp_path, proactive={"enabled": False})
    try:
        await chat(bot, (1, "随便说一句"))
        state = bot._heartbeat._states.get(42)
        assert state is not None and state.seq == 1 and state.last_inbound > 0
        assert state.next_due is None
        assert bot._heartbeat._task is None
    finally:
        await bot.stop()


# ------------------------------------------------------------ 模型协议与提示词


def test_proactive_verdict_parsing_safety():
    """speak 只认显式布尔 true；JSON 坏 → 静默（宁可不说话）。"""
    assert AIClient._proactive_from_args({"speak": True, "intent": " 想接话 "}) == (
        ProactiveVerdict(speak=True, intent="想接话")
    )
    assert AIClient._proactive_from_args({"speak": "true", "intent": "x"}).speak is False
    assert AIClient._proactive_from_args({}).speak is False
    assert AIClient._parse_proactive("乱七八糟").speak is False
    assert AIClient._parse_proactive('{"speak": false}').speak is False
    assert AIClient._parse_proactive('前缀废话 {"speak": true, "intent": "关心"} 后缀').speak is True


async def test_proactive_generation_goes_through_ai_flavor(tmp_path):
    """自发言走真实 generate_reply：历史层不剥「当前消息」、L4 换自发言变体，
    AI 味拦截与重写对自发言同样生效。"""
    settings = make_settings(tmp_path)
    client = AIClient(
        models=settings.models,
        generation=settings.generation,
        multimodal_mode=settings.multimodal.mode,
    )
    records = [
        ChatRecord(
            message_id=i, group_id=42, user_id=1000 + i, nickname=f"u{i}",
            text=f"消息{i}", ts=float(i),
        )
        for i in range(1, 4)
    ]
    seen: list[tuple[int, object, str]] = []

    async def fake_call(static, runtime, history, current, build_text_part):
        seen.append((len(history), current, build_text_part(False)))
        if len(seen) == 1:
            return ReplyDraft("作为AI我想提醒大家多喝水")  # 必中默认 AI 味规则
        return ReplyDraft("对了，记得多喝水")

    client._reply_call = fake_call
    draft = await client.generate_reply(
        "L1", "L2", records, None, "2026-08-30 10:00:00",
        forced=False, proactive_intent="关心大家喝水",
    )
    assert draft is not None and "作为AI" not in draft.text  # 被拦截稿不泄漏
    assert len(seen) == 2  # AI 味拦截触发了一次重生成
    assert seen[0][0] == len(records) and seen[0][1] is None  # 全量历史、无当前消息
    assert "【主动发言】" in seen[0][2] and "关心大家喝水" in seen[0][2]
    assert "【需要重新生成】" in seen[1][2]  # 重写要求同样附进自发言 L4


def test_proactive_prompts_shape_and_isolation():
    """两段新 L4 独立成形；既有 final_user_prompt_reply 文本一字未动。"""
    tool_prompt = final_user_prompt_proactive_judge("2026-08-30 10:00:00", via_tool=True)
    json_prompt = final_user_prompt_proactive_judge("2026-08-30 10:00:00", via_tool=False)
    assert "submit_proactive" in tool_prompt
    assert '"speak"' in json_prompt
    assert "默认保持沉默" in tool_prompt and "绝不开场白式自我介绍" in tool_prompt

    reply = final_user_prompt_proactive_reply("2026-08-30 10:00:00", "关心爬山", via_tool=True)
    assert "【主动发言】没人问你话" in reply and "关心爬山" in reply
    assert "不要引用消息、不要@任何人" in reply

    msg = ChatRecord(
        message_id=1, group_id=42, user_id=1000, nickname="小王",
        text="在吗", ts=1.0,
    )
    plain = final_user_prompt_reply("2026-08-30 10:00:00", msg, forced=True)
    assert "主动发言" not in plain  # 既有回复文本保持字节稳定
