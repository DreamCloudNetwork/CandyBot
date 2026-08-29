"""决策层三项增强的单测：发送前新鲜度检查、观望重评、重复抑制。

全链路集成风格（真实 CandyBot 编排 + 假 MCP/AI），LLM 调用全部 mock。
复用 test_integration 的配置与假件基建。
"""

from __future__ import annotations

import asyncio
import time

from candybot.ai import JudgeVerdict, ReplyDraft
from candybot.bot import (
    CandyBot,
    _already_replied_to,
    _self_reply_after,
)
from candybot.models import ChatRecord, NormalizedMessage
from candybot.prompts import (
    final_user_prompt_reply,
    repetition_warning_block,
)
from tests.test_integration import (
    QueuedAI,
    FakeSnowluma,
    drain_tick,
    group_event,
    make_settings,
    recall_event,
    tune_group,
    wait_until,
)


def rec(mid: int, uid: int, text: str = "话题", *, self_: bool = False) -> ChatRecord:
    return ChatRecord(
        message_id=mid,
        group_id=42,
        user_id=uid,
        nickname="糖糖" if self_ else f"用户{uid}",
        text=text,
        ts=float(mid),
        is_self=self_,
    )


class DraftQueuedAI(QueuedAI):
    """judge 按预置序列出判定；reply 按序列出腹稿。

    每次生成调用前先 await hook（模拟生成期间群里进了新消息；同一条事件
    重复注入会被 dedup 忽略，hook 因此无需自行保证一次性）。另记录每次
    judge 看到的上下文条数，用于断言观望取的是届时最新 tail。
    """

    def __init__(self, verdicts: list[JudgeVerdict], drafts: list[str], hook=None):
        super().__init__(verdicts)
        self.drafts = list(drafts)
        self.hook = hook
        self.judge_recent_lens: list[int] = []

    async def judge_interest(self, *args, **kwargs) -> JudgeVerdict:
        self.judge_recent_lens.append(len(args[2]))
        return await super().judge_interest(*args, **kwargs)

    async def generate_reply(self, *args, **kwargs) -> ReplyDraft:
        if self.hook is not None:
            await self.hook()
        await super().generate_reply(*args, **kwargs)
        return ReplyDraft(self.drafts.pop(0) if self.drafts else "兜底回复")


async def build(tmp_path, *, generation_overrides=None) -> CandyBot:
    settings = make_settings(tmp_path, generation_overrides)
    bot = CandyBot(settings)
    bot._snowluma = FakeSnowluma()
    return bot


# ------------------------------------------------------------ 任务 A：新鲜度检查


async def test_freshness_regenerates_on_mention_during_generation(tmp_path):
    """生成期间被 @：新消息并入上下文重生成一次，发出的是重生成稿。"""
    bot = await build(tmp_path)
    try:
        bot._ai = DraftQueuedAI(
            [JudgeVerdict(9, "值得回")],
            ["草稿1", "草稿2", "草稿3"],
            hook=lambda: bot._on_event(group_event(2, "糖糖你看这个", at_me=True)),
        )

        await bot._on_event(group_event(1, "讨论个话题"))
        await wait_until(lambda: len(bot._snowluma.sent) == 2, timeout=3)

        # msg1 的回复被重生成一次（2 次生成），随后 msg2 的 @必答（第 3 次）
        assert len(bot._ai.reply_calls) == 3
        assert bot._ai.reply_calls[0]["recent_len"] == 1  # 原稿只见 msg1
        assert bot._ai.reply_calls[1]["recent_len"] == 2  # 重生成含插进来的 @
        texts = [msg for _, msg in bot._snowluma.sent]
        assert texts == ["草稿2", "草稿3"]  # 发出的是重生成后的稿子
    finally:
        await bot.stop()


async def test_freshness_ignores_plain_new_messages(tmp_path):
    """生成期间进的普通新话题不触发重生成（宁可稍旧也不要无限拖延）。"""
    bot = await build(tmp_path)
    try:
        bot._ai = DraftQueuedAI(
            [JudgeVerdict(9, "值得回"), JudgeVerdict(2, "与己无关")],
            ["草稿1", "草稿2"],
            hook=lambda: bot._on_event(group_event(2, "另一个话题", uid=1001)),
        )

        await bot._on_event(group_event(1, "讨论个话题"))
        await wait_until(lambda: len(bot._snowluma.sent) == 1, timeout=3)
        await drain_tick()  # msg2 被判定为不值得回（冷却 60 也拦着）

        assert len(bot._ai.reply_calls) == 1  # 没有重生成
        assert bot._snowluma.sent[0][1] == "草稿1"
    finally:
        await bot.stop()


async def test_freshness_at_most_one_regeneration_per_reply(tmp_path):
    """重生成期间又被 @：每条回复至多重生成一次，绝不循环。"""
    bot = await build(tmp_path)
    try:
        ai = DraftQueuedAI(
            [JudgeVerdict(9, "值得回")],
            ["稿1", "稿2", "稿3", "稿4"],
        )

        async def hook() -> None:
            # 只在 msg1 的原稿与那一次重生成的生成期间注入新 @；
            # 之后各消息的必答生成恢复正常
            if len(ai.reply_calls) < 2:
                mid = 10 + len(ai.reply_calls)
                await bot._on_event(group_event(mid, f"糖糖看{mid}", at_me=True))

        ai.hook = hook
        bot._ai = ai

        await bot._on_event(group_event(1, "讨论个话题"))
        await wait_until(lambda: len(bot._snowluma.sent) == 3, timeout=3)

        # msg1：原稿 + 唯一一次重生成（重生成期间进的 msg11 不再叠第三次）；
        # msg10 / msg11：各自必答，共 4 次生成
        assert len(ai.reply_calls) == 4
        assert ai.judge_calls == 1  # 只有 msg1 的首评
        texts = [msg for _, msg in bot._snowluma.sent]
        assert texts == ["稿2", "稿3", "稿4"]
    finally:
        await bot.stop()


async def test_freshness_disabled_by_config(tmp_path):
    """freshness_check_enabled=False 时行为与引入前完全一致。"""
    bot = await build(tmp_path, generation_overrides={"freshness_check_enabled": False})
    try:
        bot._ai = DraftQueuedAI(
            [JudgeVerdict(9, "值得回")],
            ["草稿1", "草稿2"],
            hook=lambda: bot._on_event(group_event(2, "糖糖你看这个", at_me=True)),
        )

        await bot._on_event(group_event(1, "讨论个话题"))
        await wait_until(lambda: len(bot._snowluma.sent) == 2, timeout=3)

        assert len(bot._ai.reply_calls) == 2  # 只有必答那次，无重生成
        assert bot._snowluma.sent[0][1] == "草稿1"
    finally:
        await bot.stop()


# ------------------------------------------------------------ 任务 B：观望重评


async def test_observe_rejudges_with_fresh_context_and_replies(tmp_path):
    """首评差一点没过线 → 观望到点后取最新上下文重判，达标则回复。"""
    bot = await build(
        tmp_path,
        generation_overrides={"recheck_enabled": False, "observe_delay_seconds": 0.5},
    )
    try:
        await tune_group(
            bot,
            proactivity_threshold=8,
            cooldown_seconds=0,
            min_gap_messages=0,
            busy_rate_per_min=0,
        )
        bot._ai = DraftQueuedAI(
            [
                JudgeVerdict(7, "差一点点"),      # msg1 首评：静默 + 安排观望
                JudgeVerdict(2, "无关话题"),      # msg2：正常判定，静默
                JudgeVerdict(9, "观望后确信"),    # msg1 二次判定：达标 → 回复
            ],
            ["好嘞"],
        )

        await bot._on_event(group_event(1, "半吊子话题"))
        await drain_tick()
        assert bot._ai.judge_calls == 1
        assert bot._snowluma.sent == []

        # 观望期间群里又来了新消息：二次判定应看得见它
        await bot._on_event(group_event(2, "另一个话题", uid=1001))
        await wait_until(lambda: len(bot._snowluma.sent) == 1, timeout=5)

        assert bot._ai.judge_calls == 3
        # 三次判定的上下文条数：首评只见 msg1；msg2 与观望重评都见两条
        assert bot._ai.judge_recent_lens == [1, 2, 2]
        assert bot._snowluma.sent[0][1] == "好嘞"
    finally:
        await bot.stop()


async def test_observe_at_most_once_per_message(tmp_path):
    """二次判定仍落在观望带内也不得安排第三次：每条消息至多观望一次。"""
    bot = await build(
        tmp_path,
        generation_overrides={"recheck_enabled": False, "observe_delay_seconds": 0.15},
    )
    try:
        await tune_group(
            bot,
            proactivity_threshold=8,
            cooldown_seconds=0,
            min_gap_messages=0,
            busy_rate_per_min=0,
        )
        bot._ai = DraftQueuedAI(
            [JudgeVerdict(7, "差一点点"), JudgeVerdict(7, "还是差一点")], ["不该发"],
        )

        await bot._on_event(group_event(1, "半吊子话题"))
        await wait_until(lambda: bot._ai.judge_calls == 2, timeout=5)
        await asyncio.sleep(0.4)  # 若再次安排观望，0.15 秒早该触发第三次

        assert bot._ai.judge_calls == 2
        assert bot._snowluma.sent == []
    finally:
        await bot.stop()


async def test_observe_cancelled_when_already_replied(tmp_path):
    """观望期间该消息一带已通过其他路径（@必答）回复过 → 取消重评。"""
    bot = await build(
        tmp_path,
        generation_overrides={"recheck_enabled": False, "observe_delay_seconds": 0.4},
    )
    try:
        await tune_group(
            bot,
            proactivity_threshold=8,
            cooldown_seconds=0,
            min_gap_messages=0,
            busy_rate_per_min=0,
        )
        bot._ai = DraftQueuedAI([JudgeVerdict(7, "差一点点")], ["必答稿"])

        await bot._on_event(group_event(1, "半吊子话题"))
        await drain_tick()
        # 观望期间被 @（另一条消息走必答路径，回复落在 msg1 之后）
        await bot._on_event(group_event(2, "糖糖你说呢", at_me=True))
        await wait_until(lambda: len(bot._snowluma.sent) == 1, timeout=3)

        await asyncio.sleep(0.6)  # 观望到点时刻已过
        assert bot._ai.judge_calls == 1  # msg2 必答不 judge；msg1 的观望被取消
        assert not bot._observe_tasks
        assert len(bot._snowluma.sent) == 1
    finally:
        await bot.stop()


async def test_observe_cancelled_when_message_recalled(tmp_path):
    """观望期间目标消息被撤回 → 取消重评，不再为它花判定。"""
    bot = await build(
        tmp_path,
        generation_overrides={"recheck_enabled": False, "observe_delay_seconds": 0.3},
    )
    try:
        await tune_group(
            bot,
            proactivity_threshold=8,
            cooldown_seconds=0,
            min_gap_messages=0,
            busy_rate_per_min=0,
        )
        bot._ai = DraftQueuedAI([JudgeVerdict(7, "差一点点")], ["不该发"])

        await bot._on_event(group_event(1, "半吊子话题"))
        await drain_tick()
        await bot._on_event(recall_event(1))

        await asyncio.sleep(0.6)
        assert bot._ai.judge_calls == 1
        assert bot._snowluma.sent == []
    finally:
        await bot.stop()


async def test_observe_skipped_when_guard_blocks(tmp_path):
    """被护栏（冷却）直接终止的消息不安排观望：首评都没走到分数线以下。"""
    bot = await build(
        tmp_path,
        generation_overrides={"recheck_enabled": False, "observe_delay_seconds": 0.2},
    )
    try:
        await tune_group(
            bot,
            proactivity_threshold=8,
            cooldown_seconds=60,  # 主动发言后进入冷却
            min_gap_messages=0,
            busy_rate_per_min=0,
        )
        bot._ai = DraftQueuedAI(
            [JudgeVerdict(9, "值得回"), JudgeVerdict(7, "差一点点")], ["回复"]
        )

        await bot._on_event(group_event(1, "聊个话题"))
        await wait_until(lambda: len(bot._snowluma.sent) == 1, timeout=3)

        await bot._on_event(group_event(2, "再来一条"))
        await drain_tick()
        await asyncio.sleep(0.4)  # 超过了观望延迟
        assert bot._ai.judge_calls == 2  # msg2 只被评了一次，没有观望重评
        assert bot._observe_tasks == {}
        assert (42, 2) not in bot._observed_once
    finally:
        await bot.stop()


async def test_observe_disabled_by_band_zero(tmp_path):
    """observe_band=0 关闭观望：落带内的分数直接放弃，不安排二次判定。"""
    bot = await build(
        tmp_path,
        generation_overrides={"recheck_enabled": False, "observe_band": 0},
    )
    try:
        await tune_group(
            bot,
            proactivity_threshold=8,
            cooldown_seconds=0,
            min_gap_messages=0,
            busy_rate_per_min=0,
        )
        bot._ai = DraftQueuedAI([JudgeVerdict(7, "差一点点")], ["不该发"])

        await bot._on_event(group_event(1, "半吊子话题"))
        await asyncio.sleep(0.3)
        assert bot._ai.judge_calls == 1
        assert bot._observe_tasks == {}
        assert bot._snowluma.sent == []
    finally:
        await bot.stop()


async def test_stop_cancels_pending_observe(tmp_path):
    """停机时未决的观望任务被取消、stop 不被 45 秒睡眠拖住。"""
    bot = await build(
        tmp_path,
        generation_overrides={"recheck_enabled": False, "observe_delay_seconds": 30},
    )
    await tune_group(
        bot,
        proactivity_threshold=8,
        cooldown_seconds=0,
        min_gap_messages=0,
        busy_rate_per_min=0,
    )
    bot._ai = DraftQueuedAI([JudgeVerdict(7, "差一点点")], ["不该发"])

    await bot._on_event(group_event(1, "半吊子话题"))
    await drain_tick()
    tasks = list(bot._observe_tasks.values())
    assert tasks, "观望任务应已挂起"

    started = time.monotonic()
    await asyncio.wait_for(bot.stop(), timeout=5)
    assert time.monotonic() - started < 2  # 不会被残余 sleep 阻塞
    assert all(t.done() for t in tasks)
    assert bot._observe_tasks == {}


# ------------------------------------------------------------ 任务 C：重复抑制


def test_already_replied_to_rules():
    """_already_replied_to 的判定规则逐条覆盖（见函数 docstring）。"""
    target = rec(1, 1000)
    other = rec(2, 1001)
    self_ = rec(-1, 99, "我回的话", self_=True)
    self2 = rec(-2, 99, "连发第二条", self_=True)

    # 规则 1：目标消息不在快照里（撤回/被淘汰）→ False
    assert _already_replied_to([other, self_], target) is False
    assert _already_replied_to([], target) is False
    # 规则 2：目标之后没有任何自己的发言 → False
    assert _already_replied_to([target], target) is False
    assert _already_replied_to([target, other], target) is False
    # 规则 4：目标之后已有自己的发言、对方没再开口 → True
    assert _already_replied_to([target, self_], target) is True
    assert _already_replied_to([target, other, self_], target) is True  # 他人不算
    assert _already_replied_to([self_, target, self2], target) is True
    # 规则 3：最后一条自己的发言之后目标用户又开口了 → False（对话往前走了）
    assert _already_replied_to([target, self_, rec(3, 1000)], target) is False
    assert _already_replied_to([target, rec(3, 1000), self_], target) is True


def test_self_reply_after_helper():
    """观望取消判定：目标在否 + 其后是否已有自己的发言。"""
    target = rec(1, 1000)
    self_ = rec(-1, 99, "答过了", self_=True)
    assert _self_reply_after([target], 1) == (True, False)
    assert _self_reply_after([target, self_], 1) == (True, True)
    assert _self_reply_after([self_, target], 1) == (True, False)  # 之前的发言不算
    assert _self_reply_after([self_], 1) == (False, False)  # 撤回后找不到目标


def test_repetition_block_format_and_byte_compat():
    """L4 注入格式：默认关闭时输出与引入前完全一致；开启时恰好多一块提醒。"""
    target = rec(1, 1000)
    plain = final_user_prompt_reply("2026-08-29 10:00:00", target, forced=True)
    warned = final_user_prompt_reply(
        "2026-08-29 10:00:00", target, forced=True, repetition_warning=True
    )
    block = "【重复提醒】你刚刚已经回复过这条消息，不要和之前的发言重复。"
    assert block == repetition_warning_block()
    assert "重复" not in plain
    assert block in warned
    # 除多出的提醒块（含其后的段落分隔）外逐字节一致
    assert warned.replace(block + "\n\n", "", 1) == plain


async def test_repetition_guard_injects_l4_warning(tmp_path):
    """同一条消息被再次处理时（@必答与主动路径叠加的边界情况），
    生成前在 L4 注入重复提醒；首次生成不受影响。"""
    bot = await build(tmp_path)
    try:
        await tune_group(
            bot, cooldown_seconds=0, min_gap_messages=0, busy_rate_per_min=0
        )
        bot._ai = DraftQueuedAI([JudgeVerdict(9, "值得回"), JudgeVerdict(9, "又回了")], ["第一次", "第二次"])

        await bot._on_event(group_event(1, "讨论个话题"))
        await wait_until(lambda: len(bot._snowluma.sent) == 1, timeout=3)
        assert bot._ai.reply_calls[-1]["repetition_warning"] is False

        # 再次处理同一条消息：msg1 之后已有自己的发言、对方没再开口
        memory = await bot._memory.get(42)
        target = next(r for r in memory.tail(20) if r.message_id == 1)
        await bot._decide_and_reply(42, NormalizedMessage(record=target, mentioned_me=False))
        await wait_until(lambda: len(bot._snowluma.sent) == 2, timeout=3)
        assert bot._ai.reply_calls[-1]["repetition_warning"] is True
    finally:
        await bot.stop()


async def test_repetition_guard_disabled_by_config(tmp_path):
    """repetition_guard_enabled=False 时完全不检查，行为与引入前一致。"""
    bot = await build(
        tmp_path, generation_overrides={"repetition_guard_enabled": False}
    )
    try:
        await tune_group(
            bot, cooldown_seconds=0, min_gap_messages=0, busy_rate_per_min=0
        )
        bot._ai = DraftQueuedAI(
            [JudgeVerdict(9, "值得回"), JudgeVerdict(9, "又回了")], ["第一次", "第二次"]
        )

        await bot._on_event(group_event(1, "讨论个话题"))
        await wait_until(lambda: len(bot._snowluma.sent) == 1, timeout=3)

        memory = await bot._memory.get(42)
        target = next(r for r in memory.tail(20) if r.message_id == 1)
        await bot._decide_and_reply(42, NormalizedMessage(record=target, mentioned_me=False))
        await wait_until(lambda: len(bot._snowluma.sent) == 2, timeout=3)
        assert all(c["repetition_warning"] is False for c in bot._ai.reply_calls)
    finally:
        await bot.stop()
