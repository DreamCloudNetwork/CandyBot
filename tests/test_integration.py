"""全链路集成：真实 CandyBot 编排 + 假 SnowLuma/AI，验证事件→决策→发送。"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace as dc_replace

from candybot.bot import CandyBot
from candybot.memory import MemoryManager
from candybot.models import load_settings
from candybot.ai import JudgeVerdict, ReplyDraft
from tests.test_models_settings import DictCfg


def make_settings(tmp_path, generation_overrides: dict | None = None, *, post_process: dict | None = None):
    cfg = {
        "bot": {"self_qq": 99, "data_dir": str(tmp_path / "data")},
        "groups": {
            "42": {
                "persona": "你是群里的测试机器人。",
                "proactivity_threshold": 6,
                "cooldown_seconds": 60,
            }
        },
        "groups_default": {"enabled": False, "persona": "默认人设"},
        "ai_backend": {"base_url": "https://api.example.com/v1", "api_key": "k"},
        "models": {"judge": "j-model", "reply": "r-model"},
        "generation": dict(generation_overrides or {}),
        "multimodal": {},
        "rate_limit": {},
        "snowluma": {
            "endpoint": "http://10.0.0.5:3000/",
            "allow_private_endpoint": True,
        },
        # 本文件各用例断言的是未引入后处理时的整条单发行为，默认关闭拆条/
        # 错别字等随机加工；后处理自身的编排用例见 test_bot_postprocess.py
        "response_post_process": dict(post_process or {"enabled": False}),
    }
    return load_settings(DictCfg(cfg))


def group_event(mid: int, text: str, *, at_me=False, uid=1000, group_id=42):
    segments = []
    if at_me:
        segments.append({"type": "at", "data": {"qq": "99"}})
    segments.append({"type": "text", "data": {"text": text}})
    return {
        "post_type": "message",
        "message_type": "group",
        "group_id": group_id,
        "user_id": uid,
        "message_id": mid,
        "time": int(time.time()),
        "sender": {"card": f"用户{uid}", "nickname": f"u{uid}"},
        "message": segments,
    }


def recall_event(mid: int, *, uid=1000, group_id=42):
    return {
        "post_type": "notice",
        "notice_type": "group_recall",
        "group_id": group_id,
        "user_id": uid,
        "operator_id": uid,
        "message_id": mid,
        "time": int(time.time()),
    }


class FakeSnowluma:
    def __init__(self):
        # sent 记录原始 message 参数（str 或 OneBot 段数组）
        self.sent: list[tuple[int, object]] = []
        self._next_id = 1000

    async def start(self):
        pass

    async def probe(self):
        pass

    async def stop(self):
        pass

    async def query_login_info(self):
        return {"user_id": 99}

    async def send_group_msg(self, group_id: int, message) -> int:
        self.sent.append((group_id, message))
        self._next_id += 1
        return self._next_id


class FakeAI:
    # bot 据此选择 reply L1 守则的输出契约措辞
    reply_tool_use = True

    def __init__(self, verdict: JudgeVerdict):
        self.verdict = verdict
        self.judge_calls = 0
        self.reply_calls: list[dict] = []
        self.reconsider_calls: list[dict] = []

    async def judge_interest(self, *args, **kwargs) -> JudgeVerdict:
        self.judge_calls += 1
        return self.verdict

    async def generate_reply(self, static_system, runtime_system, recent,
                             current_message, now_text, *, forced=False,
                             engaged=False, score=None, reason="",
                             expression_hints=(), jargon_hints=(),
                             repetition_warning=False, person_hints=()):
        self.reply_calls.append({
            "static_system": static_system,
            "runtime_system": runtime_system,
            "recent_len": len(recent),
            # 模型实际收到的历史（含末条当前消息），供历史过滤类断言
            "recent_texts": [r.text for r in recent],
            "current": f"{current_message.nickname}：{current_message.text}",
            "forced": forced,
            "engaged": engaged,
            "score": score,
            "expression_hints": list(expression_hints),
            "jargon_hints": list(jargon_hints),
            "repetition_warning": repetition_warning,
            "person_hints": [
                (name, list(facts)) for name, facts in person_hints
            ],
        })
        return ReplyDraft("哈哈确实")

    async def reconsider_reply(self, static_system, runtime_system, recent,
                               now_text, *, sent_segments, pending_segments):
        self.reconsider_calls.append({
            "recent_len": len(recent),
            "sent": tuple(sent_segments),
            "pending": tuple(pending_segments),
        })
        # 默认「一字不改地继续」：bot 据此沿用原计划，等价于没有重想这回事
        return ReplyDraft("\n".join(pending_segments))

    async def describe_image(self, data_url: str):
        return "一张图"


class QueuedAI(FakeAI):
    """judge_interest 按预置序列逐条返回判定；耗尽后退回首个值。"""

    def __init__(self, verdicts: list[JudgeVerdict]):
        super().__init__(verdicts[0])
        self.verdicts = list(verdicts)
        self.judge_kwargs: list[dict] = []

    async def judge_interest(self, *args, **kwargs) -> JudgeVerdict:
        self.judge_kwargs.append(kwargs)
        self.judge_calls += 1
        return self.verdicts.pop(0) if self.verdicts else self.verdict


async def wait_until(cond, timeout=2.0, step=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        await asyncio.sleep(step)
    raise TimeoutError("条件未在超时前满足")


async def build_bot(tmp_path, verdict_score=None, generation_overrides=None):
    settings = make_settings(tmp_path, generation_overrides)
    bot = CandyBot(settings)
    bot._snowluma = FakeSnowluma()
    bot._ai = FakeAI(JudgeVerdict(verdict_score if verdict_score is not None else 9, "感兴趣"))
    return bot


async def drain_tick():
    await asyncio.sleep(0.15)


async def test_mention_triggers_forced_reply(tmp_path):
    bot = await build_bot(tmp_path)
    try:
        await bot._on_event(group_event(1, "你好呀", at_me=True))
        await wait_until(lambda: len(bot._snowluma.sent) == 1)

        gid, text = bot._snowluma.sent[0]
        assert gid == 42
        assert text == "哈哈确实"

        # @必答不应消耗 judge 调用
        assert bot._ai.judge_calls == 0
        # 回复调用的分层正确：L1 含人设；forced=True
        call = bot._ai.reply_calls[0]
        assert "测试机器人" in call["static_system"]
        assert call["forced"] is True
        assert "用户1000" in call["current"]
        # 自己的发言已写入记忆（is_self）
        memory = await bot._memory.get(42)
        assert memory.last().is_self
    finally:
        await bot.stop()


async def test_proactive_path_uses_judge_and_cooldown(tmp_path):
    bot = await build_bot(tmp_path)
    try:
        # 普通消息 → judge 打 9 分（阈值 6）→ 主动回复
        await bot._on_event(group_event(11, "有人在讨论 Rust 吗"))
        await wait_until(lambda: len(bot._snowluma.sent) == 1)
        assert bot._ai.judge_calls == 1
        _, text = bot._snowluma.sent[0]
        assert text == "哈哈确实"

        # 冷却期内的第二条高分消息：judge 照常评估（用来识别对话延续），
        # 但非延续对话 → 被冷却拦下，不发送
        await bot._on_event(group_event(12, "再来一条劲爆消息"))
        await drain_tick()
        assert bot._ai.judge_calls == 2
        assert len(bot._snowluma.sent) == 1
    finally:
        await bot.stop()


async def test_engagement_bypasses_cooldown_and_gap(tmp_path):
    """判定「在和我说话」的消息不受冷却与间隔限制，也不刷新主动冷却。"""
    bot = await build_bot(tmp_path)
    try:
        bot._ai = QueuedAI([
            JudgeVerdict(9, "新话题值得接"),            # evt11: 主动插话
            JudgeVerdict(8, "在追问刚才的话", True),     # evt12: 冷却+间隔期内延续对话
            JudgeVerdict(9, "冷却期里的无关话题"),        # evt13: 应被冷却拦下
        ])

        await bot._on_event(group_event(11, "聊聊游戏"))
        await wait_until(lambda: len(bot._snowluma.sent) == 1)

        # 群友文字接话（无 @）：冷却与间隔均在生效，但判定为与我对话 → 放行
        await bot._on_event(group_event(12, "那你觉得 Rust 难学吗"))
        await wait_until(lambda: len(bot._snowluma.sent) == 2)
        engaged_call = bot._ai.reply_calls[-1]
        assert engaged_call["forced"] is False
        assert engaged_call["engaged"] is True

        # 同一冷却窗口内的普通话题（非与我对话）仍被拦下
        await bot._on_event(group_event(13, "隔壁组的进度怎么样"))
        await drain_tick()
        assert bot._ai.judge_calls == 3
        assert len(bot._snowluma.sent) == 2
    finally:
        await bot.stop()


async def tune_group(bot: CandyBot, **overrides) -> None:
    """替换群 42 的配置（Settings.groups 是可变 dict），隔离护栏测试。"""
    profile = bot._settings.profile_for(42)
    bot._settings.groups[42] = dc_replace(profile, **overrides)


async def test_min_gap_blocks_immediate_rechatter(tmp_path):
    """主动发言后需攒够 min_gap_messages 条他人消息才会再次插话。"""
    bot = await build_bot(tmp_path)
    try:
        await tune_group(
            bot, cooldown_seconds=0, busy_rate_per_min=0, min_gap_messages=3
        )

        # 首条消息不受间隔约束（本进程从未发言）→ 正常判定并回复
        await bot._on_event(group_event(11, "聊点啥"))
        await wait_until(lambda: len(bot._snowluma.sent) == 1)
        assert bot._ai.judge_calls == 1

        # 发言后的头 3 条消息：judge 照常评估（防漏掉对我的喊话），
        # 但不属于对话延续 → 被间隔护栏拦下
        for mid in (12, 13, 14):
            await bot._on_event(group_event(mid, "继续聊"))
            await drain_tick()
            assert len(bot._snowluma.sent) == 1
        assert bot._ai.judge_calls == 4

        # 第 4 条起恢复允许插话，高分消息正常回复
        await bot._on_event(group_event(15, "再来一条"))
        await wait_until(lambda: len(bot._snowluma.sent) == 2)
        assert bot._ai.judge_calls == 5
    finally:
        await bot.stop()


async def test_busy_group_suppresses_proactive_only(tmp_path):
    """热闹期的主动插话被压制；对话延续依旧放行。"""
    bot = await build_bot(tmp_path, verdict_score=9)
    try:
        await tune_group(
            bot,
            cooldown_seconds=0,
            min_gap_messages=0,
            busy_rate_per_min=3,
            proactivity_threshold=6,
        )
        # 第一条：未达热闹阈值 → 正常回复（此过程计入窗口）
        await bot._on_event(group_event(21, "聊个新技术"))
        await wait_until(lambda: len(bot._snowluma.sent) == 1)

        bot._ai = QueuedAI([
            JudgeVerdict(9, "无关联的高分话题", False),   # evt22: 窗口 2 条 < 3 → 放行
            JudgeVerdict(8, "热闹里的又一话题", False),   # evt23: 窗口满 → 拦下
            JudgeVerdict(7, "热闹中有人叫我", True),      # evt24: 对话延续 → 放行
        ])

        await bot._on_event(group_event(22, "这条不算热闹"))
        await wait_until(lambda: len(bot._snowluma.sent) == 2)

        await bot._on_event(group_event(23, "群里开始刷屏了"))
        await drain_tick()
        assert bot._ai.judge_calls == 2          # 已评估，但被热闹护栏拦下
        assert len(bot._snowluma.sent) == 2

        await bot._on_event(group_event(24, "糖糖你说是不是", at_me=False))
        await wait_until(lambda: len(bot._snowluma.sent) == 3)
        assert bot._ai.reply_calls[-1]["engaged"] is True
    finally:
        await bot.stop()


async def test_low_score_no_reply(tmp_path):
    bot = await build_bot(tmp_path, verdict_score=2)
    try:
        await bot._on_event(group_event(21, "随便聊聊天气"))
        await drain_tick()
        assert bot._ai.judge_calls == 1
        assert bot._snowluma.sent == []
        # 但消息本身已进上下文
        assert len(await bot._memory.get(42)) == 1
    finally:
        await bot.stop()


# ------------------------------------------------------------ 门槛复核


async def test_recheck_confirms_and_replies(tmp_path):
    """首评分数落在 5 与门槛之间时复核；复评确信则按复评结果回复。"""
    bot = await build_bot(tmp_path)
    try:
        await tune_group(bot, proactivity_threshold=8)
        bot._ai = QueuedAI([
            JudgeVerdict(7, "话题开放可以接"),   # 首评：5 < 7 < 8 → 触发复核
            JudgeVerdict(9, "确有可说之处"),     # 复评：确认值得回复
        ])

        await bot._on_event(group_event(71, "群友在问开源方案"))
        await wait_until(lambda: len(bot._snowluma.sent) == 1)

        assert bot._ai.judge_calls == 2
        # 复核请求必须携带首评结论与本群真实门槛
        kw = bot._ai.judge_kwargs[-1]
        assert kw["prev_verdict"].score == 7
        assert kw["threshold"] == 8
        # 回复层采用复评的分数
        assert bot._ai.reply_calls[-1]["score"] == 9
    finally:
        await bot.stop()


async def test_recheck_downgrade_keeps_silent(tmp_path):
    """复评不认可首评的高分 → 维持沉默，消息照常入记忆。"""
    bot = await build_bot(tmp_path)
    try:
        await tune_group(bot, proactivity_threshold=8)
        bot._ai = QueuedAI([
            JudgeVerdict(7, "看起来挺热闹"),
            JudgeVerdict(3, "没人等我，只是想插话"),
        ])

        await bot._on_event(group_event(72, "随便唠两句"))
        await drain_tick()

        assert bot._ai.judge_calls == 2
        assert bot._snowluma.sent == []
        assert len(await bot._memory.get(42)) == 1
    finally:
        await bot.stop()


async def test_scores_outside_band_skip_recheck(tmp_path):
    """达到门槛或不超过中线的分数各只调一次 judge，不触发复核。"""
    bot = await build_bot(tmp_path)
    try:
        await tune_group(
            bot,
            proactivity_threshold=8,
            cooldown_seconds=0,
            min_gap_messages=0,
            busy_rate_per_min=0,
        )
        bot._ai = QueuedAI([
            JudgeVerdict(9, "明确该回"),   # 直接达标 → 无需复核
            JudgeVerdict(5, "可回可不回"),  # 未过中线 → 无需复核
        ])

        await bot._on_event(group_event(81, "有人在等我答话"))
        await wait_until(lambda: len(bot._snowluma.sent) == 1)

        await bot._on_event(group_event(82, "日常播报一条"))
        await drain_tick()

        assert bot._ai.judge_calls == 2
        assert len(bot._snowluma.sent) == 1
    finally:
        await bot.stop()


async def test_recheck_failure_defaults_to_silent(tmp_path):
    """复核调用异常时安全侧处理：维持首评（未达标）不发言。"""

    class ErrOnRecheckAI(FakeAI):
        async def judge_interest(self, *args, **kwargs) -> JudgeVerdict:
            self.judge_calls += 1
            if self.judge_calls >= 2:
                raise RuntimeError("网络中断")
            return JudgeVerdict(7, "似乎可以接一句")

    bot = await build_bot(tmp_path)
    try:
        await tune_group(bot, proactivity_threshold=8)
        bot._ai = ErrOnRecheckAI(JudgeVerdict(0, ""))

        await bot._on_event(group_event(83, "聊两句"))
        await drain_tick()

        assert bot._ai.judge_calls == 2
        assert bot._snowluma.sent == []
    finally:
        await bot.stop()


async def test_recheck_disabled_by_config(tmp_path):
    """generation.recheck_enabled=false → 首评落带内也直接采信，不二次调用。"""
    bot = await build_bot(
        tmp_path,
        generation_overrides={"recheck_enabled": False},
    )
    try:
        await tune_group(bot, proactivity_threshold=8)
        bot._ai = QueuedAI([JudgeVerdict(7, "看起来挺热闹")])

        await bot._on_event(group_event(84, "群友在问开源方案"))
        await drain_tick()

        assert bot._ai.judge_calls == 1
        assert bot._snowluma.sent == []
        assert len(await bot._memory.get(42)) == 1
    finally:
        await bot.stop()


async def test_recheck_min_score_from_config(tmp_path):
    """复评下限可配置：首评须严格高于该下限才进入复核。"""
    bot = await build_bot(tmp_path, generation_overrides={"recheck_min_score": 6})
    try:
        await tune_group(bot, proactivity_threshold=8)

        # 首评 6 分：不高于下限 6 → 不复核，直接按未达标静默
        bot._ai = QueuedAI([JudgeVerdict(6, "可回可不回")])
        await bot._on_event(group_event(85, "随便聊聊"))
        await drain_tick()
        assert bot._ai.judge_calls == 1
        assert bot._snowluma.sent == []

        # 首评 7 分：高于下限且低于门槛 → 触发复核；复评压分则维持安静
        bot._ai = QueuedAI([
            JudgeVerdict(7, "话题开放可以接"),
            JudgeVerdict(5, "其实没什么兴趣"),
        ])
        await bot._on_event(group_event(86, "再聊两句"))
        await drain_tick()
        assert bot._ai.judge_calls == 2
        assert bot._snowluma.sent == []
        kw = bot._ai.judge_kwargs[-1]
        assert kw["min_score"] == 6
        assert kw["prev_verdict"].score == 7
        assert kw["threshold"] == 8
    finally:
        await bot.stop()


async def test_whitelist_blocks_other_groups(tmp_path):
    bot = await build_bot(tmp_path)
    try:
        await bot._on_event(group_event(31, "隔壁群说话", group_id=777, at_me=True))
        await drain_tick()
        assert bot._snowluma.sent == []
        assert 777 not in bot._group_queues
    finally:
        await bot.stop()


async def test_duplicate_events_processed_once(tmp_path):
    bot = await build_bot(tmp_path)
    try:
        evt = group_event(41, "重复推送", at_me=True)
        await bot._on_event(evt)
        await bot._on_event(evt)  # 同一事件重复上报
        await wait_until(lambda: len(bot._snowluma.sent) == 1)
        await drain_tick()
        assert len(bot._snowluma.sent) == 1      # 只回了一次
    finally:
        await bot.stop()


async def test_group_recall_removes_local_record(tmp_path):
    bot = await build_bot(tmp_path)
    try:
        await bot._on_event(group_event(51, "这条稍后会被撤回"))
        await drain_tick()  # 决策链路跑完（高分消息会合成回复入记忆）
        await bot._on_event(recall_event(51))
        memory = await bot._memory.get(42)
        assert all(r.message_id != 51 for r in memory.tail(20))
        # 库里同步删除，重启后不会复活
        mgr2 = MemoryManager(tmp_path / "data")
        mem2 = await mgr2.get(42)
        assert all(r.message_id != 51 for r in mem2.tail(20))
        await mgr2.close()
    finally:
        await bot.stop()


async def test_recall_ignores_unknown_and_other_notices(tmp_path):
    bot = await build_bot(tmp_path, verdict_score=2)
    try:
        await bot._on_event(group_event(61, "留在记忆里"))
        # 撤回一条从未记录过的消息：不应报错，也不影响已有记录
        await bot._on_event(recall_event(9999))
        # 其他通知类型（如入群）应被静默忽略
        other = {
            "post_type": "notice",
            "notice_type": "group_increase",
            "group_id": 42,
            "user_id": 61,
            "time": int(time.time()),
        }
        await bot._on_event(other)
        memory = await bot._memory.get(42)
        assert [r.message_id for r in memory.tail(10)] == [61]
    finally:
        await bot.stop()
