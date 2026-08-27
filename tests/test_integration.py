"""全链路集成：真实 CandyBot 编排 + 假 MCP/AI，验证事件→决策→发送。"""

from __future__ import annotations

import asyncio
import time

from candybot.bot import CandyBot
from candybot.models import load_settings
from candybot.ai import JudgeVerdict
from tests.test_models_settings import DictCfg


def make_settings(tmp_path):
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
        "generation": {},
        "multimodal": {},
        "rate_limit": {},
        "snowluma": {
            "endpoint": "http://10.0.0.5:3000/",
            "mode": "write",
            "allow_private_endpoint": True,
        },
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


class FakeSnowluma:
    def __init__(self):
        self.sent: list[tuple[int, str]] = []

    async def start(self):
        pass

    async def probe(self):
        pass

    async def stop(self):
        pass

    async def query_login_info(self):
        return {"user_id": 99}

    async def send_group_msg(self, group_id: int, text: str) -> None:
        self.sent.append((group_id, text))


class FakeAI:
    def __init__(self, verdict: JudgeVerdict):
        self.verdict = verdict
        self.judge_calls = 0
        self.reply_calls: list[dict] = []

    async def judge_interest(self, *args, **kwargs) -> JudgeVerdict:
        self.judge_calls += 1
        return self.verdict

    async def generate_reply(self, static_system, runtime_system, recent,
                             current_message, now_text, *, forced=False,
                             score=None, reason=""):
        self.reply_calls.append({
            "static_system": static_system,
            "runtime_system": runtime_system,
            "recent_len": len(recent),
            "current": f"{current_message.nickname}：{current_message.text}",
            "forced": forced,
            "score": score,
        })
        return "哈哈确实"

    async def describe_image(self, data_url: str):
        return "一张图"


async def wait_until(cond, timeout=2.0, step=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        await asyncio.sleep(step)
    raise TimeoutError("条件未在超时前满足")


async def build_bot(tmp_path, verdict_score=None):
    settings = make_settings(tmp_path)
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
        assert bot._memory.get(42).last().is_self
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

        # 冷却期内第二条高分消息：judge 不再调用、不发送
        await bot._on_event(group_event(12, "再来一条劲爆消息"))
        await drain_tick()
        assert bot._ai.judge_calls == 1          # 冷却前置短路，judge 未被调用
        assert len(bot._snowluma.sent) == 1      # 也没有新发言
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
        assert len(bot._memory.get(42)) == 1
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
