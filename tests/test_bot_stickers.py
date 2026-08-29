"""表情包跟发链路的集成风格单测：真实 CandyBot 编排 + 假 SnowLuma/AI。

覆盖：文字回复成功后的跟发与「[表情包]」占位写回、跟发前的挑图打字延迟、
概率 0 关闭、收藏为空不发、发送失败不影响记账（复用 test_integration 的
配置与假件基建；sleep 打桩手法沿用 test_bot_postprocess）。
"""

from __future__ import annotations

import asyncio
from dataclasses import replace as dc_replace

from candybot import bot as bot_module
from candybot.ai import JudgeVerdict
from candybot.bot import CandyBot
from candybot.models import StickerSettings
from candybot.postprocess import estimate_typing_time
from candybot.stickers import STICKER_RECORD_TEXT
from tests.test_integration import (
    FakeAI,
    FakeSnowluma,
    group_event,
    make_settings,
    wait_until,
)
from tests.test_stickers import _record_with, png_url

# wait_until 依赖真实 sleep 让出事件循环；打桩 sleep 之后必须仍交还控制权
_REAL_SLEEP = asyncio.sleep
# wait_until 的轮询步长：全局 sleep 打桩会被它一并记录，断言时需过滤
_POLL_STEP = 0.02


def _business_events(events: list) -> list:
    """剔除 wait_until 轮询产生的 sleep 记录，只留业务事件序列。"""
    return [(kind, p) for kind, p in events if not (kind == "sleep" and p == _POLL_STEP)]


class FailingOnSegments(FakeSnowluma):
    """模拟端点不支持本机 file:// 图片：段数组一律拒绝发送。"""

    async def send_group_msg(self, group_id: int, message):
        if isinstance(message, list):
            raise RuntimeError("endpoint rejects local file image segment")
        return await super().send_group_msg(group_id, message)


async def _build(
    tmp_path,
    monkeypatch,
    stickers: StickerSettings,
    *,
    snowluma: FakeSnowluma | None = None,
    post_process: dict | None = None,
) -> tuple[CandyBot, list]:
    """构造表情包测试机器人；asyncio.sleep 与群发消息都记录到 events。"""
    settings = dc_replace(
        make_settings(tmp_path, post_process=post_process), stickers=stickers
    )
    bot = CandyBot(settings)
    bot._snowluma = snowluma or FakeSnowluma()
    bot._ai = FakeAI(JudgeVerdict(9, "值得回"))
    events: list = []

    async def fake_sleep(delay):
        events.append(("sleep", delay))
        await _REAL_SLEEP(0)

    chained_send = bot._snowluma.send_group_msg

    async def record_send(group_id, message):
        events.append(("send", message))
        return await chained_send(group_id, message)

    monkeypatch.setattr(bot_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(bot._snowluma, "send_group_msg", record_send)
    return bot, events


async def _collect_one(bot: CandyBot) -> str:
    await bot._memory.get(42)  # 生产上 bot.start() 已建表；测试里手动触发
    url = png_url(64)
    record = _record_with(url, ts=100.0, summary="猫表情包")
    assert await bot._stickers.collect(record, (True,)) == 1
    return url


async def test_followup_send_and_placeholder_writeback(tmp_path, monkeypatch):
    """send_probability=1.0：文字发出后跟发 image 段，并写回「[表情包]」占位。"""
    bot, _ = await _build(tmp_path, monkeypatch, StickerSettings(send_probability=1.0))
    try:
        url = await _collect_one(bot)
        await bot._on_event(group_event(1, "糖糖你好", at_me=True))
        await wait_until(lambda: len(bot._snowluma.sent) == 2, timeout=3)

        assert bot._snowluma.sent[0] == (42, "哈哈确实")  # 文字先走现有链路
        gid, segments = bot._snowluma.sent[1]
        assert gid == 42
        assert segments[0]["type"] == "image"
        file_uri = segments[0]["data"]["file"]
        assert file_uri.startswith("file://")
        # file:// URI 指向刚收藏的那个表情包文件（内容指纹命名）
        entries = await bot._memory.db.load_stickers(42)
        assert entries and entries[0].sha256 in file_uri

        # 写回：模型历史里留下 is_self 的占位，不含路径与 base64
        memory = await bot._memory.get(42)
        tail = memory.tail(10)
        assert tail[-1].is_self and tail[-1].text == "[表情包]"
        assert tail[-1].images == ()
        # 使用统计记了一笔
        entries = await bot._memory.db.load_stickers(42)
        assert len(entries) == 1 and entries[0].use_count == 1
    finally:
        await bot.stop()


async def test_followup_send_waits_pick_delay(tmp_path, monkeypatch):
    """选中表情包后不秒发：先按「[表情包]」的估算打字时长 sleep 再发 image 段，
    事件序列严格为「文字 send → 挑图 sleep → 图片 send」。"""
    bot, events = await _build(tmp_path, monkeypatch, StickerSettings(send_probability=1.0))
    try:
        await _collect_one(bot)
        await bot._on_event(group_event(1, "糖糖你好", at_me=True))
        await wait_until(lambda: len(bot._snowluma.sent) == 2, timeout=3)

        speed = bot._settings.response_post_process.typing_speed  # 默认 1.0
        expected = estimate_typing_time(STICKER_RECORD_TEXT, speed)
        assert expected > 0
        assert _business_events(events) == [
            ("send", "哈哈确实"),
            ("sleep", expected),
            ("send", bot._snowluma.sent[1][1]),
        ]
    finally:
        await bot.stop()


async def test_followup_delay_zero_when_typing_speed_zero(tmp_path, monkeypatch):
    """typing_speed=0 时估算值即 0：完全不 sleep，跟发照常进行。"""
    bot, events = await _build(
        tmp_path,
        monkeypatch,
        StickerSettings(send_probability=1.0),
        post_process={"enabled": False, "typing_speed": 0.0},
    )
    try:
        await _collect_one(bot)
        await bot._on_event(group_event(1, "糖糖你好", at_me=True))
        await wait_until(lambda: len(bot._snowluma.sent) == 2, timeout=3)

        assert not [e for e in _business_events(events) if e[0] == "sleep"]
        assert bot._snowluma.sent[1][1][0]["type"] == "image"
    finally:
        await bot.stop()


async def test_send_probability_zero_follows_up_nothing(tmp_path, monkeypatch):
    bot, _ = await _build(tmp_path, monkeypatch, StickerSettings(send_probability=0.0))
    try:
        await _collect_one(bot)
        await bot._on_event(group_event(1, "糖糖你好", at_me=True))
        await wait_until(lambda: len(bot._snowluma.sent) == 1, timeout=3)
        await _REAL_SLEEP(0.15)  # 让出事件循环等待后台任务收尾（真实 sleep，不记入事件）
        assert len(bot._snowluma.sent) == 1  # 只发了文字
    finally:
        await bot.stop()


async def test_empty_collection_sends_nothing(tmp_path, monkeypatch):
    """掷点命中但该群收藏为空：静默跳过，不报错也不发东西（更不延迟）。"""
    bot, events = await _build(tmp_path, monkeypatch, StickerSettings(send_probability=1.0))
    try:
        await bot._on_event(group_event(1, "糖糖你好", at_me=True))
        await wait_until(lambda: len(bot._snowluma.sent) == 1, timeout=3)
        await _REAL_SLEEP(0.15)  # 让出事件循环等待后台任务收尾（真实 sleep，不记入事件）
        assert len(bot._snowluma.sent) == 1
        assert not [e for e in _business_events(events) if e[0] == "sleep"]
    finally:
        await bot.stop()


async def test_collection_disabled_skips_storage(tmp_path, monkeypatch):
    """enabled=False：既不收集（即便掷点在跟发处短路）也不跟发。"""
    bot, _ = await _build(
        tmp_path, monkeypatch, StickerSettings(enabled=False, send_probability=1.0)
    )
    try:
        record = _record_with(png_url(64), ts=100.0)
        assert await bot._stickers.collect(record, (True,)) == 0
        await bot._on_event(group_event(1, "糖糖你好", at_me=True))
        await wait_until(lambda: len(bot._snowluma.sent) == 1, timeout=3)
        await _REAL_SLEEP(0.15)  # 让出事件循环等待后台任务收尾（真实 sleep，不记入事件）
        assert len(bot._snowluma.sent) == 1
    finally:
        await bot.stop()


async def test_sticker_send_failure_does_not_disturb_reply(tmp_path, monkeypatch):
    """image 段被端点拒绝：文字回复的记账与记忆写回照常，仅不写占位。"""
    bot, _ = await _build(
        tmp_path,
        monkeypatch,
        StickerSettings(send_probability=1.0),
        snowluma=FailingOnSegments(),
    )
    try:
        await _collect_one(bot)
        await bot._on_event(group_event(1, "糖糖你好", at_me=True))
        await wait_until(lambda: len(bot._snowluma.sent) == 1, timeout=3)
        await _REAL_SLEEP(0.15)  # 让出事件循环等待后台任务收尾（真实 sleep，不记入事件）

        memory = await bot._memory.get(42)
        tail = memory.tail(10)
        assert tail[-1].is_self and tail[-1].text == "哈哈确实"  # 文字记账在
        assert not any(r.text == "[表情包]" for r in tail)  # 没发出去就不写占位
        # 护栏记账：发言间隔已归零（文字确实发出去了）
        assert bot._runtimes[42].msgs_since_reply == 0
    finally:
        await bot.stop()
