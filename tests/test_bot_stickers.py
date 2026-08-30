"""表情包跟发链路的集成风格单测：真实 CandyBot 编排 + 假 SnowLuma/AI。

覆盖：文字回复成功后的跟发与「[表情包]」占位写回、跟发前的挑图打字延迟、
概率 0 关闭、收藏为空不发、发送失败不影响记账；以及任务 2 的 smart 链路
（模型选图跟发、语境文本组装、选择不发作罢、选图失败退回随机、random 模式
下模型完全不参与）（复用 test_integration 的配置与假件基建；sleep 打桩
手法沿用 test_bot_postprocess）。
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import replace as dc_replace

from candybot import bot as bot_module
from candybot.ai import JudgeVerdict, StickerAssessment
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
from tests.test_stickers import _record_with, png_bytes, png_url

# wait_until 依赖真实 sleep 让出事件循环；打桩 sleep 之后必须仍交还控制权
_REAL_SLEEP = asyncio.sleep
# wait_until 的轮询步长：全局 sleep 打桩会被它一并记录，断言时需过滤
_POLL_STEP = 0.02


def _business_events(events: list) -> list:
    """剔除 wait_until 轮询产生的 sleep 记录，只留业务事件序列。"""
    return [(kind, p) for kind, p in events if not (kind == "sleep" and p == _POLL_STEP)]


class FailingOnSegments(FakeSnowluma):
    """模拟端点不支持 image 消息段：段数组一律拒绝发送。"""

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
        # 默认 base64 模式：图片字节内嵌在发送请求里，端点无需读本机文件
        file_ref = segments[0]["data"]["file"]
        assert file_ref.startswith("base64://")
        assert base64.b64decode(file_ref[len("base64://") :]) == png_bytes(64, 64)

        await _REAL_SLEEP(0.15)  # 占位写回在 image 段发出之后，让出事件循环等收尾
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


async def test_followup_send_http_mode(tmp_path, monkeypatch):
    """send_mode=http：image 段填事件服务的只读外链，跨机部署无需共享磁盘。"""
    bot, _ = await _build(
        tmp_path,
        monkeypatch,
        StickerSettings(
            send_probability=1.0,
            send_mode="http",
            http_base_url="http://192.168.1.20:5700",
        ),
    )
    try:
        await _collect_one(bot)
        await bot._on_event(group_event(1, "糖糖你好", at_me=True))
        await wait_until(lambda: len(bot._snowluma.sent) == 2, timeout=3)
        entries = await bot._memory.db.load_stickers(42)
        assert (
            bot._snowluma.sent[1][1][0]["data"]["file"]
            == f"http://192.168.1.20:5700/stickers/42/{entries[0].sha256}.png"
        )
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


# ---------------------------------------------------------------- smart 选图（任务 2）


class SmartFakeAI(FakeAI):
    """FakeAI + 表情包审核/选图两个能力：pick 决定选图结果，记录全部调用。"""

    def __init__(self, verdict: JudgeVerdict, *, pick=(0, "得意配得意"), pick_error=None):
        super().__init__(verdict)
        self.pick = pick
        self.pick_error = pick_error
        self.assess_calls: list[str] = []
        self.pick_calls: list[tuple[str, list[tuple[str, str]]]] = []

    async def assess_sticker(self, data_url: str):
        self.assess_calls.append(data_url)
        return StickerAssessment(True, "柴犬歪头疑惑", "无语")

    async def pick_sticker(self, context_text, entries):
        self.pick_calls.append((context_text, list(entries)))
        if self.pick_error is not None:
            raise self.pick_error
        return self.pick


async def test_smart_followup_sends_model_picked_sticker(tmp_path, monkeypatch):
    """smart 命中：掷点后把语境与候选交给模型，跟发它选中的那张。"""
    bot, _ = await _build(
        tmp_path,
        monkeypatch,
        StickerSettings(send_probability=1.0, select_mode="smart"),
    )
    ai = SmartFakeAI(JudgeVerdict(9, "值得回"), pick=(0, "语境相衬"))
    bot._ai = ai  # StickerStore 经回调现取，替换后立即可见
    try:
        url = await _collect_one(bot)  # 收藏走审核 → meta 入库
        assert ai.assess_calls == [url]
        await bot._on_event(group_event(1, "糖糖你好", at_me=True))
        await wait_until(lambda: len(bot._snowluma.sent) == 2, timeout=3)

        # 选图输入：最近聊天 + 明确标注的「刚发出的回复」，候选带描述与情绪
        context, entries = ai.pick_calls[0]
        assert "【你刚发出的】哈哈确实" in context
        assert "用户1000" in context  # 触发消息（group_event 的 card 昵称）
        assert entries == [("柴犬歪头疑惑", "无语")]
        # 模型选中的就是那张有 meta 的收藏；发送、记账与占位写回链路不变
        file_ref = bot._snowluma.sent[1][1][0]["data"]["file"]
        assert base64.b64decode(file_ref[len("base64://"):]) == png_bytes(64, 64)
        await _REAL_SLEEP(0.15)
        (entry,) = await bot._memory.db.load_stickers(42)
        assert entry.use_count == 1
    finally:
        await bot.stop()


async def test_smart_model_declines_abstains(tmp_path, monkeypatch, caplog):
    """模型回答「不发」：本次跟发作罢（只有文字那条发送），INFO 带理由。"""
    bot, _ = await _build(
        tmp_path,
        monkeypatch,
        StickerSettings(send_probability=1.0, select_mode="smart"),
    )
    ai = SmartFakeAI(JudgeVerdict(9, "值得回"), pick=(None, "没有一张和刚说的话相衬"))
    bot._ai = ai
    try:
        await _collect_one(bot)
        with caplog.at_level(logging.INFO):
            await bot._on_event(group_event(1, "糖糖你好", at_me=True))
            await wait_until(lambda: len(ai.pick_calls) == 1, timeout=3)
            await _REAL_SLEEP(0.15)
        assert len(bot._snowluma.sent) == 1  # 只发了文字
        assert "不跟发表情包" in caplog.text and "没有一张和刚说的话相衬" in caplog.text
    finally:
        await bot.stop()


async def test_smart_pick_failure_falls_back_to_random(tmp_path, monkeypatch, caplog):
    """选图调用失败：WARNING 后退回一次随机抽发，表情包照发（与现状一致）。"""
    bot, _ = await _build(
        tmp_path,
        monkeypatch,
        StickerSettings(send_probability=1.0, select_mode="smart"),
    )
    ai = SmartFakeAI(JudgeVerdict(9, "值得回"), pick_error=RuntimeError("learning 挂了"))
    bot._ai = ai
    try:
        await _collect_one(bot)
        with caplog.at_level(logging.WARNING):
            await bot._on_event(group_event(1, "糖糖你好", at_me=True))
            await wait_until(lambda: len(bot._snowluma.sent) == 2, timeout=3)
        assert bot._snowluma.sent[1][1][0]["type"] == "image"
        assert "退回随机抽选" in caplog.text
    finally:
        await bot.stop()


async def test_random_mode_never_calls_pick_sticker(tmp_path, monkeypatch):
    """select_mode=random（默认）回归：有 meta 也不动模型选图，与改动前一致。"""
    bot, _ = await _build(tmp_path, monkeypatch, StickerSettings(send_probability=1.0))
    ai = SmartFakeAI(JudgeVerdict(9, "值得回"))
    bot._ai = ai
    try:
        await _collect_one(bot)  # 收藏照常过审核入 meta
        assert len(ai.assess_calls) == 1
        await bot._on_event(group_event(1, "糖糖你好", at_me=True))
        await wait_until(lambda: len(bot._snowluma.sent) == 2, timeout=3)
        assert ai.pick_calls == []  # 掷点命中也走随机，不请求选图模型
        assert bot._snowluma.sent[1][1][0]["type"] == "image"
    finally:
        await bot.stop()
