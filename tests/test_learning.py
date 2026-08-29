"""记忆与学习机制：溢出触发收集、每日印象、表达/黑话学习与 L2/L4 注入。

全部 LLM 调用以桩对象/mock 替代，不发真实请求。
"""

from __future__ import annotations

import time
from dataclasses import replace as dc_replace
from datetime import date, datetime, timedelta

import pytest

from candybot.ai import AIClient
from candybot.bot import CandyBot, GroupRuntime
from candybot.learning import LearningService, day_bounds
from candybot.memory import GroupMemory, MemoryManager
from candybot.models import (
    ChatRecord,
    GenerationSettings,
    ModelConfig,
    ModelSettings,
    Settings,
    load_settings,
)
from candybot.prompts import (
    expression_hint_block,
    final_user_prompt_reply,
    jargon_hint_block,
    learning_chat_text,
    runtime_system_prompt,
)
from tests.deterministic_rng import SeededRng
from tests.test_models_settings import DictCfg


# ---------------------------------------------------------------- 公共构造


def make_record(mid: int, text: str = "hello", *, group_id: int = 42, **kw) -> ChatRecord:
    base = dict(
        message_id=mid,
        group_id=group_id,
        user_id=1000 + mid,
        nickname=f"u{mid}",
        text=text,
        ts=time.time() + mid,
    )
    base.update(kw)
    return ChatRecord(**base)


def make_settings(tmp_path, **learning_over) -> Settings:
    cfg = {
        "bot": {"self_qq": 99, "data_dir": str(tmp_path / "data")},
        "groups": {"42": {"persona": "测试人设", "proactivity_threshold": 6}},
        "groups_default": {"enabled": False, "persona": "默认人设"},
        "ai_backend": {"base_url": "https://api.example.com/v1", "api_key": "k"},
        "models": {"judge": "j", "reply": "r"},
        "generation": {},
        "multimodal": {},
        "rate_limit": {},
        "snowluma": {
            "endpoint": "http://10.0.0.5:3000/",
            "allow_private_endpoint": True,
        },
    }
    if learning_over:
        cfg["learning"] = dict(learning_over)
    return load_settings(DictCfg(cfg))


class StubAI:
    """learning 角色各方法的桩实现：返回值可编程、调用全部计数。"""

    def __init__(self):
        self.expression_batches: list[str] = []
        self.jargon_batches: list[str] = []
        self.reviews: list[tuple[str, str]] = []
        self.inferences: list[str] = []
        self.alone_calls: list[str] = []
        self.compare_calls: list[tuple[str, str]] = []
        self.impressions: list[tuple[str, str]] = []
        self.expressions: list[tuple[str, str]] = []
        self.review_suitable = True
        self.jargon_terms: list[str] = []
        # term -> (meaning, no_info)
        self.with_context: dict[str, tuple[str | None, bool]] = {}
        self.alone: dict[str, str | None] = {}
        self.similar_terms: set[str] = set()
        self.impression_text = "那是一段今天的群聊印象。"

    async def learn_expressions(self, chat_text: str):
        self.expression_batches.append(chat_text)
        return list(self.expressions)

    async def review_expression(self, situation: str, style: str) -> bool:
        self.reviews.append((situation, style))
        return self.review_suitable

    async def extract_jargon_terms(self, chat_text: str):
        self.jargon_batches.append(chat_text)
        return list(self.jargon_terms)

    async def infer_jargon_with_context(self, term: str, context_text: str):
        self.inferences.append(term)
        return self.with_context.get(term, ("通用含义", False))

    async def infer_jargon_alone(self, term: str):
        self.alone_calls.append(term)
        return self.alone.get(term, "通用含义")

    async def compare_jargon_inference(self, m1: str, m2: str) -> bool:
        self.compare_calls.append((m1, m2))
        # 两次含义一致才判 similar
        return m1 == m2 and m1 != ""

    async def summarize_impression(self, day: str, chat_text: str, max_chars: int):
        self.impressions.append((day, chat_text))
        return self.impression_text


@pytest.fixture
async def mgr(tmp_path):
    manager = MemoryManager(tmp_path)
    await manager.db.create_tables()
    yield manager
    await manager.close()


def make_service(mgr, settings, ai=None) -> LearningService:
    return LearningService(mgr, lambda: ai, lambda: settings)


async def drain_batch(svc: LearningService, group_id: int) -> None:
    task = svc._batch_tasks.get(group_id)
    if task is not None:
        await task


# ---------------------------------------------------------------- 任务 B-1：溢出触发收集


async def test_group_memory_evicts_oldest_to_listener(tmp_path):
    evicted: list[tuple[int, int]] = []
    manager = MemoryManager(tmp_path, default_capacity=8)
    await manager.db.create_tables()
    manager.evict_listener = lambda gid, rec: evicted.append((gid, rec.message_id))
    mem = await manager.get(42)
    for i in range(12):
        await mem.append(make_record(i))
    # 容量 8，第 9..12 条入队时依次挤出最旧的 0..3 条
    assert evicted == [(42, 0), (42, 1), (42, 2), (42, 3)]
    assert [r.message_id for r in mem.tail(10)] == list(range(4, 12))
    # 撤回不触发淘汰回调（删除不是溢出）
    await mem.remove(5)
    assert len(evicted) == 4
    await manager.close()


async def test_evict_not_fired_on_duplicate_or_priming(tmp_path):
    evicted: list[int] = []
    manager = MemoryManager(tmp_path, default_capacity=8)
    await manager.db.create_tables()
    manager.evict_listener = lambda gid, rec: evicted.append(rec.message_id)
    mem = await manager.get(42)
    for i in range(8):
        await mem.append(make_record(i))
    assert evicted == []
    await mem.append(make_record(3))  # 重复入库：缓存未增长，不触发
    assert evicted == []
    # 启动回放导致的挤出不算实时淘汰
    replay = GroupMemory(43, manager.db, 8)
    replay._prime([make_record(100 + i, group_id=43) for i in range(20)])
    await manager.close()


async def test_pending_accumulates_until_batch_threshold(tmp_path, mgr):
    settings = make_settings(tmp_path, expression_batch_size=10)
    ai = StubAI()
    ai.expressions = [("表示惊叹", "我嘞个豆")]
    svc = make_service(mgr, settings, ai)
    for i in range(9):
        svc.note_evicted(42, make_record(i))
    assert 42 not in svc._batch_tasks  # 不足 10 条不触发
    assert ai.expression_batches == []
    svc.note_evicted(42, make_record(9))
    await drain_batch(svc, 42)
    assert len(ai.expression_batches) == 1
    learned = await mgr.db.load_expressions(42)
    assert [(e.situation, e.style) for e in learned] == [("表示惊叹", "我嘞个豆")]
    await svc.stop()


async def test_expression_learning_filters_and_reviews(tmp_path, mgr):
    settings = make_settings(tmp_path)
    ai = StubAI()
    ai.expressions = [("表示惊叹", "我嘞个豆"), ("敷衍应答", "呃呃")]
    svc = make_service(mgr, settings, ai)
    await svc._learn_expressions(42, [make_record(i) for i in range(10)])
    # 默认开启自审：每条候选各审一次
    assert ai.reviews == ai.expressions
    assert len(await mgr.db.load_expressions(42)) == 2

    # 自审拒收：不入库；关闭开关：不再调用 review
    ai.review_suitable = False
    await svc._learn_expressions(42, [make_record(i) for i in range(10)])
    assert len(await mgr.db.load_expressions(42)) == 2
    no_review = dc_replace(
        settings, learning=dc_replace(settings.learning, expression_self_review=False)
    )
    svc2 = make_service(mgr, no_review, ai)
    before = len(ai.reviews)
    await svc2._learn_expressions(42, [make_record(i) for i in range(10)])
    assert len(ai.reviews) == before
    entries = await mgr.db.load_expressions(42)
    assert len(entries) == 2 and all(e.weight == 2 for e in entries)  # 不审直接入库、权重累计
    await svc2.stop()


async def test_expression_learning_excludes_self_text(tmp_path, mgr):
    """学习文本里机器人自己的发言标成【你自己】，供提示词排除。"""
    settings = make_settings(tmp_path)
    ai = StubAI()
    svc = make_service(mgr, settings, ai)
    await svc._learn_expressions(
        42,
        [make_record(1, "群友的话"), make_record(2, "我说的", is_self=True, nickname="糖糖")],
    )
    text = ai.expression_batches[0]
    assert "u1(1001)：群友的话" in text
    assert "【你自己】：我说的" in text
    await svc.stop()


async def test_expression_dedup_accumulates_weight(tmp_path, mgr):
    settings = make_settings(tmp_path, expression_batch_size=10)
    ai = StubAI()
    ai.expressions = [("表示惊叹", "我嘞个豆")]
    svc = make_service(mgr, settings, ai)
    for _ in range(2):
        for i in range(10):
            svc.note_evicted(42, make_record(i))
        await drain_batch(svc, 42)
    learned = await mgr.db.load_expressions(42)
    assert len(learned) == 1
    assert learned[0].weight == 2  # 同群按内容去重，权重累计
    await svc.stop()


# ---------------------------------------------------------------- 任务 B/C-4：L4 注入


def test_l2_impression_snapshot_stable_within_day():
    today = "2026-08-29"
    runtime = GroupRuntime()
    assert runtime.impressions_snapshot(today) is None  # 未缓存要求刷新
    snapshot = (("2026-08-27", "印象甲"), ("2026-08-28", "印象乙"))
    runtime.remember_impressions(today, snapshot)
    a = runtime_system_prompt(42, today, ["甲"], impressions=runtime.impressions_snapshot(today))
    b = runtime_system_prompt(42, today, ["甲"], impressions=runtime.impressions_snapshot(today))
    assert a == b  # 同一天重建字节级相同
    assert "【最近群聊印象】" in a
    assert a.endswith("2026-08-27：印象甲\n2026-08-28：印象乙")
    assert runtime.impressions_snapshot("2026-08-30") is None  # 跨天要刷新
    # 无印象时与引入机制之前完全一致
    assert runtime_system_prompt(42, today, ["甲"]) == runtime_system_prompt(
        42, today, ["甲"], impressions=()
    )


async def test_bot_impression_snapshot_byte_stable(tmp_path):
    bot = CandyBot(make_settings(tmp_path))
    try:
        await bot._memory.db.create_tables()
        runtime = bot._runtimes[42]
        day = (date.today() - timedelta(days=1)).isoformat()
        await bot._memory.db.save_impression(42, day, "第一版印象")
        first = await bot._impressions_for(42, runtime)
        # 天内库里内容变了也不重取：天内字节级稳定
        await bot._memory.db.save_impression(42, day, "第二版印象")
        second = await bot._impressions_for(42, runtime)
        assert first == second == ((day, "第一版印象"),)
        # 跨过零点：快照键变化后重取
        runtime.remember_impressions("2000-01-01", ())
        third = await bot._impressions_for(42, runtime)
        assert third == ((day, "第二版印象"),)
        # 关闭开关后不注入
        bot._settings = dc_replace(
            bot._settings,
            learning=dc_replace(bot._settings.learning, impression_enabled=False),
        )
        assert await bot._impressions_for(42, runtime) == ()
    finally:
        await bot.stop()


# ------------------------------------------------------ 零点竞态防护


async def test_impression_waits_for_yesterday_before_solidifying(tmp_path):
    """昨天有聊天但印象尚未就位：快照暂不固化，就位后重取并固化。"""
    bot = CandyBot(make_settings(tmp_path))
    try:
        await bot._memory.db.create_tables()
        runtime = bot._runtimes[42]
        today = date.today().isoformat()
        yesterday = date.today() - timedelta(days=1)
        start_ts, end_ts = day_bounds(yesterday)
        memory = await bot._memory.get(42)
        await memory.append(
            make_record(5, "昨天的闲聊", ts=start_ts + (end_ts - start_ts) / 2)
        )

        # 印象还没生成：返回当前不完整的快照但不固化
        assert await bot._impressions_for(42, runtime) == ()
        assert runtime.impressions_snapshot(today) is None

        # 印象就位：下一次取用即完成固化
        await bot._memory.db.save_impression(42, yesterday.isoformat(), "昨日印象")
        got = await bot._impressions_for(42, runtime)
        assert got == ((yesterday.isoformat(), "昨日印象"),)
        assert runtime.impressions_snapshot(today) == got

        # 固化之后，天内库里内容再变也不重取（字节级稳定恢复）
        await bot._memory.db.save_impression(42, yesterday.isoformat(), "改写的印象")
        assert await bot._impressions_for(42, runtime) == got
    finally:
        await bot.stop()


async def test_impression_no_yesterday_activity_caches_immediately(tmp_path):
    """昨天群里没聊天：空快照照常固化，不会被竞态防护拦着天天重查。"""
    bot = CandyBot(make_settings(tmp_path))
    try:
        await bot._memory.db.create_tables()
        runtime = bot._runtimes[42]
        assert await bot._impressions_for(42, runtime) == ()
        assert runtime.impressions_snapshot(date.today().isoformat()) == ()
    finally:
        await bot.stop()


async def test_has_day_records(tmp_path, mgr):
    """has_day_records 按 [start, end) 半开区间判断，且区分群。"""
    memory = await mgr.get(42)
    yesterday = date.today() - timedelta(days=1)
    start_ts, end_ts = day_bounds(yesterday)
    await memory.append(make_record(6, "午休闲聊", ts=(start_ts + end_ts) / 2))
    assert await mgr.db.has_day_records(42, start_ts, end_ts) is True
    assert await mgr.db.has_day_records(42, end_ts, end_ts + 3600.0) is False
    assert await mgr.db.has_day_records(99, start_ts, end_ts) is False


def test_l4_hint_blocks_and_injection_format():
    msg = make_record(7, "当前消息")
    exprs = [("表示惊叹", "我嘞个豆"), ("讽刺赞同", "对对对")]
    jargons = [("yyds", "永远的神")]
    prompt = final_user_prompt_reply(
        "2026-08-29 10:00:00",
        msg,
        forced=True,
        expression_hints=exprs,
        jargon_hints=jargons,
    )
    assert expression_hint_block(exprs) in prompt
    assert jargon_hint_block(jargons) in prompt
    assert prompt.index("【需要回应的消息】") < prompt.index("【表达习惯参考") < prompt.index("【黑话参考】")
    assert '当"表示惊叹"时，可以用"我嘞个豆"' in prompt
    assert "不用完全遵守" in prompt
    assert "yyds：永远的神" in prompt
    # 无注入时与旧输出字节级一致（提示词契约尾部紧跟消息块）
    plain = final_user_prompt_reply("2026-08-29 10:00:00", msg, forced=True)
    assert "【表达习惯参考" not in plain and "【黑话参考】" not in plain
    assert plain.endswith(f"【需要回应的消息】来自 u7(1007)：\n当前消息\n\n{plain.splitlines()[-1]}")


async def test_learning_hints_injected_into_reply_call(tmp_path):
    bot = CandyBot(make_settings(tmp_path))
    try:
        await bot._memory.db.create_tables()
        await bot._memory.db.record_expression(42, "表示惊叹", "我嘞个豆", time.time())
        await bot._memory.db.record_jargon(42, "nb", "厉害", time.time(), 50)
        hints, jargons = await bot._learning_hints(42, [make_record(1, "这个操作真nb")])
        assert hints == [("表示惊叹", "我嘞个豆")]
        assert jargons == [("nb", "厉害")]
        # 关闭学习总开关后不注入
        bot._settings = dc_replace(
            bot._settings, learning=dc_replace(bot._settings.learning, enabled=False)
        )
        assert await bot._learning_hints(42, [make_record(1, "这个操作真nb")]) == ([], [])
    finally:
        await bot.stop()


async def test_pick_expressions_weighted_and_touched(tmp_path, mgr):
    settings = make_settings(tmp_path)
    svc = make_service(mgr, settings, StubAI())
    svc.rng = SeededRng(42)  # 固定种子保证可重复
    now = time.time()
    for i in range(5):
        await mgr.db.record_expression(42, f"情境{i}", f"风格{i}", now - 100 + i)
    # 加权随机抽 ≤3 条、无重复
    picks = await svc.pick_expressions(42, 3)
    assert len(picks) == 3
    assert len({p for p in picks}) == 3
    # 被选中的条目刷新 last_active_time
    entries = {e.situation: e for e in await mgr.db.load_expressions(42)}
    assert all(entries[s].last_active_time > now for s, _ in picks)
    unselected = [e for e in entries.values() if (e.situation, e.style) not in picks]
    assert all(e.last_active_time < now for e in unselected)
    assert await svc.pick_expressions(42, 0) == []
    await svc.stop()


# ---------------------------------------------------------------- 任务 C：黑话


async def test_jargon_double_inference_agreement_gate(tmp_path, mgr):
    """两次推断一致才入库；信息不足/仅词条无果/不一致都拒收。"""
    settings = make_settings(tmp_path)
    ai = StubAI()
    ai.jargon_terms = ["一致的", "不一致的", "信息不足的", "仅词条没懂的"]
    ai.with_context = {
        "一致的": ("圈内含义", False),
        "不一致的": ("上下文含义A", False),
        "信息不足的": (None, True),
        "仅词条没懂的": ("有含义", False),
    }
    ai.alone = {
        "一致的": "圈内含义",  # 与第一路相同 → similar → 入库
        "不一致的": "词条含义B",  # 不同 → 不入库
        "信息不足的": "",
        "仅词条没懂的": None,  # 第二路没结果 → 不入库
    }
    svc = make_service(mgr, settings, ai)
    await svc._learn_jargons(42, [make_record(i) for i in range(10)])
    entries = await mgr.db.load_jargons(42)
    assert [(e.term, e.meaning) for e in entries] == [("一致的", "圈内含义")]
    # 信息不足的连第二路都不该调用
    assert "信息不足的" not in ai.alone_calls
    await svc.stop()


def test_jargon_pattern_matching_is_mechanical(tmp_path, mgr):
    settings = make_settings(tmp_path)
    svc = make_service(mgr, settings, StubAI())
    # 中文词条按包含匹配
    assert svc._term_pattern("社死").search("今天彻底社死了")
    assert not svc._term_pattern("社死").search("今天很快乐")
    # 西文按词边界、大小写不敏感：nb 不命中 unbalanced
    pattern = svc._term_pattern("nb")
    assert pattern.search("这也太 nb 了") and pattern.search("NB!")
    assert not pattern.search("unbalanced")


async def test_match_jargons_hits_touches_and_limited(tmp_path, mgr):
    settings = make_settings(tmp_path, jargon_max_inject=2)
    svc = make_service(mgr, settings, StubAI())
    now = time.time()
    await mgr.db.record_jargon(42, "nb", "厉害", now - 50, 50)
    await mgr.db.record_jargon(42, "社死", "当众出丑", now - 50, 50)
    await mgr.db.record_jargon(42, "yyds", "永远的神", now - 50, 50)
    hits = await svc.match_jargons(42, "这个操作真nb，我当场社死", 2)
    assert [(t, m) for t, m in hits] == [("nb", "厉害"), ("社死", "当众出丑")]
    entries = {e.term: e for e in await mgr.db.load_jargons(42)}
    assert entries["nb"].last_hit_time >= now and entries["yyds"].last_hit_time == pytest.approx(now - 50)
    assert await svc.match_jargons(42, "   ", 5) == []
    await svc.stop()


# ---------------------------------------------------------------- 任务 A：每日印象


async def test_summarize_day_generates_and_prunes(tmp_path, mgr):
    settings = make_settings(tmp_path, impression_days=3)
    ai = StubAI()
    svc = make_service(mgr, settings, ai)
    day = date.today() - timedelta(days=1)
    day_noon = datetime.combine(day, datetime.min.time()) + timedelta(hours=12)
    ts = day_noon.timestamp()
    await mgr.db.insert_record(make_record(1, "聊 Rust", ts=ts))
    await mgr.db.insert_record(make_record(2, "好呀", ts=ts + 1, is_self=True, nickname="糖糖"))
    await mgr.db.insert_record(make_record(3, "白名单外", group_id=99, ts=ts))
    await mgr.db.insert_record(make_record(4, "不属于这一天", ts=ts - 86400))

    generated = await svc.summarize_day(day)
    assert generated == 1
    assert len(ai.impressions) == 1
    summary_day, chat_text = ai.impressions[0]
    assert summary_day == day.isoformat()
    assert "聊 Rust" in chat_text and "不属于这一天" not in chat_text
    assert "白名单外" not in chat_text  # 群 99 不在白名单，整体不总结
    entries = await mgr.db.load_impressions(42, 3)
    assert [(e.day, e.summary) for e in entries] == [(day.isoformat(), ai.impression_text)]
    # 重复运行幂等：已有当天印象不再调用 LLM
    assert await svc.summarize_day(day) == 0
    assert len(ai.impressions) == 1
    await svc.stop()


async def test_impression_load_order_limit_and_prune(tmp_path, mgr):
    db = mgr.db
    for offset, day in enumerate(("2026-08-26", "2026-08-27", "2026-08-28", "2026-08-29")):
        await db.save_impression(42, day, f"印象{offset}")
    recent = await db.load_impressions(42, 3)
    assert [e.day for e in recent] == ["2026-08-27", "2026-08-28", "2026-08-29"]
    assert await db.has_impression(42, "2026-08-26") is True
    assert await db.has_impression(42, "2026-08-25") is False
    removed = await db.prune_impressions("2026-08-28")
    assert removed == 2
    assert [e.day for e in await db.load_impressions(42, 10)] == ["2026-08-28", "2026-08-29"]
    # 同天重写是覆盖不是新增
    await db.save_impression(42, "2026-08-29", "改过的")
    latest = await db.load_impressions(42, 1)
    assert latest[0].summary == "改过的" and len(latest) == 1


async def test_jargon_cap_evicts_oldest_hit(tmp_path, mgr):
    db = mgr.db
    now = time.time()
    for i in range(5):
        await db.record_jargon(42, f"词{i}", f"含义{i}", now, max_entries=10)
    ids = {e.term: e.id for e in await db.load_jargons(42)}
    for i in range(5):  # 设定命中时间：词0 最久未命中
        await db.touch_jargons([ids[f"词{i}"]], now - 50 + i)
    await db.record_jargon(42, "词5", "含义5", now + 1, max_entries=3)
    entries = await db.load_jargons(42)
    assert [e.term for e in entries] == ["词3", "词4", "词5"]  # 最久未命中的先淘汰
    # 已有词条只更新含义并累计权重，不触发新条目插入与淘汰
    await db.record_jargon(42, "词3", "新含义", now + 2, max_entries=3)
    entries = await db.load_jargons(42)
    updated = [e for e in entries if e.term == "词3"][0]
    assert updated.meaning == "新含义" and updated.count == 2 and len(entries) == 3


# ---------------------------------------------------------------- 配置与角色继承


def test_learning_settings_defaults_and_validation(tmp_path):
    s = make_settings(tmp_path)
    ls = s.learning
    assert ls.enabled and ls.impression_enabled and ls.expression_enabled
    assert ls.jargon_enabled and ls.expression_self_review
    assert ls.impression_days == 3
    assert ls.impression_max_chars == 300
    assert ls.expression_batch_size == 10
    assert ls.expression_max_inject == 3
    assert ls.jargon_max_entries == 50
    assert ls.jargon_max_inject == 5

    s2 = make_settings(tmp_path, enabled=False, impression_days=7, jargon_max_entries=20)
    assert s2.learning.enabled is False
    assert s2.learning.impression_days == 7
    assert s2.learning.jargon_max_entries == 20
    for bad in (
        {"impression_days": 0},
        {"expression_batch_size": -3},
        {"jargon_max_entries": 0},
        {"enabled": "yes"},
    ):
        with pytest.raises(ValueError):
            make_settings(tmp_path, **bad)


def test_models_learning_optional_override():
    cfg = {
        "bot": {"self_qq": 99},
        "groups": {},
        "groups_default": {"persona": "p"},
        "ai_backend": {"base_url": "https://api.example.com/v1", "api_key": "k"},
        "models": {"judge": "j", "reply": "r"},
        "generation": {},
        "multimodal": {},
        "rate_limit": {},
        "snowluma": {
            "endpoint": "http://10.0.0.5:3000/",
            "allow_private_endpoint": True,
        },
    }
    s = load_settings(DictCfg(cfg))
    assert s.models.learning is None  # 不配则无该角色（运行时继承 judge）
    cfg2 = dict(cfg, models=dict(cfg["models"], learning={"model": "learn", "max_output_tokens": 300}))
    s2 = load_settings(DictCfg(cfg2))
    assert s2.models.learning is not None
    assert s2.models.learning.model == "learn"
    assert s2.models.learning.base_url == "https://api.example.com/v1"  # 照常继承 ai_backend
    assert s2.models.learning.max_output_tokens == 300

    # AIClient：未配置 learning 角色时复用 judge 配置
    gen = GenerationSettings(
        reply_max_tokens=500, temperature=0.8, max_context_chars=8000, timeout_seconds=60
    )
    judge = ModelConfig("j", "https://api.example.com/v1", "k", None, None)
    plain = AIClient(
        models=ModelSettings(judge=judge, reply=judge, vision=None), generation=gen
    )
    assert plain._learning is judge
    custom = ModelConfig("learn", "https://api.example.com/v1", "k", None, None)
    with_role = AIClient(
        models=ModelSettings(judge=judge, reply=judge, vision=None, learning=custom),
        generation=gen,
    )
    assert with_role._learning is custom


# ---------------------------------------------------------------- ai.py 解析层


def _parse_client() -> AIClient:
    gen = GenerationSettings(
        reply_max_tokens=500, temperature=0.8, max_context_chars=8000, timeout_seconds=60
    )
    cfg = ModelConfig("j", "https://api.example.com/v1", "k", None, None)
    return AIClient(models=ModelSettings(judge=cfg, reply=cfg, vision=None), generation=gen)


def _patch_learning_call(ai: AIClient, outputs: list[str]):
    """mock 掉真正的 LLM 请求：按队列返回预置正文。"""
    remaining = list(outputs)

    async def fake(prompt: str, *, default_max_tokens: int) -> str:
        return remaining.pop(0) if remaining else ""

    ai._learning_call = fake


async def test_ai_learning_parsing_tolerates_noise():
    ai = _parse_client()
    # 带解释文字与代码块围栏的 JSON 数组照收；超长裁剪到 20 字、最多 10 条
    long_pair = json_pair("惊" * 40, "叹" * 40)
    _patch_learning_call(
        ai,
        [
            f"好的，以下是总结：\n```json\n[{long_pair}, "
            '{"situation": "表示惊叹", "style": "使用 我嘞个xxxx"}]\n```'
        ],
    )
    result = await ai.learn_expressions("chat")
    assert result == [("惊" * 20, "叹" * 20), ("表示惊叹", "使用 我嘞个xxxx")]
    # 输出不可解析 → 空产出（不抛异常）
    _patch_learning_call(ai, ["完全不是 JSON"])
    assert await ai.learn_expressions("chat") == []

    _patch_learning_call(ai, ['前置说明 {"suitable": true, "reason": "自然"} 尾巴'])
    assert await ai.review_expression("情境", "风格") is True
    _patch_learning_call(ai, ['{"is_similar": false}'])
    assert await ai.compare_jargon_inference("甲", "乙") is False

    _patch_learning_call(
        ai,
        [
            '[{"content": "yyds"}, {"content": "yyds"}, {"content": "很长的词条超出十六个字符应该被丢掉哦真的很长"},'
            + ",".join(f'{{"content": "x{i}"}}' for i in range(12))
            + "]"
        ],
    )
    terms = await ai.extract_jargon_terms("chat")
    assert len(terms) == 10 and terms[0] == "yyds"  # 去重、限长、封顶 10

    _patch_learning_call(ai, ['{"meaning": "永远的神", "no_info": false}', '{"meaning": "永远的神"}', '{"is_similar": true}'])
    meaning, no_info = await ai.infer_jargon_with_context("yyds", "ctx")
    assert (meaning, no_info) == ("永远的神", False)
    assert await ai.infer_jargon_alone("yyds") == "永远的神"
    assert await ai.compare_jargon_inference("永远的神", "永远的神") is True


def json_pair(situation: str, style: str) -> str:
    import json

    return json.dumps({"situation": situation, "style": style}, ensure_ascii=False)


async def test_impression_prompt_and_chat_text_format():
    records = [make_record(1, "今晚开黑吗"), make_record(2, "来", is_self=True, nickname="糖糖")]
    text = learning_chat_text(records)
    assert text == "u1(1001)：今晚开黑吗\n【你自己】：来"
    from candybot.prompts import impression_summary_prompt

    prompt = impression_summary_prompt("2026-08-28", text, 300)
    assert "2026-08-28" in prompt and "今日群聊印象" in prompt and text in prompt
