"""任务 3：人物长期记忆——事实抽取挂批、衰减遗忘、画像注入 L4。

全部 LLM 调用以桩对象替代，不发真实请求；衰减数学用假时钟验证。
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import replace as dc_replace

import pytest

from candybot.ai import AIClient, ReplyDraft
from candybot.bot import CandyBot
from candybot.learning import (
    PERSON_FACT_BLOCK_CHAR_BUDGET,
    LearningService,
    person_fact_privacy_rejected,
    person_fact_score,
    person_profile_line_chars,
)
from candybot.models import (
    Decision,
    GenerationSettings,
    ModelConfig,
    ModelSettings,
    NormalizedMessage,
    Settings,
)
from candybot.prompts import (
    final_user_prompt_reply,
    person_profile_block,
)
from tests.test_learning import (
    StubAI,
    make_record,
    make_service,
    make_settings,
    mgr,  # noqa: F401  复用 MemoryManager 夹具
)

DAY = 86400.0
T0 = 1_700_000_000.0  # 假时钟基准


# ---------------------------------------------------------------- 测试桩


class PersonStubAI(StubAI):
    """StubAI 之上补人物学习桩：候选可编程、调用可断言。"""

    def __init__(self):
        super().__init__()
        self.person_calls: list[str] = []
        self.person_facts: list[tuple[str, str]] = []
        self.person_reviews: list[tuple[str, str]] = []
        self.person_review_suitable = True

    async def learn_person_facts(self, chat_text: str):
        self.person_calls.append(chat_text)
        return list(self.person_facts)

    async def review_person_fact(self, nickname: str, fact: str) -> bool:
        self.person_reviews.append((nickname, fact))
        return self.person_review_suitable


class PersonEmbedAI(PersonStubAI):
    """带可编程 embed 的人物学习桩：向量按关键词构造，cosine 只有 1 与 0。"""

    def __init__(self):
        super().__init__()
        self.embed_requests: list[list[str]] = []

    async def embed(self, texts):
        self.embed_requests.append(list(texts))
        return [_keyword_vector(t) for t in texts]


def _keyword_vector(text: str) -> list[float]:
    return [float("奶茶" in text), float("香菜" in text)]


def with_embedding(settings: Settings, model: str = "emb-m") -> Settings:
    emb = ModelConfig(model, "https://api.example.com/v1", "k", None, None)
    return dc_replace(settings, models=dc_replace(settings.models, embedding=emb))


async def seed_fact(
    mgr, fact: str, *, uid: int = 1001, group_id: int | None = 42, ts: float = T0,
    nickname: str = "小明",
) -> tuple[int, bool]:
    return await mgr.db.record_person_fact(group_id, uid, nickname, fact, ts)


# ---------------------------------------------------------------- 存储层


async def test_person_fact_record_dedup_and_touch(tmp_path, mgr):
    first_id, created = await mgr.db.record_person_fact(42, 1001, "小明", "小明是大三学生", T0)
    assert created is True
    # 同 (scope 键, user_id, 文本) 重复学到：不新增行，count+1、刷新时间与昵称
    second_id, created = await mgr.db.record_person_fact(
        42, 1001, "小明同学", "小明是大三学生", T0 + 10
    )
    assert second_id == first_id and created is False
    rows = await mgr.db.load_person_facts(1001, 42)
    assert len(rows) == 1 and rows[0].count == 2 and rows[0].last_hit_time == T0 + 10
    assert rows[0].nickname == "小明同学"
    # 不同群（group 作用域键）各存各的
    other_id, created = await mgr.db.record_person_fact(43, 1001, "小明", "小明是大三学生", T0)
    assert created is True and other_id != first_id
    assert [r.group_id for r in await mgr.db.load_person_facts(1001, 43)] == [43]
    # global 作用域（group_id=None）行不被 group 检索取到，反之亦然
    gid, _ = await mgr.db.record_person_fact(None, 1001, "小明", "小明住在上海", T0)
    assert gid not in [r.id for r in await mgr.db.load_person_facts(1001, 42)]
    assert gid in [r.id for r in await mgr.db.load_person_facts(1001, None)]
    # 注入即强化：刷新命中时间并累计 hit_count
    await mgr.db.touch_person_facts([first_id, gid], T0 + 99)
    by_id = {r.id: r for r in await mgr.db.load_person_facts(1001, None)}
    assert by_id[first_id].last_hit_time == T0 + 99 and by_id[first_id].hit_count == 1
    assert by_id[gid].hit_count == 1


async def test_person_fact_embedding_roundtrip(tmp_path, mgr):
    from candybot.database import pack_vector, unpack_vector

    fact_id, _ = await seed_fact(mgr, "小美喜欢喝奶茶")
    assert await mgr.db.load_person_fact_vectors(1001, 42, "emb-m") == {}
    await mgr.db.upsert_person_fact_embeddings([(fact_id, pack_vector([1.0, 0.0]), 2, "emb-m")])
    vectors = await mgr.db.load_person_fact_vectors(1001, 42, "emb-m")
    assert vectors[fact_id] == pytest.approx([1.0, 0.0])
    # 换模型：旧向量不再被认账（视为缺向量，由学习入库路径懒算重算）
    assert await mgr.db.load_person_fact_vectors(1001, 42, "emb-other") == {}
    # 同 fact_id 覆盖而非新增（含维数变化）
    await mgr.db.upsert_person_fact_embeddings([(fact_id, pack_vector([0.5]), 1, "emb-m")])
    again = await mgr.db.load_person_fact_vectors(1001, 42, "emb-m")
    assert list(again) == [fact_id]
    assert unpack_vector(pack_vector(again[fact_id]), 1) == pytest.approx(again[fact_id])


# ---------------------------------------------------------------- 抽取解析（ai.py）


def _person_parse_client() -> AIClient:
    gen = GenerationSettings(
        reply_max_tokens=500, temperature=0.8, max_context_chars=8000, timeout_seconds=60
    )
    cfg = ModelConfig("j", "https://api.example.com/v1", "k", None, None)
    return AIClient(models=ModelSettings(judge=cfg, reply=cfg, vision=None), generation=gen)


def _patch_learning_call(ai: AIClient, outputs: list[str]):
    remaining = list(outputs)

    async def fake(prompt: str, *, default_max_tokens: int) -> str:
        return remaining.pop(0) if remaining else ""

    ai._learning_call = fake


async def test_ai_learn_person_facts_parsing():
    ai = _person_parse_client()
    # 带解释文字与围栏的 JSON 数组照收；超长（fact 或昵称 >30 字）整条
    # 丢弃、绝不裁剪（残句误导画像，截半的昵称还可能误挂到重名前缀的人）；
    # 空昵称/空事实丢弃
    long_fact = "小明" + "事" * 40
    payload = json.dumps(
        [
            {"nickname": "小明", "fact": long_fact},
            {"nickname": "名" * 31, "fact": "超长昵称整条丢弃"},
            {"nickname": "", "fact": "没名字的丢弃"},
            {"nickname": "小红", "fact": "  "},
            {"nickname": "小红", "fact": "小红喜欢喝蜜雪冰城"},
            "不是对象",
        ],
        ensure_ascii=False,
    )
    _patch_learning_call(ai, [f"以下是结果：\n```json\n{payload}\n```"])
    result = await ai.learn_person_facts("chat")
    assert result == [("小红", "小红喜欢喝蜜雪冰城")]
    # 输出不可解析 → 空产出（不抛异常，后台任务容忍）
    _patch_learning_call(ai, ["完全不是 JSON"])
    assert await ai.learn_person_facts("chat") == []
    # 最多 10 条
    many = json.dumps(
        [{"nickname": f"n{i}", "fact": f"fact{i}"} for i in range(15)], ensure_ascii=False
    )
    _patch_learning_call(ai, [many])
    assert len(await ai.learn_person_facts("chat")) == 10


# ---------------------------------------------------------------- 挂批与昵称解析


async def test_learn_person_facts_resolution(tmp_path, mgr, caplog):
    settings = make_settings(tmp_path)
    ai = PersonStubAI()
    ai.person_facts = [
        ("小明", "小明是大三学生"),          # 唯一匹配 → 挂 user_id
        ("小王", "小王讨厌吃香菜"),          # 重名歧义 → 丢弃
        ("路人", "路人喜欢猫"),              # 匹配不到 → 丢弃
        ("糖糖", "糖糖是大学生"),            # bot 昵称（is_self）→ 丢弃
        ("小刚", "小刚家住幸福小区3栋"),     # 隐私硬拦截 → 丢弃
        ("小明", "小明是大三学生"),          # 同批重复 → 去重只处理一次
    ]
    svc = make_service(mgr, settings, ai)
    batch = [
        make_record(1, "我大三了", user_id=1001, nickname="小明"),
        make_record(2, "附议", user_id=2001, nickname="小王"),
        make_record(3, "附议", user_id=2002, nickname="小王"),
        make_record(4, "大家好", user_id=99, nickname="糖糖", is_self=True),
        make_record(5, "随便说说", user_id=3001, nickname="小刚"),
    ]
    with caplog.at_level(logging.DEBUG, logger="candybot.learning"):
        await svc._learn_person_facts(42, batch)
    rows = await mgr.db.load_person_facts(1001, 42)
    assert [(r.fact, r.user_id, r.group_id) for r in rows] == [("小明是大三学生", 1001, 42)]
    # 其余用户一条不挂（宁缺勿错）
    assert await mgr.db.load_person_facts(2001, 42) == []
    assert await mgr.db.load_person_facts(3001, 42) == []
    text = caplog.text
    assert "重名歧义" in text and "匹配不到" in text and "隐私硬拦截" in text
    await svc.stop()


async def test_learn_person_facts_self_review_switch(tmp_path, mgr):
    settings = make_settings(tmp_path)
    ai = PersonStubAI()
    ai.person_facts = [("小明", "小明喜欢打篮球")]
    svc = make_service(mgr, settings, ai)
    batch = [make_record(1, "打球去", user_id=1001, nickname="小明")]
    await svc._learn_person_facts(42, batch)
    assert ai.person_reviews == []  # 默认关：不调自审
    assert len(await mgr.db.load_person_facts(1001, 42)) == 1

    on = dc_replace(
        settings, learning=dc_replace(settings.learning, person_self_review=True)
    )
    svc2 = make_service(mgr, on, ai)
    ai.person_facts = [("小明", "小明在准备考研")]
    ai.person_review_suitable = False
    await svc2._learn_person_facts(42, batch)
    assert ai.person_reviews == [("小明", "小明在准备考研")]  # 开了才审
    assert len(await mgr.db.load_person_facts(1001, 42)) == 1  # 拒收不入库
    ai.person_review_suitable = True
    await svc2._learn_person_facts(42, batch)
    assert len(await mgr.db.load_person_facts(1001, 42)) == 2
    await svc2.stop()


async def test_learn_person_facts_semantic_merge(tmp_path, mgr):
    """配置 embedding 后：语义近重（cosine>0.92）合并为累加 count、不新增行。"""
    settings = with_embedding(make_settings(tmp_path))
    ai = PersonEmbedAI()
    svc = make_service(mgr, settings, ai)
    batch = [make_record(1, "喝奶茶", user_id=1001, nickname="小美")]
    ai.person_facts = [("小美", "小美喜欢喝奶茶")]
    await svc._learn_person_facts(42, batch)
    assert len(await mgr.db.load_person_facts(1001, 42)) == 1
    assert await mgr.db.load_person_fact_vectors(1001, 42, "emb-m") != {}

    # 换一种说法的近义事实：合并 → count=2、仍一行、不新增向量行
    ai.person_facts = [("小美", "小美常喝奶茶")]
    ai.embed_requests.clear()
    await svc._learn_person_facts(42, batch)
    rows = await mgr.db.load_person_facts(1001, 42)
    assert len(rows) == 1 and rows[0].count == 2
    assert len(ai.embed_requests) == 1  # 新事实向量化一次

    # 无关事实照常新增
    ai.person_facts = [("小美", "小美讨厌吃香菜")]
    await svc._learn_person_facts(42, batch)
    rows = await mgr.db.load_person_facts(1001, 42)
    assert len(rows) == 2 and {r.fact for r in rows} == {"小美喜欢喝奶茶", "小美讨厌吃香菜"}
    await svc.stop()


async def test_learn_person_facts_without_embedding_uses_exact_dedup(tmp_path, mgr):
    """未配置 embedding（默认）：只做文本精确去重，绝不调 embed。"""
    settings = make_settings(tmp_path)
    ai = PersonStubAI()  # 没有 embed 方法：一旦被调用直接 AttributeError 挂批次
    svc = make_service(mgr, settings, ai)
    batch = [make_record(1, "喝奶茶", user_id=1001, nickname="小美")]
    ai.person_facts = [("小美", "小美喜欢喝奶茶")]
    await svc._learn_person_facts(42, batch)
    ai.person_facts = [("小美", "小美喜欢喝奶茶")]  # 文本全同 → count 累加
    await svc._learn_person_facts(42, batch)
    rows = await mgr.db.load_person_facts(1001, 42)
    assert len(rows) == 1 and rows[0].count == 2
    await svc.stop()


# ---------------------------------------------------------------- 批调度与开关


async def test_learn_batch_person_gating(tmp_path, mgr):
    """person_enabled 关闭时 _learn_batch 与引入前完全一致（不碰人物路径）。"""

    class FlagAI(PersonStubAI):
        def __init__(self):
            super().__init__()
            self.person_touched = False

        async def learn_person_facts(self, chat_text):
            self.person_touched = True
            return []

    ai = FlagAI()
    off = dc_replace(
        make_settings(tmp_path),
        learning=dc_replace(make_settings(tmp_path).learning, person_enabled=False),
    )
    svc = make_service(mgr, off, ai)
    await svc._learn_batch(42, [make_record(i) for i in range(10)])
    assert ai.person_touched is False
    assert ai.expression_batches and ai.jargon_batches  # 其余两路照常
    await svc.stop()

    svc2 = make_service(mgr, make_settings(tmp_path), ai)  # 默认开
    await svc2._learn_batch(42, [make_record(i) for i in range(10)])
    assert ai.person_touched is True
    await svc2.stop()


async def test_note_evicted_triggers_with_person_only(tmp_path, mgr):
    """表达/黑话全关、只开人物：淘汰批仍照常攒批触发（新增一路参与总开关判断）。"""
    settings = make_settings(
        tmp_path, expression_enabled=False, jargon_enabled=False, expression_batch_size=10
    )
    ai = PersonStubAI()
    svc = make_service(mgr, settings, ai)
    for i in range(10):
        svc.note_evicted(42, make_record(i))
    task = svc._batch_tasks.get(42)
    assert task is not None
    await task
    assert len(ai.person_calls) == 1
    assert ai.expression_batches == []
    await svc.stop()

    # 三项全关：连攒批都不发生（note_evicted 直接返回）
    all_off = make_settings(
        tmp_path, expression_enabled=False, jargon_enabled=False, person_enabled=False
    )
    svc2 = make_service(mgr, all_off, PersonStubAI())
    for i in range(20):
        svc2.note_evicted(42, make_record(i))
    assert not svc2._pending.get(42) and 42 not in svc2._batch_tasks
    await svc2.stop()


# ---------------------------------------------------------------- 衰减数学


def test_person_fact_score_math():
    gain = 1.0 + math.log(2.0)  # count=1 的 log(1+count) 增益
    fresh = person_fact_score(1.0, 1, T0, T0, T0, 30)
    assert fresh == pytest.approx(gain)
    # 恰到一个半衰期：权重减半
    one_half = person_fact_score(1.0, 1, T0, T0, T0 + 30 * DAY, 30)
    assert one_half == pytest.approx(fresh / 2)
    two_halves = person_fact_score(1.0, 1, T0, T0, T0 + 60 * DAY, 30)
    assert two_halves == pytest.approx(fresh / 4)
    # 从未命中过（last_hit_time=0）按 created_ts 起步衰减，绝不永远满分
    created = person_fact_score(1.0, 1, 0.0, T0, T0 + 30 * DAY, 30)
    assert created == pytest.approx(fresh / 2)
    # 反复学到更抗遗忘：count 大者权重高
    assert person_fact_score(1.0, 10, T0, T0, T0 + 30 * DAY, 30) > one_half
    # 未来时间不放大（days 钳在 0）
    assert person_fact_score(1.0, 1, T0, T0, T0 - 999, 30) == pytest.approx(fresh)


# ---------------------------------------------------------------- 画像选取与注入


async def test_pick_person_profiles_decay_and_touch(tmp_path, mgr):
    settings = make_settings(tmp_path)
    svc = make_service(mgr, settings, PersonStubAI())
    await seed_fact(mgr, "小明是新近学到的事实", ts=T0)
    picked = await svc.pick_person_profiles(42, [(1001, "小明")], now=T0 + 80 * DAY)
    assert picked == [("小明", ["小明是新近学到的事实"])]
    rows = await mgr.db.load_person_facts(1001, 42)
    assert rows[0].last_hit_time == T0 + 80 * DAY and rows[0].hit_count == 1
    # 超过半衰期衰减到阈值（0.25）以下：整块不再出现，也不 touch
    await seed_fact(mgr, "小明很久没被想起", uid=1002, ts=T0)
    stale = await svc.pick_person_profiles(42, [(1002, "路人甲")], now=T0 + 100 * DAY)
    assert stale == []
    rows2 = await mgr.db.load_person_facts(1002, 42)
    assert rows2[0].last_hit_time == T0 and rows2[0].hit_count == 0
    # 半衰期可配置（距上次命中 100 天）：10 天半衰期早已衰减到阈值下；
    # 1000 天半衰期几乎不衰减、仍然命中
    fast = dc_replace(
        settings,
        learning=dc_replace(settings.learning, person_fact_half_life_days=10.0),
    )
    svc._settings = lambda: fast
    assert await svc.pick_person_profiles(42, [(1001, "小明")], now=T0 + 180 * DAY) == []
    slow = dc_replace(
        settings,
        learning=dc_replace(settings.learning, person_fact_half_life_days=1000.0),
    )
    svc._settings = lambda: slow
    assert len(await svc.pick_person_profiles(42, [(1001, "小明")], now=T0 + 180 * DAY)) == 1
    await svc.stop()


async def test_pick_person_profiles_priority_limits_budget(tmp_path, mgr):
    settings = make_settings(
        tmp_path, person_fact_max_inject_per_person=2
    )
    svc = make_service(mgr, settings, PersonStubAI())
    # 四个人各有事实：只取目标序的前三个（≤3 人）
    for i, uid in enumerate((1001, 1002, 1003, 1004)):
        await seed_fact(mgr, f"u{uid}的事实甲", uid=uid, nickname=f"昵{i}")
        await seed_fact(mgr, f"u{uid}的事实乙", uid=uid, nickname=f"昵{i}", ts=T0 + 1)
        await seed_fact(mgr, f"u{uid}的事实丙", uid=uid, nickname=f"昵{i}", ts=T0 + 2)
    targets = [(1001, "甲"), (1002, "乙"), (1003, "丙"), (1004, "丁")]
    picked = await svc.pick_person_profiles(42, targets, now=T0 + 3)
    assert [name for name, _ in picked] == ["甲", "乙", "丙"]  # 优先级序、≤3 人
    # 每人 ≤max_inject_per_person 条，且衰减分高（最近命中）的两条胜出
    assert [fs for _, fs in picked] == [
        [f"u{uid}的事实丙", f"u{uid}的事实乙"] for uid in (1001, 1002, 1003)
    ]
    # 目标去重：同一人只算一次
    dup = await svc.pick_person_profiles(42, [(1001, "甲"), (1001, "甲")], now=T0 + 3)
    assert len(dup) == 1
    # 长事实触发 ≤200 字合计预算截断：条数上限（4）内仍被预算卡掉最低分的一条
    budget_settings = make_settings(tmp_path)  # person_fact_max_inject_per_person=4
    svc2 = make_service(mgr, budget_settings, PersonStubAI())
    long_fact = "大个" + "事" * 57  # 每条整 60 字
    for j in range(4):
        await mgr.db.record_person_fact(42, 2001, "大个", f"{long_fact}{j}", T0 + j)
    picked2 = await svc2.pick_person_profiles(42, [(2001, "大个")], now=T0 + 4)
    assert len(picked2) == 1
    _name, facts = picked2[0]
    assert facts == [f"{long_fact}3", f"{long_fact}2", f"{long_fact}1"]  # 高分三条
    assert person_profile_line_chars("大个", facts) <= PERSON_FACT_BLOCK_CHAR_BUDGET
    assert person_profile_line_chars("大个", facts + [f"{long_fact}0"]) > PERSON_FACT_BLOCK_CHAR_BUDGET
    # 被舍弃的最低分那条不被 touch（注入=强化只对被选中者）
    rows = {r.fact: r for r in await mgr.db.load_person_facts(2001, 42)}
    assert rows[f"{long_fact}3"].last_hit_time == T0 + 4
    assert rows[f"{long_fact}0"].last_hit_time == T0  # 入库时间原样未动
    await svc.stop()
    await svc2.stop()


async def test_pick_person_profiles_scope(tmp_path, mgr):
    """scope=group 不跨群泄漏；scope=global 同 user_id 跨群命中。"""
    settings = make_settings(tmp_path)
    svc = make_service(mgr, settings, PersonStubAI())
    await seed_fact(mgr, "小明是群42认识的人", group_id=42)
    # group（默认）：群 43 查不到
    assert await svc.pick_person_profiles(43, [(1001, "小明")], now=T0) == []
    assert len(await svc.pick_person_profiles(42, [(1001, "小明")], now=T0)) == 1
    # global：任何群都能查到该 user_id 的全部行
    glob = dc_replace(
        settings, learning=dc_replace(settings.learning, person_fact_scope="global")
    )
    svc._settings = lambda: glob
    picked = await svc.pick_person_profiles(43, [(1001, "小明")], now=T0)
    assert picked == [("小明", ["小明是群42认识的人"])]
    await svc.stop()


# ---------------------------------------------------------------- 提示词块与回归


def test_person_profile_block_format():
    hints = [("小明", ["是大三学生，最近考研", "喜欢喝蜜雪冰城"]), ("小红", ["在玩原神"])]
    block = person_profile_block(hints)
    assert block.startswith("【人物画像-内部参考】\n")
    assert "关于 小明：- 是大三学生，最近考研 - 喜欢喝蜜雪冰城" in block
    assert "关于 小红：- 在玩原神" in block
    assert "不要向群友逐字复述或点名宣读" in block
    assert "与当前对话冲突时以当前对话为准" in block


def test_l4_person_block_position_and_byte_regression():
    msg = make_record(7, "当前消息")
    exprs = [("表示惊叹", "我嘞个豆")]
    jargons = [("yyds", "永远的神")]
    hints = [("小明", ["喜欢喝蜜雪冰城"])]
    prompt = final_user_prompt_reply(
        "2026-08-29 10:00:00", msg, forced=True,
        expression_hints=exprs, jargon_hints=jargons, person_hints=hints,
    )
    assert (
        prompt.index("【需要回应的消息】")
        < prompt.index("【表达习惯参考")
        < prompt.index("【黑话参考】")
        < prompt.index("【人物画像-内部参考】")
    )
    assert person_profile_block(hints) in prompt
    # person_hints 为空 → 与不传该参数的输出逐字节一致（回归：关闭时 L4 无差异）
    without = final_user_prompt_reply(
        "2026-08-29 10:00:00", msg, forced=True,
        expression_hints=exprs, jargon_hints=jargons,
    )
    assert without == final_user_prompt_reply(
        "2026-08-29 10:00:00", msg, forced=True,
        expression_hints=exprs, jargon_hints=jargons, person_hints=(),
    )
    assert "【人物画像" not in without


def test_person_fact_privacy_rejected():
    for leak in (
        "小明家住幸福小区3栋",
        "小红的学校是北京市第四中学",
        "小刚就读于华南理工大学",
        "小美在宏达集团有限公司上班",
        "小强微信号是 qqq123",
        "小李的手机号是13800000000",
    ):
        assert person_fact_privacy_rejected(leak), leak
    for ok in (
        "小明是大三学生，正在考研",
        "小红喜欢喝蜜雪冰城",
        "小刚住在上海",
        "小美是四川人",
    ):
        assert not person_fact_privacy_rejected(ok), ok


# ---------------------------------------------------------------- bot 接线


def _recorder_ai() -> tuple[object, list[dict]]:
    calls: list[dict] = []

    class RecorderAI:
        reply_tool_use = True

        async def generate_reply(self, *args, **kwargs):
            calls.append(kwargs)
            return ReplyDraft("好的")

    return RecorderAI(), calls


async def test_bot_person_targets_and_injection(tmp_path):
    """目标优先级（发送者→被@→被回复）、bot 自己排除、经 generate_reply 进 L4。"""
    import time as _time

    recent_ts = _time.time()  # 真实时钟下要足够新，否则被衰减过滤掉
    bot = CandyBot(make_settings(tmp_path))
    try:
        await bot._memory.db.create_tables()
        await bot._memory.db.record_person_fact(42, 1001, "小明", "小明喜欢打篮球", recent_ts)
        await bot._memory.db.record_person_fact(42, 2001, "小刚", "小刚讨厌吃香菜", recent_ts)
        await bot._memory.db.record_person_fact(42, 3001, "小红", "小红是设计师", recent_ts)
        sender = make_record(1, "在吗", user_id=1001, nickname="小明", ts=recent_ts)
        recent = [make_record(2, "打球去", user_id=2001, nickname="小刚", ts=recent_ts)]
        msg = NormalizedMessage(
            record=sender,
            mentioned_me=True,
            at_user_ids=(2001, 99),        # 99 = bot 自己，应被排除
            reply_user_id=3001,
            reply_nickname="小红",
        )
        hints = await bot._person_profile_hints(42, msg, recent)
        assert [name for name, _ in hints] == ["小明", "小刚", "小红"]

        # 接线到生成调用：person_hints 原样进 generate_reply kwargs
        bot._ai, calls = _recorder_ai()
        profile = bot._settings.profile_for(42)
        runtime = bot._runtimes[42]
        memory = await bot._memory.get(42)
        await memory.append(sender)
        await bot._compose_reply(
            42, msg, profile, runtime,
            Decision(should_reply=True, forced=True), [], [], [("小明", ["小明喜欢打篮球"])],
        )
        assert calls and calls[-1]["person_hints"] == [("小明", ["小明喜欢打篮球"])]

        # 关闭单项开关：不注入（返回空）
        bot._settings = dc_replace(
            bot._settings,
            learning=dc_replace(bot._settings.learning, person_enabled=False),
        )
        assert await bot._person_profile_hints(42, msg, recent) == []
    finally:
        await bot.stop()


async def test_bot_person_targets_empty_when_no_facts(tmp_path):
    """一条都不过阈值 → 整块不出现（hints 为空，L4 与现状一致）。"""
    bot = CandyBot(make_settings(tmp_path))
    try:
        await bot._memory.db.create_tables()
        msg = NormalizedMessage(record=make_record(1, "随便聊聊", ts=T0), mentioned_me=False)
        assert await bot._person_profile_hints(42, msg, [msg.record]) == []
    finally:
        await bot.stop()


async def test_bot_person_hints_failure_degrades_to_none(tmp_path):
    """画像读取失败只记 WARNING、退化为不注入，绝不阻断回复链路。"""
    bot = CandyBot(make_settings(tmp_path))
    try:
        await bot._memory.db.create_tables()

        async def boom(group_id, targets):
            raise RuntimeError("库炸了")

        bot._learning.pick_person_profiles = boom
        msg = NormalizedMessage(record=make_record(1, "随便聊聊", ts=T0), mentioned_me=False)
        assert await bot._person_profile_hints(42, msg, [msg.record]) == []
    finally:
        await bot.stop()


# ---------------------------------------------------------------- 配置解析


def test_person_settings_defaults_and_validation(tmp_path):
    ls = make_settings(tmp_path).learning
    assert ls.person_enabled is True
    assert ls.person_fact_scope == "group"
    assert ls.person_fact_max_inject_per_person == 4
    assert ls.person_fact_half_life_days == pytest.approx(30.0)
    assert ls.person_fact_min_weight == pytest.approx(0.25)
    assert ls.person_self_review is False

    custom = make_settings(
        tmp_path,
        person_enabled=False,
        person_fact_scope="GLOBAL",
        person_fact_max_inject_per_person=2,
        person_fact_half_life_days=7.5,
        person_fact_min_weight=0.5,
        person_self_review=True,
    )
    cl = custom.learning
    assert cl.person_enabled is False and cl.person_fact_scope == "global"
    assert cl.person_fact_max_inject_per_person == 2
    assert cl.person_fact_half_life_days == pytest.approx(7.5)
    assert cl.person_fact_min_weight == pytest.approx(0.5)
    assert cl.person_self_review is True

    for bad in (
        {"person_fact_scope": "world"},
        {"person_fact_max_inject_per_person": 0},
        {"person_fact_half_life_days": 0},
        {"person_fact_half_life_days": float("nan")},
        {"person_fact_min_weight": -0.1},
        {"person_enabled": "yes"},
    ):
        with pytest.raises(ValueError):
            make_settings(tmp_path, **bad)
