"""表达选择的 vector 语义检索模式：假 embedder 全场景 + 加权随机回归。

embedding 调用以确定性伪向量桩实现（二维：「游」「学」各计数），不发真实请求。
"""

from __future__ import annotations

import logging
import struct
import time
from dataclasses import replace as dc_replace
from types import SimpleNamespace

import pytest

from candybot.ai import AIClient
from candybot.bot import CandyBot
from candybot.database import pack_vector, unpack_vector
from candybot.learning import _cosine_similarity, expression_embed_text
from candybot.models import (
    GenerationSettings,
    ModelConfig,
    ModelSettings,
    Settings,
    load_settings,
)
from tests.deterministic_rng import SeededRng
from tests.test_learning import (
    StubAI,
    make_record,
    make_service,
    make_settings,
    mgr,  # noqa: F401  复用 MemoryManager 夹具
)
from tests.test_models_settings import DictCfg

MODEL = "emb-model"


# ---------------------------------------------------------------- 公共构造


def make_vector_settings(tmp_path, *, embedding: str | None = MODEL, **learning_over) -> Settings:
    """配置了 embedding 角色、表达选取走 vector 模式的最小可用配置。"""
    models: dict = {"judge": "j", "reply": "r"}
    if embedding is not None:
        models["embedding"] = embedding
    learning: dict = {"expression_selection_mode": "vector", **learning_over}
    cfg = {
        "bot": {"self_qq": 99, "data_dir": str(tmp_path / "data")},
        "groups": {"42": {"persona": "测试人设", "proactivity_threshold": 6}},
        "groups_default": {"enabled": False, "persona": "默认人设"},
        "ai_backend": {"base_url": "https://api.example.com/v1", "api_key": "k"},
        "models": models,
        "generation": {},
        "multimodal": {},
        "rate_limit": {},
        "snowluma": {"endpoint": "http://10.0.0.5:3000/", "allow_private_endpoint": True},
        "learning": learning,
    }
    return load_settings(DictCfg(cfg))


def fake_vector(text: str) -> list[float]:
    """确定性伪向量：[「游」计数, 「学」计数]（不含任何一项时为 [0,0]）。"""
    return [float(text.count("游")), float(text.count("学"))]


class FakeEmbedAI(StubAI):
    """在 StubAI 之上补一个可编程的 embed()：按关键词计数、记录每次请求。"""

    def __init__(self):
        super().__init__()
        self.embed_requests: list[list[str]] = []
        self.embed_fail = False

    async def embed(self, texts):
        self.embed_requests.append(list(texts))
        if self.embed_fail:
            raise RuntimeError("embedding 服务不可用")
        return [fake_vector(t) for t in texts]


async def seed_two_expressions(mgr, now: float) -> None:
    """一条语境相关（游戏）低权重、一条无关（学习）高权重：
    加权随机几乎必抽后者，vector 模式应抽前者。"""
    await mgr.db.record_expression(42, "约游戏", "上号开游", now)
    for _ in range(50):
        await mgr.db.record_expression(42, "提到学习", "卷不动了", now)


# ---------------------------------------------------------------- 检索选择


async def test_vector_mode_prefers_context_relevant_entry(tmp_path, mgr, caplog):
    settings = make_vector_settings(tmp_path)
    ai = FakeEmbedAI()
    svc = make_service(mgr, settings, ai)
    svc.rng = SeededRng(42)
    await seed_two_expressions(mgr, time.time())
    await svc._backfill_embeddings()
    recent = [make_record(98, "有人一起吗"), make_record(99, "今晚一起打游戏吗")]
    with caplog.at_level(logging.DEBUG, logger="candybot.learning"):
        picks = await svc.pick_expressions(42, 3, recent, trigger=recent[-1])
    # 无关条目权重再高也不入选；相似度 1.000 出现在 DEBUG 日志
    assert picks == [("约游戏", "上号开游")]
    text = caplog.text
    assert "表达向量召回" in text and "约游戏→上号开游=1.000" in text
    assert "表达 L4 注入（向量召回，含相似度）" in text
    await svc.stop()


async def test_vector_mode_returns_empty_when_all_below_threshold(tmp_path, mgr):
    settings = make_vector_settings(tmp_path)
    ai = FakeEmbedAI()
    svc = make_service(mgr, settings, ai)
    now = time.time()
    await seed_two_expressions(mgr, now)
    await svc._backfill_embeddings()
    recent = [make_record(99, "今天吃啥好啊")]  # 与两个条目都正交（零向量）
    assert await svc.pick_expressions(42, 3, recent, trigger=recent[-1]) == []
    # 不注入也不 touch
    entries = await mgr.db.load_expressions(42)
    assert all(e.last_active_time <= now for e in entries)
    await svc.stop()


async def test_vector_mode_falls_back_when_no_query_text(tmp_path, mgr):
    settings = make_vector_settings(tmp_path)
    ai = FakeEmbedAI()
    svc = make_service(mgr, settings, ai)
    svc.rng = SeededRng(7)
    await seed_two_expressions(mgr, time.time())
    await svc._backfill_embeddings()
    # 没有任何上下文：无法构造查询 → 退回加权随机（非空、不抛异常）
    picks = await svc.pick_expressions(42, 1, [], trigger=None)
    assert len(picks) == 1
    await svc.stop()


async def test_weighted_mode_identical_and_ignores_context(tmp_path, mgr):
    """回归：默认模式下新参数完全不起作用、不调 embed、抽取轨迹与改造前一致。"""
    settings = make_settings(tmp_path)  # 不配置任何新字段
    assert settings.learning.expression_selection_mode == "weighted_random"
    now = time.time()
    for i in range(5):
        await mgr.db.record_expression(42, f"情境{i}", f"风格{i}", now - 100 + i)
    ai = FakeEmbedAI()
    svc_a = make_service(mgr, settings, ai)
    svc_a.rng = SeededRng(42)
    plain = await svc_a.pick_expressions(42, 3)
    svc_b = make_service(mgr, settings, ai)
    svc_b.rng = SeededRng(42)
    with_ctx = await svc_b.pick_expressions(
        42, 3, [make_record(98, "今晚一起打游戏吗")], trigger=make_record(99, "打游戏")
    )
    assert plain == with_ctx
    assert ai.embed_requests == []
    await svc_a.stop()
    await svc_b.stop()


async def test_query_embedding_cached_per_group_and_message(tmp_path, mgr):
    settings = make_vector_settings(tmp_path)
    ai = FakeEmbedAI()
    svc = make_service(mgr, settings, ai)
    await seed_two_expressions(mgr, time.time())
    await svc._backfill_embeddings()
    ai.embed_requests.clear()
    r98 = make_record(98, "有人一起吗")
    r99 = make_record(99, "今晚一起打游戏吗")
    recent = [r98, r99]
    await svc.pick_expressions(42, 1, recent, trigger=r99)
    assert len(ai.embed_requests) == 1  # 第一次：查询向量化一次
    await svc.pick_expressions(42, 1, recent, trigger=r99)
    assert len(ai.embed_requests) == 1  # 同一触发消息再生成：不再请求
    r100 = make_record(100, "明天也来游")
    await svc.pick_expressions(42, 1, [r98, r99, r100], trigger=r100)
    assert len(ai.embed_requests) == 2  # 新触发消息：算一次
    await svc.stop()


async def test_query_text_truncated_to_tail(tmp_path, mgr):
    settings = make_vector_settings(tmp_path)
    ai = FakeEmbedAI()
    svc = make_service(mgr, settings, ai)
    svc.query_text_budget = 12
    await seed_two_expressions(mgr, time.time())
    await svc._backfill_embeddings()
    ai.embed_requests.clear()
    r = make_record(99, "今晚要不要一起打游戏呀大家说一下时间安排")
    await svc.pick_expressions(42, 1, [r], trigger=r)
    query = ai.embed_requests[-1][0]
    assert len(query) <= 12 and query == str(r.text)[-12:]
    await svc.stop()


# ---------------------------------------------------------------- 向量补算与缓存


async def test_missing_vectors_backfilled_and_model_change_recomputes(tmp_path, mgr):
    settings = make_vector_settings(tmp_path, embedding="emb-1")
    ai = FakeEmbedAI()
    svc = make_service(mgr, settings, ai)
    await seed_two_expressions(mgr, time.time())
    assert await mgr.db.load_expression_vectors(42, "emb-1") == {}
    await svc._backfill_embeddings()
    assert len(await mgr.db.load_expression_vectors(42, "emb-1")) == 2
    count = len(ai.embed_requests)
    await svc._backfill_embeddings()
    assert len(ai.embed_requests) == count  # 不缺了：不再请求
    # 触发一次检索以建立每群向量缓存与查询缓存
    r = make_record(99, "今晚一起打游戏吗")
    await svc.pick_expressions(42, 1, [r], trigger=r)
    assert svc._expr_vectors and svc._query_vectors
    # 热切换 embedding 模型：全部向量缓存整体作废
    settings2 = dc_replace(
        settings,
        models=dc_replace(
            settings.models,
            embedding=dc_replace(settings.models.embedding, model="emb-2"),
        ),
    )
    svc._settings = lambda: settings2
    assert svc._embedding_model_name() == "emb-2"
    assert svc._expr_vectors == {} and svc._query_vectors == {}
    count = len(ai.embed_requests)
    await svc._backfill_embeddings()
    assert len(ai.embed_requests) == count + 1  # 按新模型重算一轮
    assert len(await mgr.db.load_expression_vectors(42, "emb-2")) == 2
    assert await mgr.db.load_expression_vectors(42, "emb-1") == {}  # upsert 覆盖旧模型
    await svc.stop()


async def test_learn_expressions_embeds_new_entries(tmp_path, mgr):
    settings = make_vector_settings(tmp_path)
    ai = FakeEmbedAI()
    ai.expressions = [("约游戏", "上号开游")]
    svc = make_service(mgr, settings, ai)
    await svc._learn_expressions(42, [make_record(1, "今晚打游戏吗")])
    learned = await mgr.db.load_expressions(42)
    assert [(e.situation, e.style) for e in learned] == [("约游戏", "上号开游")]
    vectors = await mgr.db.load_expression_vectors(42, MODEL)
    assert len(vectors) == 1  # 入库尾部即完成补算（_guarded 内联，无需再等任务）
    assert expression_embed_text("约游戏", "上号开游") in ai.embed_requests[-1]
    await svc.stop()


async def test_learn_expressions_without_embedding_silent(tmp_path, mgr):
    """未配置 embedding（默认配置）：学习照常入库、embedding 静默跳过。
    StubAI 根本没有 embed 方法，一旦被调用即 AttributeError 让测试失败。"""
    settings = make_settings(tmp_path)
    ai = StubAI()
    ai.expressions = [("约游戏", "上号开游")]
    svc = make_service(mgr, settings, ai)
    await svc._learn_expressions(42, [make_record(1, "今晚打游戏吗")])
    assert len(await mgr.db.load_expressions(42)) == 1
    await svc.stop()


async def test_embed_runtime_failure_falls_back_to_weighted(tmp_path, mgr, caplog):
    """运行期 embedding 服务挂掉：记 WARNING、退回加权随机，绝不抛异常。"""
    settings = make_vector_settings(tmp_path)
    ai = FakeEmbedAI()
    svc = make_service(mgr, settings, ai)
    svc.rng = SeededRng(7)
    await seed_two_expressions(mgr, time.time())
    await svc._backfill_embeddings()
    ai.embed_fail = True
    with caplog.at_level(logging.WARNING, logger="candybot.learning"):
        picks = await svc.pick_expressions(
            42, 2, [make_record(99, "今晚一起打游戏吗")], trigger=make_record(99)
        )
    assert len(picks) == 2  # 全池加权随机照常有产出
    assert "退回加权随机" in caplog.text
    await svc.stop()


# ---------------------------------------------------------------- 数据库与字节格式


async def test_expression_embedding_db_roundtrip(tmp_path, mgr):
    await mgr.db.record_expression(42, "s1", "t1", time.time())
    [entry] = await mgr.db.load_expressions(42)
    assert await mgr.db.list_expressions_missing_embedding("m") == [
        (entry.id, 42, "s1", "t1")
    ]
    assert await mgr.db.load_expression_vectors(42, "m") == {}
    await mgr.db.upsert_expression_embeddings([(entry.id, pack_vector([1.0, 2.0]), 2, "m")])
    assert await mgr.db.list_expressions_missing_embedding("m") == []
    vectors = await mgr.db.load_expression_vectors(42, "m")
    assert vectors[entry.id] == pytest.approx([1.0, 2.0])
    # 换模型：同一条目重新视为待补
    assert [t[0] for t in await mgr.db.list_expressions_missing_embedding("m2")] == [
        entry.id
    ]
    # upsert 按 expression_id 覆盖而非新增
    await mgr.db.upsert_expression_embeddings([(entry.id, pack_vector([3.0]), 1, "m2")])
    assert await mgr.db.load_expression_vectors(42, "m") == {}
    assert (await mgr.db.load_expression_vectors(42, "m2"))[entry.id] == [3.0]
    # 分群装载互不串台
    await mgr.db.record_expression(43, "s2", "t2", time.time())
    assert list((await mgr.db.load_expression_vectors(43, "m2")).values()) == []


def test_pack_vector_format():
    packed = pack_vector([1.0, -2.5, 0.125])
    assert packed == struct.pack("<3f", 1.0, -2.5, 0.125)  # float32 小端
    assert len(packed) == 12
    assert unpack_vector(packed, 3) == [1.0, -2.5, 0.125]
    # float32 精度回环：近似而非逐位相等
    assert unpack_vector(pack_vector([0.1]), 1)[0] == pytest.approx(0.1, abs=1e-6)


def test_cosine_similarity():
    assert _cosine_similarity([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.0)
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0  # 零向量
    assert _cosine_similarity([1.0], [1.0, 2.0]) == 0.0  # 维数不一致
    assert _cosine_similarity([], []) == 0.0


# ---------------------------------------------------------------- 配置解析与校验


def test_vector_mode_requires_embedding(tmp_path):
    with pytest.raises(ValueError, match="models.embedding"):
        make_vector_settings(tmp_path, embedding=None)
    # 未配置 embedding 但模式是默认 weighted_random：一切照常（验收标准 1）
    s = make_settings(tmp_path)
    assert s.models.embedding is None


def test_learning_vector_settings_validation(tmp_path):
    with pytest.raises(ValueError):
        make_vector_settings(tmp_path, expression_selection_mode="random")
    with pytest.raises(ValueError):
        make_vector_settings(tmp_path, expression_min_similarity=1.5)
    with pytest.raises(ValueError):
        make_vector_settings(tmp_path, expression_min_similarity=-0.1)
    with pytest.raises(ValueError):
        make_vector_settings(tmp_path, expression_vector_top_k=0)
    # 默认值：不写任何新字段的旧配置逐字段不变
    s = make_settings(tmp_path)
    assert s.learning.expression_selection_mode == "weighted_random"
    assert s.learning.expression_vector_top_k == 10
    assert s.learning.expression_min_similarity == pytest.approx(0.30)


def test_embedding_role_config_inherits_backend(tmp_path):
    s = make_vector_settings(tmp_path, embedding="glm-embedding")
    assert isinstance(s.models.embedding, ModelConfig)
    assert s.models.embedding.model == "glm-embedding"
    assert s.models.embedding.base_url == "https://api.example.com/v1"  # 继承 ai_backend


# ---------------------------------------------------------------- AIClient.embed


def _gen() -> GenerationSettings:
    return GenerationSettings(
        reply_max_tokens=500, temperature=0.8, max_context_chars=8000, timeout_seconds=60
    )


def _embed_client(embedding: ModelConfig | None) -> AIClient:
    cfg = ModelConfig("j", "https://api.example.com/v1", "k", None, None)
    return AIClient(
        models=ModelSettings(judge=cfg, reply=cfg, vision=None, embedding=embedding),
        generation=_gen(),
    )


async def test_ai_embed_posts_to_embeddings_and_orders_by_index():
    captured: dict = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[0.0, 1.0]),
                SimpleNamespace(index=0, embedding=[1.0, 0.0]),
            ]
        )

    emb = ModelConfig("emb", "https://emb.example.com/v1", "ke", None, None)
    ai = _embed_client(emb)
    ai._client_for = lambda cfg: SimpleNamespace(embeddings=SimpleNamespace(create=fake_create))
    vectors = await ai.embed(["a", "b"])
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]  # 乱序响应按 index 归位
    assert captured["model"] == "emb"
    assert captured["input"] == ["a", "b"]
    assert captured["timeout"] == 60


async def test_ai_embed_requires_role():
    ai = _embed_client(None)
    with pytest.raises(RuntimeError, match="models.embedding 未配置"):
        await ai.embed(["x"])


async def test_ai_embed_count_mismatch_raises():
    async def fake_create(**kwargs):
        return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0])])

    emb = ModelConfig("emb", "https://emb.example.com/v1", "ke", None, None)
    ai = _embed_client(emb)
    ai._client_for = lambda cfg: SimpleNamespace(embeddings=SimpleNamespace(create=fake_create))
    with pytest.raises(RuntimeError, match="不符"):
        await ai.embed(["a", "b"])


# ---------------------------------------------------------------- bot 接线


async def test_learning_hints_vector_mode_through_bot(tmp_path):
    """_learning_hints 把触发消息传下去后，vector 模式的语境检索在
    bot 链路上生效（辅助能力语义不变：失败只退化为不注入）。"""
    bot = CandyBot(make_vector_settings(tmp_path))
    try:
        bot._ai = FakeEmbedAI()
        await bot._memory.db.create_tables()
        await bot._memory.db.record_expression(42, "约游戏", "上号开游", time.time())
        await bot._memory.db.record_expression(42, "提到学习", "卷不动了", time.time())
        await bot._learning._backfill_embeddings()
        recent = [make_record(50, "今晚一起打游戏吗")]
        hints, jargons = await bot._learning_hints(42, recent, trigger=recent[0])
        assert hints == [("约游戏", "上号开游")]
        assert jargons == []
        # 不传 trigger 也照常工作（退回语境=recent 尾部）
        hints2, _ = await bot._learning_hints(42, recent)
        assert hints2 == [("约游戏", "上号开游")]
    finally:
        await bot.stop()
