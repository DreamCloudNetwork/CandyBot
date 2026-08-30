"""任务 2（表情包 v2）：收藏审核 + 描述入库 + smart 模型选图。

覆盖：sticker_meta 读写与级联删除、收集路径的审核（拒绝不收藏 / 通过入
meta / vision 失败维持现状）、direct 模式合并调用只发生一次 vision 请求、
ai 层的 assess_sticker / pick_sticker 协议与解析、smart 选图与候选轮换、
新配置项的默认值与校验。

不发真实网络请求：AI 层用 test_ai_providers 的假 AsyncOpenAI 打桩，
store/normalize 层用假客户端，DB 与图片文件走 tmp_path。
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from candybot.ai import AIClient, ImageAssessment, StickerAssessment
from candybot.database import CandyDatabase, image_fingerprint
from candybot.models import (
    ModelConfig,
    ModelSettings,
    MultimodalSettings,
    StickerSettings,
    load_settings,
)
from candybot.normalize import normalize_group_message
from candybot.stickers import StickerStore
import candybot.normalize as norm_mod
from tests.deterministic_rng import SeededRng
from tests.test_ai_providers import _content_msg, _gen, _install_fake, _models, _tool_msg
from tests.test_models_settings import DictCfg, base_cfg
from tests.test_stickers import _no_ref, _record_with, png_url


# ---------------------------------------------------------------- 假 AI 客户端


class FakeStickerAI:
    """按脚本应答的假审核/选图客户端（记录调用，不碰网络）。

    assessments：assess_sticker 逐次返回的 StickerAssessment（耗尽或为 None
    表示「拿不到结论」；放 Exception 实例则抛出）；
    pick：pick_sticker 的固定返回，或 target 模式下按描述匹配编号。
    """

    def __init__(self, assessments=(), pick=None, pick_target=None, pick_error=None):
        self.assessments = list(assessments)
        self.pick = pick
        self.pick_target = pick_target
        self.pick_error = pick_error
        self.assess_calls: list[str] = []
        self.pick_calls: list[tuple[str, list[tuple[str, str]]]] = []

    async def assess_sticker(self, data_url):
        self.assess_calls.append(data_url)
        if not self.assessments:
            return None
        item = self.assessments.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def pick_sticker(self, context_text, entries):
        self.pick_calls.append((context_text, list(entries)))
        if self.pick_error is not None:
            raise self.pick_error
        if self.pick_target is not None:
            for position, (description, _emotion) in enumerate(entries):
                if description == self.pick_target:
                    return position, "按描述选中"
            raise AssertionError(f"候选里没有目标：{self.pick_target}")
        return self.pick


def _settings(**over) -> SimpleNamespace:
    return SimpleNamespace(stickers=StickerSettings(**over))


async def _make_store(tmp_path, ai=None, **sticker_over):
    db = CandyDatabase(tmp_path / "candy.db")
    await db.create_tables()
    store = StickerStore(
        tmp_path / "stickers",
        db,
        lambda: _settings(**sticker_over),
        (lambda: ai) if ai is not None else None,
    )
    return store, db


# ---------------------------------------------------------------- 收藏审核


async def test_moderation_rejects_and_logs_reason(tmp_path, caplog):
    """审核不合格：不收藏、不落文件，DEBUG 日志带模型给出的理由。"""
    ai = FakeStickerAI([StickerAssessment(False, "游戏截图，大段文字", "")])
    store, db = await _make_store(tmp_path, ai)
    url = png_url(64)
    with caplog.at_level(logging.DEBUG):
        assert await store.collect(_record_with(url, ts=100.0), (True,)) == 0
    assert await db.load_stickers(42) == []
    assert ai.assess_calls == [url]
    assert "审核未通过" in caplog.text and "游戏截图" in caplog.text
    assert not (tmp_path / "stickers" / "42" / f"{image_fingerprint(url)}.png").exists()
    await db.close()


async def test_moderation_pass_saves_meta(tmp_path):
    """审核通过：收藏照常入库，「描述 + 情绪」进 sticker_meta（smart 候选）。"""
    ai = FakeStickerAI([StickerAssessment(True, "柴犬歪头疑惑", "无语")])
    store, db = await _make_store(tmp_path, ai)
    assert await store.collect(_record_with(png_url(64), ts=100.0), (True,)) == 1
    (entry,) = await db.load_stickers(42)
    (meta,) = await db.load_stickers_with_meta(42)
    assert meta.entry.id == entry.id
    assert meta.description == "柴犬歪头疑惑" and meta.emotion == "无语"
    await db.close()


async def test_moderation_failure_keeps_collecting_without_meta(tmp_path, caplog):
    """审核调用失败：WARNING 一次，维持现状收藏、无 meta（只参与随机兜底）。"""
    ai = FakeStickerAI([RuntimeError("vision 挂了")])
    store, db = await _make_store(tmp_path, ai)
    with caplog.at_level(logging.DEBUG):
        assert await store.collect(_record_with(png_url(64), ts=100.0), (True,)) == 1
    assert len(ai.assess_calls) == 1
    assert await db.load_stickers_with_meta(42) == []
    assert "审核调用失败" in caplog.text
    await db.close()


async def test_moderation_vision_unavailable_collects_without_meta(tmp_path, caplog):
    """vision 未配置（assess_sticker 恒返回 None）：DEBUG 一次、照常收藏无 meta。"""
    ai = FakeStickerAI([])
    store, db = await _make_store(tmp_path, ai)
    with caplog.at_level(logging.DEBUG):
        assert await store.collect(_record_with(png_url(64), ts=100.0), (True,)) == 1
    assert await db.load_stickers_with_meta(42) == []
    assert "未产出结论" in caplog.text
    await db.close()


async def test_moderation_disabled_skips_vision_entirely(tmp_path):
    """moderation_enabled=false：一次审核调用都不发，收集与现状逐字一致。"""
    ai = FakeStickerAI([StickerAssessment(True, "不该被调用", "得意")])
    store, db = await _make_store(tmp_path, ai, moderation_enabled=False)
    assert await store.collect(_record_with(png_url(64), ts=100.0), (True,)) == 1
    assert ai.assess_calls == []
    assert await db.load_stickers_with_meta(42) == []
    await db.close()


async def test_precomputed_meta_skips_second_call_and_persists(tmp_path):
    """direct 合并结论传入：收集端不再调 assess_sticker，meta 直接入库。"""
    ai = FakeStickerAI([StickerAssessment(True, "绝不该被调用", "狂喜")])
    store, db = await _make_store(tmp_path, ai)
    metas = (StickerAssessment(True, "熊猫摊手", "无所谓"),)
    assert await store.collect(_record_with(png_url(64), ts=100.0), (True,), metas) == 1
    assert ai.assess_calls == []
    (meta,) = await db.load_stickers_with_meta(42)
    assert meta.description == "熊猫摊手" and meta.emotion == "无所谓"
    await db.close()


async def test_eviction_cascades_meta(tmp_path):
    """超上限替换：被淘汰条目的 meta 级联删除（外键约束下漏删会直接报错）。"""
    ai = FakeStickerAI(
        [
            StickerAssessment(True, "第一张", "得意"),
            StickerAssessment(True, "第二张", "无语"),
            StickerAssessment(True, "第三张", "狂喜"),
        ]
    )
    store, db = await _make_store(tmp_path, ai, max_count=2)
    for size, ts in ((10, 100.0), (11, 200.0), (12, 300.0)):
        assert await store.collect(_record_with(png_url(size), ts=ts), (True,)) == 1
    metas = await db.load_stickers_with_meta(42)
    assert [m.description for m in metas] == ["第二张", "第三张"]  # 最久未使用的先走
    shas = {e.sha256 for e in await db.load_stickers(42)}
    assert {m.entry.sha256 for m in metas} == shas
    await db.close()


# ---------------------------------------------------------------- normalize 合并链路


async def _normalize_direct(monkeypatch, data_url, assess_image):
    async def fake_download(session, url, **_kwargs):
        return data_url

    monkeypatch.setattr(norm_mod, "_download_as_data_url", fake_download)
    return await normalize_group_message(
        {
            "post_type": "message",
            "message_type": "group",
            "group_id": 42,
            "user_id": 1000,
            "message_id": 1,
            "time": 1700000000,
            "sender": {"card": "小明"},
            "message": [
                {"type": "image", "data": {"url": "https://img.example.com/a.png"}}
            ],
        },
        self_qq=99,
        multimodal=MultimodalSettings(mode="direct", download_media=True),
        find_by_message_id=_no_ref,
        http_session=object(),
        assess_image=assess_image,
    )


async def test_direct_merged_single_vision_call(tmp_path, monkeypatch):
    """direct 收集一张表情包全程只发生一次 vision 请求：合并评估给 meta，
    收集端不再独立调 assess_sticker。"""
    calls = {"n": 0}

    async def assess(_url):
        calls["n"] += 1
        return ImageAssessment(
            "柴犬歪头梗图",
            keep_raw=False,
            is_sticker=True,
            sticker_assessment=StickerAssessment(True, "柴犬歪头疑惑", "无语"),
        )

    res = await _normalize_direct(monkeypatch, png_url(64), assess)
    assert calls["n"] == 1
    assert res.sticker_flags == (True,)
    assert res.sticker_metas[0].description == "柴犬歪头疑惑"

    ai = FakeStickerAI([StickerAssessment(True, "绝不该被调用", "狂喜")])
    store, db = await _make_store(tmp_path, ai)
    assert await store.collect(res.record, res.sticker_flags, res.sticker_metas) == 1
    assert ai.assess_calls == [] and calls["n"] == 1
    (meta,) = await db.load_stickers_with_meta(42)
    assert meta.description == "柴犬歪头疑惑"
    await db.close()


async def test_direct_merged_rejection_blocks_collect(monkeypatch, caplog):
    """合并结论 acceptable=false：normalize 处即压掉 flag 并记 DEBUG 理由。"""

    async def assess(_url):
        return ImageAssessment(
            "网页截图",
            keep_raw=False,
            is_sticker=True,
            sticker_assessment=StickerAssessment(False, "网页截图，不是表情包", ""),
        )

    with caplog.at_level(logging.DEBUG):
        res = await _normalize_direct(monkeypatch, png_url(64), assess)
    assert res.sticker_flags == (False,)
    assert res.sticker_metas == (None,)
    assert "审核未通过" in caplog.text and "网页截图" in caplog.text


# ---------------------------------------------------------------- smart 选图（store 层）


async def _collect_with_meta(store, size: int, ts: float, description: str, emotion: str):
    metas = (StickerAssessment(True, description, emotion),)
    assert await store.collect(_record_with(png_url(size), ts=ts), (True,), metas) == 1


async def test_smart_pick_selected_by_model(tmp_path):
    """smart 命中：候选（描述, 情绪）交给模型，按其返回编号取对应条目。"""
    ai = FakeStickerAI(pick_target="熊猫摊手")
    store, db = await _make_store(tmp_path, ai, select_mode="smart")
    await _collect_with_meta(store, 10, 100.0, "柴犬歪头", "疑惑")
    await _collect_with_meta(store, 11, 200.0, "熊猫摊手", "无语")
    picked = await store.pick_for_send_smart(42, "语境文本", SeededRng(7))
    assert picked is not None
    assert picked.sha256 == image_fingerprint(png_url(11))  # 模型选的就是它
    context, entries = ai.pick_calls[0]
    assert context == "语境文本"
    assert {(d, e) for d, e in entries} == {("柴犬歪头", "疑惑"), ("熊猫摊手", "无语")}
    await db.close()


async def test_smart_model_declines_returns_none(tmp_path, caplog):
    """模型明确「不发」：返回 None 作罢，INFO 日志带理由，不退回随机。"""
    ai = FakeStickerAI(pick=(None, "语境里没合适的"))
    store, db = await _make_store(tmp_path, ai, select_mode="smart")
    await _collect_with_meta(store, 10, 100.0, "柴犬歪头", "疑惑")
    with caplog.at_level(logging.INFO):
        assert await store.pick_for_send_smart(42, "语境", SeededRng(1)) is None
    assert len(ai.pick_calls) == 1
    assert "本次不跟发表情包" in caplog.text and "语境里没合适的" in caplog.text
    await db.close()


async def test_smart_failure_falls_back_to_random(tmp_path, caplog):
    """选图调用失败：WARNING 后退回一次随机抽选（与现状一致，仍把图发出）。"""
    ai = FakeStickerAI(pick_error=RuntimeError("learning 端点挂了"))
    store, db = await _make_store(tmp_path, ai, select_mode="smart")
    await _collect_with_meta(store, 10, 100.0, "柴犬歪头", "疑惑")
    with caplog.at_level(logging.WARNING):
        picked = await store.pick_for_send_smart(42, "语境", SeededRng(1))
    assert picked is not None and picked.sha256 == image_fingerprint(png_url(10))
    assert "退回随机抽选" in caplog.text
    await db.close()


async def test_smart_without_meta_candidates_uses_random(tmp_path):
    """收藏全是无 meta 条目：不进模型选图，直接随机兜底。"""
    ai = FakeStickerAI(pick=(0, "不该被调用"))
    store, db = await _make_store(tmp_path, ai, select_mode="smart")
    # 无合并 meta 传入、审核脚本恒 None（vision 不可用）→ 收藏但无 meta
    assert await store.collect(_record_with(png_url(64), ts=100.0), (True,)) == 1
    assert await store.pick_for_send_smart(42, "语境", SeededRng(1)) is not None
    assert ai.pick_calls == []
    await db.close()


async def test_smart_no_ai_provider_uses_random(tmp_path):
    """未注入 AI 客户端（如假端点部署）：smart 静默退回随机，行为同现状。"""
    db = CandyDatabase(tmp_path / "candy.db")
    await db.create_tables()
    settings = lambda: _settings(select_mode="smart")
    ai = FakeStickerAI()
    store_ai = StickerStore(tmp_path / "stickers", db, settings, lambda: ai)
    await _collect_with_meta(store_ai, 10, 100.0, "柴犬歪头", "疑惑")
    store_no_ai = StickerStore(tmp_path / "stickers", db, settings, None)
    assert await store_no_ai.pick_for_send_smart(42, "语境", SeededRng(1)) is not None
    assert ai.pick_calls == []  # 一次模型请求都没发
    await db.close()


async def test_smart_candidates_rotate(tmp_path):
    """候选超过上限时按最久未使用优先轮换：刚用过的掉出池子、新面孔补进来。"""
    sha_to_desc = {}
    ai = FakeStickerAI(pick=(0, "固定选编号一"))
    store, db = await _make_store(
        tmp_path, ai, select_mode="smart", smart_max_candidates=25
    )
    for offset, size in enumerate(range(10, 40)):  # 30 张，各带唯一描述 meta
        await _collect_with_meta(store, size, 100.0 + offset, f"描述{size}", "得意")
        sha_to_desc[image_fingerprint(png_url(size))] = f"描述{size}"
    picked_first = await store.pick_for_send_smart(42, "语境", SeededRng(99))
    first_batch = {desc for desc, _ in ai.pick_calls[-1][1]}
    assert picked_first is not None and len(first_batch) == 25
    # 被选中并使用过的那张 last_used 刷新 → 掉出「最久未使用前 25」，
    # 从没入过选的新面孔补进来——模型每次都只看到编号一也不会总用同一批
    await store.mark_used(picked_first)
    await store.pick_for_send_smart(42, "语境", SeededRng(99))
    second_batch = {desc for desc, _ in ai.pick_calls[-1][1]}
    used = sha_to_desc[picked_first.sha256]
    assert used not in second_batch
    assert second_batch != first_batch
    await db.close()


async def test_smart_respects_max_candidates_cap(tmp_path):
    ai = FakeStickerAI(pick=(0, "第一张"))
    store, db = await _make_store(
        tmp_path, ai, select_mode="smart", smart_max_candidates=3
    )
    for offset, size in enumerate(range(10, 15)):  # 5 张里只取 3 张进候选
        await _collect_with_meta(store, size, 100.0 + offset, f"描述{size}", "得意")
    await store.pick_for_send_smart(42, "语境", SeededRng(5))
    assert len(ai.pick_calls[-1][1]) == 3
    await db.close()


# ---------------------------------------------------------------- AIClient 协议层


def _vision_cfg(**over) -> ModelConfig:
    return ModelConfig("v", "https://vision.example.com/v1", "kv", None, None, **over)


def _called(instances):
    return [c for c in instances if c.create_kwargs is not None]


async def test_assess_image_merged_request_once(monkeypatch):
    """合并调用：一次请求、工具参数表带上审核三字段、结论解析进 meta。"""
    instances = _install_fake(
        monkeypatch,
        {
            "kv": _tool_msg(
                "submit_assessment",
                '{"summary": "狗歪头", "keep": false, "sticker": true,'
                ' "acceptable": true, "sticker_description": "柴犬歪头疑惑",'
                ' "emotion": "无语"}',
            )
        },
    )
    ai = AIClient(models=_models(vision=_vision_cfg()), generation=_gen())
    assessment = await ai.assess_image(
        "data:image/png;base64,QQ==", want_sticker_meta=True
    )
    (client,) = _called(instances)
    props = client.create_kwargs["tools"][0]["function"]["parameters"]["properties"]
    assert {"acceptable", "sticker_description", "emotion"} <= set(props)
    assert {"summary", "keep", "sticker"} <= set(
        client.create_kwargs["tools"][0]["function"]["parameters"]["required"]
    )
    assert assessment.is_sticker and assessment.keep_raw is False
    assert assessment.sticker_assessment == StickerAssessment(True, "柴犬歪头疑惑", "无语")


async def test_assess_image_without_want_keeps_old_contract(monkeypatch):
    """不要求合并（审核关闭）：工具与提示词与改动前一致。"""
    instances = _install_fake(
        monkeypatch,
        {
            "kv": _tool_msg(
                "submit_assessment", '{"summary": "x", "keep": true, "sticker": false}'
            )
        },
    )
    ai = AIClient(models=_models(vision=_vision_cfg()), generation=_gen())
    assessment = await ai.assess_image("data:image/png;base64,QQ==")
    (client,) = _called(instances)
    props = client.create_kwargs["tools"][0]["function"]["parameters"]["properties"]
    assert set(props) == {"summary", "keep", "sticker"}
    text = client.create_kwargs["messages"][0]["content"][0]["text"]
    assert "acceptable" not in text
    assert assessment.sticker_assessment is None


async def test_assess_image_merged_plain_text_fallback(monkeypatch):
    """纯文本协议下也能解析合并结论（端点不支持 tools 时）。"""
    _install_fake(
        monkeypatch,
        {
            "kv": (
                '{"summary": "狗歪头", "keep": false, "sticker": true,'
                ' "acceptable": false, "sticker_description": "真人照片", "emotion": ""}'
            )
        },
    )
    ai = AIClient(models=_models(vision=_vision_cfg(tool_use=False)), generation=_gen())
    assessment = await ai.assess_image(
        "data:image/png;base64,QQ==", want_sticker_meta=True
    )
    assert assessment.sticker_assessment == StickerAssessment(False, "真人照片", "")


async def test_assess_sticker_tool_and_text(monkeypatch):
    """独立审核：工具与纯文本两种契约都能解析；无可用结论返回 None。"""
    instances = _install_fake(
        monkeypatch,
        {
            "kv": _tool_msg(
                "submit_sticker_assessment",
                '{"acceptable": true, "description": "猫捂脸偷笑", "emotion": "得意"}',
            )
        },
    )
    ai = AIClient(models=_models(vision=_vision_cfg()), generation=_gen())
    result = await ai.assess_sticker("data:image/png;base64,QQ==")
    (client,) = _called(instances)
    assert (
        client.create_kwargs["tools"][0]["function"]["name"]
        == "submit_sticker_assessment"
    )
    assert result == StickerAssessment(True, "猫捂脸偷笑", "得意")

    _install_fake(
        monkeypatch,
        {"kv": '{"acceptable": false, "description": "二维码广告图", "emotion": ""}'},
    )
    ai2 = AIClient(models=_models(vision=_vision_cfg(tool_use=False)), generation=_gen())
    result2 = await ai2.assess_sticker("data:image/png;base64,QQ==")
    assert result2 == StickerAssessment(False, "二维码广告图", "")

    # 通过却没有描述：meta 无使用价值，按「无结论」退回（WARNING 由调用侧记）
    _install_fake(monkeypatch, {"kv": '{"acceptable": true, "description": "  "}'})
    ai3 = AIClient(models=_models(vision=_vision_cfg(tool_use=False)), generation=_gen())
    assert await ai3.assess_sticker("data:image/png;base64,QQ==") is None


async def test_assess_sticker_unparseable_returns_none(monkeypatch, caplog):
    _install_fake(monkeypatch, {"kv": "这不是 JSON"})
    ai = AIClient(models=_models(vision=_vision_cfg(tool_use=False)), generation=_gen())
    with caplog.at_level(logging.WARNING):
        assert await ai.assess_sticker("data:image/png;base64,QQ==") is None
    assert "表情包审核输出无法解析" in caplog.text


async def test_assess_sticker_without_vision_returns_none():
    ai = AIClient(models=_models(), generation=_gen())
    assert await ai.assess_sticker("data:image/png;base64,QQ==") is None  # 不配置：None


def _learning_ai(learning: ModelConfig | None = None):
    return AIClient(
        models=ModelSettings(
            judge=ModelConfig("j", "https://j/v1", "kj", None, None),
            reply=ModelConfig("r", "https://r/v1", "kr", None, None),
            vision=None,
            learning=learning,
        ),
        generation=_gen(),
    )


async def test_pick_sticker_tool_protocol(monkeypatch):
    """smart 选图：learning 角色（未配置时继承 judge）强制工具调用。"""
    instances = _install_fake(
        monkeypatch,
        {"kj": _tool_msg("submit_sticker_pick", '{"pick": 2, "reason": "相衬"}')},
    )
    ai = _learning_ai()
    index, reason = await ai.pick_sticker(
        "小明: 在吗\n【你刚发出的】在的", [("柴犬歪头", "疑惑"), ("熊猫摊手", "无语")]
    )
    (client,) = _called(instances)
    assert client.base_url == "https://j/v1"  # 走 learning（继承 judge）的提供商
    assert client.create_kwargs["tools"][0]["function"]["name"] == "submit_sticker_pick"
    assert index == 1 and reason == "相衬"
    prompt = client.create_kwargs["messages"][0]["content"]
    assert "1. 柴犬歪头【疑惑】" in prompt and "2. 熊猫摊手【无语】" in prompt
    assert "宁可不发" in prompt


async def test_pick_sticker_text_protocol_no_tools(monkeypatch):
    """learning.tool_use=False：不带 tools 参数，正文 JSON 照常解析。"""
    instances = _install_fake(monkeypatch, {"kl": '{"pick": 1, "reason": "得意配得意"}'})
    learning = ModelConfig(
        "l", "https://learn.example.com/v1", "kl", None, None, tool_use=False
    )
    ai = _learning_ai(learning)
    index, reason = await ai.pick_sticker(
        "语境", [("柴犬歪头", "疑惑"), ("熊猫摊手", "无语")]
    )
    assert (index, reason) == (0, "得意配得意")
    (client,) = _called(instances)
    assert client.base_url == "https://learn.example.com/v1"
    assert "tools" not in client.create_kwargs


async def test_pick_sticker_bad_output_raises(monkeypatch):
    """解析歧义/编号越界一律抛错（由调用方退回随机），绝不与「不发」混淆。"""
    _install_fake(
        monkeypatch,
        {"kj": _tool_msg("submit_sticker_pick", '{"pick": 99, "reason": "x"}')},
    )
    with pytest.raises(ValueError):
        await _learning_ai().pick_sticker("语境", [("a", "b"), ("c", "d")])

    _install_fake(monkeypatch, {"kj": _content_msg("完全不是 JSON")})
    with pytest.raises(ValueError):
        await _learning_ai().pick_sticker("语境", [("a", "b")])

    _install_fake(
        monkeypatch, {"kj": _tool_msg("submit_sticker_pick", '{"reason": "忘了填"}')}
    )
    with pytest.raises(ValueError):
        await _learning_ai().pick_sticker("语境", [("a", "b")])


async def test_pick_sticker_degrades_on_tools_rejection(monkeypatch):
    """端点拒绝 tools：sticker_pick 协议开关独立降级，当次补发不丢选图。"""
    calls: list[dict] = []

    class _RejectingOpenAI:
        def __init__(self, *, base_url=None, api_key=None):
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            calls.append(kwargs)
            if "tools" in kwargs:
                raise RuntimeError("This model does not support tools")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=_content_msg('{"pick": 1, "reason": "行"}'))
                ]
            )

    monkeypatch.setattr("candybot.ai.AsyncOpenAI", _RejectingOpenAI)
    ai = _learning_ai()
    index, reason = await ai.pick_sticker("语境", [("a", "b"), ("c", "d")])
    assert (index, reason) == (0, "行")
    assert "tools" in calls[0] and "tools" not in calls[1]
    assert ai._tools_on["sticker_pick"] is False
    assert ai._tools_on["judge"] is True  # 与 judge/reply 的降级状态互不牵连


# ---------------------------------------------------------------- 配置解析


def test_sticker_v2_settings_defaults():
    default = StickerSettings()
    assert default.moderation_enabled is True  # 默认开审核（未配 vision 自然退回）
    assert default.select_mode == "random"  # 默认仍是随机跟发
    assert default.smart_max_candidates == 25


def test_sticker_v2_settings_parsed():
    cfg = base_cfg(
        stickers={
            "select_mode": "smart",
            "moderation_enabled": False,
            "smart_max_candidates": 8,
        }
    )
    settings = load_settings(DictCfg(cfg))
    assert settings.stickers.select_mode == "smart"
    assert settings.stickers.moderation_enabled is False
    assert settings.stickers.smart_max_candidates == 8


@pytest.mark.parametrize(
    "bad",
    [{"select_mode": "clever"}, {"smart_max_candidates": 0}, {"moderation_enabled": "yes"}],
)
def test_sticker_v2_settings_validation(bad):
    with pytest.raises(ValueError):
        load_settings(DictCfg(base_cfg(stickers=bad)))
