"""图片记忆管理：base64 入库、展示状态机、模型 drop/recall 标记链路。"""

from __future__ import annotations

import asyncio
import time

import pytest

from candybot.ai import ImageAssessment, AIClient, ReplyDraft, split_image_ops
from candybot.memory import MemoryManager
from candybot.models import ChatRecord, load_settings
from candybot.normalize import normalize_group_message
from candybot.prompts import HistoryTurn, reply_history_turns

IMG_A = "data:image/png;base64,AAAA"
IMG_B = "data:image/png;base64,BBBB"
IMG_C = "data:image/png;base64,CCCC"


def img_record(mid: int, *, images=(IMG_A,), states=(), summaries=None, text="看图") -> ChatRecord:
    return ChatRecord(
        message_id=mid,
        group_id=42,
        user_id=1000 + mid,
        nickname=f"u{mid}",
        text=text,
        ts=time.time() + mid,
        images=tuple(images),
        image_states=tuple(states),
        image_summaries=summaries,
    )


# ---------------------------------------------------------------- 模型与入库


async def test_record_roundtrip_persists_base64_and_states(tmp_path):
    manager = MemoryManager(tmp_path)
    mem = await manager.get(42)
    rec = img_record(1, images=(IMG_A, IMG_B), states=("show", "summarized"),
                     summaries={1: "一张猫图"})
    await mem.append(rec)

    # 重启后 base64、状态、总结全部原样恢复
    mgr2 = MemoryManager(tmp_path)
    loaded = await (await mgr2.get(42)).find_by_message_id(1)
    assert loaded is not None
    assert loaded.images == (IMG_A, IMG_B)
    assert [loaded.state_of(i) for i in range(2)] == ["show", "summarized"]
    assert loaded.summary_of(1) == "一张猫图"
    await manager.close()
    await mgr2.close()


def test_set_image_state_and_placeholder_default():
    rec = img_record(7, images=(IMG_A, IMG_B))
    rec.set_image_state(1, "placeholder")
    assert rec.state_of(1) == "placeholder" and rec.state_of(0) == "show"
    try:
        rec.set_image_state(0, "bogus")
        raise AssertionError("应拒绝非法状态")
    except ValueError:
        pass


# ---------------------------------------------------------------- 历史层渲染


def test_reply_turn_renders_each_image_shape():
    rec = img_record(
        9,
        images=(IMG_A, IMG_B, IMG_C),
        states=("summarized", "placeholder", "show"),
        summaries={0: "一只猫"},
        text="前文\n[图片]",
    )
    turns, _ = reply_history_turns([rec], max_chars=10**6, max_images=8)
    (turn,) = turns
    assert turn.images == (IMG_C,)
    body_lines = turn.content.splitlines()
    assert body_lines[0] == "u9(1009)：前文"
    assert "[图片：一只猫]" in body_lines[1:]
    assert "[图片]" in body_lines[1:]
    assert turn.content.count("[图片]") == 1  # 占位行不再重复出现


def test_reply_turn_renders_pruned_image_slots():
    """保留期回收后的槽位（数据为空）按总结/占位符渲染，绝不进入原图块。"""
    rec = img_record(
        6,
        images=("", IMG_B),
        states=("summarized", "show"),
        summaries={0: "回收的猫图"},
        text="看图",
    )
    turns, _ = reply_history_turns([rec], max_chars=10**6, max_images=8)
    (turn,) = turns
    assert turn.images == (IMG_B,)
    assert "[图片：回收的猫图]" in turn.content


def test_reply_turn_plain_when_no_extra_notes():
    rec = img_record(2, images=(IMG_A,), states=("placeholder",), text="纯文本")
    plain = img_record(4, images=(), text="没图")
    turns, _ = reply_history_turns([rec, plain], max_chars=10**6, max_images=8)
    assert turns[0].images == ()
    assert turns[1].content == "u4(1004)：没图" and isinstance(turns[1], HistoryTurn)


def test_reply_history_image_cap_drops_oldest_first():
    recs = [
        img_record(i, images=(f"data:x;base64,{i}",), states=("show",), text=f"m{i}")
        for i in range(3)
    ]
    turns, _ = reply_history_turns(recs, max_chars=10**6, max_images=2)
    attached = [bool(t.images) for t in turns]
    assert attached == [False, True, True]  # 最旧的被摘除


def test_reply_history_multi_image_partial_keeps_newer_tail():
    rec = img_record(1, images=(IMG_A, IMG_B, IMG_C), states=("show",) * 3)
    old = img_record(0, images=(IMG_A,), states=("show",))
    # 共 4 张、上限 3 → 摘掉最旧一条的附图，新消息整条保留
    turns, _ = reply_history_turns([old, rec], max_chars=10**6, max_images=3)
    assert turns[0].images == ()
    assert turns[1].images == (IMG_A, IMG_B, IMG_C)
    # 上限再收紧到 2 → 新消息内部也从头部摘，只留较新的尾部
    turns, _ = reply_history_turns([old, rec], max_chars=10**6, max_images=2)
    assert turns[0].images == () and turns[1].images == (IMG_B, IMG_C)


def test_reply_history_char_truncation_still_applies():
    recs = [img_record(i, text="x" * 100) for i in range(5)]
    turns, truncated = reply_history_turns(recs, max_chars=120, max_images=8)
    assert truncated and len(turns) < 5


# ---------------------------------------------------------------- 标记解析


def test_split_image_ops_extracts_and_cleans():
    clean, ops = split_image_ops("哈哈\n<drop_img 12345>")
    assert clean == "哈哈" and ops[0].action == "drop_img" and ops[0].message_id == 12345

    clean, ops = split_image_ops("<recall_img 77>让我看看")
    assert clean == "让我看看" and [(o.action, o.message_id) for o in ops] == [("recall_img", 77)]

    clean, ops = split_image_ops("a\n\n<drop_img 1>\n\n<recall_img 2>\n\nb")
    assert clean == "a\nb" and len(ops) == 2


def test_split_image_ops_strips_malformed_tags():
    clean, ops = split_image_ops("<drop_img abc>正文</recall_img>")
    assert clean == "正文" and ops == []


# ---------------------------------------------------------------- 收图评估解析


def test_parse_assessment_variants():
    ok = AIClient._parse_assessment('{"summary":"猫","keep":false}')
    assert ok == ImageAssessment(summary="猫", keep_raw=False)
    thinked = AIClient._parse_assessment('<think>x</think>{"summary":"狗"}')
    assert thinked.keep_raw is True and thinked.summary == "狗"
    fallback = AIClient._parse_assessment("完全不是 JSON")
    assert fallback == ImageAssessment(summary=None, keep_raw=True)


# ---------------------------------------------------------------- 归一化入库


async def _normalize_with_assess(mode: str, assessor):
    from candybot.models import MultimodalSettings

    async def no_ref(_i):
        return None

    mm = MultimodalSettings(mode=mode, download_media=True)
    event = {
        "post_type": "message",
        "message_type": "group",
        "group_id": 42,
        "user_id": 1000,
        "message_id": 11,
        "time": 1700000000,
        "sender": {"card": "小明"},
        "message": [{"type": "image", "data": {"url": f"https://example.com/{mode}.jpg"}}],
    }
    return await normalize_group_message(
        event,
        self_qq=99,
        multimodal=mm,
        find_by_message_id=no_ref,
        # 会话对象仅作非空标记：下载已被 monkeypatch，不会真正使用
        http_session=object(),
        assess_image=assessor,
    )


def test_direct_intake_keep_and_drop(monkeypatch):
    import candybot.normalize as norm_mod

    async def fake_download(session, url, **_kwargs):
        return IMG_A

    monkeypatch.setattr(norm_mod, "_download_as_data_url", fake_download)

    async def keep(data_url):
        return ImageAssessment(summary="关键截图", keep_raw=True)

    result = asyncio.run(_normalize_with_assess("direct", keep))
    assert result.record.images == (IMG_A,)
    assert result.record.state_of(0) == "show"

    async def drop(data_url):
        return ImageAssessment(summary="梗图，总结即可", keep_raw=False)

    result = asyncio.run(_normalize_with_assess("direct", drop))
    assert result.record.state_of(0) == "summarized"
    assert result.record.summary_of(0) == "梗图，总结即可"
    assert "[图片]" in result.record.text  # 正文仍保留占位标识


def test_direct_intake_assess_failure_keeps_raw(monkeypatch):
    import candybot.normalize as norm_mod

    async def fake_download(session, url, **_kwargs):
        return IMG_A

    monkeypatch.setattr(norm_mod, "_download_as_data_url", fake_download)

    async def boom(data_url):
        raise RuntimeError("vision 挂了")

    result = asyncio.run(_normalize_with_assess("direct", boom))
    assert result.record.state_of(0) == "show"


def test_placeholder_mode_persists_base64_but_no_states(monkeypatch):
    import candybot.normalize as norm_mod

    async def fake_download(session, url, **_kwargs):
        return IMG_A

    monkeypatch.setattr(norm_mod, "_download_as_data_url", fake_download)

    async def never(data_url):  # placeholder 不该触发评估
        raise AssertionError("placeholder 模式不应调用评估")

    result = asyncio.run(_normalize_with_assess("placeholder", never))
    assert result.record.images == (IMG_A,)   # 落盘存档仍要有
    assert result.record.image_states == ("show",)  # 默认形态，展示层忽略


# ---------------------------------------------------------------- 记忆层切换


async def make_mem(tmp_path):
    manager = MemoryManager(tmp_path)
    mem = await manager.get(42)
    await mem.append(
        img_record(31, images=(IMG_A, IMG_B),
                   states=("show", "summarized"), summaries={1: "已有总结"})
    )
    return manager, mem


async def test_transition_drop_keeps_summary_else_placeholder(tmp_path):
    manager, mem = await make_mem(tmp_path)
    assert await mem.transition_images(31, "drop") is True
    rec = await mem.find_by_message_id(31)
    assert rec.state_of(0) == "placeholder"   # 无总结 → 占位符
    assert rec.state_of(1) == "summarized"    # 有总结 → 总结
    # 幂等：再 drop 无变化
    assert await mem.transition_images(31, "drop") is False
    await manager.close()


async def test_transition_recall_restores_original(tmp_path):
    manager, mem = await make_mem(tmp_path)
    await mem.transition_images(31, "drop")
    assert await mem.transition_images(31, "recall") is True
    rec = await mem.find_by_message_id(31)
    assert all(rec.state_of(i) == "show" for i in range(len(rec.images)))
    # 库已同步：新实例读到的也是召回后的形态
    fresh = MemoryManager(tmp_path)
    reloaded = await (await fresh.get(42)).find_by_message_id(31)
    assert all(reloaded.state_of(i) == "show" for i in range(len(reloaded.images)))
    await fresh.close()
    await manager.close()


async def test_transition_unknown_message_or_direction(tmp_path):
    manager, mem = await make_mem(tmp_path)
    assert await mem.transition_images(999, "drop") is False
    with pytest.raises(ValueError):
        await mem.transition_images(31, "nonsense")
    await manager.close()


# ---------------------------------------------------------------- bot 链路（假 AI）


def make_settings(tmp_path, multimodal_mode):
    cfg = {
        "bot": {"self_qq": 99, "data_dir": str(tmp_path / "data")},
        "groups": {
            "42": {
                "persona": "测试人设",
                "proactivity_threshold": 6,
                "context_size": 20,
            }
        },
        "groups_default": {"enabled": False, "persona": "默认"},
        "ai_backend": {"base_url": "https://api.example.com/v1", "api_key": "k"},
        "models": {"judge": "j", "reply": "r"},
        "generation": {},
        "multimodal": {"mode": multimodal_mode, "download_media": True},
        "rate_limit": {},
        "snowluma": {
            "endpoint": "http://10.0.0.5:3000/",
            "allow_private_endpoint": True,
        },
        # 本文件断言的是发送原文（标记剥除等），关闭拟人化后处理的
        # 拆条/错别字随机加工；后处理自身用例见 test_bot_postprocess.py
        "response_post_process": {"enabled": False},
    }

    class DictCfg:
        def __init__(self, data):
            self._d = data

        def __getattr__(self, name):
            return self._d[name]

    return load_settings(DictCfg(cfg))


class MarkerAI:
    """reply 固定返回带标记的草稿；记录每次调用看到的历史快照。"""

    # bot 据此选择 reply L1 守则的输出契约措辞
    reply_tool_use = True

    def __init__(self, drafts):
        self.drafts = list(drafts)
        self.calls: list[list[tuple]] = []

    async def generate_reply(self, static_system, runtime_system, recent,
                             current_message, now_text, **kwargs):
        self.calls.append(
            [
                (r.message_id, tuple(r.state_of(i) for i in range(len(r.images))))
                for r in recent
            ]
        )
        return ReplyDraft(self.drafts.pop(0) if self.drafts else "兜底回复")


class FakeSnow:
    def __init__(self):
        self.sent: list[str] = []

    async def start(self):
        pass

    async def probe(self):
        pass

    async def stop(self):
        pass

    async def query_login_info(self):
        return None

    async def send_group_msg(self, group_id, text):
        self.sent.append(text)


async def build_marker_bot(tmp_path, mode, drafts):
    from candybot.bot import CandyBot

    bot = CandyBot(make_settings(tmp_path, mode))
    bot._snowluma = FakeSnow()
    bot._ai = MarkerAI(drafts)
    return bot


def at_event(mid, text):
    return {
        "post_type": "message",
        "message_type": "group",
        "group_id": 42,
        "user_id": 1000,
        "message_id": mid,
        "time": int(time.time()),
        "sender": {"card": "小明"},
        "message": [{"type": "at", "data": {"qq": "99"}},
                    {"type": "text", "data": {"text": text}}],
    }


async def wait_until(cond, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        await asyncio.sleep(0.02)
    raise TimeoutError()


async def test_bot_applies_drop_marker_and_sends_clean_text(tmp_path):
    bot = await build_marker_bot(tmp_path, "direct", ["好可爱\n<drop_img 51>"])
    mem = await bot._memory.get(42)
    await mem.append(img_record(51, summaries={0: "一只橘猫"}))
    try:
        await bot._on_event(at_event(52, "这张图太搞笑了"))
        await wait_until(lambda: bool(bot._snowluma.sent))
        assert bot._snowluma.sent == ["好可爱"]            # 标记未泄进群里
        rec = await mem.find_by_message_id(51)
        assert rec.state_of(0) == "summarized"             # 已降级为总结
    finally:
        await bot.stop()


async def test_bot_recalls_image_and_regenerates_once(tmp_path):
    bot = await build_marker_bot(
        tmp_path, "direct", ["先看看<recall_img 9>", "看到了"]
    )
    mem = await bot._memory.get(42)
    await mem.append(img_record(9, states=("placeholder",)))
    try:
        await bot._on_event(at_event(10, "把之前那张图翻出来看看"))
        await wait_until(lambda: len(bot._ai.calls) >= 2)
        await wait_until(lambda: bool(bot._snowluma.sent))
        assert len(bot._ai.calls) == 2                     # 召回触发一次重写
        first_snapshot = dict(bot._ai.calls[0])
        second_snapshot = dict(bot._ai.calls[1])
        assert first_snapshot[9] == ("placeholder",)
        assert second_snapshot[9] == ("show",)             # 二稿已能看到原图
        assert bot._snowluma.sent == ["看到了"]            # 发送的是干净二稿
    finally:
        await bot.stop()


async def test_bot_ignores_markers_outside_direct_mode(tmp_path):
    bot = await build_marker_bot(tmp_path, "placeholder", ["哈哈<drop_img 9>"])
    mem = await bot._memory.get(42)
    await mem.append(img_record(9, states=("show",)))
    try:
        await bot._on_event(at_event(11, "随便说说"))
        await wait_until(lambda: bool(bot._snowluma.sent))
        assert bot._snowluma.sent == ["哈哈"]              # 标记照样剥除
        assert (await mem.find_by_message_id(9)).state_of(0) == "show"
        assert len(bot._ai.calls) == 1                     # 未触发重生成
    finally:
        await bot.stop()


async def test_self_message_id_strictly_decreasing_under_clock_rollback(
    tmp_path, monkeypatch
):
    """时钟回拨时合成负 id 仍严格递减，不会撞 (group_id, message_id) 唯一键。"""
    bot = await build_marker_bot(tmp_path, "direct", [])
    try:
        first = bot._next_self_message_id()
        assert first < 0
        # 模拟回拨：time_ns 返回比第一次更早的时刻 → 候选 id 反而更大
        monkeypatch.setattr(time, "time_ns", lambda: -first - 10**9)
        assert bot._next_self_message_id() == first - 1
        assert bot._next_self_message_id() == first - 2
    finally:
        await bot.stop()
