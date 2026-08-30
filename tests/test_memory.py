from __future__ import annotations

import asyncio
import sqlite3
import time

import pytest
from sqlalchemy.exc import IntegrityError

from candybot.memory import MemoryManager
from candybot.models import ChatRecord

IMG_A = "data:image/png;base64,AAAA"
IMG_B = "data:image/png;base64,BBBB"
IMG_C = "data:image/png;base64,CCCC"


def make_record(mid: int, text: str = "hello", **kw) -> ChatRecord:
    base = dict(
        message_id=mid,
        group_id=42,
        user_id=1000 + mid,
        nickname=f"u{mid}",
        text=text,
        ts=time.time() + mid,
    )
    base.update(kw)
    return ChatRecord(**base)


def img_record(mid: int, *, images=(IMG_A,), states=(), summaries=None,
               text="看图", ts=None) -> ChatRecord:
    return ChatRecord(
        message_id=mid,
        group_id=42,
        user_id=1000 + mid,
        nickname=f"u{mid}",
        text=text,
        ts=ts if ts is not None else time.time() + mid,
        images=tuple(images),
        image_states=tuple(states),
        image_summaries=summaries,
    )


@pytest.fixture
async def mgr(tmp_path):
    manager = MemoryManager(tmp_path)
    yield manager
    await manager.close()


def blob_count(tmp_path) -> int:
    """直接查库：image_blob 里的原图份数（去重后的存储量）。"""
    conn = sqlite3.connect(tmp_path / "candy.db")
    try:
        return int(conn.execute("SELECT count(*) FROM image_blob").fetchone()[0])
    finally:
        conn.close()


def history_count(tmp_path, group_id: int) -> int:
    conn = sqlite3.connect(tmp_path / "candy.db")
    try:
        cur = conn.execute(
            "SELECT count(*) FROM chat_history WHERE group_id = ?", (group_id,)
        )
        return int(cur.fetchone()[0])
    finally:
        conn.close()


# ---------------------------------------------------------------- 基本读写


async def test_append_and_reload(tmp_path, mgr):
    mem = await mgr.get(42)
    for i in range(5):
        await mem.append(make_record(i))
    assert len(mem) == 5

    # 新实例模拟重启恢复
    mgr2 = MemoryManager(tmp_path)
    mem2 = await mgr2.get(42)
    assert [r.message_id for r in mem2.tail(10)] == list(range(5))
    await mgr2.close()


async def test_text_history_retained_beyond_memory_capacity(tmp_path):
    """热缓存有界，但库里的文本历史全量保留。"""
    mgr_small = MemoryManager(tmp_path, default_capacity=8)
    mem = await mgr_small.get(42)
    for i in range(20):
        await mem.append(make_record(i))
    assert len(mem) == 8  # 内存仍有界

    # 同等容量的新实例只回放最近 8 条
    mgr2 = MemoryManager(tmp_path, default_capacity=8)
    assert [r.message_id for r in (await mgr2.get(42)).tail(10)] == list(range(12, 20))
    # 更大容量的新实例能取回全部 20 条 → 文本历史未被裁剪
    mgr3 = MemoryManager(tmp_path, default_capacity=64)
    assert [r.message_id for r in (await mgr3.get(42)).tail(64)] == list(range(20))
    for m in (mgr_small, mgr2, mgr3):
        await m.close()


async def test_tail_excluding_last(mgr):
    mem = await mgr.get(1)
    for i in range(4):
        await mem.append(make_record(i))
    got = mem.tail_excluding_last(3)
    assert [r.message_id for r in got] == [0, 1, 2]


async def test_find_by_message_id_includes_self_records(mgr):
    mem = await mgr.get(1)
    await mem.append(make_record(9, is_self=True))
    found = await mem.find_by_message_id(9)
    assert found is not None and found.is_self


async def test_find_by_message_id_falls_back_to_db(tmp_path):
    """超出热缓存容量的历史消息仍能按 id 查到（回复引用早于启动的消息）。"""
    manager = MemoryManager(tmp_path, default_capacity=8)
    mem = await manager.get(42)
    for i in range(20):
        await mem.append(make_record(i))
    found = await mem.find_by_message_id(3)  # 不在热缓存里
    assert found is not None and found.text == "hello"
    await manager.close()


# ---------------------------------------------------------------- 撤回与去重


async def test_remove_deletes_record_and_db_row(tmp_path, mgr):
    mem = await mgr.get(42)
    for i in range(5):
        await mem.append(make_record(i))
    assert await mem.remove(2) is True
    assert [r.message_id for r in mem.tail(10)] == [0, 1, 3, 4]

    # 库已同步：新实例（模拟重启）后撤回的消息不会复活
    mgr2 = MemoryManager(tmp_path)
    assert [r.message_id for r in (await mgr2.get(42)).tail(10)] == [0, 1, 3, 4]
    await mgr2.close()


async def test_remove_missing_returns_false(mgr):
    mem = await mgr.get(3)
    await mem.append(make_record(0))
    assert await mem.remove(99) is False
    assert [r.message_id for r in mem.tail(10)] == [0]


async def test_remove_beyond_memory_capacity(tmp_path):
    """全量历史入库后，早于启动的旧消息也能撤回。"""
    manager = MemoryManager(tmp_path, default_capacity=8)
    mem = await manager.get(42)
    for i in range(20):
        await mem.append(make_record(i))
    assert await mem.remove(3) is True
    mgr2 = MemoryManager(tmp_path, default_capacity=64)
    ids = [r.message_id for r in (await mgr2.get(42)).tail(64)]
    assert 3 not in ids and len(ids) == 19
    await manager.close()
    await mgr2.close()


async def test_duplicate_message_id_ignored(mgr, tmp_path):
    mem = await mgr.get(42)
    await mem.append(make_record(7))
    await mem.append(make_record(7))  # 同群同 id 重复入库
    assert len(mem) == 1
    assert history_count(tmp_path, 42) == 1


async def test_same_message_id_across_groups(mgr):
    mem42 = await mgr.get(42)
    mem43 = await mgr.get(43)
    await mem42.append(make_record(7))
    await mem43.append(make_record(7, group_id=43))  # 不同群互不冲突
    assert len(mem42) == 1 and len(mem43) == 1


# ---------------------------------------------------------------- 图片存储


async def test_identical_image_stored_once(tmp_path, mgr):
    mem = await mgr.get(42)
    await mem.append(img_record(1))
    await mem.append(img_record(2))  # 同一张图再来一帖
    assert blob_count(tmp_path) == 1  # 全库只有一份 base64

    mgr2 = MemoryManager(tmp_path)
    got = (await mgr2.get(42)).tail(10)
    assert [r.images for r in got] == [(IMG_A,), (IMG_A,)]  # 引用全部还原
    await mgr2.close()


async def test_same_message_duplicate_images_deduped(tmp_path, mgr):
    mem = await mgr.get(42)
    await mem.append(img_record(5, images=(IMG_A, IMG_B, IMG_A)))
    assert blob_count(tmp_path) == 2
    rec = await mem.find_by_message_id(5)
    assert rec.images == (IMG_A, IMG_B, IMG_A)


async def test_blob_survives_when_one_reference_removed(tmp_path, mgr):
    """撤回其中一条引用后，另一条记录的原图仍在（引用计数语义）。"""
    mem = await mgr.get(42)
    await mem.append(img_record(1))
    await mem.append(img_record(2))
    assert await mem.remove(1) is True
    fresh = MemoryManager(tmp_path)
    rec = await (await fresh.get(42)).find_by_message_id(2)
    assert rec is not None and rec.images == (IMG_A,)
    await fresh.close()


# ---------------------------------------------------------------- 图片形态切换


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


# ---------------------------------------------------------------- 保留期回收


async def test_prune_expires_old_images_keeps_text_and_summary(tmp_path, mgr):
    now = time.time()
    mem = await mgr.get(42)
    await mem.append(
        img_record(1, images=(IMG_A, IMG_B), summaries={0: "老图总结"},
                   ts=now - 10 * 86400)  # 超过保留期
    )
    await mem.append(img_record(2, images=(IMG_C,), ts=now))

    degraded, freed = await mgr.prune_expired_images()
    assert (degraded, freed) == (2, 2)  # 旧记录两张图降级、两份原图释放

    mgr2 = MemoryManager(tmp_path)
    recs = {r.message_id: r for r in (await mgr2.get(42)).tail(10)}
    old, new = recs[1], recs[2]
    assert old.images == ("", "")          # 文本记录保留，原图数据清空
    assert old.state_of(0) == "summarized"  # 有总结 → 总结
    assert old.summary_of(0) == "老图总结"   # 总结属于文本语义，永久保留
    assert old.state_of(1) == "placeholder"  # 无总结 → 占位符
    assert old.text == "看图"
    assert new.images == (IMG_C,)           # 保留期内的新图不受影响
    await mgr2.close()


async def test_prune_keeps_blob_shared_with_recent_message(tmp_path, mgr):
    now = time.time()
    mem = await mgr.get(42)
    await mem.append(img_record(1, ts=now - 10 * 86400))
    await mem.append(img_record(2, ts=now))  # 同一张图的新引用

    degraded, freed = await mgr.prune_expired_images()
    assert (degraded, freed) == (1, 0)  # 旧槽位降级，但原图被新引用保住

    mgr2 = MemoryManager(tmp_path)
    recs = {r.message_id: r for r in (await mgr2.get(42)).tail(10)}
    assert recs[1].images == ("",)
    assert recs[2].images == (IMG_A,)
    await mgr2.close()


async def test_retention_days_floor(tmp_path):
    mgr_small = MemoryManager(tmp_path, image_retention_days=0)
    assert mgr_small.image_retention_days == 1
    await mgr_small.close()


# ---------------------------------------------------------------- 边界情况


async def test_prune_skips_records_with_unknown_time(tmp_path, mgr):
    """事件缺 time 字段（ts=0）的消息不参与回收：时间未知 ≠ 过期。"""
    mem = await mgr.get(42)
    await mem.append(img_record(1, ts=0.0))

    degraded, freed = await mgr.prune_expired_images()
    assert (degraded, freed) == (0, 0)
    recs = await mgr.db.load_recent(42, 10)
    assert recs[0].images == (IMG_A,)
    assert blob_count(tmp_path) == 1


async def test_append_survives_db_failure(tmp_path, mgr, monkeypatch):
    """库写入失败时退化为仅热缓存：消息不丢、不向事件层抛异常。"""
    mem = await mgr.get(42)

    async def boom(record):
        raise RuntimeError("db down")

    monkeypatch.setattr(mem._db, "insert_record", boom)
    rec = make_record(7, text="只存内存")
    await mem.append(rec)
    assert mem.last() is rec
    assert [r.message_id for r in mem.tail(5)] == [7]

    # 仅存于热缓存的记录，撤回时也算真的删了
    monkeypatch.undo()
    assert await mem.remove(7) is True
    assert mem.last() is None


async def test_insert_record_reraises_non_unique_integrity_error(mgr):
    """非唯一键冲突（如 NOT NULL）不得被误判成『重复入库』而吞掉。"""
    await mgr.db.create_tables()
    bad = make_record(1)
    bad.ts = None  # ts 列 NOT NULL
    with pytest.raises(IntegrityError):
        await mgr.db.insert_record(bad)


async def test_concurrent_appends_keep_cache_and_db_order(tmp_path, mgr):
    """同群并发 append：热缓存顺序必须与库内 row_id 顺序一致。"""
    mem = await mgr.get(42)
    records = [make_record(i, ts=1000.0 + i) for i in range(6)]
    await asyncio.gather(*(mem.append(r) for r in records))
    assert [r.message_id for r in mem.tail(10)] == [0, 1, 2, 3, 4, 5]
    rows = await mgr.db.load_recent(42, 10)
    assert [r.message_id for r in rows] == [0, 1, 2, 3, 4, 5]


async def test_write_slots_downgrades_show_without_data(tmp_path, mgr):
    """库内不变量：show 槽位必有原图；show + 空数据落库时强制降级。"""
    await mgr.db.create_tables()
    rec = img_record(5, images=("",), states=("show",))
    assert await mgr.db.insert_record(rec) is True
    conn = sqlite3.connect(tmp_path / "candy.db")
    try:
        (state,) = conn.execute("SELECT state FROM chat_image").fetchone()
    finally:
        conn.close()
    assert state == "placeholder"


async def test_recall_recycled_image_is_noop(tmp_path):
    """原图已过保留期回收的槽位，recall 视为无操作，状态不升回 show。"""
    mgr = MemoryManager(tmp_path)
    mem = await mgr.get(42)
    await mem.append(img_record(1, ts=time.time() - 10 * 86400))
    assert await mgr.prune_expired_images() == (1, 1)

    mgr2 = MemoryManager(tmp_path)
    mem2 = await mgr2.get(42)
    assert await mem2.transition_images(1, "recall") is False
    rec = mem2.last()
    assert rec.images == ("",)
    assert rec.state_of(0) == "placeholder"
    await mgr2.close()
    await mgr.close()


# ---------------------------------------------------------------- 命令消息与模型上下文


async def test_is_command_persisted_and_replayed(tmp_path):
    """is_command 标记随记录入库，重启回放后仍在（排除模式可继续过滤）。"""
    mgr = MemoryManager(tmp_path)
    mem = await mgr.get(42)
    await mem.append(make_record(1, "/hi", is_command=True))
    await mem.append(make_record(2, "你好！", is_self=True, is_command=True))
    await mem.append(make_record(3, "随便聊聊"))
    rows = await mgr.db.load_recent(42, 10)
    assert [(r.text, r.is_command) for r in rows] == [
        ("/hi", True),
        ("你好！", True),
        ("随便聊聊", False),
    ]
    found = await mgr.db.find_record(42, 2)
    assert found is not None and found.is_command and found.is_self
    await mgr.close()

    mgr2 = MemoryManager(tmp_path)
    mem2 = await mgr2.get(42)
    assert [r.is_command for r in mem2.tail(5)] == [True, True, False]
    await mgr2.close()


async def test_model_tail_filters_commands(mgr):
    """model_tail：include_commands=False 时命令消息与回复不进模型历史层。"""
    mem = await mgr.get(42)
    await mem.append(make_record(1, "/hi", is_command=True))
    await mem.append(make_record(2, "你好！", is_self=True, is_command=True))
    await mem.append(make_record(3, "你们在聊啥"))
    assert [r.text for r in mem.model_tail(10)] == ["/hi", "你好！", "你们在聊啥"]
    assert [r.text for r in mem.model_tail(10, include_commands=False)] == [
        "你们在聊啥"
    ]
    # 过滤先于截取：命令再多也不挤占 context_size 名额
    for i in range(4, 8):
        await mem.append(make_record(i, f"/ping {i}", is_command=True))
    assert [r.text for r in mem.model_tail(2, include_commands=False)] == [
        "你们在聊啥"
    ]
    assert mem.model_tail(0) == []
    # 缺省 include_commands=True：与 tail 一致，命令照常进历史层
    assert [r.text for r in mem.model_tail(2)] == ["/ping 6", "/ping 7"]
