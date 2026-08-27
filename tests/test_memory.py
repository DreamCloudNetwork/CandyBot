from __future__ import annotations

import json
import time

from candybot.memory import MemoryManager, _safe_memory_file
from candybot.models import ChatRecord


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


def test_append_and_reload(tmp_path):
    mgr = MemoryManager(tmp_path)
    mem = mgr.get(42)
    for i in range(5):
        mem.append(make_record(i))
    assert len(mem) == 5

    # 新实例模拟重启恢复
    mgr2 = MemoryManager(tmp_path)
    mem2 = mgr2.get(42)
    assert [r.message_id for r in mem2.tail(10)] == list(range(5))


def test_capacity_bound(tmp_path):
    mgr = MemoryManager(tmp_path)
    mem = mgr.get(7)
    mem.capacity = 8
    import collections

    mem._records = collections.deque(maxlen=8)  # 模拟小容量群
    for i in range(20):
        mem.append(make_record(i))
    assert len(mem) == 8
    assert [r.message_id for r in mem.tail(8)] == list(range(12, 20))
    # 内存有界；手动触发压缩后文件也收敛到容量以内
    mem._rewrite()
    lines = (tmp_path / "memory" / "7.jsonl").read_text().strip().splitlines()
    assert len(lines) == 8


def test_tail_excluding_last(tmp_path):
    mgr = MemoryManager(tmp_path)
    mem = mgr.get(1)
    for i in range(4):
        mem.append(make_record(i))
    got = mem.tail_excluding_last(3)
    assert [r.message_id for r in got] == [0, 1, 2]


def test_find_by_message_id_includes_self_records(tmp_path):
    mgr = MemoryManager(tmp_path)
    mem = mgr.get(1)
    mem.append(make_record(9, is_self=True))
    found = mem.find_by_message_id(9)
    assert found is not None and found.is_self


def test_corrupt_line_skipped(tmp_path):
    d = tmp_path / "memory"
    d.mkdir()
    recs = [make_record(i) for i in range(3)]
    (d / "5.jsonl").write_text(
        "\n".join(["{broken json"] + [json.dumps(r.to_json()) for r in recs]),
        encoding="utf-8",
    )
    mgr = MemoryManager(tmp_path)
    mem = mgr.get(5)
    assert [r.message_id for r in mem.tail(10)] == [0, 1, 2]


def test_safe_memory_file_rejects_traversal(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        _safe_memory_file(tmp_path, "../evil.jsonl")
    with pytest.raises(ValueError):
        _safe_memory_file(tmp_path, "a/b.jsonl")
    with pytest.raises(ValueError):
        _safe_memory_file(tmp_path, "../../etc/passwd")


def test_duplicate_lines_dedup_on_load(tmp_path):
    d = tmp_path / "memory"
    d.mkdir()
    rec = make_record(3)
    body = json.dumps(rec.to_json())
    (d / "6.jsonl").write_text(f"{body}\n{body}\n", encoding="utf-8")
    mgr = MemoryManager(tmp_path)
    assert len(mgr.get(6)) == 1
