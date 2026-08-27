import pytest

from candybot.dedup import MessageDedup


def test_first_seen_false_then_true():
    d = MessageDedup()
    assert d.check_and_mark(1) is False
    assert d.check_and_mark(1) is True
    assert d.check_and_mark(2) is False


def test_capacity_eviction():
    d = MessageDedup(capacity=4)
    for i in range(4):
        assert d.check_and_mark(i) is False
    # 满：每次未命中插入都按 FIFO 挤出当前最旧的
    assert d.check_and_mark(4) is False   # 挤出 0
    assert d.check_and_mark(1) is True    # 1 仍在记录中 → 命中
    assert d.check_and_mark(0) is False   # 重新记录已挤出的 0（顺带挤出 1）
    assert d.check_and_mark(1) is False   # 刚被挤出的 1 可重录（顺带挤出 2）
    assert d.check_and_mark(3) is True    # 3 尚未被波及，仍命中
