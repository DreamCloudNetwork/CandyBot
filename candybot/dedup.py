"""message_id 去重：OneBot 有时会重复上报同一事件。"""

from __future__ import annotations

from collections import deque


class MessageDedup:
    """固定容量的已见 message_id 集合，O(1) 判定、FIFO 淘汰。"""

    def __init__(self, capacity: int = 4096):
        if capacity <= 0:
            raise ValueError("capacity 必须为正数")
        self._capacity = capacity
        self._seen: set[int] = set()
        self._order: deque[int] = deque(maxlen=self._capacity)

    def check_and_mark(self, message_id: int) -> bool:
        """首次出现返回 False 并记录；重复出现返回 True。"""
        if message_id in self._seen:
            return True
        if len(self._order) == self._order.maxlen and self._order.maxlen is not None:
            evicted = self._order[0]
            self._seen.discard(evicted)
        self._order.append(message_id)
        self._seen.add(message_id)
        return False
