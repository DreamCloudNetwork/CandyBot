"""每群上下文记忆：内存 deque + JSONL 追加落盘，重启后恢复最近记录。

文件名只允许 ``<纯数字群号>.jsonl``，全部经由 :func:`_safe_memory_file`
构造：正则校验 + 解析后目录包含校验，杜绝外部路径成分进入文件系统。
压缩时先删除旧文件、再以追加模式重放内存队列，全程不出现可变写路径。
"""

from __future__ import annotations

import json
import logging
import re
from collections import deque
from pathlib import Path

from .models import ChatRecord

logger = logging.getLogger(__name__)

_COMPACT_THRESHOLD = 2000  # 追加次数超过该值时触发压缩

_NAME_RE = re.compile(r"^-?\d+\.jsonl$")


def _safe_memory_file(root: Path, filename: str) -> Path:
    """校验并解析存储根目录下的记忆文件名，返回绝对路径。"""
    if not _NAME_RE.fullmatch(filename):
        raise ValueError(f"非法的记忆文件名：{filename!r}")
    path = (root / filename).resolve()
    if path.parent != root:
        raise ValueError(f"记忆文件越出存储根目录：{filename!r}")
    return path


class GroupMemory:
    """单个群的记忆（有界队列 + 对应 JSONL 文件）。"""

    def __init__(self, group_id: int, root: Path, capacity: int):
        self.group_id = group_id
        self.root = root
        self.path = _safe_memory_file(root, f"{int(group_id)}.jsonl")
        self.capacity = max(capacity, 8)
        self._records: deque[ChatRecord] = deque(maxlen=self.capacity)
        self._appends_since_compact = 0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        loaded: list[ChatRecord] = []
        seen: set[int] = set()
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = ChatRecord.from_json(json.loads(line))
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        logger.warning("跳过 %s 中一条损坏的记录", self.path)
                        continue
                    if record.message_id in seen:
                        continue
                    seen.add(record.message_id)
                    loaded.append(record)
        except OSError as exc:
            logger.warning("读取记忆文件失败 %s：%s", self.path, exc)
            return
        for record in loaded[-self.capacity :]:
            self._records.append(record)
        if len(loaded) > len(self._records):
            self._rewrite()  # 启动时顺手把旧文件压到容量以内

    def append(self, record: ChatRecord) -> None:
        self._records.append(record)
        line = json.dumps(record.to_json(), ensure_ascii=False)
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as exc:
            logger.error("写入记忆文件失败 %s：%s", self.path, exc)
            return
        self._appends_since_compact += 1
        if (
            self._records.maxlen is not None
            and len(self._records) == self._records.maxlen
            and self._appends_since_compact >= _COMPACT_THRESHOLD
        ):
            self._rewrite()

    def _rewrite(self) -> None:
        """压缩：删除旧文件后把内存队列原样追加回全新文件（语义等同重写）。"""
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.error("删除旧记忆文件失败 %s：%s", self.path, exc)
            return
        self._appends_since_compact = 0
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                for record in self._records:
                    f.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.error("压缩记忆文件失败 %s：%s", self.path, exc)

    def tail(self, n: int) -> list[ChatRecord]:
        """返回最近 n 条的不可变快照（时间正序）。"""
        if n <= 0 or not self._records:
            return []
        return list(self._records)[-n:]

    def tail_excluding_last(self, n: int) -> list[ChatRecord]:
        """返回除最后一条外的最近 n 条；用于构造提示词历史（当前消息单独走指令层）。"""
        if len(self._records) <= 1 or n <= 0:
            return []
        return list(self._records)[-n - 1 : -1]

    def last(self) -> ChatRecord | None:
        return self._records[-1] if self._records else None

    def find_by_message_id(self, message_id: int) -> ChatRecord | None:
        for record in reversed(self._records):
            if record.message_id == message_id:
                return record
        return None

    def __len__(self) -> int:
        return len(self._records)


class MemoryManager:
    """管理所有群的记忆文件（<data_dir>/memory/<group_id>.jsonl）。"""

    def __init__(self, data_dir: str | Path, default_capacity: int = 64):
        self.root = (Path(data_dir) / "memory").resolve()
        self.default_capacity = default_capacity
        self.root.mkdir(parents=True, exist_ok=True)
        self._groups: dict[int, GroupMemory] = {}

    def get(self, group_id: int) -> GroupMemory:
        gid = int(group_id)
        memory = self._groups.get(gid)
        if memory is None:
            memory = GroupMemory(gid, self.root, self.default_capacity)
            self._groups[gid] = memory
        return memory
