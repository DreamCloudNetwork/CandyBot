"""每群上下文记忆：内存 deque 热缓存 + SQLite（data/candy.db）持久层。

文本历史在库里全量保留，永不清裁；图片原图按保留期
（storage.image_retention_days，默认 7 天）回收，槽位降级为总结/占位符。
deque 只保存最近 capacity 条作为提示词热缓存，重启后从库里回放；
tail/last 等读操作全部命中内存，写操作同步落库。
跨群的图片以内容指纹（SHA-256）在 image_blob 表里全库去重，同一张图
只存一份 base64；撤回/召回时槽位与库同步，引用计数负责原图的生死。
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

from .database import CandyDatabase
from .models import (
    IMAGE_STATE_PLACEHOLDER,
    IMAGE_STATE_SHOW,
    IMAGE_STATE_SUMMARIZED,
    ChatRecord,
)

logger = logging.getLogger(__name__)


class GroupMemory:
    """单个群的记忆（有界内存队列 + candy.db 持久化）。"""

    def __init__(
        self,
        group_id: int,
        db: CandyDatabase,
        capacity: int,
        on_evict: Callable[[int, ChatRecord], None] | None = None,
    ):
        self.group_id = group_id
        self.capacity = max(capacity, 8)
        self._db = db
        self._records: deque[ChatRecord] = deque(maxlen=self.capacity)
        # 热缓存溢出回调：被 deque 静默丢弃的最旧消息交给监听者
        # （学习任务据此积累「被淘汰消息」）。必须同步且 O(1) 廉价，
        # 在 append 锁内调用，绝不能在这里 await 或做重活。
        self._on_evict = on_evict
        # 串行化同群的「入库 + 进缓存」，保证 deque 顺序与库内 row_id
        # 顺序一致（insert 的 flush 与 commit 之间存在 await，无锁时
        # 两条并发消息可能顺序互换，重启回放后翻转）
        self._append_lock = asyncio.Lock()

    def _prime(self, records: list[ChatRecord]) -> None:
        """启动回放：把库里最近的历史装进热缓存（仅由 MemoryManager 调用）。

        回放导致的挤出属于装载而非实时淘汰，不触发 on_evict。
        """
        self._records.extend(records)

    def _push(self, record: ChatRecord) -> None:
        """进热缓存；溢出前捕获即将被挤出的最旧消息并交给监听者。"""
        evicted: ChatRecord | None = None
        if self._on_evict is not None and len(self._records) == self._records.maxlen:
            evicted = self._records[0]
        self._records.append(record)
        if evicted is not None:
            self._on_evict(self.group_id, evicted)

    async def append(self, record: ChatRecord) -> None:
        """入库并进热缓存。库写入失败时退化为仅热缓存（与旧版落盘失败
        同语义），不让单条存储故障中断消息处理。"""
        async with self._append_lock:
            try:
                inserted = await self._db.insert_record(record)
            except Exception:
                logger.warning(
                    "群 %d 消息 %d 入库失败，仅保留在内存热缓存",
                    self.group_id,
                    record.message_id,
                    exc_info=True,
                )
                self._push(record)
                return
            if not inserted:
                logger.debug(
                    "群 %d 消息 %d 重复入库，跳过", self.group_id, record.message_id
                )
                return
            self._push(record)

    async def remove(self, message_id: int) -> bool:
        """按 message_id 删除记录（内存与库同步），返回是否真的删了。

        全量历史都在库里，即使消息早于本次启动（不在热缓存中）也能删；
        入库失败而仅存于热缓存的记录，删热缓存也算真的删了。
        同一把 append 锁：撤回与进行中的 append 串行，避免刚入库的记录
        在进缓存前被重建的 deque 漏掉而残留为幽灵。
        """
        async with self._append_lock:
            deleted = await self._db.delete_record(self.group_id, message_id)
            remaining = tuple(r for r in self._records if r.message_id != message_id)
            removed_from_cache = len(remaining) < len(self._records)
            if removed_from_cache:
                self._records = deque(remaining, maxlen=self.capacity)
        return deleted or removed_from_cache

    def tail(self, n: int) -> list[ChatRecord]:
        """返回最近 n 条的不可变快照（时间正序）。"""
        if n <= 0 or not self._records:
            return []
        return list(self._records)[-n:]

    def model_tail(
        self, n: int, *, include_commands: bool = True
    ) -> list[ChatRecord]:
        """送入模型历史层的最近 n 条快照（时间正序）。

        include_commands=False（plugins.include_commands_in_history 关）时
        跳过命令插件产生的消息（ChatRecord.is_command：用户的命令消息与
        机器人的命令回复）——它们照常入库，供审计与印象统计，只是不进
        模型上下文。过滤先于截取：排除的命令不占用 context_size 名额。
        """
        if n <= 0:
            return []
        records = list(self._records)
        if not include_commands:
            records = [r for r in records if not r.is_command]
        return records[-n:]

    def tail_excluding_last(self, n: int) -> list[ChatRecord]:
        """返回除最后一条外的最近 n 条；用于构造提示词历史（当前消息单独走指令层）。"""
        if len(self._records) <= 1 or n <= 0:
            return []
        return list(self._records)[-n - 1 : -1]

    def last(self) -> ChatRecord | None:
        return self._records[-1] if self._records else None

    async def find_by_message_id(self, message_id: int) -> ChatRecord | None:
        """先查热缓存，未命中再查库（引用/撤回早于启动的消息仍可找到）。"""
        for record in reversed(self._records):
            if record.message_id == message_id:
                return record
        return await self._db.find_record(self.group_id, message_id)

    async def transition_images(self, message_id: int, direction: str) -> bool:
        """按模型标记整体切换一条消息里全部图片的展示形态。

        direction="drop"：把仍在展示原图（show）的图片降级为总结，
        没有总结的退化为占位符——「总结（或替换为占位符）并不再展示原图」；
        direction="recall"：把该消息任意形态的图片重新升回原图展示。
        返回是否有实际变更；有则把槽位整体同步回库，保证重启后状态不回退。
        """
        # 只查热缓存即可：模型只能引用 tail(context_size) 内出现过的
        # message_id，而缓存容量 ≥ 2×context_size 保证其必然还在缓存中
        record = None
        for candidate in reversed(self._records):
            if candidate.message_id == message_id:
                record = candidate
                break
        if record is None or not record.images:
            return False
        changed = False
        for index in range(len(record.images)):
            current = record.state_of(index)
            if direction == "recall":
                if not record.images[index]:
                    continue  # 原图已按保留期回收，数据不在，无可召回
                new_state = IMAGE_STATE_SHOW
            elif direction == "drop":
                if current != IMAGE_STATE_SHOW:
                    continue
                new_state = (
                    IMAGE_STATE_SUMMARIZED
                    if record.summary_of(index)
                    else IMAGE_STATE_PLACEHOLDER
                )
            else:
                raise ValueError(f"未知的图片状态切换方向：{direction!r}")
            if new_state != current:
                record.set_image_state(index, new_state)
                changed = True
        if changed:
            await self._db.replace_image_slots(record)
        return changed

    def __len__(self) -> int:
        return len(self._records)


def seconds_until_next_midnight() -> float:
    now = datetime.now()
    next_midnight = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return max((next_midnight - now).total_seconds(), 1.0)


class MemoryManager:
    """管理 candy.db 持久层与所有群的内存记忆。

    首次使用时惰性建表；start() 后每日零点定时回收过期原图，
    启动时也会先回收一次。
    """

    def __init__(
        self,
        data_dir: str | Path,
        default_capacity: int = 64,
        image_retention_days: int = 7,
    ):
        self.default_capacity = default_capacity
        self.image_retention_days = max(int(image_retention_days), 1)
        self.db = CandyDatabase(Path(data_dir) / "candy.db")
        self._groups: dict[int, GroupMemory] = {}
        self._lock = asyncio.Lock()
        self._ready = False
        self._prune_task: asyncio.Task[None] | None = None
        # 热缓存淘汰监听（签名 (group_id, 被淘汰记录)）：由 bot 接线到学习
        # 服务；设置后对新创建的群记忆生效。必须同步且 O(1) 廉价。
        self.evict_listener: Callable[[int, ChatRecord], None] | None = None

    async def get(self, group_id: int) -> GroupMemory:
        """取某群记忆；首次访问时从库回放最近历史（并发安全）。"""
        gid = int(group_id)
        memory = self._groups.get(gid)
        if memory is not None:
            return memory
        await self._ensure_ready()
        async with self._lock:
            memory = self._groups.get(gid)
            if memory is None:
                memory = GroupMemory(
                    gid, self.db, self.default_capacity, on_evict=self.evict_listener
                )
                memory._prime(await self.db.load_recent(gid, memory.capacity))
                self._groups[gid] = memory
            return memory

    async def start(self) -> None:
        """建表并启动每日图片回收循环（幂等，可重复调用）。"""
        await self._ensure_ready()
        if self._prune_task is None or self._prune_task.done():
            self._prune_task = asyncio.create_task(
                self._prune_loop(), name="memory-image-prune"
            )

    async def prune_expired_images(self) -> tuple[int, int]:
        """按保留期回收一次过期原图；失败仅记日志，不影响主流程。"""
        try:
            return await self.db.prune_expired_images(self.image_retention_days)
        except Exception:
            logger.exception("图片回收失败")
            return (0, 0)

    async def _prune_loop(self) -> None:
        await self.prune_expired_images()  # 启动即回收一次
        while True:
            await asyncio.sleep(seconds_until_next_midnight())
            await self.prune_expired_images()

    async def close(self) -> None:
        """停掉回收循环并释放连接池。"""
        if self._prune_task is not None:
            self._prune_task.cancel()
            try:
                await self._prune_task
            except asyncio.CancelledError:
                pass
            self._prune_task = None
        await self.db.close()

    async def _ensure_ready(self) -> None:
        if self._ready:
            return
        async with self._lock:
            if not self._ready:
                await self.db.create_tables()
                self._ready = True
