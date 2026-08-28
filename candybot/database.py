"""SQLite 持久层：SQLModel 表定义与异步数据操作（data 目录下的 candy.db）。

三个表：
- chat_history  每条群聊消息（含机器人自己发出的）；文本历史全量保留。
- chat_image    消息内每个图片槽位（展示状态、总结、指向原图的指纹）；
                随消息永久保留，回收图片时只摘除数据引用、降级展示状态。
- image_blob    以内容指纹（SHA-256）为主键的原图 base64；同一张图全库只存
                一份，按保留期回收——没有任何槽位引用时即删除。

展示状态与总结属于历史语义内容，回收后仍然保留（占位/总结随历史照常
送入模型），只有 base64 数据消失；恢复后的记录里对应槽位 images 为空串。
全部数据操作经由 SQLModel/SQLAlchemy 表达式完成，SQL 一律由框架参数
绑定，不存在任何手工拼装的查询语句。
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Sequence

from sqlalchemy import UniqueConstraint, event
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import Field, SQLModel, col, select
from sqlmodel.ext.asyncio.session import AsyncSession as SQLModelAsyncSession

from .models import (
    IMAGE_STATE_PLACEHOLDER,
    IMAGE_STATE_SHOW,
    IMAGE_STATE_SUMMARIZED,
    ChatRecord,
)

logger = logging.getLogger(__name__)


def image_fingerprint(data_url: str) -> str:
    """图片内容指纹：对整条 data URL 取 SHA-256，作 image_blob 主键去重。"""
    return hashlib.sha256(data_url.encode("utf-8")).hexdigest()


class ChatHistoryRow(SQLModel, table=True):
    """chat_history：一条群聊消息。文本与元数据永久保留。"""

    __tablename__ = "chat_history"
    __table_args__ = (UniqueConstraint("group_id", "message_id"),)

    row_id: int | None = Field(default=None, primary_key=True)
    group_id: int = Field(nullable=False)
    message_id: int = Field(nullable=False)  # 机器人自己发言为合成负 id
    user_id: int = Field(nullable=False)
    nickname: str = Field(default="", nullable=False)
    text: str = Field(default="", nullable=False)
    ts: float = Field(nullable=False, index=True)
    is_self: bool = Field(default=False, nullable=False)


class ChatImageRow(SQLModel, table=True):
    """chat_image：一条消息里的一个图片槽位（按 position 对齐）。

    sha256 为 NULL 表示原图已回收；state/summary 是历史语义内容，永久保留。
    """

    __tablename__ = "chat_image"

    id: int | None = Field(default=None, primary_key=True)
    chat_row_id: int = Field(
        foreign_key="chat_history.row_id", nullable=False, index=True
    )
    position: int = Field(default=0, nullable=False)
    state: str = Field(default="show", nullable=False)
    summary: str | None = Field(default=None)
    # 索引：SQLite 不会自动为外键子列建索引，删除 image_blob 行时的
    # 外键校验、以及 GC 的引用集合查询都依赖它
    sha256: str | None = Field(
        default=None, foreign_key="image_blob.sha256", index=True
    )


class ImageBlobRow(SQLModel, table=True):
    """image_blob：原图 base64（按内容指纹去重，全库一份）。"""

    __tablename__ = "image_blob"

    sha256: str = Field(primary_key=True)
    data_url: str = Field(nullable=False)
    created_ts: float = Field(default=0.0, nullable=False)


def _set_sqlite_pragmas(dbapi_conn, _record) -> None:
    """每条新连接启用 WAL、外键级联与忙等待，写并发下更稳。"""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


class CandyDatabase:
    """candy.db 的异步访问入口：建表、按群读写与图片回收。"""

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # URL.create 构造连接串：路径作为参数传入，不经字符串拼接
        self._engine = create_async_engine(
            URL.create(drivername="sqlite+aiosqlite", database=str(self.path))
        )
        event.listens_for(self._engine.sync_engine, "connect")(_set_sqlite_pragmas)
        self._sessions = async_sessionmaker(
            self._engine, class_=SQLModelAsyncSession, expire_on_commit=False
        )

    async def create_tables(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    async def close(self) -> None:
        await self._engine.dispose()

    # ------------------------------------------------------------ 读

    async def load_recent(self, group_id: int, limit: int) -> list[ChatRecord]:
        """某群最近 limit 条记录（时间正序），含图片原图与展示状态。"""
        if limit <= 0:
            return []
        async with self._sessions() as session:
            rows = list(
                (
                    await session.exec(
                        select(ChatHistoryRow)
                        .where(ChatHistoryRow.group_id == group_id)
                        .order_by(col(ChatHistoryRow.row_id).desc())
                        .limit(limit)
                    )
                ).all()
            )[::-1]  # row_id 升序即插入序，与时间正序一致
            slot_map = await self._load_slots(session, [r.row_id for r in rows])
            blob_map: dict[str, str] = {}
            for slots in slot_map.values():
                blob_map.update(await self._load_blobs(session, slots))
            return [
                self._record_from_row(row, slot_map.get(row.row_id, []), blob_map)
                for row in rows
            ]

    async def find_record(self, group_id: int, message_id: int) -> ChatRecord | None:
        """按 (群号, message_id) 精确查一条；不在内存热缓存的历史也可查到。"""
        async with self._sessions() as session:
            row = (
                await session.exec(
                    select(ChatHistoryRow).where(
                        ChatHistoryRow.group_id == group_id,
                        ChatHistoryRow.message_id == message_id,
                    )
                )
            ).first()
            if row is None:
                return None
            slot_map = await self._load_slots(session, [row.row_id])
            slots = slot_map.get(row.row_id, [])
            blobs = await self._load_blobs(session, slots)
            return self._record_from_row(row, slots, blobs)

    # ------------------------------------------------------------ 写

    async def insert_record(self, record: ChatRecord) -> bool:
        """写入一条记录（含图片槽位与原图），重复 message_id 返回 False。"""
        async with self._sessions() as session:
            row = ChatHistoryRow(
                group_id=record.group_id,
                message_id=record.message_id,
                user_id=record.user_id,
                nickname=record.nickname,
                text=record.text,
                ts=record.ts,
                is_self=record.is_self,
            )
            session.add(row)
            try:
                await session.flush()
            except IntegrityError as exc:
                await session.rollback()
                if "unique constraint" not in str(exc.orig).lower():
                    raise  # 非唯一键冲突如实上抛，由调用方决定降级方式
                return False  # (group_id, message_id) 重复
            await self._write_slots(session, row.row_id, record)
            await session.commit()
        return True

    async def replace_image_slots(self, record: ChatRecord) -> None:
        """以内存中的记录为准整体重写其图片槽位（状态/总结/原图引用）。

        召回（recall）时槽位重新挂上原图指纹；内存里有数据而库里没有的
        原图会随之补写进 image_blob，保证两边一致。
        """
        async with self._sessions() as session:
            row = (
                await session.exec(
                    select(ChatHistoryRow).where(
                        ChatHistoryRow.group_id == record.group_id,
                        ChatHistoryRow.message_id == record.message_id,
                    )
                )
            ).first()
            if row is None:  # 已被撤回，无处可写
                return
            await self._delete_slots(session, row.row_id)
            await self._write_slots(session, row.row_id, record)
            await session.commit()

    async def delete_record(self, group_id: int, message_id: int) -> bool:
        """删除一条记录及其槽位，并回收不再被引用的原图；返回是否删除。"""
        async with self._sessions() as session:
            row = (
                await session.exec(
                    select(ChatHistoryRow).where(
                        ChatHistoryRow.group_id == group_id,
                        ChatHistoryRow.message_id == message_id,
                    )
                )
            ).first()
            if row is None:
                return False
            await self._delete_slots(session, row.row_id)
            # 先落库子行删除：未声明 Relationship，UoW 不保证父后于子删除
            await session.flush()
            await session.delete(row)
            await session.commit()
        try:
            await self._gc_blobs()
        except Exception:
            # 行删除已提交；此处回收失败只留孤儿原图，由每日回收兜底
            logger.warning("撤回后原图回收失败（将由每日回收兜底）", exc_info=True)
        return True

    # ------------------------------------------------------------ 图片回收

    async def prune_expired_images(self, retention_days: int) -> tuple[int, int]:
        """回收超过保留期的原图：槽位降级（保留总结），无引用的原图删除。

        返回（降级槽位数, 释放原图数）。
        """
        cutoff = time.time() - max(int(retention_days), 1) * 86400
        async with self._sessions() as session:
            expired = list(
                (
                    await session.exec(
                        select(ChatImageRow).where(
                            col(ChatImageRow.sha256).is_not(None),
                            col(ChatImageRow.chat_row_id).in_(
                    select(ChatHistoryRow.row_id).where(
                        col(ChatHistoryRow.ts) < cutoff,
                        # ts=0（事件缺 time 字段）视为时间未知，不参与回收
                        col(ChatHistoryRow.ts) > 0,
                    )
                            ),
                        )
                    )
                ).all()
            )
            for slot in expired:
                slot.sha256 = None
                slot.state = (
                    IMAGE_STATE_SUMMARIZED if slot.summary else IMAGE_STATE_PLACEHOLDER
                )
                session.add(slot)
            await session.commit()
            freed = await self._gc_blobs(session)
        if expired or freed:
            logger.info(
                "图片回收：降级 %d 个槽位，释放 %d 份原图（保留期 %d 天）",
                len(expired),
                freed,
                retention_days,
            )
        return len(expired), freed

    async def _gc_blobs(self, session: SQLModelAsyncSession | None = None) -> int:
        """删除不再被任何槽位引用的原图。可复用调用方已打开的会话。

        先只取指纹集合做比对（绝不把全表 base64 载入内存），再逐个按
        主键取行删除；同一时刻至多一张原图进入内存。删除外键父行由
        chat_image.sha256 上的索引支撑，并发插入若刚重建了同一指纹的
        引用，外键约束会让删除失败，由调用方容错。
        """
        if session is None:
            async with self._sessions() as owned:
                return await self._gc_blobs(owned)
        referenced = set(
            (
                await session.exec(
                    select(ChatImageRow.sha256).where(
                        col(ChatImageRow.sha256).is_not(None)
                    )
                )
            ).all()
        )
        stale_shas = [
            sha
            for sha in (await session.exec(select(ImageBlobRow.sha256))).all()
            if sha not in referenced
        ]
        freed = 0
        for sha in stale_shas:
            blob = await session.get(ImageBlobRow, sha)
            if blob is not None:  # 并发写入可能刚重建了引用
                await session.delete(blob)
                freed += 1
        if freed:
            await session.commit()
        return freed

    # ------------------------------------------------------------ 内部工具

    async def _delete_slots(
        self, session: SQLModelAsyncSession, chat_row_id: int
    ) -> None:
        """删除一条记录的全部槽位实例（ORM 级联语义之外的显式清理）。"""
        for slot in (
            await session.exec(
                select(ChatImageRow).where(ChatImageRow.chat_row_id == chat_row_id)
            )
        ).all():
            await session.delete(slot)

    async def _write_slots(
        self, session: SQLModelAsyncSession, chat_row_id: int, record: ChatRecord
    ) -> None:
        """按内存记录写图片槽位；原图指纹不存在时补写 image_blob。"""
        for index, data_url in enumerate(record.images):
            sha = image_fingerprint(data_url) if data_url else None
            state = record.state_of(index)
            if sha is None and state == IMAGE_STATE_SHOW:
                # 防御：show 但原图数据已不在（被保留期回收），降级后落库，
                # 保证「show 槽位必有原图」这一不变量在库内成立
                state = (
                    IMAGE_STATE_SUMMARIZED
                    if record.summary_of(index)
                    else IMAGE_STATE_PLACEHOLDER
                )
            if sha is not None and await session.get(ImageBlobRow, sha) is None:
                session.add(
                    ImageBlobRow(sha256=sha, data_url=data_url, created_ts=record.ts)
                )
                # 原图先落库：槽位行的外键引用它，不能等 autoflush 决定次序
                await session.flush()
            session.add(
                ChatImageRow(
                    chat_row_id=chat_row_id,
                    position=index,
                    state=state,
                    summary=record.summary_of(index),
                    sha256=sha,
                )
            )

    async def _load_slots(
        self, session: SQLModelAsyncSession, row_ids: Sequence[int | None]
    ) -> dict[int, list[ChatImageRow]]:
        ids = [rid for rid in row_ids if rid is not None]
        if not ids:
            return {}
        rows = (
            await session.exec(
                select(ChatImageRow).where(col(ChatImageRow.chat_row_id).in_(ids))
            )
        ).all()
        out: dict[int, list[ChatImageRow]] = {}
        for row in rows:
            out.setdefault(row.chat_row_id, []).append(row)
        for slots in out.values():
            slots.sort(key=lambda s: s.position)
        return out

    async def _load_blobs(
        self, session: SQLModelAsyncSession, slots: list[ChatImageRow]
    ) -> dict[str, str]:
        shas = [s.sha256 for s in slots if s.sha256]
        if not shas:
            return {}
        rows = (
            await session.exec(
                select(ImageBlobRow).where(col(ImageBlobRow.sha256).in_(shas))
            )
        ).all()
        return {r.sha256: r.data_url for r in rows}

    def _record_from_row(
        self,
        row: ChatHistoryRow,
        slots: list[ChatImageRow],
        blobs: dict[str, str],
    ) -> ChatRecord:
        """行 + 槽位 → 领域记录。原图缺失的槽位写空串并降级展示状态。"""
        images: list[str] = []
        states: list[str] = []
        summaries: dict[int, str] = {}
        for slot in slots:
            data = blobs.get(slot.sha256, "") if slot.sha256 else ""
            state = slot.state
            if not data:
                # 原图已回收（或丢失）：降级为总结/占位符，绝不再以 show 出现
                state = (
                    IMAGE_STATE_SUMMARIZED if slot.summary else IMAGE_STATE_PLACEHOLDER
                )
            images.append(data)
            states.append(state)
            if slot.summary:
                summaries[len(images) - 1] = slot.summary
        return ChatRecord(
            message_id=row.message_id,
            group_id=row.group_id,
            user_id=row.user_id,
            nickname=row.nickname,
            text=row.text,
            ts=row.ts,
            is_self=row.is_self,
            images=tuple(images),
            image_states=tuple(states),
            image_summaries=summaries or None,
        )
