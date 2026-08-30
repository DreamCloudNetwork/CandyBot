"""SQLite 持久层：SQLModel 表定义与异步数据操作（data 目录下的 candy.db）。

八个表：
- chat_history     每条群聊消息（含机器人自己发出的）；文本历史全量保留；
                   is_command 标记命令插件产生的消息（补列与存量回填见
                   migrations.py 的启动迁移）。
- chat_image       消息内每个图片槽位（展示状态、总结、指向原图的指纹）；
                   随消息永久保留，回收图片时只摘除数据引用、降级展示状态。
- image_blob       以内容指纹（SHA-256）为主键的原图 base64；同一张图全库
                   只存一份，按保留期回收——没有任何槽位引用时即删除。
- group_impression 每日群聊印象（中期记忆）：每群每天一条 ≤300 字总结，
                   最近 N 天注入提示词 L2，更旧的按保留期清理。
- expressions      表达学习成果：「当 situation 时，可以用 style」规律，
                   同群按内容去重，count 作加权随机抽取的权重。
- jargons          黑话词条：term + meaning，同群去重，超出条目上限时
                   淘汰最久未命中的条目。
- sticker          表情包收藏（最小版）：同群按 (群, 内容指纹) 去重，
                   全局数量超上限时替换最久未使用的条目——本表只删记录，
                   图片文件的写入与删除由 stickers.StickerStore 负责。
- schema_migration 已执行的迁移名单（见 migrations.py）：每个迁移按名字
                   只跑一次，启动建表后自动补跑未执行的迁移。

展示状态与总结属于历史语义内容，回收后仍然保留（占位/总结随历史照常
送入模型），只有 base64 数据消失；恢复后的记录里对应槽位 images 为空串。
全部数据操作经由 SQLModel/SQLAlchemy 表达式完成，SQL 一律由框架参数
绑定，不存在任何手工拼装的查询语句。
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
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
    # 命令插件产生的消息（命令消息与命令回复）标记；旧库无此列、存量
    # 记录的回填标记由 migrations.py 的启动迁移完成
    is_command: bool = Field(default=False, nullable=False)


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


class GroupImpressionRow(SQLModel, table=True):
    """group_impression：每日「今日群聊印象」（中期记忆）。

    day 为 YYYY-MM-DD 字符串（字典序即时间序），每群每天一条；
    注入提示词 L2 时取最近 N 天，天内字节级不变。
    """

    __tablename__ = "group_impression"
    __table_args__ = (UniqueConstraint("group_id", "day"),)

    id: int | None = Field(default=None, primary_key=True)
    group_id: int = Field(nullable=False)
    day: str = Field(nullable=False)
    summary: str = Field(default="", nullable=False)
    created_ts: float = Field(default=0.0, nullable=False)


class ExpressionRow(SQLModel, table=True):
    """expressions：表达学习成果「当 situation 时，可以用 style」。

    同群按 (group_id, situation, style) 内容去重，重复学到时只累计
    count（即加权随机抽取的权重）；last_active_time 记录最近一次被
    选中注入回复的时刻。
    """

    __tablename__ = "expressions"
    __table_args__ = (UniqueConstraint("group_id", "situation", "style"),)

    id: int | None = Field(default=None, primary_key=True)
    group_id: int = Field(nullable=False)
    situation: str = Field(nullable=False)
    style: str = Field(nullable=False)
    count: int = Field(default=1, nullable=False)
    created_ts: float = Field(default=0.0, nullable=False)
    last_active_time: float = Field(default=0.0, nullable=False)


class JargonRow(SQLModel, table=True):
    """jargons：群内黑话词条（双路含义推断一致才入库）。

    同群按 (group_id, term) 去重；last_hit_time 是条目超上限时
    「淘汰最久未命中」的排序键。
    """

    __tablename__ = "jargons"
    __table_args__ = (UniqueConstraint("group_id", "term"),)

    id: int | None = Field(default=None, primary_key=True)
    group_id: int = Field(nullable=False)
    term: str = Field(nullable=False)
    meaning: str = Field(default="", nullable=False)
    count: int = Field(default=1, nullable=False)
    created_ts: float = Field(default=0.0, nullable=False)
    last_hit_time: float = Field(default=0.0, nullable=False)


class StickerRow(SQLModel, table=True):
    """sticker：表情包收藏（最小版）。

    同群按 (group_id, sha256) 去重（sha256 为整条 data URL 的内容指纹，
    与 image_blob 同一算法）；path 为相对表情包根目录的文件路径，
    文件实体由 stickers.StickerStore 读写。(last_used_time, created_ts)
    是全局超上限时「替换最久未使用」的排序键。
    """

    __tablename__ = "sticker"
    __table_args__ = (UniqueConstraint("group_id", "sha256"),)

    id: int | None = Field(default=None, primary_key=True)
    group_id: int = Field(nullable=False, index=True)
    sha256: str = Field(default="", nullable=False)
    path: str = Field(default="", nullable=False)
    summary: str = Field(default="", nullable=False)
    use_count: int = Field(default=0, nullable=False)
    created_ts: float = Field(default=0.0, nullable=False)
    last_used_time: float = Field(default=0.0, nullable=False)


@dataclass(frozen=True)
class ImpressionEntry:
    """一条群印象（day 为 YYYY-MM-DD）。"""

    day: str
    summary: str


@dataclass(frozen=True)
class ExpressionEntry:
    """一条候选表达。weight 即学习到的次数 count。"""

    id: int
    situation: str
    style: str
    weight: int
    last_active_time: float


@dataclass(frozen=True)
class JargonEntry:
    """一条黑话词条。"""

    id: int
    term: str
    meaning: str
    count: int
    last_hit_time: float


@dataclass(frozen=True)
class StickerEntry:
    """一条表情包收藏。path 为相对表情包根目录的文件路径。"""

    id: int
    group_id: int
    sha256: str
    path: str
    summary: str
    use_count: int


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

    @property
    def engine(self):
        """只读暴露给独立迁移模块（candybot/migrations.py）使用。"""
        return self._engine

    @property
    def sessions(self):
        """同 engine：迁移模块补列与回填标记需要直接操作会话。"""
        return self._sessions

    async def create_tables(self) -> None:
        """建出全部缺失的表（含 schema_migration，幂等）。

        注意 create_all 只建新表、不会 ALTER 已有表——给旧表补列与存量
        数据回填由 candybot/migrations.py 在启动建表后统一执行。
        """
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
                is_command=record.is_command,
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

    # ------------------------------------------------------------ 中期记忆与学习

    async def list_group_ids(self) -> list[int]:
        """库里出现过的全部群号（每日印象任务据此枚举白名单内的群）。"""
        async with self._sessions() as session:
            return list(
                (await session.exec(select(col(ChatHistoryRow.group_id)).distinct())).all()
            )

    async def load_day_records(
        self, group_id: int, start_ts: float, end_ts: float
    ) -> list[ChatRecord]:
        """某群 [start_ts, end_ts) 内的全部消息（时间正序，不含图片数据）。"""
        async with self._sessions() as session:
            rows = (
                await session.exec(
                    select(ChatHistoryRow)
                    .where(
                        ChatHistoryRow.group_id == group_id,
                        ChatHistoryRow.ts >= start_ts,
                        ChatHistoryRow.ts < end_ts,
                    )
                    .order_by(col(ChatHistoryRow.row_id))
                )
            ).all()
            return [self._record_from_row(row, [], {}) for row in rows]

    async def has_day_records(
        self, group_id: int, start_ts: float, end_ts: float
    ) -> bool:
        """某群在 [start_ts, end_ts) 内是否有任何消息记录（廉价存在性查询）。

        供 L2 印象快照的零点竞态防护使用：昨日有聊天却还没有印象，说明
        每日印象任务还没轮到/还没跑完该群，快照暂不完整、不宜固化。
        """
        async with self._sessions() as session:
            row = (
                await session.exec(
                    select(ChatHistoryRow.row_id)
                    .where(
                        ChatHistoryRow.group_id == group_id,
                        ChatHistoryRow.ts >= start_ts,
                        ChatHistoryRow.ts < end_ts,
                    )
                    .limit(1)
                )
            ).first()
            return row is not None

    async def save_impression(
        self, group_id: int, day: str, summary: str, ts: float | None = None
    ) -> None:
        """写入/覆盖某群某天的印象（定时任务重跑幂等）。"""
        async with self._sessions() as session:
            row = (
                await session.exec(
                    select(GroupImpressionRow).where(
                        GroupImpressionRow.group_id == group_id,
                        GroupImpressionRow.day == day,
                    )
                )
            ).first()
            if row is None:
                row = GroupImpressionRow(group_id=group_id, day=day)
            row.summary = summary
            row.created_ts = ts if ts is not None else time.time()
            session.add(row)
            await session.commit()

    async def load_impressions(self, group_id: int, limit: int) -> list[ImpressionEntry]:
        """某群最近 limit 天的印象，按 day 升序（旧→新）返回。"""
        if limit <= 0:
            return []
        async with self._sessions() as session:
            rows = list(
                (
                    await session.exec(
                        select(GroupImpressionRow)
                        .where(GroupImpressionRow.group_id == group_id)
                        .order_by(col(GroupImpressionRow.day).desc())
                        .limit(limit)
                    )
                ).all()
            )[::-1]
            return [ImpressionEntry(day=r.day, summary=r.summary) for r in rows]

    async def has_impression(self, group_id: int, day: str) -> bool:
        async with self._sessions() as session:
            row = (
                await session.exec(
                    select(GroupImpressionRow.id).where(
                        GroupImpressionRow.group_id == group_id,
                        GroupImpressionRow.day == day,
                    )
                )
            ).first()
            return row is not None

    async def prune_impressions(self, before_day: str) -> int:
        """删除 day 早于 before_day 的印象（day 是 ISO 格式，字典序即时间序）。"""
        async with self._sessions() as session:
            rows = (
                await session.exec(
                    select(GroupImpressionRow).where(
                        col(GroupImpressionRow.day) < before_day
                    )
                )
            ).all()
            for row in rows:
                await session.delete(row)
            await session.commit()
            return len(rows)

    async def load_expressions(self, group_id: int) -> list[ExpressionEntry]:
        async with self._sessions() as session:
            rows = (
                await session.exec(
                    select(ExpressionRow).where(ExpressionRow.group_id == group_id)
                )
            ).all()
            return [
                ExpressionEntry(
                    id=row.id,
                    situation=row.situation,
                    style=row.style,
                    weight=max(row.count, 1),
                    last_active_time=row.last_active_time,
                )
                for row in rows
            ]

    async def record_expression(
        self, group_id: int, situation: str, style: str, ts: float
    ) -> None:
        """记录学到的一条表达；同群内容完全一致时只累计权重（去重）。"""
        async with self._sessions() as session:
            row = (
                await session.exec(
                    select(ExpressionRow).where(
                        ExpressionRow.group_id == group_id,
                        ExpressionRow.situation == situation,
                        ExpressionRow.style == style,
                    )
                )
            ).first()
            if row is None:
                session.add(
                    ExpressionRow(
                        group_id=group_id,
                        situation=situation,
                        style=style,
                        count=1,
                        created_ts=ts,
                        last_active_time=0.0,
                    )
                )
            else:
                row.count += 1
                session.add(row)
            await session.commit()

    async def touch_expressions(self, ids: Sequence[int], ts: float) -> None:
        """更新被选中表达条目的最近使用时间。"""
        if not ids:
            return
        async with self._sessions() as session:
            for row_id in ids:
                row = await session.get(ExpressionRow, row_id)
                if row is not None:
                    row.last_active_time = ts
                    session.add(row)
            await session.commit()

    async def load_jargons(self, group_id: int) -> list[JargonEntry]:
        async with self._sessions() as session:
            rows = (
                await session.exec(
                    select(JargonRow).where(JargonRow.group_id == group_id)
                )
            ).all()
            return [
                JargonEntry(
                    id=row.id,
                    term=row.term,
                    meaning=row.meaning,
                    count=row.count,
                    last_hit_time=row.last_hit_time,
                )
                for row in rows
            ]

    async def record_jargon(
        self,
        group_id: int,
        term: str,
        meaning: str,
        ts: float,
        max_entries: int,
    ) -> None:
        """写入一个黑话词条；同群已有词条只更新含义，新词插入后按上限
        淘汰最久未命中的条目。"""
        async with self._sessions() as session:
            row = (
                await session.exec(
                    select(JargonRow).where(
                        JargonRow.group_id == group_id, JargonRow.term == term
                    )
                )
            ).first()
            if row is None:
                session.add(
                    JargonRow(
                        group_id=group_id,
                        term=term,
                        meaning=meaning,
                        count=1,
                        created_ts=ts,
                        last_hit_time=ts,
                    )
                )
                await session.commit()
            else:
                row.meaning = meaning
                row.count += 1
                session.add(row)
                await session.commit()
            if max_entries > 0:
                await self._evict_jargons(session, group_id, max_entries)

    async def _evict_jargons(
        self, session: SQLModelAsyncSession, group_id: int, max_entries: int
    ) -> None:
        """某群黑话超出上限时淘汰最久未命中的条目。"""
        rows = list(
            (
                await session.exec(
                    select(JargonRow).where(JargonRow.group_id == group_id)
                )
            ).all()
        )
        overflow = len(rows) - max_entries
        if overflow <= 0:
            return
        rows.sort(key=lambda r: (r.last_hit_time, r.created_ts))
        for row in rows[:overflow]:
            await session.delete(row)
        await session.commit()

    async def touch_jargons(self, ids: Sequence[int], ts: float) -> None:
        """更新被机械匹配命中的黑话条目的最近命中时间。"""
        if not ids:
            return
        async with self._sessions() as session:
            for row_id in ids:
                row = await session.get(JargonRow, row_id)
                if row is not None:
                    row.last_hit_time = ts
                    session.add(row)
            await session.commit()

    # ------------------------------------------------------------ 表情包收藏

    async def insert_sticker(
        self, group_id: int, sha256: str, path: str, summary: str, ts: float
    ) -> bool:
        """新增一条表情包收藏；同群已有相同内容指纹时返回 False（去重）。

        last_used_time 以入库时间起步：刚收藏的图绝不因为「还没用过」
        被新来的收藏立刻挤掉，LRU 淘汰按「用过或收来」的最久时间排序。
        """
        async with self._sessions() as session:
            row = StickerRow(
                group_id=group_id,
                sha256=sha256,
                path=path,
                summary=summary,
                created_ts=ts,
                last_used_time=ts,
            )
            session.add(row)
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                if "unique constraint" not in str(exc.orig).lower():
                    raise  # 非唯一键冲突如实上抛
                return False
            return True

    async def load_stickers(self, group_id: int) -> list[StickerEntry]:
        """某群收藏的全部表情包（顺序无所谓：抽发是随机的）。"""
        async with self._sessions() as session:
            rows = (
                await session.exec(
                    select(StickerRow).where(StickerRow.group_id == group_id)
                )
            ).all()
            return [
                StickerEntry(
                    id=row.id,
                    group_id=row.group_id,
                    sha256=row.sha256,
                    path=row.path,
                    summary=row.summary,
                    use_count=row.use_count,
                )
                for row in rows
            ]

    async def touch_sticker(self, sticker_id: int, ts: float) -> None:
        """记录一次使用：使用数 +1、刷新最近使用时间（LRU 的排序键）。"""
        async with self._sessions() as session:
            row = await session.get(StickerRow, sticker_id)
            if row is None:  # 并发替换刚把它删了：本次使用统计作废
                return
            row.use_count += 1
            row.last_used_time = ts
            session.add(row)
            await session.commit()

    async def evict_stickers_over(self, max_total: int) -> list[str]:
        """全局收藏超过 max_total 时按「最久未使用」删到上限，
        返回被删条目的相对文件路径（图片文件由调用方删除）。

        排序键 (last_used_time, created_ts)；last_used_time 以入库时间
        起步（见 insert_sticker），因此先淘汰「最久既没用过也没收进来」
        的条目。
        """
        async with self._sessions() as session:
            rows = list(
                (
                    await session.exec(
                        select(StickerRow)
                        .order_by(col(StickerRow.last_used_time), col(StickerRow.created_ts))
                    )
                ).all()
            )
            overflow = len(rows) - max_total
            if overflow <= 0:
                return []
            removed_paths: list[str] = []
            for row in rows[:overflow]:
                removed_paths.append(row.path)
                await session.delete(row)
            await session.commit()
            return removed_paths

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
            is_command=bool(row.is_command),
            images=tuple(images),
            image_states=tuple(states),
            image_summaries=summaries or None,
        )
