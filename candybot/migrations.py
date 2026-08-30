"""candy.db 的独立迁移：schema 补列与存量数据回填（仅手动执行）。

MemoryManager 启动时只负责建表（create_all 只建新表、不会 ALTER 已存在
的表）；给旧表补列、对存量行回填标记一律集中在本模块，且不在启动时自动
执行——CandyBot.start 只做兼容检查（pending_migrations），发现库结构
落后（如旧库缺 is_command 列，消息入库会直接报错）就拒绝启动并提示。
升级流程：

    1. 停止机器人；
    2. python -m candybot.migrations [candy.db 路径] [--dry-run 先预览]；
    3. 重新启动机器人。

不传路径时取 config.json5 的 bot.data_dir；--dry-run 一律不写库：不建
表、不补列、不更新、不记名单（缺表按「从未迁移」处理）。每个迁移按
名字记入 schema_migration 表、只执行一次（幂等）。

新增迁移：定义一个名字常量与 _m_xxx 协程，在 run_migrations 里按
「未执行过才跑、跑完记录名单」的模式挂上，并把对应的 schema 缺口加进
pending_migrations（启动检查据此拒绝落后结构）。数据查询与更新一律走
SQLModel/SQLAlchemy 表达式；仅 DDL（ALTER TABLE 补列）允许手写，且必须
是全字面量 SQL，不含任何外部输入。

存量命令消息的打标规则（与 bot._detect_command 的拦截语义对齐）：
- 他人消息：命令名命中注册表（/help 等内置命令同样在册）即为命令消息；
- 自己消息：紧跟命令消息、直到下一条他人消息为止的连续自发言视为
  命令回复。插件在册时命令消息不会走大模型，该区间内的自发言通常
  就是命令输出；上一轮连发的尾巴恰好落进区间是回填的可接受误差。
命令名集合来自运行时的插件注册表（与判定同源）：plugins.enabled=false
或注册表为空时不做回填。升级前的旧记录只有按这套规则能对上号的才会
带标记。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

from sqlalchemy import inspect as sa_inspect
from sqlmodel import Field, SQLModel, col, select, update

from .commandline import detect_command_name
from .database import CandyDatabase, ChatHistoryRow

logger = logging.getLogger(__name__)

# 迁移一：chat_history 补 is_command 列 + 按注册表回填存量标记
M_IS_COMMAND = "chat_history_is_command"

_UPDATE_CHUNK = 500  # 批量打标每批行数


class SchemaMigrationRow(SQLModel, table=True):
    """schema_migration：已执行迁移的名字与时间（字典序无意义，仅存档）。"""

    __tablename__ = "schema_migration"

    name: str = Field(primary_key=True)
    applied_ts: float = Field(default=0.0, nullable=False)


async def run_migrations(
    db: CandyDatabase,
    *,
    command_names: Iterable[str] = (),
    dry_run: bool = False,
) -> list[tuple[str, int]]:
    """执行全部未跑的迁移，返回 [(迁移名, 打标行数)]（已跑过的不出现）。

    command_names：插件注册表里的命令名集合；插件总开关关闭时传空集合，
    按规则不会产生任何标记。dry_run 一律不写库：不建表、不补列、不更新、
    不记录迁移名单（缺表时按「从未迁移过」处理）。
    """
    if not dry_run:
        await db.create_tables()  # 幂等；schema_migration 表也在这里建出
    applied = await _applied_names(db)
    results: list[tuple[str, int]] = []
    if M_IS_COMMAND not in applied:
        tagged = await _m_is_command(db, frozenset(command_names), dry_run)
        results.append((M_IS_COMMAND, tagged))
        logger.info(
            "迁移 %s：%s打标存量命令消息与回复 %d 条",
            M_IS_COMMAND,
            "试运行，" if dry_run else "",
            tagged,
        )
        if not dry_run:
            await _record_applied(db, M_IS_COMMAND)
    return results


async def _table_exists(db: CandyDatabase, name: str) -> bool:
    async with db.engine.begin() as conn:
        return bool(await conn.run_sync(lambda c: sa_inspect(c).has_table(name)))


async def pending_migrations(db: CandyDatabase) -> list[str]:
    """启动兼容检查：返回当前库仍缺失的 schema 能力描述（非迁移名单）。

    CandyBot.start 据此拒绝在落后结构上启动；空列表表示库结构满足当前
    模型定义。注意它看的是实际结构而非迁移名单：全新库从没跑过迁移，
    但建表自带全部新列，同样通过检查。
    """
    gaps: list[str] = []
    if await _table_exists(db, "chat_history"):
        async with db.engine.begin() as conn:
            columns = await conn.run_sync(_chat_history_columns)
        if "is_command" not in columns:
            gaps.append("chat_history.is_command（迁移 chat_history_is_command）")
    return gaps


async def _applied_names(db: CandyDatabase) -> set[str]:
    if not await _table_exists(db, "schema_migration"):
        return set()  # 还没建过表（dry_run 路径）：视为从未迁移
    async with db.sessions() as session:
        rows = (await session.exec(select(SchemaMigrationRow))).all()
    return {row.name for row in rows}


async def _record_applied(db: CandyDatabase, name: str) -> None:
    async with db.sessions() as session:
        session.add(SchemaMigrationRow(name=name, applied_ts=time.time()))
        await session.commit()


async def _m_is_command(
    db: CandyDatabase, command_names: frozenset[str], dry_run: bool
) -> int:
    if not await _table_exists(db, "chat_history"):
        logger.warning("chat_history 表不存在（空库？），跳过回填")
        return 0
    await _ensure_is_command_column(db, dry_run)
    async with db.sessions() as session:
        # 只读规则需要的列：不碰 is_command，dry_run 未补列时也能跑
        rows = (
            await session.exec(
                select(
                    ChatHistoryRow.row_id,
                    ChatHistoryRow.group_id,
                    ChatHistoryRow.is_self,
                    ChatHistoryRow.text,
                ).order_by(ChatHistoryRow.group_id, ChatHistoryRow.row_id)
            )
        ).all()
    row_ids = command_row_ids(rows, command_names)
    if row_ids and not dry_run:
        await _mark_rows(db, row_ids)
    return len(row_ids)


def _chat_history_columns(sync_conn) -> set[str]:
    return {c["name"] for c in sa_inspect(sync_conn).get_columns("chat_history")}


async def _ensure_is_command_column(db: CandyDatabase, dry_run: bool) -> None:
    async with db.engine.begin() as conn:
        columns = await conn.run_sync(_chat_history_columns)
        if "is_command" in columns or dry_run:
            return
        await conn.exec_driver_sql(
            "ALTER TABLE chat_history ADD COLUMN is_command BOOLEAN NOT NULL DEFAULT 0"
        )


async def _mark_rows(db: CandyDatabase, row_ids: Sequence[int]) -> None:
    async with db.sessions() as session:
        for start in range(0, len(row_ids), _UPDATE_CHUNK):
            await session.exec(
                update(ChatHistoryRow)
                .where(
                    col(ChatHistoryRow.row_id).in_(
                        list(row_ids[start : start + _UPDATE_CHUNK])
                    )
                )
                .values(is_command=True)
            )
        await session.commit()


def command_row_ids(
    rows: Sequence[tuple[int, int, bool, str]], command_names: frozenset[str]
) -> list[int]:
    """按 (row_id, group_id, is_self, text) 的有序序列算出应打标的行。

    规则见模块 docstring；纯函数，迁移与单测共用。
    """
    out: list[int] = []
    in_reply_zone = False
    zone_group: int | None = None
    for row_id, group_id, is_self, text in rows:
        if group_id != zone_group:  # 换群：回复区间不跨群
            in_reply_zone = False
            zone_group = group_id
        if is_self:
            if in_reply_zone:
                out.append(row_id)
            continue
        hit = bool(command_names) and detect_command_name(text or "") in command_names
        if hit:
            out.append(row_id)
        in_reply_zone = hit
    return out


# ---------------------------------------------------------------- 命令行入口


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m candybot.migrations",
        description="对 candy.db 执行待运行的迁移（存量 is_command 回填等）",
    )
    parser.add_argument(
        "db",
        nargs="?",
        default=None,
        help="candy.db 路径（缺省取 config.json5 的 bot.data_dir/candy.db）",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只统计将被打标的行数，不写库"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from config import Config  # 仓库根的配置加载器（-m 从仓库根运行）

    from .models import load_settings
    from .plugin_api import CommandRegistry, build_registry

    try:
        settings = load_settings(Config)
    except ValueError as exc:
        print(f"配置有误：{exc}")
        return 2
    command_names = (
        frozenset(build_registry(settings, CommandRegistry()).names())
        if settings.plugins.enabled
        else frozenset()
    )
    db_path = Path(args.db) if args.db else Path(settings.bot.data_dir) / "candy.db"

    async def _run() -> list[tuple[str, int]]:
        db = CandyDatabase(db_path)
        try:
            return await run_migrations(
                db, command_names=command_names, dry_run=args.dry_run
            )
        finally:
            await db.close()

    results = asyncio.run(_run())
    if not results:
        print(f"{db_path}：没有待执行的迁移")
    for name, tagged in results:
        print(
            f"{db_path}：迁移 {name} "
            + ("（试运行）" if args.dry_run else "")
            + f"打标 {tagged} 条"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
