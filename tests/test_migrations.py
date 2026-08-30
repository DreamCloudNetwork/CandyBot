"""candybot/migrations.py：补列迁移与存量 is_command 回填。

覆盖：旧 schema 补列 + 按注册表打标；命令回复区间规则（跨群断开、
未知命令不打标、插件关闭不回填）；dry-run 不落一字节；重复执行幂等。
种子 SQL 全为固定字面量（还原 is_command 引入前的表结构与存量行），
测试数据不含任何外部输入。
"""

from __future__ import annotations

import sqlite3
import time

from sqlmodel import select

from candybot.database import CandyDatabase
from candybot.migrations import (
    M_IS_COMMAND,
    SchemaMigrationRow,
    command_row_ids,
    pending_migrations,
    run_migrations,
)
from candybot.models import ChatRecord


def build_old_schema_db(path) -> None:
    """建一份「is_command 列引入之前」的库：旧表结构 + 固定存量行。

    群 42：普通消息 → /hi → 自发言×2 → 普通消息 → 自发言；
    群 43：/hi → 自发言（验证回复区间不跨群）。
    """
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE chat_history (row_id INTEGER PRIMARY KEY, group_id INTEGER NOT NULL, message_id INTEGER NOT NULL, user_id INTEGER NOT NULL, nickname VARCHAR NOT NULL, text VARCHAR NOT NULL, ts FLOAT NOT NULL, is_self BOOLEAN NOT NULL, UNIQUE (group_id, message_id))"
        )
        conn.execute(
            "INSERT INTO chat_history (group_id, message_id, user_id, nickname, text, ts, is_self) VALUES (42, 1, 1001, 'u1', '在吗', 1.0, 0)"
        )
        conn.execute(
            "INSERT INTO chat_history (group_id, message_id, user_id, nickname, text, ts, is_self) VALUES (42, 2, 1001, 'u1', '/hi', 2.0, 0)"
        )
        conn.execute(
            "INSERT INTO chat_history (group_id, message_id, user_id, nickname, text, ts, is_self) VALUES (42, 3, 99, '糖糖', '你好！', 3.0, 1)"
        )
        conn.execute(
            "INSERT INTO chat_history (group_id, message_id, user_id, nickname, text, ts, is_self) VALUES (42, 4, 99, '糖糖', '再补一句', 3.5, 1)"
        )
        conn.execute(
            "INSERT INTO chat_history (group_id, message_id, user_id, nickname, text, ts, is_self) VALUES (42, 5, 1002, 'u2', '哈哈', 4.0, 0)"
        )
        conn.execute(
            "INSERT INTO chat_history (group_id, message_id, user_id, nickname, text, ts, is_self) VALUES (42, 6, 99, '糖糖', '确实', 5.0, 1)"
        )
        conn.execute(
            "INSERT INTO chat_history (group_id, message_id, user_id, nickname, text, ts, is_self) VALUES (43, 7, 1001, 'u1', '/hi', 6.0, 0)"
        )
        conn.execute(
            "INSERT INTO chat_history (group_id, message_id, user_id, nickname, text, ts, is_self) VALUES (43, 8, 99, '糖糖', '你好！', 7.0, 1)"
        )
        conn.commit()
    finally:
        conn.close()


def make_record(
    mid: int, text: str, *, is_self: bool = False, is_command: bool = False,
    group_id: int = 42,
) -> ChatRecord:
    return ChatRecord(
        message_id=mid,
        group_id=group_id,
        user_id=1000 + mid,
        nickname="糖糖" if is_self else f"u{mid}",
        text=text,
        ts=time.time() + mid,
        is_self=is_self,
        is_command=is_command,
    )


async def applied_names(db: CandyDatabase) -> list[str]:
    async with db.sessions() as session:
        rows = (await session.exec(select(SchemaMigrationRow))).all()
    return sorted(row.name for row in rows)


# ---------------------------------------------------------------- 旧 schema


async def test_old_schema_column_added_and_backfilled(tmp_path):
    """旧库补列成功，且命令消息与其后的命令回复按规则回填标记。"""
    db_path = tmp_path / "candy.db"
    build_old_schema_db(db_path)
    db = CandyDatabase(db_path)
    results = await run_migrations(db, command_names=frozenset({"hi"}))
    assert results == [(M_IS_COMMAND, 5)]

    rows42 = await db.load_recent(42, 10)
    assert [(r.text, r.is_command) for r in rows42] == [
        ("在吗", False),
        ("/hi", True),
        ("你好！", True),   # 紧跟命令消息的自发言：视为命令回复
        ("再补一句", True),  # 区间未断，第二条同样打标
        ("哈哈", False),
        ("确实", False),    # 区间已断：普通自发言不打标
    ]
    rows43 = await db.load_recent(43, 10)  # 换群后区间重新计
    assert [(r.text, r.is_command) for r in rows43] == [("/hi", True), ("你好！", True)]
    assert await applied_names(db) == [M_IS_COMMAND]
    await db.close()


# ---------------------------------------------------------------- 新 schema 回填


async def test_backfill_skips_unknown_and_unregistered(tmp_path):
    """未知/未注册命令名不打标；其后的自发言也不算命令回复。"""
    db = CandyDatabase(tmp_path / "candy.db")
    await db.create_tables()
    await db.insert_record(make_record(1, "/hi"))
    await db.insert_record(make_record(-1, "你好！", is_self=True))
    await db.insert_record(make_record(2, "/roll 3"))          # 不在注册表
    await db.insert_record(make_record(-2, "点数：5", is_self=True))
    await db.insert_record(make_record(3, "/hi --upper"))      # 带参数也命中

    results = await run_migrations(db, command_names=frozenset({"hi"}))
    assert results == [(M_IS_COMMAND, 3)]
    rows = await db.load_recent(42, 10)
    assert [(r.text, r.is_command) for r in rows] == [
        ("/hi", True),
        ("你好！", True),
        ("/roll 3", False),
        ("点数：5", False),
        ("/hi --upper", True),
    ]
    await db.close()


async def test_no_backfill_without_command_names(tmp_path):
    """插件总开关关闭（命令名集合为空）：什么都不回填。"""
    db = CandyDatabase(tmp_path / "candy.db")
    await db.create_tables()
    await db.insert_record(make_record(1, "/hi"))
    await db.insert_record(make_record(-1, "你好！", is_self=True))

    results = await run_migrations(db, command_names=frozenset())
    assert results == [(M_IS_COMMAND, 0)]
    assert [r.is_command for r in await db.load_recent(42, 10)] == [False, False]
    await db.close()


# ---------------------------------------------------------------- dry-run 与幂等


async def test_dry_run_writes_nothing(tmp_path):
    """dry_run：只返回将打标行数，不补列、不更新、不记迁移名单。"""
    db_path = tmp_path / "candy.db"
    build_old_schema_db(db_path)
    db = CandyDatabase(db_path)
    results = await run_migrations(
        db, command_names=frozenset({"hi"}), dry_run=True
    )
    assert results == [(M_IS_COMMAND, 5)]
    conn = sqlite3.connect(db_path)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(chat_history)")}
        tables = {
            r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        conn.close()
    assert "is_command" not in cols  # dry_run 连 DDL 都不做
    assert "schema_migration" not in tables  # 也不建表、不记名单
    await db.close()


async def test_migration_runs_once(tmp_path):
    """跑过即记录名单：第二次调用不再执行（换命令名集合也不重算）。"""
    db = CandyDatabase(tmp_path / "candy.db")
    await db.create_tables()
    await db.insert_record(make_record(1, "/hi"))
    assert await run_migrations(db, command_names=frozenset({"hi"})) == [
        (M_IS_COMMAND, 1)
    ]
    assert await run_migrations(db, command_names=frozenset({"hi", "echo"})) == []
    assert [r.is_command for r in await db.load_recent(42, 10)] == [True]
    await db.close()


# ---------------------------------------------------------------- 启动兼容检查


async def test_pending_migrations_gap_detection(tmp_path):
    """启动检查看实际结构而非迁移名单：旧库缺列报缺口，迁移后/全新库不报。"""
    old_path = tmp_path / "old.db"
    build_old_schema_db(old_path)
    db_old = CandyDatabase(old_path)
    assert await pending_migrations(db_old) == [
        "chat_history.is_command（迁移 chat_history_is_command）"
    ]
    await run_migrations(db_old, command_names=frozenset({"hi"}))
    assert await pending_migrations(db_old) == []
    await db_old.close()

    # 全新库：建表自带全部新列，哪怕从未跑过迁移也不算落后
    db_new = CandyDatabase(tmp_path / "new.db")
    await db_new.create_tables()
    assert await pending_migrations(db_new) == []
    await db_new.close()

    # 空目录：chat_history 尚未建出，交给启动建表，不算缺口
    db_empty = CandyDatabase(tmp_path / "empty.db")
    assert await pending_migrations(db_empty) == []
    await db_empty.close()


# ---------------------------------------------------------------- 纯函数规则


def test_command_row_ids_zone_rules():
    """command_row_ids：区间规则与命令名形态的直接单测。"""
    rows = [
        (1, 42, False, "/hi"),        # 命令 → 打标
        (2, 42, True, "你好！"),      # 区间内自发言 → 打标
        (3, 42, False, "普通消息"),   # 断开区间
        (4, 42, True, "模型回复"),    # 不再打标
        (5, 42, False, " /hi"),       # lstrip 后仍命中（与 _detect_command 一致）
        (6, 42, True, "好。"),
        (7, 43, False, "  "),         # 空白消息：不算命令
        (8, 43, False, "/ hi"),       # / 后紧跟空白：不算命令
    ]
    ids = command_row_ids(rows, frozenset({"hi"}))
    assert ids == [1, 2, 5, 6]
    # 首条即自发言（无前导命令）：不打标
    assert command_row_ids([(1, 42, True, "在吗？")], frozenset({"hi"})) == []
    # 命令名不在集合：其自发言也不进区间
    assert command_row_ids(
        [(1, 42, False, "/roll"), (2, 42, True, "点数：5")], frozenset({"hi"})
    ) == []
