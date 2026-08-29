"""配置热重载测试。

覆盖三层：config.ConfigFileEventHandler 的事件过滤、原子保存形态兼容与
解析失败保旧配置；CandyBot.reload_settings 的成功替换、失败沿用与
settings_loader 缺失时的告警；以及 main.py 接线的端到端回归（真实
watchdog Observer → call_soon_threadsafe → 运行中的 bot 换配置）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from types import SimpleNamespace

from watchdog.observers import Observer

from candybot import bot as bot_module
from candybot.models import load_settings
from config import Config, ConfigFileEventHandler
from tests.test_integration import FakeSnowluma, make_settings


def _file_event(path: str, *, is_directory: bool = False, dest: str | None = None):
    """模拟 watchdog 的文件事件（handler 只读这三个字段）。"""
    return SimpleNamespace(
        is_directory=is_directory, src_path=path, dest_path=dest or path
    )


# ---------------------------------------------------------------- bot 层


async def test_reload_swaps_settings_and_rebuilds_ai(tmp_path):
    old = make_settings(tmp_path)  # response_post_process 默认关闭
    new = make_settings(tmp_path, post_process={"enabled": True, "max_split": 5})
    bot = bot_module.CandyBot(old, settings_loader=lambda: new)
    try:
        ai_before = bot._ai
        assert bot._settings.response_post_process.enabled is False

        assert bot.reload_settings() is True
        assert bot._settings is new
        # 模型/生成参数烘在 AIClient 构造里，必须随配置重建
        assert bot._ai is not ai_before
        assert bot._settings.response_post_process.max_split == 5
    finally:
        # 未 start 直接 stop 关闭 aiohttp 等资源（与既有 bot 测试同款收尾）
        await bot.stop()


async def test_reload_failure_keeps_old_settings(tmp_path, caplog):
    old = make_settings(tmp_path)
    ai_before = None

    def boom():
        raise ValueError("config.json5 第 3 行语法错误")

    bot = bot_module.CandyBot(old, settings_loader=boom)
    ai_before = bot._ai
    try:
        with caplog.at_level(logging.WARNING):
            assert bot.reload_settings() is False
        # 失败后一切照旧：旧快照、旧 AI 客户端，错误完整出现在日志里
        assert bot._settings is old
        assert bot._ai is ai_before
        assert "继续使用旧配置" in caplog.text
    finally:
        await bot.stop()


async def test_reload_without_loader_warns(tmp_path, caplog):
    old = make_settings(tmp_path)
    bot = bot_module.CandyBot(old)  # 测试里常见的无 loader 构造
    try:
        with caplog.at_level(logging.WARNING):
            assert bot.reload_settings() is False
        assert "热重载不可用" in caplog.text
    finally:
        await bot.stop()


# ---------------------------------------------------------------- 事件处理器层


def _write_cfg(tmp_path, content: str) -> str:
    cfg_file = tmp_path / "config.json5"
    cfg_file.write_text(content, encoding="utf-8")
    return str(cfg_file)


def test_handler_reloads_only_target_file(tmp_path, monkeypatch):
    path = _write_cfg(tmp_path, '{"bot": {"self_qq": 1}}')
    monkeypatch.setattr(Config, "_config_file", path)
    monkeypatch.setattr(Config, "_config", {"stale": True})
    fired: list[int] = []
    handler = ConfigFileEventHandler(
        config_file=path, on_reload=lambda: fired.append(1)
    )

    # 同目录的其他文件与目录事件都不该触发重载
    handler.on_modified(_file_event(str(tmp_path / "other.json5")))
    handler.on_created(_file_event(str(tmp_path), is_directory=True))
    assert fired == []

    handler.on_modified(_file_event(path))
    assert fired == [1]
    assert Config._config == {"bot": {"self_qq": 1}}


def test_handler_atomic_save_via_move(tmp_path, monkeypatch):
    path = _write_cfg(tmp_path, '{"bot": {"self_qq": 2}}')
    monkeypatch.setattr(Config, "_config_file", path)
    monkeypatch.setattr(Config, "_config", {})
    fired: list[int] = []
    handler = ConfigFileEventHandler(
        config_file=path, on_reload=lambda: fired.append(1)
    )

    # 原子保存：临时文件被 rename 到目标路径，src 是临时名、dest 才是配置
    handler.on_moved(
        _file_event(str(tmp_path / ".tmpXXXX"), dest=path)
    )
    assert fired == [1]
    assert Config._config == {"bot": {"self_qq": 2}}


def test_handler_parse_error_keeps_config_and_skips_callback(
    tmp_path, monkeypatch, caplog
):
    path = _write_cfg(tmp_path, "{ 这不是合法的 json5 !!")
    monkeypatch.setattr(Config, "_config_file", path)
    monkeypatch.setattr(Config, "_config", {"bot": {"self_qq": 42}})
    fired: list[int] = []
    handler = ConfigFileEventHandler(
        config_file=path, on_reload=lambda: fired.append(1)
    )

    with caplog.at_level(logging.ERROR):
        handler.on_modified(_file_event(path))
    # 解析失败：不回调、不污染内存中的旧配置，错误完整记录
    assert fired == []
    assert Config._config == {"bot": {"self_qq": 42}}
    assert "配置文件解析失败" in caplog.text


# ---------------------------------------------------------------- 端到端接线


def _cfg_dict(tmp_path, persona: str) -> dict:
    """一份对 load_settings 合法的最小配置（与 make_settings 同构，可直接落盘）。"""
    return {
        "bot": {"self_qq": 99, "data_dir": str(tmp_path / "data")},
        "groups": {
            "42": {
                "persona": persona,
                "proactivity_threshold": 6,
                "cooldown_seconds": 60,
            }
        },
        "groups_default": {"enabled": False, "persona": "默认人设"},
        "ai_backend": {"base_url": "https://api.example.com/v1", "api_key": "k"},
        "models": {"judge": "j-model", "reply": "r-model"},
        "generation": {},
        "multimodal": {},
        "rate_limit": {},
        "snowluma": {
            "endpoint": "http://10.0.0.5:3000/",
            "allow_private_endpoint": True,
        },
        "response_post_process": {"enabled": False},
    }


async def test_end_to_end_observer_swaps_running_settings(tmp_path, monkeypatch):
    """真实 watchdog Observer 的端到端接线回归。

    守住的正是 main.py 热重载的三个关键点：监听目录而非文件（编辑器原子
    保存换 inode）、回调经 call_soon_threadsafe 回到事件循环、替换与消息
    处理串行。in-place 写文件会先触发一次「截断后的空文件」事件，解析失败
    被容忍并沿用旧配置，随后写入完成的事件才完成替换——5 秒轮询超时兜住
    调度抖动。
    """
    cfg_path = tmp_path / "config.json5"

    def write(persona: str) -> None:
        cfg_path.write_text(
            json.dumps(_cfg_dict(tmp_path, persona), ensure_ascii=False),
            encoding="utf-8",
        )

    write("旧人设")
    monkeypatch.setattr(Config, "_config_file", str(cfg_path))
    Config.load_config()
    bot = bot_module.CandyBot(
        load_settings(Config), settings_loader=lambda: load_settings(Config)
    )
    bot._snowluma = FakeSnowluma()
    loop = asyncio.get_running_loop()
    observer = Observer()
    observer.schedule(
        ConfigFileEventHandler(
            config_file=str(cfg_path),
            on_reload=lambda: loop.call_soon_threadsafe(bot.reload_settings),
        ),
        str(tmp_path),
        recursive=False,
    )
    observer.start()
    try:
        await asyncio.sleep(0.2)  # 等 observer 完成注册
        write("新人设")
        deadline = time.monotonic() + 5.0
        while (
            bot._settings.groups[42].persona != "新人设"
            and time.monotonic() < deadline
        ):
            await asyncio.sleep(0.05)
    finally:
        observer.stop()
        observer.join()
        await bot.stop()
    assert bot._settings.groups[42].persona == "新人设"
