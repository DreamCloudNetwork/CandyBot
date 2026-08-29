"""CandyBot 启动入口。"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

from watchdog.observers import Observer

from candybot import __version__
from candybot.bot import build_bot
from config import Config, ConfigFileEventHandler

logging.basicConfig(
    level=logging.INFO,  # 配置加载前先用默认级别；读取 bot.log_level 后再覆盖
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
noisy_modules = ("openai", "httpx", "httpcore", "httpcore2", "asyncio", "aiosqlite", "watchdog")
# 屏蔽依赖库自身的 DEBUG 噪音，只保留 CandyBot 的调试输出
for noisy in noisy_modules:
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("candybot")

_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def apply_log_level(level_name: str) -> None:
    """按配置设置根 logger 级别（模型请求的 DEBUG 日志由此开关）。"""
    level = level_name.upper()
    if level not in _LEVELS:
        logger.warning("未知的日志级别 %r，保持 INFO", level_name)
        return
    logging.getLogger().setLevel(getattr(logging, level))
    if level == "DEBUG":
        # DEBUG 下依赖库的连接级日志仍然太吵，维持压制
        for noisy in noisy_modules:
            logging.getLogger(noisy).setLevel(logging.WARNING)


async def run() -> int:
    try:
        bot = build_bot()
    except ValueError as exc:
        logger.error("配置有误：%s", exc)
        return 2
    apply_log_level(bot.log_level)
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:  # Windows 兼容
            signal.signal(sig, lambda *_: stop_event.set())

    logger.info("CandyBot v%s 启动中…", __version__)
    await bot.start()
    logger.info("CandyBot 已就绪，等待事件…")
    # 配置热重载：watchdog 观察线程触发 → call_soon_threadsafe 调回事件循环
    # 原子替换运行时配置（与消息处理串行，见 CandyBot.reload_settings），
    # 日志级别随之更新。监听的是配置文件所在目录而非文件本身，原因见
    # config.ConfigFileEventHandler 的 docstring。
    config_path = os.path.abspath(Config._config_file)

    def on_config_reload() -> None:
        if bot.reload_settings():
            apply_log_level(bot.log_level)

    observer = Observer()
    observer.schedule(
        ConfigFileEventHandler(
            config_file=config_path,
            on_reload=lambda: loop.call_soon_threadsafe(on_config_reload),
        ),
        os.path.dirname(config_path),
        recursive=False,
    )
    observer.start()
    try:
        await stop_event.wait()
    finally:
        logger.info("正在退出…")
        await bot.stop()
        observer.stop()
        observer.join()
    return 0


def main() -> None:
    sys.exit(asyncio.run(run()))


if __name__ == "__main__":
    main()
