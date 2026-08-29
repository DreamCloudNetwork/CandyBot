"""JSON5 配置单例与配置文件变更事件入口。

ConfigFileEventHandler 只负责「认出目标配置文件变了 → 重解析原始 JSON →
通知回调」；把配置落到运行时（candybot.bot.CandyBot.reload_settings 重建
Settings 快照与 AI 客户端）由 main.py 注册的回调完成，配置模块不反向
依赖 bot。
"""

import json5
import logging
import os
from typing import Any, Callable

from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)


class ConfigClass:
    def __init__(self, config_file: str = 'config.json5'):
        self._config: dict[str, Any] = {}
        self._config_file: str = config_file
        self.load_config()

    def __getattr__(self, name: str) -> Any:
        return self.__dict__['_config'][name]

    def __setattr__(self, name: str, value: Any) -> None:
        self.__dict__[name] = value

    def load_config(self) -> None:
        with open(self._config_file, encoding='utf-8') as infile:
            self._config = json5.load(infile)


class ConfigFileEventHandler(FileSystemEventHandler):
    """监听配置文件所在目录，仅响应目标文件的变化。

    为什么不直接监听文件本身：编辑器普遍是原子保存（写临时文件后 rename
    替换 inode），inotify 挂在旧 inode 上的单文件 watch 会随之静默失效，
    保存一次之后就再也收不到事件。监听目录并按文件名过滤才稳定；原子保存
    在不同编辑器下可能呈现为 modified / created / moved 三种事件，这里统一
    收编。

    on_reload 回调只在配置**解析成功**后触发，运行于 watchdog 的观察线程
    （调用方需要自行调度回自己的事件循环）；解析失败不触发，内存中的旧配置
    原样保留，修好后再次保存自然重试。
    """

    def __init__(
        self,
        config_file: str | None = None,
        on_reload: Callable[[], None] | None = None,
    ):
        path = os.path.abspath(config_file or Config._config_file)
        self._name = os.path.basename(path)
        self._on_reload = on_reload

    def _is_config_event(self, event: Any) -> bool:
        return (
            not event.is_directory
            and os.path.basename(str(event.src_path)) == self._name
        )

    def on_created(self, event: Any) -> None:
        if self._is_config_event(event):
            self._reload()

    def on_modified(self, event: Any) -> None:
        if self._is_config_event(event):
            self._reload()

    def on_moved(self, event: Any) -> None:
        # rename 式原子保存里 src_path 是临时文件，落在目标路径的是 dest
        dest = str(getattr(event, "dest_path", "") or "")
        if not event.is_directory and os.path.basename(dest) == self._name:
            self._reload()

    def on_deleted(self, event: Any) -> None:
        if self._is_config_event(event):
            logger.warning(
                "请勿删除配置文件，内存中的配置为 %s", Config._config
            )

    def _reload(self) -> None:
        try:
            Config.load_config()
        except Exception:
            # 保存到一半/配置写坏不该让常驻服务下葬：完整记录异常，
            # 旧配置继续服役，修正后再保存会自动生效
            logger.exception("配置文件解析失败，继续使用内存中的旧配置")
            return
        logger.info("配置文件被修改，正在重载")
        if self._on_reload is not None:
            self._on_reload()


Config = ConfigClass()
