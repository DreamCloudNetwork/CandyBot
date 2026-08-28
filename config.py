import json5
from typing import Any
from watchdog.events import FileSystemEventHandler
import logging
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
    def on_created(self, event):
        pass

    def on_modified(self, event):
        logger.info("配置文件被修改，正在重载")
        Config.load_config()

    def on_deleted(self, event):
        logger.warning("请勿删除配置文件，内存中的配置为 %s", Config.__dict__['_config'])

Config = ConfigClass()
