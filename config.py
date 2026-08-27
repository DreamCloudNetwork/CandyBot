import json5
from typing import Any


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


Config = ConfigClass()
