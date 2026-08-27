import json
from typing import Any


class ConfigClass:
    def __init__(self, config_file: str = 'config.json'):
        self._json_config: dict[Any, Any] = {}
        self._config_file: str = config_file
        self.load_config()

    def __getattr__(self, name: str) -> Any:
        if name == "load_config" or name == "save_config":
            return self.__dict__[name]
        return self.__dict__['_json_config'][name]

    def __setattr__(self, name: str, value: Any) -> None:
        self.__dict__[name] = value

    def load_config(self) -> None:
        self._json_config = json.load(open(self._config_file))

    def save_config(self) -> None:
        with open(self._config_file, 'w') as outfile:
            json.dump(self._json_config, outfile, indent=4)


Config = ConfigClass()
