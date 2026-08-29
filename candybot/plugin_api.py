"""命令插件 SDK：注册表、命令声明与插件目录加载。

插件作者在插件文件里写 `from candybot.plugin_api import command, CommandParam`，
用装饰器注册命令；机器人收到以 / 开头且命令名命中的群消息时跳过大模型
（见 bot._run_command），按 unix 终端命令风格解析参数并调用 handler，
把返回值（文本 / OneBot 段列表 / None）发到群里。

进程内有一个共享的 `default_registry`：不带自定义注册表构造的 CandyBot
（生产路径）直接装载它。测试或特殊装配可以传入独立 CommandRegistry，
加载插件目录时装饰器会临时指向该注册表（_loading_target 机制），
互不污染。
"""

from __future__ import annotations

import importlib.util
import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)

# handler 入参 ctx、返回 str（纯文本）/ list[dict]（OneBot 段数组）/ None（不发消息）；
# 允许同步或异步（异步时受 plugins.timeout_seconds 约束）。
CommandHandler = Callable[..., "str | list[dict] | None"]

# 合法的参数类型名（commandline.py 据此映射到 argparse 的 type 转换器）
PARAM_TYPES = ("str", "int", "float")
# 位置参数支持的 nargs（"?" 语义含糊，不提供）
POSITIONAL_NARGS = (None, "+", "*")

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class CommandParam:
    """命令的一个参数声明。

    name 以 "-" 开头的自动识别为选项（可省略 kind）：位置参数按顺序取值，
    nargs="+" 表示至少一个、"*" 表示任意个（含零个），取值恒为 list；带
    default 的位置参数自动成为可选参数。选项可同时给逗号分隔的长短别名
    （如 "--count,-c"）；store_true=True 表示开关型（不带值，出现即 True）。
    """

    name: str
    kind: str = ""  # 留空自动识别；也可显式 "positional" | "option"
    type: str = "str"  # "str" | "int" | "float"
    default: Any = None
    help: str = ""
    nargs: str | None = None
    store_true: bool = False

    def __post_init__(self) -> None:
        auto_kind = "option" if self.name.lstrip().startswith("-") else "positional"
        if self.kind == "":
            object.__setattr__(self, "kind", auto_kind)
        elif self.kind != auto_kind:
            raise ValueError(
                f"CommandParam：name {self.name!r} 与 kind {self.kind!r} 矛盾"
            )
        if self.kind not in ("positional", "option"):
            raise ValueError(f"CommandParam.kind 非法：{self.kind!r}")
        if self.type not in PARAM_TYPES:
            raise ValueError(f"CommandParam.type 非法：{self.type!r}")
        if self.kind == "positional":
            if self.nargs not in POSITIONAL_NARGS:
                raise ValueError(f"位置参数 nargs 只能是 {POSITIONAL_NARGS}：{self!r}")
            if self.store_true:
                raise ValueError("store_true 只能用于选项参数")
            if not _NAME_RE.fullmatch(self.name):
                raise ValueError(
                    f"位置参数名需为小写字母/数字/_/-，实际是 {self.name!r}"
                )
        else:
            flags = [part.strip() for part in self.name.split(",")]
            if not flags or any(
                not re.fullmatch(r"--?[a-z0-9][a-z0-9_-]*", flag) for flag in flags
            ):
                raise ValueError(
                    f"选项参数需形如 '--count' 或 '--count,-c'，实际是 {self.name!r}"
                )
            if self.store_true and self.type != "str":
                raise ValueError("store_true 开关不带值，不能声明 type")
            if self.nargs is not None:
                raise ValueError("选项参数不支持 nargs")

    @property
    def dest(self) -> str:
        """结果字典里的键名：去掉选项的横线前缀。"""
        if self.kind == "positional":
            return self.name
        first = self.name.split(",")[0].strip()
        return first.lstrip("-")

    @property
    def flags(self) -> tuple[str, ...]:
        """option 参数交给 argparse 的全部 flag（含别名）。"""
        return tuple(part.strip() for part in self.name.split(","))


@dataclass(slots=True)
class CommandSpec:
    """一条已注册的命令。"""

    name: str
    handler: CommandHandler
    params: tuple[CommandParam, ...] = ()
    help: str = ""
    plugin: str = "builtin"  # 来源插件名（/help 展示与日志用）

    def __post_init__(self) -> None:
        if not _NAME_RE.fullmatch(self.name):
            raise ValueError(
                f"命令名需为小写字母/数字/_/-（不含 /），实际是 {self.name!r}"
            )


@dataclass(slots=True)
class CommandContext:
    """handler 的入参：本次调用的上下文。

    args 是解析好的参数字典（键为 CommandParam.dest）；registry/settings 为
    只读快照，db 是 CandyDatabase（插件需要持久状态时用，可为 None，如测试
    里的裸注册表）。
    """

    group_id: int
    user_id: int
    nickname: str
    text: str  # 原始命令全文（已 lstrip）
    args: dict[str, Any]
    registry: "CommandRegistry"
    settings: Any  # Settings 快照（避免循环导入不做类型标注）
    db: Any = None


class CommandRegistry:
    """命令名 → CommandSpec 的注册表。重名先到者胜，后来者拒绝并记日志。"""

    def __init__(self) -> None:
        self._specs: dict[str, CommandSpec] = {}

    def register(self, spec: CommandSpec) -> CommandSpec:
        existing = self._specs.get(spec.name)
        if existing is not None:
            # 重复注册：进程内多实例重建时插件文件被再次导入属正常，
            # 静默保留先到者；真正的撞名由构建期的 error 日志提示。
            logger.debug(
                "命令 /%s 已由插件 %s 注册，忽略 %s 的同名注册",
                spec.name,
                existing.plugin,
                spec.plugin,
            )
            return existing
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> CommandSpec | None:
        return self._specs.get(name)

    def names(self) -> list[str]:
        return sorted(self._specs)

    def specs(self) -> list[CommandSpec]:
        """按命令名字典序排列的全部命令（/help 展示顺序与此一致）。"""
        return [self._specs[name] for name in self.names()]

    def command(
        self,
        name: str,
        params: tuple[CommandParam, ...] = (),
        help: str = "",
        *,
        plugin: str = "builtin",
    ) -> Callable[[CommandHandler], CommandHandler]:
        """装饰器工厂：`@registry.command("echo", ...)` 或经模块级
        `command` 装饰器（插件文件的常规写法）。"""

        def decorate(handler: CommandHandler) -> CommandHandler:
            self.register(
                CommandSpec(
                    name=name,
                    handler=handler,
                    params=tuple(params),
                    help=help,
                    plugin=plugin,
                )
            )
            return handler

        return decorate


# ------------------------------------------------------------ 默认注册表与加载

default_registry = CommandRegistry()

# 插件文件里 `from candybot.plugin_api import command` 拿到的装饰器在装饰
# 时刻决定注册去向：构建期通过 _loading_target 指向本次装配的注册表，
# 平时（直接 import 使用）落到 default_registry。
_loading_target: CommandRegistry | None = None
_loading_plugin_label: str | None = None


@contextmanager
def _loading_into(registry: CommandRegistry, plugin_label: str) -> Iterator[None]:
    global _loading_target, _loading_plugin_label
    _loading_target, _loading_plugin_label = registry, plugin_label
    try:
        yield
    finally:
        _loading_target, _loading_plugin_label = None, None


def command(
    name: str,
    params: tuple[CommandParam, ...] = (),
    help: str = "",
) -> Callable[[CommandHandler], CommandHandler]:
    """插件侧装饰器：注册命令到当前加载目标（通常为 default_registry）。"""
    registry = _loading_target if _loading_target is not None else default_registry
    plugin = _loading_plugin_label or "builtin"
    return registry.command(name, params=params, help=help, plugin=plugin)


def format_usage(spec: CommandSpec) -> str:
    """由参数声明生成一行 unix 风格用法，如 `/roll <次数> [--面数 N]`。"""
    parts = [f"/{spec.name}"]
    for p in spec.params:
        if p.kind == "positional":
            metavar = p.dest.upper()
            piece = f"<{metavar}...>" if p.nargs == "+" else f"<{metavar}>"
            if p.nargs == "*":
                piece = f"[<{metavar}...>]"
            parts.append(piece)
        elif p.store_true:
            parts.append(f"[{p.flags[0]}]")
        else:
            parts.append(f"[{p.flags[0]} {p.dest.upper()}]")
    return " ".join(parts)


def load_plugin_dir(registry: CommandRegistry, directory: str | Path) -> list[str]:
    """逐个导入目录下的插件 .py 文件（按文件名排序，`_` 开头跳过）。

    插件模块用 importlib 按文件路径加载（不进 sys.modules，重复构建注册表
    时重新执行注册语句即可，重名先到者胜）。单个插件导入失败只记 error
    并跳过——坏插件绝不拖垮机器人。返回成功装载的插件文件名列表。
    """
    root = Path(directory)
    if not root.is_dir():
        logger.debug("插件目录 %s 不存在，跳过加载", root)
        return []
    loaded: list[str] = []
    for path in sorted(root.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module_name = f"candybot_plugin__{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"无法为 {path} 构造模块规格")
            module = importlib.util.module_from_spec(spec)
            with _loading_into(registry, path.stem):
                spec.loader.exec_module(module)
        except Exception:
            logger.error("加载插件 %s 失败，已跳过", path, exc_info=True)
            continue
        loaded.append(path.name)
    logger.info("插件目录 %s 已加载 %d 个插件：%s", root, len(loaded), loaded)
    return loaded


def build_registry(
    settings: Any, registry: CommandRegistry | None = None
) -> CommandRegistry:
    """构建本次装配的命令注册表：内置命令 + plugins.enabled 时扫描插件目录。

    registry=None（生产路径）使用进程共享的 default_registry。
    """
    from candybot.builtin_plugins import help as help_plugin

    target = registry if registry is not None else default_registry
    help_plugin.register(target)
    ps = getattr(settings, "plugins", None)
    if ps is not None and ps.enabled:
        load_plugin_dir(target, ps.dir)
    return target
