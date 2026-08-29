"""unix 终端风格的命令行解析：/ 前缀识别 + shlex 切词 + argparse 参数校验。

两步分工：
- detect_command_name：只按第一个空白切出命令名（不经 shlex，引号不影响
  命中判断），bot._on_event 据此查注册表决定是否短路大模型；
- parse_invocation：命令名命中后才做完整解析（shlex 支持引号包参、
  argparse 按声明校验位置参数与 --选项）。
"""

from __future__ import annotations

import argparse
import logging
import re
import shlex
from typing import Any

from candybot.plugin_api import CommandSpec, format_usage

logger = logging.getLogger(__name__)

_TYPE_MAP: dict[str, Any] = {"str": str, "int": int, "float": float}

# 形似选项但不该参与置换的 token：负数（-3、-1.5）与单独的 "-"
_NUMBER_RE = re.compile(r"^-\d+(\.\d+)?$")


def _looks_like_option(token: str) -> bool:
    return token.startswith("-") and not _NUMBER_RE.match(token)


class CommandUsageError(ValueError):
    """参数用法错误：消息直接作为提示发回群里（含用法行与 /help 引导）。"""


def detect_command_name(text: str) -> str | None:
    """`/echo hi` → "echo"；非 / 开头、/ 后紧跟空白等情况返回 None。"""
    stripped = text.lstrip()
    if not stripped.startswith("/"):
        return None
    rest = stripped[1:]
    if not rest or rest[0].isspace():
        return None
    return rest.split(None, 1)[0]


def _build_parser(spec: CommandSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"/{spec.name}",
        add_help=False,
        # exit_on_error=False：参数错误抛 ArgumentError 而不是 SystemExit，
        # 绝不能让一次群聊输入把机器人进程问停
        exit_on_error=False,
        # 关掉 --ab 之类的前缀缩写，选项必须写全，行为更可预期
        allow_abbrev=False,
    )
    for p in spec.params:
        kwargs: dict[str, Any] = {"help": p.help or argparse.SUPPRESS}
        if p.kind == "positional":
            nargs = p.nargs
            if nargs is None and p.default is not None:
                nargs = "?"
                kwargs["default"] = p.default
            if nargs is not None:
                kwargs["nargs"] = nargs
            kwargs["type"] = _TYPE_MAP[p.type]
            parser.add_argument(p.dest, **kwargs)
        elif p.store_true:
            kwargs["action"] = "store_true"
            kwargs["default"] = bool(p.default) if p.default is not None else False
            parser.add_argument(*p.flags, **kwargs)
        else:
            kwargs["type"] = _TYPE_MAP[p.type]
            if p.default is not None:
                kwargs["default"] = p.default
            parser.add_argument(*p.flags, **kwargs)
    return parser


def _usage_error(spec: CommandSpec, message: str) -> CommandUsageError:
    """面向群聊用户的错误：原因 + 用法行 + /help 引导。"""
    return CommandUsageError(
        f"{message}\n用法：{format_usage(spec)}\n用 /help {spec.name} 查看参数说明"
    )


def _permute(spec: CommandSpec, argv: list[str]) -> list[str]:
    """GNU 风格的选项/位置参数自由混排置换。

    argparse 对「贪心位置参数 + 夹在中间的选项」有历史局限（nargs="+" 后
    再出现的裸位置参数会被报 unrecognized）；unix 用户的直觉是选项可以出现
    在任何位置，所以先把选项连同其值挪到最前、位置参数按原相对顺序排在后。
    只识别声明过的 flag（含 --flag=value 形式与短别名）；带值的未知 token
    留在原位交给 argparse 报「unrecognized arguments」。"--" 之后的一切
    按标准语义不再置换。负数（如 /roll -3）不是选项，留给类型校验报错。
    """
    flags_value = set()  # 需要吞一个值的选项 flag
    flags_bare = set()  # store_true 开关
    for p in spec.params:
        if p.kind != "option":
            continue
        (flags_bare if p.store_true else flags_value).update(p.flags)
    opts: list[str] = []
    positionals: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--":
            # 标准语义：-- 之后的 token 一律按位置参数处理，原样带上交给 argparse
            positionals.extend(argv[i:])
            break
        name, sep, value = token.partition("=")
        if name in flags_bare and (not sep or not value):
            opts.append(token)
        elif name in flags_value:
            opts.append(token)
            if sep == "" and i + 1 < len(argv) and not _looks_like_option(argv[i + 1]):
                opts.append(argv[i + 1])  # --sides 20 的值跟着 flag 走
                i += 1
        elif _looks_like_option(token):
            opts.append(token)  # 未知选项：留在选项组里，交给 argparse 报错
        else:
            positionals.append(token)
        i += 1
    return opts + positionals


def parse_invocation(spec: CommandSpec, text: str) -> dict[str, Any]:
    """把命令全文解析成参数字典（键为 dest）。

    任何解析失败（引号不配对、未知选项、类型不符、缺少参数）统一抛
    CommandUsageError，消息面向群聊用户：错误原因 + 用法行 + /help 引导。
    """
    try:
        tokens = shlex.split(text.lstrip(), posix=True)
    except ValueError:
        tokens = None
    if tokens is None or not tokens or tokens[0].lstrip("/") != spec.name:
        # shlex 不配对（"unbalanced quote"）时连命令名都取不出；首 token
        # 与声明不符只可能发生在调用方用错 spec，防御性归为用法错误
        raise _usage_error(spec, "命令参数解析失败（引号不配对或参数为空？）")
    parser = _build_parser(spec)
    try:
        ns = parser.parse_args(_permute(spec, tokens[1:]))
    except argparse.ArgumentError as exc:
        raise _usage_error(spec, f"参数有误：{exc}") from exc
    except Exception as exc:  # argparse 边缘行为的兜底，绝不向上抛原始异常
        logger.exception("命令 /%s 解析出现预期外的错误", spec.name)
        raise _usage_error(spec, f"参数有误：{exc}") from exc
    return vars(ns)
