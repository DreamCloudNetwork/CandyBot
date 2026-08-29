"""unix 风格命令解析单测：命令名识别 + shlex 切词 + 选项置换 + argparse 校验。"""

from __future__ import annotations

import pytest

from candybot.commandline import CommandUsageError, detect_command_name, parse_invocation
from candybot.plugin_api import CommandParam, CommandSpec


def _spec(*params: CommandParam, name: str = "t") -> CommandSpec:
    return CommandSpec(name=name, handler=lambda ctx: None, params=params)


ECHO = _spec(
    CommandParam("text", nargs="+", help="要复读的文本"),
    CommandParam("--upper, -u", store_true=True, help="转大写"),
    name="echo",
)
ROLL = _spec(
    CommandParam("times", type="int", default=1, help="次数"),
    CommandParam("--sides", type="int", default=6, help="面数"),
    name="roll",
)


# ------------------------------------------------------------ detect_command_name


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/echo hi", "echo"),
        ("  /ping  ", "ping"),
        ("/a-b_c1 x", "a-b_c1"),
        ("普通消息", None),
        (" /  ", None),  # / 后只有空白
        ("/", None),
        ("", None),
        ("不是开头 /echo", None),
        ("/echo--upper hi", "echo--upper"),  # 粘连不算选项分隔，交给注册表未命中
    ],
)
def test_detect_command_name(text, expected):
    assert detect_command_name(text) == expected


# ------------------------------------------------------------ 正常解析


def test_positional_only():
    assert parse_invocation(ECHO, "/echo 你好 世界") == {"text": ["你好", "世界"], "upper": False}


def test_quotes_group_words():
    got = parse_invocation(ECHO, '/echo 你好 "带空格 的词"')
    assert got["text"] == ["你好", "带空格 的词"]


def test_option_after_positionals():
    got = parse_invocation(ECHO, "/echo hello --upper 世界")
    assert got == {"text": ["hello", "世界"], "upper": True}


def test_short_alias_and_equals_value():
    assert parse_invocation(ECHO, "/echo -u hello")["upper"] is True
    assert parse_invocation(ROLL, "/roll --sides=20 3") == {"times": 3, "sides": 20}


def test_flag_with_explicit_value_is_usage_error():
    # 开关型不带值：-u=1 会被 argparse 拒绝，不会静默吞掉
    with pytest.raises(CommandUsageError):
        parse_invocation(ECHO, "/echo -u=1 hi")


def test_defaults_apply():
    assert parse_invocation(ROLL, "/roll") == {"times": 1, "sides": 6}
    assert parse_invocation(ROLL, "/roll 5") == {"times": 5, "sides": 6}


def test_double_dash_forces_positional():
    got = parse_invocation(ECHO, "/echo a -- --upper -c")
    assert got["text"] == ["a", "--upper", "-c"]


def test_negative_number_is_not_option():
    # 负数不当选项做置换；到类型校验处按 int 正常解析
    assert parse_invocation(ROLL, "/roll -3")["times"] == -3


# ------------------------------------------------------------ 用法错误


@pytest.mark.parametrize(
    ("spec", "text", "frag"),
    [
        (ECHO, "/echo", "the following arguments are required"),
        (ECHO, "/echo 'unbalanced", "引号不配对"),
        (ROLL, "/roll abc", "invalid int value"),
        (ROLL, "/roll --nope", "unrecognized arguments"),
        (ROLL, "/roll 1 2", "unrecognized arguments"),  # 多余裸位置参数
    ],
)
def test_usage_errors(spec, text, frag):
    with pytest.raises(CommandUsageError) as exc:
        parse_invocation(spec, text)
    message = str(exc.value)
    assert frag in message
    # 面向群聊用户：带用法行与 /help 引导
    assert "用法：/" in message and "/help" in message


def test_wrong_spec_defensive():
    # 调用方拿错 spec（首 token 对不上命令名）也归为用法错误而非崩溃
    with pytest.raises(CommandUsageError):
        parse_invocation(ROLL, "/echo hi")
