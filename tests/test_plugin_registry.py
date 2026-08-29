"""插件 SDK 单测：参数/命令声明校验、注册表去重、插件目录加载、用法生成。"""

from __future__ import annotations

import pytest

from candybot.plugin_api import (
    CommandContext,
    CommandParam,
    CommandRegistry,
    CommandSpec,
    build_registry,
    format_usage,
    load_plugin_dir,
)


def _spec(name="t", **over):
    return CommandSpec(name=name, handler=lambda ctx: None, **over)


def _ctx(reg: CommandRegistry, text: str = "", **args) -> CommandContext:
    return CommandContext(
        group_id=1,
        user_id=2,
        nickname="n",
        text=text,
        args=args,
        registry=reg,
        settings=None,
    )


# ------------------------------------------------------------ 声明校验


def test_param_kind_auto_detection():
    assert CommandParam("text").kind == "positional"
    assert CommandParam("--count").kind == "option"
    assert CommandParam("--count,-c").dest == "count"
    assert CommandParam("--count,-c").flags == ("--count", "-c")
    assert CommandParam("-n").dest == "n"
    # 显式 kind 与 name 矛盾要报错
    with pytest.raises(ValueError, match="矛盾"):
        CommandParam("--count", kind="positional")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "大写NAME"},
        {"name": "pos", "nargs": "?"},
        {"name": "--sw", "store_true": True, "type": "int"},
        {"name": "--sw", "nargs": "+"},
        {"name": "x", "type": "bool"},
        {"name": "x", "kind": "weird"},
    ],
)
def test_param_validation_at_construction(kwargs):
    # frozen dataclass 的 __post_init__ 在构造时即触发校验
    with pytest.raises(ValueError):
        CommandParam(**kwargs)


@pytest.mark.parametrize("bad_name", ["Echo", "含中文", "a b", "", "/x"])
def test_command_spec_name_validation(bad_name):
    with pytest.raises(ValueError):
        _spec(bad_name)


# ------------------------------------------------------------ 注册表


def test_registry_first_registration_wins():
    reg = CommandRegistry()
    first = reg.register(_spec("dup", plugin="p1"))
    second = reg.register(_spec("dup", plugin="p2"))
    assert reg.get("dup") is first
    assert second.plugin == "p1"  # register 返回的是先到者
    assert reg.names() == ["dup"]


def test_registry_decorator_and_lookup():
    reg = CommandRegistry()

    @reg.command("hi", params=(CommandParam("who"),), help="打招呼")
    def hi(ctx):
        return f"hi {ctx.args['who']}"

    assert reg.get("hi") is not None
    assert hi(_ctx(reg, who="x")) == "hi x"
    assert reg.get("nope") is None
    assert reg.specs()[0].name == "hi"


# ------------------------------------------------------------ 用法生成


def test_format_usage():
    spec = _spec(
        "roll",
        params=(
            CommandParam("times", type="int", default=1),
            CommandParam("text", nargs="+"),
            CommandParam("tail", nargs="*"),
            CommandParam("--sides", type="int", default=6),
            CommandParam("--verbose,-v", store_true=True),
        ),
    )
    assert (
        format_usage(spec)
        == "/roll <TIMES> <TEXT...> [<TAIL...>] [--sides SIDES] [--verbose]"
    )


# ------------------------------------------------------------ 插件目录加载

GOOD_PLUGIN = """
from candybot.plugin_api import command, CommandParam

@command("ping", params=(CommandParam("--loud", store_true=True),), help="pong")
def ping(ctx):
    return "PONG" if ctx.args.get("loud") else "pong"
"""

BROKEN_PLUGIN = "raise ImportError('故意写坏的插件')\n"


def _make_plugin_dir(tmp_path):
    d = tmp_path / "plug"
    d.mkdir()
    (d / "good.py").write_text(GOOD_PLUGIN, encoding="utf-8")
    (d / "_internal.py").write_text(BROKEN_PLUGIN, encoding="utf-8")  # 下划线开头跳过
    return d


def test_load_plugin_dir_skips_broken_and_underscore(tmp_path, caplog):
    d = _make_plugin_dir(tmp_path)
    (d / "bad.py").write_text(BROKEN_PLUGIN, encoding="utf-8")
    reg = CommandRegistry()
    loaded = load_plugin_dir(reg, d)
    assert loaded == ["good.py"]  # bad.py 导入失败只记日志、不抛
    spec = reg.get("ping")
    assert spec is not None and spec.plugin == "good"
    assert spec.handler(_ctx(reg, loud=True)) == "PONG"
    assert any("加载插件" in r.message or "失败" in r.message for r in caplog.records)


def test_load_plugin_dir_missing_dir_is_noop(tmp_path):
    assert load_plugin_dir(CommandRegistry(), tmp_path / "nonexistent") == []


def test_build_registry_scans_dir_but_disabled_skips(tmp_path):
    from candybot.models import PluginSettings

    class Snap:
        def __init__(self, plugins):
            self.plugins = plugins

    _make_plugin_dir(tmp_path)
    reg = build_registry(
        Snap(PluginSettings(enabled=True, dir=str(tmp_path / "plug"))),
        CommandRegistry(),
    )
    assert reg.get("ping") is not None and reg.get("help") is not None

    reg2 = build_registry(Snap(PluginSettings(enabled=False)), CommandRegistry())
    assert reg2.get("ping") is None and reg2.get("help") is not None
