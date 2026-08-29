"""/help —— 列出全部已注册命令及用法。

注册表里的命令随时可能被别的插件扩充，所以用法文本在每次调用时按
ctx.registry 现算，不做构建期缓存。
"""

from __future__ import annotations

from candybot.plugin_api import CommandContext, CommandParam, CommandRegistry, CommandSpec, format_usage

HELP_PARAMS = (
    CommandParam("command", nargs="*", help="要查看用法的具体命令名"),
)


def _describe(spec: CommandSpec) -> str:
    lines = [format_usage(spec)]
    if spec.help:
        lines.append(f"  {spec.help}")
    for p in spec.params:
        flags = "/".join(p.flags) if p.kind == "option" else p.dest
        piece = f"{flags}"
        if p.store_true:
            piece += "（开关）"
        elif p.kind == "positional" and p.nargs:
            piece += "（一个或多个）" if p.nargs == "+" else "（零个或多个）"
        elif p.type != "str":
            piece += f"（{p.type}）"
        if p.help:
            piece += f"：{p.help}"
        lines.append(f"  {piece}")
    return "\n".join(lines)


def _handle(ctx: CommandContext) -> str:
    names: list[str] = ctx.args.get("command") or []
    if not names:
        specs = ctx.registry.specs()
        if not specs:
            return "当前没有可用命令。"
        lines = [f"/{s.name} —— {s.help}" if s.help else f"/{s.name}" for s in specs]
        return "可用命令：\n" + "\n".join(lines) + "\n用 /help <命令> 查看用法。"
    name = names[0]
    spec = ctx.registry.get(name)
    if spec is None:
        return f"没有 /{name} 这个命令，用 /help 查看可用命令。"
    return _describe(spec)


def register(registry: CommandRegistry) -> None:
    registry.register(
        CommandSpec(
            name="help",
            handler=_handle,
            params=HELP_PARAMS,
            help="列出可用命令；/help <命令> 查看某个命令的用法",
            plugin="builtin",
        )
    )
