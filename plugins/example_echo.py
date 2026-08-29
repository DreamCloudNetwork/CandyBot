"""示例命令插件：演示如何写一个 CandyBot 插件。

放进 plugins/ 目录即被自动加载（文件名以 `_` 开头的会被跳过；改完文件
需重启机器人生效）。命令以 / 开头在群里触发，例如：

    /echo 你好世界
    /echo --upper 你好 世界
    /roll 3
    /roll 100 --sides 6

handler 收到 CommandContext（群号、发送者、解析好的 args 等），返回
str 或 OneBot 段列表即原样发到群里，返回 None 则不发消息。
"""

from __future__ import annotations

import random

from candybot.plugin_api import CommandContext, CommandParam, command


@command(
    "echo",
    params=(
        CommandParam("text", nargs="+", help="要复读的文本（可用引号包带空格的词）"),
        CommandParam("--upper, -u", store_true=True, help="把文本转成大写"),
    ),
    help="复读机：把参数原样发回群里",
)
async def echo(ctx: CommandContext) -> str:
    words: list[str] = ctx.args["text"]
    out = " ".join(words)
    if ctx.args.get("upper"):
        out = out.upper()
    return out


@command(
    "roll",
    params=(
        CommandParam("times", type="int", default=1, help="骰子次数（缺省 1）"),
        CommandParam("--sides", type="int", default=6, help="骰子面数"),
    ),
    help="掷骰子：/roll [次数] [--sides 面数]",
)
def roll(ctx: CommandContext) -> str:
    times = min(int(ctx.args["times"]), 20)  # 同步 handler 也支持
    sides = max(int(ctx.args["sides"]), 2)
    results = [str(random.randint(1, sides)) for _ in range(times)]
    return f"🎲 {times} 次 {sides} 面骰：{'、'.join(results)}"
