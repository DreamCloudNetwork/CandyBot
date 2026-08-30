"""命令插件全链路：真实 CandyBot 编排 + 假 SnowLuma/AI + 独立注册表。

验证「/ 命中注册表 → 取消大模型自主回复 → 解析参数 → 调用插件 → 原样
发群 → 写回记忆」，以及未知命令回落、总开关关闭、各类失败提示。
沿用 test_integration 的 FakeSnowluma/FakeAI/group_event 模式。
"""

from __future__ import annotations

import asyncio

from candybot.ai import JudgeVerdict
from candybot.bot import CandyBot
from candybot.models import load_settings
from candybot.plugin_api import CommandParam, CommandRegistry
from tests.test_integration import (
    FakeAI,
    FakeSnowluma,
    group_event,
    wait_until,
)
from tests.test_models_settings import DictCfg


def make_settings(tmp_path, plugins_over: dict | None = None):
    cfg = {
        "bot": {"self_qq": 99, "data_dir": str(tmp_path / "data")},
        "groups": {
            "42": {
                "persona": "你是群里的测试机器人。",
                "proactivity_threshold": 6,
                "cooldown_seconds": 60,
            }
        },
        "groups_default": {"enabled": False, "persona": "默认人设"},
        "ai_backend": {"base_url": "https://api.example.com/v1", "api_key": "k"},
        "models": {"judge": "j-model", "reply": "r-model"},
        "generation": {},
        "multimodal": {},
        "rate_limit": {},
        "snowluma": {
            "endpoint": "http://10.0.0.5:3000/",
            "allow_private_endpoint": True,
        },
        "response_post_process": {"enabled": False},
        # 指向空的临时目录：与真实 plugins/ 隔离，用例各自手工注册命令
        "plugins": {"dir": str(tmp_path / "plug"), **(plugins_over or {})},
    }
    return load_settings(DictCfg(cfg))


async def build(tmp_path, *, plugins_over=None, score=2):
    """假依赖的 CandyBot + 独立注册表。score 是 FakeAI 的 judge 分（默认
    低分：未命中命令的消息走 judge 后保持沉默，便于单独断言走了大模型）。"""
    settings = make_settings(tmp_path, plugins_over)
    reg = CommandRegistry()
    bot = CandyBot(settings, registry=reg)
    bot._snowluma = FakeSnowluma()
    bot._ai = FakeAI(JudgeVerdict(score, "感兴趣"))
    return bot, reg


async def test_command_bypasses_llm_and_replies(tmp_path):
    bot, reg = await build(tmp_path)
    try:
        seen = []

        @reg.command("hi", help="打招呼")
        async def hi(ctx):
            seen.append(ctx)
            return "你好！"

        await bot._on_event(group_event(1, "/hi"))
        await wait_until(lambda: bot._snowluma.sent)
        assert bot._snowluma.sent == [(42, "你好！")]
        # 完全不走大模型：judge 与 reply 都没有一次调用
        assert bot._ai.judge_calls == 0
        assert bot._ai.reply_calls == []
        # ctx 携带调用者身份
        assert seen[0].group_id == 42 and seen[0].user_id == 1000
        assert seen[0].nickname == "用户1000" and seen[0].text == "/hi"
        # 命令消息与插件回复都进了记忆（缺省配置：照常送入模型上下文），
        # 且都带 is_command 标记
        memory = await bot._memory.get(42)
        records = memory.tail(5)
        assert records[0].text == "/hi" and not records[0].is_self
        assert records[0].is_command
        assert records[-1].text == "你好！" and records[-1].is_self
        assert records[-1].is_command
    finally:
        await bot.stop()


async def test_commands_excluded_from_model_context_when_disabled(tmp_path):
    """include_commands_in_history=false：命令消息与回复照常入库（带
    is_command 标记），但不进模型的历史上下文、不占 context_size 名额。"""
    bot, reg = await build(
        tmp_path,
        plugins_over={"include_commands_in_history": False},
        score=9,
    )
    try:
        @reg.command("hi", help="打招呼")
        def hi(ctx):
            return "你好！"

        await bot._on_event(group_event(1, "/hi"))
        await wait_until(lambda: bot._snowluma.sent)
        assert bot._snowluma.sent == [(42, "你好！")]
        # 照常写入记忆，且两者都带 is_command 标记
        memory = await bot._memory.get(42)
        records = memory.tail(5)
        assert [(r.text, r.is_self, r.is_command) for r in records] == [
            ("/hi", False, True),
            ("你好！", True, True),
        ]
        # 持久层同样保留标记（重启回放后仍可按配置过滤）
        rows = await bot._memory.db.load_recent(42, 10)
        assert [r.is_command for r in rows] == [True, True]
        # 之后的普通消息：命令消息不出现在模型历史里（recent_len 只算
        # 「你们在聊啥」这一条），L2 须知换成排除措辞
        await bot._on_event(group_event(2, "你们在聊啥"))
        await wait_until(lambda: len(bot._snowluma.sent) > 1)
        assert bot._ai.reply_calls[0]["recent_len"] == 1
        assert bot._ai.reply_calls[0]["recent_texts"] == ["你们在聊啥"]
        runtime_system = bot._ai.reply_calls[0]["runtime_system"]
        assert "【命令功能】" in runtime_system
        assert "都不会出现在你的聊天记录里" in runtime_system
    finally:
        await bot.stop()


async def test_unknown_command_not_marked_when_excluded(tmp_path):
    """未知 /命令 不算插件产生的消息：照常走大模型、不带 is_command 标记。"""
    bot, _reg = await build(
        tmp_path,
        plugins_over={"include_commands_in_history": False},
        score=9,
    )
    try:
        await bot._on_event(group_event(3, "/这命令不存在 说点什么"))
        await wait_until(lambda: bot._snowluma.sent)
        assert bot._ai.reply_calls[0]["recent_texts"] == [
            "/这命令不存在 说点什么"
        ]
        memory = await bot._memory.get(42)
        records = memory.tail(5)
        assert records[0].text == "/这命令不存在 说点什么" and not records[0].is_self
        assert not records[0].is_command
        assert records[-1].is_self and not records[-1].is_command
    finally:
        await bot.stop()


async def test_args_parsed_with_quotes_and_options(tmp_path):
    bot, reg = await build(tmp_path)
    try:
        @reg.command("echo", params=(
            CommandParam("text", nargs="+"),
            CommandParam("--upper,-u", store_true=True),
        ))
        def echo(ctx):
            out = " ".join(ctx.args["text"])
            return out.upper() if ctx.args["upper"] else out

        await bot._on_event(group_event(2, '/echo hello "带空格 的词" --upper'))
        await wait_until(lambda: bot._snowluma.sent)
        assert bot._snowluma.sent == [(42, "HELLO 带空格 的词")]
        assert bot._ai.judge_calls == 0
    finally:
        await bot.stop()


async def test_unknown_command_still_goes_to_llm(tmp_path):
    """未注册的 /命令 不作否决：照常走 judge 链路（行为与引入前一致）。"""
    bot, reg = await build(tmp_path, score=9)
    try:
        await bot._on_event(group_event(3, "/这命令不存在 说点什么"))
        await wait_until(lambda: bot._snowluma.sent)
        assert bot._ai.judge_calls == 1
        # 最终回复来自大模型链路
        assert bot._snowluma.sent == [(42, "哈哈确实")]
        # plugins.enabled 时 L2 注入了「命令功能」须知（judge 与 reply 共用）
        assert "【命令功能】" in bot._ai.reply_calls[0]["runtime_system"]
    finally:
        await bot.stop()


async def test_plugin_disabled_sends_slash_to_llm(tmp_path):
    """plugins.enabled=false：即使命令名在册也不拦截，全部交给大模型。"""
    bot, reg = await build(tmp_path, plugins_over={"enabled": False}, score=2)
    try:
        called = False

        @reg.command("hi")
        def hi(ctx):
            nonlocal called
            called = True
            return "不该出现"

        await bot._on_event(group_event(4, "/hi"))
        await asyncio.sleep(0.15)
        assert not called
        assert bot._snowluma.sent == []
        assert bot._ai.judge_calls == 1  # 低分 → 沉默，但走了大模型判定
    finally:
        await bot.stop()


async def test_usage_error_replies_hint_without_handler(tmp_path):
    bot, reg = await build(tmp_path)
    try:
        called = False

        @reg.command("add", params=(CommandParam("n", type="int"),))
        def add(ctx):
            nonlocal called
            called = True
            return str(int(ctx.args["n"]) + 1)

        await bot._on_event(group_event(5, "/add 不是数字"))
        await wait_until(lambda: bot._snowluma.sent)
        assert not called
        hint = bot._snowluma.sent[0][1]
        assert "参数有误" in hint and "用法：/add" in hint and "/help add" in hint
    finally:
        await bot.stop()


async def test_handler_crash_gets_generic_message(tmp_path):
    bot, reg = await build(tmp_path)
    try:
        @reg.command("boom")
        def boom(ctx):
            raise RuntimeError("插件内部炸了")

        await bot._on_event(group_event(6, "/boom"))
        await wait_until(lambda: bot._snowluma.sent)
        assert bot._snowluma.sent == [(42, "/boom 执行失败了（内部错误）。")]
        # 失败提示也写回记忆（群里确实说过）
        memory = await bot._memory.get(42)
        assert memory.last().is_self
    finally:
        await bot.stop()


async def test_handler_timeout(tmp_path):
    bot, reg = await build(tmp_path, plugins_over={"timeout_seconds": 1})
    try:
        @reg.command("slow")
        async def slow(ctx):
            await asyncio.sleep(5)
            return "太慢了"

        await bot._on_event(group_event(7, "/slow"))
        await wait_until(lambda: bot._snowluma.sent, timeout=4)
        assert bot._snowluma.sent == [(42, "/slow 执行超时，稍后再试吧。")]
    finally:
        await bot.stop()


async def test_none_result_sends_nothing(tmp_path):
    bot, reg = await build(tmp_path)
    try:
        @reg.command("silent")
        def silent(ctx):
            return None

        await bot._on_event(group_event(8, "/silent"))
        await asyncio.sleep(0.15)
        assert bot._snowluma.sent == []
        memory = await bot._memory.get(42)
        assert [r.text for r in memory.tail(5)] == ["/silent"]
    finally:
        await bot.stop()


async def test_illegal_return_type_ignored(tmp_path):
    """handler 乱写返回值：只记日志不发垃圾，也不许炸掉 worker。"""
    bot, reg = await build(tmp_path)
    try:
        @reg.command("weird")
        def weird(ctx):
            return 123

        await bot._on_event(group_event(9, "/weird"))
        await asyncio.sleep(0.15)
        assert bot._snowluma.sent == []
        # 后续消息处理不受影响
        await bot._on_event(group_event(10, "/weird"))
        await asyncio.sleep(0.15)
        assert bot._snowluma.sent == []
    finally:
        await bot.stop()


async def test_segment_result_sent_and_memory_placeholder(tmp_path):
    bot, reg = await build(tmp_path)
    try:
        segments = [
            {"type": "text", "data": {"text": "看图"}},
            {"type": "image", "data": {"file": "file:///tmp/x.png"}},
        ]

        @reg.command("pic")
        def pic(ctx):
            return segments

        await bot._on_event(group_event(11, "/pic"))
        await wait_until(lambda: bot._snowluma.sent)
        assert bot._snowluma.sent == [(42, segments)]
        memory = await bot._memory.get(42)
        assert memory.last().is_self and memory.last().text == "看图"

        @reg.command("purepic")
        def purepic(ctx):
            return [segments[1]]

        await bot._on_event(group_event(12, "/purepic"))
        await wait_until(lambda: len(bot._snowluma.sent) == 2)
        memory = await bot._memory.get(42)
        assert memory.last().text == "[命令消息]"
    finally:
        await bot.stop()


async def test_builtin_help_lists_commands(tmp_path):
    bot, reg = await build(tmp_path)
    try:
        @reg.command("alpha", help="甲命令")
        def alpha(ctx):
            return "a"

        await bot._on_event(group_event(13, "/help"))
        await wait_until(lambda: bot._snowluma.sent)
        listing = bot._snowluma.sent[0][1]
        assert "/alpha —— 甲命令" in listing and "/help" in listing

        bot._snowluma.sent.clear()
        await bot._on_event(group_event(14, "/help alpha"))
        await wait_until(lambda: bot._snowluma.sent)
        assert bot._snowluma.sent[0][1].startswith("/alpha")

        bot._snowluma.sent.clear()
        await bot._on_event(group_event(15, "/help no-such"))
        await wait_until(lambda: bot._snowluma.sent)
        assert "没有 /no-such 这个命令" in bot._snowluma.sent[0][1]
        assert bot._ai.judge_calls == 0
    finally:
        await bot.stop()


async def test_command_ordering_after_llm_reply(tmp_path):
    """同群串行队列保序：慢速大模型回复先发，命令输出排在它后面。"""
    bot, reg = await build(tmp_path, score=9)
    try:
        @reg.command("hi")
        async def hi(ctx):
            return "你好！"

        release = asyncio.Event()

        original = bot._ai.generate_reply

        async def slow_reply(*args, **kwargs):
            await release.wait()
            return await original(*args, **kwargs)

        bot._ai.generate_reply = slow_reply

        await bot._on_event(group_event(16, "聊个值得回复的话题"))
        await asyncio.sleep(0.05)  # worker 已取走第一条，正卡在生成里
        await bot._on_event(group_event(17, "/hi"))
        await asyncio.sleep(0.15)
        # 命令不会抢在大模型回复前面发出
        assert bot._snowluma.sent == []
        release.set()
        await wait_until(lambda: len(bot._snowluma.sent) == 2)
        assert bot._snowluma.sent[0] == (42, "哈哈确实")
        assert bot._snowluma.sent[1] == (42, "你好！")
    finally:
        await bot.stop()
