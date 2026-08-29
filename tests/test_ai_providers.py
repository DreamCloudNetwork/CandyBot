"""多提供商 AIClient：按 (base_url, api_key) 复用客户端、按模型配置窗口与输出上限。"""

from __future__ import annotations

import pytest
from types import SimpleNamespace

import candybot.ai as ai_mod
from candybot.ai import AIClient
from candybot.models import ChatRecord, GenerationSettings, ModelConfig, ModelSettings


def _gen(**over) -> GenerationSettings:
    values = dict(
        reply_max_tokens=500,
        temperature=0.8,
        max_context_chars=8000,
        timeout_seconds=60,
    )
    values.update(over)
    return GenerationSettings(**values)


def _models(
    judge: ModelConfig | None = None,
    reply: ModelConfig | None = None,
    vision: ModelConfig | None = None,
) -> ModelSettings:
    return ModelSettings(
        judge=judge or ModelConfig("j-model", "https://judge.example.com/v1", "kj", None, None),
        reply=reply or ModelConfig("r-model", "https://reply.example.com/v1", "kr", None, None),
        vision=vision,
    )


def _record(text: str = "在吗") -> ChatRecord:
    return ChatRecord(message_id=1, group_id=2, user_id=3, nickname="群友", text=text, ts=0)


# ---------------------------------------------------------------- 历史预算


def test_history_chars_falls_back_to_global_limit():
    ai = AIClient(models=_models(), generation=_gen())
    assert ai._history_chars(_models().judge, 100) == 8000


def test_history_chars_window_budget():
    """预算 = 窗口 - 输出预留 - 指令层字符 - 固定开销，并与全局上限取小。"""
    windowed = ModelConfig("r", "https://b/v1", "k", 2000, 300)
    ai = AIClient(models=_models(reply=windowed), generation=_gen())
    # 2000 - 300(输出) - 100(prompt) - 128(格式开销) = 1472
    assert ai._history_chars(windowed, 100) == 1472
    # 窗口远大于全局字符上限 → 取全局上限
    huge = ModelConfig("r", "https://b/v1", "k", 10**9, 1)
    assert ai._history_chars(huge, 0) == 8000
    # 预算算出负数 → 夹到 0（历史层仍至少保留一条，由裁剪函数保证）
    tiny = ModelConfig("r", "https://b/v1", "k", 50, 30)
    assert ai._history_chars(tiny, 10) == 0


# ---------------------------------------------------------------- 客户端复用


def test_client_cache_shares_same_provider():
    ai = AIClient(models=_models(), generation=_gen())
    same_cfg = ModelConfig("j2", "https://judge.example.com/v1", "kj", None, None)
    assert ai._client_for(same_cfg) is ai._client_for(_models().judge)
    other = ModelConfig("x", "https://other.example.com/v1", "kj", None, None)
    assert ai._client_for(other) is not ai._client_for(_models().judge)


def test_empty_api_key_falls_back_to_env_then_placeholder(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ai = AIClient(models=_models(), generation=_gen())
    cfg = ModelConfig("m", "https://local.example.com/v1", "", None, None)
    assert ai._client_for(cfg).api_key == "EMPTY"
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    fresh = AIClient(models=_models(), generation=_gen())
    assert fresh._client_for(cfg).api_key == "from-env"


# ---------------------------------------------------------------- 调用参数


_JUDGE_JSON = '{"score": 8, "to_me": false, "reason": "测试"}'


def _content_msg(text: str) -> SimpleNamespace:
    """不支持工具调用的端点：只有正文。"""
    return SimpleNamespace(content=text, tool_calls=None)


def _tool_msg(name: str, arguments: str) -> SimpleNamespace:
    """强制工具调用路径：结论在 tool_calls 的参数里，正文为空。"""
    return SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))],
    )


def _install_fake(monkeypatch, responses: dict[str, object]) -> list:
    """替换 candybot.ai.AsyncOpenAI：记录构造与 create 参数，按 api_key 返回预设响应。

    值为字符串时按正文消息返回（回退解析路径）；为 SimpleNamespace 时
    直接作为 message 返回（工具调用路径）。
    """

    class _FakeAsyncOpenAI:
        instances: list = []

        def __init__(self, *, base_url=None, api_key=None):
            self.base_url = base_url
            self.api_key = api_key
            self.create_kwargs: dict | None = None
            value = responses.get(api_key or "", "")
            self._message = value if not isinstance(value, str) else _content_msg(value)
            outer = self

            class _Completions:
                @staticmethod
                async def create(**kwargs):
                    outer.create_kwargs = kwargs
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=outer._message)]
                    )

            self.chat = SimpleNamespace(completions=_Completions())
            _FakeAsyncOpenAI.instances.append(self)

    monkeypatch.setattr(ai_mod, "AsyncOpenAI", _FakeAsyncOpenAI)
    return _FakeAsyncOpenAI.instances


def _called_instances(instances) -> list:
    return [c for c in instances if c.create_kwargs is not None]


async def test_judge_routes_to_own_provider_with_default_max_tokens(monkeypatch):
    instances = _install_fake(monkeypatch, {"kj": _JUDGE_JSON})
    ai = AIClient(models=_models(), generation=_gen())
    current = _record()
    verdict = await ai.judge_interest("L1", "L2", [current], current, "now")
    assert verdict.score == 8
    (called,) = _called_instances(instances)
    assert called.base_url == "https://judge.example.com/v1"
    assert called.api_key == "kj"
    assert called.create_kwargs["model"] == "j-model"
    assert called.create_kwargs["max_tokens"] == 1000
    # 结构化结论一律通过强制工具调用提交
    assert called.create_kwargs["tools"][0]["function"]["name"] == "submit_judgment"
    assert called.create_kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_judgment"},
    }


async def test_forced_tool_choice_disabled_uses_auto(monkeypatch):
    """forced_tool_choice=False：仍携带 tools，但 tool_choice 用 auto（思考模式兼容）。"""
    instances = _install_fake(
        monkeypatch,
        {"kj": _tool_msg("submit_judgment", '{"score": 9, "to_me": true, "reason": "在等我"}')},
    )
    ai = AIClient(
        models=_models(
            judge=ModelConfig(
                "j",
                "https://judge.example.com/v1",
                "kj",
                None,
                None,
                forced_tool_choice=False,
            )
        ),
        generation=_gen(),
    )
    current = _record()
    verdict = await ai.judge_interest("L1", "L2", [current], current, "now")
    assert verdict.score == 9
    (called,) = _called_instances(instances)
    assert called.create_kwargs["tools"][0]["function"]["name"] == "submit_judgment"
    assert called.create_kwargs["tool_choice"] == "auto"
    assert ai.judge_tool_use is True  # 仍走工具协议，提示词契约不变


async def test_judge_prefers_tool_call_arguments(monkeypatch):
    instances = _install_fake(
        monkeypatch,
        {"kj": _tool_msg("submit_judgment", '{"score": 9, "to_me": true, "reason": "在等我"}')},
    )
    ai = AIClient(models=_models(), generation=_gen())
    current = _record()
    verdict = await ai.judge_interest("L1", "L2", [current], current, "now")
    assert verdict.score == 9
    assert verdict.to_me is True
    assert verdict.reason == "在等我"


async def test_judge_tool_args_missing_score_scores_zero(monkeypatch):
    instances = _install_fake(monkeypatch, {"kj": _tool_msg("submit_judgment", '{"reason": "x"}')})
    ai = AIClient(models=_models(), generation=_gen())
    current = _record()
    verdict = await ai.judge_interest("L1", "L2", [current], current, "now")
    assert verdict.score == 0


async def test_reply_routes_to_own_provider_with_fallback_max_tokens(monkeypatch):
    instances = _install_fake(monkeypatch, {"kr": "好"})
    ai = AIClient(models=_models(), generation=_gen())
    current = _record()
    reply = await ai.generate_reply("L1", "L2", [current], current, "now", forced=True)
    assert reply is not None and reply.text == "好"
    (called,) = _called_instances(instances)
    assert called.base_url == "https://reply.example.com/v1"
    assert called.api_key == "kr"
    assert called.create_kwargs["model"] == "r-model"
    assert called.create_kwargs["max_tokens"] == 500  # 未配置 → generation.reply_max_tokens
    assert called.create_kwargs["tools"][0]["function"]["name"] == "send_reply"
    assert called.create_kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "send_reply"},
    }


async def test_reply_reads_tool_call_text_and_image_ops(monkeypatch):
    instances = _install_fake(
        monkeypatch,
        {
            "kr": _tool_msg(
                "send_reply",
                '{"text": "哈哈", "drop_img": [12, "x"], "recall_img": [7]}',
            )
        },
    )
    ai = AIClient(models=_models(), generation=_gen())
    current = _record()
    reply = await ai.generate_reply("L1", "L2", [current], current, "now", forced=True)
    assert reply is not None and reply.text == "哈哈"
    assert [(o.action, o.message_id) for o in reply.ops] == [
        ("drop_img", 12),
        ("recall_img", 7),
    ]


async def test_reply_tool_text_strips_residual_markers(monkeypatch):
    """模型沿用旧习惯把标记写进 text 参数时也要剥除收编，绝不发进群里。"""
    _install_fake(
        monkeypatch,
        {"kr": _tool_msg("send_reply", '{"text": "好可爱\\n<drop_img 51>"}')},
    )
    ai = AIClient(models=_models(), generation=_gen())
    current = _record()
    reply = await ai.generate_reply("L1", "L2", [current], current, "now", forced=True)
    assert reply is not None and reply.text == "好可爱"
    assert [(o.action, o.message_id) for o in reply.ops] == [("drop_img", 51)]


async def test_reconsider_reply_keeps_full_history_and_uses_tool(monkeypatch):
    """重想调用：历史层保留到当下的全部记录（含插话与自己的连发片段），
    指令层转述腹稿——被打断的上下文必须完整交给模型。"""
    instances = _install_fake(
        monkeypatch,
        {"kr": _tool_msg("send_reply", '{"text": "行，那这句还是要说的"}')},
    )
    ai = AIClient(models=_models(), generation=_gen())
    mine = ChatRecord(1, 2, 99, "糖糖", "涨是涨了", 0, is_self=True)
    interrupt = _record("不是")
    reply = await ai.reconsider_reply(
        "L1", "L2", [mine, interrupt], "now",
        sent_segments=["涨是涨了"],
        pending_segments=["但跟别家比还是便宜得离谱"],
    )
    assert reply is not None and reply.text == "行，那这句还是要说的"
    (called,) = _called_instances(instances)
    messages = called.create_kwargs["messages"]
    assert {"role": "assistant", "content": "涨是涨了"} in messages
    # 插话没有被当「当前消息」剥出历史：它就是最后一回合
    assert messages[-2] == {"role": "user", "content": "群友(3)：不是"}
    assert "但跟别家比还是便宜得离谱" in messages[-1]["content"]
    assert called.create_kwargs["tools"][0]["function"]["name"] == "send_reply"


async def test_reconsider_reply_empty_text_means_abort_not_failure(monkeypatch):
    """模型明确选择「不发了」：返回空正文的 ReplyDraft，而非 None。"""
    _install_fake(monkeypatch, {"kr": _tool_msg("send_reply", '{"text": ""}')})
    ai = AIClient(models=_models(), generation=_gen())
    reply = await ai.reconsider_reply(
        "L1", "L2", [_record()], "now", sent_segments=[], pending_segments=["还没发的腹稿"]
    )
    assert reply is not None
    assert reply.text == ""


async def test_per_model_max_output_tokens_override(monkeypatch):
    instances = _install_fake(monkeypatch, {"kj": _JUDGE_JSON, "kr": "好"})
    ai = AIClient(
        models=_models(
            judge=ModelConfig("j", "https://judge.example.com/v1", "kj", None, 1234),
            reply=ModelConfig("r", "https://reply.example.com/v1", "kr", None, 800),
        ),
        generation=_gen(),
    )
    current = _record()
    await ai.judge_interest("L1", "L2", [current], current, "now")
    await ai.generate_reply("L1", "L2", [current], current, "now", forced=True)
    judge_call, reply_call = _called_instances(instances)
    assert judge_call.create_kwargs["max_tokens"] == 1234
    assert reply_call.create_kwargs["max_tokens"] == 800


async def test_vision_uses_own_provider_and_limit_override(monkeypatch):
    instances = _install_fake(monkeypatch, {"kv": "好"})
    vision = ModelConfig("v", "https://vision.example.com/v1", "kv", None, 999)
    ai = AIClient(models=_models(vision=vision), generation=_gen())
    assert await ai.describe_image("data:image/png;base64,QQ==") == "好"
    describe_kwargs = dict(ai._client_for(vision).create_kwargs)
    await ai.assess_image("data:image/png;base64,QQ==")
    assess_kwargs = dict(ai._client_for(vision).create_kwargs)
    # 两个调用共享同一提供商客户端（同 base_url/api_key 复用连接池）
    (client,) = _called_instances(instances)
    assert client.base_url == "https://vision.example.com/v1"
    assert client.api_key == "kv"
    assert describe_kwargs["model"] == "v"
    assert describe_kwargs["max_tokens"] == 999
    assert assess_kwargs["model"] == "v"
    assert assess_kwargs["max_tokens"] == 999
    # 转述保持纯文本；入库评估则强制工具调用
    assert "tools" not in describe_kwargs
    assert assess_kwargs["tools"][0]["function"]["name"] == "submit_assessment"


async def test_assess_reads_tool_call_arguments(monkeypatch):
    _install_fake(
        monkeypatch,
        {"kv": _tool_msg("submit_assessment", '{"summary": "猫", "keep": false}')},
    )
    vision = ModelConfig("v", "https://vision.example.com/v1", "kv", None, None)
    ai = AIClient(models=_models(vision=vision), generation=_gen())
    assessment = await ai.assess_image("data:image/png;base64,QQ==")
    assert assessment.summary == "猫"
    assert assessment.keep_raw is False


# ------------------------------------------------- 不支持工具调用的模型


async def test_tool_use_disabled_uses_text_contract(monkeypatch):
    """tool_use=False：不发 tools 参数，按旧约定从正文解析（提示词同套契约）。"""
    instances = _install_fake(monkeypatch, {"kj": _JUDGE_JSON, "kr": "好"})
    ai = AIClient(
        models=_models(
            judge=ModelConfig(
                "j", "https://judge.example.com/v1", "kj", None, None, tool_use=False
            ),
            reply=ModelConfig(
                "r", "https://reply.example.com/v1", "kr", None, None, tool_use=False
            ),
        ),
        generation=_gen(),
    )
    current = _record()
    verdict = await ai.judge_interest("L1", "L2", [current], current, "now")
    assert verdict.score == 8
    reply = await ai.generate_reply("L1", "L2", [current], current, "now", forced=True)
    assert reply is not None and reply.text == "好"
    judge_call, reply_call = _called_instances(instances)
    assert "tools" not in judge_call.create_kwargs
    assert "tool_choice" not in judge_call.create_kwargs
    assert "tools" not in reply_call.create_kwargs
    assert ai.judge_tool_use is False and ai.reply_tool_use is False


async def test_degrades_to_text_contract_on_tools_rejection(monkeypatch):
    """端点报 tools 相关错误：降级为纯文本协议并立即补发，本次判定不丢。"""
    calls: list[dict] = []

    class _RejectingOpenAI:
        def __init__(self, *, base_url=None, api_key=None):
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            calls.append(kwargs)
            if "tools" in kwargs:
                raise RuntimeError("This model does not support tools")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=_content_msg(_JUDGE_JSON))]
            )

    monkeypatch.setattr(ai_mod, "AsyncOpenAI", _RejectingOpenAI)
    ai = AIClient(models=_models(), generation=_gen())
    current = _record()
    verdict = await ai.judge_interest("L1", "L2", [current], current, "now")
    assert verdict.score == 8
    assert ai.judge_tool_use is False
    assert "tools" in calls[0] and "tools" not in calls[1]
    # 补发请求换用纯文本输出契约：不再要求调用工具，改为正文输出 JSON
    assert "submit_judgment" not in calls[1]["messages"][-1]["content"]
    assert "JSON" in calls[1]["messages"][-1]["content"]


async def test_reply_degrades_with_in_place_text_retry(monkeypatch):
    """reply 工具请求被拒：立即按纯文本契约补发，本轮回复不丢。"""
    calls: list[dict] = []

    class _RejectingOpenAI:
        def __init__(self, *, base_url=None, api_key=None):
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            calls.append(kwargs)
            if "tools" in kwargs:
                raise RuntimeError("This model does not support tools")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=_content_msg("好"))]
            )

    monkeypatch.setattr(ai_mod, "AsyncOpenAI", _RejectingOpenAI)
    ai = AIClient(models=_models(), generation=_gen())
    current = _record()
    reply = await ai.generate_reply("L1", "L2", [current], current, "now", forced=True)
    assert reply is not None and reply.text == "好"
    assert ai.reply_tool_use is False
    assert "tools" in calls[0] and "tools" not in calls[1]
    # 补发请求的 L4 指令换成纯文本契约
    assert "send_reply" not in calls[1]["messages"][-1]["content"]


async def test_assess_degrades_with_in_place_text_retry(monkeypatch):
    """入库评估工具请求被拒：立即按纯文本契约补发，本次评估不丢。"""
    calls: list[dict] = []

    class _RejectingOpenAI:
        def __init__(self, *, base_url=None, api_key=None):
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            calls.append(kwargs)
            if "tools" in kwargs:
                raise RuntimeError("This model does not support tools")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=_content_msg('{"summary": "猫", "keep": false}')
                    )
                ]
            )

    monkeypatch.setattr(ai_mod, "AsyncOpenAI", _RejectingOpenAI)
    vision = ModelConfig("v", "https://vision.example.com/v1", "kv", None, None)
    ai = AIClient(models=_models(vision=vision), generation=_gen())
    assessment = await ai.assess_image("data:image/png;base64,QQ==")
    assert assessment.summary == "猫"
    assert assessment.keep_raw is False
    assert ai._tools_on["vision"] is False
    assert "tools" in calls[0] and "tools" not in calls[1]


async def test_degrades_when_endpoint_ignores_tools(monkeypatch):
    """端点接受 tools 却只回正文：本次按回退解析成功，角色随后降级。"""
    calls: list[dict] = []

    class _IgnoringOpenAI:
        def __init__(self, *, base_url=None, api_key=None):
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=_content_msg(_JUDGE_JSON))]
            )

    monkeypatch.setattr(ai_mod, "AsyncOpenAI", _IgnoringOpenAI)
    ai = AIClient(models=_models(), generation=_gen())
    current = _record()
    verdict = await ai.judge_interest("L1", "L2", [current], current, "now")
    assert verdict.score == 8
    assert ai.judge_tool_use is False
    await ai.judge_interest("L1", "L2", [current], current, "now")
    assert "tools" in calls[0] and "tools" not in calls[1]


async def test_unrelated_error_does_not_degrade(monkeypatch):
    """与工具无关的调用异常（如网络错误）不应触发降级。"""

    class _FlakyOpenAI:
        def __init__(self, *, base_url=None, api_key=None):
            self.chat = SimpleNamespace(completions=self)

        async def create(self, **kwargs):
            raise RuntimeError("connection reset by peer")

    monkeypatch.setattr(ai_mod, "AsyncOpenAI", _FlakyOpenAI)
    ai = AIClient(models=_models(), generation=_gen())
    current = _record()
    with pytest.raises(RuntimeError):
        await ai.judge_interest("L1", "L2", [current], current, "now")
    assert ai.judge_tool_use is True
