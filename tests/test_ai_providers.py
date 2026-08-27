"""多提供商 AIClient：按 (base_url, api_key) 复用客户端、按模型配置窗口与输出上限。"""

from __future__ import annotations

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


def _install_fake(monkeypatch, content_by_key: dict[str, str]) -> list:
    """替换 candybot.ai.AsyncOpenAI：记录构造与 create 参数，按 api_key 决定返回文本。"""

    class _FakeAsyncOpenAI:
        instances: list = []

        def __init__(self, *, base_url=None, api_key=None):
            self.base_url = base_url
            self.api_key = api_key
            self.create_kwargs: dict | None = None
            self._content = content_by_key.get(api_key or "", "")
            outer = self

            class _Completions:
                @staticmethod
                async def create(**kwargs):
                    outer.create_kwargs = kwargs
                    return SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
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


async def test_reply_routes_to_own_provider_with_fallback_max_tokens(monkeypatch):
    instances = _install_fake(monkeypatch, {"kr": "好"})
    ai = AIClient(models=_models(), generation=_gen())
    current = _record()
    assert await ai.generate_reply("L1", "L2", [current], current, "now", forced=True) == "好"
    (called,) = _called_instances(instances)
    assert called.base_url == "https://reply.example.com/v1"
    assert called.api_key == "kr"
    assert called.create_kwargs["model"] == "r-model"
    assert called.create_kwargs["max_tokens"] == 500  # 未配置 → generation.reply_max_tokens


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
