from __future__ import annotations

import pytest

from candybot.models import (
    GroupProfile,
    Settings,
    load_settings,
    validate_request_url,
)


def base_cfg(**over):
    cfg = {
        "bot": {"self_qq": 99},
        "groups": {},
        "groups_default": {
            "persona": "测试人设",
            "proactivity_threshold": 6,
            "cooldown_seconds": 60,
            "context_size": 10,
        },
        "ai_backend": {"base_url": "https://api.example.com/v1", "api_key": "k"},
        "models": {"judge": "j", "reply": "r"},
        "generation": {},
        "multimodal": {},
        "rate_limit": {},
        "snowluma": {
            "endpoint": "http://10.0.0.5:3000/",
            "api_key": "t",
            "mode": "write",
            "allow_private_endpoint": True,
        },
    }
    for section, values in over.items():
        cfg[section].update(values)
    return cfg


class DictCfg(dict):
    """模拟 ConfigClass 的按段名取属性（缺 key 抛 KeyError）的行为。"""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise KeyError(name) from exc

    def __getitem__(self, name):
        if name not in self:
            raise KeyError(name)
        return super().__getitem__(name)


def test_load_settings_full():
    s: Settings = load_settings(DictCfg(base_cfg()))
    assert s.bot.self_qq == 99
    assert s.groups_default.persona == "测试人设"
    assert s.snowluma.allow_private_endpoint is True
    assert s.snowluma.mode == "write"
    assert s.multimodal.mode == "placeholder"
    assert s.rate_limit.global_daily_limit is None


def test_profile_strict_whitelist_and_override():
    """groups 非空 → 严格白名单；条目覆盖默认值，其余继承。"""
    cfg = base_cfg()
    cfg["groups"] = {
        "111": {"persona": "专属人设", "proactivity_threshold": 3},
    }
    s = load_settings(DictCfg(cfg))

    assert s.profile_for(222) is None         # 白名单外拒绝（哪怕默认开启）

    p_over = s.profile_for(111)
    assert p_over.persona == "专属人设"        # 覆盖
    assert p_over.proactivity_threshold == 3
    assert p_over.cooldown_seconds == 60      # 哨兵 -1 继承默认


def test_whitelist_empty_allows_all_when_default_enabled():
    """groups 为空 + groups_default.enabled=True → 全量兜底模式。"""
    s = load_settings(DictCfg(base_cfg()))    # base_cfg 里 groups={} 且 enabled=True
    p = s.profile_for(88888)
    assert p is not None
    assert p.persona == "测试人设"


def test_whitelist_empty_and_default_disabled_blocks_everything():
    cfg = base_cfg()
    cfg["groups_default"] = dict(cfg["groups_default"], enabled=False)
    s = load_settings(DictCfg(cfg))
    assert s.profile_for(1) is None


def test_missing_section_raises():
    cfg = base_cfg()
    del cfg["models"]
    with pytest.raises(ValueError):
        load_settings(DictCfg(cfg))


def test_mode_read_rejected():
    cfg = base_cfg(snowluma={"mode": "read", "endpoint": "http://example.com/"})
    with pytest.raises(ValueError):
        load_settings(DictCfg(cfg))


def test_validate_request_url_rejects_private():
    validate_request_url("https://example.com/img.jpg")
    for bad in (
        "http://127.0.0.1/x",
        "http://localhost/x",
        "http://10.1.2.3/x",
        "http://192.168.1.1/x",
        "http://169.254.1.1/x",
        "ftp://example.com/x",
        "file:///etc/passwd",
    ):
        with pytest.raises(ValueError):
            validate_request_url(bad)


def test_endpoint_requires_explicit_private_optin():
    cfg = base_cfg()  # endpoint 是私网但 allow_private=True → OK
    load_settings(DictCfg(cfg))
    cfg["snowluma"]["allow_private_endpoint"] = False
    with pytest.raises(ValueError):
        load_settings(DictCfg(cfg))


def test_disabled_group_profile_returns_none():
    cfg = base_cfg(groups={"55": {"enabled": False}})
    s = load_settings(DictCfg(cfg))
    assert s.profile_for(55) is None
    assert s.profile_for(56) is None  # 白名单外同样拒绝


def test_log_level_default_and_validation():
    # 默认 INFO
    s = load_settings(DictCfg(base_cfg()))
    assert s.bot.log_level == "INFO"
    # 合法值大小写不敏感
    s2 = load_settings(DictCfg(base_cfg(bot={"log_level": "debug"})))
    assert s2.bot.log_level == "DEBUG"
    # 非法值启动即报错
    with pytest.raises(ValueError):
        load_settings(DictCfg(base_cfg(bot={"log_level": "VERBOSE"})))


def test_generation_emoji_defaults():
    """两个 emoji 键缺失时走默认值：概率 0.25，上限 2。"""
    s = load_settings(DictCfg(base_cfg()))
    assert s.generation.emoji_chance == 0.25
    assert s.generation.emoji_max == 2


def test_generation_emoji_boundary_values_accepted():
    cfg = base_cfg(generation={"emoji_chance": 0, "emoji_max": 0})
    s = load_settings(DictCfg(cfg))
    assert s.generation.emoji_chance == 0
    assert s.generation.emoji_max == 0
    s2 = load_settings(DictCfg(base_cfg(generation={"emoji_chance": 1})))
    assert s2.generation.emoji_chance == 1


def test_generation_emoji_validation():
    with pytest.raises(ValueError):
        load_settings(DictCfg(base_cfg(generation={"emoji_chance": 1.5})))
    with pytest.raises(ValueError):
        load_settings(DictCfg(base_cfg(generation={"emoji_chance": -0.1})))
    with pytest.raises(ValueError):
        load_settings(DictCfg(base_cfg(generation={"emoji_max": -1})))
