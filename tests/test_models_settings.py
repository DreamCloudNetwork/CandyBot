from __future__ import annotations

import pytest

from candybot.models import (
    AI_FLAVOR_RULES_DEFAULT,
    CONTEXT_SIZE_DEFAULT,
    MULTIPLE_REPLY_STYLE_DEFAULT,
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
        "storage": {},
        "rate_limit": {},
        "snowluma": {
            "endpoint": "http://10.0.0.5:3000/",
            "api_key": "t",
            "mode": "write",
            "allow_private_endpoint": True,
        },
    }
    for section, values in over.items():
        cfg.setdefault(section, {}).update(values)
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


def test_guardrail_defaults_override_and_disable():
    """反插嘴护栏：缺省走内置默认值；单群可覆盖，显式 0 表示关闭。"""
    cfg = base_cfg(
        groups={
            "111": {"min_gap_messages": 5},
            "222": {"busy_rate_per_min": 0},
        }
    )
    s = load_settings(DictCfg(cfg))
    assert s.groups_default.min_gap_messages == 3     # 内置默认
    assert s.groups_default.busy_rate_per_min == 6
    assert s.profile_for(111).min_gap_messages == 5   # 单群覆盖生效
    assert s.profile_for(111).busy_rate_per_min == 6  # 未覆盖项继承默认
    assert s.profile_for(222).busy_rate_per_min == 0  # 显式关闭

    # 缺省键的哨兵值 -1 也应落到内置默认
    cfg2 = base_cfg(groups_default={"min_gap_messages": -1})
    s2 = load_settings(DictCfg(cfg2))
    assert s2.groups_default.min_gap_messages == 3


def test_context_size_builtin_default():
    """groups_default 缺省/哨兵 -1 的 context_size 落到内置默认，绝不静默清空历史。"""
    # 整个键不写
    cfg = base_cfg()
    del cfg["groups_default"]["context_size"]
    s = load_settings(DictCfg(cfg))
    assert s.groups_default.context_size == CONTEXT_SIZE_DEFAULT
    # 全量兜底模式（groups 为空）解析出的每群条数同样带上默认值
    assert s.profile_for(88888).context_size == CONTEXT_SIZE_DEFAULT

    # 显式写哨兵 -1 与不写等价
    cfg2 = base_cfg(groups_default={"context_size": -1})
    assert load_settings(DictCfg(cfg2)).groups_default.context_size == CONTEXT_SIZE_DEFAULT

    # 单群条目：哨兵继承 groups_default，显式值覆盖
    cfg3 = base_cfg(
        groups_default={"context_size": -1},
        groups={"111": {"context_size": 5}, "222": {}},
    )
    s3 = load_settings(DictCfg(cfg3))
    assert s3.profile_for(111).context_size == 5
    assert s3.profile_for(222).context_size == CONTEXT_SIZE_DEFAULT


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


def test_generation_recheck_defaults():
    """复核键缺失时走默认值：开关打开，触发下限 5。"""
    s = load_settings(DictCfg(base_cfg()))
    assert s.generation.recheck_enabled is True
    assert s.generation.recheck_min_score == 5


def test_generation_recheck_custom_values():
    cfg = base_cfg(generation={"recheck_enabled": False, "recheck_min_score": 3})
    s = load_settings(DictCfg(cfg))
    assert s.generation.recheck_enabled is False
    assert s.generation.recheck_min_score == 3


def test_generation_recheck_validation():
    with pytest.raises(ValueError):
        load_settings(DictCfg(base_cfg(generation={"recheck_min_score": -1})))
    with pytest.raises(ValueError):
        load_settings(DictCfg(base_cfg(generation={"recheck_min_score": 11})))
    with pytest.raises(ValueError):
        load_settings(DictCfg(base_cfg(generation={"recheck_enabled": "yes"})))


def test_generation_decision_layers_defaults():
    """新鲜度检查 / 观望 / 重复抑制：键缺失时全部走默认值且默认开启。"""
    s = load_settings(DictCfg(base_cfg()))
    g = s.generation
    assert g.freshness_check_enabled is True
    assert g.observe_band == 2
    assert g.observe_delay_seconds == 45.0
    assert g.repetition_guard_enabled is True


def test_generation_decision_layers_custom_values():
    cfg = base_cfg(
        generation={
            "freshness_check_enabled": False,
            "observe_band": 0,               # 0 = 关闭观望
            "observe_delay_seconds": 5.5,
            "repetition_guard_enabled": False,
        }
    )
    s = load_settings(DictCfg(cfg))
    g = s.generation
    assert g.freshness_check_enabled is False
    assert g.observe_band == 0
    assert g.observe_delay_seconds == 5.5
    assert g.repetition_guard_enabled is False


def test_generation_decision_layers_validation():
    for bad in (
        {"observe_band": -1},
        {"observe_band": 11},
        {"observe_band": "wide"},
        # 观望延时喂给 asyncio.sleep：inf/nan 会挂死或行为未定，负数非法
        {"observe_delay_seconds": float("inf")},
        {"observe_delay_seconds": float("nan")},
        {"observe_delay_seconds": -0.1},
        {"observe_delay_seconds": "45s"},
        {"freshness_check_enabled": "yes"},
        {"repetition_guard_enabled": 1},
    ):
        with pytest.raises(ValueError):
            load_settings(DictCfg(base_cfg(generation=bad)))


# ---------------------------------------------------------------- 存储


def test_storage_image_retention_defaults():
    """storage 段可整体省略，保留期默认 7 天。"""
    s = load_settings(DictCfg(base_cfg()))
    assert s.storage.image_retention_days == 7
    s2 = load_settings(DictCfg(base_cfg(storage={"image_retention_days": 30})))
    assert s2.storage.image_retention_days == 30


def test_storage_image_retention_validation():
    """保留期必须 ≥ 1，0 与负数一律拒绝。"""
    for bad in (0, -1, -100):
        with pytest.raises(ValueError):
            load_settings(DictCfg(base_cfg(storage={"image_retention_days": bad})))


# ---------------------------------------------------------------- 输出层后处理


def test_response_post_process_defaults_when_section_absent():
    """response_post_process 整段省略：enabled 默认开启、各率走默认值。"""
    s = load_settings(DictCfg(base_cfg()))
    pp = s.response_post_process
    assert pp.enabled is True
    assert pp.typing_speed == 1.0
    assert pp.max_split == 3
    assert pp.max_length == 120
    assert pp.keep_strong_punctuation is True
    assert pp.typo_error_rate == 0.05
    assert pp.typo_tone_error_rate == 0.3
    assert pp.typo_word_replace_rate == 0.2
    assert pp.typo_correction_probability == 0.5
    assert pp.typo_polyphone_mode == "word_reading"
    assert "呃呃" in pp.lazy_replies


def test_response_post_process_custom_values():
    cfg = base_cfg(
        response_post_process={
            "enabled": False,
            "typing_speed": 2.5,
            "max_split": 5,
            "max_length": 200,
            "keep_strong_punctuation": False,
            "typo_error_rate": 0,
            "typo_correction_probability": 1,
            "typo_polyphone_mode": "skip",
            "lazy_replies": ["哦"],
        }
    )
    s = load_settings(DictCfg(cfg))
    pp = s.response_post_process
    assert pp.enabled is False
    assert pp.typing_speed == 2.5
    assert pp.max_split == 5
    assert pp.max_length == 200
    assert pp.keep_strong_punctuation is False
    assert pp.typo_error_rate == 0
    assert pp.typo_correction_probability == 1
    assert pp.typo_polyphone_mode == "skip"
    assert pp.lazy_replies == ("哦",)


def test_response_post_process_validation():
    for bad in (
        {"enabled": "yes"},
        {"typing_speed": -0.1},
        {"max_split": 0},
        {"max_length": -5},
        {"typo_error_rate": 1.5},
        {"typo_tone_error_rate": -0.2},
        {"typo_word_replace_rate": 2},
        {"typo_correction_probability": 1.1},
        {"typo_polyphone_mode": "first"},
        {"typing_speed": []},
        # json5 能写出 Infinity / NaN 字面量：inf 会挂死连发的 sleep，
        # nan 会静默关闭延迟，都必须按非法值拒收
        {"typing_speed": float("inf")},
        {"typing_speed": float("-inf")},
        {"typing_speed": float("nan")},
        {"typo_error_rate": "abc"},
        {"lazy_replies": []},
        {"lazy_replies": ["  "]},
        {"lazy_replies": "呃呃"},
        {"lazy_replies": [1]},
    ):
        with pytest.raises(ValueError):
            load_settings(DictCfg(base_cfg(response_post_process=bad)))


def test_response_post_process_error_messages_name_the_key():
    """类型写错的数字项也要报出键名（而不是 float() 的原生报错）。"""
    with pytest.raises(ValueError, match="typing_speed"):
        load_settings(DictCfg(base_cfg(response_post_process={"typing_speed": []})))
    with pytest.raises(ValueError, match="typo_error_rate"):
        load_settings(DictCfg(base_cfg(response_post_process={"typo_error_rate": "abc"})))
    with pytest.raises(ValueError, match="typo_polyphone_mode"):
        load_settings(
            DictCfg(base_cfg(response_post_process={"typo_polyphone_mode": "first"}))
        )


# ---------------------------------------------------------------- 多提供商模型
def test_models_string_form_inherits_backend():
    """字符串写法 = 仅模型名，提供商继承 ai_backend（向后兼容）。"""
    s = load_settings(DictCfg(base_cfg(models={"judge": "j", "reply": "r"})))
    assert s.models.judge.model == "j"
    assert s.models.judge.base_url == "https://api.example.com/v1"
    assert s.models.judge.api_key == "k"
    assert s.models.judge.context_window is None
    assert s.models.judge.max_output_tokens is None
    assert s.models.reply.model == "r"
    assert s.models.vision is None


def test_models_object_form_per_provider():
    """对象写法可按模型覆盖 base_url / api_key 与限额。"""
    cfg = base_cfg(
        models={
            "judge": {
                "model": "glm-4-flash",
                "context_window": 8192,
                "max_output_tokens": 1000,
            },
            "reply": {
                "model": "deepseek-chat",
                "base_url": "https://api.deepseek.com/v1",
                "api_key": "kr",
                "context_window": 64000,
                "max_output_tokens": 800,
            },
            "vision": {"model": "glm-4v-flash", "api_key": "kv"},
        }
    )
    s = load_settings(DictCfg(cfg))
    assert s.models.judge.base_url == "https://api.example.com/v1"   # 未写 → 继承
    assert s.models.judge.context_window == 8192
    assert s.models.judge.max_output_tokens == 1000
    assert s.models.reply.base_url == "https://api.deepseek.com/v1"  # 覆盖
    assert s.models.reply.api_key == "kr"
    assert s.models.reply.context_window == 64000
    assert s.models.reply.max_output_tokens == 800
    assert s.models.vision.model == "glm-4v-flash"
    assert s.models.vision.api_key == "kv"                           # 单项覆盖，其余继承
    assert s.models.vision.base_url == "https://api.example.com/v1"


def test_model_tool_use_flag_parsing():
    """tool_use 默认 true；显式 false 表示该角色走纯文本协议；非布尔值拒绝。"""
    s = load_settings(DictCfg(base_cfg(models={"judge": "j", "reply": "r"})))
    assert s.models.judge.tool_use is True
    s = load_settings(
        DictCfg(
            base_cfg(
                models={"judge": {"model": "j", "tool_use": False}, "reply": "r"}
            )
        )
    )
    assert s.models.judge.tool_use is False
    with pytest.raises(ValueError):
        load_settings(
            DictCfg(
                base_cfg(
                    models={"judge": {"model": "j", "tool_use": "yes"}, "reply": "r"}
                )
            )
        )


def test_model_forced_tool_choice_flag_parsing():
    """forced_tool_choice 默认 true；显式 false 表示改用 auto 工具选择；非布尔值拒绝。"""
    s = load_settings(DictCfg(base_cfg(models={"judge": "j", "reply": "r"})))
    assert s.models.judge.forced_tool_choice is True
    s = load_settings(
        DictCfg(
            base_cfg(
                models={
                    "judge": {"model": "j", "forced_tool_choice": False},
                    "reply": "r",
                }
            )
        )
    )
    assert s.models.judge.forced_tool_choice is False
    with pytest.raises(ValueError):
        load_settings(
            DictCfg(
                base_cfg(
                    models={
                        "judge": {"model": "j", "forced_tool_choice": "no"},
                        "reply": "r",
                    }
                )
            )
        )


def test_models_without_ai_backend_section():
    """三个模型都自带提供商时，ai_backend 段可整个省略。"""
    cfg = base_cfg()
    del cfg["ai_backend"]
    cfg["models"] = {
        "judge": {"model": "j", "base_url": "https://j.example.com/v1", "api_key": "kj"},
        "reply": {"model": "r", "base_url": "https://r.example.com/v1", "api_key": "kr"},
    }
    s = load_settings(DictCfg(cfg))
    assert s.models.judge.base_url == "https://j.example.com/v1"
    assert s.models.reply.base_url == "https://r.example.com/v1"


def test_models_missing_base_url_rejected():
    """模型与其继承的 ai_backend 都没有 base_url 时启动即报错。"""
    cfg = base_cfg(ai_backend={"base_url": "", "api_key": ""})
    with pytest.raises(ValueError):
        load_settings(DictCfg(cfg))
    # api_key 可以为空（本地无密钥端点），base_url 不行
    ok = base_cfg(ai_backend={"base_url": "https://api.example.com/v1", "api_key": ""})
    s = load_settings(DictCfg(ok))
    assert s.models.reply.api_key == ""


def test_models_entry_validation():
    missing_judge = base_cfg()
    del missing_judge["models"]["judge"]
    with pytest.raises(ValueError):
        load_settings(DictCfg(missing_judge))
    with pytest.raises(ValueError):
        load_settings(DictCfg(base_cfg(models={"judge": "", "reply": "r"})))
    with pytest.raises(ValueError):
        load_settings(DictCfg(base_cfg(models={"judge": 123, "reply": "r"})))
    with pytest.raises(ValueError):
        load_settings(DictCfg(base_cfg(models={"judge": {"base_url": "https://x/v1"}})))


def test_models_limits_validation():
    """窗口/输出上限必须为正，且输出上限必须小于窗口。"""
    for bad in (
        {"model": "r", "context_window": 0},
        {"model": "r", "context_window": -100},
        {"model": "r", "max_output_tokens": 0},
        {"model": "r", "context_window": 100, "max_output_tokens": 100},
        {"model": "r", "context_window": 100, "max_output_tokens": 500},
    ):
        with pytest.raises(ValueError):
            load_settings(DictCfg(base_cfg(models={"reply": bad})))


# ---------------------------------------------------------------- 风格多样性与内容拦截


def test_style_flavor_sticker_defaults():
    """整段不写时全部落到带默认值的向后兼容配置。"""
    s = load_settings(DictCfg(base_cfg()))
    gen = s.generation
    assert gen.multiple_probability == 0.0            # 临时风格默认关闭
    assert gen.multiple_reply_style == MULTIPLE_REPLY_STYLE_DEFAULT  # 内置示例池
    assert len(gen.multiple_reply_style) >= 3
    assert all(isinstance(x, str) and x.strip() for x in gen.multiple_reply_style)
    assert gen.ai_flavor_rules == AI_FLAVOR_RULES_DEFAULT  # AI 味检测默认开
    assert gen.ai_flavor_retries == 1
    st = s.stickers
    assert st.enabled is True
    assert st.send_probability == 0.05
    assert st.max_count == 64


def test_style_flavor_sticker_custom_values():
    cfg = base_cfg(
        generation={
            "multiple_reply_style": [" 只回一个字 ", "用反问接话"],
            "multiple_probability": 0.3,
            "ai_flavor_rules": ["哈哈"],
            "ai_flavor_retries": 0,
        },
        stickers={"enabled": False, "send_probability": 0.5, "max_count": 10},
    )
    s = load_settings(DictCfg(cfg))
    assert s.generation.multiple_reply_style == ("只回一个字", "用反问接话")  # 自动 strip
    assert s.generation.multiple_probability == 0.3
    assert s.generation.ai_flavor_rules == ("哈哈",)
    assert s.generation.ai_flavor_retries == 0
    assert s.stickers.enabled is False
    assert s.stickers.send_probability == 0.5
    assert s.stickers.max_count == 10

    # 空列表是合法显式值：分别关闭风格注入与 AI 味检测
    cfg2 = base_cfg(generation={"multiple_reply_style": [], "ai_flavor_rules": []})
    s2 = load_settings(DictCfg(cfg2))
    assert s2.generation.multiple_reply_style == ()
    assert s2.generation.ai_flavor_rules == ()


def test_style_flavor_sticker_validation_errors():
    for bad_gen, frag in [
        ({"multiple_probability": 1.5}, "multiple_probability"),
        ({"multiple_probability": -0.1}, "multiple_probability"),
        ({"multiple_reply_style": "字符串不是列表"}, "multiple_reply_style"),
        ({"multiple_reply_style": ["ok", "  "]}, "multiple_reply_style"),
        ({"ai_flavor_rules": ["("]}, "ai_flavor_rules"),
        ({"ai_flavor_rules": [42]}, "ai_flavor_rules"),
        ({"ai_flavor_retries": -1}, "ai_flavor_retries"),
    ]:
        with pytest.raises(ValueError, match=frag):
            load_settings(DictCfg(base_cfg(generation=bad_gen)))
    for bad_st, frag in [
        ({"send_probability": 2.0}, "send_probability"),
        ({"max_count": 0}, "max_count"),
        ({"max_count": -5}, "max_count"),
        ({"enabled": "yes"}, "stickers.enabled"),
    ]:
        with pytest.raises(ValueError, match=frag):
            load_settings(DictCfg(base_cfg(stickers=bad_st)))
