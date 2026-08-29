"""AI 味内容检测（任务 B）：在内容层拦住「太像机器人」的输出。

回复生成并经过现有清洗（_strip_noise、emoji 处理）之后过一轮规则检测，
命中即在 ai.AIClient.generate_reply 里把被拦截回复与原因附进 L4 重新生成
一次（与网络重试 _generate_with_retry 相互独立）。规则列表可配置：
正则字符串列表（generation.ai_flavor_rules，内置默认见 models.
AI_FLAVOR_RULES_DEFAULT），统一按 re.MULTILINE 编译，^ 类规则能命中任意
一行行首（markdown 残留常在多行正文的中间行）。

非法正则在 load_settings 装载配置时即拒绝；本模块不再重复校验，缓存
pattern → 已编译正则复用（规则集很小且稳定，无需封顶）。
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_COMPILED: dict[str, re.Pattern[str]] = {}

# 拦截原因里转述的命中片段长度上限（进日志也进 L4，不能太长）
_FRAGMENT_MAX_CHARS = 40


def detect_ai_flavor(text: str, rules: Sequence[str]) -> str | None:
    """按顺序检测文本，返回首个命中规则的拦截原因；无命中返回 None。

    原因带规则原文与命中片段，供 DEBUG/WARNING 日志和重生成约束文案
    共同引用，方便据日志调规则。
    """
    for pattern in rules:
        regex = _COMPILED.get(pattern)
        if regex is None:
            regex = re.compile(pattern, re.MULTILINE)
            _COMPILED[pattern] = regex
        match = regex.search(text)
        if match:
            fragment = " ".join(match.group(0).split())[:_FRAGMENT_MAX_CHARS]
            return f"命中规则 {pattern!r}（片段「{fragment}」）"
    return None
