from __future__ import annotations

from candybot.ai import format_messages_for_log


def test_text_messages_rendered_with_roles():
    messages = [
        {"role": "system", "content": "L1 静态层"},
        {"role": "user", "content": "你好"},
    ]
    text = format_messages_for_log(messages)
    assert "──[0] system──" in text
    assert "──[1] user──" in text
    assert "L1 静态层" in text and "你好" in text


def test_image_base64_not_dumped_into_log():
    """direct 多模态的 base64 大块只显示长度与头部，不得整块进日志。"""
    big_b64 = "A" * 20000
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "看看这张图"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{big_b64}"}},
            ],
        }
    ]
    text = format_messages_for_log(messages)
    assert big_b64 not in text                      # 完整 base64 绝不出现
    assert "共 " + str(len(f"data:image/png;base64,{big_b64}")) + " 字符" in text
    assert "data:image/png;base64,AAAA" in text     # 只保留头部片段
    assert "看看这张图" in text
