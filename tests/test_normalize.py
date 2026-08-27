from __future__ import annotations

import asyncio

import pytest

from candybot.models import MultimodalSettings, NormalizedMessage
from candybot.normalize import normalize_group_message


def run(coro):
    return asyncio.run(coro)


def base_event(**over):
    event = {
        "post_type": "message",
        "message_type": "group",
        "group_id": 42,
        "user_id": 1000,
        "message_id": 1,
        "time": 1700000000,
        "sender": {"card": "小明", "nickname": "xiaoming"},
        "message": [{"type": "text", "data": {"text": "大家好"}}],
    }
    event.update(over)
    return event


MM = MultimodalSettings(mode="placeholder", download_media=False)


def norm(event, **kw):
    defaults = dict(self_qq=99, multimodal=MM, find_by_message_id=lambda _id: None)
    defaults.update(kw)
    return run(normalize_group_message(event, **defaults))


def test_basic_text():
    result = norm(base_event())
    assert isinstance(result, NormalizedMessage)
    assert result.record.text == "大家好"
    assert result.record.nickname == "小明"
    assert result.mentioned_me is False


def test_at_me_marks_mentioned():
    segs = [
        {"type": "at", "data": {"qq": "99"}},
        {"type": "text", "data": {"text": " 在吗"}},
    ]
    result = norm(base_event(message=segs))
    assert result.mentioned_me is True
    assert "@糖糖" in result.record.text


def test_at_other_no_mention():
    segs = [
        {"type": "at", "data": {"qq": "12345"}},
        {"type": "text", "data": {"text": " 你怎么看"}},
    ]
    result = norm(base_event(message=segs))
    assert result.mentioned_me is False
    assert "@QQ12345" in result.record.text


def test_reply_to_self_via_memory():
    class FakeMem:
        def find_by_message_id(self, mid):
            from candybot.models import ChatRecord

            return ChatRecord(1, 42, 99, "糖糖", "我之前说的话", 0.0, is_self=True)

    segs = [
        {"type": "reply", "data": {"id": 777}},
        {"type": "text", "data": {"text": "不同意"}},
    ]
    result = norm(base_event(message=segs), find_by_message_id=FakeMem().find_by_message_id)
    assert result.mentioned_me is True
    assert "[回复糖糖]" in result.record.text


def test_reply_to_other_shows_snippet():
    class FakeMem:
        def find_by_message_id(self, mid):
            from candybot.models import ChatRecord

            return ChatRecord(2, 42, 1234, "小红", "原始内容abc", 0.0)

    segs = [{"type": "reply", "data": {"id": 888}}, {"type": "text", "data": {"text": "+"}}]
    result = norm(base_event(message=segs), find_by_message_id=FakeMem().find_by_message_id)
    assert result.mentioned_me is False
    assert "[回复 小红(1234)：原始内容abc]" in result.record.text


def test_image_placeholder_mode():
    segs = [
        {"type": "image", "data": {"url": "https://example.com/a.jpg", "file": "a.jpg"}},
    ]
    result = norm(base_event(message=segs))
    assert result.record.text == "[图片]"
    assert result.record.images == ()


def test_image_direct_mode_requires_download(tmp_path):
    segs = [
        {"type": "image", "data": {"url": "http://127.0.0.1/x.jpg"}},  # 私网地址必被拒
    ]
    mm = MultimodalSettings(mode="direct", download_media=True)
    import aiohttp

    async def with_session():
        async with aiohttp.ClientSession() as s:
            return await normalize_group_message(
                base_event(message=segs),
                self_qq=99,
                multimodal=mm,
                find_by_message_id=lambda _i: None,
                http_session=s,
            )

    result = run(with_session())
    # 下载被 SSRF 校验拦截 → 降级为占位符
    assert result.record.images == ()
    assert "[图片]" in result.record.text


def test_non_group_ignored():
    assert norm(base_event(message_type="private")) is None


def test_meta_ignored():
    assert norm({"post_type": "meta_event", "meta_event_type": "heartbeat"}) is None


def test_self_echo_ignored():
    assert norm(base_event(user_id=99)) is None


def test_cq_string_fallback():
    result = norm(base_event(message="纯文本CQ码兼容"))
    assert result.record.text == "纯文本CQ码兼容"


def test_empty_message_returns_none():
    assert norm(base_event(message=[])) is None
