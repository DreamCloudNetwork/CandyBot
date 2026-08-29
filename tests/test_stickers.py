"""任务 C（表情包最小版）：识别启发式、StickerStore 收集与上限替换、normalize 判定。

不发真实网络请求：下载环节直接 monkeypatch 掉，数据库用 tmp_path 真库。
"""

from __future__ import annotations

import base64
from types import SimpleNamespace

from candybot.ai import ImageAssessment
from candybot.database import CandyDatabase, image_fingerprint
from candybot.models import ChatRecord, MultimodalSettings, StickerSettings
from candybot.normalize import normalize_group_message
from candybot.stickers import (
    StickerStore,
    image_dimensions,
    is_small_image,
    is_sticker_by_summary,
    parse_data_url,
)
import candybot.normalize as norm_mod
from tests.deterministic_rng import SeededRng


def png_bytes(width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\x0dIHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x06\x00\x00\x00"
        + b"\x00" * 8
    )


def png_url(width: int = 64, height: int = 64) -> str:
    return "data:image/png;base64," + base64.b64encode(
        png_bytes(width, height)
    ).decode()


# ---------------------------------------------------------------- 尺寸启发式


def test_dimensions_png_gif_bmp_webp():
    assert image_dimensions(png_bytes(123, 45)) == (123, 45)
    gif = b"GIF89a" + (100).to_bytes(2, "little") + (80).to_bytes(2, "little") + b"\x00" * 8
    assert image_dimensions(gif) == (100, 80)
    bmp = (
        b"BM"
        + b"\x00" * 16
        + (60).to_bytes(4, "little", signed=True)
        + (-40).to_bytes(4, "little", signed=True)  # 顶部原点 BMP 的负高
    )
    assert image_dimensions(bmp) == (60, 40)
    webp = (
        b"RIFF"
        + (40).to_bytes(4, "little")
        + b"WEBP"
        + b"VP8X"
        + (10).to_bytes(4, "little")
        + b"\x00"
        + b"\x00\x00\x00"
        + (99).to_bytes(3, "little")
        + (49).to_bytes(3, "little")
    )
    assert image_dimensions(webp) == (100, 50)


def test_dimensions_jpeg_sof():
    jpeg = (
        b"\xff\xd8"
        + b"\xff\xe0"
        + (6).to_bytes(2, "big")
        + b"JFIF\x00\x01"
        + b"\xff\xc0"
        + (17).to_bytes(2, "big")
        + b"\x08"
        + (100).to_bytes(2, "big")
        + (200).to_bytes(2, "big")
        + b"\x03\x11\x00\x00"
    )
    assert image_dimensions(jpeg) == (200, 100)


def test_dimensions_unknown_returns_none():
    assert image_dimensions(b"not an image at all") is None


def test_is_small_image():
    assert is_small_image(png_url(64, 64)) is True
    assert is_small_image(png_url(512, 200)) is True  # 较长边恰好压线
    assert is_small_image(png_url(600, 200)) is False
    assert is_small_image("https://example.com/x.png") is False  # 非 data URL
    assert is_small_image("data:image/png;base64,!!!坏数据") is False
    assert is_small_image("data:image/png,非base64") is False


def test_parse_data_url():
    mime, data = parse_data_url(png_url(8, 8))
    assert mime == "image/png"
    assert data == png_bytes(8, 8)
    assert parse_data_url("http://example.com/a.png") is None


def test_is_sticker_by_summary():
    assert is_sticker_by_summary("一张柴犬表情包") is True
    assert is_sticker_by_summary("搞笑梗图，配字哈哈") is True
    assert is_sticker_by_summary("meme 风格的猫") is True
    assert is_sticker_by_summary("一张 GIF 动图") is True
    assert is_sticker_by_summary("代码截图，报错栈") is False
    assert is_sticker_by_summary(None) is False
    assert is_sticker_by_summary("") is False


# ---------------------------------------------------------------- normalize 判定


def _image_event():
    return {
        "post_type": "message",
        "message_type": "group",
        "group_id": 42,
        "user_id": 1000,
        "message_id": 1,
        "time": 1700000000,
        "sender": {"card": "小明"},
        "message": [{"type": "image", "data": {"url": "https://img.example.com/a.png"}}],
    }


async def _no_ref(_id):
    return None


async def _normalize(monkeypatch, mode, data_url, *, describe_image=None, assess_image=None):
    async def fake_download(session, url):
        return data_url

    monkeypatch.setattr(norm_mod, "_download_as_data_url", fake_download)
    return await normalize_group_message(
        _image_event(),
        self_qq=99,
        multimodal=MultimodalSettings(mode=mode, download_media=True),
        find_by_message_id=_no_ref,
        http_session=object(),
        describe_image=describe_image,
        assess_image=assess_image,
    )


async def test_placeholder_flag_by_size(monkeypatch):
    small = await _normalize(monkeypatch, "placeholder", png_url(64))
    assert small.record.images == (png_url(64),)  # base64 仍随记忆落盘
    assert small.sticker_flags == (True,)

    big = await _normalize(monkeypatch, "placeholder", png_url(1200, 900))
    assert big.sticker_flags == (False,)


async def test_describe_flag_by_summary_and_persists_image(monkeypatch):
    async def desc(_url):
        return "一张猫的表情包"

    res = await _normalize(monkeypatch, "describe", png_url(64), describe_image=desc)
    assert res.sticker_flags == (True,)
    assert res.record.images == (png_url(64),)  # describe 模式 base64 同样入库
    assert "[图片：一张猫的表情包" in res.record.text

    async def desc_shot(_url):
        return "一张代码截图"

    res2 = await _normalize(monkeypatch, "describe", png_url(64), describe_image=desc_shot)
    assert res2.sticker_flags == (False,)

    async def boom(_url):
        raise RuntimeError("vision 挂了")

    res3 = await _normalize(monkeypatch, "describe", png_url(64), describe_image=boom)
    assert res3.sticker_flags == (False,)  # 转述失败：不收集，消息本身照常入库


async def test_direct_flag_from_assessment(monkeypatch):
    async def assess(_url):
        return ImageAssessment(summary="狗梗图", keep_raw=False, is_sticker=True)

    res = await _normalize(monkeypatch, "direct", png_url(64), assess_image=assess)
    assert res.sticker_flags == (True,)

    async def assess_info(_url):
        return ImageAssessment(summary="报错截图", keep_raw=True)

    res2 = await _normalize(monkeypatch, "direct", png_url(64), assess_image=assess_info)
    assert res2.sticker_flags == (False,)


async def test_direct_without_vision_falls_back_to_size(monkeypatch):
    big = await _normalize(monkeypatch, "direct", png_url(1200, 800))
    assert big.sticker_flags == (False,)
    small = await _normalize(monkeypatch, "direct", png_url(64))
    assert small.sticker_flags == (True,)


async def test_no_download_media_no_flags(monkeypatch):
    mm_off = await normalize_group_message(
        _image_event(),
        self_qq=99,
        multimodal=MultimodalSettings(mode="placeholder", download_media=False),
        find_by_message_id=_no_ref,
        http_session=object(),
    )
    assert mm_off.sticker_flags == ()


# ---------------------------------------------------------------- StickerStore


def _record_with(url: str, *, ts: float, summary: str | None = None) -> ChatRecord:
    return ChatRecord(
        message_id=5,
        group_id=42,
        user_id=1,
        nickname="u",
        text="[图片]",
        ts=ts,
        images=(url,),
        image_summaries={0: summary} if summary else None,
    )


async def _make_store(tmp_path, **sticker_over):
    db = CandyDatabase(tmp_path / "candy.db")
    await db.create_tables()
    settings = SimpleNamespace(stickers=StickerSettings(**sticker_over))
    store = StickerStore(tmp_path / "stickers", db, lambda: settings)
    return store, db


async def test_collect_writes_file_and_row(tmp_path):
    store, db = await _make_store(tmp_path)
    url = png_url(64)
    rec = _record_with(url, ts=100.0, summary="猫表情包")
    assert await store.collect(rec, (True,)) == 1
    entries = await db.load_stickers(42)
    assert len(entries) == 1 and entries[0].summary == "猫表情包"
    path = store.absolute_path(entries[0])
    assert path.parent == tmp_path / "stickers" / "42"  # 按群分目录
    assert path.read_bytes() == png_bytes(64, 64)
    # 同群同图只收藏一次；False 槽位跳过
    assert await store.collect(rec, (True,)) == 0
    assert await store.collect(_record_with(png_url(65), ts=101.0), (False,)) == 0
    assert len(await db.load_stickers(42)) == 1
    await db.close()


async def test_collect_disabled_does_nothing(tmp_path):
    store, db = await _make_store(tmp_path, enabled=False)
    assert await store.collect(_record_with(png_url(64), ts=100.0), (True,)) == 0
    assert await db.load_stickers(42) == []
    await db.close()


async def test_cap_evicts_least_recently_used(tmp_path):
    """全局上限 2：收第三张时最久未使用的被替换，文件同步删除。"""
    store, db = await _make_store(tmp_path, max_count=2)
    urls = [png_url(10, 10), png_url(11, 11), png_url(12, 12)]
    for url, ts in zip(urls, (100.0, 200.0, 300.0)):
        await store.collect(_record_with(url, ts=ts), (True,))
    entries = await db.load_stickers(42)
    assert {e.sha256 for e in entries} == {
        image_fingerprint(u) for u in urls[1:]
    }
    gone = tmp_path / "stickers" / "42" / f"{image_fingerprint(urls[0])}.png"
    assert not gone.exists()  # 被淘汰的表情包文件已删除
    assert all(store.absolute_path(e).exists() for e in entries)
    await db.close()


async def test_used_sticker_survives_eviction(tmp_path):
    """被用过的（last_used 更新）不再是最久未使用，该轮到另一张被淘汰。"""
    store, db = await _make_store(tmp_path, max_count=2)
    a, b = png_url(10, 10), png_url(11, 11)
    await store.collect(_record_with(a, ts=100.0), (True,))
    await store.collect(_record_with(b, ts=200.0), (True,))
    entry_a = next(
        e for e in await db.load_stickers(42) if e.sha256 == image_fingerprint(a)
    )
    await store.mark_used(entry_a)  # A 被用过 → last_used 刷新到现在
    await store.collect(_record_with(png_url(12, 12), ts=300.0), (True,))
    shas = {e.sha256 for e in await db.load_stickers(42)}
    assert image_fingerprint(b) not in shas  # 从没再用过的 B 先走
    assert image_fingerprint(a) in shas
    assert not (
        tmp_path / "stickers" / "42" / f"{image_fingerprint(b)}.png"
    ).exists()
    await db.close()


async def test_pick_and_image_segment(tmp_path):
    store, _db = await _make_store(tmp_path)
    rng = SeededRng(7)
    assert await store.pick_for_send(42, rng) is None  # 收藏为空
    url = png_url(64)
    await store.collect(_record_with(url, ts=100.0), (True,))
    entry = await store.pick_for_send(42, rng)
    assert entry is not None
    segment = store.image_segment(entry)
    assert segment["type"] == "image"
    file_value = segment["data"]["file"]
    assert file_value.startswith("file://")
    assert entry.sha256 in file_value  # 内容指纹命名的绝对路径
    assert await store.pick_for_send(999, rng) is None  # 别群收藏不共享
    await _db.close()


async def test_mark_used_counts(tmp_path):
    store, db = await _make_store(tmp_path)
    await store.collect(_record_with(png_url(64), ts=100.0), (True,))
    (entry,) = await db.load_stickers(42)
    await store.mark_used(entry)
    (after,) = await db.load_stickers(42)
    assert after.use_count == 1
    await db.close()
