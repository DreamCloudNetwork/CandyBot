"""每群上下文记忆：内存 deque + JSONL 追加落盘，重启后恢复最近记录。

文件名只允许 ``<纯数字群号>.jsonl``，全部经由 :func:`_safe_memory_file`
构造：正则校验 + 解析后目录包含校验，杜绝外部路径成分进入文件系统。
压缩时先删除旧文件、再以追加模式重放内存队列，全程不出现可变写路径。
图片去重：同一张图（整条 data URL 的 SHA-256 相同）在本群文件里只有
最早一次落盘保存原始 base64，其余位置一律写 ``ref:sha256:<指纹>``
引用；加载按行序重放解析，压缩/删除触发的重写会重新锚定最早幸存的副本。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import deque
from pathlib import Path
from typing import Any

from .models import (
    IMAGE_STATE_PLACEHOLDER,
    IMAGE_STATE_SHOW,
    IMAGE_STATE_SUMMARIZED,
    ChatRecord,
)

logger = logging.getLogger(__name__)

_COMPACT_THRESHOLD = 2000  # 追加次数超过该值时触发压缩

_NAME_RE = re.compile(r"^-?\d+\.jsonl$")

# 去重引用前缀：指向本文件中更早一行里的同一张图
_IMAGE_REF_PREFIX = "ref:sha256:"


def _safe_memory_file(root: Path, filename: str) -> Path:
    """校验并解析存储根目录下的记忆文件名，返回绝对路径。"""
    if not _NAME_RE.fullmatch(filename):
        raise ValueError(f"非法的记忆文件名：{filename!r}")
    path = (root / filename).resolve()
    if path.parent != root:
        raise ValueError(f"记忆文件越出存储根目录：{filename!r}")
    return path


def _image_fingerprint(data_url: str) -> str:
    return hashlib.sha256(data_url.encode("utf-8")).hexdigest()


def _dedupe_images(images: Any, known: dict[str, str]) -> list[str]:
    """把图片序列写成落盘形式：首次出现存原图，重复出现存引用。"""
    out: list[str] = []
    for url in images:
        fingerprint = _image_fingerprint(url)
        if fingerprint in known:
            out.append(_IMAGE_REF_PREFIX + fingerprint)
        else:
            known[fingerprint] = url
            out.append(url)
    return out


def _resolve_image_refs(obj: dict[str, Any], known: dict[str, str]) -> None:
    """就地解析行内图片引用，把 obj 还原成完整的记录 JSON。

    引用无法解析（如文件被截断）时丢弃该图及对应的状态与总结，
    避免下标错位；full 与 resolved 图片都会进入指纹表供后续行引用。
    """
    images = obj.get("images")
    if not isinstance(images, list):
        return
    states_raw = obj.get("image_states")
    states = states_raw if isinstance(states_raw, list) else []
    sums_raw = obj.get("image_summaries")
    sums = sums_raw if isinstance(sums_raw, dict) else {}
    new_images: list[str] = []
    new_states: list[Any] = []
    new_sums: dict[str, str] = {}
    for index, value in enumerate(images):
        if isinstance(value, str) and value.startswith(_IMAGE_REF_PREFIX):
            value = known.get(value[len(_IMAGE_REF_PREFIX):], "")
        if not (isinstance(value, str) and value.startswith("data:")):
            logger.warning("跳过 %s 中一张无法解析的图片引用", obj.get("message_id"))
            continue
        known.setdefault(_image_fingerprint(value), value)
        position = len(new_images)
        new_images.append(value)
        if index < len(states):
            new_states.append(states[index])
        summary = sums.get(str(index))
        if isinstance(summary, str):
            new_sums[str(position)] = summary
    obj["images"] = new_images
    obj["image_states"] = new_states
    obj["image_summaries"] = new_sums or None


class GroupMemory:
    """单个群的记忆（有界队列 + 对应 JSONL 文件）。"""

    def __init__(self, group_id: int, root: Path, capacity: int):
        self.group_id = group_id
        self.root = root
        self.path = _safe_memory_file(root, f"{int(group_id)}.jsonl")
        self.capacity = max(capacity, 8)
        self._records: deque[ChatRecord] = deque(maxlen=self.capacity)
        self._appends_since_compact = 0
        # 指纹 → 原始 data URL：记录「当前文件里哪张图是原图锚点」
        self._img_map: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        loaded: list[ChatRecord] = []
        seen: set[int] = set()
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        _resolve_image_refs(obj, self._img_map)
                        record = ChatRecord.from_json(obj)
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        logger.warning("跳过 %s 中一条损坏的记录", self.path)
                        continue
                    if record.message_id in seen:
                        continue
                    seen.add(record.message_id)
                    loaded.append(record)
        except OSError as exc:
            logger.warning("读取记忆文件失败 %s：%s", self.path, exc)
            return
        for record in loaded[-self.capacity :]:
            self._records.append(record)
        if len(loaded) > len(self._records):
            self._rewrite()  # 启动时顺手把旧文件压到容量以内

    def _record_to_json(self, record: ChatRecord) -> dict[str, Any]:
        """序列化并按指纹去重：重复图片只写引用，首次出现的存原图。"""
        obj = record.to_json()
        if obj.get("images"):
            obj["images"] = _dedupe_images(obj["images"], self._img_map)
        return obj

    def append(self, record: ChatRecord) -> None:
        self._records.append(record)
        line = json.dumps(self._record_to_json(record), ensure_ascii=False)
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as exc:
            logger.error("写入记忆文件失败 %s：%s", self.path, exc)
            return
        self._appends_since_compact += 1
        if (
            self._records.maxlen is not None
            and len(self._records) == self._records.maxlen
            and self._appends_since_compact >= _COMPACT_THRESHOLD
        ):
            self._rewrite()

    def _rewrite(self) -> None:
        """压缩：删除旧文件后把内存队列原样追加回全新文件（语义等同重写）。

        重写同时重建图片锚点表——原图挂靠到幸存记录中最早的那条，
        被撤回的锚点记录不会留下悬空引用。
        """
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.error("删除旧记忆文件失败 %s：%s", self.path, exc)
            return
        self._appends_since_compact = 0
        self._img_map = {}
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                for record in self._records:
                    f.write(
                        json.dumps(self._record_to_json(record), ensure_ascii=False)
                        + "\n"
                    )
        except OSError as exc:
            logger.error("压缩记忆文件失败 %s：%s", self.path, exc)

    def tail(self, n: int) -> list[ChatRecord]:
        """返回最近 n 条的不可变快照（时间正序）。"""
        if n <= 0 or not self._records:
            return []
        return list(self._records)[-n:]

    def tail_excluding_last(self, n: int) -> list[ChatRecord]:
        """返回除最后一条外的最近 n 条；用于构造提示词历史（当前消息单独走指令层）。"""
        if len(self._records) <= 1 or n <= 0:
            return []
        return list(self._records)[-n - 1 : -1]

    def last(self) -> ChatRecord | None:
        return self._records[-1] if self._records else None

    def find_by_message_id(self, message_id: int) -> ChatRecord | None:
        for record in reversed(self._records):
            if record.message_id == message_id:
                return record
        return None

    def remove(self, message_id: int) -> bool:
        """按 message_id 删除记录（内存与落盘同步），返回是否真的删了。

        JSONL 是纯追加的，删除只能整文件重放；撤回是低频操作，可接受。
        """
        if self.find_by_message_id(message_id) is None:
            return False
        self._records = deque(
            (r for r in self._records if r.message_id != message_id),
            maxlen=self.capacity,
        )
        self._rewrite()
        return True

    def transition_images(self, message_id: int, direction: str) -> bool:
        """按模型标记整体切换一条消息里全部图片的展示形态。

        direction="drop"：把仍在展示原图（show）的图片降级为总结，
        没有总结的退化为占位符——「总结（或替换为占位符）并不再展示原图」；
        direction="recall"：把该消息任意形态的图片重新升回原图展示。
        返回是否有实际变更；有则重写落盘，保证重启后状态不回退。
        """
        record = self.find_by_message_id(message_id)
        if record is None or not record.images:
            return False
        changed = False
        for index in range(len(record.images)):
            current = record.state_of(index)
            if direction == "recall":
                new_state = IMAGE_STATE_SHOW
            elif direction == "drop":
                if current != IMAGE_STATE_SHOW:
                    continue
                new_state = (
                    IMAGE_STATE_SUMMARIZED
                    if record.summary_of(index)
                    else IMAGE_STATE_PLACEHOLDER
                )
            else:
                raise ValueError(f"未知的图片状态切换方向：{direction!r}")
            if new_state != current:
                record.set_image_state(index, new_state)
                changed = True
        if changed:
            self._rewrite()
        return changed

    def __len__(self) -> int:
        return len(self._records)


class MemoryManager:
    """管理所有群的记忆文件（<data_dir>/memory/<group_id>.jsonl）。"""

    def __init__(self, data_dir: str | Path, default_capacity: int = 64):
        self.root = (Path(data_dir) / "memory").resolve()
        self.default_capacity = default_capacity
        self.root.mkdir(parents=True, exist_ok=True)
        self._groups: dict[int, GroupMemory] = {}

    def get(self, group_id: int) -> GroupMemory:
        gid = int(group_id)
        memory = self._groups.get(gid)
        if memory is None:
            memory = GroupMemory(gid, self.root, self.default_capacity)
            self._groups[gid] = memory
        return memory
