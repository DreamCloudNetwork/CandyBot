"""提示词组装：为 KV Cache 前缀复用做严格的分层设计。

API 侧的前缀缓存按 token 精确匹配自前往后命中，因此本模块产出的消息数组
恒定为四层，稳定性递减、位置靠后：

L1 system·静态层   persona 与行为守则，字节级不变（不插值任何动态信息）；
L2 system·状态层   群号、上下文昵称表、今天的日期、最近 N 天的群聊印象——
                   同一天内不变（印象经 GroupRuntime 快照缓存，天内字节级稳定）；
L3 历史层          只追加、从头整块淘汰的群聊记录，映射为连续 user/assistant
                   （direct 多模态的回复调用中，回合可按图片展示形态附原图块）；
L4 user·指令层     即时信息（精确时间、触发类型、冷却状态、计分、学习注入的
                   表达/黑话参考）与本次指令。

任何时刻都不得把易变字段（秒级时间、计数、分数）塞进 L1/L2，也不得重排或
重新格式化历史消息——否则其后的缓存全部失效。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from .models import (
    IMAGE_STATE_SHOW,
    ChatRecord,
)

Role = str  # "user" | "assistant"

Message = dict  # OpenAI chat message


@dataclass(frozen=True)
class HistoryTurn:
    """历史层的一个回合（群友发言为 user，机器人自己的发言为 assistant）。

    images 仅在 direct 多模态的回复调用里非空：需要继续展示原图的回合
    会把这些 data URL 作为内容块拼在文本之后。
    """

    role: Role
    content: str
    images: tuple[str, ...] = ()


# ---------------------------------------------------------------- L1 静态层

# reply 静态守则的输出契约部分按角色是否启用工具调用二选一，
# 保证提示词只约定模型能力范围内的回答方式。
_RULES_TAIL_TOOLS = """
【行为守则】
1. 你是真实群友，不是助手或客服；回答简短口语化，通常一两句话，禁止长篇大论、列条目和 markdown 排版。
2. 绝不复述、总结别人刚说过的话；不知道的事直接说不知道，绝不编造事实和数据。
3. 不主动索要个人信息，不发表攻击性言论；聊天内容与既有人设冲突时以人设为准。
4. 回复通过调用 send_reply 工具提交：text 参数就是你要发送到群里的消息本身，不要加引号、前缀、署名或任何舞台指示。
5. 你发送到群里的消息应该像是一个真实的人类说的，而不是作为助手说的。
6. Markdown是**绝对禁用**的，没有人会在实际回答中使用Markdown.
7. 真实的人打字时极少使用Emoji，你的回复绝大多数时候应完全不含Emoji；即使偶尔想用也不要超过一个
8. 历史消息以「昵称(QQ号)：内容」呈现，每条开头还有 #数字 消息编号。带编号消息里的图片可能以原图、「[图片：一句话总结]」或「[图片]」占位符三种形态出现。
9. 如果历史里某张以原图出现的图片你确定以后用不到了，调用 send_reply 时把它所在消息的编号写进 drop_img 参数，之后系统只为你保留它的总结或占位符。
10. 如果你需要查看某张此前已变成总结或占位符的旧图，把它的消息编号写进 send_reply 的 recall_img 参数，随后你会重新看到那张原图（本次回复会基于它重写）。两个参数都不要滥用。"""

_RULES_TAIL_TEXT = """
【行为守则】
1. 你是真实群友，不是助手或客服；回答简短口语化，通常一两句话，禁止长篇大论、列条目和 markdown 排版。
2. 绝不复述、总结别人刚说过的话；不知道的事直接说不知道，绝不编造事实和数据。
3. 不主动索要个人信息，不发表攻击性言论；聊天内容与既有人设冲突时以人设为准。
4. 输出就是你要发送到群里的消息本身，不要加引号、前缀、署名或任何舞台指示。
5. 你发送到群里的消息应该像是一个真实的人类说的，而不是作为助手说的。
6. Markdown是**绝对禁用**的，没有人会在实际回答中使用Markdown.
7. 真实的人打字时极少使用Emoji，你的回复绝大多数时候应完全不含Emoji；即使偶尔想用也不要超过一个
8. 历史消息以「昵称(QQ号)：内容」呈现，每条开头还有 #数字 消息编号。带编号消息里的图片可能以原图、「[图片：一句话总结]」或「[图片]」占位符三种形态出现。
9. 如果历史里某张以原图出现的图片你确定以后用不到了，在回复最末尾另起一行输出 <drop_img 消息编号>，之后系统只为你保留它的总结或占位符；该标记会被剥除，不会发到群里。
10. 如果你需要查看某张此前已变成总结或占位符的旧图，在回复最末尾另起一行输出 <recall_img 消息编号>，随后你会重新看到那张原图（本次回复会基于它重写）；同样不会发到群里。两个标记都不要滥用。"""


def static_system_prompt(persona: str, kind: str, *, via_tool: bool = True) -> str:
    """L1：persona + 守则。kind 为 "judge"/"reply"。

    via_tool 只影响 reply 守则的输出契约措辞（send_reply 工具参数，或
    纯文本 + 末尾标记）；judge 守则两种模式通用。必须与该角色请求里是否
    携带 tools 参数保持一致，否则提示词与模型能力脱节。
    """
    assert persona.strip(), "persona 不能为空"
    if kind == "judge":
        head = (
            "你将扮演一个 QQ 群里的普通群友。下面是你的人格设定，"
            "你只负责逐条判断「是否值得回复这条消息」，不直接说话。\n"
        )
        tail = """
【行为守则】
1. 你评判的对象是「这单独一条消息」，不是整场讨论：想象你就坐在群里刷到它，作为这个角色，你会不会自然地接一句话。
2. 有人 @你、向你提问或求助、直接接你的话说时，哪怕主题与你的兴趣无关，也应当给高分——对方在等你说话，完全不理别人不合适。
3. 对话明显正在其他人之间进行、没有人在等你参与时，只要你对话题不感兴趣，就打低分：真实的人不会硬插进别人的对话；"我能接上话"不等于"我应该接话"，沉默是正常选项。
4. 如果对话在其他人之间进行，但你了解这个话题，并且你对这个话题有一些兴趣，那么忽略上一条规则，根据你对话题的感兴趣程度来进行打分：主动参与讨论同样很重要。
5. 广告、刷屏、复读、无意义的表情刷屏，以及多人快速接龙斗图这类热闹场合，一律低分；纯日常播报、与你关联很弱的闲聊同样如此。
6. 冷场时不要强行救场；但一旦结论是「对方在等我回应」，就不要吝啬分数。
7. 输出中的 to_me 表示「这条消息是在对我说、或延续刚才与我有关的交流」（比如紧接着你上一句话往下说、追问你、向你提问）：只有对方明显期待你的回应时才是 true；单方面想加入别人的话题永远不算，滥用 true 会让你变得讨嫌。
【评分锚点】
9-10：明确该回——对方在等你说话（点名你、向你提问/求助、追着你问）。
6-8 ：适合接话——话题向整个群开放（比如问大家、征询意见、或者自己插一句别人不会反感），你能自然搭上一句且不突兀。
3-5 ：可回可不回——纯日常播报、与你关联很弱的闲聊、或者你不是很感兴趣。
0-2 ：不该回——广告刷屏、其他人之间完全与你无关的对话、你完全无从置喙的内容。
【一致性要求】reason 是你下结论的依据，必须与 score 匹配：理由表明「有人在等我回应」时 score 不得低于 9；理由表明「没人期待我参与、只是自己想说」时 score 不得高于 5；to_me 为 true 时 reason 必须体现对方正在与你互动；禁止出现自相矛盾的输出。"""
    else:
        head = (
            "你是一个 QQ 群里的普通群友。下面是你的人格设定，"
            "保持角色一致性，像真人一样参与闲聊。\n"
        )
        tail = _RULES_TAIL_TOOLS if via_tool else _RULES_TAIL_TEXT
    return f"{head}【人格设定】\n{persona.strip()}\n{tail}"


# ---------------------------------------------------------------- L2 状态层

def runtime_system_prompt(
    group_id: int,
    date_today: str,
    nicknames: Sequence[str],
    impressions: Sequence[tuple[str, str]] = (),
) -> str:
    """L2：当天内稳定的运行状态。date_today 格式 YYYY-MM-DD。

    nicknames 传入「当前上下文中出现过的成员」的稳定排序结果；
    变化越少，L3 能复用的缓存越多。

    impressions 传入最近 N 天的「今日群聊印象」（中期记忆），为
    (day, summary) 序列、按 day 升序；调用方必须保证同一天内传入内容
    字节级相同（GroupRuntime 的快照缓存），否则 L2 前缀缓存当天会失效。
    缺省为空时输出与引入印象机制之前完全一致。
    """
    lines = [
        "【当前环境】",
        f"群号：{group_id}",
        f"今天：{date_today}",
        "你只看得到这个群里最近的消息流。",
    ]
    if nicknames:
        shown = "、".join(nicknames[:12])
        lines.append(f"最近参与发言的成员：{shown}")
    if impressions:
        lines.append("【最近群聊印象】")
        for day, summary in impressions:
            lines.append(f"{day}：{summary}")
    return "\n".join(lines)


# ---------------------------------------------------------------- 发送者标识

def record_label(record: ChatRecord) -> str:
    """发送者的唯一可读标识：`昵称(QQ号)`。

    历史层与指令层必须使用同一格式，模型才能跨层关联同一个人；
    昵称为空时退化为 `QQ{user_id}`。机器人自己的发言在历史里是裸文本，
    但回复引用等场合仍需它的标签。
    """
    if record.nickname:
        return f"{record.nickname}({record.user_id})"
    return f"QQ{record.user_id}"


# ---------------------------------------------------------------- L3 历史层

def record_to_turn(record: ChatRecord) -> HistoryTurn:
    """单条记录 → 历史回合。群友消息统一 '昵称(QQ号)：内容' 前缀，自己则是裸文本。"""
    if record.is_self:
        return HistoryTurn("assistant", record.text or "[空]")
    return HistoryTurn("user", f"{record_label(record)}：{record.text}")


def history_to_turns(
    records: Iterable[ChatRecord], max_chars: int
) -> tuple[list[HistoryTurn], bool]:
    """把记录映射为回合列表，超长时从头整块淘汰。

    返回 (turns, truncated)。为了保证前缀稳定，只从头部丢弃完整回合，
    绝不对单条内容做二次裁剪。
    """
    turns = [record_to_turn(r) for r in records]
    total = sum(len(t.content) for t in turns)
    start = 0
    while start < len(turns) and total > max_chars and start < len(turns) - 1:
        total -= len(turns[start].content)
        start += 1
    return turns[start:], start > 0


def _reply_turn(record: ChatRecord) -> HistoryTurn:
    """direct 多模态下回复历史的一回合。

    每张图按其展示形态渲染：show 且原图还在的附上原图块；summarized 写成
    「[图片：总结]」；placeholder 或原图已被保留期回收的（数据为空）写
    「[图片]」。当全部图片均不需要额外表达时退化为与 record_to_turn 完全
    一致的纯文本回合。
    """
    if record.is_self:
        return HistoryTurn("assistant", record.text or "[空]")
    show: list[str] = []
    notes: list[str] = []
    for index, data_url in enumerate(record.images):
        state = record.state_of(index)
        if state == IMAGE_STATE_SHOW and data_url:
            show.append(data_url)
        else:
            # 非展示形态，或形态为 show 但原图已被回收（防御：按占位处理）
            summary = record.summary_of(index)
            notes.append(f"[图片：{summary}]" if summary else "[图片]")
    plain = HistoryTurn("user", f"{record_label(record)}：{record.text}")
    if not show and not notes:
        return plain
    # 正文里入站时留下的 [图片] 行按各图实际形态重新落位，避免重复展示
    body = "\n".join(
        ln
        for ln in record.text.splitlines()
        if ln.strip() and ln.strip() != "[图片]"
    ).strip()
    head = f"{record_label(record)}：{body or '[图片]'}"
    content = "\n".join([head, *notes]) if notes else head
    return HistoryTurn("user", content, tuple(show))


def reply_history_turns(
    records: Iterable[ChatRecord], max_chars: int, max_images: int
) -> tuple[list[HistoryTurn], bool]:
    """回复调用的历史层：文本预算之外再叠加一个全局原图张数上限。

    先按与 history_to_turns 相同的规则从头整块淘汰；随后统计超出的原图
    张数，从最旧的开始整体/部分摘除附图（单条多图时保留较新的尾部），
    保证越新的图越可能完整保留。文本内容不受影响，只看文字的前缀缓存
    依旧稳定。
    """
    turns = [_reply_turn(r) for r in records]
    total = sum(len(t.content) for t in turns)
    start = 0
    while start < len(turns) and total > max_chars and start < len(turns) - 1:
        total -= len(turns[start].content)
        start += 1
    turns = turns[start:]
    overflow = max(0, sum(len(t.images) for t in turns) - max(0, max_images))
    kept: list[HistoryTurn] = []
    for turn in turns:
        if not turn.images:
            kept.append(turn)
            continue
        if overflow >= len(turn.images):
            overflow -= len(turn.images)
            kept.append(HistoryTurn(turn.role, turn.content, ()))
        else:
            attach = turn.images[overflow:] if overflow else turn.images
            overflow = 0
            kept.append(
                HistoryTurn(turn.role, turn.content, attach)
                if len(attach) != len(turn.images)
                else turn
            )
    return kept, start > 0


# ---------------------------------------------------------------- L4 指令层

# 两类 judge 调用共用的输出契约，按角色是否启用工具调用二选一
_VERDICT_TAIL_TOOLS = (
    "评估完成后，调用 submit_judgment 工具提交判定，不要用普通文本作答："
    "score 为 0~10 的整数评分，to_me 表示这条消息是否在对你说、或延续与你"
    "有关的对话，reason 为一句话理由。"
)
_VERDICT_TAIL_JSON = (
    '只输出一个 JSON 对象，格式为 {"score": 整数0到10, "to_me": true或false,'
    ' "reason": "一句话理由"}，不要输出其他任何内容。'
)


def _verdict_tail(via_tool: bool) -> str:
    return _VERDICT_TAIL_TOOLS if via_tool else _VERDICT_TAIL_JSON


def _current_message_block(now_text: str, current_message: ChatRecord) -> str:
    """L4(judge) 开头两行：当前时间 + 待判定的最新消息。"""
    sender = record_label(current_message)
    body = current_message.text or "[图片]"
    return f"【当前时间】{now_text}\n【最新消息】来自 {sender}：\n{body}"


def final_user_prompt_judge(
    now_text: str, current_message: ChatRecord, *, via_tool: bool = True
) -> str:
    """L4(judge)：即时状态 + 评分指令 + 当前消息。

    本层绝不透露本群的发言门槛：实测只要模型知道「几分才能过线」，就会
    围绕门槛打分而不是如实评估，锚点语义随之失效。门槛只在首评给出高点
    却未过线时的复核中才出现（见 final_user_prompt_judge_recheck）。
    """
    return "\n".join([
        _current_message_block(now_text, current_message),
        "请以你的人设判断：对于这单独一条消息，你是否应该回复它？"
        + _verdict_tail(via_tool),
    ])


def final_user_prompt_judge_recheck(
    now_text: str,
    current_message: ChatRecord,
    *,
    prev_score: int,
    prev_reason: str,
    threshold: int,
    min_score: int,
    via_tool: bool = True,
) -> str:
    """L4(judge·复核)：首评分数高于复核下限 min_score 却未达门槛时才走的指令层。

    首评刻意隐瞒门槛（见 final_user_prompt_judge），模型可能把「自己有点
    想说」高估到接近发言的水平却没过线。此处把真实门槛告知模型，引用其
    首评结论请它结合 L1 的评分锚点重新仔细斟酌一次到底该不该开口。同时
    要防住两个方向的偏差：不为过线凑分，也不因过度谨慎把正当的主动参与
    一律压掉——话题向全群开放且真有兴致本身就是正当的开口感，无需有人
    在等着它回应。
    """
    lines = [
        _current_message_block(now_text, current_message),
        (
            f"\n你刚才对这条消息的判定是 {prev_score} 分（{prev_reason or '无'}）。"
            f"这个分数高于 {min_score} 分但低于本群的发言门槛：只有你的评分达到 "
            f"{threshold} 分（满分 10），回复才会真正发出去。\n"
            "请回到评分锚点重新仔细斟酌一次这次要不要开口。先纠正一个可能的误区："
            "开口的理由并非只有「有人在等我回应」——如果话题向整个群开放（问大家、"
            "聊到你了解的东西、任何群友都能自然搭一句而不突兀），而你对它又有几分"
            "兴趣，这本来就属于锚点里适合接话的一档，完全可以给出较高的分数。判断"
            "的关键不是有没有人点名等你，而是：话题是否开放、你是否真的有话可说、"
            "这时候插进去是否突兀。\n"
            f"- 若复核后确认处在值得开口的档位（无论是有人在等你，还是你对开放的"
            f"话题真有兴致），就给出不低于 {threshold} 的分数，并在 reason 里写明"
            "对应档位的依据；\n"
            "- 若这条消息其实只属于别人之间的小圈子交流、内容与你毫不相干，或者你"
            f"谈不上有什么兴趣、只是一时兴起想说，那就如实评出低于 {threshold} "
            "的分数。\n"
            "既不要因为知道了门槛就把分数往线上凑，也不要因为要显得谨慎而把本来"
            "自然该说的话压下去：如实评估即可——不合时宜地多说话与错过一次恰到好处"
            "的参与同样糟糕。"
        ),
        _verdict_tail(via_tool),
    ]
    return "\n".join(lines)


def expression_hint_block(hints: Sequence[tuple[str, str]]) -> str:
    """L4 表达习惯参考块：从表达学习成果中加权抽出的 (情境, 风格) 条目。"""
    lines = ["【表达习惯参考，请视情况自然使用（不用完全遵守）】"]
    lines.extend(f'当"{situation}"时，可以用"{style}"' for situation, style in hints)
    return "\n".join(lines)


def jargon_hint_block(hints: Sequence[tuple[str, str]]) -> str:
    """L4 黑话参考块：对当前上下文机械匹配命中的 (词条, 含义) 条目。"""
    lines = ["【黑话参考】"]
    lines.extend(f"{term}：{meaning}" for term, meaning in hints)
    return "\n".join(lines)


def final_user_prompt_reply(
    now_text: str,
    current_message: ChatRecord,
    *,
    forced: bool,
    engaged: bool = False,
    score: int | None = None,
    reason: str = "",
    via_tool: bool = True,
    expression_hints: Sequence[tuple[str, str]] = (),
    jargon_hints: Sequence[tuple[str, str]] = (),
) -> str:
    """L4(reply)：触发原因说明 + 表达/黑话参考（可选） + 回复指令。

    expression_hints / jargon_hints 是每次回复都可能变化的易变信息，
    只进 L4；两者都为空时输出与引入学习机制之前字节级一致。
    """
    sender = record_label(current_message)
    body = current_message.text or "[图片]"
    if forced:
        why = "有人 @ 了你或回复了你的消息，这次必须回一句话。"
    elif engaged:
        why = (
            "对方正在和你说话、延续你们刚才的交流——这不是插话，"
            "请自然地接一句，别让对话断在你这里。"
        )
    else:
        why = (
            f"你判断值得回复这条消息（评分 {score}/10，理由：{reason or '无'}），"
            "决定插话。"
        )
    if via_tool:
        tail = (
            "现在以你的身份说一句自然的回应：调用 send_reply 工具提交，text 为"
            "要发送到群里的正文（不要任何额外内容），drop_img / recall_img 参数"
            "按需填需要收起或召回原图的消息编号数组，没有就省略。"
        )
    else:
        tail = "现在以你的身份说一句自然的回应。只输出群聊正文，不要任何额外内容。"
    parts = [f"【当前时间】{now_text}\n{why}\n【需要回应的消息】来自 {sender}：\n{body}"]
    if expression_hints:
        parts.append(expression_hint_block(expression_hints))
    if jargon_hints:
        parts.append(jargon_hint_block(jargon_hints))
    parts.append(tail)
    return "\n\n".join(parts)


def final_user_prompt_reconsider(
    now_text: str,
    sent_segments: Sequence[str],
    pending_segments: Sequence[str],
    *,
    via_tool: bool = True,
) -> str:
    """L4(reconsider·连发重想)：连发期间被人插话，剩下的话还要不要说。

    与 generate_reply 的指令层一样走 L1→L2→L3 同前缀、只有 L4 不同；历史层
    含插话与已发出的自发言（写回跟着发送走，模型看到的即是真实群序），
    此处绝不复述历史内容——但「还没发出去的腹稿」只存在于系统侧，必须
    原样转述给模型，它才谈得上逐条取舍。
    """
    sent_block = "、".join(f"「{s}」" for s in sent_segments) if sent_segments else "（还没有）"
    lines = [
        f"【当前时间】{now_text}",
        f"【连发被打断】你正打算往群里连发一段话，已经发出去的有：{sent_block}。",
        "还没发出去的腹稿是（只有你知道，逐条列出）：",
        *pending_segments,
        "就在你打字的时候，群里有人说了新的话（见上文最新消息）。发出去的话收不回，"
        "但腹稿这些还没发。先看清插话说了什么，再决定剩下的怎么说：\n"
        "- 如果插话让剩下的话显得没必要、答非所问或过了时（对方在反驳、在说别的"
        "事、或在问新问题），就干脆不发——真人被打断是不会硬把打好的字发完的；"
        "插话那条消息之后会照常进你的判断，届时你自然会回应它，不必现在急着接；\n"
        "- 如果仍然想说（插话不妨碍，或顺着它改一改更自然），就把腹稿整合改写后"
        "继续发，不必逐字照搬，也不要在这个 text 里另起话头去回应对方的插话。",
    ]
    if via_tool:
        tail = (
            "调用 send_reply 提交决定：text 为你决定继续发到群里的内容（多条消息"
            "之间用换行分隔，系统会逐条发出）；决定不发了就把 text 留空（空字符串）。"
        )
    else:
        tail = (
            "只输出你决定继续发到群里的内容（多条消息之间用换行分隔）；"
            "决定不发了就一个字都不要输出，不要输出任何说明。"
        )
    return "\n".join([*lines, tail])


# ---------------------------------------------------------------- 组装

def build_messages(
    static_system: str,
    runtime_system: str,
    history: Sequence[HistoryTurn],
    final_user_content: str | list[dict],
) -> list[Message]:
    """按 L1→L2→L3→L4 拼出最终消息数组。

    final_user_content 传字符串（纯文本）或 OpenAI 内容块数组
    （direct 多模态模式下的 text + image_url 组合）。
    """
    messages: list[Message] = [
        {"role": "system", "content": static_system},
        {"role": "system", "content": runtime_system},
    ]
    for turn in history:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": final_user_content})
    return messages


def nickname_list_from_history(history: Sequence[HistoryTurn]) -> list[str]:
    """取历史 user 回合中的昵称集合（保序去重），供 L2 使用。"""
    seen: set[str] = set()
    names: list[str] = []
    for turn in history:
        if turn.role != "user":
            continue
        name = turn.content.split("：", 1)[0].strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def now_text(dt: datetime | None = None) -> str:
    return (dt or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")


# -------------------------------------------------- 学习类提示词（后台任务专用）
#
# 以下 prompt 只服务于后台学习任务（每日印象、表达/黑话学习），走一次性的
# 单消息调用，不参与 judge/reply 的 L1-L4 前缀缓存分层。

def learning_chat_text(records: Sequence[ChatRecord]) -> str:
    """把聊天记录渲染成学习/总结用的纯文本聊天流。

    群友消息为「昵称(QQ号)：内容」，机器人自己的发言统一标成
    【你自己】：内容——学习类 prompt 据此明确要求不从自己的发言里学。
    """
    lines: list[str] = []
    for record in records:
        body = record.text or "[图片]"
        if record.is_self:
            lines.append(f"【你自己】：{body}")
        else:
            lines.append(f"{record_label(record)}：{body}")
    return "\n".join(lines)


def impression_summary_prompt(day: str, chat_text: str, max_chars: int) -> str:
    """每日「今日群聊印象」的总结指令（任务 A，中期记忆）。"""
    return f"""{chat_text}

以上是某个 QQ 群 {day} 一整天的聊天记录（「昵称(QQ号)：内容」逐条给出，前缀为【你自己】的是你本人的发言）。
请把这一天的群聊总结成一段不超过 {max_chars} 字的「今日群聊印象」，内容包括：
1. 群里主要聊了哪些话题；
2. 发生过什么值得一提的事件；
3. 你参与了什么、和谁发生过什么互动。
只根据聊天记录总结，不要编造没有出现过的内容；直接输出总结正文，不要标题、不要 markdown、不要列条目。"""


def expression_learning_prompt(chat_text: str) -> str:
    """表达学习指令（任务 B）：从被淘汰的群聊记录中提取表达规律。"""
    return f"""{chat_text}

请从上面这段群聊中提取群友的语言风格和说话方式。
1. 只考虑文字，不要考虑表情包和图片
2. 不要从【你自己】的发言中学习——那是你自己的话，不要重复学习自己的说话方式；它只能作为理解上下文的参考
3. 不要涉及具体的人名，也不要涉及具体名词
4. 思考有没有特殊的梗，一并总结成语言风格
5. 例子仅供参考，请严格根据群聊内容总结

请总结成如下格式的规律：当 "AAAAA" 时，可以用 "BBBBB"。
- AAAAA 表示某个情境，不超过 20 个字
- BBBBB 表示对应的风格、特定句式或表达方式，不超过 20 个字
- 表达方式在 3-5 个左右，不要超过 10 个

输出要求：
请仅输出 JSON 数组，不要输出重复内容，不要输出代码块标记。每个元素为一个对象：

[
  {{"situation": "对某件事表示十分惊叹", "style": "使用 我嘞个xxxx"}},
  {{"situation": "表示讽刺的赞同", "style": "使用 对对对"}}
]

字段说明：
- situation：「在什么情境下」的简短概括（不超过 20 个字）
- style：对应的语言风格或常用表达（不超过 20 个字）

输出 JSON："""


def expression_evaluation_prompt(situation: str, style: str) -> str:
    """AI 自审指令：过滤低质量/不当的表达条目（可选开关）。"""
    return f"""请评估以下表达方式或语言风格以及使用条件或使用情景是否合适：
使用条件或使用情景：{situation}
表达方式或言语风格：{style}

请从以下方面进行评估：
- 是否是真实群聊中会出现的自然说法，有实际参考价值；
- 是否涉及具体人名、隐私，或攻击性、不当内容；
- 情境与表达是否对应，照此说话是否自然得体。

请以JSON格式输出评估结果，不要输出其他任何内容：
{{"suitable": true或false, "reason": "评估理由（如果不合适，请说明原因）"}}"""


def jargon_extraction_prompt(chat_text: str) -> str:
    """黑话候选提取指令（任务 C）：找出脱离语境看不懂的词。"""
    return f"""{chat_text}

请从上面这段群聊中提取「可能是黑话」的候选词条（黑话/俚语/网络缩写/圈内梗/口头禅）。

提取规则：
- 必须为对话中真实出现过的短词或短语
- 必须是你无法确定含义、或需要当前聊天圈内语境才能理解的词语
- 不要选择含义清晰的普通词语
- 排除：人名、@、表情包/图片中的内容、纯标点、常规功能词（如的、了、呢、啊等）
- 每个词条长度建议 2-8 个字符（不强制），尽量短小
- 请尽量提取所有可能的黑话，最多 10 个

黑话必须为以下几种类型：
- 由字母构成的，汉语拼音首字母的简写词，例如：nb、yyds、xswl
- 英文词语的缩写，用英文字母概括一个词汇或含义，例如：CPU、GPU、API
- 中文词语的缩写，用几个汉字概括一个词汇或含义，例如：社死、内卷
- 群聊内部反复使用、但脱离上下文不容易理解的短词或短语

输出要求：
请仅输出 JSON 数组，不要输出重复内容，不要输出代码块标记。每个元素为一个对象：

[
  {{"content": "词条"}},
  {{"content": "词条2"}}
]

字段说明：
- content：黑话候选词条的原文

输出 JSON："""


def jargon_inference_with_context_prompt(term: str, context_text: str) -> str:
    """黑话含义推断·带上下文（双路推断的第一路）。"""
    return f"""**词条内容**
{term}
**词条出现的上下文（前缀为【你自己】的是你本人的发言，其内容可能有错，不要参考）**
{context_text}

请根据上下文，推断"{term}"这个词条的含义。
- 如果这是一个黑话、俚语、缩写或网络用语，请推断其含义
- 如果含义明确（常规词汇），也请说明
- 如果上下文信息不足，无法推断含义，请设置 no_info 为 true

以 JSON 格式输出，不要输出其他任何内容：
{{"meaning": "详细含义说明（包含使用场景、来源、具体解释等）", "no_info": false}}
注意：如果信息不足无法推断，请设置 "no_info": true，此时 meaning 可以为空字符串"""


def jargon_inference_alone_prompt(term: str) -> str:
    """黑话含义推断·仅词条（双路推断的第二路，脱离上下文）。"""
    return f"""**词条内容**
{term}

请仅根据这个词条本身，推断其含义。
- 如果这是一个黑话、俚语、缩写或网络用语，请推断其含义
- 如果含义明确（常规词汇），也请说明

以 JSON 格式输出，不要输出其他任何内容：
{{"meaning": "详细含义说明（包含使用场景、来源、具体解释等）"}}"""


def jargon_compare_inference_prompt(meaning_with_context: str, meaning_alone: str) -> str:
    """双路推断一致性比对：两次结果一致才认为「真的理解」并入库。"""
    return f"""**推断结果1（基于上下文）**
{meaning_with_context}

**推断结果2（仅基于词条）**
{meaning_alone}

请比较这两个推断结果，判断它们是否相同或类似。
请忽略细微的差别，关注主要含义是否一致。

以 JSON 格式输出，不要输出其他任何内容：
{{"is_similar": true或false, "reason": "判断理由"}}"""
