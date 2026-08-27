# CandyBot

一个通过 SnowLuma 参与 QQ 群聊讨论的 AI 机器人。

- **收消息**：SnowLuma 通过 OneBot v11 HTTP POST 把事件上报到 CandyBot 内置的 aiohttp 服务；
- **发消息**：全部经由 SnowLuma 的 MCP server（stdio 启动 `@snowluma/mcp`，write 模式 `invoke_action("send_group_msg")`）；
- **大脑**：OpenAI 兼容 API。`judge` 模型逐条评估「是否值得回复这条消息」打 0-10 分，超过阈值才插话；@机器人/回复机器人必答，由 `reply` 模型生成回复；可选 `vision` 模型把图片转成文字描述。

## 快速开始

```bash
uv sync                          # 安装依赖
uv run pytest                    # 跑测试（48 个）
uv run main.py                   # 启动
```

### 前置条件

1. **SnowLuma 实例**已部署并登录 QQ（WebUI 默认 <http://127.0.0.1:5099>）。
2. 在 SnowLuma WebUI 的「网络适配器」里：
   - **HTTP 服务端**已启用（默认 `http://127.0.0.1:3000/`），记下 accessToken；
   - **HTTP 上报（client）** 新增一条，URL 填 `http://127.0.0.1:5700/onebot/event`，勾选消息事件。
3. Node.js ≥ 22（MCP 子进程需要）。
4. 编辑 `config.json`（见下文逐项说明），至少改掉 `bot.self_qq`、`ai_backend.api_key`、`snowluma.api_key`，并把要服务的群号写进 `groups` 且 `enabled: true`。

## 配置项说明（config.json）

| 段 | 字段 | 说明 |
|---|---|---|
| bot | self_qq | 机器人自己的 QQ 号，用于识别 @我 / 回复我 / 过滤自己消息 |
| | listen_host / listen_port | 事件上报服务监听地址（默认仅本机） |
| | event_secret | 非 null 时校验上报请求的 HMAC-SHA1 签名（OneBot v11 标准） |
| | data_dir | 记忆等运行时数据的目录 |
| | log_level | 日志级别：DEBUG/INFO/WARNING/ERROR/CRITICAL（默认 INFO）。设 DEBUG 可查看每次 LLM 请求的完整 prompt 与收到的消息 |
| groups | （群号为键） | **白名单**。只有列出的群会被服务；条目内可覆盖 persona/threshold/cooldown/context_size，留空/-1/null 表示继承 groups_default；`enabled: false` 单独禁用该群 |
| | — 特例 | `groups` 为空对象且 `groups_default.enabled=true` 时服务所有群（不建议） |
| groups_default | … | 兜底人设与参数；groups 为空时即全群配置 |
| ai_backend | base_url / api_key | OpenAI 兼容 API 地址与密钥 |
| models | judge | 判断是否参与话题的模型（建议便宜快速的，如 glm-4-flash） |
| | reply | 参与回答生成回复的模型 |
| | vision | describe 多模态模式用的视觉模型，其他模式可不填 |
| generation | reply_max_tokens / temperature | 回复生成长度与随机性 |
| | max_context_chars | 喂给模型的历史上下文总字符上限 |
| | timeout_seconds | 单次 LLM 调用超时 |
| multimodal | mode | `placeholder` 图片→"[图片]"；`describe` 视觉模型转文字；`direct` 图片 base64 直传（要求 reply 模型多模态） |
| | download_media | 是否下载图片（direct/describe 需要） |
| rate_limit | global_daily_limit | 全局每日主动发言上限，null 不限（@必答不受限） |
| snowluma | mcp_command / mcp_args | MCP server 启动命令（默认 npx -y @snowluma/mcp） |
| | endpoint | SnowLuma OneBot HTTP 端点；**允许私网需显式设置 allow_private_endpoint=true** |
| | api_key | accessToken，作为 Bearer token 传给 MCP |
| | mode | 必须 `"write"`，否则无法发言 |

`bot.self_qq` 也可在机器人名字上体现——发送的记忆里机器人昵称固定为「糖糖」，若你在 persona 里另取了名字请保持一致或修改 `candybot/bot.py` 中的昵称常量。

## 行为逻辑

```
事件上报 → 过滤链(非群聊/白名单外/自己/重复 → 丢弃)
        → 归一化(@/回复/图片处理，写入该群记忆并落盘 JSONL)
        → 决策：
            @我或回复我的消息 ──────────────► 必答
            其他消息 → judge 模型评估"是否该回复这条消息"(0-10) ─┐
                       score ≥ 阈值 且 冷却已过 ──► reply 模型生成 → invoke_action 发送
                       否则 ──► 只留在上下文
        → 发送成功后把自己发言也写回记忆
```

- 每个群一条串行决策队列，保证顺序；回复生成异步执行不阻塞后续消息。
- judge 失败按不发言处理；reply 失败重试 2 次；发送失败重试 3 次。
- 重启后每群记忆自动从 `data/memory/<群号>.jsonl` 恢复最近上下文。

## KV Cache 优化

提示词按稳定性严格分四层（详见 `candybot/prompts.py` 模块注释）：

1. system·静态层：persona + 守则，字节级不变；
2. system·状态层：群号、当天日期、成员昵称表（天内稳定）；
3. 历史层：只追加、从头整块淘汰，绝不重排；
4. user·指令层：秒级时间、触发类型等易变信息全部压到最后一层。

相邻两次调用中 L1-L3 构成完全相同的前缀，API 侧前缀缓存命中率最大化。

## 人工验收清单

真机行为需要你手动验证：

1. **启动自检**
   ```bash
   uv run main.py
   ```
   日志应依次出现：「SnowLuma MCP 会话已建立」→ 工具列表含 `invoke_action` → 「事件服务已启动」→ 「CandyBot 已就绪」。若报配置错误按提示改 config.json。

2. **@必答**：在白名单群里 @机器人 说一句话 → 几秒内收到回复，日志无「兴趣评分」行（judge 未被调用）。

3. **阈值行为**：普通闲聊 → 日志出现 `回复判定 X/阈值 8`，X<8 时不发言；有人向群里提问、分享、或抛出容易接话的内容时，分数应明显更高 → 主动插话。judge 的提示词里已内置「评分锚点 + 与门槛校准」（见 prompts.py）：只要它理由写的是"需要回应"，分数就必须达到门槛；若实机仍偏保守，优先把该群的 `proactivity_threshold` 降到 6-7 观察，其次再改 persona。

4. **冷却**：主动发言后立刻制造高分话题 → 日志应无评分记录（冷却期直接跳过判断）；等待 `cooldown_seconds` 后恢复。

5. **记忆持久化**：和它聊几句 → Ctrl+C 退出 → 再启动后引用它上一场说的话（回复那条消息），它能理解语境。

6. **多模态**：切 `multimodal.mode=describe` 并配 vision 模型 → 发一张图，日志里该消息文本变为 `[图片：<描述>]`。

7. **调积极性**：觉得它话太少就把 `proactivity_threshold` 从 8 调到 5-6；太吵就调高 + 加大 `cooldown_seconds`。

## 开发

```bash
uv run pytest           # 全部测试
uv run pytest -k names  # 单测命名过滤
```

### 查看每次请求的完整 Prompt

把 config.json 的 `bot.log_level` 设为 `"DEBUG"` 后重启即可在日志里看到 judge / reply / vision 每次请求的完整消息数组（含分层内容；多模态图片只显示长度和头部片段），以及每条收到消息、冷却跳过等细节：

```json
"bot": { "log_level": "DEBUG", ... }
```

配置错误（非法级别名）会在启动时直接报错，便于及时发现。

模块速览：`models.py`(领域模型+配置校验+SSRF 校验) · `normalize.py`(OneBot→内部消息) · `memory.py`(JSONL 记忆) · `events_server.py`(aiohttp 接收) · `snowluma.py`(MCP 客户端) · `prompts.py`(KV Cache 分层提示词) · `ai.py`(LLM 三角色) · `bot.py`(编排)。
