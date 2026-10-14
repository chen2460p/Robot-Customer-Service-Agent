# 智扫通 · 扫地机器人智能客服（Robot Customer Service Agent）

一个用 Python 实现的**教学型扫地机器人客服 Agent**，在网页里用自然语言咨询选购、故障、保养问题，也能一句话生成个人使用报告。

- 知识问答基于 **RAG 检索增强**（Chroma 向量库 + 通义千问 Embedding），回答来自知识库而非模型记忆
- 7 个工具由 LangChain **ReAct Agent 自主编排**，模型自己决定调哪个、调几次
- **运行时切换人设**：日常是客服，一旦判定用户要生成报告，自动切为报告写手——靠中间件实现，不改 Agent 主体代码
- **三级数据源降级**：Redis → MySQL → CSV，任一层挂掉都不阻断主流程

---

## 目录

- [功能特性](#功能特性)
- [整体架构](#整体架构)
- [目录结构](#目录结构)
- [环境准备](#环境准备)
- [配置项说明（.env）](#配置项说明env)
- [快速开始](#快速开始)
- [对话全流程（真实链路）](#对话全流程真实链路)
- [关键设计与缺陷修复](#关键设计与缺陷修复)
- [数据降级矩阵](#数据降级矩阵)
- [模块说明](#模块说明)
- [核心数据结构](#核心数据结构)
- [能力边界](#能力边界)
- [常见问题](#常见问题)
- [免责声明](#免责声明)

---

## 功能特性

- **多轮对话**：携带上下文窗口（`config/agent.yml` 的 `max_history_messages`，默认最近 10 条），支持「那它呢」「刚才说的那个」这类指代性追问
- **RAG 知识问答**：用户问题 → 向量检索 top-3 相关片段 → 拼进提示词 → 模型总结。回答可追溯到具体资料片段，不靠模型自由发挥
- **工具自主编排**：7 个工具由模型按需调用，含 RAG 检索、天气查询、用户数据查询等；调用与否由模型判断，不是硬编码流程
- **运行时提示词切换**：注册一个「哨兵工具」，模型调用它即代表意图切换；中间件拦截后改写运行时上下文，下一次生成提示词时自动换成报告写手人设
- **流式输出**：Streamlit 前端逐字打字机效果，只输出模型自己产出的 `AIMessage`，过滤掉用户提问与工具原始返回
- **真实天气集成**：`get_weather` 调用 Open-Meteo 公开 API（**免密钥**），含超时控制、指数退避重试、两级 TTL 缓存与优雅降级
- **三级数据源降级**：使用记录按 Redis 缓存 → MySQL → CSV 逐层回退，缓存层或数据库不可用时自动降级，不阻断对话

## 整体架构

```
                        用户自然语言（浏览器）
                                │
                    ┌───────────▼───────────┐
                    │   app.py（Streamlit）  │
                    │   聊天界面 + 流式展示   │
                    └───────────┬───────────┘
                                │ execute_stream(query, history)
                    ┌───────────▼───────────┐
                    │  ReactAgent           │
                    │  agent/react_agent.py │
                    │  model+prompt+tools   │
                    │  +middleware 组装      │
                    └───────────┬───────────┘
                                │
   ┌────────────────┬───────────┼───────────┬────────────────┐
   ▼                ▼           ▼           ▼                ▼
model/factory.py  tools/      tools/      rag/            db/
ChatModel         7 个工具     3 个中间件   RAG 检索+总结    MySQL/Redis
通义千问          @tool       monitor     检索→拼上下文     连接池
qwen3.8-max      weather_     log          →LCEL→总结      + 缓存
Embedding        client       prompt_switch
v4               Open-Meteo  客服↔报告     Chroma 向量库
```

**Agent 执行链路**：`create_agent` 内部是 ReAct 循环——模型「思考 → 决定调工具 → 观察工具结果 → 再思考」，直到给出最终回答。中间件挂在这条链路的固定切点上：

```
工具调用 ──► [monitor_tool]     记录工具名/入参；若调的是哨兵工具则置 context["report"]=True
   │
模型调用 ──► [log_before_model] 记录此刻消息条数与最后一条消息
   │
生成提示词 ► [report_prompt_switch] 读 context["report"]，决定用「客服」还是「报告写手」提示词
```

## 目录结构

```
.
├── README.md                   # 本文档
├── .env.example                # API Key 模板（复制为 .env 后填 Key）
├── requirements.txt            # 依赖清单（锁定实测版本）
├── app.py                      # 入口：Streamlit 聊天界面 + 流式打字机
├── bench_rag.py                # RAG 性能基准脚本（产出可复现指标）
├── agent/
│   ├── react_agent.py          # 组装 model + prompt + tools + middleware 的 ReAct 封装
│   └── tools/
│       ├── agent_tools.py      # 7 个 @tool 工具定义
│       ├── middleware.py       # 3 个中间件：工具监控 / 调用日志 / 动态提示词切换
│       └── weather_client.py   # 天气客户端：超时 / 指数退避 / 两级缓存 / 降级
├── rag/
│   ├── rag_service.py          # RAG 总结服务：LCEL 链（PromptTemplate → model → 解析）
│   └── vector_store.py         # Chroma 建库 / MD5 增量入库 / 检索器
├── model/
│   └── factory.py              # 抽象工厂：产出 chat_model / embed_model 单例
├── db/
│   ├── init_db.py              # 建库建表 + CSV 数据迁移（幂等）
│   ├── mysql_manager.py        # MySQL 连接池 + 通用 CRUD
│   └── redis_manager.py        # Redis 连接池 + 通用缓存 / 业务缓存
├── utils/
│   ├── config_handler.py       # 配置加载中心（一次读入全部 YAML）
│   ├── path_tool.py            # 项目根锚定，跨启动目录可用
│   ├── logger_handler.py       # 双通道日志（控制台 INFO / 文件 DEBUG 按日切）
│   ├── prompt_loader.py        # 提示词按需加载
│   └── file_handler.py         # PDF/TXT 加载 + 文件 MD5 指纹
├── config/                     # YAML 配置（模型名 / 向量库 / Agent / DB）
├── prompts/                    # 提示词文本（客服 / RAG 总结 / 报告写手）
├── data/                       # 知识库文档（txt/pdf）+ 外部数据 CSV
└── logs/                       # 运行日志（按日切分，已 gitignore）
```

## 环境准备

- **Python 3.13**（开发环境为 Windows 11 + Python 3.13.9）
- 依赖安装：

```powershell
pip install -r requirements.txt
```

- **MySQL 与 Redis 可选**：不启动也能跑完整对话，使用记录会自动降级到读 CSV（见 [数据降级矩阵](#数据降级矩阵)）
- 需要联网：调用通义千问 API 与 Open-Meteo 天气接口

## 配置项说明（.env）

复制模板：

```powershell
Copy-Item .env.example .env   # PowerShell
```

| 配置项 | 必填 | 使用模块 | 作用 |
|---|---|---|---|
| `DASHSCOPE_API_KEY` | **是** | `model/factory.py` | 通义千问 API Key，对话模型与 Embedding 都靠它 |

Key 申请地址：

- DashScope：`https://bailian.console.aliyun.com/` → API-KEY 管理

除 `.env` 外，还有两份 YAML 配置：

| 文件 | 说明 |
|---|---|
| `config/db.example.yml` | MySQL / Redis 连接参数模板，**复制为 `config/db.yml`** 后修改；含密码，已被 `.gitignore` 排除 |
| `config/rag.yml` | 模型名（`chat_model_name` / `embedding_model_name`） |
| `config/chroma.yml` | 向量库参数（集合名、持久化目录、分片大小、top_k） |
| `config/agent.yml` | 上下文窗口条数、天气接口地址与超时/重试/缓存参数 |

`.env` 与 `config/db.yml` 均已被 `.gitignore` 忽略，不会提交；仓库里只提供对应的 `.example` 模板。

## 快速开始

```powershell
# Windows PowerShell（中文环境建议开启 UTF-8）
$env:PYTHONUTF8 = "1"

# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 Key（编辑 .env 填入 DASHSCOPE_API_KEY）
Copy-Item .env.example .env

# 3. 配置数据库（可选；不配则自动降级用 CSV）
Copy-Item config/db.example.yml config/db.yml

# 4. 初始化数据库（可选，首次一次）
python db/init_db.py

# 5. 构建向量库（首次一次；新增文档后重跑即可增量入库）
python rag/vector_store.py

# 6. 启动
streamlit run app.py
```

浏览器打开 `http://localhost:8501`。试试问：

```
小户型适合哪些扫地机器人？        # 触发 RAG 检索
深圳今天适合开扫地机器人吗？      # 触发天气工具
给我生成我的使用报告              # 触发提示词切换 + 用户数据查询
```

## 对话全流程（真实链路）

以「帮我看看我这个月用得怎么样」为例，展示报告场景的完整工具链：

```
你 > 给我生成我的使用报告
       → 模型判定意图为「生成报告」，按提示词强约束走固定工具顺序：

         get_user_id()                 取用户ID（本项目为构造数据）
              │
         get_current_month()           取月份（本项目为构造数据）
              │
         fill_context_for_report()     哨兵工具：本身无业务
              │                        → 中间件置 context["report"]=True
              │                        → 下一次生成提示词时切为「报告写手」
              ▼
         fetch_external_data(user_id, month)
              │                        三级取数：Redis → MySQL → CSV
              ▼
       报告写手人设的模型整合数据，输出结构化使用报告

多轮对话示例：

你 > 小户型适合哪些扫地机器人？
       → RAG 检索知识库，返回基于资料的推荐

你 > 那它多久要清理一次滤网？      # 指代性追问，依赖上下文窗口
       → 携带历史，模型理解「它」指上文的机型
```

天气工具的真实链路：

```
你 > 深圳今天适合开扫地机器人吗？
       → get_weather("深圳")
          ├─ 地理编码：城市名 → 经纬度（缓存 1 天，实测冷调用约 1.3s）
          ├─ 实时天气 + 未来 6 小时最大降雨概率
          └─ 空气质量（可选增强，失败不影响主天气返回）
       → 返回中文天气描述，模型据此给出建议

# 任何一层失败都不会抛异常中断对话，而是返回降级文案：
# 「天气服务暂时不可用，未能获取到实时天气数据，请稍后再试。」
```

## 关键设计与缺陷修复

前四个问题的共同点是：**程序照常运行、不报任何错，但功能静默失效**——排查成本远高于修复成本。

**RAG 侧（两处）**

- **向量库路径随 CWD 漂移**：Chroma 会按*当前工作目录*解析相对路径，导致从不同目录启动会指向不同的库——应用照常回答、检索却恒为空且不报错。修复方式是统一经 `get_abs_path()` 锚定到项目根。
- **MD5 去重记录与向量库生命周期不一致**：记录文件固定在项目根，而向量库在 `persist_directory`。一旦换库/清库，记录仍声称「已入库」，新库对所有文件一律跳过——库是空的却没有任何报错。修复方式是把记录文件收进向量库同一目录，二者同生共死。

**Agent 侧（两处）**

- **无状态导致上下文丢失**：每轮调用只提交当前提问，多轮对话下「那它呢」这类**指代性追问直接失效**。修复方式是让 `execute_stream` 接收历史并按窗口截断后再提交。
- **流式输出未按消息类型过滤**：`stream_mode="values"` 吐出的每个状态快照里，`messages[-1]` 可能是**用户原始提问**（助手气泡会复述一遍问题）或**工具原始返回**（检索到的资料、天气串会直接倒进聊天窗口）。修复方式是只对 `AIMessage` 做流式输出，并按消息 id 去重。

**动态提示词切换的实现**（本项目核心亮点）

模型无法直接修改运行时上下文，只能通过「调用工具」这个动作发出信号。于是：

1. 注册空壳工具 `fill_context_for_report` 作为哨兵；
2. 模型判定要写报告时调用它；
3. `monitor_tool` 中间件拦截到这次调用，向 `runtime.context["report"]` 写入 `True`；
4. 下一次生成提示词前，`report_prompt_switch`（`@dynamic_prompt`）读该标记，返回报告写手的系统提示词。

**用「工具调用」作为状态切换信号**，全程不改 Agent 主体代码。

## 数据降级矩阵

| 组件不可用 | 影响面 | 降级行为 |
|---|---|---|
| Redis 未启动 | 缓存层 | 跳过缓存，直接查 MySQL（`get_json` 失败返回 `None`，不抛异常） |
| MySQL 未启动 | 使用记录 | 回退到读 `data/external/records.csv` |
| MySQL + Redis 均不可用 | 使用记录 | 读 CSV；CSV 也没有则返回空串，模型如实告知查不到 |
| 天气 API 超时/重试用尽 | 天气工具 | 返回配置的降级文案，如实告知用户，**不编造假天气** |
| 空气质量接口失败 | 天气工具 | 单独 try 包裹，仅少一项信息，主天气正常返回 |
| 城市名无法识别 | 天气工具 | 地理编码查不到 → 返回降级文案 |

设计原则：**缓存与非核心工具挂掉不应阻断主流程**，但**绝不能假装成功**。

## 模块说明

| 文件 | 关键内容 |
|---|---|
| `agent/react_agent.py` | `ReactAgent`：`create_agent(model, system_prompt, tools, middleware)` 组装；`execute_stream()` 组装消息列表（历史窗口截断）、按 `stream_mode="values"` 消费、仅对 `AIMessage` 流式输出并去重 |
| `agent/tools/agent_tools.py` | 7 个 `@tool`：`rag_summarize` / `get_weather` / `get_user_location` / `get_user_id` / `get_current_month` / `fetch_external_data` / `fill_context_for_report`；`fetch_external_data` 实现 Redis→MySQL→CSV 三级回退与惰性加载 |
| `agent/tools/middleware.py` | `monitor_tool`（`@wrap_tool_call`，记录工具调用 + 哨兵工具置位）、`log_before_model`（`@before_model`）、`report_prompt_switch`（`@dynamic_prompt`） |
| `agent/tools/weather_client.py` | `WeatherClient`：`_get_json` 指数退避重试、`_geocode` 城市名→经纬度（长缓存）、两级 TTL 缓存 + `threading.Lock`、WMO 天气代码映射、AQI 分档 |
| `rag/rag_service.py` | `RagSummarizeService`：LCEL 链 `PromptTemplate \| print_prompt \| chat_model \| StrOutputParser`；检索结果拼参考资料并编号；Redis 缓存查询结果 |
| `rag/vector_store.py` | `VectorStoreService`：`load_document()` 扫描 → 加载 → MD5 去重 → 递归分片 → 入库；`get_retriever()` 返回 k=3 检索器 |
| `model/factory.py` | `BaseModelFactory` 抽象基类 + `ChatModelFactory`（ChatOpenAI 走 DashScope 兼容模式）+ `EmbeddingsFactory`；模块级单例 `chat_model` / `embed_model` |
| `db/mysql_manager.py` | `PooledDB` 连接池；`query/query_one/execute/execute_many` 全部异常收敛，失败返回空值不打断上层 |
| `db/redis_manager.py` | `ConnectionPool`（`protocol=2` 兼容 Redis 5.x）；通用 `get/set/get_json/set_json/delete` + 业务缓存键（天气/RAG） |
| `db/init_db.py` | 建库 → 建表（`user_usage_records`，`user_id+month` 唯一键）→ CSV 批量导入（`ON DUPLICATE KEY UPDATE` 幂等）→ 打印统计 |
| `utils/path_tool.py` | `get_project_root()` 以文件自身位置为锚向上推算，保证任意目录启动都指向同一项目根 |
| `utils/logger_handler.py` | `get_logger()` 控制台 INFO + 文件 DEBUG 双通道；按 `if logger.handlers` 去重避免重复打印 |
| `utils/file_handler.py` | `get_file_md5_hex`（4KB 分块，不撑爆内存）、`listdir_with_allowed_type`、`pdf_loader`/`txt_loader` |
| `app.py` | `st.session_state` 缓存 Agent 实例与对话历史；`capture()` 生成器实现逐字打字机；`[:-1]` 排除本轮提问避免重复 |
| `bench_rag.py` | 四类指标：知识库构建 / 增量入库 / 检索延迟 / 端到端延迟；`--persist-dir` 隔离目录以复现冷启动 |

## 核心数据结构

```python
# Agent 输入：统一以「消息列表」为上下文格式，兼容多轮与工具消息
{"messages": [
    {"role": "user", "content": "..."},         # 历史（已按窗口截断）
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "本轮提问"},     # 追加在末尾
]}

# 运行时上下文：跨中间件共享的自定义标记
context = {"report": False}    # 哨兵工具被调用后 → True → 触发提示词切换

# 使用记录（fetch_external_data 的返回值，按 用户ID → 月份 两级索引）
{"1001": {"2025-01": {"特征": ..., "效率": ..., "耗材": ..., "对比": ...}}}
```

关键配置项：

```yaml
# config/chroma.yml
chunk_size: 200          # 分片大小（字符）
chunk_overlap: 20        # 相邻分片重叠
k: 3                     # 检索返回条数

# config/agent.yml
max_history_messages: 10 # 上下文窗口（user + assistant 合计）
weather: {timeout: 8, retries: 3, cache_ttl: 600, geo_cache_ttl: 86400}
```

## 能力边界

这部分刻意写明，避免误解项目完成度。

**真实实现（可放心声称已接入）**

- `rag_summarize` —— 真实 Chroma 向量检索
- `get_weather` —— 真实调用 Open-Meteo 公开 API，含超时 / 指数退避重试 / 两级 TTL 缓存 / 降级
- `fetch_external_data` —— 真实的三级数据源读取（Redis / MySQL / CSV）

**简化实现（不得声称已完成）**

- `get_user_location` / `get_user_id` / `get_current_month` —— 返回**构造数据**（`random.choice`），仅用于演示工具编排机制。真实业务中用户身份应从登录态或 Token 获取，本项目**未做鉴权**
- `data/external/records.csv` 为构造数据，未对接真实业务系统
- `data/` 下的知识文档为公开资料整理，非真实企业知识库

## 常见问题

**Q：MySQL / Redis 没启动，能跑吗？**
A：能。使用记录会自动降级：Redis 不通 → 直查 MySQL；MySQL 不通 → 回退读 `data/external/records.csv`。只是拿不到缓存加速。

**Q：检索结果总是空的，但程序不报错？**
A：多半是向量库没建。先跑一次 `python rag/vector_store.py` 把 `data/` 下的文档入库；若已建库仍为空，检查是否在别的目录下启动过（历史版本的 CWD 漂移问题，现已修复）。

**Q：新增了知识文档，要重新建库吗？**
A：把文件放进 `data/`，重跑一次 `python rag/vector_store.py` 即可。靠 MD5 去重，**只有新文件会被向量化**，已有文件自动跳过。

**Q：想换模型怎么办？**
A：改 `config/rag.yml` 的 `chat_model_name` / `embedding_model_name` 两行即可，业务代码零改动（模型创建集中在 `model/factory.py`）。

**Q：天气查不到？**
A：Open-Meteo 是公开接口、无需 Key，查不到通常是网络问题或城市名无法识别。程序会返回「天气服务暂时不可用」的降级文案，不会编造数据。

**Q：Windows 终端中文乱码？**
A：先执行 `$env:PYTHONUTF8 = "1"` 再运行；日志文件本身固定以 UTF-8 写入，不受控制台编码影响。

**Q：性能指标从哪来？**
A：跑 `python bench_rag.py`（加 `--persist-dir tmp_bench` 可隔离目录、反复复现冷启动）。它测量入库耗时、检索延迟 P50/P95、端到端延迟，指标可复现。

## 免责声明

本项目为**教学演示项目**：主体跟随公开教程实现，并在此基础上做了若干工程化改造（见 [关键设计与缺陷修复](#关键设计与缺陷修复)）。知识文档为公开资料整理，使用记录为构造数据。请勿将 `.env`、`config/db.yml` 或任何真实密钥提交到公开仓库。

## License

[MIT](LICENSE)
