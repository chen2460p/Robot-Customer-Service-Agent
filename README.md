# 智扫通 · 扫地机器人智能客服

基于 **ReAct Agent + RAG** 的扫地机器人客服系统。用户可以用自然语言咨询选购、故障、保养问题，也可以一句话生成个人使用报告。

Agent 会自主判断该调用哪些工具——查知识库、查天气、查用户使用记录——并在**运行过程中动态切换自己的"人设"**：日常是客服，一旦判定用户要生成报告，就切换成报告写手。

> 这是一个教学项目，主体跟随公开教程实现，并在此基础上做了若干工程化改造（详见 [关键设计](#关键设计) 与 [能力边界](#能力边界)）。

---

## 功能特性

- **多轮对话**：携带上下文窗口，支持"那它呢"这类指代性追问
- **RAG 检索增强**：Chroma 向量库 + 通义千问 Embedding，回答基于检索到的资料而非模型记忆
- **工具自主编排**：7 个工具由模型自主决定调用时机与顺序
- **运行时提示词切换**：靠中间件 + 哨兵工具实现，无需改 Agent 主体代码
- **三级数据源降级**：Redis 缓存 → MySQL → CSV，任一层挂掉都不阻断主流程
- **流式输出**：Streamlit 前端逐字打字机效果

## 架构

```
                            ┌─────────────────┐
                            │   app.py        │  Streamlit 前端
                            │  聊天界面 + 流式  │
                            └────────┬────────┘
                                     │ execute_stream()
                            ┌────────▼────────┐
                            │  ReactAgent     │  agent/react_agent.py
                            │  组装 + 编排     │
                            └────────┬────────┘
                                     │
        ┌──────────────┬─────────────┼──────────────┬──────────────┐
        │              │             │              │              │
   ┌────▼────┐   ┌─────▼─────┐  ┌────▼────┐   ┌─────▼─────┐  ┌─────▼─────┐
   │ 模型工厂 │   │  工具集    │  │ 中间件   │   │  RAG 服务  │  │ DB 管理器  │
   │ factory │   │ 7 个 tool │  │ 3 个      │   │ 检索+总结  │  │ MySQL     │
   │         │   │           │  │           │   │           │  │ Redis     │
   └────┬────┘   └─────┬─────┘  └────┬────┘   └─────┬─────┘  └───────────┘
        │              │             │              │
   通义千问        天气 API       提示词切换    Chroma 向量库
   qwen3.8-max    Open-Meteo    客服↔报告      + 本地知识库
```

### 目录结构

```
.
├── app.py                      # Streamlit 前端入口（界面 + 流式展示）
├── bench_rag.py                # RAG 链路性能基准脚本（产出可复现指标）
├── agent/
│   ├── react_agent.py          # 组装 model + prompt + tools + middleware
│   └── tools/
│       ├── agent_tools.py      # 7 个 @tool 工具定义
│       ├── middleware.py       # 3 个中间件（工具监控 / 调用日志 / 提示词切换）
│       └── weather_client.py   # 天气客户端（Open-Meteo，含重试与两级缓存）
├── rag/
│   ├── rag_service.py          # LCEL 链：检索 → 拼上下文 → 模型总结
│   └── vector_store.py         # Chroma 建库 / MD5 增量入库 / 检索器
├── model/
│   └── factory.py              # 抽象工厂，产出 chat_model / embed_model 单例
├── db/
│   ├── init_db.py              # 建库建表 + CSV 数据迁移
│   ├── mysql_manager.py        # MySQL 连接池 + 通用 CRUD
│   └── redis_manager.py        # Redis 连接池 + 缓存封装
├── utils/                      # 配置/路径/日志/提示词/文件 工具
├── config/                     # YAML 配置（模型名、向量库、Agent、DB）
├── prompts/                    # 提示词文本（客服 / RAG 总结 / 报告写手）
├── data/                       # 知识库文档（txt/pdf）+ 外部数据 CSV
└── logs/                       # 运行日志（按日切分，已 gitignore）
```

## 技术栈

| 层次 | 选型 |
|---|---|
| Agent 编排 | LangChain 1.x（`create_agent`）+ LangGraph |
| 大模型 | 通义千问 `qwen3.8-max`（DashScope OpenAI 兼容模式） |
| 向量模型 | 通义千问 `text-embedding-v4` |
| 向量库 | Chroma（本地持久化） |
| 数据层 | MySQL（PyMySQL + DBUtils 连接池）· Redis（缓存） |
| 前端 | Streamlit |
| 外部 API | Open-Meteo（天气 / 空气质量，**免密钥**） |

## 快速开始

**环境要求**：Python 3.13（开发环境实测 3.13.9）

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

项目读取环境变量 `DASHSCOPE_API_KEY`（[阿里云百炼](https://bailian.console.aliyun.com/) → API-KEY 管理 获取）。

推荐用 `.env` 文件：

```bash
cp .env.example .env
# 编辑 .env，填入真实 Key
```

也可以直接设置系统环境变量：

```powershell
# Windows PowerShell
$env:DASHSCOPE_API_KEY = "sk-xxxxxxxx"
```
```bash
# macOS / Linux
export DASHSCOPE_API_KEY="sk-xxxxxxxx"
```

### 3. 配置数据库

```bash
cp config/db.example.yml config/db.yml
# 编辑 config/db.yml，改成你本机的 MySQL / Redis 连接参数
```

> `config/db.yml` 含密码，已被 `.gitignore` 排除，不会进版本库。

### 4. 初始化数据库（首次运行一次）

```bash
python db/init_db.py
```

建库、建表，并把 `data/external/records.csv` 导入 MySQL（幂等，可重复执行）。

### 5. 构建向量库（首次运行一次）

```bash
python rag/vector_store.py
```

扫描 `data/` 下的知识文档，切片、向量化、写入 Chroma。靠 MD5 去重，**新增文档后重跑一次即可增量入库**。

### 6. 启动

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501`。试试问：

- 「小户型适合哪些扫地机器人？」（触发 RAG 检索）
- 「深圳今天适合开扫地机器人吗？」（触发天气工具）
- 「给我生成我的使用报告」（触发提示词切换 + 用户数据查询）

## 关键设计

### 运行时提示词切换（核心亮点）

同一个 Agent 要在"客服"和"报告写手"两种身份间切换，但模型无法直接修改运行时上下文。这里的做法是：

1. 注册一个空壳的**哨兵工具** `fill_context_for_report`；
2. 模型判定要写报告时调用它；
3. `monitor_tool` 中间件拦截到这次调用，向 `runtime.context["report"]` 写入 `True`；
4. 下一次生成提示词前，`report_prompt_switch`（`@dynamic_prompt`）读取该标记，返回报告写手的系统提示词。

**用"工具调用"作为状态切换信号**，全程不改 Agent 主体代码。

### 多轮对话的上下文窗口

历史会随对话无限增长，持续推高 token 消耗并最终撑爆上下文。这里截断到最近 N 条（`config/agent.yml` 的 `max_history_messages`，默认 10），是成本与效果之间最简单的折中。

### RAG 的两处隐蔽缺陷修复

- **向量库路径随 CWD 漂移**：Chroma 会按*当前工作目录*解析相对路径，导致从不同目录启动会指向不同的库——应用照常回答、检索却恒为空且不报错。修复方式是统一经 `get_abs_path()` 锚定到项目根。
- **MD5 去重记录与向量库生命周期不一致**：记录文件固定在项目根，而向量库在 `persist_directory`。一旦换库/清库，记录仍声称"已入库"，新库对所有文件一律跳过——库是空的却没有任何报错。修复方式是把记录文件收进向量库同一目录，二者同生共死。

### 三级数据源降级

`fetch_external_data` 按 **Redis 缓存 → MySQL → CSV** 的顺序取数，前两层各自独立 try 包裹：

| 层 | 作用 | 挂掉时 |
|---|---|---|
| Redis | 缓存热点查询（TTL 1h） | 跳过缓存，直接查 MySQL |
| MySQL | 主数据源 | 回退到读 CSV |
| CSV | 兜底 | 返回空串，模型如实告知查不到 |

同理，天气工具把非核心的空气质量接口单独 try——它失败只是少一项信息，不影响主天气返回。

## 性能指标

`bench_rag.py` 是可复现的性能基准脚本，测量四类指标：

```bash
python bench_rag.py                        # 用真实配置
python bench_rag.py --persist-dir tmp_dir  # 隔离目录，可反复复现"冷启动"
python bench_rag.py --queries 50 --e2e 5   # 自定义采样次数
```

- **知识库构建**：各文件切片数、冷启动全量入库耗时、向量数
- **增量入库**：二次运行（全部命中 MD5）耗时与文件跳过率
- **检索延迟**：向量检索的 P50 / P95（含 Embedding 调用）
- **端到端延迟**：RAG 总结的总耗时，即用户实际等待时间

> `--persist-dir` 的意义：向量库是持久化的，第二次运行必然"已入库"，测不到冷启动。指定全新目录即可复现冷启动，且不污染真实向量库。

## 能力边界

这部分是刻意写明的，避免误解项目的完成度。

**真实的**：

- `rag_summarize` —— 真实 Chroma 向量检索
- `get_weather` —— 真实调用 Open-Meteo 公开 API，含超时 / 指数退避重试 / 两级 TTL 缓存 / 降级
- `fetch_external_data` —— 真实的三级数据源读取

**简化的**：

- `get_user_location` / `get_user_id` / `get_current_month` —— 返回**构造数据**（`random.choice`），用于演示工具编排机制。真实业务中用户身份应从登录态或 Token 获取，本项目未做鉴权。
- `data/external/records.csv` 为构造数据，未对接真实业务系统。
- `data/` 下的知识文档为公开资料整理，非真实企业知识库。

## License

[MIT](LICENSE)
