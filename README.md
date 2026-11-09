# 教育客服 Agent

一个用 LangGraph 显式状态机实现的教育机构客服 Agent：
学员问一句话，系统先判断意图，**咨询**走「问题改写 → 双路检索 → 精排 → 带引用生成」，
**退费**走「大模型调工具查单 → 政策判定 → 人工审批中断 → 执行退款」。
状态全程落 SQLite，所以可以挂起、跨进程恢复、逐条溯源。

完整的流程说明（数据怎么入库、每个节点做什么、为什么这么设计、有哪些坑）见
**[docs/项目全流程.md](docs/项目全流程.md)** —— 这是项目唯一的主文档。

## 目录

```text
core/                 编排与业务核心
  agent_workflow.py     LangGraph 状态机：8 个节点 + 条件边 + HITL
  rag_engine.py         BM25 + 向量双路召回 → RRF → 覆盖率精排 → 阈值
  memory_manager.py     长期记忆：画像 + 情景记忆（long_term_memory.db）
  memory_tools.py       3 个记忆工具（读画像 / 存记忆 / 检索记忆）
  refund_service.py     订单查询 / 退款政策判定 / 退款申请登记
  tools.py              4 个 @tool（闭包注入 user_id）
server/               FastAPI 网关与数据层
  main.py               /api/chat、/api/audit/resume + 调试接口
  mock_db.py            内存版订单库（2 条假订单）
  refund_store.py       退款流水（refund_records.db，主键防重复）
scripts/
  ingest.py             离线建库：data/ → 切片 → chroma_db/ + 切片台账
  interactive_cli.py    终端对话，内置人工审批控制台
  eval_rag.py           离线评测：Recall / MRR / 引用率 / 拒答率 / 延迟
  eval_memory.py        长期记忆检索评测：Recall@K / MRR / 误召回 / 用户隔离
  render_flow.py        生成问答链路 SVG（docs/流程图.svg）
  setup_repo.ps1        一次性引导（测试 → requirements → git init）
data/                 知识库源文件、干扰文档、评测标注集
docs/项目全流程.md     主文档
docs/流程图.png        一页纸问答链路图（render_flow.py 生成）
docs/长期记忆.md       长期记忆设计与评测
tests/                单测（不花钱）+ 真模型回归测试
demo_rerank.py        打印 RRF → 精排 → 阈值的中间分，讲原理用
```

## 快速开始

```bash
conda activate agent
python scripts/ingest.py          # 必须先建库：rag_engine 在 import 时就把库读进内存
pytest -q                         # 不花钱的测试
python scripts/interactive_cli.py # 命令行对话（退费会触发人工审批窗口）
uvicorn server.main:app --reload  # HTTP 服务
```

`.env` 需要 `DEEPSEEK_API_KEY`（照 `.env.example` 复制一份）。Langfuse 是可选的，没配也能跑。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/chat` | 一轮问答；高风险退费返回 `{"status": "SUSPENDED", ...}` |
| POST | `/api/audit/resume` | 人工审批后恢复：`{"thread_id": ..., "approved": true}` |
| GET | `/api/memory/{user_id}` | 查看画像与最近记忆 |
| DELETE | `/api/memory/{user_id}/items/{memory_id}` | 删除一条记忆（带归属校验） |
| POST | `/api/memory/{user_id}/prune` | 遗忘：删「超过 N 天且重要性低」的记忆 |
| GET | `/api/health` | 存活检查 |
| GET/POST | `/api/debug/*` | 订单、退款流水、会话状态、重置（`ENABLE_DEBUG_API` 控制） |

```bash
curl -s localhost:8000/api/chat -H "Content-Type: application/json" -d '{
  "thread_id": "t-1", "user_id": "USER_B",
  "message": "这个课我不想学了，帮我退款", "order_id": "ORD_002"}'
```

## 评测基线

7 份文档 / 30 个切片，easy 16 题、hard 18 题：检索 `Recall@2` 100%，MRR 0.917 / 0.962，
答案通过率 100% / 94.4%，P95 约 5.2 / 5.5 秒。复现：

```bash
python scripts/eval_rag.py                        # 只评检索层（免费）
python scripts/eval_rag.py --with-llm             # 再评答案质量
python scripts/eval_rag.py --mode dense           # 单路对照实验
```

## 长期记忆

画像（学习目标 / 基础 / 偏好）每轮注入咨询与售后 Prompt；情景记忆（退款进度、关键事实）
按当前问题检索 Top-3，不把全部历史塞进上下文。退款申请、成功、驳回由代码确定性写入，
模型也可以通过 `save_memory` 记录用户透露的事实；遗忘用 TTL + 重要性。

```bash
python scripts/eval_memory.py          # Recall@3 / MRR / 误召回 / 用户隔离
curl localhost:8000/api/memory/USER_A  # 查看画像与最近记忆
```

完整设计见 [docs/长期记忆.md](docs/长期记忆.md)。

## 已知限制

订单在内存、退款流水在磁盘（重启后两者口径不一致）；政策阈值在代码与文档各写了一份；
`/api/audit/resume` 未校验审批人身份；Chroma 默认英文 embedding。
逐条说明与修复方向见 [docs/项目全流程.md §12](docs/项目全流程.md)。
