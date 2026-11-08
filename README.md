# **为什么引入Langfuse？**

在实际的业务落地中，单纯依靠控制台的print日志无法满足工业级生产需求。Langfuse为本项目提供了以下三大核心功能：

## **1.多阶段调用链追踪：**

* 将一次用户的交互细分为：**Intent Router (意图分类) → Vector Search (向量召回) → RAG Answer (溯源生成)。

* 毫秒级展示每个阶段的延迟，精准定位系统性能瓶颈。

## **2.Token 与成本精细化核算：**

* 自动统计 Prompt Tokens、Completion Tokens 及总开销，支持针对退费等高频流程进行 ROI 评估。

## **输入输出与幻觉核查：**

* 完整记录大模型最终接收到的组装 Prompt (包含检索出的上下文片段与页码标签)，便于人工复盘与 Bad Case 归因。


# **代码块树 [文档产物] ingest.py 生成的切片明细，供人工审查**


```text
│
├── edu_policy.pdf          # 原始业务规章（PDF 知识库源文件）py 离线生成的本地向量库文件夹
│
├── mock_db.py              # [业务层] 模拟外部订单数据库与幂等锁
├── rag_engine.py           # [检索层] 秒级只读本地向量库，执行距离打分
├── agent.py                # [核心引擎] LangGraph 显式状态机、HITL 挂起逻辑
│
├── main.py                 #
├── ingest.py               # [ETL离线工具] 解析PDF，导出Markdown并持久化入库 [网关层] FastAPI 提供外部 RESTful 与审批恢复接口
├── test_regression.py      # [评测层] Pytest 自动化回归测试（带规则断言）
└── interactive_cli.py      # [交互层] 
├── processed_chunks.md     # [文档产物] ingest.py 生成的切片明细，供人工审查
├── chroma_db/              # [向量数据库] ingest.py 离线生成的本地向量库文件夹
│
├── mock_db.py              # [业务层] 模拟外部订单数据库与幂等锁
├── rag_engine.py           # [检索层] 秒级只读本地向量库，执行距离打分
├── agent.py                # [核心引擎] LangGraph 显式状态机、HITL 挂起逻辑
命令行自主测试终端（支持打字问答与审批中断）