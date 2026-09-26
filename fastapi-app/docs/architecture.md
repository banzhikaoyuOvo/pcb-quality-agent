# PCB 智能质检多 Agent 协同平台 - 架构文档

> LangGraph 5-Agent Orchestrator-Worker 架构 · YOLOv8n + DeepSeek + IPC-A-610G 知识库 · SSE 流式反馈

---

## 一、项目概述

### 1.1 项目背景

传统 PCB 质检依赖人工目检，效率低（单板 30-60s）、漏检率高（15-20%）。本项目用 **5-Agent 协同架构**自动化质检全流程：

```text
缺陷检测 → 标准匹配 → 根因分析 → 处置决策 → 报告生成
```

### 1.2 核心指标

| 指标 | 值 | 环境 |
|------|-----|------|
| YOLOv8n mAP@50 | **0.896** | PKU-Market-PCB 数据集 |
| Precision | **0.943** | - |
| 单帧推理延迟 | **1.2ms** | RTX 4060 Ti |
| 特征向量维度 | **256** | Neck Hook Layer 21 |
| IPC 知识库检索 | **<1ms** | YAML O(1) 精确匹配 |
| LLM 根因分析 | **<3s** | DeepSeek API |
| 端到端单次质检 | **3-5s** | 含 YOLO + LLM 调用 |

### 1.3 数据集

- **来源**：[PKU-Market-PCB](https://www.kaggle.com/datasets/akhatova/pcb-defects)
- **规模**：693 张图片，6 类缺陷
- **类别**：`missing_part` / `bent_lead` / `crack` / `short_circuit` / `wrong_placement` / `contamination`

---

## 二、系统架构

### 2.1 5-Agent Orchestrator-Worker 编排

```text
┌────────────────────────────────────────────────────────────┐
│                       START                                │
│                         ↓                                  │
│            ┌────────────────────────┐                      │
│            │   Decision Agent       │  (Supervisor)        │
│            │   动态路由 + 状态检查   │                      │
│            └───────────┬────────────┘                      │
│                        │                                   │
│    ┌───────────────────┼───────────────────┐               │
│    ▼                   ▼                   ▼               │
│ ┌────────┐         ┌─────────┐        ┌──────────┐         │
│ │Visual  │         │Standard │        │RootCause │         │
│ │YOLOv8n │         │IPC YAML │        │DeepSeek  │         │
│ │256-dim │         │ O(1)    │        │ 推理     │         │
│ └────┬───┘         └────┬────┘        └─────┬────┘         │
│      │                  │                   │              │
│      │        ┌─────────┴────────┐          │              │
│      │        │  DB Store        │          │              │
│      │        │  Milvus Lite     │          │              │
│      │        └─────────┬────────┘          │              │
│      └──────────────────┼───────────────────┘              │
│                         ▼                                  │
│                ┌─────────────────┐                         │
│                │  Report Agent   │                         │
│                │  Markdown 模板   │                         │
│                └────────┬────────┘                         │
│                         ▼                                  │
│                       END                                  │
└────────────────────────────────────────────────────────────┘
```

### 2.2 Agent 职责

| Agent | 技术 | 输入 | 输出 |
|-------|------|------|------|
| **Decision** | LangGraph 条件边 | State | 下一个节点名 |
| **Visual** | YOLOv8n + Hook Layer 21 | PCB 图片 | 缺陷列表 + 256 维特征向量 |
| **DB Store** | pymilvus-lite | 特征向量 | Milvus Upsert 结果 |
| **Standard** | YAML 内存字典 | 缺陷类型 | IPC-A-610G 标准条目 |
| **RootCause** | DeepSeek API | 缺陷 + 标准 | 根因分析（500+ 字） |
| **Report** | Python f-string | 全部结果 | Markdown 报告 |

### 2.3 编排特性

- **回环调度**：Worker 完成后通过回边 `add_edge("visual", "decision")` 强制回到 Decision
- **状态键存在性路由**：Decision 通过 `if "xxx" not in state` 判断下一步，避免空结果误判
- **异常隔离**：单节点 try-except + `state["error"]` 短路到 END
- **断点续传**：中断后重新 invoke 同一 State，从断点继续

---

## 三、关键设计决策

### 3.1 为什么用 YAML 精确匹配而不是向量 RAG？

**决策依据**：

| 维度 | 向量 RAG | YAML 字典 | 结论 |
|------|---------|-----------|------|
| 适用场景 | 非结构化文档 | 结构化标准条目 | YAML ✅ |
| 检索方式 | 语义相似 | 精确匹配 | YAML ✅ |
| 检索延迟 | 10-100ms | **<1ms** | YAML ✅ |
| 准确率 | 85-95% | **100%** | YAML ✅ |
| 幻觉风险 | 有 | **零** | YAML ✅ |

**核心原理**：

- PCB 缺陷类型是**确定的 6 类枚举值**
- IPC-A-610G 标准按缺陷类型**一一对应**
- 无需语义检索，O(1) 精确匹配即可

**选型原则**：

```text
查询键确定性 + 结果唯一性  →  精确匹配（YAML/字典）
查询键模糊   + 结果多样性  →  语义检索（向量 RAG）
```

**反例**：如果用向量 RAG，`short`（短路）和 `open_circuit`（开路）在向量空间可能高相似，导致误匹配。

### 3.2 为什么用 Orchestrator-Worker 模式？

**对比方案**：

| 方案 | 实现 | 问题 |
|------|------|------|
| 线性脚本 | `if-else` 顺序调用 | 无状态管理、无重试、无断点续传 |
| 纯 Chain | LangChain LCEL | 不支持回环、条件路由复杂 |
| **LangGraph StateGraph** | **有向图 + 条件边** | **天然支持回环、路由、状态增量** |

**关键优势**：

- **回环**：Worker → Decision 汇报 → Decision 决定下一步
- **条件路由**：`add_conditional_edges` 根据 State 动态选择
- **状态增量**：每个节点返回 dict，State 自动 merge
- **断点续传**：State 键存在性判断进度

### 3.3 YOLOv8n 特征提取方案

**问题**：YOLOv8 Neck 是 DAG 结构（含 Concat 多输入模块），常规 Hook 会被 `predict()` 静默跳过。

**方案：物理截断模型**

```python
# 1. 找输出通道为 256 的最后一个 C2f 层（idx=21）
target_idx = _find_neck_layer(model, 256)

# 2. 用 nn.ModuleList 封装前 22 层
truncated = nn.ModuleList([model.model.model[i] for i in range(target_idx + 1)])

# 3. 逐层前向，多输入模块从缓存取
y = {}
for i, m in enumerate(truncated):
    if hasattr(m, "f") and m.f != -1:
        x = y[m.f]  # 从缓存取输入
    x = m(x)
    y[i] = x
```

**特征后处理**：

- 全局平均池化：`feat_map.mean(dim=(2,3)).squeeze(0)` → `[256]`
- L2 归一化：`vec / np.linalg.norm(vec)`
- 存储：Milvus Lite + COSINE 度量 + FLAT 索引

**为什么不用原始图片**：
- 像素维度高、冗余大
- 对光照、角度、背景敏感
- 像素距离不等于语义相似
- 特征向量压缩语义，同类缺陷在向量空间更近

### 3.4 State Schema 设计

**`PCBQualityState`（TypedDict）**：

```python
class PCBQualityState(TypedDict, total=False):
    # ── 输入 ──
    image_path: str
    
    # ── Visual Agent 输出 ──
    defects: list[dict]           # [{"defect_id", "type", "confidence", "bbox"}]
    feature_vectors: list[list[float]]  # 256 维向量
    visual_done: bool             # 显式执行标记
    
    # ── DB Store ──
    db_write_count: int           # >0 成功 / 0 未执行 / -1 失败
    
    # ── Standard Agent ──
    matched_standards: list[dict]
    
    # ── RootCause Agent ──
    root_cause: str
    
    # ── Report Agent ──
    final_report: str
    
    # ── 路由 ──
    next_step: str                # "visual" | "db_store" | "standard" | "rootcause" | "report" | "END"
    error: str                    # 异常信息
```

**关键字段**：

- **`visual_done: bool`** —— 显式执行标记，避免空列表被当 falsy 导致死循环
- **`db_write_count: int`** —— 哨兵值（>0 成功 / 0 未执行 / -1 失败）

### 3.5 Decision Agent 路由逻辑

```python
def decision_agent(state):
    # 1. 异常短路
    if state.get("error"):
        return {"next_step": "END"}
    
    # 2. Visual 未执行 → 执行
    if not state.get("visual_done"):
        return {"next_step": "visual"}
    
    # 3. 有缺陷但未写 DB → 写 DB
    if state.get("defects"):
        db_count = state.get("db_write_count", 0)
        if db_count == 0:
            return {"next_step": "db_store"}
        # db_count == -1 时降级跳过
    
    # 4. 未匹配标准 → 匹配
    if "matched_standards" not in state:
        return {"next_step": "standard"}
    
    # 5. 未分析根因 → 分析
    if "root_cause" not in state:
        return {"next_step": "rootcause"}
    
    # 6. 未生成报告 → 生成
    if "final_report" not in state:
        return {"next_step": "report"}
    
    return {"next_step": "END"}
```

---

## 四、目录结构

```text
fastapi-app/
├── app/                          # 核心代码
│   ├── main.py                  # FastAPI 入口 + SSE 端点
│   ├── graph.py                 # LangGraph 5-Agent 编排
│   ├── state.py                 # PCBQualityState 唯一状态源
│   ├── db.py                    # Milvus Lite 连接 + Upsert
│   ├── feature_extractor.py     # YOLOv8 截断模型特征提取
│   ├── knowledge_base.py        # IPC YAML 单例
│   ├── llm_client.py            # DeepSeek 客户端配置
│   ├── root_cause_agent.py      # 根因分析封装
│   ├── schemas.py               # Pydantic 数据模型
│   └── static/
│       └── index.html           # 单页前端（Tailwind + SSE）
│
├── data/                         # 知识库
│   ├── ipc_standards.yaml       # IPC-A-610G 6 类缺陷标准
│   └── pcb_dataset.yaml         # YOLOv8 训练配置
│
├── scripts/                      # 数据处理脚本
│   ├── convert_coco_to_yolo.py
│   └── split_images.py
│
├── tests/                        # 单元测试
│   ├── test_feature_extractor.py
│   ├── test_knowledge_base.py
│   └── test_root_cause_agent.py
│
├── runs/detect/runs/pcb_detect/train_v1/weights/
│   └── best.pt                  # 微调后的 YOLOv8n 权重（5.97 MB）
│
├── Dockerfile
├── .dockerignore
├── .env.example
├── requirements.txt
├── requirements-dev.txt
└── start.sh
```

---

## 五、Docker 部署

### 5.1 docker-compose.yml

```yaml
name: smart-agent

services:
  api:
    build:
      context: ./fastapi-app
      dockerfile: Dockerfile
    container_name: pcb_api
    restart: unless-stopped
    ports:
      - "8000:8000"
    env_file:
      - ./fastapi-app/.env
    environment:
      - MODEL_PATH=/app/models/best.pt
    volumes:
      - ./fastapi-app/runs/detect/runs/pcb_detect/train_v1/weights/best.pt:/app/models/best.pt:ro
      - ./fastapi-app/uploaded_images:/app/uploaded_images
      - ./fastapi-app/milvus_data.db:/app/milvus_data.db
      - ./fastapi-app/app:/app/app
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 60s
      timeout: 5s
      retries: 3
      start_period: 30s
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 5.2 关键工程实践

| 实践 | 说明 |
|------|------|
| **`.dockerignore` 优化** | 排除 `data/datasets/`（907MB），build context 从 2.87GB → 10MB，构建时间 24min → 2-3min |
| **Volume 挂载代码** | 开发阶段改代码只需 `docker restart`（5s），不用 `docker compose build`（2-3min） |
| **GPU 穿透** | `deploy.resources.reservations.devices` 让容器访问宿主机 GPU |
| **健康检查** | 每 60s 检查一次 `/health`，`start_period=30s` 给模型加载留时间 |
| **`.env` 运行时注入** | `env_file` 挂载，不进镜像，符合 12-Factor App |

### 5.3 一键启动脚本

**`start.ps1`**：Docker 检测 → 幂等启动 → 健康检查轮询 → 自动打开浏览器

```powershell
cd D:\smart-agent
.\start.ps1     # 启动
.\stop.ps1      # 停止
```

---

## 六、性能优化

### 6.1 推理尺寸调优

| 问题 | 现象 | 修复 |
|------|------|------|
| 训练 `imgsz=640`，PCB 原图 2592×1944 | letterbox 后缺陷像素太小，同一张图从 0 检出 | 强制 `INFERENCE_IMGSZ=1280`，检出 5 个缺陷 |

### 6.2 置信度阈值策略

- **推理层**：`CONF_THRESHOLD=0.1`（低阈值保召回）
- **展示层**：`DISPLAY_CONF_THRESHOLD=0.20`（双重过滤）
- **保底**：全部低于阈值时保留最高分缺陷，防止下游收到空列表

### 6.3 异步化改造

| 操作 | 异步方案 |
|------|---------|
| YOLO 推理 | `asyncio.to_thread()` 放入线程池 |
| Milvus 写入 | `asyncio.to_thread()` |
| DeepSeek 调用 | `await _llm.ainvoke()` 原生异步 |
| FastAPI 端点 | `async def` + `run_in_threadpool` |

**目的**：保证 SSE 心跳和并发请求不被阻塞。

---

## 七、关键踩坑记录

| # | 问题 | 根因 | 修复 |
|---|------|------|------|
| 1 | 新版 `index.html` 不生效 | 容器里是旧版，磁盘是新版 | 重建镜像 |
| 2 | 缺 `slowapi` 依赖 | `requirements.txt` 未声明 | 加 `slowapi` + `limits` |
| 3 | `KeyError: 'label'` | 字段名硬访问 | `fd.get("label") or fd.get("type")` |
| 4 | LangSmith 403 刷屏 | key 无效但 tracing 开着 | `.env` 设 `LANGCHAIN_TRACING_V2=false` |
| 5 | 业务日志被过滤 | 默认 WARNING 级别 | `logging.basicConfig(level=INFO)` |
| 6 | build context 2.87GB | `.dockerignore` 没排 `data/datasets/` | 排除数据集 |
| 7 | 容器名前缀异常 | compose 在父目录 | compose 加 `name: smart-agent` |
| 8 | 改代码要重建镜像 | Dockerfile `COPY app/` 固化代码 | Volume 挂载 `./app:/app/app` |
| 9 | `.gitignore` 把 best.pt 忽略 | 父目录被忽略，`!` 否定无效 | 逐层 `!` 解锁 |
| 10 | `GraphRecursionError` 死循环 | 空列表被当 falsy | 键存在性判断 + `visual_done` |

---

## 八、面试话术

### 8.1 一句话介绍

> "基于 LangGraph 5-Agent Orchestrator-Worker 架构的 PCB 智能质检平台，YOLOv8n 微调 mAP@50 达 0.896，IPC-A-610G 知识库 O(1) 精确匹配，DeepSeek 根因推理，Docker Compose + GPU 穿透部署。"

### 8.2 核心亮点

1. **5-Agent 编排**：Decision 作为 Supervisor 动态路由，Worker 完成后强制回环汇报
2. **特征提取**：截断模型提取 256 维特征，L2 归一化 + COSINE 度量
3. **YAML 精确匹配**：摒弃向量 RAG，O(1) 精确匹配，零幻觉
4. **异常隔离**：单节点 try-except + error 短路，断点续传
5. **Docker 化**：GPU 穿透 + Volume 挂载 + 健康检查

### 8.3 工程难点

- **GraphRecursionError 死循环排查**：空列表被当 falsy → 键存在性判断
- **Docker build context 2.87GB 优化**：`.dockerignore` 排除数据集 → 10MB
- **YOLOv8 Hook 失效**：Neck DAG 结构 → 截断模型逐层执行

---

## 九、License

MIT