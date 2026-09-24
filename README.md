# 🔬 PCB 智能质检多 Agent 协同平台

> LangGraph 5-Agent Orchestrator-Worker 架构 · YOLOv8n + DeepSeek + IPC-A-610G 知识库 · SSE 流式反馈 · Docker 一键部署

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange)](https://langchain-ai.github.io/langgraph/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.3-red)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

面向工业质检场景的 PCB 缺陷检测系统：上传一张 PCB 图片，5 个 Agent 协同完成 **缺陷检测 → 标准匹配 → 根因分析 → 处置决策 → 报告生成** 全链路，SSE 实时流式反馈。

## ✨ 核心亮点

- **5-Agent Orchestrator-Worker 编排**：Decision Agent 作为 Supervisor，统一调度 4 个 Worker，所有 Worker 完成后强制回环汇报
- **YOLOv8n 微调**：在 PKU-Market-PCB 数据集（6 类工业缺陷）上微调，mAP@50 达 **0.896**，Precision **0.943**，单帧推理 **1.2ms**（RTX 4060 Ti）
- **特征提取 + Milvus Lite 向量库**：从 YOLOv8n Neck 末端 Hook Layer 21 提取 **256 维特征**，L2 归一化后存 Milvus Lite，支持历史案例检索
- **IPC-A-610G 知识库（YAML O(1) 匹配）**：摒弃向量 RAG，采用结构化 YAML + 内存字典实现精确匹配，检索延迟 **<1ms**，零幻觉风险
- **DeepSeek 根因推理**：结构化 Prompt 注入缺陷列表 + IPC 证据链，基于工业知识做工程根因分析
- **SSE 流式反馈**：`astream_events` 逐 Token 推送报告生成过程，前端实时打字机效果
- **接口限流 + 文件校验**：slowapi 单 IP 5 次/分钟限流 + 10MB 文件大小限制 + 格式白名单
- **Docker Compose + GPU 穿透**：一键部署，容器内 CUDA 12.6 + RTX 4060 Ti

## 📐 系统架构

```text
┌─────────────────────────────────────────────────────────────┐
│                        START                                │
│                          ↓                                  │
│              ┌───────────────────────┐                      │
│              │   Decision Agent      │  (Supervisor)        │
│              │   动态路由 + 状态检查   │                      │
│              └───────────┬───────────┘                      │
│                          │                                  │
│      ┌───────────────────┼───────────────────┐              │
│      ▼                   ▼                   ▼              │
│  ┌────────┐         ┌─────────┐        ┌──────────┐        │
│  │Visual  │         │Standard │        │RootCause │        │
│  │YOLOv8n │         │IPC YAML │        │DeepSeek  │        │
│  │256-dim │         │ O(1)    │        │ 推理     │        │
│  └────┬───┘         └────┬────┘        └─────┬────┘        │
│       │                  │                   │              │
│       │        ┌─────────┴────────┐          │              │
│       │        │  DB Store        │          │              │
│       │        │  Milvus Lite     │          │              │
│       │        └─────────┬────────┘          │              │
│       └──────────────────┼───────────────────┘              │
│                          ▼                                  │
│                 ┌─────────────────┐                         │
│                 │  Report Agent   │                         │
│                 │  Markdown 模板   │                         │
│                 └────────┬────────┘                         │
│                          ▼                                  │
│                        END                                  │
└─────────────────────────────────────────────────────────────┘
```

**编排特性**：
- Worker 完成后通过回边 `add_edge("visual", "decision")` 强制回到 Decision
- Decision 通过 **State 键存在性**判断下一步（避免空结果误判死循环）
- 异常隔离：单节点失败写入 `state["error"]` 并短路到 END
- 支持断点续传：中断后重新 invoke 同一 State，从断点继续

## 🚀 快速开始

### 前置依赖

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | 3.13 | 推荐 conda 环境隔离 |
| Docker Desktop | 最新 | 用于容器化部署 |
| NVIDIA GPU | 可选 | 推荐 RTX 3060+，用于 YOLOv8 推理 |
| CUDA | 12.6 | 与 PyTorch cu126 匹配 |
| DeepSeek API Key | - | [注册地址](https://platform.deepseek.com/) |

### 方式一：Docker Compose（推荐）

```bash
# 1. 克隆项目
git clone https://github.com/banzhikaoyuOvo/pcb-quality-agent.git
cd pcb-quality-agent

# 2. 配置环境变量
cp fastapi-app/.env.example fastapi-app/.env
# 编辑 fastapi-app/.env，填入 DEEPSEEK_API_KEY

# 3. 一键启动（推荐）
.\start.ps1

# 或手动启动
docker compose up -d

# 4. 等 30 秒确认健康状态
docker ps | grep pcb_api
# 应显示 (healthy)

# 5. 浏览器访问
# http://localhost:8000
```

### 方式二：本地开发

```bash
# 1. 创建 conda 环境
conda create -n pcb python=3.13 -y
conda activate pcb

# 2. 安装依赖
cd fastapi-app
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 配置环境变量
cp .env.example .env
# 填入 DEEPSEEK_API_KEY

# 4. 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 5. 浏览器访问
# http://localhost:8000
```

### 验证服务

```bash
# 健康检查
curl http://localhost:8000/health
# 预期: {"status":"ok"}

# 同步质检接口（返回完整 JSON）
curl -F "file=@path/to/pcb.jpg" http://localhost:8000/inspect

# 流式质检接口（SSE）
curl -N -F "file=@path/to/pcb.jpg" http://localhost:8000/inspect_stream
```

## 🧠 核心设计

### 5-Agent 职责

| Agent | 技术 | 输入 | 输出 |
|-------|------|------|------|
| **Visual** | YOLOv8n + Hook Layer 21 | PCB 图片 | 缺陷列表 + 256 维特征向量 |
| **DB Store** | pymilvus-lite | 特征向量 | Milvus Upsert 结果 |
| **Standard** | YAML 内存字典 | 缺陷类型 | IPC-A-610G 标准条目 |
| **RootCause** | DeepSeek API | 缺陷 + 标准 | 根因分析（500+ 字） |
| **Report** | Python f-string | 全部结果 | Markdown 报告 |

### 为什么用 YAML 精确匹配而非向量 RAG？

**决策依据**：

| 维度 | 向量 RAG | YAML 字典 |
|------|---------|-----------|
| 适用场景 | 非结构化文档 | 结构化标准条目 |
| 检索方式 | 语义相似 | 精确匹配 |
| 检索延迟 | 10-100ms | **<1ms** |
| 准确率 | 85-95% | **100%** |
| 幻觉风险 | 有 | **零** |

**PCB 缺陷类型是确定的 6 类枚举值，IPC 标准按缺陷类型一一对应**，无需语义检索。选型原则：**查询键确定性 + 结果唯一性 → 精确匹配；查询键模糊 + 结果多样 → 向量检索**。

### 特征提取方案

**YOLOv8 Neck 是 DAG 结构**（含 Concat 多输入模块），常规 Hook 会被 `predict()` 静默跳过。

**最终方案：物理截断模型**：
```python
# 找到输出通道为 256 的最后一个 C2f 层（idx=21）
modules = [model.model.model[i] for i in range(22)]
truncated = nn.ModuleList(modules).eval()

# 逐层前向，多输入模块从缓存取
for i, m in enumerate(truncated):
    if hasattr(m, "f") and m.f != -1:
        x = y[m.f]  # 从缓存取
    x = m(x)
    y[i] = x
```

**特征后处理**：
- 全局平均池化：`feat_map.mean(dim=(2,3)).squeeze(0)` → `[256]`
- L2 归一化：`vec / np.linalg.norm(vec)`
- COSINE 度量 + FLAT 索引（Milvus Lite 小数据集推荐）

### SSE 流式协议

后端通过 `astream_events(version="v2")` 捕获细粒度事件：

| 事件类型 | 触发时机 | 前端展示 |
|---------|---------|---------|
| `start` | 任务接收 | "🚀 任务已接收" |
| `node_start` | 每个 Agent 开始 | "👁️ 视觉 Agent 正在运行..." |
| `llm_token` | LLM 逐 Token 输出 | 打字机逐字显示 |
| `complete` | 全流程结束 | "✅ 流水线执行完毕" |
| `error` | 异常 | "❌ 错误信息" |

前端用 `response.body.getReader()` 解析 SSE 流，Markdown 渲染器 `StreamMarkdownRenderer` 做防抖 + 代码块检测 + 打字机光标。

## 📊 技术指标

| 指标 | 值 | 环境 |
|------|-----|------|
| YOLOv8n mAP@50 | **0.896** | PKU-Market-PCB 数据集 |
| YOLOv8n Precision | **0.943** | - |
| 单帧推理延迟 | **1.2ms** | RTX 4060 Ti |
| 特征向量维度 | **256** | Neck Hook Layer 21 |
| IPC 知识库检索 | **<1ms** | YAML O(1) |
| LLM 根因分析 | **<3s** | DeepSeek API |
| 报告生成 | **<1ms** | 纯 Python 模板 |
| 端到端单次质检 | **3-5s** | 含 YOLO + LLM 调用 |

## 📁 项目结构

```text
pcb-quality-agent/
├── docker-compose.yml          # Docker Compose 编排
├── milvus.yaml                 # Milvus 配置（WSL2 调优）
├── LICENSE
├── README.md
│
└── fastapi-app/
    ├── app/                    # 核心代码
    │   ├── main.py            # FastAPI 入口 + SSE 端点
    │   ├── graph.py           # LangGraph 5-Agent 编排
    │   ├── state.py           # PCBQualityState 唯一状态源
    │   ├── db.py              # Milvus Lite 连接 + Upsert
    │   ├── feature_extractor.py # YOLOv8 截断模型特征提取
    │   ├── knowledge_base.py  # IPC YAML 单例
    │   ├── llm_client.py      # DeepSeek 客户端配置
    │   ├── root_cause_agent.py # 根因分析封装
    │   ├── schemas.py         # Pydantic 数据模型
    │   └── static/
    │       └── index.html     # 单页前端（Tailwind + SSE）
    │
    ├── data/                   # 知识库
    │   ├── ipc_standards.yaml # IPC-A-610G 6 类缺陷标准
    │   └── pcb_dataset.yaml   # YOLOv8 训练配置
    │
    ├── scripts/                # 数据处理脚本
    │   ├── convert_coco_to_yolo.py
    │   └── split_images.py
    │
    ├── tests/                  # 单元测试
    │   ├── test_feature_extractor.py
    │   ├── test_knowledge_base.py
    │   └── test_root_cause_agent.py
    │
    ├── runs/detect/runs/pcb_detect/train_v1/weights/
    │   └── best.pt            # 微调后的 YOLOv8n 权重（5.97 MB）
    │
    ├── Dockerfile
    ├── .dockerignore
    ├── .env.example
    ├── requirements.txt
    ├── requirements-dev.txt
    └── start.sh
```

## 🛠️ 技术栈

| 层级 | 组件 | 版本 |
|------|------|------|
| **运行时** | Python | 3.13 |
| **Web 框架** | FastAPI | 0.115.6 |
| **ASGI Server** | Uvicorn | 0.34.0 |
| **Agent 编排** | LangGraph | 0.2.61 |
| **LLM 框架** | LangChain | 0.3.14 |
| **向量库** | pymilvus-lite | 3.2+ |
| **视觉模型** | Ultralytics YOLOv8 | 8.3.40 |
| **深度学习** | PyTorch | 2.14.0 + CUDA 12.6 |
| **LLM** | DeepSeek V3 | API |
| **限流** | slowapi | 0.1.9 |
| **容器化** | Docker Compose | 最新 |

## 🎯 演示示例

### 输入

上传一张 PCB 图片（例如 `01_missing_hole_08.jpg`），前端调用 `/inspect_stream`。

### 前端展示

```text
┌────────────────────────────────────────────┐
│ 1. 上传 PCB 图像                            │
│ [预览图]                                    │
│ [🚀 开始智能质检]                            │
├────────────────────────────────────────────┤
│ 2. Agent 执行流水线                          │
│ 🎯 决策 Agent: 规划下一步...                 │
│ 👁️ 视觉 Agent: 正在运行 YOLO 检测...         │
│ 💾 数据库 Agent: 正在将特征写入 Milvus...    │
│ 📖 标准 Agent: 正在匹配 IPC 规则库...        │
│ 🧠 根因 Agent: 正在调用 DeepSeek...         │
│ 📝 报告 Agent: 正在生成 Markdown 报告...     │
├────────────────────────────────────────────┤
│ 3. 结构化质检报告                            │
│ [打字机逐字生成 Markdown 报告]                │
└────────────────────────────────────────────┘
```

### 输出报告结构

```markdown
# PCB 质检报告

## 检测概要
- 图片：01_missing_hole_08.jpg
- 缺陷数量：1 个
- 检测时间：2026-09-24 06:45:07

## 缺陷明细

| # | 类型 | 置信度 | IPC 等级 | 处置建议 |
|---|------|--------|---------|---------|
| 1 | open_circuit | 27.19% | IPC-A-610G 6.1.1 | 对于非 BGA 区域的简单走线，可使用飞线按 IPC-7711/21 返工 |

## 根因分析
[DeepSeek 基于 IPC 证据链的 500+ 字分析]

## 综合判定
[处置决策 + 复核建议]
```

## 🐳 Docker 部署要点

### 容器编排

```yaml
services:
  api:
    build: ./fastapi-app
    container_name: pcb_api
    ports: ["8000:8000"]
    volumes:
      - ./fastapi-app/app:/app/app              # 代码热更新（开发）
      - ./fastapi-app/uploaded_images:/app/uploaded_images
      - ./fastapi-app/milvus_data.db:/app/milvus_data.db
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 工程实践

- **`.dockerignore` 优化**：排除 `data/datasets/`（907MB）、`runs/`、`weights/`，build context 从 **2.87GB → 10MB**，构建时间从 24min → 2-3min
- **Volume 挂载代码**：开发阶段改代码只需 `docker restart`（5s），不用 `docker compose build`（2-3min）
- **LangSmith tracing 关闭**：`.env` 设 `LANGCHAIN_TRACING_V2=false`，避免无效 key 导致 403 刷屏
- **日志分级**：`logging.basicConfig(level=INFO)` + 第三方库降噪

### 运维命令

```bash
# ⚠️ 必须在父目录执行（compose 文件位置）
cd pcb-quality-agent

# 一键启动/停止（Windows）
.\start.ps1              # 启动（含健康检查 + 自动开浏览器）
.\stop.ps1               # 停止

# 或手动操作
docker compose up -d
docker compose stop
docker compose restart

# 重建镜像（改了 requirements.txt 或 Dockerfile）
docker compose build
docker compose up -d

# 只改代码（挂了 volume）
docker restart pcb_api

# 日志
docker logs pcb_api --tail 50
docker logs pcb_api -f

# GPU 验证
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

## 📦 数据集与模型

### 数据集

- **来源**：[PKU-Market-PCB](https://www.kaggle.com/datasets/akhatova/pcb-defects)
- **规模**：693 张图片，6 类缺陷
- **类别**：`missing_part` / `bent_lead` / `crack` / `short_circuit` / `wrong_placement` / `contamination`
- **说明**：数据集未上传仓库（907MB），需自行下载

### 模型权重

- **`best.pt`**（5.97 MB）已包含在仓库
- 路径：`fastapi-app/runs/detect/runs/pcb_detect/train_v1/weights/best.pt`

### 重新训练

```bash
# 1. 下载数据集后，COCO → YOLO 格式转换
python scripts/convert_coco_to_yolo.py

# 2. 切分训练/验证集
python scripts/split_images.py

# 3. 微调训练
yolo detect train data=data/pcb_dataset.yaml model=yolov8n.pt epochs=100 imgsz=640

# 4. 权重会保存在 runs/detect/train/weights/best.pt
```

## 🧪 测试

```bash
cd fastapi-app
pytest tests/ -v
```

覆盖：
- `test_feature_extractor.py`：Hook Layer 索引、向量 shape、L2 模长
- `test_knowledge_base.py`：单例模式、YAML 加载、已知/未知缺陷
- `test_root_cause_agent.py`：early return 分支 + LLM 真实调用



## 📄 License

MIT

## 👤 作者

**俞晓兴** · AI 应用开发工程师

- GitHub: [@banzhikaoyuOvo](https://github.com/banzhikaoyuOvo)
- 邮箱：3341428887@qq.com

---

**如果这个项目对你有帮助，欢迎 ⭐ Star！**