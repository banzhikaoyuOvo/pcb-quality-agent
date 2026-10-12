# app/main.py
"""
FastAPI 应用入口
=================

【职责】
    暴露 HTTP API（同步 + SSE 流式），托管前端静态文件，
    限流保护，以及将 LangGraph 工作流的执行过程通过 SSE 推送给前端。

【关键设计】
    1. lifespan: 启动时初始化 Milvus 集合，关闭时清理连接
    2. 双端点: /inspect (同步) 和 /inspect_stream (SSE 流式)
    3. astream_events v2: 捕获节点开始、LLM Token、节点结束等细粒度事件
    4. slowapi 限流: 单 IP 5 次/分钟，防止滥用
    5. 文件校验: 10MB 大小上限 + 格式白名单

【启动顺序注意】
    load_dotenv() 必须在 import graph 之前执行，因为 graph.py 在模块级
    读取环境变量（os.getenv）初始化 LLM 客户端和模型路径。
"""

# -----------------------------------------------------------------------------
# ⚠️ 必须最先执行: 加载 .env，且必须在 import graph 之前
# -----------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

import shutil
import json
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.runnables import RunnableConfig

# ✅ 接口限流
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app import db
from app.graph import graph
import logging

# -----------------------------------------------------------------------------
# 日志配置
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# 降低第三方库日志级别，避免刷屏
logging.getLogger("langsmith").setLevel(logging.CRITICAL)   # 关闭无效 key 的 403 刷屏
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# -----------------------------------------------------------------------------
# 限流器初始化
# -----------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)
# key_func=get_remote_address: 按客户端 IP 限流。
# 分布式部署时应改用 Redis 后端（slowapi 支持 storage_uri 参数）。


# -----------------------------------------------------------------------------
# 路径常量
# -----------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = _BASE_DIR / "uploaded_images"
STATIC_DIR = _BASE_DIR / "app" / "static"


# -----------------------------------------------------------------------------
# 生命周期管理
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期钩子。

    Startup:
        - 创建上传目录和静态目录（幂等）
        - 初始化 Milvus 集合

    Shutdown:
        - 关闭 Milvus 客户端连接

    Why lifespan 而非 @app.on_event("startup")？
        FastAPI 0.93+ 推荐 lifespan，支持 async 上下文管理，
        且 on_event 已 deprecated。
    """
    UPLOAD_DIR.mkdir(exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    db.init_db()
    print("🚀 FastAPI 启动完成，Milvus Lite 集合就绪")
    yield
    # ---------- Shutdown ----------
    try:
        db.client.close()
    except Exception:
        pass
    print("👋 FastAPI 已关闭")


# -----------------------------------------------------------------------------
# FastAPI 应用实例
# -----------------------------------------------------------------------------
app = FastAPI(
    title="智能制造质检多 Agent 协同平台",
    description="PCB 缺陷检测子场景 · LangGraph 6-Agent 编排 (SSE 流式版)",
    version="0.4.0",
    lifespan=lifespan,
)

# 挂载限流器与静态文件
app.state.limiter = limiter
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# -----------------------------------------------------------------------------
# 全局限流异常处理
# -----------------------------------------------------------------------------
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """限流触发时返回 429，附带友好提示。"""
    return JSONResponse(
        status_code=429,
        content={"detail": "请求过于频繁，请稍后再试 (上限: 5次/分钟)"},
    )


# -----------------------------------------------------------------------------
# 基础路由
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    """返回前端单页应用 index.html。"""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    """
    轻量健康检查端点（纯内存，不查任何外部依赖）。

    用途: K8s liveness probe / Docker healthcheck。
    深度健康检查请用 /ready。
    """
    return {"status": "ok"}


# =============================================================================
# SSE 流式质检端点（核心）
# =============================================================================

@app.post("/inspect_stream")
@limiter.limit("5/minute")  # 🔒 企业级安全：单 IP 每分钟最多 5 次
async def inspect_pcb_stream(request: Request, file: UploadFile = File(...)):
    """
    PCB 质检 SSE 流式入口。

    Args:
        request:  FastAPI 注入，供限流器使用
        file:     上传的 PCB 图片

    Returns:
        StreamingResponse (text/event-stream)

    【SSE 事件类型】
        - start:       任务已接收
        - node_start:  某 Agent 开始执行
        - llm_token:   LLM 逐 Token 输出
        - complete:    全流程结束，附带最终 State
        - error:       异常

    【R42 安全加固】
        - 文件大小限制 10MB
        - 扩展名白名单 (.jpg/.jpeg/.png/.bmp)
        - trace_id 日志追踪
    """
    import uuid

    # ---------- 1. 安全校验: 文件大小限制 10MB ----------
    MAX_FILE_SIZE = 10 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大: {len(content)} bytes，上限 10MB",
        )

    # ---------- 2. 安全校验: 扩展名白名单 ----------
    allowed = {".jpg", ".jpeg", ".png", ".bmp"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {suffix}")

    # ---------- 3. 保存上传文件 ----------
    UPLOAD_DIR.mkdir(exist_ok=True)
    save_path = UPLOAD_DIR / file.filename
    save_path.write_bytes(content)

    # ---------- 4. 生成 trace_id 用于日志追踪 ----------
    trace_id = str(uuid.uuid4())[:8]
    print(f"🔍 [{trace_id}] SSE 请求开始: {file.filename}, size={len(content)}", flush=True)

    async def event_generator():
        """
        SSE 事件生成器 —— 包装 graph.astream_events 并转换为 SSE 协议。

        【为什么用 astream_events v2 而非 astream？】
            astream 只推送节点级的状态变更（黑盒）。
            astream_events 推送细粒度事件：
            - on_chain_start/end: 节点开始/结束
            - on_chat_model_stream: LLM Token（实现打字机效果的关键）
            前端体验更好。
        """
        # 首个事件: 告知前端任务已接收
        yield f"data: {json.dumps({'event': 'start', 'trace_id': trace_id, 'message': '🚀 任务已接收，正在初始化流水线...'})}\n\n"

        initial_state = {"image_path": str(save_path)}
        # ✅ LangSmith 会自动捕获 metadata 中的 trace_id，便于跨系统追踪
        config = RunnableConfig(callbacks=[], metadata={"trace_id": trace_id})
        final_state = {}
        report_has_streamed = False  # 标记报告是否已通过 LLM Token 流式输出

        try:
            # ---------- 遍历图事件 ----------
            async for event in graph.astream_events(initial_state, config=config, version="v2"):
                kind = event.get("event")
                metadata = event.get("metadata", {})
                node_name = metadata.get("langgraph_node", "")

                # ---------- 事件 1: 节点开始 ----------
                if kind == "on_chain_start" and node_name:
                    # ✅ 过滤嵌套事件：只处理"节点本身"的 on_chain_start，
                    # 节点内部的 RunnableSequence / RunnableLambda 也会触发
                    # on_chain_start，且 metadata.langgraph_node 相同，
                    # 若不过滤会导致同一节点在流水线面板重复 2-3 次。
                    if event.get("name") != node_name:
                        continue
                    # ✅ 过滤 LangGraph 内部节点（__start__ / __end__ 等）
                    if node_name.startswith("__"):
                        continue
                    node_map = {
                        "decision": "🎯 决策 Agent: 规划下一步...",
                        "visual": "👁️ 视觉 Agent: 正在运行 YOLO 检测与特征提取...",
                        "db_store": "💾 数据库 Agent: 正在将特征写入 Milvus...",
                        "standard": "📖 标准 Agent: 正在匹配 IPC 规则库...",
                        "rootcause": "🧠 根因 Agent: 正在调用 DeepSeek 分析根因...",
                        "report": "📝 报告 Agent: 正在生成结构化 Markdown 报告...",
                    }
                    msg = node_map.get(node_name, f"执行节点: {node_name}")
                    yield f"data: {json.dumps({'event': 'node_start', 'node': node_name, 'message': msg})}\n\n"
                    # report 节点开始时重置标志（防止上一轮干扰）
                    if node_name == "report":
                        report_has_streamed = False
                    await asyncio.sleep(0.02)  # 轻微延迟让前端有时间渲染

                # ---------- 事件 2: LLM Token 流式输出 ----------
                elif kind == "on_chat_model_stream":
                    chunk_content = event["data"]["chunk"].content
                    if chunk_content and isinstance(chunk_content, str):
                        current_node = metadata.get("langgraph_node", "")
                        if current_node == "report":
                            report_has_streamed = True
                            yield f"data: {json.dumps({'event': 'llm_token', 'content': chunk_content})}\n\n"

                # ---------- 事件 3: report 节点结束 ----------
                elif kind == "on_chain_end" and node_name == "report":
                    output = event.get("data", {}).get("output", {})
                    report_text = None
                    if isinstance(output, dict):
                        # 兼容多种字段名（历史版本遗留）
                        report_text = (
                            output.get("final_report")
                            or output.get("report")
                            or (output.get("messages", [{}])[-1].get("content") if output.get("messages") else None)
                        )
                    elif isinstance(output, str):
                        report_text = output

                    # 若 LLM 未流式输出（如异常降级），则模拟分块推送
                    if report_text and not report_has_streamed:
                        chunk_size = 20
                        for i in range(0, len(report_text), chunk_size):
                            chunk = report_text[i:i + chunk_size]
                            yield f"data: {json.dumps({'event': 'llm_token', 'content': chunk})}\n\n"
                            await asyncio.sleep(0.03)
                    if isinstance(output, dict):
                        final_state.update(output)

                # ---------- 事件 4: 其他节点结束（累积 final_state）----------
                elif kind == "on_chain_end" and node_name and node_name != "report":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        final_state.update(output)

            # ---------- 全流程完成 ----------
            yield f"data: {json.dumps({'event': 'complete', 'trace_id': trace_id, 'state': final_state})}\n\n"
            print(f"✅ [{trace_id}] SSE 请求完成", flush=True)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ [{trace_id}] SSE 异常: {e}", flush=True)
            yield f"data: {json.dumps({'event': 'error', 'trace_id': trace_id, 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# =============================================================================
# 同步质检端点（兼容旧调用）
# =============================================================================

@app.post("/inspect")
@limiter.limit("5/minute")  # 🔒 同步接口同样限流
async def inspect_pcb(request: Request, file: UploadFile = File(...)):
    """
    PCB 质检同步入口 —— 一次性返回完整结果。

    Args:
        request: FastAPI 注入（限流用）
        file:    上传图片

    Returns:
        {"message": "质检完成", "state": {...完整 State...}}

    用途:
        - 兼容不支持 SSE 的老客户端
        - 便于 curl / Postman 调试
        - 供 Gradio 等前端框架调用（无需 SSE 解析）
    """
    import sys, traceback

    print(f"📥 [/inspect] 收到请求: {file.filename}, size={file.size}", flush=True)

    # ---------- 格式校验 ----------
    allowed = {".jpg", ".jpeg", ".png", ".bmp"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {suffix}")

    # ---------- 保存文件 ----------
    UPLOAD_DIR.mkdir(exist_ok=True)
    save_path = UPLOAD_DIR / file.filename

    try:
        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"💾 [/inspect] 文件已保存: {save_path}", flush=True)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"❌ [/inspect] 文件保存失败:\n{tb}", file=sys.stderr, flush=True)
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    # ---------- 执行图 ----------
    initial_state = {"image_path": str(save_path)}

    try:
        print(f"🚀 [/inspect] 开始调用 graph.invoke...", flush=True)
        result = graph.invoke(initial_state)
        print(f"✅ [/inspect] graph.invoke 完成, defects={len(result.get('defects', []))}", flush=True)
        return {"message": "质检完成", "state": result}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"❌ [/inspect] graph.invoke 失败:\n{tb}", file=sys.stderr, flush=True)
        print(f"❌ [/inspect] graph.invoke 失败:\n{tb}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 深度健康检查
# =============================================================================

@app.get("/ready")
def readiness():
    """
    深度健康检查 —— 验证所有外部依赖是否可用。

    与 /health 的区别:
        /health  纯内存检查，用于 liveness probe（进程存活即 OK）
        /ready   检查 Milvus 等外部依赖，用于 readiness probe（可接收流量才 OK）

    Returns:
        JSON，状态码 200 (全 OK) 或 503 (有依赖故障)
    """
    checks = {}

    # 检查 Milvus
    try:
        from app.db import client
        client.list_collections()
        checks["milvus"] = "ok"
    except Exception as e:
        checks["milvus"] = f"error: {str(e)}"

    all_ok = all(v == "ok" for v in checks.values())
    status_code = 200 if all_ok else 503
    return JSONResponse(content=checks, status_code=status_code)