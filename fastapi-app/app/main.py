# app/main.py

from dotenv import load_dotenv
load_dotenv()  # 必须在 import graph 之前执行

import shutil
import json
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.runnables import RunnableConfig

# ✅ 新增：接口限流
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app import db
from app.graph import graph
import logging

# 配置 root logger（让业务日志可见）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# 降低第三方库日志级别，避免刷屏
logging.getLogger("langsmith").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# ✅ 初始化限流器：按客户端 IP 限流
limiter = Limiter(key_func=get_remote_address)

_BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = _BASE_DIR / "uploaded_images"
STATIC_DIR = _BASE_DIR / "app" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    db.init_db()
    print("🚀 FastAPI 启动完成，Milvus Lite 集合就绪")
    yield
    try:
        db.client.close()
    except Exception:
        pass
    print("👋 FastAPI 已关闭")


app = FastAPI(
    title="智能制造质检多 Agent 协同平台",
    description="PCB 缺陷检测子场景 · LangGraph 6-Agent 编排 (SSE 流式版)",
    version="0.4.0",  # ✅ 版本号升级
    lifespan=lifespan,
)

# ✅ 挂载限流器与静态文件
app.state.limiter = limiter
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ✅ 全局限流异常处理
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "请求过于频繁，请稍后再试 (上限: 5次/分钟)"}
    )


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    """纯内存健康检查端点"""
    return {"status": "ok"}


# ✅ 核心改动：增加 @limiter.limit 装饰器
@app.post("/inspect_stream")
@limiter.limit("5/minute")  # 🔒 企业级安全：单IP每分钟最多5次
async def inspect_pcb_stream(request: Request, file: UploadFile = File(...)):
    """PCB 质检 SSE 流式入口（R42 安全加固 + 可观测性增强版）"""
    import uuid

    # ✅ 1. 安全校验：文件大小限制 10MB
    MAX_FILE_SIZE = 10 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"文件过大: {len(content)} bytes，上限 10MB")

    allowed = {".jpg", ".jpeg", ".png", ".bmp"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {suffix}")

    UPLOAD_DIR.mkdir(exist_ok=True)
    save_path = UPLOAD_DIR / file.filename
    save_path.write_bytes(content)

    trace_id = str(uuid.uuid4())[:8]
    print(f"🔍 [{trace_id}] SSE 请求开始: {file.filename}, size={len(content)}", flush=True)

    async def event_generator():
        yield f"data: {json.dumps({'event': 'start', 'trace_id': trace_id, 'message': '🚀 任务已接收，正在初始化流水线...'})}\n\n"

        initial_state = {"image_path": str(save_path)}
        # ✅ LangSmith 会自动捕获 metadata 中的 trace_id
        config = RunnableConfig(callbacks=[], metadata={"trace_id": trace_id})
        final_state = {}
        report_has_streamed = False

        try:
            async for event in graph.astream_events(initial_state, config=config, version="v2"):
                kind = event.get("event")
                metadata = event.get("metadata", {})
                node_name = metadata.get("langgraph_node", "")

                if kind == "on_chain_start" and node_name:
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
                    if node_name == "report":
                        report_has_streamed = False
                    await asyncio.sleep(0.02)

                elif kind == "on_chat_model_stream":
                    chunk_content = event["data"]["chunk"].content
                    if chunk_content and isinstance(chunk_content, str):
                        current_node = metadata.get("langgraph_node", "")
                        if current_node == "report":
                            report_has_streamed = True
                        yield f"data: {json.dumps({'event': 'llm_token', 'content': chunk_content})}\n\n"

                elif kind == "on_chain_end" and node_name == "report":
                    output = event.get("data", {}).get("output", {})
                    report_text = None
                    if isinstance(output, dict):
                        report_text = (
                            output.get("final_report")
                            or output.get("report")
                            or (output.get("messages", [{}])[-1].get("content") if output.get("messages") else None)
                        )
                    elif isinstance(output, str):
                        report_text = output

                    if report_text and not report_has_streamed:
                        chunk_size = 20
                        for i in range(0, len(report_text), chunk_size):
                            chunk = report_text[i:i + chunk_size]
                            yield f"data: {json.dumps({'event': 'llm_token', 'content': chunk})}\n\n"
                            await asyncio.sleep(0.03)
                    if isinstance(output, dict):
                        final_state.update(output)

                elif kind == "on_chain_end" and node_name and node_name != "report":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        final_state.update(output)

            yield f"data: {json.dumps({'event': 'complete', 'trace_id': trace_id, 'state': final_state})}\n\n"
            print(f"✅ [{trace_id}] SSE 请求完成", flush=True)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ [{trace_id}] SSE 异常: {e}", flush=True)
            yield f"data: {json.dumps({'event': 'error', 'trace_id': trace_id, 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/inspect")
@limiter.limit("5/minute")  # 🔒 同步接口同样限流
async def inspect_pcb(request: Request, file: UploadFile = File(...)):
    """PCB 质检同步入口（兼容旧调用）"""
    import sys, traceback
    
    print(f"📥 [/inspect] 收到请求: {file.filename}, size={file.size}", flush=True)
    
    allowed = {".jpg", ".jpeg", ".png", ".bmp"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {suffix}")

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


@app.get("/ready")
def readiness():
    """深度健康检查：验证所有外部依赖是否可用"""
    checks = {}
    try:
        from app.db import client
        client.list_collections()
        checks["milvus"] = "ok"
    except Exception as e:
        checks["milvus"] = f"error: {str(e)}"
    
    all_ok = all(v == "ok" for v in checks.values())
    status_code = 200 if all_ok else 503
    return JSONResponse(content=checks, status_code=status_code)