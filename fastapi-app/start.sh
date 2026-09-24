#!/bin/bash
# ✅ 修改：移除 Gradio 后台启动，仅保留 Uvicorn
# 原代码可能类似：python -m app.gradio_app & 
# 现在直接删除上面那行，只保留下面的 Uvicorn 启动命令

echo "🚀 Starting FastAPI server with SSE support..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1