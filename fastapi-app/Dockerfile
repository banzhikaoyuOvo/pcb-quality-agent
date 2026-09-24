FROM python:3.13-slim

# 🔥 新增：安装 OpenCV 运行所需的系统级依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgl1 \
    libxrender-dev \
    libgomp1 \
    # libxcb.so.1 库所在的包
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ✅ 回归 R31.2 锁定决策：使用 PyTorch cu126 GPU 版本
RUN pip install --no-cache-dir \
    torch==2.14.0 \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple
    
# ✅ Step 2: 安装其余依赖（走清华源加速，torch 已满足条件会自动跳过）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# ... 后续内容保持不变

# Copy application code & knowledge base
COPY app/ ./app/
COPY data/ ./data/

# ✅ R38-B 新增：复制并赋予执行权限
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Create runtime directories (will be overridden by volume mounts)
RUN mkdir -p uploaded_images milvus_data.db

# Default model path inside container (can be overridden via env/volume)
ENV MODEL_PATH=/app/models/best.pt
ENV PYTHONUNBUFFERED=1

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/docs')" || exit 1

# ✅ R38-B 修改：使用启动脚本替代直接 uvicorn
CMD ["/app/start.sh"]