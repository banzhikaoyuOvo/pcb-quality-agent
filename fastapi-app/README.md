# fastapi-app

PCB 智能质检多 Agent 协同平台的核心代码目录。

**完整项目介绍请查看**：[../README.md](../README.md)

## 快速导航

- **核心代码**：`app/*.py`
- **前端**：`app/static/index.html`
- **知识库**：`data/ipc_standards.yaml`
- **权重**：`runs/detect/runs/pcb_detect/train_v1/weights/best.pt`
- **测试**：`tests/*.py`

## 启动

```bash
# 本地开发
conda activate pcb
uvicorn app.main:app --reload

# Docker（在父目录执行）
cd ..
docker compose up -d
```

## 相关文档

- [项目完整 README](../README.md)
- [Docker 部署](../docker-compose.yml)