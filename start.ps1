# PCB 智能质检平台 - 一键启动脚本
# 用法: .\start.ps1

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  🔬 PCB 智能质检多 Agent 平台 - 启动" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: 检查 Docker Desktop ──
Write-Host "[1/4] 检查 Docker Desktop..." -ForegroundColor Yellow
try {
    docker info *>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker 未运行"
    }
    Write-Host "      ✅ Docker 正在运行" -ForegroundColor Green
} catch {
    Write-Host "      ❌ Docker Desktop 未启动" -ForegroundColor Red
    Write-Host ""
    Write-Host "请先启动 Docker Desktop，等待鲸鱼图标变绿后重试。" -ForegroundColor Yellow
    Write-Host "或执行: Start-Process 'Docker Desktop'" -ForegroundColor Gray
    Write-Host ""
    exit 1
}

# ── Step 2: 检查并启动容器 ──
Write-Host ""
Write-Host "[2/4] 启动容器..." -ForegroundColor Yellow

$containerStatus = docker ps --filter "name=pcb_api" --format "{{.Status}}" 2>$null

if ($containerStatus -match "Up.*healthy") {
    Write-Host "      ✅ 容器已在运行 (healthy)" -ForegroundColor Green
} elseif ($containerStatus -match "Up") {
    Write-Host "      容器已启动但未就绪: $containerStatus" -ForegroundColor Gray
} else {
    if ($containerStatus) {
        Write-Host "      当前状态: $containerStatus" -ForegroundColor Gray
    }
    
    $result = docker compose up -d 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "      ❌ 容器启动失败" -ForegroundColor Red
        Write-Host ""
        Write-Host $result -ForegroundColor DarkGray
        exit 1
    }
    Write-Host "      ✅ 容器已启动" -ForegroundColor Green
}

# ── Step 3: 等待健康检查 ──
Write-Host ""
Write-Host "[3/4] 等待健康检查..." -ForegroundColor Yellow

$maxWait = 90
$elapsed = 0
$healthy = $false
$lastStatus = ""

while ($elapsed -lt $maxWait) {
    Start-Sleep -Seconds 3
    $elapsed += 3
    
    $status = docker ps --filter "name=pcb_api" --format "{{.Status}}" 2>$null
    
    if ($status -match "healthy") {
        $healthy = $true
        Write-Host "      ✅ 容器已就绪 (耗时 ${elapsed}s)" -ForegroundColor Green
        break
    } elseif ($status -match "Restarting|Exited") {
        Write-Host "      ⚠️  容器异常: $status" -ForegroundColor Red
        break
    } elseif ($status -ne $lastStatus) {
        Write-Host "      ⏳ ${elapsed}s: $status" -ForegroundColor Gray
        $lastStatus = $status
    }
}

if (-not $healthy) {
    Write-Host ""
    Write-Host "❌ 健康检查超时或容器异常" -ForegroundColor Red
    Write-Host ""
    Write-Host "查看日志定位问题：" -ForegroundColor Yellow
    Write-Host "  docker logs pcb_api --tail 50" -ForegroundColor White
    Write-Host ""
    exit 1
}

# ── Step 4: 打开浏览器 ──
Write-Host ""
Write-Host "[4/4] 打开浏览器..." -ForegroundColor Yellow
Start-Process "http://localhost:8000"
Write-Host "      ✅ 已打开 http://localhost:8000" -ForegroundColor Green

# ── 完成 ──
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  ✅ 启动完成！" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  🎯 主界面:    http://localhost:8000" -ForegroundColor White
Write-Host "  📋 健康检查:  http://localhost:8000/health" -ForegroundColor White
Write-Host "  📖 API 文档:  http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "  实时日志: docker logs pcb_api -f" -ForegroundColor DarkGray
Write-Host "  停止服务: .\stop.ps1" -ForegroundColor DarkGray
Write-Host ""