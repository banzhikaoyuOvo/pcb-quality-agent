# PCB 智能质检平台 - 停止脚本
# 用法: .\stop.ps1

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host ""
Write-Host "停止 PCB 质检平台..." -ForegroundColor Yellow
docker compose stop

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ 已停止" -ForegroundColor Green
    Write-Host ""
    Write-Host "  🔄 重新启动: .\start.ps1" -ForegroundColor DarkGray
    Write-Host "  🗑️  彻底删除: docker compose down" -ForegroundColor DarkGray
    Write-Host ""
} else {
    Write-Host "❌ 停止失败" -ForegroundColor Red
}