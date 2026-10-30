# PepeShop 1-Click PowerShell Setup Script
$ErrorActionPreference = "Stop"

Write-Host "==============================================================" -ForegroundColor Cyan
Write-Host "  PepeShop - 1-Click Docker Setup & Environment Launcher" -ForegroundColor Cyan
Write-Host "==============================================================" -ForegroundColor Cyan

# 1. Create directories
Write-Host "[1/4] Проверка структуры папок..." -ForegroundColor Yellow
$dirs = @("models", "output", "logs")
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) {
        New-Item -ItemType Directory -Path $d | Out-Null
        Write-Host "  Создана папка: $d" -ForegroundColor Green
    }
}

# 2. Check Docker
Write-Host "`n[2/4] Проверка Docker..." -ForegroundColor Yellow
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "[-] Docker не обнаружен! Установите Docker Desktop: https://www.docker.com/products/docker-desktop/" -ForegroundColor Red
    Pause
    Exit 1
}

# 3. Check Docker status
try {
    docker info | Out-Null
    Write-Host "  [+] Docker daemon активен." -ForegroundColor Green
} catch {
    Write-Host "[-] Docker daemon не запущен. Пожалуйста, запустите Docker Desktop." -ForegroundColor Red
    Pause
    Exit 1
}

# 4. Deploy Docker
Write-Host "`n[3/4] Сборка и запуск контейнеров..." -ForegroundColor Yellow
docker compose -f docker/docker-compose.yml up --build -d

# 5. Open browser
Write-Host "`n[4/4] Открытие редактора в браузере..." -ForegroundColor Yellow
Start-Sleep -Seconds 3
Start-Process "http://localhost:8000"

Write-Host "`n[✓] PepeShop запущен на http://localhost:8000" -ForegroundColor Green
Write-Host "  Логи пишутся в: logs/log.txt" -ForegroundColor Green
Write-Host "  Выходные файлы сохраняются в: output/" -ForegroundColor Green
