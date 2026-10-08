# Установка Multimodal Behavior Analyzer (Windows 10/11 x64).
#   powershell -ExecutionPolicy Bypass -File scripts\install.ps1 [-Gpu] [-SkipFrontend] [-Models]
# -Gpu     : torch CUDA 12.8 + cuBLAS/cuDNN для faster-whisper (RTX 3080)
# -Models  : заранее скачать веса (whisper, emotion2vec, MediaPipe) и Qwen3 через Ollama
param([switch]$Gpu, [switch]$SkipFrontend, [switch]$Models)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:UV_CACHE_DIR = "$Root\.cache\uv"
$env:UV_PYTHON_INSTALL_DIR = "$Root\.cache\python"
$env:UV_LINK_MODE = "copy"

Write-Host "== uv / Python 3.11" -ForegroundColor Cyan
python -m pip install --user -q uv
python -m uv python install 3.11
if (-not (Test-Path ".venv\Scripts\python.exe")) { python -m uv venv .venv --python 3.11 }
$Py = "$Root\.venv\Scripts\python.exe"

Write-Host "== Python-зависимости" -ForegroundColor Cyan
python -m uv pip install --python $Py -r requirements.txt
if ($Gpu) {
    python -m uv pip install --python $Py torch torchaudio --index-url https://download.pytorch.org/whl/cu128
    # CTranslate2 (faster-whisper) на Windows требует cuBLAS 12 и cuDNN 9 DLL
    python -m uv pip install --python $Py nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
} else {
    python -m uv pip install --python $Py torch torchaudio --index-url https://download.pytorch.org/whl/cpu
}

if (-not $SkipFrontend) {
    Write-Host "== Frontend (Node.js >= 18)" -ForegroundColor Cyan
    Push-Location frontend
    npm install --no-audit --no-fund
    npm run build
    Pop-Location
}

Write-Host "== Тестовые фикстуры" -ForegroundColor Cyan
& $Py scripts\make_fixtures.py
if (-not (Test-Path "tests\fixtures\reading_av.mp4")) { & $Py scripts\make_reading_fixture.py }

if ($Models) {
    Write-Host "== Загрузка моделей" -ForegroundColor Cyan
    & $Py scripts\download_models.py --profile $(if ($Gpu) { "gpu" } else { "cpu" })
}

Write-Host "== Проверка окружения" -ForegroundColor Cyan
& $Py scripts\check_env.py
