# Запуск backend (API + собранный UI на http://127.0.0.1:8000)
param([int]$Port = 8000, [switch]$Reload)
$Root = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force "$Root\data" | Out-Null
Set-Location "$Root\backend"
$args_ = @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", $Port)
if ($Reload) { $args_ += "--reload" }
& "$Root\.venv\Scripts\python.exe" @args_
