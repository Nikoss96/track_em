# Dev-сервер UI с hot reload: http://localhost:5173 (проксирует /api и /ws на backend :8000)
$Root = Split-Path -Parent $PSScriptRoot
Set-Location "$Root\frontend"
npm run dev
