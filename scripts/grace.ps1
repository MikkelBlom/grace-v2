# Starts Grace: llama-server, the speech server and the embed server (only those not already
# running), then the engine. The servers hold the models and stay up between engine restarts.
#
#   ./scripts/grace.ps1            # run Grace
#   ./scripts/grace.ps1 -SelfTest  # silent startup check
#   ./scripts/grace.ps1 -Restart   # restart the speech server first (new voice, code changes)
#   ./scripts/grace.ps1 -Memory data/replay-memory.db   # separate memory (unattended test runs)
param([switch]$SelfTest, [switch]$Restart, [string]$Memory)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force logs | Out-Null

# Fast local port check (Test-NetConnection takes ~10 s per closed port).
function Test-Port($port) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try { return $client.ConnectAsync('127.0.0.1', $port).Wait(200) } catch { return $false } finally { $client.Dispose() }
}
function Wait-Port($port, $what, $seconds) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while (-not (Test-Port $port)) {
        if ((Get-Date) -gt $deadline) { throw "$what did not start within $seconds s; see logs/" }
        Start-Sleep -Milliseconds 500
    }
}

if (-not (Test-Port 8080)) {
    Write-Host 'starting llama-server (Gemma 4 12B) ...'
    $llamaArgs = @('-m','.runtime\models\gemma-4-12b-it-qat-q4_0.gguf','-ngl','99','-c','32768','-np','2','-fa','on','-ctk','q8_0','-ctv','q8_0','--reasoning','off','--swa-full','--host','127.0.0.1','--port','8080')
    # The vision projector lets the vision tools send screenshots (docs/tools/vision.md). -ub 1120
    # gives an image its full token budget (capped at 512 otherwise): small screen text read
    # 16/28 lines exactly vs 2/28. Text TTFT measured unchanged (341 vs 343 ms p50).
    $mmproj = '.runtime\models\mmproj-gemma-4-12b-it-qat-q4_0.gguf'
    if (Test-Path $mmproj) { $llamaArgs += @('--mmproj', $mmproj, '-ub', '1120') }
    Start-Process -WindowStyle Hidden -FilePath .runtime\llama.cpp\b11539\llama-server.exe `
        -ArgumentList $llamaArgs `
        -RedirectStandardError logs\llama-server.log -RedirectStandardOutput logs\llama-server.out.log
}
# Restart the speech server with -Restart (e.g. after changing config/speech.json).
if ($Restart -and (Test-Port 7810)) {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*speech_server.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    Start-Sleep -Seconds 1
}
if (-not (Test-Port 7810)) {
    Write-Host 'starting speech server (edda STT + Pocket TTS) ...'
    Start-Process -WindowStyle Hidden -FilePath bench\.venv\Scripts\python.exe -ArgumentList 'services\speech_server.py' `
        -RedirectStandardError logs\speech-server.log -RedirectStandardOutput logs\speech-server.out.log
}
if (-not (Test-Port 7812)) {
    Write-Host 'starting embed server (harrier-270m, memory search) ...'
    Start-Process -WindowStyle Hidden -FilePath bench\.venv-embed\Scripts\python.exe -ArgumentList 'services\embed_server.py' `
        -RedirectStandardError logs\embed-server.log -RedirectStandardOutput logs\embed-server.out.log
}
# Tools (ADR 0019, 0020): SearXNG in Docker + the tools server (JevK5 4B on the Arc iGPU).
# Grace still runs without them: no Docker means DuckDuckGo/Wikipedia only, no tools server
# means no tools this session.
if (-not (Test-Port 8888)) {
    docker info *> $null
    if ($LASTEXITCODE -eq 0) {
        Write-Host 'starting SearXNG (web search) ...'
        docker compose -f docker/searxng/docker-compose.yml up -d *> logs\searxng.log
    } else { Write-Host 'Docker is not running: web search falls back to DuckDuckGo and Wikipedia' }
}
$toolsVenv = Test-Path bench\.venv-openvino\Scripts\python.exe
if ($toolsVenv -and -not (Test-Port 7813)) {
    Write-Host 'starting tools server (JevK5 4B on the Arc iGPU, web search, apps, files) ...'
    Start-Process -WindowStyle Hidden -FilePath bench\.venv-openvino\Scripts\python.exe -ArgumentList 'services\tools_server.py' `
        -RedirectStandardError logs\tools-server.log -RedirectStandardOutput logs\tools-server.out.log
}
Wait-Port 8080 'llama-server' 120
Wait-Port 7811 'speech server' 120
Wait-Port 7812 'embed server' 120
if ($toolsVenv) { Wait-Port 7813 'tools server' 120 }
# llama-server opens its port before the model is loaded; wait for /health.
while ($true) { try { if ((Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/health).StatusCode -eq 200) { break } } catch {} ; Start-Sleep -Milliseconds 500 }

# Build, then run the engine directly (not via `cargo run`): Ctrl+C must reach Grace herself.
cargo build --release -q -p grace
if ($LASTEXITCODE -ne 0) { throw 'engine build failed' }
$engineArgs = @()
if ($SelfTest) { $engineArgs += '--selftest' }
if ($Memory) { $engineArgs += '--memory', $Memory }
& ./target/release/grace.exe @engineArgs
