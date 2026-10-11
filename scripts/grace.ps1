# Starts Grace: llama-server and the speech server (only if not already running), then the engine.
# The two servers hold the big models and stay up between engine restarts.
#
#   ./scripts/grace.ps1            # run Grace
#   ./scripts/grace.ps1 -SelfTest  # silent startup check
#   ./scripts/grace.ps1 -Restart   # restart the speech server first (new voice, code changes)
param([switch]$SelfTest, [switch]$Restart)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force logs | Out-Null

function Test-Port($port) { (Test-NetConnection 127.0.0.1 -Port $port -WarningAction SilentlyContinue).TcpTestSucceeded }
function Wait-Port($port, $what, $seconds) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while (-not (Test-Port $port)) {
        if ((Get-Date) -gt $deadline) { throw "$what did not start within $seconds s; see logs/" }
        Start-Sleep -Milliseconds 500
    }
}

if (-not (Test-Port 8080)) {
    Write-Host 'starting llama-server (Gemma 4 12B) ...'
    Start-Process -WindowStyle Hidden -FilePath .runtime\llama.cpp\b11539\llama-server.exe `
        -ArgumentList '-m','.runtime\models\gemma-4-12b-it-qat-q4_0.gguf','-ngl','99','-c','16384','-np','2','-fa','on','-ctk','q8_0','-ctv','q8_0','--reasoning','off','--swa-full','--host','127.0.0.1','--port','8080' `
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
Wait-Port 8080 'llama-server' 120
Wait-Port 7811 'speech server' 120
# llama-server opens its port before the model is loaded; wait for /health.
while ($true) { try { if ((Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/health).StatusCode -eq 200) { break } } catch {} ; Start-Sleep -Milliseconds 500 }

# Build, then run the engine directly (not via `cargo run`): Ctrl+C must reach Grace herself.
cargo build --release -q -p grace
if ($LASTEXITCODE -ne 0) { throw 'engine build failed' }
$engineArgs = @()
if ($SelfTest) { $engineArgs += '--selftest' }
& ./target/release/grace.exe @engineArgs
