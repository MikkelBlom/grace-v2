# Starts the servers the text benches need (grace-tools-sim, grace-chat-bench), all hidden:
# llama-server with Grace's flags, the embed server, SearXNG and a dry-run tools server on 7814.
# Overrides are for A/B runs (docs/research/small-models/README.md); -Restart replaces a running
# llama-server / tools server so a changed flag takes effect.
#
#   ./scripts/bench-services.ps1
#   ./scripts/bench-services.ps1 -Restart -KvType f16
#   ./scripts/bench-services.ps1 -Restart -Model .runtime\models\gemma-4-12b-it-UD-Q4_K_XL.gguf
#   ./scripts/bench-services.ps1 -Stop
param([switch]$Restart, [switch]$Stop, [string]$KvType = 'q8_0', [string]$Model = '.runtime\models\gemma-4-12b-it-qat-q4_0.gguf',
      [string[]]$Extra = @(), [string]$Verbosity = '')
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force logs | Out-Null

function Test-Port($port) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try { return $client.ConnectAsync('127.0.0.1', $port).Wait(200) } catch { return $false } finally { $client.Dispose() }
}
function Stop-Matching($pattern) {
    Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like $pattern } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

if ($Stop -or $Restart) {
    Stop-Matching '*llama-server.exe*'
    Stop-Matching '*tools_server.py*--port*7814*'
    Start-Sleep -Seconds 2
    if ($Stop) { Stop-Matching '*embed_server.py*'; Write-Host 'stopped'; return }
}

if (-not (Test-Port 8080)) {
    # Same flags as scripts/grace.ps1, except the overrides.
    $llamaArgs = @('-m', $Model, '-ngl', '99', '-c', '32768', '-np', '2', '-fa', 'on', '-ctk', $KvType, '-ctv', $KvType,
                   '--reasoning', 'off', '--swa-full', '--host', '127.0.0.1', '--port', '8080')
    $mmproj = '.runtime\models\mmproj-gemma-4-12b-it-qat-q4_0.gguf'
    if (Test-Path $mmproj) { $llamaArgs += @('--mmproj', $mmproj, '-ub', '1120') }
    if ($Verbosity) { $llamaArgs += @('-lv', $Verbosity) }
    $llamaArgs += $Extra
    Write-Host "llama-server: $($llamaArgs -join ' ')"
    Start-Process -WindowStyle Hidden -FilePath .runtime\llama.cpp\b11539\llama-server.exe -ArgumentList $llamaArgs `
        -RedirectStandardError logs\llama-server.log -RedirectStandardOutput logs\llama-server.out.log
}
if (-not (Test-Port 7812)) {
    Start-Process -WindowStyle Hidden -FilePath bench\.venv-embed\Scripts\python.exe -ArgumentList 'services\embed_server.py' `
        -RedirectStandardError logs\embed-server.log -RedirectStandardOutput logs\embed-server.out.log
}
if (-not (Test-Port 8888)) {
    docker compose -f docker/searxng/docker-compose.yml up -d *> logs\searxng.log
}
while ($true) { try { if ((Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/health).StatusCode -eq 200) { break } } catch {} ; Start-Sleep -Milliseconds 500 }
if (-not (Test-Port 7814)) {
    $env:GRACE_TOOLS_SCHEDULER = '0'  # no reminders firing during benches
    Start-Process -WindowStyle Hidden -FilePath bench\.venv-openvino\Scripts\python.exe `
        -ArgumentList 'services/tools_server.py', '--dry-run', '--port', '7814' `
        -RedirectStandardError logs\tools-server-dry.log -RedirectStandardOutput logs\tools-server-dry.out.log
}
$deadline = (Get-Date).AddSeconds(180)
while (-not (Test-Port 7814)) { if ((Get-Date) -gt $deadline) { throw 'tools server did not start; see logs/tools-server-dry.log' }; Start-Sleep 1 }
while (-not (Test-Port 7812)) { if ((Get-Date) -gt $deadline) { throw 'embed server did not start' }; Start-Sleep 1 }
Write-Host 'bench services ready'
