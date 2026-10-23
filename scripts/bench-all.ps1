# One labelled run of the text benches against the bench services (scripts/bench-services.ps1):
# tools sim on the group scenarios, the built-in scenarios and the held-out set, then the chat
# bench with tools. Prints each report's totals; results land in bench-results/.
#
#   ./scripts/bench-all.ps1 -Label baseline
param([Parameter(Mandatory)][string]$Label, [int]$Seeds = 2, [switch]$NoChat)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$sim = 'target\release\grace-tools-sim.exe'
foreach ($set in @(@('groups', 'fixtures/tools/scenarios'), @('builtin', 'fixtures/tools/scenarios.yaml'), @('heldout', 'fixtures/tools/heldout'))) {
    $name = "$Label-$($set[0])"
    & $sim --label $name --scenarios $set[1] --seeds $Seeds --rules-from data/memory.db *> "logs\sim-$name.txt"
    $r = Get-Content "bench-results\tools\$name\report.json" | ConvertFrom-Json
    Write-Host ("{0,-28} all {1,-8} tools {2,-8} args {3,-8} answer {4}" -f $name, $r.all_ok, $r.tools_ok, $r.args_ok, $r.answer_ok)
}
if (-not $NoChat) {
    # 13 (opening hours) is skipped: its live search trips the antivirus on radarsuper.com.
    & target\release\grace-chat-bench.exe --label "$Label-chat" --tools 127.0.0.1:7814 --skip 13 *> "logs\chat-$Label.txt"
    Get-Content "logs\chat-$Label.txt" | Select-String '"score"|"hard"' | ForEach-Object { Write-Host "$Label-chat $($_.Line.Trim())" }
}
