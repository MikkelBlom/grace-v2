#Requires -RunAsAdministrator
# Installs the four Intel driver updates from Downloads, in a safe order, one at a time.
# Each installer opens its own window: click through it as usual. The script checks every file
# first (present, signed by Intel), records the driver version before and after, and logs
# everything to logs\intel-drivers-<time>.log plus a .json summary for Claude to read.
#
#   Right-click PowerShell -> Run as administrator, then:
#   & "C:\Users\mikke\Desktop\Projects\Grace\scripts\update-intel-drivers.ps1"
#
# It never restarts the PC itself; restart once at the end.

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
New-Item -ItemType Directory -Force (Join-Path $root 'logs') | Out-Null
$log = Join-Path $root "logs\intel-drivers-$stamp.log"
$jsonPath = Join-Path $root "logs\intel-drivers-$stamp.json"
$downloads = Join-Path $env:USERPROFILE 'Downloads'

# Order: NPU first (what we need), graphics next, Wi-Fi last because it drops the network.
$packages = @(
    @{ Name = 'NPU';       File = 'npu_win_32.0.100.5540.exe';              Device = 'AI Boost';              Target = '32.0.100.5540' },
    @{ Name = 'Graphics';  File = 'gfx_win_101.9150.exe';                   Device = 'Arc\(TM\) 140T';         Target = '32.0.101.9150' },
    @{ Name = 'Bluetooth'; File = 'BT-24.70.0-64UWD-Win10-Win11.exe';       Device = '^Intel\(R\) Wireless Bluetooth'; Target = '24.70.0' },
    @{ Name = 'Wi-Fi';     File = 'WiFi-24.70.0-Driver64-Win10-Win11.exe';  Device = 'Wi-Fi 7 BE201';         Target = '24.70.0' }
)

$results = @()
function Say($level, $text) {
    $line = "{0} [{1}] {2}" -f (Get-Date -Format 'HH:mm:ss'), $level, $text
    Add-Content -Path $log -Value $line -Encoding utf8
    $color = @{ OK = 'Green'; WARN = 'Yellow'; FAIL = 'Red'; INFO = 'Gray' }[$level]
    Write-Host $line -ForegroundColor $color
}

function Get-DriverVersion($pattern) {
    $d = Get-CimInstance Win32_PnPSignedDriver -ErrorAction SilentlyContinue |
        Where-Object { $_.DeviceName -match $pattern } | Select-Object -First 1
    if ($d) { $d.DriverVersion } else { $null }
}

function At-Least($have, $want) {
    if (-not $have) { return $false }
    try {
        $h = [version](($have -split '[^0-9.]')[0])
        $w = [version](($want -split '[^0-9.]')[0])
        # Wi-Fi/Bluetooth report 24.70.0.x; compare only as many parts as the target has.
        $parts = $w.ToString().Split('.').Count
        $hTrim = [version](($h.ToString().Split('.')[0..($parts - 1)]) -join '.')
        return $hTrim -ge $w
    } catch { return $false }
}

Say INFO "Intel driver update started. Log: $log"
Say INFO ("Windows {0}, PC {1}" -f [Environment]::OSVersion.Version, $env:COMPUTERNAME)
$pending = (Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending') -or
           (Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired')
if ($pending) { Say WARN 'Windows already has a restart pending. Installers may fail; consider restarting first.' }

foreach ($p in $packages) {
    $r = [ordered]@{ name = $p.Name; file = $p.File; target = $p.Target; before = $null; after = $null;
                     exit_code = $null; seconds = $null; status = $null; note = '' }
    Say INFO "---- $($p.Name) ----"
    $path = Join-Path $downloads $p.File
    $r.before = Get-DriverVersion $p.Device
    Say INFO "Installed now: $($r.before)"

    if (-not (Test-Path $path)) {
        $r.status = 'FAIL'; $r.note = "file not found: $path"
        Say FAIL $r.note; $results += [pscustomobject]$r; continue
    }
    $sig = Get-AuthenticodeSignature $path
    if ($sig.Status -ne 'Valid' -or $sig.SignerCertificate.Subject -notmatch 'Intel Corporation') {
        $r.status = 'FAIL'; $r.note = "signature not valid Intel: $($sig.Status) $($sig.SignerCertificate.Subject)"
        Say FAIL "$($r.note) - skipped, not running it."; $results += [pscustomobject]$r; continue
    }
    Say OK "File present and signed by Intel Corporation."

    if (At-Least $r.before $p.Target) {
        $r.status = 'OK'; $r.after = $r.before; $r.note = 'already up to date, skipped'
        Say OK $r.note; $results += [pscustomobject]$r; continue
    }
    if ($p.Name -eq 'Wi-Fi') { Say WARN 'The network will drop for a moment during this install.' }

    Say INFO "Starting the installer. Click through its window; the script waits for it to close."
    $t0 = Get-Date
    try {
        $proc = Start-Process -FilePath $path -Wait -PassThru
        $r.exit_code = $proc.ExitCode
    } catch {
        $r.status = 'FAIL'; $r.note = "could not start: $($_.Exception.Message)"
        Say FAIL $r.note; $results += [pscustomobject]$r; continue
    }
    $r.seconds = [int]((Get-Date) - $t0).TotalSeconds
    Start-Sleep -Seconds 3
    $r.after = Get-DriverVersion $p.Device

    switch ($r.exit_code) {
        0       { $codeNote = 'installer reported success' }
        3010    { $codeNote = 'installer reported success, restart required' }
        1641    { $codeNote = 'installer reported success and started a restart' }
        1602    { $codeNote = 'cancelled by the user' }
        default { $codeNote = "installer exit code $($r.exit_code)" }
    }
    if (At-Least $r.after $p.Target) {
        $r.status = 'OK'; $r.note = "$codeNote; now $($r.after)"
        Say OK $r.note
    } elseif ($r.exit_code -in 0, 3010, 1641) {
        $r.status = 'WARN'; $r.note = "$codeNote, but the device still reports $($r.after); check again after the restart"
        Say WARN $r.note
    } else {
        $r.status = 'FAIL'; $r.note = "$codeNote; device still reports $($r.after)"
        Say FAIL $r.note
    }
    $results += [pscustomobject]$r
}

Say INFO '==== Summary ===='
foreach ($r in $results) {
    Say $r.status ("{0,-9} {1} -> {2} (target {3}) {4}" -f $r.name, $r.before, $r.after, $r.target, $r.note)
}
$results | ConvertTo-Json | Set-Content -Path $jsonPath -Encoding utf8
$bad = @($results | Where-Object status -ne 'OK').Count
if ($bad) { Say WARN "$bad of $($results.Count) need attention (see above). Restart, then tell Claude." }
else { Say OK "All $($results.Count) done. Restart the PC, then tell Claude." }
Say INFO "Log: $log"
Say INFO "Summary for Claude: $jsonPath"
Read-Host 'Press Enter to close'
