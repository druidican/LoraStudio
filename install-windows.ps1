# SPDX-License-Identifier: GPL-3.0-only · Copyright (C) 2026 L. G. H. Druidican · see LICENSE
# LoRA Studio for Windows. LoRA Studio version 1.261026s
#
# LoRA Studio and the trainers are Linux programs. On Windows they run inside WSL2 (Windows' own Linux, from
# Microsoft), which is the way sd-scripts, musubi-tuner and PyTorch work best on Windows. This script:
#   1. installs WSL2 with Ubuntu 24.04 when it is missing (needs administrator rights and one restart),
#   2. checks the graphics card and Ollama on the Windows side,
#   3. runs install.sh inside Ubuntu (it asks the same questions as on Linux),
#   4. puts a "LoRA Studio" starter on the desktop: it starts LoRA Studio in Ubuntu and opens it in your browser.
#
# Start it with "Install LoRA Studio.cmd" (double-click), or:  powershell -ExecutionPolicy Bypass -File install-windows.ps1
# Options: -Distro Ubuntu-24.04   -Dir ~/LoraStudio (inside Ubuntu)   -Port 7860
param(
    [string]$Distro = "Ubuntu-24.04",
    [string]$Dir = "~/LoraStudio",
    [int]$Port = 7860
)
$ErrorActionPreference = "Stop"
$env:WSL_UTF8 = "1"                               # wsl.exe writes UTF-8 instead of UTF-16
$script:Rpt = New-Object System.Collections.Generic.List[string]     # the install report, written at the end
function Say($t) { Write-Host "   $t" }
function Ok($t) { Write-Host "   ok " -ForegroundColor Green -NoNewline; Write-Host $t; $script:Rpt.Add("- ok $t") }
function Warn($t) { Write-Host "   !  " -ForegroundColor Yellow -NoNewline; Write-Host $t; $script:Rpt.Add("- ! $t") }
function Step($t) { Write-Host ""; Write-Host "== $t" -ForegroundColor Cyan; $script:Rpt.Add(""); $script:Rpt.Add("**$t**") }
function Write-Report($result, $linux = "") {
    # install_report_windows_<time>.md next to this script (user name and home folder removed), and printed
    try {
        $os = Get-CimInstance Win32_OperatingSystem
        $gpu = (Get-CimInstance Win32_VideoController | ForEach-Object { "$($_.Name) (driver $($_.DriverVersion))" }) -join " · "
        $lines = @("# LoRA Studio install report (Windows) · $(Get-Date -Format 'yyyy-MM-dd HH:mm')", "",
            "Please look at this install of LoRA Studio on Windows (WSL2; a Gradio app that trains LoRAs).", "",
            "- Result: **$result**",
            "- Windows: $($os.Caption) $($os.Version) (build $($os.BuildNumber)) · WSL distro: $Distro",
            "- Graphics: $gpu") + $script:Rpt
        if ($linux) { $lines += @("", "## The Linux part (install.sh inside $Distro)", "", $linux) }
        $text = ($lines -join "`r`n") -replace [regex]::Escape($env:USERPROFILE), "%USERPROFILE%" `
            -replace "(hf_|sk-|ghp_|github_pat_)[A-Za-z0-9_-]{16,}", "[token removed]"
        if ($env:USERNAME -and $env:USERNAME.Length -ge 3) { $text = $text -replace ("\b" + [regex]::Escape($env:USERNAME) + "\b"), "<user>" }
        $f = Join-Path $here ("install_report_windows_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".md")
        Set-Content -Path $f -Value $text -Encoding UTF8
        try { Set-Clipboard -Value $text; $clip = " (also copied: paste it into a chat with Ctrl+V)" } catch { $clip = "" }
        Write-Host ""
        Say "Install report: $f$clip"
    } catch { Say "(the install report could not be written: $_)" }
}
function Stop-Here($t) {
    Write-Host ""; Write-Host "Stopped: $t" -ForegroundColor Red
    $script:Rpt.Add(""); $script:Rpt.Add("**Stopped:** $t"); Write-Report "stopped"
    Read-Host "Press Enter to close"; exit 1
}
function Ask($q, $def = "y") {
    $a = Read-Host "   $q $(if ($def -eq 'y') {'(Y/n)'} else {'(y/N)'})"
    if (-not $a) { $a = $def }
    return $a -match '^[YyJj]'
}
function Wsl-Lines([string[]]$a) { (& wsl.exe @a 2>$null) | ForEach-Object { ($_ -replace "`0", "").Trim() } | Where-Object { $_ } }

Write-Host ""
Write-Host "LoRA Studio for Windows (runs inside WSL2 with $Distro)" -ForegroundColor White
$here = $PSScriptRoot
if (-not $here) { $here = (Get-Location).Path }
if (-not (Test-Path (Join-Path $here "install.sh"))) { Stop-Here "install.sh is not next to this script: unpack the whole LoraStudio.zip first." }

# ---------------------------------------------------------------- 1. WSL2 and Ubuntu
Step "1. WSL2 (Linux inside Windows)"
$build = [int](Get-CimInstance Win32_OperatingSystem).BuildNumber
if ($build -lt 19041) { Stop-Here "Windows 10 version 2004 (build 19041) or newer is needed for WSL2; this is build $build." }
$distros = @()
if (Get-Command wsl.exe -ErrorAction SilentlyContinue) { $distros = @(Wsl-Lines @("-l", "-q")) }
if ($distros -notcontains $Distro) {
    Say "$Distro is not installed in WSL yet."
    $admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $admin) {
        Say "Installing WSL needs administrator rights: Windows asks for them now."
        Start-Process powershell.exe -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Distro $Distro -Dir $Dir -Port $Port"
        exit 0
    }
    if (-not (Ask "Install WSL2 with $Distro now? (about 1-2 GB; Windows must restart afterwards)")) { Stop-Here "WSL2 is needed." }
    & wsl.exe --install -d $Distro
    Write-Host ""
    Say "Next:"
    Say "  1. Restart Windows."
    Say "  2. Ubuntu opens by itself (or start 'Ubuntu' from the Start menu): choose a user name and a password."
    Say "  3. Double-click 'Install LoRA Studio.cmd' again: it continues from here."
    Read-Host "Press Enter to close"
    exit 0
}
$ver = (Wsl-Lines @("-l", "-v")) | Where-Object { $_ -match "^\*?\s*$([regex]::Escape($Distro))\s" } | Select-Object -First 1
if ($ver -and $ver -match "\s1$") {
    Say "$Distro runs as WSL1; LoRA Studio needs WSL2 (for the GPU)."
    & wsl.exe --set-version $Distro 2
}
$user = (Wsl-Lines @("-d", $Distro, "--", "whoami")) | Select-Object -First 1
if (-not $user) { Stop-Here "$Distro does not start. Open 'Ubuntu' from the Start menu once (it finishes its setup), then run this again." }
if ($user -eq "root") { Warn "$Distro logs in as root: better open Ubuntu once and make a normal user (LoRA Studio works as root too)." }
Ok "WSL2 with $Distro, user $user"

# ---------------------------------------------------------------- 2. graphics card and Ollama (Windows side)
Step "2. Graphics card and Ollama"
$gpus = @(Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name })
foreach ($g in $gpus) { Say "GPU: $g" }
if ($gpus -match "NVIDIA") {
    Ok "NVIDIA: the normal Windows driver also serves WSL (nothing to install inside Ubuntu)"
} elseif ($gpus -match "AMD|Radeon") {
    Warn "AMD Radeon in WSL: training needs AMD's 'ROCm on WSL' (Radeon RX 7000/9000 with a recent Adrenalin driver):"
    Say "   https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-radeon.html"
    Say "   If that is not possible, answer 'none' for sd-scripts and musubi-tuner: LoRA Studio then makes the launch"
    Say "   scripts and prepares datasets, and you train on a Linux computer."
} else {
    Warn "no NVIDIA or AMD GPU found: answer 'none' for sd-scripts and musubi-tuner (training needs a GPU)"
}

$olArgs = @()
$olWin = $false
try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 "http://127.0.0.1:11434/api/tags" | Out-Null; $olWin = $true } catch { }
if (-not $olWin -and (Get-Command ollama.exe -ErrorAction SilentlyContinue)) { $olWin = $true; Say "Ollama is installed (not running right now)." }
if ($olWin) {
    Ok "Ollama runs in Windows"
    # Ubuntu reaches Windows' Ollama at 127.0.0.1 with 'mirrored' networking (Windows 11 22H2 and newer)
    $cfg = Join-Path $env:USERPROFILE ".wslconfig"
    $txt = if (Test-Path $cfg) { Get-Content $cfg -Raw } else { "" }
    if ($txt -match "networkingMode\s*=\s*mirrored") {
        Ok "WSL uses mirrored networking: Ollama is at http://127.0.0.1:11434 for LoRA Studio too"
        $olArgs = @("--ollama-url", "http://127.0.0.1:11434")
    } elseif ($build -ge 22621 -and (Ask "Let Ubuntu reach Windows' Ollama at 127.0.0.1 (sets networkingMode=mirrored in .wslconfig, restarts WSL)?")) {
        if ($txt -match "(?m)^\s*\[wsl2\]") { $txt = $txt -replace "(?m)^\s*\[wsl2\]\s*$", "[wsl2]`r`nnetworkingMode=mirrored" }
        else { $txt = $txt.TrimEnd() + "`r`n[wsl2]`r`nnetworkingMode=mirrored`r`n" }
        Set-Content -Path $cfg -Value $txt.TrimStart() -Encoding ASCII
        & wsl.exe --shutdown
        Ok "mirrored networking on"
        $olArgs = @("--ollama-url", "http://127.0.0.1:11434")
    } else {
        Say "Without mirrored networking Ubuntu reaches Windows by its WSL address; Ollama must then listen on it:"
        Say "  set OLLAMA_HOST=0.0.0.0 for your Windows user and restart Ollama."
        if (Ask "Set OLLAMA_HOST=0.0.0.0 now?") {
            [Environment]::SetEnvironmentVariable("OLLAMA_HOST", "0.0.0.0", "User")
            Ok "set: quit Ollama (tray icon) and start it again"
        }
        $gw = (Wsl-Lines @("-d", $Distro, "--", "sh", "-c", "ip route show default | awk '{print `$3}'")) | Select-Object -First 1
        if ($gw) { $olArgs = @("--ollama-url", "http://${gw}:11434"); Say "Ollama for LoRA Studio: http://${gw}:11434 (this address can change after a restart: Settings › AI helpers)" }
    }
} else {
    Say "Ollama is not installed. It is optional (the detailed sort, captions, the consistency check)."
    Say "Install it in Windows (not inside Ubuntu): https://ollama.com/download/windows"
    Say "Guide: https://github.com/ollama/ollama/blob/main/docs/windows.md"
    Say "Then run this installer again, or tick Ollama in LoRA Studio › Settings › Start and network."
}

# ---------------------------------------------------------------- 3. install.sh inside Ubuntu
Step "3. LoRA Studio inside $Distro"
$src = (Wsl-Lines @("-d", $Distro, "--", "wslpath", "-a", ($here -replace "\\", "/"))) | Select-Object -First 1
if (-not $src) { Stop-Here "Ubuntu cannot see $here." }
Say "Running install.sh in Ubuntu (it asks about sd-scripts, musubi-tuner and Ollama; it may ask for your Ubuntu password)."
$extra = ($olArgs | ForEach-Object { "'$_'" }) -join " "
& wsl.exe -d $Distro -- bash -lc "cd '$src' && sed -i 's/\r$//' install.sh start.sh 2>/dev/null; bash ./install.sh --dir $Dir $extra"
$rc = $LASTEXITCODE
$linux = (& wsl.exe -d $Distro -- bash -lc "ls -t $Dir/userdata/reports/install_report_*.md 2>/dev/null | head -1 | xargs -r cat") -join "`r`n"
if ($rc -ne 0) {
    $script:Rpt.Add(""); $script:Rpt.Add("**Stopped:** install.sh inside $Distro stopped (exit code $rc)")
    Write-Report "stopped in install.sh" $linux
    Write-Host ""; Write-Host "Stopped: install.sh stopped (its message is above). Fix the cause and run this again: what worked is kept." -ForegroundColor Red
    Read-Host "Press Enter to close"; exit 1
}
Ok "install.sh finished inside $Distro"

# ---------------------------------------------------------------- 4. the starter
Step "4. Desktop starter"
$starter = Join-Path $here "LoRA Studio.cmd"
@"
@echo off
rem LoRA Studio: starts it inside WSL ($Distro) and opens it in the browser. Close this window to stop LoRA Studio.
title LoRA Studio (close this window to stop it)
start "" powershell -NoProfile -WindowStyle Hidden -Command "for(`$i=0;`$i -lt 90;`$i++){try{Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://localhost:$Port | Out-Null; Start-Process http://localhost:$Port; break}catch{Start-Sleep 2}}"
wsl.exe -d $Distro -- bash -lc "cd $Dir && ./start.sh --no-browser"
"@ | Set-Content -Path $starter -Encoding ASCII
$desk = [Environment]::GetFolderPath("Desktop")
$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut((Join-Path $desk "LoRA Studio.lnk"))
$lnk.TargetPath = $starter
$lnk.WorkingDirectory = $here
$ico = Join-Path $here "logo\lorastudio.ico"
if (Test-Path $ico) { $lnk.IconLocation = $ico }
$lnk.Save()
Ok "'LoRA Studio' is on your desktop (it opens http://localhost:$Port when LoRA Studio is ready)"
Write-Host ""
Say "Your files: in Ubuntu, Windows' C: drive is /mnt/c (for example /mnt/c/Users/$env:USERNAME/Pictures)."
Say "Datasets and models train much faster inside Ubuntu (\\wsl$\$Distro\home\$user in the Explorer) than on /mnt/c."
Write-Report "finished" $linux
Read-Host "Done. Press Enter to close"
