# Windows setup and update for the heavy-machine scenarios - no git needed.
#
# Downloads the latest scripts\heavy_machines and assets\heavy_machines from GitHub (branch
# claude/review-last-request-I6mXJ) into the Isaac Lab folder, then starts the render. Nothing else in
# the folder changes: no git commands, no stash, no checkout. An up-to-date copy is not downloaded again.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File setup_windows.ps1 [-Repo F:\Projects\IsaacLab] [-NoRender]
#
# Log: %TEMP%\heavy_machines_setup.log (kept open for the whole run, so other programs that briefly lock
# new files cannot block it). Windows PowerShell 5.1 syntax, plain ASCII.
param([string]$Repo = "F:\Projects\IsaacLab", [switch]$NoRender)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"      # Windows PowerShell 5.1 downloads crawl with a progress bar
$owner = "seyedalicheraghi"; $name = "vla-datagen"; $branch = "claude/review-last-request-I6mXJ"
$folders = @("scripts/heavy_machines/", "assets/heavy_machines/")

$tmp = [System.IO.Path]::GetTempPath()
$logPath = Join-Path $tmp "heavy_machines_setup.log"
$logStream = New-Object System.IO.FileStream($logPath, [System.IO.FileMode]::Create,
    [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite)
$log = New-Object System.IO.StreamWriter($logStream)
$log.AutoFlush = $true

function Say([string]$msg) {
    Write-Host "[setup] $msg" -ForegroundColor Cyan
    $log.WriteLine("[setup] $msg")
}

function Stop-Setup([string]$msg) {
    Say "STOPPED: $msg"
    Say "Nothing was deleted. Log: $logPath"
    $log.Close()
    exit 1
}

try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch { $null = $_ }

if (-not (Test-Path -LiteralPath $Repo)) { Stop-Setup "folder $Repo not found - run with -Repo <your IsaacLab folder>" }
$Repo = (Resolve-Path -LiteralPath $Repo).Path
$hmDir = Join-Path (Join-Path $Repo "scripts") "heavy_machines"
$launcher = Join-Path $hmDir "render_windows.ps1"
$stampPath = Join-Path $hmDir ".installed_commit"
Say "Isaac Lab folder: $Repo | PowerShell $($PSVersionTable.PSVersion)"

# 1. which version is on GitHub, and is it already here?
$sha = $null
try {
    $resp = Invoke-WebRequest -UseBasicParsing -TimeoutSec 30 -Headers @{ "User-Agent" = "heavy-machines-setup" } `
        -Uri "https://api.github.com/repos/$owner/$name/commits/$branch"
    $sha = ($resp.Content | ConvertFrom-Json).sha
} catch {
    Say "could not ask GitHub for the latest version ($($_.Exception.Message)) - downloading anyway"
}
$installed = $null
if (Test-Path -LiteralPath $stampPath) { $installed = Get-Content -LiteralPath $stampPath -TotalCount 1 }

if ($sha -and ($installed -eq $sha) -and (Test-Path -LiteralPath $launcher)) {
    Say "already up to date ($($sha.Substring(0, 7)))"
} else {
    # 2. download the branch as a ZIP and extract only the heavy-machine folders
    $zip = Join-Path $tmp "heavy_machines_branch.zip"
    Say "downloading the latest version from GitHub (about 40 MB) ..."
    try {
        Invoke-WebRequest -UseBasicParsing -TimeoutSec 900 -OutFile $zip `
            -Uri "https://codeload.github.com/$owner/$name/zip/refs/heads/$branch"
    } catch {
        Stop-Setup "download failed: $($_.Exception.Message)"
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::OpenRead($zip)
    $count = 0
    $failed = $null
    foreach ($entry in $archive.Entries) {
        $rel = $entry.FullName.Substring($entry.FullName.IndexOf("/") + 1)      # drop "<repo>-<branch>/"
        $wanted = @($folders | Where-Object { $rel.StartsWith($_, [System.StringComparison]::Ordinal) })
        if ($wanted.Count -eq 0 -or $rel.EndsWith("/")) { continue }
        $dest = Join-Path $Repo ($rel.Replace("/", [string][System.IO.Path]::DirectorySeparatorChar))
        $dir = Split-Path -Parent $dest
        if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
        $ok = $false
        for ($attempt = 1; ($attempt -le 10) -and (-not $ok); $attempt++) {
            try {
                [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $dest, $true)
                $ok = $true
            } catch {
                Start-Sleep -Milliseconds 500        # another program (antivirus, sync) briefly holds the file
            }
        }
        if (-not $ok) { $failed = $dest; break }
        $count++
    }
    $archive.Dispose()
    Remove-Item -LiteralPath $zip -ErrorAction SilentlyContinue
    if ($failed) { Stop-Setup "could not write $failed - is it open in another program?" }
    if ($count -lt 50 -or -not (Test-Path -LiteralPath $launcher)) { Stop-Setup "the download looks incomplete ($count files)" }
    if ($sha) { Set-Content -LiteralPath $stampPath -Value $sha -Encoding ASCII }
    $shown = "unknown version"
    if ($sha) { $shown = $sha.Substring(0, 7) }
    Say "installed $count files ($shown) into $hmDir and assets\heavy_machines"
}

if ($NoRender) { $log.Close(); exit 0 }

# 3. render (same PowerShell executable as this one)
Say "starting the render ..."
Say "from now on you only need: $Repo\scripts\heavy_machines\render_windows.bat"
$log.Close()
$env:HM_SKIP_UPDATE = "1"
$shell = (Get-Process -Id $PID).Path
& $shell -NoProfile -ExecutionPolicy Bypass -File $launcher @args
exit $LASTEXITCODE
