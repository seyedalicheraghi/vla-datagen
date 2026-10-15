# One-time Windows setup: put the repo on the heavy-machines branch, then start the first render.
# Safe to run again at any time, from any repo state. Every step is printed, checked and logged to
#   <repo>\heavy_machine_videos\setup_windows.log
#
#   1. local edits of tracked files are saved with "git stash" (untracked files such as .venv stay as they are)
#   2. the branch is fetched over HTTPS into the remote "vla" (no SSH key needed)
#   3. untracked files that would block the checkout are moved to <repo>\_moved_aside_<time>\ (never deleted)
#   4. the branch is checked out and the launcher is verified on disk
#   5. scripts\heavy_machines\render_windows.ps1 runs (same as render_windows.bat)
#
# Windows PowerShell 5.1 syntax, plain ASCII.
param([string]$Repo = "F:\Projects\IsaacLab")

$ErrorActionPreference = "Continue"
$url = "https://github.com/seyedalicheraghi/vla-datagen.git"
$branch = "claude/review-last-request-I6mXJ"
$env:GIT_TERMINAL_PROMPT = "0"; $env:GCM_INTERACTIVE = "never"     # never wait on a credential prompt

if (-not (Test-Path (Join-Path $Repo ".git"))) {
    Write-Host "[setup] $Repo is not a git repository - run with -Repo <path to your IsaacLab clone>"
    exit 1
}
Set-Location $Repo
$logDir = Join-Path $Repo "heavy_machine_videos"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "setup_windows.log"
Set-Content -Path $log -Value "setup_windows.ps1 $(Get-Date -Format s)" -Encoding UTF8

function Say([string]$msg) {
    Write-Host "[setup] $msg" -ForegroundColor Cyan
    Add-Content -Path $log -Value "[setup] $msg" -Encoding UTF8
}

function Stop-Setup([string]$msg) {
    Say "STOPPED: $msg"
    Say "Nothing was deleted. Send Claude this file: $log"
    exit 1
}

function RunGit {
    # run git with the given arguments; show and log its output; return its exit code
    Add-Content -Path $log -Value "> git $args" -Encoding UTF8
    $out = & git @args 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
    foreach ($line in $out) {
        Write-Host "    $line"
        Add-Content -Path $log -Value "    $line" -Encoding UTF8
    }
    Add-Content -Path $log -Value "    (exit code $code)" -Encoding UTF8
    return $code
}

# 0. what we are working with
$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitCmd) { Stop-Setup "git is not on PATH" }
Say "repo $Repo | PowerShell $($PSVersionTable.PSVersion) | git at $($gitCmd.Source)"
$c = RunGit --version
$c = RunGit status --short --branch --untracked-files=no

# 1. save local edits of tracked files
$dirty = & git status --porcelain --untracked-files=no
if ($dirty) {
    Say "saving your local edits with git stash (get them back later with: git stash pop)"
    $c = RunGit stash push -m "local edits saved by setup_windows.ps1"
    if ($c -ne 0) { Stop-Setup "git stash failed" }
} else {
    Say "no local edits to save"
}

# 2. fetch the branch over HTTPS
& git remote get-url vla *> $null
if ($LASTEXITCODE -eq 0) { $c = RunGit remote set-url vla $url } else { $c = RunGit remote add vla $url }
if ($c -ne 0) { Stop-Setup "could not configure the remote 'vla'" }
Say "downloading branch $branch from $url ..."
$c = RunGit fetch vla "+refs/heads/${branch}:refs/remotes/vla/${branch}"
if ($c -ne 0) { Stop-Setup "could not download the branch from GitHub" }
$target = "refs/remotes/vla/$branch"

# 3. move aside untracked files that the checkout would otherwise refuse to overwrite
$tracked = New-Object 'System.Collections.Generic.HashSet[string]'
foreach ($p in (& git ls-files)) { [void]$tracked.Add($p) }
$incoming = & git diff --name-only --no-renames HEAD $target
$blocking = @($incoming | Where-Object { -not $tracked.Contains($_) -and (Test-Path -LiteralPath (Join-Path $Repo $_) -PathType Leaf) })
if ($blocking.Count -gt 0) {
    $aside = Join-Path $Repo ("_moved_aside_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
    Say "moving $($blocking.Count) untracked file(s) that block the checkout to $aside"
    foreach ($p in $blocking) {
        $dest = Join-Path $aside $p
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
        Move-Item -LiteralPath (Join-Path $Repo $p) -Destination $dest
        Say "    $p"
    }
}

# 4. check out the branch and verify
$c = RunGit checkout -B $branch $target
if ($c -ne 0) { Stop-Setup "git checkout failed (see the lines above)" }
$c = RunGit branch "--set-upstream-to=vla/$branch" $branch
$c = RunGit log --oneline -1
$launcher = Join-Path (Join-Path (Join-Path $Repo "scripts") "heavy_machines") "render_windows.ps1"
if (-not (Test-Path $launcher)) { Stop-Setup "checkout finished but $launcher is missing" }
Say "ready: branch $branch checked out. From now on just run: .\scripts\heavy_machines\render_windows.bat"

# 5. first render (same PowerShell executable as this one, so it also works from pwsh)
$shell = (Get-Process -Id $PID).Path
Say "starting the render launcher ..."
& $shell -NoProfile -ExecutionPolicy Bypass -File $launcher @args
exit $LASTEXITCODE
