# Render a heavy-machine scenario on Windows with the Isaac Sim pip install in <repo>\.venv.
#
# Run through render_windows.bat (works in any PowerShell or cmd window, no venv activation, no
# execution-policy change):
#     scripts\heavy_machines\render_windows.bat                                 # adt_approach_hex, all cameras
#     scripts\heavy_machines\render_windows.bat --renderer rtx --cameras front  # extra args go to render_scenario.py
#
# Steps: update the code (fast-forward only), check the venv, make sure ffmpeg is available for the
# mp4, render, then open the output folder.  Isaac Sim runs directly in this console (not piped), so
# its first-run licence prompt and progress show exactly as if typed by hand; render_scenario.py keeps
# its own log in heavy_machine_videos\<scenario>\render.log.
#
# Kept to Windows PowerShell 5.1 syntax and plain ASCII (5.1 reads BOM-less scripts as ANSI).

$ErrorActionPreference = "Continue"
$repo = (Resolve-Path (Join-Path $PSScriptRoot (Join-Path ".." ".."))).Path
Set-Location $repo
$py = Join-Path (Join-Path (Join-Path $repo ".venv") "Scripts") "python.exe"
$render = Join-Path (Join-Path (Join-Path (Join-Path $repo "scripts") "heavy_machines") "scenarios") "render_scenario.py"

function Say([string]$msg) { Write-Host "[render_windows] $msg" -ForegroundColor Cyan }

if (-not (Test-Path $py)) {
    Say "No Python venv at $py"
    Say "Create it where Isaac Sim is installed, or edit `$py at the top of this script."
    exit 1
}

# 1. latest code from the tracked branch; never touches local edits
$branch = (git rev-parse --abbrev-ref HEAD 2>$null)
if ($LASTEXITCODE -eq 0) {
    Say "updating branch $branch ..."
    $env:GIT_TERMINAL_PROMPT = "0"; $env:GCM_INTERACTIVE = "never"     # never wait on a credential prompt
    git pull --ff-only --quiet
    if ($LASTEXITCODE -ne 0) { Say "git pull skipped (offline, no upstream or local changes) - using the code on disk" }
}

# 2. the venv must hold Isaac Sim and Isaac Lab
& $py -c "import importlib.metadata as m; print('[render_windows] Python', __import__('sys').version.split()[0], '| isaacsim', m.version('isaacsim'), '| isaaclab', m.version('isaaclab'))"
if ($LASTEXITCODE -ne 0) {
    Say "Isaac Sim or Isaac Lab is not installed in $repo\.venv (see the error above)."
    exit 1
}

# 3. ffmpeg for the mp4 (frames are written either way)
& $py -c "import shutil, sys; import importlib.util as u; sys.exit(0 if shutil.which('ffmpeg') or u.find_spec('imageio_ffmpeg') else 1)"
if ($LASTEXITCODE -ne 0) {
    Say "no ffmpeg found - installing imageio-ffmpeg into .venv for the mp4 encoding ..."
    & $py -m pip --version *> $null
    if ($LASTEXITCODE -eq 0) {
        & $py -m pip install --quiet imageio-ffmpeg
    } elseif (Get-Command uv -ErrorAction SilentlyContinue) {
        uv pip install --quiet --python $py imageio-ffmpeg
    } else {
        Say "neither pip nor uv available - the render will write PNG frames only"
    }
}

# 4. render (output straight to this console)
$env:PYTHONUNBUFFERED = "1"
Say "rendering: python $render --headless $args"
& $py $render --headless @args
$code = $LASTEXITCODE

# 5. result
$videos = Join-Path $repo "heavy_machine_videos"
if ($code -eq 0) {
    Say "done. Videos:"
    Get-ChildItem -Path $videos -Recurse -Filter *.mp4 -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | ForEach-Object { Write-Host "    $($_.FullName)" }
    if ($env:OS -eq "Windows_NT" -and (Test-Path $videos)) { Invoke-Item $videos }
} else {
    Say "FAILED (exit code $code). Send this file to Claude:"
    Get-ChildItem -Path $videos -Recurse -Filter render.log -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { Write-Host "    $($_.FullName)" }
}
exit $code
