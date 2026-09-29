$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$PythonLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($PythonLauncher) {
    & py -3.10 --version
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.10 was not found. Install Python 3.10 or newer and retry."
    }
    if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
        & py -3.10 -m venv .venv
    }
} else {
    $PythonCommand = Get-Command python -ErrorAction Stop
    $PythonVersion = & $PythonCommand.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ([version]$PythonVersion -lt [version]"3.10") {
        throw "Python 3.10 or newer is required; found $PythonVersion."
    }
    if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
        & $PythonCommand.Source -m venv .venv
    }
}

$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
& $VenvPython -m pip install --disable-pip-version-check --progress-bar off -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Pinned package installation failed." }
& $VenvPython -m pip install --disable-pip-version-check --progress-bar off --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw "Project installation failed." }

Write-Host "Setup complete. Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "Next run: python -m teacher_reliability.env_check"
