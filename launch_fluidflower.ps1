param(
    [ValidateSet('studio', 'experiments', 'native')][string]$Application = 'studio',
    [string]$SelectModel = ''
)

$ErrorActionPreference = 'Stop'
$repository = $PSScriptRoot
$environmentRoot = Join-Path $repository '.venv-fluidflower'
$python = Join-Path $environmentRoot 'Scripts\python.exe'
$venvConfig = Get-Content -LiteralPath (Join-Path $environmentRoot 'pyvenv.cfg')
$baseHome = ($venvConfig | Where-Object { $_ -match '^home\s*=' } | Select-Object -First 1) -replace '^home\s*=\s*', ''
$pythonw = Join-Path $baseHome 'pythonw.exe'
if (-not (Test-Path -LiteralPath $pythonw)) {
    throw "Install the FluidFlower environment first: $environmentRoot (see FLUIDFLOWER.md)."
}

# Use installed packages, never inherit the former ff_ac source bootstrap.
$env:PYTHONPATH = $null
$env:PYTHONHOME = $null
$env:VIRTUAL_ENV = $environmentRoot
$env:DARSIA_FLUIDFLOWER_PYTHON = $python
$env:PYTHONFAULTHANDLER = '1'
if ($env:QT_QPA_PLATFORM -eq 'offscreen') { $env:QT_QPA_PLATFORM = $null }
$logRoot = Join-Path $env:LOCALAPPDATA "DarSIA\logs\central-$Application"
$workingRoot = if ($env:DARSIA_FLUIDFLOWER_CACHE) { $env:DARSIA_FLUIDFLOWER_CACHE } else { Join-Path $env:LOCALAPPDATA 'DarSIA\cache' }
New-Item -ItemType Directory -Path $logRoot, $workingRoot -Force | Out-Null

# Reuse only a current Studio from this installation. Other windows may own
# unsaved reviews or jobs and are never killed by this launcher.
if ($Application -eq 'studio' -and -not $SelectModel) {
    $source = Join-Path $repository 'src\darsia\fluidflower\ui\model_training.py'
    $sourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
    $latestCode = (Get-ChildItem -LiteralPath (Join-Path $repository 'src\darsia\fluidflower') -Filter '*.py' -File -Recurse |
        Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1).LastWriteTimeUtc
    foreach ($candidate in (Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -like '*Phase Model Studio*' })) {
        $receipt = Join-Path $env:LOCALAPPDATA "DarSIA\logs\phase-model-studio\runtime-$($candidate.Id).json"
        if (-not (Test-Path -LiteralPath $receipt)) { continue }
        $identity = Get-Content -LiteralPath $receipt -Raw | ConvertFrom-Json
        if ($identity.training_page_source -eq $source -and $identity.training_page_sha256 -eq $sourceHash -and $candidate.StartTime.ToUniversalTime() -ge $latestCode) {
            $shell = New-Object -ComObject WScript.Shell
            $null = $shell.AppActivate($candidate.Id)
            @{pid=$candidate.Id; reused=$true; repository=$repository} | ConvertTo-Json
            return
        }
    }
}
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$stdout = Join-Path $logRoot "$stamp.stdout.log"
$stderr = Join-Path $logRoot "$stamp.stderr.log"
$runner = Join-Path $repository 'run_fluidflower.pyw'
$arguments = @(('"' + $runner + '"'), $Application)
if ($SelectModel) {
    if ($Application -ne 'studio') { throw 'SelectModel applies only to Studio.' }
    $modelPath = (Resolve-Path -LiteralPath $SelectModel).ProviderPath
    $arguments += @('--select-model', ('"' + $modelPath + '"'))
}
# This is the interactive Studio window, not a background worker/service.
# The real pythonw interpreter creates no console; Qt must receive a visible
# startup state or Windows hides its first top-level window.
$process = Start-Process -FilePath $pythonw -ArgumentList $arguments -WorkingDirectory $workingRoot -WindowStyle Normal -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
@{pid=$process.Id; reused=$false; repository=$repository; python=$pythonw; stdout=$stdout; stderr=$stderr} | ConvertTo-Json
