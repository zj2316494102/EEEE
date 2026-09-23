[CmdletBinding()]
param(
    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$RunId
)

$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$localResults = Join-Path $projectRoot 'results'
New-Item -ItemType Directory -Force -Path $localResults | Out-Null

$rcloneCommand = Get-Command rclone.exe -ErrorAction SilentlyContinue
if ($rcloneCommand) {
    $rclone = $rcloneCommand.Source
} else {
    $rclone = Get-ChildItem `
        -Path (Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages') `
        -Filter 'rclone.exe' `
        -Recurse `
        -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}

if ([string]::IsNullOrWhiteSpace($rclone) -or -not (Test-Path -LiteralPath $rclone)) {
    throw 'rclone was not found. Reopen the terminal or install Rclone.'
}

if ([string]::IsNullOrWhiteSpace($RunId)) {
    $remote = 'eeeeee-sftp:/home/user/EEEEEE/runs'
    $target = $localResults
    Write-Host "Downloading all remote run results: $remote -> $target"
} else {
    & ssh eeeeee-remote "test -f '/home/user/EEEEEE/runs/$RunId/DONE'"
    if ($LASTEXITCODE -ne 0) {
        throw "Run $RunId has no DONE marker; incomplete results will not be downloaded."
    }

    $remote = "eeeeee-sftp:/home/user/EEEEEE/runs/$RunId"
    $target = Join-Path $localResults $RunId
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    Write-Host "Downloading run result: $remote -> $target"
}

& $rclone copy $remote $target `
    --progress `
    --create-empty-src-dirs `
    --checkers 8 `
    --transfers 4 `
    --retries 3 `
    --low-level-retries 10

if ($LASTEXITCODE -ne 0) {
    throw "Result download failed with exit code $LASTEXITCODE"
}

Write-Host 'Result download completed. Local historical results were preserved.'
