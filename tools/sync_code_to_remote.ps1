[CmdletBinding()]
param(
    [string]$ProjectRoot
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
} else {
    $ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
}

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

$remote = 'eeeeee-sftp:/home/user/EEEEEE'
$dataDirName = -join ([char[]](0x45, 0x9898, 0x6570, 0x636E))
$arguments = @(
    'sync',
    $ProjectRoot,
    $remote,
    '--progress',
    '--create-empty-src-dirs',
    '--checkers', '8',
    '--transfers', '4',
    '--retries', '3',
    '--low-level-retries', '10',
    '--exclude', '.git/**',
    '--exclude', '.venv/**',
    '--exclude', 'venv/**',
    '--exclude', '__pycache__/**',
    '--exclude', '*.pyc',
    '--exclude', '.env',
    '--exclude', '.DS_Store',
    '--exclude', '**/.DS_Store',
    '--exclude', "$dataDirName/**",
    '--exclude', 'results/**',
    '--exclude', 'runs/**',
    '--exclude', 'outputs/**',
    '--exclude', 'logs/**',
    '--exclude', 'checkpoints/**',
    '--exclude', '.vscode/**',
    '--exclude', 'tools/**'
)

Write-Host "Syncing code: $ProjectRoot -> $remote"
& $rclone @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Code sync failed with exit code $LASTEXITCODE"
}

Write-Host 'Code sync completed. Remote data and run results were preserved.'
