# Self-healing evidence runner: keeps (re)launching the paired A/B until the
# final report is written. Any silent kill (session cleanup, reboot, crash)
# loses at most checkpoint_every turns, because every segment resumes from
# runs\paired_ab_prose.ckpt.json.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File tools\resume_evidence.ps1
#   # or with custom args:
#   powershell -ExecutionPolicy Bypass -File tools\resume_evidence.ps1 -Output runs\mine.json -Model prism-ml/bonsai-27b -Convs 2 -MaxTurns 45
#
# The final report (with "metrics") replaces the partial one at $Output.

param(
    [string]$Output = "runs\paired_ab_prose.json",
    [string]$Checkpoint = "runs\paired_ab_prose.ckpt.json",
    [string]$Corpus = "hivebench\tests\fixtures\generated_prose_horizon",
    [string]$Model = "prism-ml/bonsai-27b",
    [int]$Convs = 2,
    [int]$MaxTurns = 45,
    [int]$MaxTokens = 120,
    [int]$CheckpointEvery = 2,
    [int]$FifoBudget = 0,
    [switch]$LiveStore
)

if (-not $env:HIVE_EVIDENCE_LOG) {
    Start-Transcript -Path "$env:TEMP\hive_evidence.log" -Force | Out-Null
}

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root
$VenvPy = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"

function Is-Final($path) {
    if (-not (Test-Path $path)) { return $false }
    try {
        $doc = Get-Content $path -Raw | ConvertFrom-Json
        return -not ($doc.PSObject.Properties.Name -contains "partial")
    } catch { return $false }
}

$attempt = 0
while (-not (Is-Final $Output)) {
    $attempt++
    $args = @(
        "-m", "experiments.paired_ab", "--live",
        "--model", $Model,
        "--conversations", $Corpus,
        "--max-convs", "$Convs",
        "--max-turns", "$MaxTurns",
        "--confidence", "off",
        "--max-tokens", "$MaxTokens",
        "--no-thinking",
        "--checkpoint-every", "$CheckpointEvery",
        "--output", $Output,
        "--checkpoint", $Checkpoint
    )
    if (Test-Path $Checkpoint) { $args += @("--resume", $Checkpoint) }
    if ($FifoBudget -gt 0) { $args += @("--fifo-budget", "$FifoBudget") }
    if ($LiveStore) { $args += @("--live-store") }
    Write-Host "== attempt ${attempt}: $(Get-Date -Format HH:mm:ss)"
    & $VenvPy @args
    $code = $LASTEXITCODE
    Write-Host "== attempt $attempt exited $code at $(Get-Date -Format HH:mm:ss)"
    if ((Is-Final $Output)) { break }
    if ($code -eq 0) { Write-Host "!! exited 0 but report not final - aborting"; break }
    Start-Sleep -Seconds 5
}
Write-Host "== done: ${Output}"