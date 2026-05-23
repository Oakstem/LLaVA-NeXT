# Run region extraction demo server via WSL
# Usage: powershell -ExecutionPolicy Bypass -File scripts/run_region_extraction_demo.ps1
Param(
    [string]$RepoRoot = "D:\Projects\LLaVA-NeXT",
    [int]$Port = 7860,
    [string]$HostAddress = "0.0.0.0",
    [string]$ModelPath = "lmms-lab/llava-onevision-qwen2-7b-ov-chat",
    [string]$AttnImplementation = "sdpa",
    [int]$AttnLayerInd = -1,
    [switch]$Load4bit,
    [switch]$Load8bit
)

$repoRootFull = (Resolve-Path -Path $RepoRoot).Path
$driveLetter = $repoRootFull.Substring(0,1).ToLowerInvariant()
$repoRootWsl = "/mnt/$driveLetter/" + $repoRootFull.Substring(3).Replace("\", "/")

$pythonBin = "/home/alonz/llava/bin/python"
$serverScript = "demo/server.py"

$wslArgs = @(
    "$pythonBin",
    "$serverScript",
    "--host", "$HostAddress",
    "--port", "$Port",
    "--model-path", "`"$ModelPath`"",
    "--attn-implementation", "$AttnImplementation",
    "--attn-layer-ind", "$AttnLayerInd"
)

if ($Load4bit) { $wslArgs += "--load-4bit" }
if ($Load8bit) { $wslArgs += "--load-8bit" }

$wslCommand = "cd $repoRootWsl && " + ($wslArgs -join " ")

Write-Host "=========================================="
Write-Host " Region Extraction Demo Server"
Write-Host "=========================================="
Write-Host "Repo:     $repoRootWsl"
Write-Host "Model:    $ModelPath"
Write-Host "Host:     $HostAddress"
Write-Host "Port:     $Port"
Write-Host "URL:      http://localhost:$Port/"
Write-Host "=========================================="
Write-Host ""
Write-Host "[WSL] Executing: $wslCommand"

$wslProcess = Start-Process -FilePath "wsl.exe" `
    -ArgumentList @("bash", "-lc", $wslCommand) `
    -NoNewWindow -PassThru -Wait

if ($wslProcess.ExitCode -ne 0) {
    Write-Host "Server exited with code $($wslProcess.ExitCode)" -ForegroundColor Red
}
