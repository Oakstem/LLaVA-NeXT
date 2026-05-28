# Run vanilla inference demo server via WSL
# Usage: powershell -ExecutionPolicy Bypass -File scripts/run_vanilla_inference_demo.ps1
Param(
    [string]$RepoRoot = "D:\Projects\LLaVA-NeXT",
    [int]$Port = 7861,
    #[string]$AdapterPath = "training_outputs/llava-20260304_225738/checkpoint-10000",
    [string]$AdapterPath = "training_outputs/llava-20260420_030228/checkpoint-6000",
    [string]$ModelPath = "lmms-lab/llava-onevision-qwen2-7b-ov-chat",
    [switch]$Load4bit,
    [switch]$Load8bit
)

$repoRootFull = (Resolve-Path -Path $RepoRoot).Path
$driveLetter = $repoRootFull.Substring(0,1).ToLowerInvariant()
$repoRootWsl = "/mnt/$driveLetter/" + $repoRootFull.Substring(3).Replace("\", "/")

$pythonBin = "/home/alonz/llava/bin/python"
$serverScript = "demo/vanilla_server.py"

$wslArgs = @(
    "$pythonBin",
    "$serverScript",
    "--port", "$Port",
    "--adapter-path", "`"$AdapterPath`"",
    "--model-path", "`"$ModelPath`""
)

if ($Load4bit) { $wslArgs += "--load-4bit" }
if ($Load8bit) { $wslArgs += "--load-8bit" }

$wslCommand = "cd $repoRootWsl && " + ($wslArgs -join " ")

Write-Host "=========================================="
Write-Host " Vanilla Inference Demo Server"
Write-Host "=========================================="
Write-Host "Repo:     $repoRootWsl"
Write-Host "Adapter:  $AdapterPath"
Write-Host "Port:     $Port"
Write-Host "URL:      http://localhost:$Port/vanilla/"
Write-Host "=========================================="
Write-Host ""
Write-Host "[WSL] Executing: $wslCommand"

$wslProcess = Start-Process -FilePath "wsl.exe" `
    -ArgumentList @("bash", "-lc", $wslCommand) `
    -NoNewWindow -PassThru -Wait

if ($wslProcess.ExitCode -ne 0) {
    Write-Host "Server exited with code $($wslProcess.ExitCode)" -ForegroundColor Red
}
