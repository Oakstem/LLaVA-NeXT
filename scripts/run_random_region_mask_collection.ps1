# run command:
# powershell -ExecutionPolicy Bypass -File scripts/run_random_region_mask_collection.ps1
Param(
    [string]$RepoRoot = "D:\Projects\LLaVA-NeXT",
    [string]$TrainDir = "D:/Projects/data/gazefollow/train",
    [string]$WindowsPython = "D:\pythonEnvs\p39\Scripts\python.exe",
    [string]$CollectorScript = "scripts/collect_random_region_masks.py",
    [string]$MaskOutputDir = "region_masks/random_samples",
    [string]$CsvPath = "region_masks/random_samples/selected_masks.csv",
    [string]$MaskSuffix = "region_mask",
    [int]$Limit = 0,
    [int]$Seed = 90426
)

function Invoke-OrFail {
    param (
        [string[]]$Command,
        [string]$Stage
    )
    Write-Host "[$Stage] Executing: $($Command -join ' ')"
    $process = Start-Process -FilePath $Command[0] -ArgumentList $Command[1..($Command.Length - 1)] -NoNewWindow -PassThru -Wait
    if ($process.ExitCode -ne 0) {
        throw "Stage '$Stage' failed with exit code $($process.ExitCode)."
    }
}

$repoRootFull = (Resolve-Path -Path $RepoRoot).Path
$collectorScriptPath = Join-Path $repoRootFull $CollectorScript
$maskOutputPath = Join-Path $repoRootFull $MaskOutputDir
$csvOutputPath = Join-Path $repoRootFull $CsvPath

if (!(Test-Path $collectorScriptPath)) {
    throw "Collector script not found at $collectorScriptPath"
}

$collectorArgs = @(
    $WindowsPython,
    $collectorScriptPath,
    "--train-dir", $TrainDir,
    "--output-dir", $maskOutputPath,
    "--csv-path", $csvOutputPath,
    "--mask-suffix", $MaskSuffix,
    "--limit", $Limit,
    "--seed", $Seed
)

Invoke-OrFail -Command $collectorArgs -Stage "Mask Collection"

Write-Host "Random mask collection completed successfully."
