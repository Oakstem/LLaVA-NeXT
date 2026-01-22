# run command:
# powershell -ExecutionPolicy Bypass -File scripts/run_single_region_workflow.ps1
Param(
    [string]$RepoRoot = "D:\Projects\LLaVA-NeXT",
    # [string]$ImagePath = "D:\Projects\data\gazefollow\train\00000093\00093143.jpg",
    # [string]$ImagePath = "D:\Projects\data\gazefollow\train\00000012\00012014.jpg",
    [string]$ImagePath = "D:/Projects/data/gazefollow/train/00000000/00000691.jpg",
    [string]$WindowsPython = "D:\pythonEnvs\p39\Scripts\python.exe",
    [string]$MaskScript = "scripts/create_region_mask_from_bbox.py",
    [string]$MaskOutputDir = "region_masks",
    [string]$MaskPrefix = "active_region_mask",
    [string]$WslScript = "scripts/run_single_region_wsl.sh",
    [string[]]$ExtractionArgs = @(),
    [switch]$NoGui,
    [switch]$OverwriteMask = $true
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
$maskScriptPath = Join-Path $repoRootFull $MaskScript
$maskOutputPath = Join-Path $repoRootFull $MaskOutputDir

if (!(Test-Path $maskScriptPath)) {
    throw "Mask script not found at $maskScriptPath"
}

$maskArgs = @(
    $WindowsPython,
    $maskScriptPath,
    "--image-path", $ImagePath,
    "--output-dir", $maskOutputPath,
    "--mask-prefix", $MaskPrefix
)

if ($NoGui) {
    $maskArgs += "--no-gui"
}

if ($OverwriteMask) {
    $maskArgs += "--overwrite"
}

Invoke-OrFail -Command $maskArgs -Stage "Mask Creation"

$driveLetter = $repoRootFull.Substring(0,1).ToLowerInvariant()
$repoRootWsl = "/mnt/$driveLetter/" + $repoRootFull.Substring(3).Replace("\", "/")
$wslScriptPath = "$repoRootWsl/$($WslScript.Replace('\', '/'))"
$wslCommand = "cd $repoRootWsl && bash ""$wslScriptPath"""

if ($ExtractionArgs.Length -gt 0) {
    $argString = ""
    foreach ($arg in $ExtractionArgs) {
        $argString += " $arg"
    }
    $wslCommand += $argString
}

Write-Host "[WSL] Executing: $wslCommand"
$wslProcess = Start-Process -FilePath "wsl.exe" -ArgumentList @("bash", "-lc", $wslCommand) -NoNewWindow -PassThru -Wait
if ($wslProcess.ExitCode -ne 0) {
    throw "WSL extraction failed with exit code $($wslProcess.ExitCode)."
}

Write-Host "Workflow completed successfully."
