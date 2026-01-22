# run command:
# powershell -ExecutionPolicy Bypass -File scripts/run_dual_region_workflow.ps1
Param(
    [string]$RepoRoot = "D:\Projects\LLaVA-NeXT",
    [string]$ImagePath = "D:/Projects/data/gazefollow/train/00000000/00000691.jpg",
    [string]$WindowsPython = "D:\pythonEnvs\p39\Scripts\python.exe",
    [string]$MaskScript = "scripts/create_dual_region_masks.py",
    [string]$MaskOutputDir = "region_masks",
    [string]$SourcePrefix = "source_region_mask",
    [string]$TargetPrefix = "target_region_mask",
    [string]$InputMode = "auto",
    [string]$WslScript = "scripts/run_dual_region_wsl.sh",
    [string[]]$ExtractionArgs = @(),
    [switch]$NoGui,
    [switch]$UniqueOutput
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

function Convert-ToWslPath {
    param ([string]$WindowsPath)
    if ($WindowsPath -match "^[A-Za-z]:") {
        $drive = $WindowsPath.Substring(0,1).ToLowerInvariant()
        $rest = $WindowsPath.Substring(2).Replace("\", "/")
        return "/mnt/$drive/$rest"
    }
    return $WindowsPath.Replace("\", "/")
}

function Get-LatestMaskPath {
    param (
        [string]$Directory,
        [string]$Prefix
    )
    $pattern = "$Prefix*.npy"
    $latest = Get-ChildItem -Path $Directory -Filter $pattern | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $latest) {
        throw "Mask file matching $pattern not found in $Directory"
    }
    return $latest.FullName
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
    "--source-prefix", $SourcePrefix,
    "--target-prefix", $TargetPrefix,
    "--input-mode", $InputMode
)

if ($NoGui) {
    $maskArgs += "--no-gui"
}

if ($UniqueOutput) {
    $maskArgs += "--unique-output"
}

Invoke-OrFail -Command $maskArgs -Stage "Mask Creation"

$sourceMaskWin = Join-Path $maskOutputPath "$SourcePrefix.npy"
$targetMaskWin = Join-Path $maskOutputPath "$TargetPrefix.npy"

if (!(Test-Path $sourceMaskWin)) {
    $sourceMaskWin = Get-LatestMaskPath -Directory $maskOutputPath -Prefix $SourcePrefix
}
if (!(Test-Path $targetMaskWin)) {
    $targetMaskWin = Get-LatestMaskPath -Directory $maskOutputPath -Prefix $TargetPrefix
}

$driveLetter = $repoRootFull.Substring(0,1).ToLowerInvariant()
$repoRootWsl = "/mnt/$driveLetter/" + $repoRootFull.Substring(3).Replace("\", "/")
$wslScriptPath = "$repoRootWsl/$($WslScript.Replace('\', '/'))"

$sourceMaskWsl = Convert-ToWslPath $sourceMaskWin
$targetMaskWsl = Convert-ToWslPath $targetMaskWin

$wslCommand = "cd $repoRootWsl && bash ""$wslScriptPath"" --source-mask-path ""$sourceMaskWsl"" --target-mask-path ""$targetMaskWsl"""

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

Write-Host "Dual-region workflow completed successfully."
