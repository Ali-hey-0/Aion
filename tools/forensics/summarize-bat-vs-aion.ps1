param(
    [string]$RunRoot = ""
)

$ErrorActionPreference = "Stop"

function Resolve-LatestRunRoot {
    $root = Join-Path $env:APPDATA "Aion\config\logs\bat-vs-aion"
    if (-not (Test-Path -LiteralPath $root -PathType Container)) {
        throw "No bat-vs-aion capture root exists at '$root'."
    }
    $latest = Get-ChildItem -LiteralPath $root -Directory |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $latest) {
        throw "No bat-vs-aion capture runs exist under '$root'."
    }
    return $latest.FullName
}

function Read-JsonLines {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return @()
    }
    $records = New-Object System.Collections.Generic.List[object]
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line.Trim().Length -eq 0) {
            continue
        }
        $records.Add(($line | ConvertFrom-Json))
    }
    return $records
}

function Read-EnvironmentFile {
    param([string]$Path)
    if (-not $Path -or $Path.StartsWith("unavailable:") -or -not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Test-RecordBelongsToProfile {
    param(
        [object]$Record,
        [string]$ProfilePath
    )
    $cmd = [string]$Record.cmdline
    if ($cmd -like "*capture.py*") {
        return $false
    }
    if ($cmd -like "*$ProfilePath*") {
        return $true
    }
    $env = Read-EnvironmentFile -Path ([string]$Record.env_path)
    if ($env -and ([string]$env.CODEX_HOME) -eq $ProfilePath) {
        return $true
    }
    return $false
}

function Convert-RecordToLine {
    param([object]$Record)
    $cmd = [string]$Record.cmdline
    if ($cmd.Length -gt 500) {
        $cmd = $cmd.Substring(0, 500) + "..."
    }
    $cwd = [string]$Record.cwd
    return "| $($Record.observed_at) | $($Record.name) | $($Record.pid) | $($Record.ppid) | ``$cwd`` | ``$cmd`` |"
}

function Format-EnvDiff {
    param(
        [object]$BatEnv,
        [object]$AionEnv
    )
    if (-not $BatEnv -or -not $AionEnv) {
        return "A complete environment diff cannot be produced until both BAT and Aion app-server environments are captured."
    }

    $batProps = $BatEnv.PSObject.Properties | ForEach-Object { $_.Name }
    $aionProps = $AionEnv.PSObject.Properties | ForEach-Object { $_.Name }
    $all = @($batProps + $aionProps) | Sort-Object -Unique
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("| Variable | BAT | Aion | Status |")
    $lines.Add("| --- | --- | --- | --- |")
    foreach ($key in $all) {
        $batValue = [string]($BatEnv.PSObject.Properties[$key].Value)
        $aionValue = [string]($AionEnv.PSObject.Properties[$key].Value)
        $status = if ($batValue -eq $aionValue) { "same" } else { "different" }
        if ($batValue.Length -gt 180) { $batValue = $batValue.Substring(0, 180) + "..." }
        if ($aionValue.Length -gt 180) { $aionValue = $aionValue.Substring(0, 180) + "..." }
        $lines.Add("| ``$key`` | ``$batValue`` | ``$aionValue`` | $status |")
    }
    return ($lines -join "`n")
}

if (-not $RunRoot) {
    $RunRoot = Resolve-LatestRunRoot
}

$RunRoot = (Resolve-Path -LiteralPath $RunRoot).Path
$manifestPath = Join-Path $RunRoot "manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "manifest.json was not found in '$RunRoot'."
}

$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$processRecords = Read-JsonLines -Path (Join-Path $RunRoot "process-timeline.jsonl")
$fsRecords = Read-JsonLines -Path (Join-Path $RunRoot "filesystem-timeline.jsonl")
$authRecords = Read-JsonLines -Path (Join-Path $RunRoot "auth-candidates.jsonl")

$batProfile = [string]$manifest.bat.profile_path
$aionProfile = [string]$manifest.aion.profile_path

$batRecords = @($processRecords | Where-Object { Test-RecordBelongsToProfile -Record $_ -ProfilePath $batProfile })
$aionRecords = @($processRecords | Where-Object { Test-RecordBelongsToProfile -Record $_ -ProfilePath $aionProfile })

$batAppServer = $batRecords | Where-Object { ([string]$_.cmdline) -like "*app-server*" } | Select-Object -First 1
$aionAppServer = $aionRecords | Where-Object { ([string]$_.cmdline) -like "*app-server*" } | Select-Object -First 1
$batEnv = if ($batAppServer) { Read-EnvironmentFile -Path ([string]$batAppServer.env_path) } else { $null }
$aionEnv = if ($aionAppServer) { Read-EnvironmentFile -Path ([string]$aionAppServer.env_path) } else { $null }

$processReport = @"
# BAT vs Aion Process Diff

Run root: $RunRoot

## Capture Status

- BAT profile: $batProfile
- Aion profile: $aionProfile
- BAT profile process records: $($batRecords.Count)
- Aion profile process records: $($aionRecords.Count)
- BAT app-server captured: $([bool]$batAppServer)
- Aion app-server captured: $([bool]$aionAppServer)

## BAT Records

| Observed | Name | PID | PPID | CWD | Command |
| --- | --- | --- | --- | --- | --- |
$((@($batRecords | Select-Object -First 20 | ForEach-Object { Convert-RecordToLine -Record $_ }) -join "`n"))

## Aion Records

| Observed | Name | PID | PPID | CWD | Command |
| --- | --- | --- | --- | --- | --- |
$((@($aionRecords | Select-Object -First 20 | ForEach-Object { Convert-RecordToLine -Record $_ }) -join "`n"))

## Current Conclusion

If Aion process records are zero, the capture has not yet observed an actual Aion UI launch. Run the interactive capture and click Launch in Aion while the sidecar is active.
"@

$envReport = @"
# BAT vs Aion Environment Diff

Run root: $RunRoot

## App-Server Environment Comparison

$(Format-EnvDiff -BatEnv $batEnv -AionEnv $aionEnv)

## Important Handling Note

Environment files are stored locally under $RunRoot\env. They may contain sensitive local paths or tokens. Do not publish raw env files.
"@

$batFs = @($fsRecords | Where-Object { $_.root -eq "bat" })
$aionFs = @($fsRecords | Where-Object { $_.root -eq "aion" })
$fsReport = @"
# BAT vs Aion Filesystem Timeline

Run root: $RunRoot

## Capture Counts

- BAT filesystem events: $($batFs.Count)
- Aion filesystem events: $($aionFs.Count)
- Auth-candidate events: $($authRecords.Count)

## First BAT Events

| Observed | Kind | Path | Size |
| --- | --- | --- | --- |
$((@($batFs | Select-Object -First 40 | ForEach-Object { $p=[string]$_.relative_path; "| $($_.observed_at) | $($_.kind) | ``$p`` | $($_.after.size) |" }) -join "`n"))

## First Aion Events

| Observed | Kind | Path | Size |
| --- | --- | --- | --- |
$((@($aionFs | Select-Object -First 40 | ForEach-Object { $p=[string]$_.relative_path; "| $($_.observed_at) | $($_.kind) | ``$p`` | $($_.after.size) |" }) -join "`n"))

## Current Conclusion

The first observable divergence can only be called after both BAT and Aion launches are captured through the same login stage. Baseline-only runs intentionally do not prove authentication persistence.
"@

$authReport = @"
# BAT vs Aion Auth Persistence

Run root: $RunRoot

## Auth-Candidate Timeline

| Observed | Root | Kind | Path | Size | SHA-256 |
| --- | --- | --- | --- | --- | --- |
$((@($authRecords | Select-Object -First 80 | ForEach-Object { $p=[string]$_.relative_path; $h=[string]$_.after.sha256; "| $($_.observed_at) | $($_.root) | $($_.kind) | ``$p`` | $($_.after.size) | ``$h`` |" }) -join "`n"))

## Current Conclusion

Codex creates multiple authentication-adjacent Chromium artifacts before login, including `Default\Network\Cookies`, `Default\Login Data`, `Default\Account Web Data`, and `Default\Local Storage\leveldb`. The actual identity artifact must be identified by capturing the moment after successful Desktop login.
"@

$finalReport = @"
# Final Root Cause

Run root: $RunRoot

## Status

Root cause is not proven unless this run contains both BAT and Aion app-server captures and a post-login filesystem/auth timeline for both profiles.

## Evidence Captured In This Run

- BAT process records: $($batRecords.Count)
- Aion process records: $($aionRecords.Count)
- BAT app-server captured: $([bool]$batAppServer)
- Aion app-server captured: $([bool]$aionAppServer)
- BAT filesystem events: $($batFs.Count)
- Aion filesystem events: $($aionFs.Count)
- Auth-candidate events: $($authRecords.Count)

## Current Forensic Finding

The launcher cannot be convicted or cleared by command-line inspection alone. The BAT baseline demonstrates that the Desktop app writes authentication-adjacent Chromium state under the profile directory before sign-in. Therefore the next required evidence is the login-completion write timeline and the matching Aion write timeline.

## Missing Evidence If Aion Records Are Zero

The sidecar did not observe an actual Aion UI launch for profile $($manifest.aion.profile_name). Run this from the project root:

powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\forensics\bat-vs-aion-live.ps1 -Mode interactive -KeepProfiles

Then launch the generated Aion profile from the Aion UI while the script is waiting.

## Release Decision

NO. Aion is not ready for public release until the first post-login divergence is captured and explained with evidence.
"@

[System.IO.File]::WriteAllText((Join-Path (Get-Location) "BAT_vs_AION_PROCESS_DIFF.md"), $processReport, [System.Text.UTF8Encoding]::new($false))
[System.IO.File]::WriteAllText((Join-Path (Get-Location) "BAT_vs_AION_ENV_DIFF.md"), $envReport, [System.Text.UTF8Encoding]::new($false))
[System.IO.File]::WriteAllText((Join-Path (Get-Location) "BAT_vs_AION_FILESYSTEM_TIMELINE.md"), $fsReport, [System.Text.UTF8Encoding]::new($false))
[System.IO.File]::WriteAllText((Join-Path (Get-Location) "BAT_vs_AION_AUTH_PERSISTENCE.md"), $authReport, [System.Text.UTF8Encoding]::new($false))
[System.IO.File]::WriteAllText((Join-Path (Get-Location) "FINAL_ROOT_CAUSE.md"), $finalReport, [System.Text.UTF8Encoding]::new($false))

Write-Host "Reports written for run root: $RunRoot"
