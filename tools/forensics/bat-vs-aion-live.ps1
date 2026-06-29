param(
    [ValidateSet("baseline", "interactive")]
    [string]$Mode = "baseline",
    [int]$BaselineSeconds = 45,
    [switch]$KeepProfiles,
    [string]$CodexExe = ""
)

$ErrorActionPreference = "Stop"

function Convert-ToNormalWindowsPath {
    param([Parameter(Mandatory = $true)][string]$PathText)
    if ($PathText.StartsWith("\\?\")) {
        return $PathText.Substring(4)
    }
    return $PathText
}

function Resolve-CodexExecutable {
    param([string]$ExplicitPath)

    if ($ExplicitPath -and (Test-Path -LiteralPath $ExplicitPath -PathType Leaf)) {
        return (Resolve-Path -LiteralPath $ExplicitPath).Path
    }

    $appConfigPath = Join-Path $env:APPDATA "Aion\config\app.json"
    if (Test-Path -LiteralPath $appConfigPath -PathType Leaf) {
        try {
            $config = Get-Content -LiteralPath $appConfigPath -Raw | ConvertFrom-Json
            if ($config.custom_codex_path) {
                $candidate = Convert-ToNormalWindowsPath ([string]$config.custom_codex_path)
                if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                    return (Resolve-Path -LiteralPath $candidate).Path
                }
            }
        } catch {
            Write-Warning "Could not read Aion custom Codex path from '$appConfigPath': $($_.Exception.Message)"
        }
    }

    $windowsApps = Join-Path $env:ProgramFiles "WindowsApps"
    $packages = Get-ChildItem -LiteralPath $windowsApps -Directory -Filter "OpenAI.Codex_*" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending
    foreach ($package in $packages) {
        $candidate = Join-Path $package.FullName "app\Codex.exe"
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }

    throw "Codex.exe was not found. Pass -CodexExe with an absolute path."
}

function Get-UnixTimeSeconds {
    return [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
}

function Write-JsonFile {
    param(
        [Parameter(Mandatory = $true)]$Value,
        [Parameter(Mandatory = $true)][string]$Path
    )
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $json = $Value | ConvertTo-Json -Depth 16
    [System.IO.File]::WriteAllText($Path, $json, [System.Text.UTF8Encoding]::new($false))
}

function New-AionDiagnosticProfile {
    param(
        [Parameter(Mandatory = $true)][string]$ProfileId,
        [Parameter(Mandatory = $true)][string]$ProfileName
    )

    $profilesDir = Join-Path $env:APPDATA "Aion\config\profiles"
    New-Item -ItemType Directory -Force -Path $profilesDir | Out-Null

    $profile = [ordered]@{
        id = $ProfileId
        name = $ProfileName
        email = "forensics+aion@example.invalid"
        color_tag = "#4F46E5"
        created_at = Get-UnixTimeSeconds
        last_launched = $null
        usage_week_hours = 0.0
        usage_5h_hours = 0.0
        activated_at = $null
        expires_at = $null
        proxy = $null
    }

    $profilePath = Join-Path $profilesDir "$ProfileId.json"
    Write-JsonFile -Value $profile -Path $profilePath
    return $profilePath
}

function New-LegacyBatchLauncher {
    param(
        [Parameter(Mandatory = $true)][string]$LauncherPath,
        [Parameter(Mandatory = $true)][string]$CodexPath,
        [Parameter(Mandatory = $true)][string]$ProfilePath
    )

    $content = @"
@echo off
set "CODEX_HOME=$ProfilePath"
start "" "$CodexPath" --user-data-dir="%CODEX_HOME%"
exit /b 0
"@
    [System.IO.File]::WriteAllText($LauncherPath, $content.Replace("`n", "`r`n"), [System.Text.Encoding]::ASCII)
}

function Write-CapturePython {
    param([Parameter(Mandatory = $true)][string]$Path)

    $python = @'
import hashlib
import json
import os
import sys
import time
from pathlib import Path

try:
    import psutil
except Exception as exc:
    psutil = None
    PSUTIL_ERROR = str(exc)
else:
    PSUTIL_ERROR = ""

out_root = Path(sys.argv[1])
stop_file = Path(sys.argv[2])
bat_profile = Path(sys.argv[3])
aion_profile = Path(sys.argv[4])
global_codex = Path(sys.argv[5])

process_log = out_root / "process-timeline.jsonl"
filesystem_log = out_root / "filesystem-timeline.jsonl"
auth_log = out_root / "auth-candidates.jsonl"
env_dir = out_root / "env"
env_dir.mkdir(parents=True, exist_ok=True)

tracked_roots = {
    "bat": bat_profile,
    "aion": aion_profile,
    "global_codex": global_codex,
}

auth_tokens = (
    "auth", "token", "session", "cookie", "local state", "login data",
    "web data", "indexeddb", "leveldb", "sqlite", "state_", "logs_",
    "memories_", ".codex", "config.toml"
)

def now_record():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()) + (".%03d" % int((time.time() % 1) * 1000))

def append_jsonl(path, value):
    value["observed_at"] = now_record()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")

def safe_hash(path):
    try:
        size = path.stat().st_size
        if size > 8 * 1024 * 1024:
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None

def snapshot_tree(root):
    result = {}
    if not root.exists():
        return result
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {"GPUCache", "GrShaderCache", "ShaderCache", "Code Cache", "Cache"}]
        for name in files:
            path = Path(current) / name
            try:
                stat = path.stat()
                rel = str(path.relative_to(root))
                lower_rel = rel.lower()
                is_auth_candidate = any(token in lower_rel for token in auth_tokens)
                result[rel] = {
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "ctime_ns": stat.st_ctime_ns,
                    "sha256": safe_hash(path) if is_auth_candidate or stat.st_size <= 1024 * 1024 else None,
                    "auth_candidate": is_auth_candidate,
                }
            except Exception as exc:
                rel = str(path)
                result[rel] = {"error": str(exc)}
    return result

def record_tree_diff(label, before, after):
    before_keys = set(before.keys())
    after_keys = set(after.keys())
    for rel in sorted(after_keys - before_keys):
        event = {"kind": "created", "root": label, "relative_path": rel, "after": after[rel]}
        append_jsonl(filesystem_log, event)
        if after[rel].get("auth_candidate"):
            append_jsonl(auth_log, event)
    for rel in sorted(before_keys - after_keys):
        event = {"kind": "deleted", "root": label, "relative_path": rel, "before": before[rel]}
        append_jsonl(filesystem_log, event)
        if before[rel].get("auth_candidate"):
            append_jsonl(auth_log, event)
    for rel in sorted(before_keys & after_keys):
        if before[rel] != after[rel]:
            event = {"kind": "modified", "root": label, "relative_path": rel, "before": before[rel], "after": after[rel]}
            append_jsonl(filesystem_log, event)
            if before[rel].get("auth_candidate") or after[rel].get("auth_candidate"):
                append_jsonl(auth_log, event)

def proc_cmdline(proc):
    try:
        return " ".join(proc.cmdline())
    except Exception:
        return ""

def capture_processes(seen_env):
    if psutil is None:
        append_jsonl(process_log, {"kind": "error", "message": f"psutil unavailable: {PSUTIL_ERROR}"})
        return seen_env

    markers = [str(bat_profile), str(aion_profile), bat_profile.name, aion_profile.name]
    for proc in psutil.process_iter(["pid", "ppid", "name", "exe", "create_time"]):
        try:
            name = proc.info.get("name") or ""
            cmdline = proc_cmdline(proc)
            lower_name = name.lower()
            if lower_name not in {"codex.exe", "aion.exe", "cmd.exe", "powershell.exe"} and not any(m in cmdline for m in markers):
                continue
            if lower_name in {"cmd.exe", "powershell.exe"} and not any(m in cmdline for m in markers):
                continue
            cwd = ""
            try:
                cwd = proc.cwd()
            except Exception:
                pass
            exe = proc.info.get("exe") or ""
            parent_pid = proc.info.get("ppid")
            env_path = None
            env_digest = None
            if lower_name in {"codex.exe", "aion.exe"} or "resources\\codex.exe" in cmdline.lower() or any(m in cmdline for m in markers):
                try:
                    env = proc.environ()
                    encoded = json.dumps(env, ensure_ascii=False, sort_keys=True)
                    env_digest = hashlib.sha256(encoded.encode("utf-8", "replace")).hexdigest()
                    key = f"{proc.pid}-{env_digest}"
                    if key not in seen_env:
                        env_path = env_dir / f"{proc.pid}-{env_digest}.json"
                        with env_path.open("w", encoding="utf-8") as handle:
                            json.dump(env, handle, ensure_ascii=False, indent=2, sort_keys=True)
                        seen_env.add(key)
                    else:
                        env_path = env_dir / f"{proc.pid}-{env_digest}.json"
                except Exception as exc:
                    env_path = f"unavailable: {exc}"
            append_jsonl(process_log, {
                "kind": "process_observed",
                "pid": proc.pid,
                "ppid": parent_pid,
                "name": name,
                "exe": exe,
                "cwd": cwd,
                "cmdline": cmdline,
                "create_time": proc.info.get("create_time"),
                "env_sha256": env_digest,
                "env_path": str(env_path) if env_path else None,
            })
        except Exception as exc:
            append_jsonl(process_log, {"kind": "process_error", "message": str(exc)})
    return seen_env

append_jsonl(process_log, {
    "kind": "capture_started",
    "pid": os.getpid(),
    "bat_profile": str(bat_profile),
    "aion_profile": str(aion_profile),
    "global_codex": str(global_codex),
})

previous = {label: snapshot_tree(root) for label, root in tracked_roots.items()}
seen_env = set()

while not stop_file.exists():
    seen_env = capture_processes(seen_env)
    current = {label: snapshot_tree(root) for label, root in tracked_roots.items()}
    for label in sorted(tracked_roots.keys()):
        record_tree_diff(label, previous.get(label, {}), current.get(label, {}))
    previous = current
    time.sleep(0.75)

seen_env = capture_processes(seen_env)
final_snapshot_path = out_root / "final-filesystem-snapshot.json"
with final_snapshot_path.open("w", encoding="utf-8") as handle:
    json.dump(previous, handle, ensure_ascii=False, indent=2, sort_keys=True)

append_jsonl(process_log, {"kind": "capture_stopped"})
'@
    [System.IO.File]::WriteAllText($Path, $python, [System.Text.UTF8Encoding]::new($false))
}

function Start-Capture {
    param(
        [Parameter(Mandatory = $true)][string]$OutRoot,
        [Parameter(Mandatory = $true)][string]$BatProfile,
        [Parameter(Mandatory = $true)][string]$AionProfile
    )

    $capturePy = Join-Path $OutRoot "capture.py"
    $stopFile = Join-Path $OutRoot "STOP"
    $globalCodex = Join-Path $env:USERPROFILE ".codex"
    Write-CapturePython -Path $capturePy
    $job = Start-Job -ScriptBlock {
        param($ScriptPath, $OutRoot, $StopFile, $BatProfile, $AionProfile, $GlobalCodex)
        python $ScriptPath $OutRoot $StopFile $BatProfile $AionProfile $GlobalCodex
    } -ArgumentList $capturePy, $OutRoot, $stopFile, $BatProfile, $AionProfile, $globalCodex

    return [pscustomobject]@{
        Job = $job
        StopFile = $stopFile
    }
}

function Stop-Capture {
    param([Parameter(Mandatory = $true)]$Capture)
    New-Item -ItemType File -Force -Path $Capture.StopFile | Out-Null
    Wait-Job -Job $Capture.Job -Timeout 10 | Out-Null
    Receive-Job -Job $Capture.Job -ErrorAction SilentlyContinue | Out-Null
    Remove-Job -Job $Capture.Job -Force -ErrorAction SilentlyContinue
}

function Write-RunManifest {
    param(
        [string]$Path,
        [string]$RunId,
        [string]$CodexPath,
        [string]$BatProfile,
        [string]$BatLauncher,
        [string]$AionProfileId,
        [string]$AionProfileName,
        [string]$AionProfilePath,
        [string]$Mode
    )

    $manifest = [ordered]@{
        run_id = $RunId
        mode = $Mode
        created_at = (Get-Date).ToString("o")
        codex_executable = $CodexPath
        bat = [ordered]@{
            profile_path = $BatProfile
            launcher = $BatLauncher
            command_model = 'set "CODEX_HOME=<profile>"; start "" "<Codex.exe>" --user-data-dir="%CODEX_HOME%"'
        }
        aion = [ordered]@{
            profile_id = $AionProfileId
            profile_name = $AionProfileName
            profile_path = $AionProfilePath
            launch_instruction = "Open Aion, select '$AionProfileName', click Launch Profile."
        }
        outputs = [ordered]@{
            process_timeline = "process-timeline.jsonl"
            filesystem_timeline = "filesystem-timeline.jsonl"
            auth_candidates = "auth-candidates.jsonl"
            environment_directory = "env"
            final_filesystem_snapshot = "final-filesystem-snapshot.json"
        }
    }
    Write-JsonFile -Value $manifest -Path $Path
}

$runId = Get-Date -Format "yyyyMMdd-HHmmss"
$outRoot = Join-Path $env:APPDATA "Aion\config\logs\bat-vs-aion\$runId"
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

$codexPath = Resolve-CodexExecutable -ExplicitPath $CodexExe
$codexPath = Convert-ToNormalWindowsPath $codexPath

$profilesRoot = Join-Path $env:USERPROFILE "CodexProfiles"
New-Item -ItemType Directory -Force -Path $profilesRoot | Out-Null

$batProfileName = "BATLive-$runId"
$batProfile = Join-Path $profilesRoot $batProfileName
$aionProfileId = [guid]::NewGuid().ToString()
$aionProfileName = "AionLive-$runId"
$aionProfile = Join-Path $profilesRoot $aionProfileId

New-Item -ItemType Directory -Force -Path $batProfile | Out-Null
New-Item -ItemType Directory -Force -Path $aionProfile | Out-Null

$aionProfileDocument = New-AionDiagnosticProfile -ProfileId $aionProfileId -ProfileName $aionProfileName
$batLauncher = Join-Path $outRoot "legacy-bat-launch.cmd"
New-LegacyBatchLauncher -LauncherPath $batLauncher -CodexPath $codexPath -ProfilePath $batProfile

Write-RunManifest `
    -Path (Join-Path $outRoot "manifest.json") `
    -RunId $runId `
    -CodexPath $codexPath `
    -BatProfile $batProfile `
    -BatLauncher $batLauncher `
    -AionProfileId $aionProfileId `
    -AionProfileName $aionProfileName `
    -AionProfilePath $aionProfile `
    -Mode $Mode

$capture = Start-Capture -OutRoot $outRoot -BatProfile $batProfile -AionProfile $aionProfile

try {
    Write-Host "Aion forensic run: $runId"
    Write-Host "Output: $outRoot"
    Write-Host "BAT profile: $batProfile"
    Write-Host "Aion profile name: $aionProfileName"
    Write-Host "Aion profile path: $aionProfile"
    Write-Host ""
    Write-Host "Launching BAT reference profile..."
    Start-Process -FilePath $batLauncher

    if ($Mode -eq "interactive") {
        Write-Host ""
        Write-Host "Complete BAT login/restart behavior now if needed, then press Enter."
        [void][System.Console]::ReadLine()
        Write-Host ""
        Write-Host "Now open Aion, select '$aionProfileName', click Launch Profile, perform the same login/restart behavior, then press Enter."
        [void][System.Console]::ReadLine()
    } else {
        Write-Host "Baseline capture is running for $BaselineSeconds seconds."
        Write-Host "For login-stage evidence, rerun with: powershell -ExecutionPolicy Bypass -File .\tools\forensics\bat-vs-aion-live.ps1 -Mode interactive"
        Start-Sleep -Seconds $BaselineSeconds
    }
} finally {
    Stop-Capture -Capture $capture

    if (-not $KeepProfiles -and $Mode -eq "baseline") {
        Remove-Item -LiteralPath $aionProfileDocument -Force -ErrorAction SilentlyContinue
    }

    Write-Host ""
    Write-Host "Capture complete."
    Write-Host "Artifacts: $outRoot"
}
