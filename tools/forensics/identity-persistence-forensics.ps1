param(
    [ValidateSet("bat", "aion", "both")]
    [string]$Target = "both",
    [string]$CodexExe = "",
    [switch]$KeepTestProfiles
)

$ErrorActionPreference = "Stop"

function Get-RunId {
    return Get-Date -Format "yyyyMMdd-HHmmss"
}

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
            Write-Warning "Unable to read Aion custom Codex path: $($_.Exception.Message)"
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

function Write-TextFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Content
    )
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    [System.IO.File]::WriteAllText($Path, $Content, [System.Text.UTF8Encoding]::new($false))
}

function Write-JsonFile {
    param(
        [Parameter(Mandatory = $true)]$Value,
        [Parameter(Mandatory = $true)][string]$Path
    )
    Write-TextFile -Path $Path -Content (($Value | ConvertTo-Json -Depth 32))
}

function Write-LegacyBatLauncher {
    param(
        [Parameter(Mandatory = $true)][string]$LauncherPath,
        [Parameter(Mandatory = $true)][string]$ProfilePath,
        [Parameter(Mandatory = $true)][string]$CodexPath
    )
    $content = @"
@echo off
set "CODEX_HOME=$ProfilePath"
start "" "$CodexPath" --user-data-dir="%CODEX_HOME%"
exit /b 0
"@
    [System.IO.File]::WriteAllText($LauncherPath, $content.Replace("`n", "`r`n"), [System.Text.Encoding]::ASCII)
}

function New-AionProfileDocument {
    param(
        [Parameter(Mandatory = $true)][string]$ProfileId,
        [Parameter(Mandatory = $true)][string]$ProfileName
    )

    $profilesDir = Join-Path $env:APPDATA "Aion\config\profiles"
    New-Item -ItemType Directory -Force -Path $profilesDir | Out-Null
    $profilePath = Join-Path $profilesDir "$ProfileId.json"
    $now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    $profile = [ordered]@{
        id = $ProfileId
        name = $ProfileName
        email = "identity-forensics@example.invalid"
        color_tag = "#4F46E5"
        created_at = $now
        last_launched = $null
        usage_week_hours = 0.0
        usage_5h_hours = 0.0
        activated_at = $null
        expires_at = $null
        proxy = $null
    }
    Write-JsonFile -Value $profile -Path $profilePath
    return $profilePath
}

function Stop-CodexProfileProcesses {
    param([Parameter(Mandatory = $true)][string]$ProfilePath)

    $all = Get-CimInstance Win32_Process | Where-Object { $_.Name -in @("Codex.exe", "codex.exe") }
    $ids = [System.Collections.Generic.HashSet[int]]::new()
    foreach ($process in $all) {
        if ($process.CommandLine -like "*$ProfilePath*") {
            [void]$ids.Add([int]$process.ProcessId)
        }
    }

    $changed = $true
    while ($changed) {
        $changed = $false
        foreach ($process in $all) {
            if ($ids.Contains([int]$process.ParentProcessId) -and -not $ids.Contains([int]$process.ProcessId)) {
                [void]$ids.Add([int]$process.ProcessId)
                $changed = $true
            }
        }
    }

    foreach ($processId in $ids) {
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }
    return @($ids | Sort-Object)
}

function Write-AnalysisPython {
    param([Parameter(Mandatory = $true)][string]$Path)

    $python = @'
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

run_root = Path(sys.argv[1])
profile_path = Path(sys.argv[2])
phase = sys.argv[3]

snapshot_dir = run_root / "snapshots"
analysis_dir = run_root / "analysis"
snapshot_dir.mkdir(parents=True, exist_ok=True)
analysis_dir.mkdir(parents=True, exist_ok=True)

skip_dirs = {
    "GPUCache", "GrShaderCache", "ShaderCache", "Code Cache", "DawnGraphiteCache",
    "DawnWebGPUCache", "BrowserMetrics"
}

candidate_tokens = (
    "auth", "token", "session", "cookie", "local state", "login data",
    "web data", "account web data", "indexeddb", "leveldb", "preferences",
    "secure preferences", "state_", ".codex", "device bound"
)

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()

def snapshot(root):
    rows = {}
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for name in files:
            path = Path(current) / name
            try:
                stat = path.stat()
                rel = str(path.relative_to(root))
                rows[rel] = {
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "ctime_ns": stat.st_ctime_ns,
                    "sha256": sha256_file(path),
                    "candidate": any(token in rel.lower() for token in candidate_tokens),
                }
            except Exception as exc:
                rows[str(path)] = {"error": str(exc)}
    return rows

def byte_diff(path_a, path_b, max_bytes=128 * 1024 * 1024):
    try:
        size_a = path_a.stat().st_size
        size_b = path_b.stat().st_size
        if max(size_a, size_b) > max_bytes:
            return {"skipped": "too_large", "size_a": size_a, "size_b": size_b}
        first = None
        differing = 0
        offset = 0
        with path_a.open("rb") as a, path_b.open("rb") as b:
            while True:
                ca = a.read(1024 * 1024)
                cb = b.read(1024 * 1024)
                if not ca and not cb:
                    break
                limit = min(len(ca), len(cb))
                for idx in range(limit):
                    if ca[idx] != cb[idx]:
                        if first is None:
                            first = offset + idx
                        differing += 1
                if len(ca) != len(cb):
                    if first is None:
                        first = offset + limit
                    differing += abs(len(ca) - len(cb))
                offset += max(len(ca), len(cb))
        return {"first_different_offset": first, "different_byte_count": differing, "size_a": size_a, "size_b": size_b}
    except Exception as exc:
        return {"error": str(exc)}

def compare_snapshots(label_a, label_b):
    a_path = snapshot_dir / f"{label_a}.json"
    b_path = snapshot_dir / f"{label_b}.json"
    if not a_path.exists() or not b_path.exists():
        return None
    a = json.loads(a_path.read_text(encoding="utf-8"))
    b = json.loads(b_path.read_text(encoding="utf-8"))
    out = {"from": label_a, "to": label_b, "created": [], "deleted": [], "modified": []}
    for rel in sorted(set(b) - set(a)):
        out["created"].append({"relative_path": rel, "after": b[rel]})
    for rel in sorted(set(a) - set(b)):
        out["deleted"].append({"relative_path": rel, "before": a[rel]})
    for rel in sorted(set(a) & set(b)):
        if a[rel].get("sha256") != b[rel].get("sha256") or a[rel].get("size") != b[rel].get("size"):
            diff = None
            pa = run_root / "copies" / label_a / rel
            pb = run_root / "copies" / label_b / rel
            if pa.exists() and pb.exists():
                diff = byte_diff(pa, pb)
            out["modified"].append({"relative_path": rel, "before": a[rel], "after": b[rel], "byte_diff": diff})
    (analysis_dir / f"diff-{label_a}-to-{label_b}.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return out

def copy_candidates(root, label, snap):
    base = run_root / "copies" / label
    for rel, meta in snap.items():
        if not meta.get("candidate"):
            continue
        source = root / rel
        target = base / rel
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        except Exception:
            pass

def inspect_sqlite(path):
    if not path.exists():
        return None
    tmp = Path(tempfile.gettempdir()) / ("aion_identity_" + re.sub(r"[^A-Za-z0-9]+", "_", str(path)))
    try:
        shutil.copy2(path, tmp)
        con = sqlite3.connect(str(tmp))
        cur = con.cursor()
        result = {}
        tables = [r[0] for r in cur.execute("select name from sqlite_master where type='table' order by name")]
        for table in tables:
            try:
                count = cur.execute(f'select count(*) from "{table}"').fetchone()[0]
                cols = [r[1] for r in cur.execute(f'pragma table_info("{table}")')]
                info = {"count": count, "columns": cols, "samples": []}
                interesting = [c for c in cols if any(x in c.lower() for x in ["host", "name", "origin", "url", "realm", "user", "email", "account", "token", "key", "value", "service", "gaia", "path"])]
                if count and interesting:
                    q = ", ".join([f'"{c}"' for c in interesting[:8]])
                    for row in cur.execute(f'select {q} from "{table}" limit 25'):
                        item = {}
                        for key, value in zip(interesting[:8], row):
                            if isinstance(value, bytes):
                                item[key] = f"<bytes:{len(value)}>"
                            elif key.lower() in {"encrypted_value", "encrypted_token", "password_value", "value"} and isinstance(value, str) and len(value) > 12:
                                item[key] = "<redacted>"
                            else:
                                text = str(value)
                                item[key] = text[:160] + ("..." if len(text) > 160 else "")
                        info["samples"].append(item)
                result[table] = info
            except Exception as exc:
                result[table] = {"error": str(exc)}
        con.close()
        return result
    except Exception as exc:
        return {"error": str(exc)}

def json_identity_hits(path):
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc)}
    hits = []
    def walk(value, current=""):
        if isinstance(value, dict):
            for key, child in value.items():
                next_path = f"{current}.{key}" if current else key
                if any(token in key.lower() for token in ["account", "auth", "signin", "token", "user", "profile", "email", "gaia", "session", "cookie", "identity"]):
                    if isinstance(child, (dict, list)):
                        rendered = f"{type(child).__name__}:{len(child)}"
                    elif isinstance(child, str) and len(child) > 80:
                        rendered = child[:32] + "...<redacted>"
                    else:
                        rendered = str(child)
                    hits.append({"path": next_path, "type": type(child).__name__, "value": rendered})
                walk(child, next_path)
        elif isinstance(value, list):
            for index, child in enumerate(value[:50]):
                walk(child, f"{current}[{index}]")
    walk(data)
    return hits

def string_identity_hits(path):
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
    except Exception as exc:
        return {"error": str(exc)}
    strings = re.findall(rb"[\x20-\x7e]{4,}", raw)
    hits = []
    for item in strings:
        low = item.lower()
        if any(token in low for token in [b"openai", b"chatgpt", b"auth", b"token", b"session", b"account", b"user", b"email", b"oai"]):
            text = item.decode("utf-8", "replace")
            text = re.sub(r"(?i)(token|secret|auth|session)[^\s]{0,24}[=:][^\s,;]{8,}", r"\1=<redacted>", text)
            hits.append(text[:220] + ("..." if len(text) > 220 else ""))
    return hits[:100]

def inspect_candidates(root):
    candidates = {
        "auth_json_exists": (root / "auth.json").exists(),
        "sqlite": {},
        "json": {},
        "strings": {},
    }
    for rel in [
        "Default/Network/Cookies",
        "Default/Network/Device Bound Sessions",
        "Default/Web Data",
        "Default/Account Web Data",
        "Default/Login Data",
        "Default/Login Data For Account",
    ]:
        candidates["sqlite"][rel] = inspect_sqlite(root / rel)
    for rel in ["Local State", "Default/Preferences", "Default/Secure Preferences"]:
        candidates["json"][rel] = json_identity_hits(root / rel)
    for rel in ["Local State", "Default/Preferences", "Default/Secure Preferences"]:
        candidates["strings"][rel] = string_identity_hits(root / rel)
    leveldb = root / "Default/Local Storage/leveldb"
    if leveldb.exists():
        for child in sorted(leveldb.iterdir()):
            if child.is_file() and child.suffix.lower() in {".log", ".ldb"}:
                candidates["strings"][str(child.relative_to(root))] = string_identity_hits(child)
    return candidates

snap = snapshot(profile_path)
(snapshot_dir / f"{phase}.json").write_text(json.dumps(snap, indent=2, sort_keys=True), encoding="utf-8")
copy_candidates(profile_path, phase, snap)
(analysis_dir / f"identity-candidates-{phase}.json").write_text(json.dumps(inspect_candidates(profile_path), indent=2, sort_keys=True), encoding="utf-8")

for previous in ["before-login", "after-login"]:
    if previous != phase:
        compare_snapshots(previous, phase)

summary = {
    "phase": phase,
    "profile": str(profile_path),
    "file_count": len(snap),
    "candidate_count": sum(1 for meta in snap.values() if meta.get("candidate")),
}
print(json.dumps(summary, indent=2))
'@
    Write-TextFile -Path $Path -Content $python
}

function Invoke-ForensicSnapshot {
    param(
        [Parameter(Mandatory = $true)][string]$RunRoot,
        [Parameter(Mandatory = $true)][string]$ProfilePath,
        [Parameter(Mandatory = $true)][string]$Phase
    )
    $scriptPath = Join-Path $RunRoot "analyze_identity.py"
    if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) {
        Write-AnalysisPython -Path $scriptPath
    }
    python $scriptPath $RunRoot $ProfilePath $Phase
}

function Write-RunInstructions {
    param(
        [Parameter(Mandatory = $true)][string]$RunRoot,
        [Parameter(Mandatory = $true)][string]$Text
    )
    Write-TextFile -Path (Join-Path $RunRoot "README-STEPS.txt") -Content $Text
}

function Invoke-BatScenario {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$CodexPath
    )

    $profileName = "IdentityBAT-$script:RunId"
    $profilePath = Join-Path (Join-Path $env:USERPROFILE "CodexProfiles") $profileName
    $runRoot = Join-Path $Root "bat"
    New-Item -ItemType Directory -Force -Path $profilePath | Out-Null
    New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
    $launcherPath = Join-Path $runRoot "legacy-launch.cmd"
    Write-LegacyBatLauncher -LauncherPath $launcherPath -ProfilePath $profilePath -CodexPath $CodexPath

    Write-RunInstructions -RunRoot $runRoot -Text @"
BAT identity persistence scenario.

Profile: $profilePath
Launcher: $launcherPath

1. Snapshot before-login is captured before launch.
2. Codex launches through the BAT-equivalent launcher.
3. Complete normal Desktop login.
4. Return to this PowerShell window and press Enter.
5. The script captures after-login, stops only this profile's processes, restarts it, then waits again.
6. Confirm the same account appears after restart, return here, and press Enter.
"@

    Write-Host ""
    Write-Host "BAT scenario profile: $profilePath"
    Write-Host "Capturing BAT before-login snapshot..."
    Invoke-ForensicSnapshot -RunRoot $runRoot -ProfilePath $profilePath -Phase "before-login"

    Write-Host "Launching BAT reference..."
    Start-Process -FilePath $launcherPath
    Write-Host "Complete Desktop login for the BAT profile. Press Enter here only after login is complete and the account is visible."
    [void][System.Console]::ReadLine()

    Write-Host "Capturing BAT after-login snapshot..."
    Invoke-ForensicSnapshot -RunRoot $runRoot -ProfilePath $profilePath -Phase "after-login"

    Write-Host "Stopping only BAT profile processes..."
    $stopped = Stop-CodexProfileProcesses -ProfilePath $profilePath
    Write-JsonFile -Value @{ stopped_process_ids = $stopped } -Path (Join-Path $runRoot "stopped-before-restart.json")
    Start-Sleep -Seconds 3

    Write-Host "Restarting BAT profile..."
    Start-Process -FilePath $launcherPath
    Write-Host "Confirm the same account is restored after restart. Press Enter here after the window is stable."
    [void][System.Console]::ReadLine()

    Write-Host "Capturing BAT after-restart snapshot..."
    Invoke-ForensicSnapshot -RunRoot $runRoot -ProfilePath $profilePath -Phase "after-restart"

    return [ordered]@{
        profile_name = $profileName
        profile_path = $profilePath
        run_root = $runRoot
        launcher = $launcherPath
    }
}

function Invoke-AionScenario {
    param([Parameter(Mandatory = $true)][string]$Root)

    $profileId = [guid]::NewGuid().ToString()
    $profileName = "IdentityAion-$script:RunId"
    $profilePath = Join-Path (Join-Path $env:USERPROFILE "CodexProfiles") $profileId
    $runRoot = Join-Path $Root "aion"
    New-Item -ItemType Directory -Force -Path $profilePath | Out-Null
    New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
    $documentPath = New-AionProfileDocument -ProfileId $profileId -ProfileName $profileName

    Write-RunInstructions -RunRoot $runRoot -Text @"
Aion identity persistence scenario.

Profile name in Aion UI: $profileName
Profile id/path: $profilePath
Profile document: $documentPath

1. Snapshot before-login is captured before launch.
2. Open/reload Aion.
3. Select $profileName and click Launch Profile.
4. Complete normal Desktop login.
5. Return to this PowerShell window and press Enter.
6. The script captures after-login, stops only this profile's processes, then waits for restart.
7. Launch the same Aion profile again.
8. Confirm which account appears and press Enter.
"@

    Write-Host ""
    Write-Host "Aion scenario profile name: $profileName"
    Write-Host "Aion scenario profile path: $profilePath"
    Write-Host "Capturing Aion before-login snapshot..."
    Invoke-ForensicSnapshot -RunRoot $runRoot -ProfilePath $profilePath -Phase "before-login"

    Write-Host "Open Aion now, select '$profileName', click Launch Profile, then complete Desktop login."
    Write-Host "Press Enter here only after login is complete and the account is visible."
    [void][System.Console]::ReadLine()

    Write-Host "Capturing Aion after-login snapshot..."
    Invoke-ForensicSnapshot -RunRoot $runRoot -ProfilePath $profilePath -Phase "after-login"

    Write-Host "Stopping only Aion forensic profile processes..."
    $stopped = Stop-CodexProfileProcesses -ProfilePath $profilePath
    Write-JsonFile -Value @{ stopped_process_ids = $stopped } -Path (Join-Path $runRoot "stopped-before-restart.json")
    Start-Sleep -Seconds 3

    Write-Host "Launch the same Aion profile '$profileName' again from Aion."
    Write-Host "After the window is stable, press Enter here."
    [void][System.Console]::ReadLine()

    Write-Host "Capturing Aion after-restart snapshot..."
    Invoke-ForensicSnapshot -RunRoot $runRoot -ProfilePath $profilePath -Phase "after-restart"

    return [ordered]@{
        profile_id = $profileId
        profile_name = $profileName
        profile_path = $profilePath
        profile_document = $documentPath
        run_root = $runRoot
    }
}

function Write-FinalComparison {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        $BatResult,
        $AionResult
    )
    $summary = [ordered]@{
        run_id = $script:RunId
        created_at = (Get-Date).ToString("o")
        bat = $BatResult
        aion = $AionResult
        next_analysis = "Inspect bat/analysis and aion/analysis JSON files. Candidate identity artifacts are summarized in identity-candidates-after-login.json and diff-before-login-to-after-login.json."
    }
    Write-JsonFile -Value $summary -Path (Join-Path $Root "run-summary.json")
}

$script:RunId = Get-RunId
$root = Join-Path $env:APPDATA "Aion\config\logs\identity-forensics\$script:RunId"
New-Item -ItemType Directory -Force -Path $root | Out-Null

$codexPath = Convert-ToNormalWindowsPath (Resolve-CodexExecutable -ExplicitPath $CodexExe)
Write-Host "Identity persistence forensic run: $script:RunId"
Write-Host "Output root: $root"
Write-Host "Codex executable: $codexPath"

$batResult = $null
$aionResult = $null

if ($Target -in @("bat", "both")) {
    $batResult = Invoke-BatScenario -Root $root -CodexPath $codexPath
}

if ($Target -in @("aion", "both")) {
    $aionResult = Invoke-AionScenario -Root $root
}

Write-FinalComparison -Root $root -BatResult $batResult -AionResult $aionResult

Write-Host ""
Write-Host "Forensic run complete."
Write-Host "Output root: $root"
Write-Host "No production code was modified by this script."
