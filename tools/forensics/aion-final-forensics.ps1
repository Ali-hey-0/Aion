param(
    [string]$OutputRoot = "",
    [int]$WaitSeconds = 10
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputRoot = Join-Path $env:APPDATA "Aion\config\logs\final-forensics\$stamp"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$inspectorSource = @"
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

public static class AionProcessInspector {
    const int PROCESS_QUERY_INFORMATION = 0x0400;
    const int PROCESS_QUERY_LIMITED_INFORMATION = 0x1000;
    const int PROCESS_VM_READ = 0x0010;
    const int TOKEN_QUERY = 0x0008;
    const int TokenIntegrityLevel = 25;

    [StructLayout(LayoutKind.Sequential)]
    struct PROCESS_BASIC_INFORMATION {
        public IntPtr Reserved1;
        public IntPtr PebBaseAddress;
        public IntPtr Reserved2_0;
        public IntPtr Reserved2_1;
        public IntPtr UniqueProcessId;
        public IntPtr InheritedFromUniqueProcessId;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct TOKEN_MANDATORY_LABEL {
        public IntPtr LabelSid;
        public uint Attributes;
    }

    [DllImport("kernel32.dll", SetLastError=true)]
    static extern IntPtr OpenProcess(int dwDesiredAccess, bool bInheritHandle, int dwProcessId);

    [DllImport("kernel32.dll", SetLastError=true)]
    static extern bool CloseHandle(IntPtr hObject);

    [DllImport("ntdll.dll")]
    static extern int NtQueryInformationProcess(IntPtr processHandle, int processInformationClass, ref PROCESS_BASIC_INFORMATION processInformation, int processInformationLength, out int returnLength);

    [DllImport("kernel32.dll", SetLastError=true)]
    static extern bool ReadProcessMemory(IntPtr hProcess, IntPtr lpBaseAddress, byte[] lpBuffer, int dwSize, out IntPtr lpNumberOfBytesRead);

    [DllImport("advapi32.dll", SetLastError=true)]
    static extern bool OpenProcessToken(IntPtr ProcessHandle, int DesiredAccess, out IntPtr TokenHandle);

    [DllImport("advapi32.dll", SetLastError=true)]
    static extern bool GetTokenInformation(IntPtr TokenHandle, int TokenInformationClass, IntPtr TokenInformation, int TokenInformationLength, out int ReturnLength);

    [DllImport("advapi32.dll", SetLastError=true)]
    static extern IntPtr GetSidSubAuthority(IntPtr pSid, UInt32 nSubAuthority);

    [DllImport("advapi32.dll", SetLastError=true)]
    static extern IntPtr GetSidSubAuthorityCount(IntPtr pSid);

    [DllImport("kernel32.dll", SetLastError=true)]
    static extern bool IsProcessInJob(IntPtr ProcessHandle, IntPtr JobHandle, out bool Result);

    static IntPtr Add(IntPtr pointer, int offset) {
        return new IntPtr(pointer.ToInt64() + offset);
    }

    static IntPtr ReadIntPtr(IntPtr process, IntPtr address) {
        byte[] buffer = new byte[IntPtr.Size];
        IntPtr bytesRead;
        if (!ReadProcessMemory(process, address, buffer, buffer.Length, out bytesRead) || bytesRead.ToInt32() != buffer.Length) {
            return IntPtr.Zero;
        }
        return IntPtr.Size == 8 ? new IntPtr(BitConverter.ToInt64(buffer, 0)) : new IntPtr(BitConverter.ToInt32(buffer, 0));
    }

    static ushort ReadUInt16(IntPtr process, IntPtr address) {
        byte[] buffer = new byte[2];
        IntPtr bytesRead;
        if (!ReadProcessMemory(process, address, buffer, buffer.Length, out bytesRead) || bytesRead.ToInt32() != buffer.Length) {
            return 0;
        }
        return BitConverter.ToUInt16(buffer, 0);
    }

    static string ReadUnicodeString(IntPtr process, IntPtr unicodeStringAddress) {
        ushort length = ReadUInt16(process, unicodeStringAddress);
        if (length == 0 || length > 32766) {
            return "";
        }
        IntPtr bufferAddress = ReadIntPtr(process, Add(unicodeStringAddress, IntPtr.Size == 8 ? 8 : 4));
        if (bufferAddress == IntPtr.Zero) {
            return "";
        }
        byte[] buffer = new byte[length];
        IntPtr bytesRead;
        if (!ReadProcessMemory(process, bufferAddress, buffer, buffer.Length, out bytesRead)) {
            return "";
        }
        return Encoding.Unicode.GetString(buffer, 0, Math.Min(length, bytesRead.ToInt32()));
    }

    static IntPtr GetProcessParameters(IntPtr process) {
        PROCESS_BASIC_INFORMATION pbi = new PROCESS_BASIC_INFORMATION();
        int returnLength;
        int status = NtQueryInformationProcess(process, 0, ref pbi, Marshal.SizeOf(typeof(PROCESS_BASIC_INFORMATION)), out returnLength);
        if (status != 0 || pbi.PebBaseAddress == IntPtr.Zero) {
            return IntPtr.Zero;
        }
        return ReadIntPtr(process, Add(pbi.PebBaseAddress, IntPtr.Size == 8 ? 0x20 : 0x10));
    }

    public static string[] GetEnvironment(int pid) {
        IntPtr process = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, false, pid);
        if (process == IntPtr.Zero) {
            return new string[] { "AION_ENV_READ_ERROR=OpenProcess failed" };
        }
        try {
            IntPtr processParameters = GetProcessParameters(process);
            if (processParameters == IntPtr.Zero) {
                return new string[] { "AION_ENV_READ_ERROR=ProcessParameters unavailable" };
            }
            IntPtr environment = ReadIntPtr(process, Add(processParameters, IntPtr.Size == 8 ? 0x80 : 0x48));
            if (environment == IntPtr.Zero) {
                return new string[] { "AION_ENV_READ_ERROR=Environment pointer unavailable" };
            }
            byte[] buffer = new byte[1024 * 1024];
            IntPtr bytesRead;
            if (!ReadProcessMemory(process, environment, buffer, buffer.Length, out bytesRead)) {
                return new string[] { "AION_ENV_READ_ERROR=ReadProcessMemory failed" };
            }
            int limit = bytesRead.ToInt32();
            int end = 0;
            for (int i = 0; i + 3 < limit; i += 2) {
                if (buffer[i] == 0 && buffer[i + 1] == 0 && buffer[i + 2] == 0 && buffer[i + 3] == 0) {
                    end = i;
                    break;
                }
            }
            if (end == 0) {
                end = limit;
            }
            string block = Encoding.Unicode.GetString(buffer, 0, end);
            return block.Split(new char[] {'\0'}, StringSplitOptions.RemoveEmptyEntries);
        } catch (Exception ex) {
            return new string[] { "AION_ENV_READ_ERROR=" + ex.GetType().Name + ": " + ex.Message };
        } finally {
            CloseHandle(process);
        }
    }

    public static string GetCurrentDirectory(int pid) {
        IntPtr process = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, false, pid);
        if (process == IntPtr.Zero) {
            return "";
        }
        try {
            IntPtr processParameters = GetProcessParameters(process);
            if (processParameters == IntPtr.Zero) {
                return "";
            }
            return ReadUnicodeString(process, Add(processParameters, IntPtr.Size == 8 ? 0x38 : 0x24));
        } catch {
            return "";
        } finally {
            CloseHandle(process);
        }
    }

    public static string GetIntegrityLevel(int pid) {
        IntPtr process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid);
        if (process == IntPtr.Zero) {
            return "UNKNOWN";
        }
        IntPtr token = IntPtr.Zero;
        try {
            if (!OpenProcessToken(process, TOKEN_QUERY, out token)) {
                return "UNKNOWN";
            }
            int needed;
            GetTokenInformation(token, TokenIntegrityLevel, IntPtr.Zero, 0, out needed);
            IntPtr buffer = Marshal.AllocHGlobal(needed);
            try {
                if (!GetTokenInformation(token, TokenIntegrityLevel, buffer, needed, out needed)) {
                    return "UNKNOWN";
                }
                TOKEN_MANDATORY_LABEL label = (TOKEN_MANDATORY_LABEL)Marshal.PtrToStructure(buffer, typeof(TOKEN_MANDATORY_LABEL));
                IntPtr countPtr = GetSidSubAuthorityCount(label.LabelSid);
                byte count = Marshal.ReadByte(countPtr);
                IntPtr ridPtr = GetSidSubAuthority(label.LabelSid, (uint)(count - 1));
                int rid = Marshal.ReadInt32(ridPtr);
                if (rid >= 0x4000) return "System";
                if (rid >= 0x3000) return "High";
                if (rid >= 0x2000) return "Medium";
                if (rid >= 0x1000) return "Low";
                return "Untrusted";
            } finally {
                Marshal.FreeHGlobal(buffer);
            }
        } catch {
            return "UNKNOWN";
        } finally {
            if (token != IntPtr.Zero) CloseHandle(token);
            CloseHandle(process);
        }
    }

    public static string IsInJob(int pid) {
        IntPtr process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid);
        if (process == IntPtr.Zero) {
            return "UNKNOWN";
        }
        try {
            bool result;
            if (!IsProcessInJob(process, IntPtr.Zero, out result)) {
                return "UNKNOWN";
            }
            return result ? "True" : "False";
        } catch {
            return "UNKNOWN";
        } finally {
            CloseHandle(process);
        }
    }
}
"@

Add-Type -TypeDefinition $inspectorSource -ErrorAction Stop

function Convert-ToSafeName {
    param([string]$Value)
    $safe = $Value -replace '[^A-Za-z0-9_.-]', '-'
    if ($safe.Length -gt 96) { $safe.Substring(0, 96) } else { $safe }
}

function Find-CodexExecutable {
    $repositoryKey = "Registry::HKEY_CURRENT_USER\Software\Classes\Local Settings\Software\Microsoft\Windows\CurrentVersion\AppModel\Repository\Packages"
    if (Test-Path -LiteralPath $repositoryKey) {
        $candidate = Get-ChildItem -LiteralPath $repositoryKey -ErrorAction SilentlyContinue |
            Where-Object { $_.PSChildName -like "OpenAI.Codex*" } |
            ForEach-Object {
                $props = Get-ItemProperty -LiteralPath $_.PSPath -ErrorAction SilentlyContinue
                if ($props.PackageRootFolder) {
                    Join-Path $props.PackageRootFolder "app\Codex.exe"
                }
            } |
            Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } |
            Select-Object -First 1
        if ($candidate) { return (Resolve-Path -LiteralPath $candidate).Path }
    }

    $windowsApps = Join-Path $env:ProgramFiles "WindowsApps"
    $candidate = Get-ChildItem -LiteralPath $windowsApps -Directory -Filter "OpenAI.Codex_*" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        ForEach-Object { Join-Path $_.FullName "app\Codex.exe" } |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
        Select-Object -First 1
    if ($candidate) { return (Resolve-Path -LiteralPath $candidate).Path }

    throw "Codex.exe not found."
}

function Get-TraceRoots {
    param([string[]]$ProfileRoots)

    $roots = New-Object System.Collections.Generic.List[string]
    foreach ($root in $ProfileRoots) {
        if ($root -and (Test-Path -LiteralPath $root)) { $roots.Add((Resolve-Path -LiteralPath $root).Path) }
    }
    foreach ($candidate in @((Join-Path $env:USERPROFILE ".codex"))) {
        if (Test-Path -LiteralPath $candidate) {
            $roots.Add((Resolve-Path -LiteralPath $candidate).Path)
        }
    }
    $roots | Sort-Object -Unique
}

function Get-FileSystemRecord {
    param([string]$Root, [System.IO.FileSystemInfo]$Item)

    $relative = $Item.FullName.Substring($Root.Length).TrimStart("\", "/")
    $hash = ""
    if (-not $Item.PSIsContainer) {
        try {
            if ($Item.Length -le 134217728) {
                $hash = (Get-FileHash -LiteralPath $Item.FullName -Algorithm SHA256).Hash
            } else {
                $hash = "SKIPPED_GT_128MB"
            }
        } catch {
            $hash = "HASH_ERROR:$($_.Exception.Message)"
        }
    }

    $acl = ""
    try { $acl = (Get-Acl -LiteralPath $Item.FullName).Sddl } catch { $acl = "ACL_ERROR:$($_.Exception.Message)" }

    $streams = @()
    if (-not $Item.PSIsContainer) {
        try {
            $streams = Get-Item -LiteralPath $Item.FullName -Stream * -ErrorAction SilentlyContinue |
                Select-Object Stream,Length
        } catch {
            $streams = @([pscustomobject]@{ Stream = "ADS_ERROR"; Length = 0 })
        }
    }

    [pscustomobject]@{
        root = $Root
        relative_path = $relative
        full_path = $Item.FullName
        item_type = if ($Item.PSIsContainer) { "Directory" } else { "File" }
        length = if ($Item.PSIsContainer) { 0 } else { $Item.Length }
        attributes = $Item.Attributes.ToString()
        created_utc = $Item.CreationTimeUtc.ToString("o")
        modified_utc = $Item.LastWriteTimeUtc.ToString("o")
        hash_sha256 = $hash
        acl_sddl = $acl
        link_type = $Item.LinkType
        target = @($Item.Target)
        streams = $streams
    }
}

function New-Snapshot {
    param([string]$Label, [string[]]$Roots)

    $records = New-Object System.Collections.Generic.List[object]
    foreach ($root in $Roots) {
        if (-not (Test-Path -LiteralPath $root)) { continue }
        Get-ChildItem -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue |
            ForEach-Object { $records.Add((Get-FileSystemRecord -Root $root -Item $_)) }
    }

    $path = Join-Path $OutputRoot "$((Get-Date -Format 'HHmmss'))-$(Convert-ToSafeName $Label).snapshot.json"
    [pscustomobject]@{
        label = $Label
        captured_utc = (Get-Date).ToUniversalTime().ToString("o")
        roots = $Roots
        records = $records
    } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $path -Encoding UTF8
    $path
}

function New-SnapshotDiff {
    param([string]$Before, [string]$After, [string]$Label)

    $beforeData = Get-Content -LiteralPath $Before -Raw | ConvertFrom-Json
    $afterData = Get-Content -LiteralPath $After -Raw | ConvertFrom-Json
    $beforeMap = @{}
    $afterMap = @{}
    foreach ($record in $beforeData.records) { $beforeMap[$record.full_path] = $record }
    foreach ($record in $afterData.records) { $afterMap[$record.full_path] = $record }

    $created = @()
    $deleted = @()
    $modified = @()

    foreach ($key in $afterMap.Keys) {
        if (-not $beforeMap.ContainsKey($key)) {
            $created += $afterMap[$key]
            continue
        }
        $left = $beforeMap[$key]
        $right = $afterMap[$key]
        if ($left.length -ne $right.length -or $left.modified_utc -ne $right.modified_utc -or $left.hash_sha256 -ne $right.hash_sha256 -or $left.acl_sddl -ne $right.acl_sddl) {
            $modified += [pscustomobject]@{ full_path = $key; before = $left; after = $right }
        }
    }
    foreach ($key in $beforeMap.Keys) {
        if (-not $afterMap.ContainsKey($key)) { $deleted += $beforeMap[$key] }
    }

    $path = Join-Path $OutputRoot "$((Get-Date -Format 'HHmmss'))-$(Convert-ToSafeName $Label).diff.json"
    [pscustomobject]@{
        label = $Label
        before = $Before
        after = $After
        created_count = $created.Count
        deleted_count = $deleted.Count
        modified_count = $modified.Count
        created = $created | Sort-Object full_path
        deleted = $deleted | Sort-Object full_path
        modified = $modified | Sort-Object full_path
    } | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $path -Encoding UTF8
    $path
}

function Get-DescendantProcessIds {
    param([int[]]$RootPids)
    $all = Get-CimInstance Win32_Process
    $known = New-Object System.Collections.Generic.HashSet[int]
    foreach ($pid in $RootPids) { [void]$known.Add($pid) }
    $changed = $true
    while ($changed) {
        $changed = $false
        foreach ($proc in $all) {
            if ($known.Contains([int]$proc.ParentProcessId) -and -not $known.Contains([int]$proc.ProcessId)) {
                [void]$known.Add([int]$proc.ProcessId)
                $changed = $true
            }
        }
    }
    $known.ToArray()
}

function Capture-ProcessSet {
    param([string]$Label, [string]$Marker)

    $all = Get-CimInstance Win32_Process
    $roots = $all | Where-Object {
        $_.Name -in @("Codex.exe", "codex.exe", "cmd.exe", "powershell.exe", "aion.exe") -and
        $_.CommandLine -like "*$Marker*"
    }
    $rootIds = @($roots | ForEach-Object { [int]$_.ProcessId })
    $ids = if ($rootIds.Count -gt 0) { Get-DescendantProcessIds -RootPids $rootIds } else { @() }

    $records = @()
    foreach ($proc in $all) {
        if ($ids -contains [int]$proc.ProcessId) {
            $env = @()
            $cwd = ""
            $integrity = "UNKNOWN"
            $inJob = "UNKNOWN"
            try { $env = [AionProcessInspector]::GetEnvironment([int]$proc.ProcessId) | Sort-Object } catch { $env = @("AION_ENV_READ_ERROR=$($_.Exception.Message)") }
            try { $cwd = [AionProcessInspector]::GetCurrentDirectory([int]$proc.ProcessId) } catch { $cwd = "" }
            try { $integrity = [AionProcessInspector]::GetIntegrityLevel([int]$proc.ProcessId) } catch { $integrity = "UNKNOWN" }
            try { $inJob = [AionProcessInspector]::IsInJob([int]$proc.ProcessId) } catch { $inJob = "UNKNOWN" }

            $interestingEnv = $env | Where-Object {
                $_ -match '^(USERPROFILE|HOME|HOMEDRIVE|HOMEPATH|APPDATA|LOCALAPPDATA|TEMP|TMP|CODEX_HOME|PATH|PWD)=' -or
                $_ -match '^(NODE_|ELECTRON_|VSCODE_|OPENAI_)' -or
                $_ -match '(?i)(auth|token|profile|codex|session)'
            }

            $records += [pscustomobject]@{
                name = $proc.Name
                pid = [int]$proc.ProcessId
                parent_pid = [int]$proc.ParentProcessId
                executable = $proc.ExecutablePath
                command_line = $proc.CommandLine
                creation_date = $proc.CreationDate
                current_directory = $cwd
                integrity_level = $integrity
                is_in_job = $inJob
                environment = $env
                interesting_environment = $interestingEnv
            }
        }
    }

    $path = Join-Path $OutputRoot "$((Get-Date -Format 'HHmmss'))-$(Convert-ToSafeName $Label).processes.json"
    $records | Sort-Object pid | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $path -Encoding UTF8
    $path
}

function Stop-ProcessSetForMarker {
    param([string]$Marker)
    $all = Get-CimInstance Win32_Process
    $roots = $all | Where-Object {
        $_.Name -in @("Codex.exe", "codex.exe", "cmd.exe") -and $_.CommandLine -like "*$Marker*"
    }
    $ids = if (@($roots).Count -gt 0) { Get-DescendantProcessIds -RootPids @($roots | ForEach-Object { [int]$_.ProcessId }) } else { @() }
    foreach ($id in ($ids | Sort-Object -Descending)) {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$id" -ErrorAction SilentlyContinue
        if ($proc -and ($proc.CommandLine -like "*$Marker*" -or $ids -contains [int]$proc.ParentProcessId)) {
            try { [void](Invoke-CimMethod -InputObject $proc -MethodName Terminate) } catch {}
        }
    }
}

function Write-RegistrySnapshot {
    param([string]$Label)
    $path = Join-Path $OutputRoot "$((Get-Date -Format 'HHmmss'))-$(Convert-ToSafeName $Label).registry.txt"
    foreach ($key in @(
        "HKCU:\Software\OpenAI",
        "HKCU:\Software\Classes\Local Settings\Software\Microsoft\Windows\CurrentVersion\AppModel\Repository\Packages",
        "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall",
        "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall",
        "HKCU:\Software\Microsoft\IdentityCRL",
        "HKCU:\Software\Microsoft\Windows\CurrentVersion\Authentication"
    )) {
        "===== $key =====" | Out-File -LiteralPath $path -Append -Encoding UTF8
        if (Test-Path $key) {
            Get-ChildItem -Path $key -Recurse -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match "Codex|OpenAI|Identity|Token|Auth|Session|AppModel" } |
                ForEach-Object {
                    $_.Name | Out-File -LiteralPath $path -Append -Encoding UTF8
                    try { Get-ItemProperty -LiteralPath $_.PSPath | Format-List * | Out-File -LiteralPath $path -Append -Encoding UTF8 } catch {}
                }
        } else {
            "MISSING" | Out-File -LiteralPath $path -Append -Encoding UTF8
        }
    }
    $path
}

function Write-CredentialSnapshot {
    param([string]$Label)
    $path = Join-Path $OutputRoot "$((Get-Date -Format 'HHmmss'))-$(Convert-ToSafeName $Label).credentials.txt"
    cmdkey /list | Out-File -LiteralPath $path -Encoding UTF8
    $path
}

function Start-LegacyBatPattern {
    param([string]$ProfilePath, [string]$CodexExe)
    $launcherRoot = Join-Path $env:TEMP "Aion\Launchers"
    New-Item -ItemType Directory -Force -Path $launcherRoot | Out-Null
    $scriptPath = Join-Path $launcherRoot ("legacy-bat-launch-{0}.cmd" -f (Split-Path -Leaf $ProfilePath))
    $script = "@echo off`r`nset `"CODEX_HOME=$ProfilePath`"`r`nstart `"`" `"$CodexExe`" --user-data-dir=`"%CODEX_HOME%`"`r`n"
    Set-Content -LiteralPath $scriptPath -Value $script -Encoding ASCII -NoNewline
    & cmd.exe /d /c $scriptPath
    $scriptPath
}

function Start-AionScriptPattern {
    param([string]$ProfilePath, [string]$CodexExe)
    $launcherRoot = Join-Path $env:TEMP "Aion\Launchers"
    New-Item -ItemType Directory -Force -Path $launcherRoot | Out-Null
    $scriptPath = Join-Path $launcherRoot ("aion-launch-{0}.cmd" -f (Split-Path -Leaf $ProfilePath))
    $script = "@echo off`r`nset `"CODEX_HOME=$ProfilePath`"`r`nstart `"`" `"$CodexExe`" --user-data-dir=`"%CODEX_HOME%`"`r`nexit /b 0`r`n"
    Set-Content -LiteralPath $scriptPath -Value $script -Encoding ASCII -NoNewline
    $process = Start-Process -FilePath "cmd.exe" -ArgumentList @("/d", "/c", $scriptPath) -PassThru
    [pscustomobject]@{ ScriptPath = $scriptPath; LauncherPid = $process.Id }
}

$codexExe = Find-CodexExecutable
$runStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$batProfile = Join-Path $env:USERPROFILE "CodexProfiles\AionForensicBat-$runStamp"
$aionProfile = Join-Path $env:USERPROFILE "CodexProfiles\AionForensicAion-$runStamp"
New-Item -ItemType Directory -Force -Path $batProfile | Out-Null
New-Item -ItemType Directory -Force -Path $aionProfile | Out-Null
$roots = Get-TraceRoots -ProfileRoots @($batProfile, $aionProfile)

$summary = [ordered]@{
    output_root = $OutputRoot
    codex_exe = $codexExe
    bat_profile = $batProfile
    aion_profile = $aionProfile
    wait_seconds = $WaitSeconds
    artifacts = [ordered]@{}
}

$summary.artifacts.registry_before = Write-RegistrySnapshot -Label "registry-before"
$summary.artifacts.credentials_before = Write-CredentialSnapshot -Label "credentials-before"

$batBefore = New-Snapshot -Label "bat-before" -Roots $roots
$legacyScript = Start-LegacyBatPattern -ProfilePath $batProfile -CodexExe $codexExe
Start-Sleep -Seconds $WaitSeconds
$summary.artifacts.bat_processes = Capture-ProcessSet -Label "bat-processes" -Marker (Split-Path -Leaf $batProfile)
$batAfter = New-Snapshot -Label "bat-after" -Roots $roots
$summary.artifacts.bat_before_snapshot = $batBefore
$summary.artifacts.bat_after_snapshot = $batAfter
$summary.artifacts.bat_diff = New-SnapshotDiff -Before $batBefore -After $batAfter -Label "bat-diff"
$summary.artifacts.legacy_script = $legacyScript
Stop-ProcessSetForMarker -Marker (Split-Path -Leaf $batProfile)
Start-Sleep -Seconds 2

$aionBefore = New-Snapshot -Label "aion-before" -Roots $roots
$aionLaunch = Start-AionScriptPattern -ProfilePath $aionProfile -CodexExe $codexExe
Start-Sleep -Seconds $WaitSeconds
$summary.artifacts.aion_processes = Capture-ProcessSet -Label "aion-processes" -Marker (Split-Path -Leaf $aionProfile)
$aionAfter = New-Snapshot -Label "aion-after" -Roots $roots
$summary.artifacts.aion_before_snapshot = $aionBefore
$summary.artifacts.aion_after_snapshot = $aionAfter
$summary.artifacts.aion_diff = New-SnapshotDiff -Before $aionBefore -After $aionAfter -Label "aion-diff"
$summary.artifacts.aion_script = $aionLaunch.ScriptPath
$summary.artifacts.aion_launcher_pid = $aionLaunch.LauncherPid
Stop-ProcessSetForMarker -Marker (Split-Path -Leaf $aionProfile)
Start-Sleep -Seconds 2

$summary.artifacts.registry_after = Write-RegistrySnapshot -Label "registry-after"
$summary.artifacts.credentials_after = Write-CredentialSnapshot -Label "credentials-after"

$summaryPath = Join-Path $OutputRoot "summary.json"
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
Write-Output $summaryPath
