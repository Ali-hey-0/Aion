param(
  [Parameter(Mandatory = $true)]
  [ValidateSet('snapshot', 'diff', 'processes', 'credentials', 'registry')]
  [string]$Mode,

  [string]$Label = '',
  [string]$Before = '',
  [string]$After = '',
  [string]$OutputRoot = ''
)

$ErrorActionPreference = 'Stop'

function Get-DefaultOutputRoot {
  Join-Path $env:APPDATA 'Aion\config\logs\forensics'
}

function New-OutputDirectory {
  param([string]$Root)
  if ([string]::IsNullOrWhiteSpace($Root)) {
    $Root = Get-DefaultOutputRoot
  }
  New-Item -ItemType Directory -Force -Path $Root | Out-Null
  $Root
}

function Convert-ToSafeName {
  param([string]$Value)
  if ([string]::IsNullOrWhiteSpace($Value)) {
    return 'snapshot'
  }
  $safe = $Value -replace '[^A-Za-z0-9_.-]', '-'
  if ($safe.Length -gt 80) {
    return $safe.Substring(0, 80)
  }
  $safe
}

function Get-CodexRoots {
  $roots = @()

  $candidateRoots = @(
    (Join-Path $env:USERPROFILE '.codex'),
    (Join-Path $env:USERPROFILE 'CodexProfiles'),
    (Join-Path $env:APPDATA 'Codex'),
    (Join-Path $env:APPDATA 'Aion'),
    (Join-Path $env:LOCALAPPDATA 'Codex'),
    (Join-Path $env:LOCALAPPDATA 'OpenAI')
  )

  foreach ($candidate in $candidateRoots) {
    if (Test-Path -LiteralPath $candidate) {
      $roots += (Resolve-Path -LiteralPath $candidate).Path
    }
  }

  $packagesRoot = Join-Path $env:LOCALAPPDATA 'Packages'
  if (Test-Path -LiteralPath $packagesRoot) {
    $packageRoots = Get-ChildItem -LiteralPath $packagesRoot -Directory -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -like 'OpenAI.Codex*' } |
      ForEach-Object { $_.FullName }
    $roots += $packageRoots
  }

  $roots | Sort-Object -Unique
}

function Get-FileRecord {
  param(
    [string]$Root,
    [System.IO.FileInfo]$File
  )

  $relative = $File.FullName.Substring($Root.Length).TrimStart('\', '/')
  $hash = ''
  try {
    if ($File.Length -le 268435456) {
      $hash = (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash
    } else {
      $hash = 'SKIPPED_GT_256MB'
    }
  } catch {
    $hash = "HASH_ERROR:$($_.Exception.Message)"
  }

  [PSCustomObject]@{
    root = $Root
    relative_path = $relative
    full_path = $File.FullName
    length = $File.Length
    created_utc = $File.CreationTimeUtc.ToString('o')
    modified_utc = $File.LastWriteTimeUtc.ToString('o')
    hash_sha256 = $hash
  }
}

function New-CodexSnapshot {
  param(
    [string]$SnapshotLabel,
    [string]$Root
  )

  $output = New-OutputDirectory $Root
  $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
  $safeLabel = Convert-ToSafeName $SnapshotLabel
  $path = Join-Path $output "$stamp-$safeLabel.snapshot.json"
  $roots = Get-CodexRoots
  $records = New-Object System.Collections.Generic.List[object]

  foreach ($rootPath in $roots) {
    Get-ChildItem -LiteralPath $rootPath -Recurse -Force -File -ErrorAction SilentlyContinue |
      ForEach-Object { $records.Add((Get-FileRecord -Root $rootPath -File $_)) }
  }

  $snapshot = [PSCustomObject]@{
    schema_version = 1
    label = $SnapshotLabel
    captured_utc = (Get-Date).ToUniversalTime().ToString('o')
    machine = $env:COMPUTERNAME
    user = $env:USERNAME
    roots = $roots
    files = $records
  }

  $snapshot | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $path -Encoding UTF8
  Write-Output $path
}

function New-CodexDiff {
  param(
    [string]$BeforePath,
    [string]$AfterPath,
    [string]$Root
  )

  $output = New-OutputDirectory $Root
  $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
  $path = Join-Path $output "$stamp-diff.json"
  $beforeData = Get-Content -LiteralPath $BeforePath -Raw | ConvertFrom-Json
  $afterData = Get-Content -LiteralPath $AfterPath -Raw | ConvertFrom-Json
  $beforeMap = @{}
  $afterMap = @{}

  foreach ($item in $beforeData.files) {
    $beforeMap[$item.full_path] = $item
  }
  foreach ($item in $afterData.files) {
    $afterMap[$item.full_path] = $item
  }

  $created = @()
  $deleted = @()
  $modified = @()

  foreach ($key in $afterMap.Keys) {
    if (-not $beforeMap.ContainsKey($key)) {
      $created += $afterMap[$key]
      continue
    }
    $beforeItem = $beforeMap[$key]
    $afterItem = $afterMap[$key]
    if (
      $beforeItem.length -ne $afterItem.length -or
      $beforeItem.modified_utc -ne $afterItem.modified_utc -or
      $beforeItem.hash_sha256 -ne $afterItem.hash_sha256
    ) {
      $modified += [PSCustomObject]@{
        full_path = $key
        before = $beforeItem
        after = $afterItem
      }
    }
  }

  foreach ($key in $beforeMap.Keys) {
    if (-not $afterMap.ContainsKey($key)) {
      $deleted += $beforeMap[$key]
    }
  }

  $diff = [PSCustomObject]@{
    schema_version = 1
    before = $BeforePath
    after = $AfterPath
    generated_utc = (Get-Date).ToUniversalTime().ToString('o')
    created_count = $created.Count
    deleted_count = $deleted.Count
    modified_count = $modified.Count
    created = $created | Sort-Object full_path
    deleted = $deleted | Sort-Object full_path
    modified = $modified | Sort-Object full_path
  }

  $diff | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $path -Encoding UTF8
  Write-Output $path
}

function Write-CodexProcesses {
  param([string]$Root, [string]$SnapshotLabel)
  $output = New-OutputDirectory $Root
  $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
  $safeLabel = Convert-ToSafeName $SnapshotLabel
  $path = Join-Path $output "$stamp-$safeLabel.processes.json"
  $processes = Get-CimInstance Win32_Process |
    Where-Object { $_.Name -in @('Codex.exe', 'codex.exe', 'aion.exe', 'node.exe', 'cargo.exe', 'cmd.exe', 'powershell.exe') } |
    Select-Object Name,ProcessId,ParentProcessId,ExecutablePath,CommandLine,CreationDate
  $processes | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $path -Encoding UTF8
  Write-Output $path
}

function Write-CredentialTargets {
  param([string]$Root, [string]$SnapshotLabel)
  $output = New-OutputDirectory $Root
  $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
  $safeLabel = Convert-ToSafeName $SnapshotLabel
  $path = Join-Path $output "$stamp-$safeLabel.credentials.txt"
  cmdkey /list | Out-File -LiteralPath $path -Encoding UTF8
  Write-Output $path
}

function Write-CodexRegistry {
  param([string]$Root, [string]$SnapshotLabel)
  $output = New-OutputDirectory $Root
  $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
  $safeLabel = Convert-ToSafeName $SnapshotLabel
  $path = Join-Path $output "$stamp-$safeLabel.registry.txt"
  $keys = @(
    'HKCU:\Software\OpenAI',
    'HKCU:\Software\Classes\Local Settings\Software\Microsoft\Windows\CurrentVersion\AppModel\Repository\Packages',
    'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall',
    'HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall'
  )
  foreach ($key in $keys) {
    "===== $key =====" | Out-File -LiteralPath $path -Append -Encoding UTF8
    if (Test-Path $key) {
      Get-ChildItem -Path $key -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match 'Codex|OpenAI' } |
        Format-List * |
        Out-File -LiteralPath $path -Append -Encoding UTF8
    } else {
      "MISSING" | Out-File -LiteralPath $path -Append -Encoding UTF8
    }
  }
  Write-Output $path
}

switch ($Mode) {
  'snapshot' { New-CodexSnapshot -SnapshotLabel $Label -Root $OutputRoot }
  'diff' { New-CodexDiff -BeforePath $Before -AfterPath $After -Root $OutputRoot }
  'processes' { Write-CodexProcesses -Root $OutputRoot -SnapshotLabel $Label }
  'credentials' { Write-CredentialTargets -Root $OutputRoot -SnapshotLabel $Label }
  'registry' { Write-CodexRegistry -Root $OutputRoot -SnapshotLabel $Label }
}
