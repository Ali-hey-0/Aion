# Aion Codex Isolation Forensic Report

Date: 2026-06-28

## Scope

This report documents the remaining profile isolation inconsistency in Aion, a Tauri/Rust launcher for OpenAI Codex Desktop profiles.

No production workaround is proposed here. The objective is evidence: identify what Codex Desktop persists, where it persists it, and whether Aion differs from the legacy batch launch.

## Important Safety Change For This Investigation

The previous Aion auth-hiding guard was disabled by default for forensic correctness. It now only activates when `AION_ENABLE_AUTH_GUARD` is explicitly set. During this investigation it was not enabled.

This prevents Aion from renaming, hiding, or modifying `%USERPROFILE%\.codex\auth.json` while evidence is collected.

## Evidence Artifacts

Diagnostic artifacts were written under:

```text
%APPDATA%\Aion\config\logs\
```

Important files/directories:

```text
%APPDATA%\Aion\config\logs\launch-diagnostics\
%APPDATA%\Aion\config\logs\forensics\
%APPDATA%\Aion\config\logs\forensics\aion-forensic-launch-output.txt
```

Test profile roots:

```text
%USERPROFILE%\CodexProfiles\ForensicBat-20260628-223234
%USERPROFILE%\CodexProfiles\a90c24e3-db10-4839-ab3c-965bdf88ec93
```

## Launch Comparison

### Legacy BAT Pattern

The known working batch pattern is:

```bat
set "CODEX_HOME=C:\Users\WebVajhegan\CodexProfiles\<profile>"
start "" "C:\Program Files\WindowsApps\OpenAI.Codex_26.623.5546.0_x64__2p2nqsd0c76g0\app\Codex.exe" --user-data-dir="%CODEX_HOME%"
```

### Aion Launch Script

Aion generated:

```bat
@echo off
set "CODEX_HOME=C:\Users\WebVajhegan\CodexProfiles\a90c24e3-db10-4839-ab3c-965bdf88ec93"
start "" "C:\Program Files\WindowsApps\OpenAI.Codex_26.623.5546.0_x64__2p2nqsd0c76g0\app\Codex.exe" --user-data-dir="%CODEX_HOME%"
exit /b 0
```

The only material difference is `exit /b 0` after `start`, which runs after process creation and is not a profile isolation input.

## Process Model Evidence

Both BAT and Aion-code launch created independent primary `Codex.exe` processes with the requested `--user-data-dir`.

### BAT Test

Profile:

```text
C:\Users\WebVajhegan\CodexProfiles\ForensicBat-20260628-223234
```

Observed process tree:

```text
Codex.exe PID 19936
  --user-data-dir="C:\Users\WebVajhegan\CodexProfiles\ForensicBat-20260628-223234"

resources\codex.exe PID 18096
  ParentProcessId: 19936
  CommandLine: app-server --analytics-default-enabled
```

### Aion-Code Test

Profile:

```text
C:\Users\WebVajhegan\CodexProfiles\a90c24e3-db10-4839-ab3c-965bdf88ec93
```

Observed process tree:

```text
Codex.exe PID 10520
  --user-data-dir="C:\Users\WebVajhegan\CodexProfiles\a90c24e3-db10-4839-ab3c-965bdf88ec93"

resources\codex.exe PID 12352
  ParentProcessId: 10520
  CommandLine: app-server --analytics-default-enabled
```

### Conclusion

The current Aion launch path does not lose `--user-data-dir`.

The current Aion launch path does not attach to only the global app-server.

The current Aion launch path creates a per-profile `resources\codex.exe app-server`, same as the BAT pattern.

## Environment Evidence

Aion diagnostic report captured environment variables for both `Codex.exe` and `resources\codex.exe app-server`.

For Aion profile `a90c24e3-db10-4839-ab3c-965bdf88ec93`:

```text
CODEX_HOME=C:\Users\WebVajhegan\CodexProfiles\a90c24e3-db10-4839-ab3c-965bdf88ec93
USERPROFILE=C:\Users\WebVajhegan
APPDATA=C:\Users\WebVajhegan\AppData\Roaming
LOCALAPPDATA=C:\Users\WebVajhegan\AppData\Local
TEMP=C:\Users\WEBVAJ~1\AppData\Local\Temp
TMP=C:\Users\WEBVAJ~1\AppData\Local\Temp
```

This proves `CODEX_HOME` reaches both the Electron host and the backend app-server.

## Filesystem Persistence Model

Codex Desktop uses two distinct persistence layers:

### 1. Chromium/Electron UI State

Controlled by:

```text
--user-data-dir=<profile-root>
```

This creates:

```text
Local State
Default\Preferences
Crashpad\
BrowserMetrics\
component_crx_cache\
GPU/Shader caches
```

### 2. Codex Backend/App-Server State

Controlled by:

```text
CODEX_HOME=<profile-root>
```

This creates:

```text
.codex-global-state.json
.codex-global-state.json.bak
config.toml
goals_1.sqlite
logs_2.sqlite
memories_1.sqlite
state_5.sqlite
installation_id
skills\
plugins\
vendor_imports\
tmp\
```

### 3. Authentication Identity

Controlled by:

```text
auth.json
```

Known structure without printing secrets:

```text
auth_mode
OPENAI_API_KEY
tokens.access_token
tokens.account_id
tokens.id_token
tokens.refresh_token
last_refresh
```

No relevant Codex/OpenAI entries were observed in Windows Credential Manager.

Registry inspection showed package metadata, not account/session credentials.

## Auth File Evidence

Observed key files:

```text
%USERPROFILE%\.codex\auth.json                  EXISTS
%USERPROFILE%\CodexProfiles\Acc1\auth.json      EXISTS
%USERPROFILE%\CodexProfiles\Acc2\auth.json      MISSING
%USERPROFILE%\CodexProfiles\Acc3\auth.json      MISSING
%USERPROFILE%\CodexProfiles\ForensicBat-...\auth.json  MISSING
%USERPROFILE%\CodexProfiles\a90c24e3-...\auth.json      MISSING
```

Fresh BAT and fresh Aion-code launches both created backend state but did not create `auth.json` without an actual successful login.

## Account Fingerprint Evidence

No token values were printed.

The account identifier was hashed for comparison.

```text
global %USERPROFILE%\.codex\auth.json
  account_id_sha256_12 = 7b939a6462b2

Acc1 %USERPROFILE%\CodexProfiles\Acc1\auth.json
  account_id_sha256_12 = 7b939a6462b2
```

This proves the observed Acc1 local auth belongs to the same account identity as the global auth file. Acc1 may look like an isolated profile operationally, but it is not evidence of a separate account.

## Rejected Hypotheses

### Hypothesis: Aion drops `--user-data-dir`

Rejected.

Process command lines show the exact per-profile `--user-data-dir` for Aion-launched primary and child Chromium processes.

### Hypothesis: Aion fails to set `CODEX_HOME`

Rejected.

Environment diagnostics show `CODEX_HOME` present in both `Codex.exe` and `resources\codex.exe app-server`.

### Hypothesis: Aion reuses only the global app-server

Rejected for the tested launch.

Aion-code launch created `resources\codex.exe app-server` with `ParentProcessId` equal to the per-profile primary `Codex.exe`.

### Hypothesis: Windows Credential Manager is the primary Codex auth store

Not supported by observed evidence.

`cmdkey /list` did not show relevant Codex/OpenAI credential targets.

### Hypothesis: Registry is the primary Codex auth store

Not supported by observed evidence.

Registry hits were AppModel/package registration metadata, not account tokens.

## Most Likely Root Cause

The remaining inconsistency is not caused by the launcher.

The persistence model is:

```text
--user-data-dir  -> Chromium/Electron profile
CODEX_HOME       -> Codex backend state
auth.json        -> Codex account identity
```

When a profile-local `auth.json` exists, that profile has a persisted auth identity.

When profile-local `auth.json` is missing, Codex can still appear logged in because a global auth exists at:

```text
%USERPROFILE%\.codex\auth.json
```

This fallback creates the observed non-determinism:

1. Fresh profile creates Chromium and backend state successfully.
2. Fresh profile has no local `auth.json`.
3. Global auth exists.
4. UI can show the global account instead of forcing a new login.
5. After login attempts, if Codex does not write a local `auth.json`, the profile will not retain an independent account.

## Confidence

High: 90%.

The only missing hard proof is a ProcMon trace of the first successful `ReadFile` against `%USERPROFILE%\.codex\auth.json` from `resources\codex.exe app-server`.

All other observable differences have been tested and rejected as root cause.

## Required Permanent Fix Direction

A production fix must treat profile authentication state as a first-class lifecycle requirement.

A profile is not fully isolated until:

```text
%USERPROFILE%\CodexProfiles\<profile>\auth.json
```

exists and belongs to the intended account.

Possible permanent approaches must be validated separately:

1. Use Codex's own login flow in a context where it writes `auth.json` to `CODEX_HOME`.
2. If Codex always falls back to `%USERPROFILE%\.codex\auth.json` when local auth is missing, first-login bootstrapping must run without the global auth being visible to that Codex process.
3. If Codex does not support disabling this fallback per process, true deterministic first-login isolation may require launching under separate Windows user profiles or another officially supported Codex auth-home override.

Do not claim a profile is account-isolated merely because `--user-data-dir` and `CODEX_HOME` are set. Those isolate UI/backend state, not account identity unless local `auth.json` exists.
