# Debugging Timeline Summary

This timeline summarizes the launch-subsystem investigation in chronological order. It intentionally excludes bulky forensic logs.

## 1. Legacy BAT Reference Identified

The original working proof-of-concept launcher is `ac-codex.bat`.

Core behavior:

```bat
set "CODEX_HOME=%USERPROFILE%\CodexProfiles\%ACC_NAME%"
start "" "%APP_DIR%" --user-data-dir="%CODEX_HOME%"
```

## 2. Initial Rust Backend Implemented

Aion added:

- profile persistence
- profile sandbox directories
- registry/package discovery for `Codex.exe`
- process scanning with `sysinfo`
- Tauri command `launch_profile`

## 3. Sandbox Root Moved Outside AppData

The Windows sandbox root was moved to:

```text
%USERPROFILE%\CodexProfiles\<UUID>
```

This avoids placing profile data under Aion's AppData config tree.

## 4. Launch Strategy Changed To BAT-Style Script

Aion's Windows launch path was changed to generate a temporary `.cmd` launcher that sets `CODEX_HOME` and calls `start ""`.

Current generated payload:

```bat
@echo off
set "CODEX_HOME=<profile path>"
start "" "<Codex.exe>" --user-data-dir="%CODEX_HOME%"
exit /b 0
```

## 5. Launcher Script Moved Outside Profile Directory

Earlier Aion builds wrote `aion-launch.cmd` inside `CODEX_HOME`.

That file was removed from the profile directory. Current launch scripts are written under:

```text
%TEMP%\Aion\Launchers\
```

## 6. Command-Line Parity Verified

Process command lines showed primary `Codex.exe` processes receiving:

```text
--user-data-dir="<profile path>"
```

Chromium child processes also showed the profile-specific user-data path.

## 7. CODEX_HOME Parity Verified

The generated launcher sets:

```text
CODEX_HOME=<profile path>
```

This matches the BAT reference's intended environment input.

## 8. Runtime Backend Delay Observed

Successful launches do not create backend artifacts immediately.

Observed successful flow:

```text
primary Codex.exe appears
Chromium artifacts appear
10-20 seconds pass
resources\codex.exe app-server appears
backend artifacts appear
```

## 9. Chromium-Only Launches Observed

Some historical profile directories contain Chromium artifacts but no backend artifacts:

```text
config.toml
.codex-global-state.json
state_*.sqlite
logs_*.sqlite
memories_*.sqlite
```

These directories show that some launches reached Chromium bootstrap but not backend initialization.

## 10. Manual Controlled Launch Succeeded

A controlled manual `npm run tauri dev` launch was captured where:

- primary `Codex.exe` appeared with the expected `--user-data-dir`
- `resources\codex.exe app-server` appeared as a child of that primary process
- backend artifacts appeared immediately after app-server startup

## 11. Start Menu / Explorer Path Compared

Windows reports registered Codex app IDs:

```text
OpenAI.Codex_2p2nqsd0c76g0!App
com.openai.codex
```

The package manifest declares:

```text
Application Id="App"
Executable="app/Codex.exe"
EntryPoint="Windows.FullTrustApplication"
```

The BAT/Aion launch path executes `app\Codex.exe` directly through `cmd.exe start`, not through Start Menu app activation.

## 12. Current Review Question

The remaining engineering review target is whether Aion's launch pipeline is materially equivalent to the BAT launch pipeline at the Windows process-launch level, and whether post-launch process detection should treat primary `Codex.exe` detection as sufficient.

