# Aion Process Diff

Collected raw evidence: `G:\others\Aion-Codex\forensics\execution-env-20260629-143228`

## Current Process Snapshot

At capture time:

| Process class | Count / status |
| --- | --- |
| `aion.exe` | `0` |
| `cargo.exe` | `0` |
| Aion profile `Codex.exe` with `CodexProfiles` in command line | `0` |
| Global `Codex.exe` | running |
| Global `resources\codex.exe app-server` | running |

Observed global Codex root process:

```text
Codex.exe PID 7940
Parent PID 11224
Command line:
"C:\Program Files\WindowsApps\OpenAI.Codex_26.623.8305.0_x64__2p2nqsd0c76g0\app\Codex.exe"
```

Observed global app-server:

```text
codex.exe PID 13392
Parent PID 7940
Command line:
"C:\Program Files\WindowsApps\OpenAI.Codex_26.623.8305.0_x64__2p2nqsd0c76g0\app\resources\codex.exe" app-server --analytics-default-enabled
```

Observed Codex Agent execution chain:

```text
OpenAI Codex desktop
  -> resources\codex.exe app-server
    -> shell / sandbox tooling
      -> PowerShell diagnostics
```

## Important Observation

Because no manual `npm run tauri dev` process was alive, a complete live parent-chain comparison between Agent-launched Aion and user-launched Aion could not be captured in this run.

## Current Launcher Behavior From Source

The Windows launcher path in `G:\others\Aion-Codex\src-tauri\src\utils.rs`:

1. Creates a temporary launcher script under `%TEMP%\Aion\Launchers`.
2. Writes:

```bat
@echo off
set "CODEX_HOME=<profile>"
start "" "<Codex.exe>" --user-data-dir="%CODEX_HOME%"
exit /b 0
```

3. Starts `cmd.exe /d /c <launcher.cmd>`.
4. Drops the `cmd.exe` child handle immediately.
5. Waits only for a primary `Codex.exe` whose command line contains the target `--user-data-dir`.

The code does not wait for:

- `resources\codex.exe app-server`
- `config.toml`
- `state_*.sqlite`
- `logs_*.sqlite`
- `memories_*.sqlite`
- `.codex-global-state.json`

## Proven Process-Level Gap

The UI/backend can mark a profile as running once the primary Electron `Codex.exe` appears, even if the backend process never starts.

This is consistent with observed failed profiles that contain Chromium artifacts but no backend artifacts.

Confidence: high.
