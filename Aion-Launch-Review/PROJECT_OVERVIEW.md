# Aion Launch Subsystem Overview

This package contains only the files needed to review how Aion launches OpenAI Codex Desktop profiles.

## Included Source Files

- `src-tauri/src/main.rs`: registers Tauri managed state and IPC commands.
- `src-tauri/src/commands.rs`: exposes `launch_profile` and related runtime commands.
- `src-tauri/src/profile.rs`: owns persistent profile storage, sandbox path generation, runtime state, and profile CRUD.
- `src-tauri/src/utils.rs`: owns Codex executable discovery, process launch, process matching, focus, and termination logic.
- `ac-codex.bat`: legacy BAT reference launcher that successfully launches isolated Codex profiles.
- `package.json`, `src-tauri/Cargo.toml`, `src-tauri/tauri.conf.json`: build/runtime configuration relevant to launching the Tauri app.

## High-Level Architecture

Aion uses Tauri managed state:

- `ProfileManager`: persistent profile/config manager.
- `RuntimeManager`: in-memory runtime state manager.
- `CodexProcessProvider`: OS integration layer for locating and launching Codex.

The frontend calls Tauri IPC commands. The launch path is:

1. Frontend invokes `launch_profile(profile_id)`.
2. `commands.rs` validates that the profile exists.
3. `ProfileManager` resolves and ensures the profile user-data directory.
4. `CodexProcessProvider` resolves the Codex executable path.
5. `RuntimeManager` marks the profile as `Launching`.
6. `CodexProcessProvider::launch_codex` launches Codex through the Windows-specific launcher path.
7. Aion waits until a primary `Codex.exe` process appears with the expected `--user-data-dir`.
8. `RuntimeManager` marks the profile as `Running` with the detected PID.
9. `ProfileManager` updates the profile last-launch metadata.

## Profile Creation

Profiles are persisted under:

```text
%APPDATA%\Aion\config\profiles\<UUID>.json
```

The profile sandbox/user-data directory is derived from the profile UUID.

On Windows, profile sandboxes are rooted at:

```text
%USERPROFILE%\CodexProfiles\<UUID>
```

The path is produced by `ProfileManager::profile_user_data_dir`, which joins the configured sandbox root with the profile UUID.

## Launch Inputs

For each profile, Aion launches Codex with:

```text
CODEX_HOME=<profile sandbox path>
--user-data-dir=<profile sandbox path>
```

The same directory is used for both values.

## CODEX_HOME Assignment

In `src-tauri/src/utils.rs`, the Windows launch path writes a temporary command script with:

```bat
set "CODEX_HOME=<profile sandbox path>"
start "" "<Codex.exe>" --user-data-dir="%CODEX_HOME%"
```

This mirrors the legacy `ac-codex.bat` launcher.

## --user-data-dir Assignment

The generated launch script passes:

```bat
--user-data-dir="%CODEX_HOME%"
```

At runtime, observed primary and Chromium child `Codex.exe` command lines contain the profile-specific `--user-data-dir`.

## Where cmd.exe Is Spawned

In `src-tauri/src/utils.rs`, `launch_isolated_codex_process` creates a temporary launcher script under:

```text
%TEMP%\Aion\Launchers\aion-launch-<profile>.cmd
```

Then it spawns:

```text
cmd.exe /d /c <launcher.cmd>
```

using Rust `std::process::Command`.

## Where Codex.exe Is Started

`Codex.exe` is not spawned directly by Rust in the Windows launch path. It is started by the generated command script via:

```bat
start "" "<Codex.exe>" --user-data-dir="%CODEX_HOME%"
```

The executable path is resolved by `CodexProcessProvider::resolve_codex_executable`, using custom config, registry/package discovery, and fallback scanning.

## Runtime Flow

Runtime state is stored in memory by `RuntimeManager`:

- `Idle`
- `Launching`
- `Running`
- `Stopping`
- `Exited`
- `Error(String)`

The launch command marks a profile as `Launching`, calls the process provider, then marks it `Running` after a primary `Codex.exe` process is detected with the profile user-data path.

## Expected Process Tree

For both the BAT reference and Aion's current Windows launch path, the relevant process tree is:

```text
Aion.exe
  cmd.exe /d /c %TEMP%\Aion\Launchers\aion-launch-<profile>.cmd
    Codex.exe --user-data-dir="<profile path>"
      Codex.exe --type=crashpad-handler ...
      Codex.exe --type=gpu-process ...
      Codex.exe --type=utility ...
      codex.exe app-server --analytics-default-enabled
```

The `codex.exe app-server` process is the backend process located under:

```text
<OpenAI.Codex package>\app\resources\codex.exe
```

## Legacy BAT Reference

The included `ac-codex.bat` performs the same core launch operation:

```bat
set "CODEX_HOME=%USERPROFILE%\CodexProfiles\%ACC_NAME%"
start "" "%APP_DIR%" --user-data-dir="%CODEX_HOME%"
```

## Relevant Configuration

Tauri config:

- `src-tauri/tauri.conf.json`
- dev frontend: `http://127.0.0.1:7391`
- bundle targets: `msi`, `nsis`

Rust dependencies relevant to launch:

- `sysinfo`: process scanning/matching.
- `windows-sys`: Windows registry, filesystem, environment, threading, and window APIs.
- `tauri`: IPC and app runtime.

