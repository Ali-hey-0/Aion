# Aion Timeline Diff

## Evidence Sources

- `G:\others\Aion-Codex\forensics\execution-env-20260629-143228`
- Current source inspection:
  - `G:\others\Aion-Codex\src-tauri\src\utils.rs`
  - `G:\others\Aion-Codex\src-tauri\src\commands.rs`
- Current profile artifact scan:
  - `C:\Users\WebVajhegan\CodexProfiles`

## Current Aion Launch Timeline

From the current implementation:

1. Aion validates the profile sandbox path.
2. Aion optionally removes stale `aion-launch.cmd` from the profile root.
3. Aion writes a launcher script outside the profile under `%TEMP%\Aion\Launchers`.
4. Aion starts `cmd.exe /d /c <launcher.cmd>`.
5. Aion drops the `cmd.exe` handle.
6. Aion polls for a primary `Codex.exe` matching `--user-data-dir=<profile>`.
7. As soon as that primary `Codex.exe` is detected, `launch_codex` returns a PID.
8. `commands.rs` marks the runtime state as `Running`.

## Successful Profile Artifact Pattern

Successful profiles contain both Chromium and backend artifacts, for example:

```text
Local State
Default\
Crashpad\
config.toml
state_*.sqlite
logs_*.sqlite
memories_*.sqlite
.codex-global-state.json
```

## Failed Profile Artifact Pattern

Failed profiles observed in `C:\Users\WebVajhegan\CodexProfiles` include:

```text
999366ca-5d46-4bee-bb95-ae838b9ed91e
76bb17c1-b380-4e59-a50f-0b7ed834bad2
```

They have:

```text
HasChromium = true
HasBackend = false
```

That means:

1. The primary Electron/Chromium process started far enough to create profile bootstrap files.
2. The Codex backend did not initialize far enough to create `config.toml`, SQLite state, logs, memories, or global state.

## Earliest Confirmed Timeline Divergence

The earliest confirmed divergence is after primary `Codex.exe` creation and before app-server/backend initialization.

In successful launches:

```text
primary Codex.exe appears
  -> resources\codex.exe app-server appears
  -> backend artifacts are created
  -> profile becomes meaningfully running
```

In failed launches:

```text
primary Codex.exe appears
  -> Chromium artifacts are created
  -> app-server/backend artifacts do not appear
  -> Aion may still mark the profile Running
```

## Conclusion

The first reliable divergence is not authentication. It is backend startup readiness. Aion currently treats primary Electron process detection as launch success, but the observed failure mode occurs after that point and before backend readiness.

Confidence: high.
