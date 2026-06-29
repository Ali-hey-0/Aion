# Codex Desktop Path Resolution

This document separates the different path systems used by Codex Desktop.

Investigation target:

`C:\Program Files\WindowsApps\OpenAI.Codex_26.623.8305.0_x64__2p2nqsd0c76g0\app\resources\app.asar`

## Path Classes

Codex Desktop uses at least three distinct path classes:

1. Chromium profile path.
2. Electron `userData` path.
3. Backend `CODEX_HOME` path.

These are not interchangeable.

## Chromium Profile Path

Input:

`--user-data-dir=<path>`

Ownership:

Chromium/Electron browser process.

Observed artifacts:

- `Local State`
- `Default\`
- `Crashpad\`
- Chromium caches and component data

Important evidence:

`bootstrap.js` does not read `--user-data-dir` when deciding Electron `userData` or the single-instance lock.

Consequence:

Passing a unique `--user-data-dir` isolates Chromium storage, but it does not by itself isolate Electron's single-instance namespace.

## Electron `userData` Path

Source:

`app.asar!.vite/build/bootstrap.js:1 @ offset 2491`

Function:

`C({ appDataPath, buildFlavor, env })`

Resolution:

1. If `process.env.CODEX_ELECTRON_USER_DATA_PATH` is set and non-empty, use `path.resolve(...)`.
2. Otherwise use `path.join(appDataPath, brand/build-flavor directory)`.
3. For agent flavor, `CODEX_ELECTRON_AGENT_RUN_ID` adds an agent subdirectory.

Applied at:

`app.asar!.vite/build/bootstrap.js:1 @ offset 13377`

Bootstrap calls `app.setPath("userData", resolvedPath)`.

Consumers:

- Electron single-instance lock.
- Browser persistence directory in `main-B6QfY4LN.js @ offset 1089996`.
- Extension lookup path in `main-B6QfY4LN.js @ offset 163761`.
- macOS installer state file path in bootstrap, not relevant on Windows.

Critical finding:

Electron `userData` is set before `requestSingleInstanceLock()`.

## Backend `CODEX_HOME`

Source:

`app.asar!.vite/build/src-CoIhwwHr.js:1 @ offset 332943`

Function:

`LT({ preferWsl = false })`

Resolution:

1. If `process.env.CODEX_HOME` exists, use it.
2. Otherwise fall back to `path.join(os.homedir(), ".codex")`.

Consumers:

- `qV({ moduleDir })` in `workspace-root-drop-handler-DeLi4ACJ.js @ offset 4305348`.
- `.codex-global-state.json` under `codexHome`.
- `config.toml` under `codexHome`.
- app-server environment and backend storage.

Supporting evidence:

`main-B6QfY4LN.js @ offset 370402` contains remote/control path templates based on `${CODEX_HOME:-$HOME/.codex}/app-server-control`.

## App-Server Executable Path

Source:

`app.asar!.vite/build/src-CoIhwwHr.js:1`

Functions:

- `rM(resourcesPath)` at offset `456251`.
- `zR(...)` at offset `578629`.
- `FR(...)` at offset `576703`.

Resolution:

- Locate bundled `codex.exe` in the Electron resources path.
- On Windows Store/MSIX paths, there is relocation logic for bundled executables.
- Spawn with arguments `app-server --analytics-default-enabled`.

## SQLite Paths

Backend sqlite artifacts are created under `CODEX_HOME`, not under Chromium `--user-data-dir`.

Relevant source:

- `workspace-root-drop-handler-DeLi4ACJ.js @ offset 4305348`: `qV(...)` initializes `codexHome`.
- `main-B6QfY4LN.js @ offset 250717`: references `state_5.sqlite` and `sqlite`.
- `src-CoIhwwHr.js @ offset 332943`: fallback path for `CODEX_HOME`.

## Path Dependency Table

| Path | Source | Main Use | Isolated by `--user-data-dir` |
| --- | --- | --- | --- |
| Chromium user-data-dir | command-line switch | Browser profile files | Yes |
| Electron userData | `CODEX_ELECTRON_USER_DATA_PATH` or appData fallback | Single-instance lock and app-level persistence | No |
| Backend CODEX_HOME | `CODEX_HOME` or `%USERPROFILE%\.codex` fallback | app-server state and config | No |
| app-server executable | bundled resources path | `resources\codex.exe app-server` | Not applicable |

## Main Conclusion

The failing profile signature is explained by isolating Chromium and backend paths while leaving Electron `userData` shared.

If Electron `userData` is shared, `requestSingleInstanceLock()` can reject a new profile before backend startup, even though the new profile has a unique `--user-data-dir` and `CODEX_HOME`.
