# Backend Startup: `resources\codex.exe app-server`

This document explains the exact app-server startup path found inside the Codex Desktop package.

Investigation target:

`C:\Program Files\WindowsApps\OpenAI.Codex_26.623.8305.0_x64__2p2nqsd0c76g0\app\resources\app.asar`

## Required Precondition

The backend is not started by `bootstrap.js`.

The backend can only start after:

1. Electron `userData` is set.
2. `requestSingleInstanceLock()` succeeds.
3. `app.whenReady()` resolves.
4. SQLite readiness guard returns true.
5. `.vite/build/main-B6QfY4LN.js` is imported.
6. `runMainAppStartup()` executes.

If any precondition before item 5 fails, the app-server code is never imported.

## Startup Entry

Source:

`app.asar!.vite/build/main-B6QfY4LN.js:1 @ offset 1676262`

Function:

`Lee()`, exported as `runMainAppStartup`.

Observed responsibilities:

- Log `Launching app`.
- Await `app.whenReady()`.
- Register the app protocol.
- Hydrate shell environment.
- Resolve desktop paths using `r.E({ moduleDir })`.
- Run state reconciliation using `Yp(j.codexHome)`.
- Build desktop window services and app-server connection infrastructure.

## Codex Home Binding

Source:

`app.asar!.vite/build/workspace-root-drop-handler-DeLi4ACJ.js:1 @ offset 4305348`

Function:

`qV({ moduleDir })`

Responsibility:

- Resolve `codexHome` through `t.Wr()`.
- Initialize `.codex-global-state.json`.
- Initialize `config.toml`.
- Return `{ codexHome, preloadPath, desktopRoot, repoRoot, globalState, settingsStore }`.

Supporting source:

`app.asar!.vite/build/src-CoIhwwHr.js:1 @ offset 332943`

Function:

`LT({ preferWsl = false })`

Behavior:

- Returns `process.env.CODEX_HOME` if set.
- Otherwise falls back to `path.join(os.homedir(), ".codex")`.

## App-Server Transport Selection

Source:

`app.asar!.vite/build/main-B6QfY4LN.js:1 @ offset 1290810`

Function:

`L2(...)`

Behavior:

- Selects websocket transport for remote SSH/control hosts.
- Selects WSL transport for WSL hosts.
- Selects local stdio transport for local hosts.

For local Windows desktop use, the selected path is:

`new t.On({ hostConfig, repoRoot, resourcesPath, defaultOriginator })`

## Local Stdio Transport

Source:

`app.asar!.vite/build/src-CoIhwwHr.js:1 @ offset 576320`

Class:

`PR`

Behavior:

- `kind = "stdio"`.
- `connect()` calls `FR(options)`.
- If no executable is found, throws an explicit error saying the Codex CLI binary cannot be located.
- Otherwise returns `new NR(...)`.

## App-Server Executable Resolution

Source:

`app.asar!.vite/build/src-CoIhwwHr.js:1`

Functions:

- `FR(options)` at offset `576703`.
- `zR(...)` at offset `578629`.
- `rM(resourcesPath)` at offset `456251`.

Behavior:

- `zR(...)` resolves the local app-server executable and arguments.
- It first honors an explicit host `codex_cli_command`.
- Otherwise it searches for bundled `codex.exe` under the resources path.
- On Windows, `rM(...)` resolves `codex.exe`.
- The selected arguments are `app-server --analytics-default-enabled`.

## Process Spawn

Source:

`app.asar!.vite/build/src-CoIhwwHr.js:1 @ offset 572678`

Class:

`NR`

Function:

`spawnProcess()`

Behavior:

- Calls Node `child_process.spawn(...)`.
- Uses stdio pipes.
- Uses the environment produced by `FR(...)`.
- Logs `stdio_transport_spawned` with executable path and PID.
- Marks the transport open when a PID exists.

## Backend Artifact Creation

The backend artifacts are downstream of `resources\codex.exe app-server`.

Expected artifacts under `CODEX_HOME` after backend startup:

- `config.toml`
- `.codex-global-state.json`
- `state_*.sqlite`
- `logs_*.sqlite`
- `memories_*.sqlite`

If these are absent and only Chromium artifacts exist, execution did not reach or complete the app-server spawn path.

## Important Distinction

The primary `Codex.exe` Chromium process and the backend `resources\codex.exe app-server` are not started by the same code path.

- Chromium/Electron profile files can appear during bootstrap.
- Backend files only appear after `runMainAppStartup()` constructs and connects the local app-server transport.

Therefore a profile with only Chromium artifacts is consistent with an exit in `bootstrap.js` before `main-B6QfY4LN.js` is loaded.
