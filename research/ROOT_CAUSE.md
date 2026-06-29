# Root Cause: Codex Desktop Stops Before Backend Startup

## Conclusion

Codex Desktop gates backend startup behind Electron's `requestSingleInstanceLock()`.

The lock is acquired after Codex sets Electron `userData`, but before Codex imports the main desktop bundle and before it creates the app-server transport.

When a launch loses this single-instance lock, Codex exits from `bootstrap.js` before `runMainAppStartup()` executes. In that state:

- Chromium may already have created profile artifacts under `--user-data-dir`.
- `resources\codex.exe app-server` is never spawned.
- Backend artifacts under `CODEX_HOME` are never created.
- The existing lock-owning Codex instance receives the launch arguments and can show/focus an already logged-in/global-looking window.

This is the root cause of the observed Chromium-only failed profiles.

## Evidence

### Evidence 1: Electron userData is independent from `--user-data-dir`

Source:

`app.asar!.vite/build/bootstrap.js:1 @ offset 2491`

Observed logic:

- `CODEX_ELECTRON_USER_DATA_PATH` overrides Electron `userData`.
- Otherwise Electron `userData` is derived from `app.getPath("appData")` plus the product/build-flavor directory.
- The function does not consume `--user-data-dir`.
- The function does not consume `CODEX_HOME`.

Confidence:

High.

### Evidence 2: Electron userData is set before the single-instance lock

Source:

`app.asar!.vite/build/bootstrap.js:1 @ offset 13377`

Observed sequence:

1. `app.setPath("userData", ...)`
2. Windows AppUserModelId setup.
3. Single-instance policy calculation.
4. `requestSingleInstanceLock()`.

Confidence:

High.

### Evidence 3: Losing the single-instance lock exits before main startup

Source:

`app.asar!.vite/build/bootstrap.js:1 @ offsets 13599-14114`

Observed branch:

- If the lock call returns false, Codex logs `Exiting second desktop instance`.
- It then calls `app.exit(0)`.
- The dynamic import of `.vite/build/main-B6QfY4LN.js` and call to `runMainAppStartup()` are in the success branch only.

Confidence:

High.

### Evidence 4: Backend spawn code lives after `runMainAppStartup()`

Sources:

- `app.asar!.vite/build/main-B6QfY4LN.js:1 @ offset 1676262`: `function Lee()` is `runMainAppStartup`.
- `app.asar!.vite/build/main-B6QfY4LN.js:1 @ offset 1678053`: resolves `j.codexHome` and runs state reconciliation.
- `app.asar!.vite/build/src-CoIhwwHr.js:1 @ offset 576320`: local stdio transport class.
- `app.asar!.vite/build/src-CoIhwwHr.js:1 @ offset 572678`: `child_process.spawn(...)` starts the app-server process.
- `app.asar!.vite/build/src-CoIhwwHr.js:1 @ offset 578629`: app-server args are `app-server --analytics-default-enabled`.

Consequence:

If bootstrap exits before importing `main-B6QfY4LN.js`, there is no code path that can spawn `resources\codex.exe app-server`.

Confidence:

High.

### Evidence 5: Backend state is bound to CODEX_HOME, but only after main startup

Sources:

- `app.asar!.vite/build/workspace-root-drop-handler-DeLi4ACJ.js:1 @ offset 4305348`: `qV({ moduleDir })` initializes `codexHome`.
- `app.asar!.vite/build/src-CoIhwwHr.js:1 @ offset 332943`: `CODEX_HOME` fallback logic.

Consequence:

Correct `CODEX_HOME` is necessary but not sufficient. The backend must start before `CODEX_HOME` can receive backend artifacts.

Confidence:

High.

## Why Failed Profiles Contain Only Chromium Artifacts

Chromium can initialize enough to create:

- `Local State`
- `Default\`
- `Crashpad\`

before the Codex main app imports and before the app-server transport is created.

If `requestSingleInstanceLock()` fails after this early Chromium setup, the profile will look partially initialized but will never contain backend files.

This exactly matches the observed failed profile signature.

## Why The Behavior Is Non-Deterministic

The result depends on whether another Codex process already owns the Electron single-instance lock for the same Electron `userData` namespace.

- If no process owns the lock, startup proceeds to `runMainAppStartup()` and backend initialization.
- If a process already owns the lock, the new launch exits before backend startup.

The profile-specific `--user-data-dir` and `CODEX_HOME` do not determine this lock namespace. Electron `userData` does.

Confidence:

High for control-flow mechanism. Exact native mutex/IPC object name is not present in the bundled JavaScript and was not asserted.

## Rejected Causes

These are not the root cause of the Chromium-only backend-missing profile state:

- Missing `--user-data-dir`: source and process evidence already show it is present.
- Missing `CODEX_HOME`: backend path source shows it is consumed later, but failed launches never reach backend startup.
- `auth.json`: authentication happens after backend and UI startup; failed profiles stop earlier.
- app-server crash after spawn: failed profiles where app-server never appears are explained by bootstrap exit before app-server spawn.
- `cmd.exe` or BAT semantics: Codex's internal lock gate can reject a launch regardless of external command-line parity.

## Root Cause Statement

Codex Desktop's backend startup is blocked by a pre-main Electron single-instance lock that is keyed to Electron `userData`, not to Chromium `--user-data-dir` and not to `CODEX_HOME`.

When multiple Aion profiles share the same Electron `userData`, a new profile launch can become a second instance. Codex then exits at `bootstrap.js` before importing `main-B6QfY4LN.js`, so `resources\codex.exe app-server` is never spawned and backend artifacts are never created.

## Confidence

Overall confidence: high.

The conclusion is based on source-level control flow from the unpacked Codex Desktop package and matches the exact failed profile artifact pattern.

The only missing low-level detail is the exact native Electron IPC/mutex object used by `requestSingleInstanceLock()`. That implementation is inside Electron native code, not inside the Codex Desktop JavaScript bundle inspected here.
