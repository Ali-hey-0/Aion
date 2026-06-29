# Aion Root Cause Final Investigation

Date: 2026-06-29

Scope: determine the exact Windows/Electron decision that makes a Codex Desktop launch proceed as an isolated instance or silently join/exit before backend startup.

Production code changes in this phase: none.

## Current Conclusion

The exact root cause is not fully proven.

The strongest source-backed pre-backend exit is Electron's `requestSingleInstanceLock()` branch in Codex Desktop `bootstrap.js`. That branch can exactly produce the observed failed-profile signature: Chromium files exist, but `resources\codex.exe app-server` never starts and backend files never appear.

However, the latest controlled evidence contradicts the narrower claim that shared Electron `userData` alone explains every failure. A test where two profiles intentionally shared the same `CODEX_ELECTRON_USER_DATA_PATH` still produced backend artifacts for both profiles. Therefore the final trigger is still `UNKNOWN`.

## Proven Facts

### 1. Chromium-only profiles are not successful Codex startups

Successful profiles create backend artifacts under `CODEX_HOME`:

- `config.toml`
- `.codex-global-state.json`
- `state_*.sqlite`
- `logs_*.sqlite`
- `memories_*.sqlite`
- `sqlite\codex-dev.db`

Failed profiles contain only early Chromium/Electron artifacts such as:

- `Local State`
- `Default\`
- `Crashpad\`

Conclusion: backend readiness, not Chromium artifact creation, is the valid success signal.

Confidence: high.

### 2. Codex Desktop sets Electron `userData` before `requestSingleInstanceLock()`

Source: `C:\Program Files\WindowsApps\OpenAI.Codex_26.623.8305.0_x64__2p2nqsd0c76g0\app\resources\app.asar!.vite\build\bootstrap.js`

Observed sequence:

1. Resolve build flavor and app brand.
2. `app.setName(...)`
3. `app.setPath("userData", computedPath)`
4. `app.setAppUserModelId(...)` on Windows.
5. `app.requestSingleInstanceLock()`

The computed path comes from `CODEX_ELECTRON_USER_DATA_PATH` when set. Otherwise it falls back to `app.getPath("appData")\Codex` for the production Codex package.

Confidence: high.

### 3. Losing `requestSingleInstanceLock()` exits before backend startup

Source: same `bootstrap.js`.

Observed branch:

- If `requestSingleInstanceLock()` returns false, Codex logs a second-instance exit and calls `app.exit(0)`.
- The dynamic import of `main-B6QfY4LN.js` happens only on the success branch.
- The backend app-server spawn path is in the main bundle, not in `bootstrap.js`.

Conclusion: if this branch is taken, `resources\codex.exe app-server` cannot start.

Confidence: high.

### 4. Electron native code ties the single-instance lock to `chrome::DIR_USER_DATA`

Primary source: Electron `electron_api_app.cc` from the Electron project.

Relevant source evidence:

- `RequestSingleInstanceLock()` fetches `chrome::DIR_USER_DATA`.
- It then calls `Browser::Get()->RequestSingleInstanceLock(...)`.
- It returns false for `LOCK_ERROR`, `PROFILE_IN_USE`, and `PROCESS_NOTIFIED`.

Source URL:

- https://raw.githubusercontent.com/electron/electron/main/shell/browser/api/electron_api_app.cc

Official docs also define `userData` as the app configuration directory, defaulting to `appData` plus app name.

Source URL:

- https://electronjs.org/docs/latest/api/app

Confidence: high.

### 5. `CODEX_ELECTRON_USER_DATA_PATH` alone did not eliminate failures as a proven cause

Latest controlled run root:

`C:\Users\WebVajhegan\CodexProfiles\AionFinalLock-20260629-221843-*`

Important observation:

- `SameElectronUserData\P1` and `SameElectronUserData\P2` intentionally used one shared Electron userData directory:
  `C:\Users\WebVajhegan\CodexProfiles\AionFinalLock-20260629-221843-SameElectronUserData\SharedElectronUserData`
- Both profile roots produced backend artifacts.
- `DifferentElectronUserData\P1` and `DifferentElectronUserData\P2` also produced backend artifacts.

Therefore:

- A unique Electron userData path is still architecturally correct.
- But the exact intermittent failure cannot be reduced to "Electron userData is shared" without further proof.

Confidence: high for the observation, medium for interpretation.

## Rejected Hypotheses

### Missing `--user-data-dir`

Rejected. Process command lines showed the switch reaches the primary `Codex.exe` and child Chromium processes.

Confidence: high.

### Missing `CODEX_HOME`

Rejected. Environment tracing showed `CODEX_HOME` reaches both `Codex.exe` and the app-server when the app-server starts.

Confidence: high.

### `launcher.cmd` inside the profile as the primary cause

Rejected as primary cause. That pollution was removed, but the user still observed the failure pattern.

Confidence: medium-high.

### `cmd.exe` or early shell exit as primary cause

Rejected by source and process evidence. Successful backend startup occurs after the shell process exits, and direct spawn has also been tested.

Confidence: medium-high.

### Authentication file fallback as the active failure point

Rejected for the Chromium-only failure class. Auth is downstream of backend startup; the failing profiles do not reach backend initialization.

Confidence: high.

### Shared Electron userData as the complete root cause

Rejected as a complete explanation by the latest controlled `SameElectronUserData` run, where both profiles created backend artifacts.

Confidence: medium-high.

## Remaining Unknowns

### Does failed `Codex.exe` exit immediately or stay alive?

UNKNOWN for the most recent user-reproduced failure. The current controlled run did not reproduce a failed launch.

Needed evidence:

- Primary `Codex.exe` PID creation timestamp.
- Exit timestamp and exit code.
- Whether a second-instance event was delivered.

### Is `requestSingleInstanceLock()` returning false in the actual failed launch?

UNKNOWN. Source control flow proves this branch can explain the signature, but no direct log of the return value exists for the user-reproduced failed run.

Needed evidence:

- Temporary JS logging inside unpacked Codex startup, or native process tracing showing primary exit before main import.

### Is `resources\codex.exe app-server` ever attempted in the actual failed launch?

UNKNOWN for the most recent failure.

Needed evidence:

- ETW/WMI process creation and termination events for `resources\codex.exe`.
- Exit code if it starts and exits.

### Does another global resource participate before backend startup?

UNKNOWN.

Candidates still requiring proof:

- Electron process singleton named pipe or mutex state.
- AppUserModelID interaction.
- MSIX package identity / package-global singleton behavior.
- Job object side effects.
- Hidden Electron startup lock file or process singleton handle.

## Final Root Cause Statement

The exact root cause remains UNKNOWN.

The proven earliest source-level gate that can stop backend startup is:

`Codex.exe -> bootstrap.js -> app.setPath("userData") -> requestSingleInstanceLock() -> app.exit(0)`

But the exact condition that makes real manual Aion launches enter that branch, or otherwise skip app-server startup, has not been captured with hard evidence.

## Confidence

- Confidence that failed launches stop before backend startup: high.
- Confidence that `requestSingleInstanceLock()` is a valid pre-backend exit: high.
- Confidence that this exact branch caused the latest user failure: medium, not proven.
- Confidence that Electron userData uniqueness alone fixes all failures: low after the latest controlled contradiction.

## No Production Patch Justification

No production patch is justified by the final evidence set.

The next valid action is not another launcher rewrite. It is targeted process-event and single-instance instrumentation around an actual failed manual launch.
