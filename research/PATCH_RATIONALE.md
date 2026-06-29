# Patch Rationale

Date: 2026-06-29

Production code changes in this phase: none.

## Decision

Do not apply another production patch yet.

The evidence proves several important launch and startup facts, but it does not prove one exact remaining trigger that explains every successful and failed launch. Applying another launcher architecture change now would be speculative.

## Proposal Review

| Proposal | Classification | Evidence-based assessment |
| --- | --- | --- |
| Keep unique `--user-data-dir` | Proven necessary | It isolates Chromium profile artifacts, but it is not sufficient for backend startup. |
| Keep unique `CODEX_HOME` | Proven necessary | Backend state is read/written under `CODEX_HOME` after main startup. |
| Treat backend artifacts as readiness | Proven | Chromium-only profiles are not successful Codex startups. |
| Set `CODEX_ELECTRON_USER_DATA_PATH` | Plausible and source-backed | Codex consumes it before `requestSingleInstanceLock()`, but latest controlled test contradicts it as the complete root cause. |
| Rewrite launcher with `cmd /c start` | Not currently justified | Prior parity work did not prove shell launch semantics cause the remaining failure. |
| Rewrite launcher with direct `Command::new` | Already current path | Current `utils.rs` uses direct spawn. It has not eliminated every observed manual failure. |
| Override `APPDATA`, `LOCALAPPDATA`, `USERPROFILE`, `HOME` | Speculative / risky | Earlier experiments broke or destabilized Electron bootstrap. Current Codex source does not require these overrides before the lock. |
| Copy or hide `auth.json` | Contradicted for this failure class | The failing Chromium-only class stops before backend/auth flow. |
| Add Windows Job Objects | Useful for lifecycle, not proven root-cause fix | Current code already assigns the child to a job object. It does not prove or disprove the startup failure. |
| Junction-based profile switching | Speculative | No current trace proves filesystem redirection is the remaining cause. |
| Start Menu / ShellExecute activation | Plausible but unproven | No trace proves MSIX activation broker differences explain the failure. |

## External Source Comparison

Electron official documentation states that `userData` defaults to appData plus app name and stores app configuration. This supports separating Electron `userData` from Chromium `--user-data-dir`.

Source:

- https://electronjs.org/docs/latest/api/app

Electron native source shows `requestSingleInstanceLock()` is implemented using `chrome::DIR_USER_DATA` and may fail with `LOCK_ERROR`, `PROFILE_IN_USE`, or `PROCESS_NOTIFIED`.

Source:

- https://raw.githubusercontent.com/electron/electron/main/shell/browser/api/electron_api_app.cc

Community evidence also shows Electron single-instance behavior on Windows has named-pipe/session edge cases, but that is only contextual. It does not prove Aion's specific failure.

Example:

- https://github.com/electron/electron/issues/33975

VS Code and portable Electron projects reinforce that applications must explicitly separate app/user data to support parallel or portable instances, but they do not prove Codex's exact internal decision.

Examples:

- https://github.com/microsoft/vscode/issues/130504
- https://github.com/microsoft/vscode/issues/97626
- https://github.com/electron-userland/electron-builder/issues/6473

## Why No Patch Is Justified Yet

The latest controlled test produced a critical contradiction:

- Two profiles with the same Electron userData override both created backend artifacts.
- Two profiles with different Electron userData overrides also created backend artifacts.

If shared Electron userData were the complete deterministic cause, the shared-userData test should have reliably produced a blocked second instance. It did not.

Therefore, the actual failure likely depends on an additional condition not yet captured:

- active lock-owner lifetime,
- app-user-model or MSIX package state,
- profile lock contention,
- native singleton object state,
- job object interaction,
- or another startup condition before app-server spawn.

All of these remain `UNKNOWN`.

## Highest-Confidence Next Step

Add temporary forensic instrumentation only, not production behavior changes.

The instrumentation must capture a user-reproduced failed manual launch and answer:

1. Does the primary `Codex.exe` exit?
2. What is its exit code?
3. Does `resources\codex.exe app-server` ever spawn?
4. Is a second-instance event delivered to another process?
5. Does the native single-instance branch return false?

Recommended capture methods:

- WMI or ETW process creation/termination trace filtered to `Codex.exe` and `codex.exe`.
- Process command line, parent PID, creation timestamp, exit timestamp, exit code.
- Filesystem artifact timestamps under the exact `CODEX_HOME`.
- Optional temporary unpacked `bootstrap.js` logging if an isolated copy of the package can be used without modifying production Aion.

## Minimal Future Patch Conditions

Only patch production after one of these is proven:

- If `requestSingleInstanceLock()` returns false for failed launches, patch the exact userData/lock namespace issue.
- If app-server starts and exits, patch the exact app-server precondition or environment issue shown by exit code/logs.
- If app-server is never attempted despite lock success, patch the main startup condition that prevents transport creation.
- If the primary process is killed by a Job Object or parent lifecycle, patch process containment.

Until one of those is proven, the correct production patch is `none`.
