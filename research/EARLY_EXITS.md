# Verified Early Exits Before App-Server Startup

This document lists only source-backed early-exit paths found inside Codex Desktop.

Investigation target:

`C:\Program Files\WindowsApps\OpenAI.Codex_26.623.8305.0_x64__2p2nqsd0c76g0\app\resources\app.asar`

## Summary

The only Windows-relevant early exit that exactly matches the failed profile signature is the Electron single-instance lock failure in `bootstrap.js`.

Failed profile signature:

- Chromium artifacts exist.
- `resources\codex.exe app-server` never starts.
- Backend files are absent:
  - `config.toml`
  - `state_*.sqlite`
  - `logs_*.sqlite`
  - `memories_*.sqlite`
  - `.codex-global-state.json`

This matches an exit after Electron/Chromium bootstrap but before importing `main-B6QfY4LN.js`.

## Early Exit 1: Electron Single-Instance Lock Failure

Source:

`app.asar!.vite/build/bootstrap.js:1 @ offsets 13377-14114`

Relevant control flow:

- Set Electron `userData`.
- Call `requestSingleInstanceLock()`.
- If the call returns false, log `Exiting second desktop instance`.
- Call `app.exit(0)`.
- Do not import `main-B6QfY4LN.js`.
- Do not call `runMainAppStartup()`.

Condition:

`app.requestSingleInstanceLock()` returns false.

Consequence:

- No `runMainAppStartup()`.
- No app-server transport construction.
- No `resources\codex.exe app-server`.
- No backend artifacts under `CODEX_HOME`.
- Existing lock-owning instance receives second-instance argv.

Evidence:

- `bootstrap.js @ offset 13377`: `app.setPath("userData", ...)`.
- `bootstrap.js @ offset 13599`: `requestSingleInstanceLock()`.
- `bootstrap.js @ offset 13641`: second instance log and `app.exit(0)`.
- `bootstrap.js @ offset 13781`: `second-instance` handler queues argv in the lock-owning process.
- `bootstrap.js @ offset 14114`: `runMainAppStartup` import occurs only after the successful branch.

Confidence:

High. This branch exactly explains Chromium-only profiles and missing backend artifacts.

## Early Exit 2: macOS Intel Warning Quit

Source:

`app.asar!.vite/build/bootstrap.js:1 @ function y(...)`

Condition:

Packaged macOS x64 build running under Rosetta and user chooses Quit.

Consequence:

`app.quit()` before main startup.

Relevance to Aion on Windows:

Not relevant. The condition is macOS-only.

Confidence:

High for source existence, not relevant to observed Windows failures.

## Early Exit 3: macOS DMG Install Flow

Source:

`app.asar!.vite/build/bootstrap.js:1 @ function ne(...)`

Condition:

Packaged macOS app is launched from a DMG/outside Applications and installation flow requires quit/relaunch.

Consequence:

Main startup is skipped.

Relevance to Aion on Windows:

Not relevant. The condition is macOS-only.

Confidence:

High for source existence, not relevant to observed Windows failures.

## Early Exit 4: SQLite Readiness Guard Quit

Source:

`app.asar!.vite/build/workspace-root-drop-handler-DeLi4ACJ.js:1 @ offset 4313378`

Export mapping:

- `bootstrap.js` calls `r.v()`.
- `workspace-root-drop-handler-DeLi4ACJ.js` maps export `v` to `OH`.
- `OH()` calls `AH(...)`, which opens the local SQLite state DB and may show a recovery dialog.

Condition:

The local SQLite readiness check fails and the user selects Quit in the recovery dialog.

Consequence:

`r.v()` returns false; bootstrap does not import `main-B6QfY4LN.js`.

Relevance to observed failure:

Possible in general, but it does not match the silent Chromium-only failure as strongly as the single-instance branch because this path is designed to show a database access dialog.

Confidence:

Medium. Source confirms the branch. No evidence from the provided symptoms shows the recovery dialog.

## Early Exit 5: Main Import Or Startup Exception

Source:

`app.asar!.vite/build/bootstrap.js:1 @ offset 14114`

Condition:

The dynamic import of `main-B6QfY4LN.js` or `runMainAppStartup()` throws.

Consequence:

- All BrowserWindows are destroyed.
- Error `Desktop bootstrap failed to start the main app` is logged.
- A failure dialog is shown.

Relevance to observed failure:

Possible in general, but not the best match. The observed failure is a normal-looking existing/global window with only Chromium artifacts in the new profile, not a visible startup failure dialog.

Confidence:

Medium. Source confirms the branch. Current symptom pattern does not point to this path.

## Early Exit Ranking For The Aion Failure

1. `requestSingleInstanceLock()` false - highest confidence.
2. SQLite readiness guard false - possible but weaker evidence.
3. main import/startup exception - possible but would normally surface as a startup failure.
4. macOS-only exits - not applicable.
