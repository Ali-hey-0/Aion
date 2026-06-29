# Aion Codex Isolation Diagnosis

## Current Conclusion

Aion is now passing `--user-data-dir` correctly to the MSIX-packaged Codex desktop process. The launched Chromium/Electron profile is being created under:

```text
C:\Users\WebVajhegan\AionProfiles\<UUID>
```

The remaining isolation failure is not primarily a Chromium `--user-data-dir` failure. It is a Codex backend/app-server home-state failure.

Codex has at least two state layers:

1. Chromium/Electron UI profile state controlled by `--user-data-dir`.
2. Codex backend/app-server state containing `auth.json`, `.codex-global-state.json`, `config.toml`, sqlite state/logs, memories, plugins, etc.

Aion currently isolates layer 1, but layer 2 still appears to resolve to the global Codex home or fails to initialize inside the Aion profile root.

## Evidence

Running Codex processes show the sandbox argument is present:

```text
"C:\Program Files\WindowsApps\OpenAI.Codex_26.623.5546.0_x64__2p2nqsd0c76g0\app\Codex.exe"
--user-data-dir="C:\Users\WebVajhegan\AionProfiles\a8b4f5e8-9fe8-46b4-a78c-75fa35d8d782"
```

Child Chromium processes also inherit the same sandbox:

```text
--type=renderer
--user-data-dir="C:\Users\WebVajhegan\AionProfiles\a8b4f5e8-9fe8-46b4-a78c-75fa35d8d782"
```

The Aion profile root contains Chromium data:

```text
Local State
Default\
Crashpad\
BrowserMetrics\
GPUPersistentCache\
...
```

But the same Aion profile root is missing Codex backend files:

```text
auth.json                         MISSING
config.toml                       MISSING
.codex-global-state.json          MISSING
state_5.sqlite                    MISSING
logs_2.sqlite                     MISSING
```

Legacy working profiles under `C:\Users\WebVajhegan\CodexProfiles\AccX` contain both Chromium data and Codex backend state:

```text
auth.json
config.toml
.codex-global-state.json
state_5.sqlite
logs_2.sqlite
Local State
Default\
...
```

Legacy `config.toml` files contain:

```toml
NODE_REPL_TRUSTED_CODE_PATHS = 'C:\Users\WebVajhegan\CodexProfiles\AccX'
CODEX_HOME = 'C:\Users\WebVajhegan\CodexProfiles\AccX'
```

The global Codex home still exists at:

```text
C:\Users\WebVajhegan\.codex
```

and contains:

```text
auth.json
config.toml
.codex-global-state.json
state_5.sqlite
logs_2.sqlite
```

## Important Legacy Batch Detail

The legacy batch file does not only pass `--user-data-dir`. It also sets `CODEX_HOME` before launching:

```bat
set "CODEX_HOME=%USERPROFILE%\CodexProfiles\%ACC_NAME%"
start "" "%APP_DIR%" --user-data-dir="%CODEX_HOME%"
```

That means the working implementation likely isolated both:

```text
Chromium profile   -> --user-data-dir
Codex backend home -> CODEX_HOME
```

## Current Rust/Bat Launcher Observation

The current generated Aion launcher also sets:

```bat
set "CODEX_HOME=C:\Users\WebVajhegan\AionProfiles\<UUID>"
start "" "%APP_DIR%" --user-data-dir="%CODEX_HOME%"
```

However, the Aion profile root still does not receive `auth.json`, `config.toml`, `.codex-global-state.json`, or sqlite state files. This suggests one of the following:

1. `CODEX_HOME` is not reaching the spawned MSIX/Electron/app-server process.
2. The app-server no longer uses `CODEX_HOME` during first bootstrap unless a compatible `config.toml` already exists.
3. The backend app-server has a global singleton or IPC/state lock and reuses the already-running global Codex home.
4. The MSIX execution path strips some environment variables when launched through the current hidden Tauri-created `cmd.exe` flow.
5. Legacy success may depend on using the exact profile root shape under `CodexProfiles\AccX`, including pre-existing `config.toml` and related app-server files.

## Recommended Next Experiment

Stop all Codex processes, then test one clean Aion profile after pre-seeding its root with a minimal `config.toml`:

```toml
[env]
CODEX_HOME = 'C:\Users\WebVajhegan\AionProfiles\<UUID>'
NODE_REPL_TRUSTED_CODE_PATHS = 'C:\Users\WebVajhegan\AionProfiles\<UUID>'
```

Also launch with a batch payload that explicitly sets:

```bat
set "CODEX_HOME=C:\Users\WebVajhegan\AionProfiles\<UUID>"
set "NODE_REPL_TRUSTED_CODE_PATHS=C:\Users\WebVajhegan\AionProfiles\<UUID>"
start "" "<Codex.exe>" --user-data-dir="%CODEX_HOME%"
```

Then verify whether these files are created inside the Aion profile root:

```text
auth.json
config.toml
.codex-global-state.json
state_5.sqlite
logs_2.sqlite
```

If they are still missing, the remaining issue is not `--user-data-dir`; it is that the MSIX-packaged Codex app-server ignores or loses `CODEX_HOME` in this launch context.

## Shareable Prompt

```text
I am building a Tauri v2 + Rust desktop app called Aion. It manages isolated profiles for OpenAI Codex Desktop on Windows. Codex is installed as an MSIX/Windows Store package:

C:\Program Files\WindowsApps\OpenAI.Codex_26.623.5546.0_x64__2p2nqsd0c76g0\app\Codex.exe

Goal:
Launch multiple Codex instances, each with separate login/auth/session state.

Legacy working batch file:

@echo off
set "CODEX_HOME=%USERPROFILE%\CodexProfiles\Acc1"
start "" "C:\Program Files\WindowsApps\OpenAI.Codex_26.623.5546.0_x64__2p2nqsd0c76g0\app\Codex.exe" --user-data-dir="%CODEX_HOME%"

This works. The profile root under C:\Users\WebVajhegan\CodexProfiles\Acc1 contains both Chromium/Electron data and Codex backend files:

auth.json
config.toml
.codex-global-state.json
state_5.sqlite
logs_2.sqlite
Local State
Default\
Crashpad\

The config.toml inside working legacy profiles contains:

NODE_REPL_TRUSTED_CODE_PATHS = 'C:\Users\WebVajhegan\CodexProfiles\Acc1'
CODEX_HOME = 'C:\Users\WebVajhegan\CodexProfiles\Acc1'

Current Aion implementation:

1. Aion stores profile sandboxes outside AppData to avoid MSIX AppData virtualization:
   C:\Users\WebVajhegan\AionProfiles\<UUID>

2. Aion generates a temporary batch file:

@echo off
set "CODEX_HOME=C:\Users\WebVajhegan\AionProfiles\<UUID>"
set "APP_DIR="
for /f "tokens=*" %%i in ('dir /b /ad "C:\Program Files\WindowsApps\OpenAI.Codex_*_x64__*" 2^>nul') do (
set "APP_DIR=C:\Program Files\WindowsApps\%%i\app\Codex.exe"
)
if defined APP_DIR (
start "" "%APP_DIR%" --user-data-dir="%CODEX_HOME%"
) else (
start "" "<validated Codex.exe path>" --user-data-dir="%CODEX_HOME%"
)

3. Process command line proves --user-data-dir is correctly passed:

"C:\Program Files\WindowsApps\OpenAI.Codex_26.623.5546.0_x64__2p2nqsd0c76g0\app\Codex.exe"
--user-data-dir="C:\Users\WebVajhegan\AionProfiles\<UUID>"

Renderer/GPU/Crashpad child processes also contain the same --user-data-dir.

Problem:
The UI opens using the same logged-in global account. The Aion profile root receives Chromium data like:

Local State
Default\
Crashpad\
BrowserMetrics\

But it does NOT receive Codex backend/auth files:

auth.json                         missing
config.toml                       missing
.codex-global-state.json          missing
state_5.sqlite                    missing
logs_2.sqlite                     missing

The global Codex home at C:\Users\WebVajhegan\.codex still contains auth.json, config.toml, global state and sqlite files.

Hypothesis:
--user-data-dir isolation is working for Chromium/Electron, but Codex auth/session is controlled by the hidden app-server process:

app\resources\codex.exe app-server --analytics-default-enabled

That app-server is resolving to the global Codex home instead of the Aion profile root. The key issue is isolating Codex backend state, not Chromium profile state.

Question:
What is the correct Windows/MSIX-safe launch strategy to force OpenAI Codex Desktop's app-server/backend state to use C:\Users\WebVajhegan\AionProfiles\<UUID> instead of C:\Users\WebVajhegan\.codex?

Please analyze:

1. Whether CODEX_HOME is sufficient or needs to be paired with config.toml pre-seeding.
2. Whether NODE_REPL_TRUSTED_CODE_PATHS or other env/config keys are required.
3. Whether MSIX/WindowsApps strips env vars when launched through cmd/start from a Tauri process.
4. Whether Codex app-server has a singleton/global IPC lock that must be killed or namespaced.
5. Whether the launcher should set HOME/USERPROFILE or avoid that because Electron may break.
6. Whether Aion should launch resources\codex.exe app-server separately with CODEX_HOME before launching Codex.exe.
7. Why the legacy CodexProfiles\AccX batch works while the Tauri-generated batch with AionProfiles\<UUID> does not.

Please propose a minimal production-grade Rust/Tauri implementation strategy.
```
```

