# Launch Bug Description

## Expected Behavior

Launching a managed Aion profile should behave like the legacy BAT launcher:

1. Each profile uses its own directory under:

   ```text
   %USERPROFILE%\CodexProfiles\<profile>
   ```

2. Codex Desktop receives:

   ```text
   CODEX_HOME=<profile directory>
   --user-data-dir=<profile directory>
   ```

3. A fresh profile should open Codex Desktop using only that profile's isolated state.

4. The profile should not reuse another profile's visible window or runtime state.

## Observed Behavior

The following behavior has been observed during manual testing:

1. Some launches create Chromium profile artifacts in the profile directory.
2. Some launches do not create backend artifacts such as:

   ```text
   config.toml
   .codex-global-state.json
   state_*.sqlite
   logs_*.sqlite
   memories_*.sqlite
   ```

3. Successful launches eventually create `resources\codex.exe app-server` under the primary `Codex.exe`.
4. Successful backend startup can take roughly 10-20 seconds after the primary `Codex.exe` appears.
5. Aion currently records launch success after detecting the primary `Codex.exe` with the expected `--user-data-dir`.

## Proven Facts Already Ruled Out

These points have been verified and should not be re-investigated unless new evidence contradicts them:

1. Aion passes a profile-specific `--user-data-dir`.
2. Aion sets `CODEX_HOME` to the profile directory in the generated launcher script.
3. The generated Aion launcher script mirrors the BAT core payload:

   ```bat
   set "CODEX_HOME=<profile>"
   start "" "<Codex.exe>" --user-data-dir="%CODEX_HOME%"
   ```

4. Aion no longer writes `aion-launch.cmd` inside the profile directory. Launch scripts are generated outside the profile under:

   ```text
   %TEMP%\Aion\Launchers\
   ```

5. The profile sandbox root is outside `%APPDATA%` on Windows:

   ```text
   %USERPROFILE%\CodexProfiles\
   ```

6. The GUI `Codex.exe` is not an AppExecutionAlias found on PATH. The package manifest declares:

   ```text
   Executable="app/Codex.exe"
   EntryPoint="Windows.FullTrustApplication"
   ```

7. The current Aion Windows path does not launch GUI Codex directly with Rust `Command::new(Codex.exe)`. Rust launches `cmd.exe`, and `cmd.exe` runs the generated `.cmd` script containing `start ""`.

## Scope For Review

Review should focus on the launch pipeline from `commands::launch_profile` through `CodexProcessProvider::launch_isolated_codex_process`, including:

- profile directory selection
- launcher script generation
- environment construction
- `cmd.exe /d /c` invocation
- `start "" "<Codex.exe>" --user-data-dir="%CODEX_HOME%"`
- process detection criteria after launch

