# Aion Execution Environment Diff

Collected raw evidence: `G:\others\Aion-Codex\forensics\execution-env-20260629-143228`

## Scope

This report compares the current Codex Agent shell environment with the environment produced by `npm run env` from the same workspace. No production code was modified.

## Binary Environment

Both logical and physical workspace paths resolve to the same debug executable:

| Path | SHA256 | Size |
| --- | --- | --- |
| `G:\others\Aion-Codex\src-tauri\target\debug\aion.exe` | `C9356544661EB930BE5DEDC2A10029832FE56F1000D045245C571D9A64779E7F` | `13419520` |
| `C:\Users\WebVajhegan\Desktop\vs code\Aion\src-tauri\target\debug\aion.exe` | `C9356544661EB930BE5DEDC2A10029832FE56F1000D045245C571D9A64779E7F` | `13419520` |

No release binary was present at either path.

Conclusion: the observed behavior is not explained by Agent and manual runs executing different Aion binaries.

Confidence: high.

## Environment Variables

Current Agent environment count: `95`

`npm run env` environment count: `121`

Differing variables: `27`

Relevant differences:

| Variable | Agent | npm run env |
| --- | --- | --- |
| `HOME` | not set | `C:\Users\WebVajhegan` |
| `INIT_CWD` | not set | `G:\others\Aion-Codex` |
| `NODE` | not set | `C:\Program Files\nodejs\node.exe` |
| `NPM_*` | not set | npm lifecycle/config/package variables |
| `PATH` | inherited Codex Agent PATH | npm prepends workspace `node_modules\.bin`, parent `node_modules\.bin`, and npm `node-gyp-bin` |

Shared important values:

| Variable | Status |
| --- | --- |
| `USERPROFILE` | present in both |
| `APPDATA` | present in both |
| `LOCALAPPDATA` | present in both |
| `TEMP` / `TMP` | present in both |
| `COMSPEC` | present in both |
| `CARGO_HOME` / `RUSTUP_HOME` | present through inherited environment |

## Effect On Codex Launch

The current Windows launcher intentionally clears the inherited environment before executing Codex and restores only a whitelist:

`ALLUSERSPROFILE`, `APPDATA`, `COMSPEC`, `HOMEDRIVE`, `HOMEPATH`, `LOCALAPPDATA`, `PATHEXT`, `PROGRAMFILES`, `SYSTEMROOT`, `TEMP`, `TMP`, `USERPROFILE`, `WINDIR`, and other OS-level variables.

It does not preserve `HOME`, `INIT_CWD`, `NODE`, or `NPM_*`.

It also filters development PATH segments such as:

- `node_modules\.bin`
- npm run-script node-gyp bin path
- `src-tauri\target\debug`
- Rust toolchain internals

Conclusion: Agent vs npm environment differences are real, but current launcher code should prevent the npm-specific variables from reaching the Codex child process. The environment hypothesis is not proven by the collected data.

Confidence: medium-high.

## Missing Evidence

At capture time, no live `aion.exe`, `cargo.exe`, or `npm run tauri dev` process existed. Therefore this report cannot claim a complete live manual-run environment diff. It compares Agent shell vs npm lifecycle environment, not an active user-launched Aion process.

Confidence for that limitation: high.
