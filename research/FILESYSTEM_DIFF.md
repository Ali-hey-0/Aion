# FILESYSTEM_DIFF

## Evidence Files

- Auth presence: `C:\Users\WebVajhegan\AppData\Roaming\Aion\config\logs\manual-forensics\20260629-050159-auth-presence.json`
- File listing: `C:\Users\WebVajhegan\AppData\Roaming\Aion\config\logs\manual-forensics\20260629-050159-current-forensic-profile-files.json`

## Fresh Profile Auth State

| Profile | `auth.json` | `.codex-global-state.json` | `config.toml` | `state_*.sqlite` | `logs_*.sqlite` |
| --- | --- | --- | --- | ---: | ---: |
| `AionForensicBat-20260629-045454` | false | true | true | 1 | 1 |
| `AionForensicAion-20260629-044559` | false | true | true | 1 | 1 |

## Created State

Both fresh profiles created the same categories of state:

- Chromium profile state: `Local State`, `Default\`, `Crashpad\`, `BrowserMetrics`, GPU/shader/cache directories.
- Codex backend state: `.codex-global-state.json`, `config.toml`, `state_5.sqlite`, `logs_2.sqlite`, `memories_1.sqlite`, `goals_1.sqlite`, `installation_id`.

Neither profile created `auth.json`.

## Script Comparison

BAT-pattern generated script:

```bat
@echo off
set "CODEX_HOME=C:\Users\WebVajhegan\CodexProfiles\AionForensicBat-20260629-045454"
start "" "C:\Program Files\WindowsApps\OpenAI.Codex_26.623.5546.0_x64__2p2nqsd0c76g0\app\Codex.exe" --user-data-dir="%CODEX_HOME%"
```

Aion-pattern generated script:

```bat
@echo off
set "CODEX_HOME=C:\Users\WebVajhegan\CodexProfiles\AionForensicAion-20260629-044559"
start "" "C:\Program Files\WindowsApps\OpenAI.Codex_26.623.5546.0_x64__2p2nqsd0c76g0\app\Codex.exe" --user-data-dir="%CODEX_HOME%"
exit /b 0
```

The only script difference is `exit /b 0` after `start`.

## Conclusion

BAT and Aion fresh profiles are filesystem-equivalent for the relevant authentication question: both create runtime/backend state, and neither creates local `auth.json`.

Confidence: **0.95**

Therefore, if the UI opens a signed-in account for either fresh profile, that identity cannot be coming from profile-local `auth.json`.

Confidence: **0.95**
