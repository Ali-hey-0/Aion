# ENVIRONMENT_DIFF

## Evidence File

- `C:\Users\WebVajhegan\AppData\Roaming\Aion\config\logs\manual-forensics\20260629-050456-psutil-env-processes.json`

## Key Environment Values

| Process | PID | `CODEX_HOME` | `USERPROFILE` | `APPDATA` | `LOCALAPPDATA` | CWD |
| --- | ---: | --- | --- | --- | --- | --- |
| Global app-server | 10192 | empty | `C:\Users\WebVajhegan` | `C:\Users\WebVajhegan\AppData\Roaming` | `C:\Users\WebVajhegan\AppData\Local` | Codex app dir |
| BAT primary | 1276 | `C:\Users\WebVajhegan\CodexProfiles\AionForensicBat-20260629-045454` | `C:\Users\WebVajhegan` | `C:\Users\WebVajhegan\AppData\Roaming` | `C:\Users\WebVajhegan\AppData\Local` | Codex app dir |
| BAT app-server | 4860 | `C:\Users\WebVajhegan\CodexProfiles\AionForensicBat-20260629-045454` | `C:\Users\WebVajhegan` | `C:\Users\WebVajhegan\AppData\Roaming` | `C:\Users\WebVajhegan\AppData\Local` | Codex app dir |
| Aion primary | 11544 | `C:\Users\WebVajhegan\CodexProfiles\AionForensicAion-20260629-044559` | `C:\Users\WebVajhegan` | `C:\Users\WebVajhegan\AppData\Roaming` | `C:\Users\WebVajhegan\AppData\Local` | Codex app dir |
| Aion app-server | 17068 | `C:\Users\WebVajhegan\CodexProfiles\AionForensicAion-20260629-044559` | `C:\Users\WebVajhegan` | `C:\Users\WebVajhegan\AppData\Roaming` | `C:\Users\WebVajhegan\AppData\Local` | Codex app dir |

## Observed Differences

No account-relevant environment difference was observed between BAT and Aion primary/app-server processes.

Both launch methods preserve the global Windows user environment:

- `USERPROFILE=C:\Users\WebVajhegan`
- `APPDATA=C:\Users\WebVajhegan\AppData\Roaming`
- `LOCALAPPDATA=C:\Users\WebVajhegan\AppData\Local`

Both launch methods set `CODEX_HOME` correctly for their primary process and dedicated app-server process.

## Important Detail

Some Chromium renderer/utility child processes do not expose the full inherited environment through `psutil`, and several show blank `CODEX_HOME`, `USERPROFILE`, and `APPDATA`. This is not the account source process. The dedicated `resources\codex.exe app-server` does carry the expected `CODEX_HOME`.

## Conclusion

The Aion-vs-BAT authentication difference is not explained by `CODEX_HOME`, `USERPROFILE`, `APPDATA`, `LOCALAPPDATA`, or working directory differences in the primary/app-server processes.

Confidence: **0.95**

The remaining account-leak vector is Codex app-server logic: when profile-local auth is missing, it can still derive account identity from a global source reachable through the unchanged `USERPROFILE`/global user context.

Confidence: **0.80**
