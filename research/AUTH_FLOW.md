# AUTH_FLOW

## Evidence Files

- Login status: `C:\Users\WebVajhegan\AppData\Roaming\Aion\config\logs\manual-forensics\20260629-050630-login-status.json`
- Auth presence: `C:\Users\WebVajhegan\AppData\Roaming\Aion\config\logs\manual-forensics\20260629-050159-auth-presence.json`

## Official Codex CLI Auth Status

| Label | `CODEX_HOME` | `auth.json` | `codex.exe login status` |
| --- | --- | --- | --- |
| global | `C:\Users\WebVajhegan\.codex` | true | `Logged in using ChatGPT` |
| BAT fresh profile | `C:\Users\WebVajhegan\CodexProfiles\AionForensicBat-20260629-045454` | false | `Not logged in` |
| Aion fresh profile | `C:\Users\WebVajhegan\CodexProfiles\AionForensicAion-20260629-044559` | false | `Not logged in` |

## Timeline From Observed Evidence

1. Launcher creates profile directory.
2. Launcher sets `CODEX_HOME=<profile>`.
3. Launcher starts `Codex.exe --user-data-dir="%CODEX_HOME%"`.
4. Primary `Codex.exe` starts Chromium child processes with the profile `--user-data-dir`.
5. Primary `Codex.exe` starts a dedicated `resources\codex.exe app-server`.
6. The dedicated app-server inherits the correct `CODEX_HOME`.
7. The profile creates backend SQLite/runtime files.
8. The profile does not create `auth.json`.
9. Official Codex CLI reports the profile as `Not logged in`.
10. Global `%USERPROFILE%\.codex\auth.json` exists and reports `Logged in using ChatGPT`.

## Reasoning Chain

If profile-local auth were created or used, `codex.exe login status` with `CODEX_HOME=<profile>` would report logged in, or at least `auth.json` would exist in the profile.

It does not.

Therefore, any signed-in UI shown by Codex Desktop for these fresh profiles is not sourced from the profile-local auth state.

The only verified signed-in auth state is global:

```text
C:\Users\WebVajhegan\.codex\auth.json
```

## Conclusion

Codex Desktop launch does not deterministically create profile-local auth for fresh profiles. It can show a signed-in account while the official profile-local CLI state remains `Not logged in`.

Confidence: **0.95**

The most likely account source for that signed-in UI is global `%USERPROFILE%\.codex\auth.json`, or a Desktop/app-server auth cache derived from it.

Confidence: **0.85**

The exact first read of that global auth source still requires ProcMon/ETW file-read tracing.

Confidence: **1.00**
