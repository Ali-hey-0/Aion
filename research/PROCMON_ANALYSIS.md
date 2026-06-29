# PROCMON_ANALYSIS

## ProcMon Availability

ProcMon was not found in:

- PATH
- `C:\Sysinternals`
- `C:\Tools`
- `C:\Users\WebVajhegan\Downloads`
- `C:\Users\WebVajhegan\Desktop`

Built-in tools available:

- `wpr.exe`
- `logman.exe`
- `tracerpt.exe`
- `wevtutil.exe`
- `auditpol.exe`

## What Was Not Proven

The investigation did not capture a kernel-level first successful `CreateFile`/`ReadFile` event for:

- `%USERPROFILE%\.codex\auth.json`
- profile-local `auth.json`
- Electron cookies/IndexedDB/local storage
- package LocalState
- DPAPI/credential APIs

Therefore, the exact first authentication read is not directly observed.

## What Was Proven Without ProcMon

The following facts were proven by process/environment/filesystem/CLI evidence:

- BAT and Aion both pass `--user-data-dir` correctly.
- BAT and Aion both set `CODEX_HOME` correctly for primary and app-server.
- BAT and Aion both spawn a dedicated app-server.
- BAT and Aion fresh profiles both create backend databases.
- BAT and Aion fresh profiles both do not create `auth.json`.
- Official Codex CLI reports both fresh profiles as `Not logged in`.
- Official Codex CLI reports global `%USERPROFILE%\.codex` as logged in.

## Evidence Gap

To prove the first authentication read, run ProcMon with filters:

```text
Process Name is Codex.exe OR codex.exe
Operation is CreateFile OR ReadFile OR QueryOpen OR Load Image OR RegOpenKey OR RegQueryValue
Path contains auth.json OR .codex OR Cookies OR IndexedDB OR Local Storage OR Session Storage OR Local State OR Login Data OR Web Data
Result is SUCCESS
```

The first successful auth-related read should be added to `FINAL_ROOT_CAUSE.md`.

## Conclusion

ProcMon-level evidence is still missing. The current root cause conclusion is high-confidence but not kernel-trace complete.

Confidence: **1.00**
