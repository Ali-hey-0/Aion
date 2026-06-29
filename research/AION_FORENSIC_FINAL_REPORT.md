# AION_FORENSIC_FINAL_REPORT

## Executive Summary

The controlled investigation did not find a material BAT-vs-Aion launcher difference. Both launch methods correctly create isolated Chromium/runtime state and both pass the correct `CODEX_HOME` and `--user-data-dir`.

The actual failure mode is authentication lifecycle: fresh profiles do not get a local `auth.json` from Desktop launch. The official Codex CLI reports those fresh profiles as `Not logged in`, while global `%USERPROFILE%\.codex` is logged in. Therefore any signed-in UI shown by those fresh profiles is coming from outside the profile-local auth state.

## Deliverables

- `PROCESS_TREE.md`
- `ENVIRONMENT_DIFF.md`
- `FILESYSTEM_DIFF.md`
- `AUTH_FLOW.md`
- `REGISTRY_TRACE.md`
- `PROCMON_ANALYSIS.md`
- `FINAL_ROOT_CAUSE.md`

## Key Evidence

| Evidence | Result |
| --- | --- |
| BAT command line | Correct `--user-data-dir=<BAT profile>` |
| Aion command line | Correct `--user-data-dir=<Aion profile>` |
| BAT app-server env | Correct `CODEX_HOME=<BAT profile>` |
| Aion app-server env | Correct `CODEX_HOME=<Aion profile>` |
| BAT fresh profile | no `auth.json`; CLI says `Not logged in` |
| Aion fresh profile | no `auth.json`; CLI says `Not logged in` |
| Global profile | has `auth.json`; CLI says `Logged in using ChatGPT` |

## Root Cause Statement

Codex Desktop does not reliably create profile-local authentication during Desktop launch. When profile-local auth is absent, Desktop can still surface the globally logged-in identity from the Windows user context, most likely `%USERPROFILE%\.codex\auth.json` or a cache derived from it.

Confidence: **0.85**

## Why This Explains the User-Observed Difference

The legacy BAT launcher appeared to work because some named BAT profiles already had state/auth or matched the same global identity. Under controlled fresh-profile conditions, BAT and Aion behave the same: both create profile runtime files, neither creates `auth.json`, and both are `Not logged in` according to the official Codex CLI.

Confidence: **0.90**

## Not Yet Proven

The first successful authentication read was not captured because ProcMon was not installed. This prevents a 100% statement such as "the first auth read is exactly `%USERPROFILE%\.codex\auth.json` at timestamp X by PID Y."

## Required Next Evidence For 100% Proof

Install/run ProcMon locally and capture:

```text
Process Name is Codex.exe OR codex.exe
Operation is CreateFile OR ReadFile OR QueryOpen OR RegOpenKey OR RegQueryValue
Result is SUCCESS
Path contains auth.json OR .codex OR Cookies OR IndexedDB OR Local Storage OR Session Storage OR Local State
```

Then record the first successful authentication-related read in `PROCMON_ANALYSIS.md` and `FINAL_ROOT_CAUSE.md`.
