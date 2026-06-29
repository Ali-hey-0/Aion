# PROCESS_TREE

## Evidence Files

- Process snapshot: `C:\Users\WebVajhegan\AppData\Roaming\Aion\config\logs\manual-forensics\20260629-050159-current-codex-processes.json`
- Environment/process snapshot: `C:\Users\WebVajhegan\AppData\Roaming\Aion\config\logs\manual-forensics\20260629-050456-psutil-env-processes.json`

## Controlled Profiles

| Launch | Profile home |
| --- | --- |
| Legacy BAT pattern | `C:\Users\WebVajhegan\CodexProfiles\AionForensicBat-20260629-045454` |
| Aion script pattern | `C:\Users\WebVajhegan\CodexProfiles\AionForensicAion-20260629-044559` |

## Existing Global Codex Tree

| Process | PID | Parent PID | Command line |
| --- | ---: | ---: | --- |
| `Codex.exe` | 8928 | 8916 | `"C:\Program Files\WindowsApps\OpenAI.Codex_26.623.5546.0_x64__2p2nqsd0c76g0\app\Codex.exe"` |
| `resources\codex.exe app-server` | 10192 | 8928 | `"C:\Program Files\WindowsApps\OpenAI.Codex_26.623.5546.0_x64__2p2nqsd0c76g0\app\resources\codex.exe" app-server --analytics-default-enabled` |

## Legacy BAT Pattern Tree

| Process | PID | Parent PID | Relevant command line |
| --- | ---: | ---: | --- |
| Primary `Codex.exe` | 1276 | 13592 | `--user-data-dir="C:\Users\WebVajhegan\CodexProfiles\AionForensicBat-20260629-045454"` |
| `resources\codex.exe app-server` | 4860 | 1276 | `app-server --analytics-default-enabled` |
| Crashpad | 12976 | 1276 | `--user-data-dir=C:\Users\WebVajhegan\CodexProfiles\AionForensicBat-20260629-045454` |
| GPU process | 14076 | 1276 | `--user-data-dir="C:\Users\WebVajhegan\CodexProfiles\AionForensicBat-20260629-045454"` |
| Network service | 6792 | 1276 | `--user-data-dir="C:\Users\WebVajhegan\CodexProfiles\AionForensicBat-20260629-045454"` |
| Storage service | 15060 | 1276 | `--user-data-dir="C:\Users\WebVajhegan\CodexProfiles\AionForensicBat-20260629-045454"` |

## Aion Script Pattern Tree

| Process | PID | Parent PID | Relevant command line |
| --- | ---: | ---: | --- |
| Primary `Codex.exe` | 11544 | 11092 | `--user-data-dir="C:\Users\WebVajhegan\CodexProfiles\AionForensicAion-20260629-044559"` |
| `resources\codex.exe app-server` | 17068 | 11544 | `app-server --analytics-default-enabled` |
| Crashpad | 20156 | 11544 | `--user-data-dir=C:\Users\WebVajhegan\CodexProfiles\AionForensicAion-20260629-044559` |
| GPU process | 10588 | 11544 | `--user-data-dir="C:\Users\WebVajhegan\CodexProfiles\AionForensicAion-20260629-044559"` |
| Network service | 5416 | 11544 | `--user-data-dir="C:\Users\WebVajhegan\CodexProfiles\AionForensicAion-20260629-044559"` |
| Storage service | 15644 | 11544 | `--user-data-dir="C:\Users\WebVajhegan\CodexProfiles\AionForensicAion-20260629-044559"` |

## Conclusion

The controlled BAT and Aion launch patterns are process-tree equivalent for the account-isolation-relevant processes:

- Both create a primary `Codex.exe` with the requested profile path.
- Both create a dedicated `resources\codex.exe app-server` under that primary process.
- Both propagate `--user-data-dir` to Chromium child processes.

Confidence: **0.95**

Updated parity note: Aion no longer writes `aion-launch.cmd` inside `CODEX_HOME`. Launch scripts are generated outside the profile directory, so this earlier script-location delta is removed.

Confidence: **0.90**
