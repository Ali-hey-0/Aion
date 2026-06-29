# BAT vs Aion Filesystem Timeline

Run root: C:\Users\WebVajhegan\AppData\Roaming\Aion\config\logs\bat-vs-aion\20260629-053543

## Capture Counts

- BAT filesystem events: 957
- Aion filesystem events: 0
- Auth-candidate events: 72

## First BAT Events

| Observed | Kind | Path | Size |
| --- | --- | --- | --- |
| 2026-06-29T05:35:59.771 | created | `..codex-global-state.json.tmp-1782698748901-27fa3515-c229-4ac2-ad97-5ec37a17a7ff` | 42 |
| 2026-06-29T05:35:59.772 | created | `Default\Account Web Data` | 77824 |
| 2026-06-29T05:35:59.773 | created | `Default\Account Web Data-journal` | 0 |
| 2026-06-29T05:35:59.774 | created | `Default\Affiliation Database` | 53248 |
| 2026-06-29T05:35:59.774 | created | `Default\Affiliation Database-journal` | 0 |
| 2026-06-29T05:35:59.774 | created | `Default\ClientCertificates\LOCK` | 0 |
| 2026-06-29T05:35:59.774 | created | `Default\ClientCertificates\LOG` | 0 |
| 2026-06-29T05:35:59.775 | created | `Default\Extension Rules\000003.log` | 76 |
| 2026-06-29T05:35:59.775 | created | `Default\Extension Rules\CURRENT` | 16 |
| 2026-06-29T05:35:59.775 | created | `Default\Extension Rules\LOCK` | 0 |
| 2026-06-29T05:35:59.776 | created | `Default\Extension Rules\LOG` | 291 |
| 2026-06-29T05:35:59.776 | created | `Default\Extension Rules\MANIFEST-000001` | 41 |
| 2026-06-29T05:35:59.776 | created | `Default\Extension Scripts\000003.log` | 38 |
| 2026-06-29T05:35:59.777 | created | `Default\Extension Scripts\CURRENT` | 16 |
| 2026-06-29T05:35:59.777 | created | `Default\Extension Scripts\LOCK` | 0 |
| 2026-06-29T05:35:59.777 | created | `Default\Extension Scripts\LOG` | 295 |
| 2026-06-29T05:35:59.778 | created | `Default\Extension Scripts\MANIFEST-000001` | 41 |
| 2026-06-29T05:35:59.778 | created | `Default\Extension State\000003.log` | 114 |
| 2026-06-29T05:35:59.778 | created | `Default\Extension State\CURRENT` | 16 |
| 2026-06-29T05:35:59.779 | created | `Default\Extension State\LOCK` | 0 |
| 2026-06-29T05:35:59.779 | created | `Default\Extension State\LOG` | 291 |
| 2026-06-29T05:35:59.779 | created | `Default\Extension State\MANIFEST-000001` | 41 |
| 2026-06-29T05:35:59.780 | created | `Default\Favicons` | 20480 |
| 2026-06-29T05:35:59.780 | created | `Default\Favicons-journal` | 0 |
| 2026-06-29T05:35:59.780 | created | `Default\History` | 163840 |
| 2026-06-29T05:35:59.781 | created | `Default\History-journal` | 0 |
| 2026-06-29T05:35:59.781 | created | `Default\LOCK` | 0 |
| 2026-06-29T05:35:59.781 | created | `Default\LOG` | 0 |
| 2026-06-29T05:35:59.782 | created | `Default\Local Storage\leveldb\000003.log` | 30 |
| 2026-06-29T05:35:59.783 | created | `Default\Local Storage\leveldb\CURRENT` | 16 |
| 2026-06-29T05:35:59.783 | created | `Default\Local Storage\leveldb\LOCK` | 0 |
| 2026-06-29T05:35:59.784 | created | `Default\Local Storage\leveldb\LOG` | 303 |
| 2026-06-29T05:35:59.785 | created | `Default\Local Storage\leveldb\MANIFEST-000001` | 41 |
| 2026-06-29T05:35:59.786 | created | `Default\Login Data` | 40960 |
| 2026-06-29T05:35:59.786 | created | `Default\Login Data For Account` | 40960 |
| 2026-06-29T05:35:59.787 | created | `Default\Login Data For Account-journal` | 0 |
| 2026-06-29T05:35:59.787 | created | `Default\Login Data-journal` | 0 |
| 2026-06-29T05:35:59.788 | created | `Default\Network\Cookies` | 20480 |
| 2026-06-29T05:35:59.789 | created | `Default\Network\Cookies-journal` | 0 |
| 2026-06-29T05:35:59.789 | created | `Default\Network\Device Bound Sessions` | 20480 |

## First Aion Events

| Observed | Kind | Path | Size |
| --- | --- | --- | --- |


## Current Conclusion

The first observable divergence can only be called after both BAT and Aion launches are captured through the same login stage. Baseline-only runs intentionally do not prove authentication persistence.