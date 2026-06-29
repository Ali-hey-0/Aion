# REGISTRY_TRACE

## Evidence Files

- Registry snapshot: `C:\Users\WebVajhegan\AppData\Roaming\Aion\config\logs\manual-forensics\20260629-050835-registry-openai-codex.txt`
- Credential snapshot: `C:\Users\WebVajhegan\AppData\Roaming\Aion\config\logs\manual-forensics\20260629-050835-credentials-after.txt`

## Registry Areas Checked

- `HKCU:\Software\OpenAI`
- `HKCU:\Software\Classes\Local Settings\Software\Microsoft\Windows\CurrentVersion\AppModel\Repository\Packages`
- `HKCU:\Software\Microsoft\Windows\CurrentVersion\Authentication`
- `HKCU:\Software\Microsoft\IdentityCRL`
- Windows uninstall metadata in earlier traces

## Findings

The registry evidence found Codex/OpenAI package/install metadata, but no verified authentication token/session source.

No evidence was found that the account identity comes from a Codex-specific registry value.

## Credential Manager

`cmdkey /list` was captured. No Codex/OpenAI credential-manager target was identified as the account source in this pass.

## Conclusion

Registry and Windows Credential Manager are not supported by current evidence as the primary Codex account source.

Confidence: **0.75**

This does not fully exclude DPAPI, Web Account Manager, MSAL, or Electron/native API credential access because those require ProcMon/API/ETW-level tracing to prove negative.

Confidence: **1.00**
