# Final Root Cause

Run root: C:\Users\WebVajhegan\AppData\Roaming\Aion\config\logs\bat-vs-aion\20260629-053543

## Status

Root cause is not proven unless this run contains both BAT and Aion app-server captures and a post-login filesystem/auth timeline for both profiles.

## Evidence Captured In This Run

- BAT process records: 38
- Aion process records: 0
- BAT app-server captured: True
- Aion app-server captured: False
- BAT filesystem events: 957
- Aion filesystem events: 0
- Auth-candidate events: 72

## Current Forensic Finding

The launcher cannot be convicted or cleared by command-line inspection alone. The BAT baseline demonstrates that the Desktop app writes authentication-adjacent Chromium state under the profile directory before sign-in. Therefore the next required evidence is the login-completion write timeline and the matching Aion write timeline.

## Missing Evidence If Aion Records Are Zero

The sidecar did not observe an actual Aion UI launch for profile AionLive-20260629-053543. Run this from the project root:

powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\forensics\bat-vs-aion-live.ps1 -Mode interactive -KeepProfiles

Then launch the generated Aion profile from the Aion UI while the script is waiting.

## Release Decision

NO. Aion is not ready for public release until the first post-login divergence is captured and explained with evidence.