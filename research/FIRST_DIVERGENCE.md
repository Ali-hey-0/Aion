# First Observable Divergence

## Question

Why does the exact same code appear to work when launched by Codex Agent but fail when launched manually with `npm run tauri dev`?

## Directly Proven

The following are proven by current evidence:

1. The debug binary reached through `G:\others\Aion-Codex` and `C:\Users\WebVajhegan\Desktop\vs code\Aion` is the same file.
2. No release binary exists, so the failure is not explained by debug vs release.
3. npm lifecycle environment differs from the Agent shell environment.
4. The launcher clears npm-specific variables before invoking Codex.
5. The current implementation returns launch success after detecting only primary `Codex.exe`.
6. Failed profile folders exist with Chromium artifacts but no backend artifacts.

## Earliest Confirmed Divergence

The earliest confirmed divergence is:

```text
primary Codex.exe detected
```

After that point:

- Successful launches continue to `resources\codex.exe app-server` and backend artifact creation.
- Failed launches stop after Chromium bootstrap artifacts and never create backend artifacts.

## Why The User Sees The Global Account

The global Codex desktop process is already running:

```text
Codex.exe PID 7940
resources\codex.exe app-server PID 13392
```

When an isolated launch only gets as far as Chromium bootstrap and does not initialize its backend, the visible stable Codex window can be the already-running global instance. Aion may still report the selected profile as running because it saw a matching primary `Codex.exe` earlier.

This explains the symptom without requiring a new authentication hypothesis.

## What Is Not Proven

It is not proven from the current capture that Agent and manual `npm run tauri dev` produce different Codex child environments. There was no live manual `aion.exe` process during capture.

It is also not proven that npm-specific variables reach Codex, because current source code explicitly clears them.

## First Divergence Confidence

Backend readiness divergence: high.

Agent-vs-manual environment as the root cause: low-to-medium with current evidence.
