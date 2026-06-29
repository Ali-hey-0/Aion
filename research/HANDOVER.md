# Aion Technical Handover

## Current System State

Aion is a Tauri v2 desktop application for managing isolated Codex desktop profiles. Each profile is represented by durable account/configuration metadata in JSON and a dedicated `--user-data-dir` folder under the user roaming profile.

Persistent storage lives at:

```text
%APPDATA%\Aion\config.json
%APPDATA%\Aion\profiles\<profile_uuid>\
```

The persistent config layer is intentionally separate from live process state:

- `ProfileManager` owns static profile CRUD and all access to `config.json`.
- `FileConfigStore` implements `ConfigStore` and performs atomic JSON writes through a temporary file replacement.
- `config.lock` serializes read-modify-write config mutations without relying on Tauri managed `Mutex` or `RwLock` guards during filesystem work.
- `RuntimeManager` owns only in-memory runtime state as `Arc<RwLock<HashMap<Uuid, RuntimeState>>>`.
- `CodexProcessProvider` implements `ProcessProvider` and owns Codex executable resolution, launch, and process inspection.

This means the JSON file is the source of truth for durable profiles, while the runtime map is a lightweight cache of currently observed Codex processes.

## Structural Boundaries

### Durable Config Schema

`config.json` uses a versioned root object:

```json
{
  "schema_version": 1,
  "custom_codex_path": "C:\\Optional\\Path\\Codex.exe",
  "profiles": [
    {
      "id": "00000000-0000-0000-0000-000000000000",
      "name": "Primary",
      "email": "account@example.com",
      "color_tag": "#4F46E5",
      "created_at": 1782580000,
      "last_launched": null,
      "usage_week_hours": 0.0,
      "usage_5h_hours": 0.0,
      "activated_at": null,
      "expires_at": null
    }
  ]
}
```

`Profile.id` is a typed UUID in Rust and serializes as a string for JSON and IPC.

### Runtime State Schema

Runtime state is not written to `config.json`.

```json
{
  "pid": 1234,
  "status": "Running",
  "is_focused": false,
  "started_at": "2026-06-27T14:42:00Z"
}
```

Allowed `ProcessStatus` values are:

- `Idle`
- `Running`
- `Exited`
- `Unknown`

### IPC Endpoints

All commands return `Result<T, String>` on the Rust side.

`create_profile`

Input:

```json
{
  "name": "Primary",
  "email": "account@example.com",
  "colorTag": "#4F46E5"
}
```

Output: full `ProfileView`.

`list_profiles`

Input: none.

Output: full array of `ProfileView` objects, including account metadata and telemetry fields. This endpoint is for initial load and CRUD refreshes, not high-frequency polling.

`get_runtime_statuses`

Input: none.

Output:

```json
[
  {
    "id": "00000000-0000-0000-0000-000000000000",
    "status": "Running",
    "pid": 1234,
    "is_focused": false
  }
]
```

This is the 2-second polling endpoint used by Zustand. It scans `Codex.exe` processes and matches `--user-data-dir` arguments against managed profile directories.

`launch_profile`

Input:

```json
{
  "profileId": "00000000-0000-0000-0000-000000000000"
}
```

Output: full `ProfileView` with running state updated after launch.

`rename_profile`

Input:

```json
{
  "profileId": "00000000-0000-0000-0000-000000000000",
  "newName": "Work Account"
}
```

Output: full `ProfileView`.

`clone_profile`

Input:

```json
{
  "profileId": "00000000-0000-0000-0000-000000000000",
  "newName": "Work Account Copy"
}
```

`newName` may be `null`.

Output: full `ProfileView`.

`delete_profile`

Input:

```json
{
  "profileId": "00000000-0000-0000-0000-000000000000"
}
```

Output: `null` on success. Running profiles are rejected.

`set_custom_codex_path`

Input:

```json
{
  "codexPath": "C:\\Path\\To\\Codex.exe"
}
```

Output: canonicalized path string.

### Frontend State Flow

`src/store/useProfileStore.js` performs:

- Initial `list_profiles`.
- Recurring `get_runtime_statuses` every 2 seconds.
- Full profile refresh through direct command responses after create, rename, clone, delete, and launch.

The store normalizes Rust snake_case response fields into frontend camelCase fields.

## Security and Reliability Notes

- Codex discovery uses the Windows registry under the current user's AppModel package repository.
- Custom executable paths are canonicalized and must resolve to a real `Codex.exe` file.
- Executable and profile data paths reject `..` traversal.
- Executable and profile data paths reject symlinks, junctions, and Windows reparse points.
- Profile clone refuses symlinked data entries.
- Runtime locks are held only for in-memory map mutation/read, never while scanning processes, spawning Codex, or performing profile filesystem operations.
- Config mutations are serialized by `config.lock` and written atomically.

## Next Evolution Steps

1. Add a dedicated `focus_profile` command that uses Win32 window enumeration to focus the matching Codex process by PID.
2. Persist runtime accounting events rather than projecting usage from `last_launched`; a future `usage_events` table/file should track start/stop intervals.
3. Add explicit `update_profile_metadata` for `activated_at` and `expires_at` instead of editing JSON manually.
4. Add frontend controls for account lifecycle dates once metadata editing exists.
5. Add capability files for Tauri v2 permissions if the application begins using filesystem or shell plugins from the frontend.
6. Add unit tests around config parsing, path validation, and runtime status merging.
7. Consider a background watcher thread for runtime refresh if process polling from IPC becomes too frequent for larger profile counts.

## Development Constraints

- Use `cargo check` for backend validation during iterative work.
- Do not use `cargo build` or `cargo tauri build` unless explicitly entering release validation.
- Preserve the separation between persistent profile data and runtime process state.
- Keep command payloads camelCase on the frontend because Tauri maps them to Rust snake_case arguments.

