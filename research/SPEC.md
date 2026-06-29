# Aion (Codex Profile Manager) - Functional Spec

## Core Features (MVP)
1. CRUD Profiles (Stored in AppData/Roaming/Aion/config.json).
2. Launch Codex with dynamic `--user-data-dir`.
3. Detect running instances via process argument parsing.
4. Auto-detect Codex.exe via Windows Registry (AppModel/Repository/Packages).

## Data Schema (Rust struct & TypeScript Interface)
Profile {
    id: String (UUID v4),
    name: String,
    color_tag: String (Hex),
    created_at: u64,
    last_launched: Option<u64>
}