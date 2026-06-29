# Aion - Technical Architecture

- **Backend**: Rust + Tauri v2
- **Frontend**: React (Vite) + TailwindCSS + shadcn/ui
- **State Management**: Zustand (Frontend) / File-based JSON (Backend)
- **Process Execution**: `std::process::Command` with detached process flags on Windows (`CREATE_NO_WINDOW` or standard spawning).