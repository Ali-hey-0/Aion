# Aion Design Philosophy

This document defines the core dogmas governing Aion's engineering and product roadmap. Every Pull Request and Architectural Shift must align with these principles.

### 1. Zero Hidden Magic (Predictability Over Cleverness)
We prefer explicit, boring, and highly predictable Rust code over complex macro-driven magic or clever abstractions. If a junior developer cannot reason about the data flow in 5 minutes, the code is too clever.

### 2. Runtime is Ephemeral, Storage is Absolute
In-memory OS runtime states (PIDs, window handles, tracking loops) must NEVER be persisted to disk. Disk storage (`config.json`, `profiles/`) is strictly reserved for user-defined intent and declarative states.

### 3. Guard the OS Boundaries
All Win32 API calls, process spawning, and filesystem mutations must stay heavily isolated inside the `infrastructure` layer. The core application logic must remain pure and fully testable without requiring mock OS contexts.

### 4. Low Cognitive Load Architecture
Minimalism is a technical constraint, not just a UI trend. Every metric shown, every nested card, and every configuration parameter must justify its existence by solving a real developer pain point. If it introduces visual or conceptual clutter, it gets rejected.

### 5. Mechanical Sympathy for Windows
We build with explicit awareness of Windows filesystem behaviors. We enforce atomic crash-safe writes (`fsync`) and treat process monitoring with adaptive backoffs to maintain a near-zero resource footprint.