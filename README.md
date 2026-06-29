<div align="center">

<h1>⚡ Aion</h1>

**Research platform for OpenAI Codex Desktop profile management and startup analysis**

<br>

[![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-0078D4?style=flat-square&logo=windows&logoColor=white)](#requirements)
[![Tauri](https://img.shields.io/badge/Tauri-v2-24C8D8?style=flat-square&logo=tauri&logoColor=white)](https://tauri.app)
[![Rust](https://img.shields.io/badge/Rust-stable-CE422B?style=flat-square&logo=rust&logoColor=white)](https://www.rust-lang.org)
[![React](https://img.shields.io/badge/React-Vite-61DAFB?style=flat-square&logo=react&logoColor=black)](https://vitejs.dev)
[![Status](https://img.shields.io/badge/status-Research%20Prototype-orange?style=flat-square)](#current-status)

<br>

[Why Aion](#-why-aion) •
[Current Status](#-current-status) •
[Features](#-features) •
[Architecture](#-architecture) •
[Research](#-research-methodology) •
[Contributing](#-contributing)

</div>

---

## 📌 Current Status

> [!IMPORTANT]
> **Research Prototype** — Complete multi-profile isolation has not yet been achieved.

Aion is an experimental Windows application built to investigate whether OpenAI Codex Desktop can be executed as multiple isolated profiles **without modifying Codex itself**.

The project successfully implements profile management, launch orchestration, runtime monitoring, authentication helpers, and extensive diagnostic tooling.

After extensive reverse engineering and forensic analysis, evidence suggests that additional internal startup behavior inside Codex Desktop affects profile isolation beyond the currently documented configuration layers.

Rather than hiding this result, Aion documents the investigation so future contributors can continue from an advanced starting point.

---

## 💡 Why Aion

OpenAI Codex Desktop stores much of its runtime state in locations designed around a **single user profile**. For developers managing multiple OpenAI accounts, client environments, testing scenarios, or research workflows, this creates practical limitations.

Initially, Aion was created to solve that problem. As development progressed, it became clear that the documented launch parameters alone were insufficient to guarantee independent Codex Desktop sessions.

Instead of abandoning the project, Aion evolved into two complementary goals:

- 🗂️ A practical **Windows profile manager** for Codex Desktop
- 🔍 An open-source **reverse engineering effort** documenting the application's startup behavior

Today, the repository contains both production code and a large collection of forensic documentation describing every investigated hypothesis.

---

## ✅ What Aion Currently Provides

| Category        | Capabilities                                                                         |
| --------------- | ------------------------------------------------------------------------------------ |
| 🗂️ **Profile**  | Profile management · Isolated directories · Isolated launch configuration            |
| 🔐 **Security** | Authentication helpers · Per-profile proxy configuration                             |
| 📊 **Runtime**  | Process monitoring · Process management · Batch operations                           |
| 🔬 **Research** | Forensic tooling · Startup diagnostics · Reproducible experiments · RE documentation |

> [!WARNING]
> Aion does **not** claim to provide guaranteed isolation between multiple OpenAI accounts. Current evidence indicates that additional behavior inside Codex Desktop influences startup and authentication.

---

## 🔍 Current Findings

### ✅ Confirmed

| #   | Finding                                                                             |
| --- | ----------------------------------------------------------------------------------- |
| 1   | `CODEX_HOME` controls Codex backend state                                           |
| 2   | `--user-data-dir` controls Chromium user data                                       |
| 3   | Electron initializes its own `userData` directory **before** the backend is started |
| 4   | Codex Desktop performs additional startup work before launching its backend process |
| 5   | The startup sequence can terminate before the backend is created                    |
| 6   | Aion correctly configures every documented launch parameter currently known         |

### 🧪 Investigated

<details>
<summary>View all investigated areas</summary>

<br>

| Area                      | Notes                             |
| ------------------------- | --------------------------------- |
| Electron startup          | Analyzed initialization sequence  |
| Backend launch            | Documented process creation       |
| Process trees             | Parent-child relationship mapping |
| Authentication flow       | Credential propagation tracing    |
| Filesystem activity       | Runtime artifact comparison       |
| Environment variables     | Propagation and override analysis |
| Runtime artifacts         | Cross-launch artifact diffing     |
| AppModel package behavior | Windows package identity analysis |
| Windows process creation  | API-level process tracing         |
| Startup timing            | Sequencing and race conditions    |
| Launch failures           | Failure mode classification       |

Multiple hypotheses were investigated, reproduced, and documented. Several were ruled out through controlled experiments.

</details>

### ❓ Still Unknown

> [!CAUTION]
> The final condition responsible for preventing reliable multi-profile isolation has **not yet been identified**.

Current evidence suggests that additional internal behavior inside Codex Desktop influences startup in ways that are not publicly documented. Future contributors may be able to identify the remaining missing component using the forensic reports included in this repository.

---

## 🚀 Features

### Production Ready

| Feature                             | Status | Description                                                                   |
| ----------------------------------- | :----: | ----------------------------------------------------------------------------- |
| Profile management                  |   ✅   | Create, rename, clone and delete profiles backed by UUIDs                     |
| Codex executable discovery          |   ✅   | Locate Codex Desktop automatically via Windows registry and AppModel metadata |
| Manual executable selection         |   ✅   | Override automatic discovery with a custom executable                         |
| Crash-safe configuration storage    |   ✅   | Atomic JSON writes with temporary files and replacement                       |
| Runtime process monitoring          |   ✅   | Track running Codex processes from inside Aion                                |
| Runtime FSM                         |   ✅   | `Idle → Launching → Running → Exited` state transitions                       |
| Window management                   |   ✅   | Focus, rename and terminate managed instances                                 |
| Launch orchestration                |   ✅   | Launch profiles with isolated runtime configuration                           |
| Profile-specific launch environment |   ✅   | Configure environment variables independently per launch                      |
| Per-profile proxy configuration     |   ✅   | Optional proxy configuration stored with each profile                         |
| Batch operations                    |   ✅   | Launch or terminate multiple managed profiles                                 |
| Reverse engineering reports         |   ✅   | Extensive forensic documentation included in the repository                   |

### 🧪 Experimental

> [!NOTE]
> These capabilities are implemented for research purposes. They work as intended from Aion's perspective, but do **not** currently guarantee complete Codex Desktop isolation.

| Feature                      | Notes                                             |
| ---------------------------- | ------------------------------------------------- |
| `CODEX_HOME` isolation       | Configured independently for every profile        |
| Electron user-data isolation | Uses `--user-data-dir`                            |
| Authentication workflow      | Separate profile authentication experiments       |
| Startup tracing              | Documents Codex Desktop startup sequence          |
| Backend startup analysis     | Documents app-server initialization               |
| Environment comparison       | Compare successful and failed launches            |
| Filesystem tracing           | Compare runtime artifacts between launches        |
| Process tree analysis        | Analyze parent-child relationships during startup |

### 🔬 Reverse Engineering Toolkit

<details>
<summary>View full investigation topic list</summary>

<br>

The repository contains a large collection of investigation reports covering:

- Startup pipeline reconstruction
- Electron initialization
- Backend launch sequence
- Authentication flow
- Environment propagation
- Process creation
- Filesystem analysis
- Runtime artifact comparison
- Launch timing
- Failure analysis
- Hypothesis testing

These documents are intended for developers interested in understanding Codex Desktop internals or continuing the investigation.

</details>

---

## 📸 Screenshots

### Dashboard

<p align="center">
  <img src="https://github.com/user-attachments/assets/a4faf236-568f-45d4-8dd0-a380527607f7" width="100%">
</p>

> The main workspace for managing profiles, monitoring runtime state, launching Codex Desktop, and reviewing profile configuration.

### Authentication

<p align="center">
  <img src="https://github.com/user-attachments/assets/d4da9dc1-ced6-432a-b63d-3180a7231985" width="430">
</p>

> Profiles can initiate an independent authentication workflow. The current implementation investigates how Codex Desktop creates and stores authentication state for each launch configuration. Authentication behavior remains one of the primary research topics of this repository.

---

## 🏗️ Architecture

Aion separates four independent responsibilities: **persistent profile storage**, **runtime state**, **operating system integration**, and **research instrumentation**. Keeping these independent allows new experiments to be added without changing the core profile manager.

```
┌──────────────────────────────────────────────────────────────┐
│              React · Zustand · Tailwind CSS                 │
│                   Frontend Interface                        │
└──────────────────────┬───────────────────────────────────────┘
                       │
                 Tauri IPC Commands
                       │
┌──────────────────────▼───────────────────────────────────────┐
│                    Rust Backend                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   Profile Management    Runtime Management    Storage        │
│   Process Discovery     Windows Integration   Launch Orch.   │
│                                                              │
└──────────┬────────────────────────────────────┬─────────────┘
           │                                    │
    Windows APIs                        Codex Desktop
           │                                    │
           └───────────────┬────────────────────┘
                           │
                  Research & Diagnostics
```

---

## 🔩 Backend Architecture

### `profile.rs` — Domain Model

Owns the application's domain model.

- Profile definitions and configuration storage
- Runtime metadata and serialization
- Profile state machine: `Idle → Launching → Running → Exited`
- Atomic configuration writes (temp file + replace)

> Persistent data and runtime state are intentionally separated. Profile configuration exists **on disk**. Runtime information exists **only in memory**.

---

### `utils.rs` — OS Boundary

Owns every interaction with Windows.

- Windows registry discovery and executable validation
- Codex process launching and runtime process inspection
- Window management and filesystem helpers
- Environment construction

> Every interaction with Windows APIs is isolated inside this module.

---

### `commands.rs` — IPC Boundary

Acts as the thin boundary between the frontend and backend.

```
Validate parameters → Delegate work → Convert errors to user messages
```

> Business logic does not live here.

---

## 🖥️ Frontend Architecture

The frontend is intentionally lightweight. Its responsibilities are limited to:

- Rendering UI and displaying runtime state
- Dispatching IPC commands
- Polling runtime information

No operating-system logic exists inside React components.

**State Management** — Zustand acts as the single source of truth, owning the profile collection, runtime information, active selections, loading states, and polling. UI components remain almost entirely presentation-only.

**Runtime Model** — Runtime state is intentionally ephemeral. No running-process information is persisted. Every application restart reconstructs runtime state by inspecting the OS rather than restoring previous memory, avoiding stale information after crashes.

---

## 💾 Storage Layout

**Application configuration:**

```
%APPDATA%\Aion
│
├── config
│   ├── app.json
│   ├── profiles
│   └── runtime
│
└── logs
```

**Codex profile directories:**

```
%USERPROFILE%\CodexProfiles
│
├── profile-1
├── profile-2
├── profile-3
└── ...
```

Each profile directory is intended to become a self-contained Codex environment. Whether every component of Codex Desktop respects this separation is one of the central questions investigated by this project.

---

## 🚀 Launch Strategy

Every profile launch attempts to isolate the runtime using the currently documented mechanisms.

| Parameter                      | Purpose                      |
| ------------------------------ | ---------------------------- |
| `CODEX_HOME`                   | Backend state isolation      |
| `NODE_REPL_TRUSTED_CODE_PATHS` | Node environment isolation   |
| `--user-data-dir`              | Chromium user data isolation |

> [!CAUTION]
> Current evidence indicates these configuration layers alone are **not sufficient** to guarantee complete isolation.

---

## 🔬 Research Methodology

One objective of Aion is reproducibility. Every conclusion is based on reproducible experiments rather than assumptions.

The investigation relied on:

| Method                        | Purpose                            |
| ----------------------------- | ---------------------------------- |
| Controlled launch experiments | Baseline isolation testing         |
| Filesystem comparison         | Artifact delta analysis            |
| Startup timing                | Race condition investigation       |
| Process inspection            | Runtime state observation          |
| Electron startup analysis     | Pre-backend initialization tracing |
| Backend artifact comparison   | State diff across launches         |
| Environment tracing           | Variable propagation verification  |
| Windows process monitoring    | API-level event capture            |

Whenever possible, competing hypotheses were tested directly and documented. Several popular assumptions were eliminated during this investigation.

---

## 📋 Research Reports

<details>
<summary>View all research topics</summary>

<br>

The repository contains numerous forensic reports covering:

| Topic                   | Description                         |
| ----------------------- | ----------------------------------- |
| Authentication flow     | Credential creation and storage     |
| Backend startup         | Process initialization sequence     |
| Electron startup        | Pre-backend initialization          |
| Startup pipeline        | End-to-end launch sequence          |
| Environment propagation | Variable inheritance mapping        |
| Runtime artifacts       | Files created during launch         |
| Filesystem differences  | Cross-launch artifact comparison    |
| Process trees           | Parent-child relationships          |
| Launch failures         | Failure classification and triggers |
| Root cause analysis     | Investigation synthesis             |

> Negative results are intentionally preserved. Knowing what does **not** explain the observed behavior is often just as valuable as knowing what does.

</details>

---

## ⚠️ Current Limitations

> [!WARNING]
> Aion should be viewed as a **research prototype** rather than a production-ready profile isolation tool.

<details>
<summary>View all known limitations</summary>

<br>

**Multi-account isolation**
Complete isolation between multiple OpenAI accounts has not been demonstrated. Although Aion configures every documented launch parameter currently known, Codex Desktop still exhibits startup behavior that is not yet fully understood.

**Authentication**
Profile-specific authentication experiments are implemented. However, authentication behavior remains partially controlled by internal Codex Desktop startup logic that has not yet been completely reverse engineered.

**Electron startup**
Investigation confirmed that Electron performs significant initialization before the backend process is created. Current evidence indicates this stage plays an important role in runtime behavior.

**Backend initialization**
The Codex backend process is created only after several earlier startup stages complete. Some failed launches terminate before backend initialization occurs. The exact condition responsible for these failures remains unknown.

**Internal behavior**
Several components of Codex Desktop appear to rely on undocumented internal behavior. Future versions may therefore change launch characteristics without notice.

**Windows only**
Aion currently targets Windows exclusively. Support for Linux or macOS is outside the current scope.

</details>

---

## 💡 What Has Been Learned

Even though the original objective has not yet been fully achieved, the project produced several useful findings.

| Finding                                                          |
| ---------------------------------------------------------------- |
| How Codex Desktop launches its backend                           |
| How Electron initializes user data                               |
| How runtime artifacts are created                                |
| Where backend state is stored                                    |
| How launch parameters propagate                                  |
| Where documented isolation boundaries exist                      |
| Which commonly suggested approaches do **not** solve the problem |

These findings significantly reduce the search space for future investigation.

---

## 🔭 Future Work

Possible directions for future investigation:

- Deeper Electron instrumentation
- Backend process tracing
- Startup event interception
- Windows API monitoring
- Comparison across Codex Desktop releases
- Automated launch experiments
- Improved diagnostic tooling

Contributions in these areas are particularly valuable.

---

## ⚙️ Requirements

| Requirement             | Notes             |
| ----------------------- | ----------------- |
| Windows 10 / 11         | Required          |
| Rust (stable)           | MSVC toolchain    |
| Node.js 20+             | Required          |
| npm                     | Required          |
| Microsoft Edge WebView2 | Required by Tauri |
| OpenAI Codex Desktop    | Installed locally |

---

## 🛠️ Building

```powershell
# Clone and build
git clone https://github.com/<Ali-hey-0>/Aion.git
cd Aion

npm install
npm run tauri build
```

**Development mode:**

```powershell
npm run tauri dev
```

---

## 👩‍💻 Development

The project is organized so experimental work can be added without affecting the core application.

**Recommended workflow:**

```
1. Verify launch behavior
2. Collect evidence
3. Compare successful and failed launches
4. Update documentation
5. Only then modify implementation
```

> Conclusions should be based on reproducible experiments rather than assumptions.

---

## 🤝 Contributing

Contributions are welcome. The most valuable contributions are not necessarily new features, but **new evidence**.

**Valuable pull requests include:**

- Reproducible experiments and startup traces
- ProcMon captures and Electron analysis
- Backend behavior analysis
- Documentation improvements
- Validation of existing hypotheses

> [!IMPORTANT]
> Please distinguish clearly between **observations**, **experimental results**, **hypotheses**, and **conclusions**. Doing so keeps the investigation scientifically reproducible.

---

## ❓ FAQ

<details>
<summary><strong>Does Aion modify Codex Desktop?</strong></summary>

<br>

No. The project intentionally avoids modifying or patching Codex Desktop.

</details>

<details>
<summary><strong>Does Aion inject code?</strong></summary>

<br>

No. No DLL injection, binary patching, or runtime modification is performed.

</details>

<details>
<summary><strong>Does Aion guarantee isolated accounts?</strong></summary>

<br>

No. That was the original objective. Current evidence indicates that additional undocumented behavior inside Codex Desktop affects profile isolation.

</details>

<details>
<summary><strong>Why publish if the original goal wasn't fully achieved?</strong></summary>

<br>

Because the investigation itself has value. The repository documents months of reverse engineering work, eliminates several incorrect assumptions, and provides a foundation for future research. Publishing both successful and unsuccessful experiments avoids duplicated effort and enables other developers to continue from a much more advanced starting point.

</details>

<details>
<summary><strong>Can this project become fully functional in the future?</strong></summary>

<br>

Possibly. The remaining obstacle appears to be understanding a small portion of Codex Desktop's startup behavior. If that behavior can be identified, the existing architecture should already be capable of supporting complete profile isolation.

</details>

---

## 📜 License

The license will be selected before the first stable public release.

---

## 🙏 Acknowledgements

This project would not exist without the work of [Tauri](https://tauri.app), [Rust](https://www.rust-lang.org), [React](https://react.dev), and [Electron](https://www.electronjs.org).

Special thanks to everyone who contributes evidence, experiments, and analysis that help improve our understanding of Codex Desktop's startup behavior.

---

<div align="center">

**Aion** — Investigating the boundaries of Codex Desktop

</div>
