# Aion backend launch/process review sources

Generated from the current workspace for architectural review. These are the complete, unmodified Rust sources that participate in Tauri command wiring, profile launch, process management, Codex.exe spawning, sandbox path construction, and runtime state.


## src-tauri/src/main.rs

```rust
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod profile;
mod utils;

fn main() {
    let profile_manager = match profile::ProfileManager::new_default() {
        Ok(manager) => manager,
        Err(error) => {
            eprintln!("Aion failed to initialize backend state: {error}");
            std::process::exit(1);
        }
    };
    let runtime_manager = profile::RuntimeManager::new();
    let process_provider = utils::CodexProcessProvider::new();

    let app_result = tauri::Builder::default()
        .manage(profile_manager)
        .manage(runtime_manager)
        .manage(process_provider)
        .invoke_handler(tauri::generate_handler![
            commands::create_profile,
            commands::list_profiles,
            commands::get_runtime_polling_metadata,
            commands::get_runtime_statuses,
            commands::delete_profile,
            commands::rename_profile,
            commands::clone_profile,
            commands::launch_profile,
            commands::focus_profile,
            commands::launch_all_profiles,
            commands::kill_all_active_instances,
            commands::discover_codex_path,
            commands::browse_codex_executable,
            commands::set_custom_codex_path
        ])
        .run(tauri::generate_context!());

    if let Err(error) = app_result {
        eprintln!("Aion runtime failed: {error}");
        std::process::exit(1);
    }
}

```


## src-tauri/src/commands.rs

```rust
use std::path::PathBuf;

use tauri::State;
use uuid::Uuid;

use crate::{
    profile::{
        parse_profile_id, ProfileManager, ProfileView, ProxyConfig, RuntimeManager,
        RuntimePollingMetadata, RuntimeStatusView,
    },
    utils::{CodexPathDiscovery, CodexProcessProvider, KillResult, LaunchOptions, ProcessProvider},
};

#[tauri::command]
pub fn create_profile(
    profiles: State<'_, ProfileManager>,
    name: String,
    email: String,
    color_tag: String,
    proxy: Option<ProxyConfig>,
) -> Result<ProfileView, String> {
    let profile = profiles.create_profile(name, email, color_tag, proxy)?;
    let user_data_dir = profiles.profile_user_data_dir(&profile.id);
    Ok(ProfileView::from_profile(&profile, &user_data_dir, None))
}

#[tauri::command]
pub fn list_profiles(
    profiles: State<'_, ProfileManager>,
    runtime: State<'_, RuntimeManager>,
) -> Result<Vec<ProfileView>, String> {
    let static_profiles = profiles.profiles()?;
    let profile_ids: Vec<Uuid> = static_profiles.iter().map(|profile| profile.id).collect();
    let runtime_by_id = runtime.states_for(&profile_ids)?;

    Ok(static_profiles
        .iter()
        .map(|profile| {
            let user_data_dir = profiles.profile_user_data_dir(&profile.id);
            let runtime_state = runtime_by_id.get(&profile.id);
            ProfileView::from_profile(profile, &user_data_dir, runtime_state)
        })
        .collect())
}

#[tauri::command]
pub fn get_runtime_polling_metadata() -> Result<RuntimePollingMetadata, String> {
    Ok(RuntimePollingMetadata::default())
}

#[tauri::command]
pub fn get_runtime_statuses(
    profiles: State<'_, ProfileManager>,
    runtime: State<'_, RuntimeManager>,
    process_provider: State<'_, CodexProcessProvider>,
) -> Result<Vec<RuntimeStatusView>, String> {
    let profile_dirs = profiles.profile_directories()?;
    let profile_ids: Vec<Uuid> = profile_dirs
        .iter()
        .map(|(profile_id, _)| *profile_id)
        .collect();
    let detected = process_provider.detect_running_profiles(&profile_dirs)?;
    runtime.merge_detected(&profile_ids, detected)
}

#[tauri::command]
pub fn delete_profile(
    profiles: State<'_, ProfileManager>,
    runtime: State<'_, RuntimeManager>,
    process_provider: State<'_, CodexProcessProvider>,
    profile_id: String,
) -> Result<(), String> {
    let parsed_profile_id = parse_profile_id(&profile_id)?;
    let profile_dirs = profiles.profile_directories()?;
    let profile_ids: Vec<Uuid> = profile_dirs
        .iter()
        .map(|(profile_id, _)| *profile_id)
        .collect();
    let detected = process_provider.detect_running_profiles(&profile_dirs)?;
    let statuses = runtime.merge_detected(&profile_ids, detected)?;
    let target_id = parsed_profile_id.to_string();
    let is_active = statuses
        .iter()
        .filter(|status| status.id == target_id)
        .any(|status| status.status.is_active());

    if is_active {
        return Err(format!(
            "Profile '{}' is currently active. Close its Codex window before deleting it.",
            profile_id
        ));
    }

    profiles.delete_profile(&parsed_profile_id)?;
    runtime.remove(&parsed_profile_id)?;
    Ok(())
}

#[tauri::command]
pub fn rename_profile(
    profiles: State<'_, ProfileManager>,
    runtime: State<'_, RuntimeManager>,
    profile_id: String,
    new_name: String,
) -> Result<ProfileView, String> {
    let parsed_profile_id = parse_profile_id(&profile_id)?;
    let profile = profiles.rename_profile(&parsed_profile_id, new_name)?;
    let user_data_dir = profiles.profile_user_data_dir(&profile.id);
    let runtime_state = runtime_state_for(&runtime, &profile.id)?;
    Ok(ProfileView::from_profile(
        &profile,
        &user_data_dir,
        runtime_state.as_ref(),
    ))
}

#[tauri::command]
pub fn clone_profile(
    profiles: State<'_, ProfileManager>,
    profile_id: String,
    new_name: Option<String>,
) -> Result<ProfileView, String> {
    let parsed_profile_id = parse_profile_id(&profile_id)?;
    let profile = profiles.clone_profile(&parsed_profile_id, new_name)?;
    let user_data_dir = profiles.profile_user_data_dir(&profile.id);
    Ok(ProfileView::from_profile(&profile, &user_data_dir, None))
}

#[tauri::command]
pub fn launch_profile(
    profiles: State<'_, ProfileManager>,
    runtime: State<'_, RuntimeManager>,
    process_provider: State<'_, CodexProcessProvider>,
    profile_id: String,
) -> Result<ProfileView, String> {
    let parsed_profile_id = parse_profile_id(&profile_id)?;
    if !profiles.profile_exists(&parsed_profile_id)? {
        return Err(format!("Profile '{}' was not found.", profile_id));
    }

    let user_data_dir = profiles.ensure_profile_user_data_dir(&parsed_profile_id)?;
    let custom_codex_path = profiles.custom_codex_path()?;
    let codex_executable =
        process_provider.resolve_codex_executable(custom_codex_path.as_deref())?;
    let static_profile = profiles
        .profiles()?
        .into_iter()
        .find(|profile| profile.id == parsed_profile_id)
        .ok_or_else(|| format!("Profile '{}' was not found.", profile_id))?;
    let options = LaunchOptions {
        profile_name: static_profile.name.clone(),
        proxy: static_profile.proxy.clone(),
    };
    runtime.mark_launching(parsed_profile_id)?;
    let launch = match process_provider.launch_codex(&codex_executable, &user_data_dir, &options) {
        Ok(launch) => launch,
        Err(error) => {
            let _ = runtime.mark_error(parsed_profile_id, error.clone());
            tracing::error!(profile_id = %parsed_profile_id, "failed to launch Codex profile: {error}");
            return Err(error);
        }
    };
    runtime.mark_running(parsed_profile_id, launch.pid, launch.started_at)?;
    let profile = profiles.mark_profile_launched(&parsed_profile_id)?;
    let mut runtime_states = runtime.states_for(&[parsed_profile_id])?;
    let runtime_state = runtime_states.remove(&parsed_profile_id);

    Ok(ProfileView::from_profile(
        &profile,
        &user_data_dir,
        runtime_state.as_ref(),
    ))
}

#[tauri::command]
pub fn focus_profile(
    profiles: State<'_, ProfileManager>,
    runtime: State<'_, RuntimeManager>,
    process_provider: State<'_, CodexProcessProvider>,
    profile_id: String,
) -> Result<bool, String> {
    let parsed_profile_id = parse_profile_id(&profile_id)?;
    let user_data_dir = profiles.profile_user_data_dir(&parsed_profile_id);
    let detected =
        process_provider.detect_running_profiles(&[(parsed_profile_id, user_data_dir)])?;

    if let Some(state) = detected.get(&parsed_profile_id).cloned() {
        let pid = state.pid();
        runtime.mark_running(parsed_profile_id, pid, state.started_at().to_string())?;
        let focused = process_provider.focus_process_window(pid)?;
        runtime.update_focus(parsed_profile_id, focused)?;
        return Ok(focused);
    }

    let statuses = runtime.statuses_for(&[parsed_profile_id])?;
    let Some(status) = statuses.first() else {
        return Err(format!("Profile '{}' has no runtime status.", profile_id));
    };

    if !status.status.is_running() || status.pid == 0 {
        return Err(format!("Profile '{}' is not running.", profile_id));
    }

    let focused = process_provider.focus_process_window(status.pid)?;
    runtime.update_focus(parsed_profile_id, focused)?;
    Ok(focused)
}

#[tauri::command]
pub fn launch_all_profiles(
    profiles: State<'_, ProfileManager>,
    runtime: State<'_, RuntimeManager>,
    process_provider: State<'_, CodexProcessProvider>,
) -> Result<Vec<ProfileView>, String> {
    let static_profiles = profiles.profiles()?;
    if static_profiles.is_empty() {
        return Ok(Vec::new());
    }

    let profile_dirs = profiles.profile_directories()?;
    let profile_ids: Vec<Uuid> = profile_dirs
        .iter()
        .map(|(profile_id, _)| *profile_id)
        .collect();
    let detected = process_provider.detect_running_profiles(&profile_dirs)?;
    let current_statuses = runtime.merge_detected(&profile_ids, detected)?;
    let running_ids: Vec<Uuid> = current_statuses
        .iter()
        .filter(|status| status.status.is_active())
        .filter_map(|status| Uuid::parse_str(&status.id).ok())
        .collect();

    let custom_codex_path = profiles.custom_codex_path()?;
    let codex_executable =
        process_provider.resolve_codex_executable(custom_codex_path.as_deref())?;

    for profile in &static_profiles {
        if running_ids
            .iter()
            .any(|profile_id| *profile_id == profile.id)
        {
            continue;
        }

        let user_data_dir = profiles.ensure_profile_user_data_dir(&profile.id)?;
        let options = LaunchOptions {
            profile_name: profile.name.clone(),
            proxy: profile.proxy.clone(),
        };
        runtime.mark_launching(profile.id)?;
        let launch = match process_provider.launch_codex(
            &codex_executable,
            &user_data_dir,
            &options,
        ) {
            Ok(launch) => launch,
            Err(error) => {
                let _ = runtime.mark_error(profile.id, error.clone());
                tracing::error!(profile_id = %profile.id, "failed to launch Codex profile: {error}");
                return Err(error);
            }
        };
        runtime.mark_running(profile.id, launch.pid, launch.started_at)?;
        profiles.mark_profile_launched(&profile.id)?;
    }

    profile_views(&profiles, &runtime)
}

#[tauri::command]
pub fn kill_all_active_instances(
    profiles: State<'_, ProfileManager>,
    runtime: State<'_, RuntimeManager>,
    process_provider: State<'_, CodexProcessProvider>,
) -> Result<Vec<KillResult>, String> {
    let profile_dirs = profiles.profile_directories()?;
    let profile_ids: Vec<Uuid> = profile_dirs
        .iter()
        .map(|(profile_id, _)| *profile_id)
        .collect();
    for status in runtime.statuses_for(&profile_ids)? {
        if status.status.is_active() {
            if let Ok(profile_id) = Uuid::parse_str(&status.id) {
                let _ = runtime.mark_stopping(profile_id);
            }
        }
    }
    let killed = process_provider.kill_profile_instances(&profile_dirs)?;
    let detected = process_provider.detect_running_profiles(&profile_dirs)?;
    runtime.merge_detected(&profile_ids, detected)?;
    Ok(killed)
}

#[tauri::command]
pub fn discover_codex_path(
    profiles: State<'_, ProfileManager>,
    process_provider: State<'_, CodexProcessProvider>,
) -> Result<CodexPathDiscovery, String> {
    let custom_codex_path = profiles.custom_codex_path()?;
    process_provider.discover_codex_executable(custom_codex_path.as_deref())
}

#[tauri::command]
pub fn browse_codex_executable(
    process_provider: State<'_, CodexProcessProvider>,
) -> Result<Option<String>, String> {
    Ok(process_provider
        .browse_codex_executable()?
        .map(|path| path.to_string_lossy().into_owned()))
}

#[tauri::command]
pub fn set_custom_codex_path(
    profiles: State<'_, ProfileManager>,
    process_provider: State<'_, CodexProcessProvider>,
    codex_path: String,
) -> Result<String, String> {
    let path = PathBuf::from(codex_path.trim());
    let safe_path = process_provider.validate_codex_executable(&path)?;
    let stored = profiles.set_custom_codex_path(safe_path)?;
    Ok(stored.to_string_lossy().into_owned())
}

fn profile_views(
    profiles: &State<'_, ProfileManager>,
    runtime: &State<'_, RuntimeManager>,
) -> Result<Vec<ProfileView>, String> {
    let static_profiles = profiles.profiles()?;
    let profile_ids: Vec<Uuid> = static_profiles.iter().map(|profile| profile.id).collect();
    let runtime_by_id = runtime.states_for(&profile_ids)?;

    Ok(static_profiles
        .iter()
        .map(|profile| {
            let user_data_dir = profiles.profile_user_data_dir(&profile.id);
            let runtime_state = runtime_by_id.get(&profile.id);
            ProfileView::from_profile(profile, &user_data_dir, runtime_state)
        })
        .collect())
}

fn runtime_state_for(
    runtime: &State<'_, RuntimeManager>,
    profile_id: &Uuid,
) -> Result<Option<crate::profile::RuntimeState>, String> {
    let mut states = runtime.states_for(&[*profile_id])?;
    Ok(states.remove(profile_id))
}

```


## src-tauri/src/utils.rs

```rust
use serde::Serialize;
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Component, Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

use sysinfo::{Process, System};
use uuid::Uuid;

use crate::profile::{iso_timestamp_from_unix, iso_timestamp_now, ProxyConfig, RuntimeState};

const CODEX_EXECUTABLE_NAME: &str = "Codex.exe";
const USER_DATA_DIR_ARG: &str = "--user-data-dir";
const MAX_DISCOVERY_DEPTH: usize = 5;
const MAX_DISCOVERY_ENTRIES: usize = 6_000;
const LAUNCH_PROCESS_DETECTION_RETRIES: usize = 40;
const LAUNCH_PROCESS_DETECTION_DELAY_MS: u64 = 250;
const WINDOW_TITLE_RETRY_COUNT: usize = 40;
const WINDOW_TITLE_RETRY_DELAY_MS: u64 = 250;

#[derive(Debug, Clone, Serialize)]
pub struct CodexPathDiscovery {
    pub path: Option<String>,
    pub source: String,
    pub auto_detected: bool,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct KillResult {
    pub id: String,
    pub pid: u32,
    pub killed: bool,
}

#[derive(Debug, Clone)]
pub struct LaunchOptions {
    pub profile_name: String,
    pub proxy: Option<ProxyConfig>,
}

impl LaunchOptions {
    fn window_title(&self) -> String {
        let mut name = self
            .profile_name
            .chars()
            .filter(|character| !character.is_control())
            .take(120)
            .collect::<String>();
        if name.trim().is_empty() {
            name = "Managed Profile".to_string();
        }
        format!("Aion - {}", name.trim())
    }
}

#[derive(Debug, Clone)]
struct MatchedProcess {
    profile_id: Uuid,
    pid: u32,
    started_at: String,
}

pub trait ProcessProvider: Send + Sync {
    fn discover_codex_executable(
        &self,
        custom_path: Option<&Path>,
    ) -> Result<CodexPathDiscovery, String>;
    fn browse_codex_executable(&self) -> Result<Option<PathBuf>, String>;
    fn resolve_codex_executable(&self, custom_path: Option<&Path>) -> Result<PathBuf, String>;
    fn validate_codex_executable(&self, path: &Path) -> Result<PathBuf, String>;
    fn launch_codex(
        &self,
        codex_executable: &Path,
        user_data_dir: &Path,
        options: &LaunchOptions,
    ) -> Result<LaunchResult, String>;
    fn focus_process_window(&self, pid: u32) -> Result<bool, String>;
    fn kill_profile_instances(
        &self,
        profile_dirs: &[(Uuid, PathBuf)],
    ) -> Result<Vec<KillResult>, String>;
    fn detect_running_profiles(
        &self,
        profile_dirs: &[(Uuid, PathBuf)],
    ) -> Result<HashMap<Uuid, RuntimeState>, String>;
}

#[derive(Debug, Clone)]
pub struct LaunchResult {
    pub pid: u32,
    pub started_at: String,
}

#[derive(Clone, Default)]
pub struct ProcessManager {
    children: Arc<Mutex<HashMap<u32, TrackedChild>>>,
}

#[cfg(not(windows))]
type TrackedChild = std::process::Child;

#[cfg(windows)]
type TrackedChild = ();

impl ProcessManager {
    #[cfg(not(windows))]
    fn track(&self, child: std::process::Child) -> Result<u32, String> {
        let pid = child.id();
        let mut children = self
            .children
            .lock()
            .map_err(|_| "Aion process manager lock was poisoned.".to_string())?;
        children.insert(pid, child);
        Ok(pid)
    }

    fn release(&self, pid: u32) -> Result<(), String> {
        let mut children = self
            .children
            .lock()
            .map_err(|_| "Aion process manager lock was poisoned.".to_string())?;
        children.remove(&pid);
        Ok(())
    }
}

#[derive(Clone, Default)]
pub struct CodexProcessProvider {
    process_manager: ProcessManager,
}

impl CodexProcessProvider {
    pub fn new() -> Self {
        Self {
            process_manager: ProcessManager::default(),
        }
    }
}

impl ProcessProvider for CodexProcessProvider {
    fn discover_codex_executable(
        &self,
        custom_path: Option<&Path>,
    ) -> Result<CodexPathDiscovery, String> {
        Ok(discover_codex_executable(custom_path))
    }

    fn browse_codex_executable(&self) -> Result<Option<PathBuf>, String> {
        let Some(path) = platform::browse_codex_executable()? else {
            return Ok(None);
        };

        self.validate_codex_executable(&path).map(Some)
    }

    fn resolve_codex_executable(&self, custom_path: Option<&Path>) -> Result<PathBuf, String> {
        if let Some(path) = custom_path {
            return self.validate_codex_executable(path);
        }

        let discovery = discover_codex_executable(None);
        match discovery.path {
            Some(path) => self.validate_codex_executable(Path::new(&path)),
            None => Err(discovery.message),
        }
    }

    fn validate_codex_executable(&self, path: &Path) -> Result<PathBuf, String> {
        validate_codex_executable(path)
    }

    fn launch_codex(
        &self,
        codex_executable: &Path,
        user_data_dir: &Path,
        options: &LaunchOptions,
    ) -> Result<LaunchResult, String> {
        let executable = validate_codex_executable(codex_executable)?;
        let sandbox_path = user_data_dir;

        reject_path_traversal(sandbox_path, "profile sandbox directory")?;
        fs::create_dir_all(sandbox_path).map_err(|error| {
            format!(
                "Failed to create profile sandbox directory '{}': {error}",
                sandbox_path.display()
            )
        })?;
        let safe_sandbox_path =
            canonicalize_existing_path(sandbox_path, "profile sandbox directory")?;
        ensure_directory_is_not_reparse_point(&safe_sandbox_path, "profile sandbox directory")?;

        let pid = self.launch_isolated_codex_process(
            &executable,
            &safe_sandbox_path,
            options.proxy.as_ref(),
        )?;
        schedule_window_title_update(pid, options.window_title());

        Ok(LaunchResult {
            pid,
            started_at: iso_timestamp_now()?,
        })
    }

    fn focus_process_window(&self, pid: u32) -> Result<bool, String> {
        if pid == 0 {
            return Err("Cannot focus a process with pid 0.".to_string());
        }

        Ok(focus_window_for_pid(pid))
    }

    fn kill_profile_instances(
        &self,
        profile_dirs: &[(Uuid, PathBuf)],
    ) -> Result<Vec<KillResult>, String> {
        let killed = kill_profile_instances(profile_dirs);
        for result in &killed {
            if result.killed {
                self.process_manager.release(result.pid)?;
            }
        }
        Ok(killed)
    }

    fn detect_running_profiles(
        &self,
        profile_dirs: &[(Uuid, PathBuf)],
    ) -> Result<HashMap<Uuid, RuntimeState>, String> {
        Ok(detect_running_profiles(profile_dirs))
    }
}

impl CodexProcessProvider {
    #[cfg(windows)]
    fn launch_isolated_codex_process(
        &self,
        executable: &Path,
        absolute_sandbox_path: &Path,
        proxy: Option<&ProxyConfig>,
    ) -> Result<u32, String> {
        let executable_arg =
            windows_command_line_path_text(executable, "absolute Codex executable path")?;
        let sandbox_arg = windows_command_line_path_text(
            absolute_sandbox_path,
            "absolute profile sandbox directory",
        )?;

        let mut command = Command::new("cmd.exe");
        command
            .arg("/c")
            .arg("start")
            .arg("")
            .arg(executable_arg)
            .arg(format!("--user-data-dir={}", sandbox_arg))
            .arg("--no-first-run");
        append_proxy_configuration(&mut command, proxy);

        let status = command.status().map_err(|error| {
            format!(
                "Failed to launch Codex through cmd.exe start from '{}' with sandbox '{}': {error}",
                executable.display(),
                absolute_sandbox_path.display()
            )
        })?;

        if !status.success() {
            return Err(format!(
                "cmd.exe start failed while launching Codex with sandbox '{}'. Exit code: {:?}",
                absolute_sandbox_path.display(),
                status.code()
            ));
        }

        wait_for_primary_codex_process_using_user_data_dir(absolute_sandbox_path)?.ok_or_else(|| {
            format!(
                "Codex was started through the Windows shell, but Aion could not verify that it adopted sandbox '{}'. Close all existing Codex windows and launch this profile again.",
                absolute_sandbox_path.display()
            )
        })
    }

    #[cfg(not(windows))]
    fn launch_isolated_codex_process(
        &self,
        executable: &Path,
        sandbox_path: &Path,
        proxy: Option<&ProxyConfig>,
    ) -> Result<u32, String> {
        let mut command = Command::new(executable);
        command.arg(format!("--user-data-dir={}", sandbox_path.display()));
        append_proxy_configuration(&mut command, proxy);

        let child = command.spawn().map_err(|error| {
            format!(
                "Failed to launch Codex from '{}' with user data directory '{}': {error}",
                executable.display(),
                sandbox_path.display()
            )
        })?;

        self.process_manager.track(child)
    }
}

pub fn validate_codex_executable(path: &Path) -> Result<PathBuf, String> {
    reject_path_traversal(path, "Codex executable path")?;
    let canonical = canonicalize_existing_path(path, "Codex executable path")?;
    ensure_file_is_not_reparse_point(&canonical, "Codex executable path")?;

    if !canonical.is_file() {
        return Err(format!(
            "Codex executable path '{}' is not a file.",
            canonical.display()
        ));
    }

    let is_codex_exe = canonical
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.eq_ignore_ascii_case(CODEX_EXECUTABLE_NAME))
        .unwrap_or(false);

    if !is_codex_exe {
        return Err(format!(
            "Expected a '{}' executable, but received '{}'.",
            CODEX_EXECUTABLE_NAME,
            canonical.display()
        ));
    }

    Ok(canonical)
}

pub fn discover_codex_executable(custom_path: Option<&Path>) -> CodexPathDiscovery {
    let mut diagnostics = Vec::new();

    if let Some(path) = custom_path {
        match validate_codex_executable(path) {
            Ok(validated) => {
                return discovery_hit(
                    validated,
                    "Configured Path",
                    false,
                    "Using the saved custom Codex.exe path.",
                );
            }
            Err(error) => diagnostics.push(format!("Saved custom path rejected: {error}")),
        }
    }

    match find_codex_executable_from_registry() {
        Ok(path) => {
            return discovery_hit(
                path,
                "Windows AppModel Registry",
                true,
                "Codex.exe was auto-detected from the registered OpenAI.Codex package.",
            );
        }
        Err(error) => diagnostics.push(error),
    }

    let uninstall_candidates = platform::find_codex_executable_from_uninstall_registry();
    if let Some((path, error_notes)) = first_valid_candidate(uninstall_candidates) {
        diagnostics.extend(error_notes);
        return discovery_hit(
            path,
            "Windows Uninstall Registry",
            true,
            "Codex.exe was auto-detected from installed application metadata.",
        );
    }

    let standard_candidates = standard_path_candidates();
    if let Some((path, error_notes)) = first_valid_candidate(standard_candidates) {
        diagnostics.extend(error_notes);
        return discovery_hit(
            path,
            "Standard Windows Paths",
            true,
            "Codex.exe was auto-detected from local application directories.",
        );
    }

    let detail = if diagnostics.is_empty() {
        "No Codex.exe candidates were found.".to_string()
    } else {
        diagnostics.join(" ")
    };

    CodexPathDiscovery {
        path: None,
        source: "Not Found".to_string(),
        auto_detected: false,
        message: format!(
            "Aion could not auto-detect Codex.exe. Use Browse or paste the executable path manually. {detail}"
        ),
    }
}

pub fn detect_running_profiles(profile_dirs: &[(Uuid, PathBuf)]) -> HashMap<Uuid, RuntimeState> {
    matched_profile_processes(profile_dirs)
        .into_iter()
        .map(|matched| {
            (
                matched.profile_id,
                RuntimeState::running(matched.pid, matched.started_at),
            )
        })
        .collect()
}

fn kill_profile_instances(profile_dirs: &[(Uuid, PathBuf)]) -> Vec<KillResult> {
    if profile_dirs.is_empty() {
        return Vec::new();
    }

    let normalized_targets = normalized_profile_targets(profile_dirs);
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    let system = System::new_all();

    for process in system.processes().values() {
        if !is_codex_process(process) {
            continue;
        }

        for profile_id in matching_profile_ids(process, &normalized_targets) {
            let pid = process.pid().as_u32();
            if !seen.insert((profile_id, pid)) {
                continue;
            }

            result.push(KillResult {
                id: profile_id.to_string(),
                pid,
                killed: process.kill(),
            });
        }
    }

    result
}

fn matched_profile_processes(profile_dirs: &[(Uuid, PathBuf)]) -> Vec<MatchedProcess> {
    if profile_dirs.is_empty() {
        return Vec::new();
    }

    let normalized_targets = normalized_profile_targets(profile_dirs);
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    let system = System::new_all();

    for process in system.processes().values() {
        if !is_codex_process(process) {
            continue;
        }

        for profile_id in matching_profile_ids(process, &normalized_targets) {
            let pid = process.pid().as_u32();
            if !seen.insert((profile_id, pid)) {
                continue;
            }

            result.push(MatchedProcess {
                profile_id,
                pid,
                started_at: iso_timestamp_from_unix(process.start_time()),
            });
        }
    }

    result
}

fn normalized_profile_targets(profile_dirs: &[(Uuid, PathBuf)]) -> Vec<(Uuid, String)> {
    profile_dirs
        .iter()
        .map(|(profile_id, dir)| (*profile_id, normalize_path_for_match(dir)))
        .collect()
}

fn matching_profile_ids(process: &Process, normalized_targets: &[(Uuid, String)]) -> Vec<Uuid> {
    let args = process_args(process);
    if !is_primary_codex_process(&args) {
        return Vec::new();
    }

    let user_data_dirs = extract_user_data_dirs(&args);
    if user_data_dirs.is_empty() {
        return Vec::new();
    }

    let mut matches = Vec::new();
    for observed_dir in user_data_dirs {
        let observed_normalized = normalize_path_string_for_match(&observed_dir);
        for (profile_id, target_dir) in normalized_targets {
            if user_data_dir_matches_target(&observed_normalized, target_dir) {
                matches.push(*profile_id);
            }
        }
    }

    matches
}

fn is_primary_codex_process(args: &[String]) -> bool {
    !args.iter().any(|arg| {
        let trimmed = trim_wrapping_quotes(arg.trim());
        trimmed == "--type" || trimmed.starts_with("--type=")
    })
}

fn user_data_dir_matches_target(observed_normalized: &str, target_normalized: &str) -> bool {
    observed_normalized == target_normalized
        || observed_normalized
            .strip_prefix(target_normalized)
            .map(|suffix| suffix.starts_with('\\'))
            .unwrap_or(false)
}

#[cfg(windows)]
fn wait_for_primary_codex_process_using_user_data_dir(
    absolute_sandbox_path: &Path,
) -> Result<Option<u32>, String> {
    for _ in 0..LAUNCH_PROCESS_DETECTION_RETRIES {
        if let Some(pid) = find_primary_codex_process_using_user_data_dir(absolute_sandbox_path) {
            return Ok(Some(pid));
        }
        thread::sleep(Duration::from_millis(LAUNCH_PROCESS_DETECTION_DELAY_MS));
    }

    Ok(None)
}

#[cfg(windows)]
fn find_primary_codex_process_using_user_data_dir(absolute_sandbox_path: &Path) -> Option<u32> {
    let target = normalize_path_for_match(absolute_sandbox_path);
    let system = System::new_all();

    for process in system.processes().values() {
        if !is_codex_process(process) {
            continue;
        }

        let args = process_args(process);
        if !is_primary_codex_process(&args) {
            continue;
        }

        let matches_target = extract_user_data_dirs(&args)
            .into_iter()
            .any(|observed_dir| {
                let observed = normalize_path_string_for_match(&observed_dir);
                user_data_dir_matches_target(&observed, &target)
            });

        if matches_target {
            return Some(process.pid().as_u32());
        }
    }

    None
}

fn discovery_hit(
    path: PathBuf,
    source: &str,
    auto_detected: bool,
    message: &str,
) -> CodexPathDiscovery {
    CodexPathDiscovery {
        path: Some(path.to_string_lossy().into_owned()),
        source: source.to_string(),
        auto_detected,
        message: message.to_string(),
    }
}

fn first_valid_candidate(candidates: Vec<PathBuf>) -> Option<(PathBuf, Vec<String>)> {
    let mut seen = HashSet::new();
    let mut errors = Vec::new();

    for candidate in candidates {
        let normalized = normalize_path_text(&candidate.to_string_lossy());
        if !seen.insert(normalized) {
            continue;
        }

        match validate_codex_executable(&candidate) {
            Ok(path) => return Some((path, errors)),
            Err(error) => errors.push(error),
        }
    }

    None
}

fn standard_path_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    let mut roots = Vec::new();

    push_env_root(&mut roots, "LOCALAPPDATA");
    push_env_root(&mut roots, "PROGRAMFILES");
    push_env_root(&mut roots, "ProgramFiles(x86)");

    for root in roots {
        candidates.extend(direct_standard_candidates(&root));

        for search_root in standard_search_roots(&root) {
            if search_root.is_dir() {
                let mut visited = 0_usize;
                if let Some(found) = find_file_bounded(
                    &search_root,
                    CODEX_EXECUTABLE_NAME,
                    MAX_DISCOVERY_DEPTH,
                    &mut visited,
                ) {
                    candidates.push(found);
                }
            }
        }
    }

    candidates
}

fn push_env_root(roots: &mut Vec<PathBuf>, name: &str) {
    if let Some(value) = std::env::var_os(name) {
        let path = PathBuf::from(value);
        if !path.as_os_str().is_empty() {
            roots.push(path);
        }
    }
}

fn direct_standard_candidates(root: &Path) -> Vec<PathBuf> {
    vec![
        root.join("Codex").join(CODEX_EXECUTABLE_NAME),
        root.join("OpenAI")
            .join("Codex")
            .join(CODEX_EXECUTABLE_NAME),
        root.join("OpenAI Codex").join(CODEX_EXECUTABLE_NAME),
        root.join("Programs")
            .join("Codex")
            .join(CODEX_EXECUTABLE_NAME),
        root.join("Programs")
            .join("OpenAI")
            .join("Codex")
            .join(CODEX_EXECUTABLE_NAME),
        root.join("Programs")
            .join("OpenAI Codex")
            .join(CODEX_EXECUTABLE_NAME),
    ]
}

fn standard_search_roots(root: &Path) -> Vec<PathBuf> {
    vec![
        root.join("Codex"),
        root.join("OpenAI"),
        root.join("OpenAI Codex"),
        root.join("Programs").join("Codex"),
        root.join("Programs").join("OpenAI"),
        root.join("Programs").join("OpenAI Codex"),
    ]
}

fn append_proxy_configuration(command: &mut Command, proxy: Option<&ProxyConfig>) {
    let Some(proxy) = proxy else {
        return;
    };

    if proxy.host_port.is_empty() {
        return;
    }

    command.arg(format!("--proxy-server=http://{}", proxy.host_port));

    let proxy_uri = proxy_env_uri(proxy);
    command.env("HTTP_PROXY", &proxy_uri);
    command.env("HTTPS_PROXY", &proxy_uri);
    command.env("ALL_PROXY", &proxy_uri);
}

fn proxy_env_uri(proxy: &ProxyConfig) -> String {
    if !proxy.has_credentials() {
        return format!("http://{}", proxy.host_port);
    }

    format!(
        "http://{}:{}@{}",
        percent_encode_ascii_component(&proxy.username),
        percent_encode_ascii_component(&proxy.password),
        proxy.host_port
    )
}

fn percent_encode_ascii_component(input: &str) -> String {
    let mut output = String::new();
    for byte in input.bytes() {
        let is_unreserved =
            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~');
        if is_unreserved {
            output.push(char::from(byte));
        } else {
            output.push_str(&format!("%{byte:02X}"));
        }
    }
    output
}

fn is_codex_process(process: &Process) -> bool {
    let name_matches = process
        .name()
        .to_string_lossy()
        .eq_ignore_ascii_case(CODEX_EXECUTABLE_NAME)
        || process
            .name()
            .to_string_lossy()
            .eq_ignore_ascii_case("Codex");

    if name_matches {
        return true;
    }

    process
        .exe()
        .and_then(|path| path.file_name())
        .and_then(|name| name.to_str())
        .map(|name| name.eq_ignore_ascii_case(CODEX_EXECUTABLE_NAME))
        .unwrap_or(false)
}

fn process_args(process: &Process) -> Vec<String> {
    process
        .cmd()
        .iter()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect()
}

fn extract_user_data_dirs(args: &[String]) -> Vec<String> {
    let mut values = Vec::new();

    for (index, arg) in args.iter().enumerate() {
        let trimmed = trim_wrapping_quotes(arg.trim());

        if let Some(value) = trimmed.strip_prefix(&format!("{USER_DATA_DIR_ARG}=")) {
            let normalized = trim_wrapping_quotes(value.trim()).to_string();
            if !normalized.is_empty() {
                values.push(normalized);
            }
            continue;
        }

        if trimmed == USER_DATA_DIR_ARG {
            if let Some(next_value) = args.get(index + 1) {
                let normalized = trim_wrapping_quotes(next_value.trim()).to_string();
                if !normalized.is_empty() {
                    values.push(normalized);
                }
            }
        }
    }

    values
}

fn trim_wrapping_quotes(input: &str) -> &str {
    let trimmed = input.trim();
    let is_double_quoted = trimmed.starts_with('"') && trimmed.ends_with('"');
    let is_single_quoted = trimmed.starts_with('\'') && trimmed.ends_with('\'');

    if trimmed.len() >= 2 && (is_double_quoted || is_single_quoted) {
        return &trimmed[1..trimmed.len() - 1];
    }

    trimmed
}

fn reject_path_traversal(path: &Path, label: &str) -> Result<(), String> {
    if path.as_os_str().is_empty() {
        return Err(format!("{label} cannot be empty."));
    }

    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(format!(
            "{} '{}' contains a path traversal segment.",
            label,
            path.display()
        ));
    }

    Ok(())
}

fn canonicalize_existing_path(path: &Path, label: &str) -> Result<PathBuf, String> {
    fs::canonicalize(path).map_err(|error| {
        format!(
            "Failed to canonicalize {} '{}': {error}",
            label,
            path.display()
        )
    })
}

fn ensure_file_is_not_reparse_point(path: &Path, label: &str) -> Result<(), String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("Failed to inspect {} '{}': {error}", label, path.display()))?;

    if metadata.file_type().is_symlink() || is_windows_reparse_point(&metadata) {
        return Err(format!(
            "Refusing to use {} '{}' because it is a symlink, junction, or reparse point.",
            label,
            path.display()
        ));
    }

    Ok(())
}

fn ensure_directory_is_not_reparse_point(path: &Path, label: &str) -> Result<(), String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("Failed to inspect {} '{}': {error}", label, path.display()))?;

    if !metadata.is_dir() {
        return Err(format!(
            "{} '{}' is not a directory.",
            label,
            path.display()
        ));
    }

    if metadata.file_type().is_symlink() || is_windows_reparse_point(&metadata) {
        return Err(format!(
            "Refusing to use {} '{}' because it is a symlink, junction, or reparse point.",
            label,
            path.display()
        ));
    }

    Ok(())
}

#[cfg(windows)]
fn is_windows_reparse_point(metadata: &fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0400;
    metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
}

#[cfg(not(windows))]
fn is_windows_reparse_point(_metadata: &fs::Metadata) -> bool {
    false
}

fn normalize_path_for_match(path: &Path) -> String {
    match fs::canonicalize(path) {
        Ok(canonical) => normalize_path_text(&canonical.to_string_lossy()),
        Err(_) => normalize_path_text(&path.to_string_lossy()),
    }
}

fn normalize_path_string_for_match(path: &str) -> String {
    let trimmed = trim_wrapping_quotes(path);
    let path_buf = PathBuf::from(trimmed);
    match fs::canonicalize(&path_buf) {
        Ok(canonical) => normalize_path_text(&canonical.to_string_lossy()),
        Err(_) => normalize_path_text(trimmed),
    }
}

fn normalize_path_text(input: &str) -> String {
    let mut text = normal_windows_path_text_from_str(input.trim()).replace('/', "\\");
    while text.len() > 3 && (text.ends_with('\\') || text.ends_with('/')) {
        text.pop();
    }
    text.to_ascii_lowercase()
}

#[cfg(windows)]
fn windows_command_line_path_text(path: &Path, label: &str) -> Result<String, String> {
    let text = normal_windows_path_text_from_str(&path.to_string_lossy());
    if text.is_empty() {
        return Err(format!("{label} cannot be empty."));
    }

    if !Path::new(&text).is_absolute() {
        return Err(format!(
            "{} '{}' must resolve to an absolute path before launch.",
            label,
            path.display()
        ));
    }

    if text.contains('"') || text.contains('\r') || text.contains('\n') {
        return Err(format!(
            "{} '{}' contains characters that cannot be injected safely into the Codex command line.",
            label,
            path.display()
        ));
    }

    Ok(text)
}

fn normal_windows_path_text_from_str(input: &str) -> String {
    if let Some(rest) = input.strip_prefix(r"\\?\UNC\") {
        return format!(r"\\{rest}");
    }

    if let Some(rest) = input.strip_prefix(r"\\?\") {
        return rest.to_string();
    }

    if let Some(rest) = input.strip_prefix(r"\??\") {
        return rest.to_string();
    }

    input.to_string()
}

fn find_file_bounded(
    root: &Path,
    file_name: &str,
    remaining_depth: usize,
    visited: &mut usize,
) -> Option<PathBuf> {
    if remaining_depth == 0 || *visited >= MAX_DISCOVERY_ENTRIES {
        return None;
    }

    let metadata = fs::symlink_metadata(root).ok()?;
    if metadata.file_type().is_symlink()
        || is_windows_reparse_point(&metadata)
        || !metadata.is_dir()
    {
        return None;
    }

    let entries = fs::read_dir(root).ok()?;
    for entry_result in entries {
        if *visited >= MAX_DISCOVERY_ENTRIES {
            return None;
        }

        let entry = match entry_result {
            Ok(entry) => entry,
            Err(_) => continue,
        };
        *visited += 1;

        let path = entry.path();
        let entry_type = match entry.file_type() {
            Ok(entry_type) => entry_type,
            Err(_) => continue,
        };

        if entry_type.is_file() {
            let matches = path
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.eq_ignore_ascii_case(file_name))
                .unwrap_or(false);
            if matches {
                return Some(path);
            }
        }

        if entry_type.is_dir() {
            if let Some(found) = find_file_bounded(&path, file_name, remaining_depth - 1, visited) {
                return Some(found);
            }
        }
    }

    None
}

pub fn find_codex_executable_from_registry() -> Result<PathBuf, String> {
    platform::find_codex_executable_from_registry()
}

#[cfg(windows)]
fn schedule_window_title_update(pid: u32, title: String) {
    thread::spawn(move || {
        for _ in 0..WINDOW_TITLE_RETRY_COUNT {
            if set_window_title_for_pid(pid, &title) {
                return;
            }
            thread::sleep(Duration::from_millis(WINDOW_TITLE_RETRY_DELAY_MS));
        }
    });
}

#[cfg(not(windows))]
fn schedule_window_title_update(_pid: u32, _title: String) {}

#[cfg(windows)]
fn set_window_title_for_pid(pid: u32, title: &str) -> bool {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::core::BOOL;
    use windows_sys::Win32::{
        Foundation::{HWND, LPARAM},
        UI::WindowsAndMessaging::{
            EnumWindows, GetWindowThreadProcessId, IsWindowVisible, SetWindowTextW,
        },
    };

    struct Context {
        pid: u32,
        title: Vec<u16>,
        matched: bool,
    }

    unsafe extern "system" fn callback(hwnd: HWND, lparam: LPARAM) -> BOOL {
        let context = &mut *(lparam as *mut Context);
        let mut process_id = 0_u32;
        GetWindowThreadProcessId(hwnd, &mut process_id);
        if process_id == context.pid && IsWindowVisible(hwnd) != 0 {
            SetWindowTextW(hwnd, context.title.as_ptr());
            context.matched = true;
            return 0;
        }
        1
    }

    let mut context = Context {
        pid,
        title: std::ffi::OsStr::new(title)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect(),
        matched: false,
    };

    unsafe {
        EnumWindows(Some(callback), &mut context as *mut Context as LPARAM);
    }
    context.matched
}

#[cfg(windows)]
fn focus_window_for_pid(pid: u32) -> bool {
    use windows_sys::core::BOOL;
    use windows_sys::Win32::{
        Foundation::{HWND, LPARAM},
        UI::WindowsAndMessaging::{
            EnumWindows, GetWindowThreadProcessId, IsWindowVisible, SetForegroundWindow,
            ShowWindow, SW_RESTORE,
        },
    };

    struct Context {
        pid: u32,
        focused: bool,
    }

    unsafe extern "system" fn callback(hwnd: HWND, lparam: LPARAM) -> BOOL {
        let context = &mut *(lparam as *mut Context);
        let mut process_id = 0_u32;
        GetWindowThreadProcessId(hwnd, &mut process_id);
        if process_id == context.pid && IsWindowVisible(hwnd) != 0 {
            ShowWindow(hwnd, SW_RESTORE);
            context.focused = SetForegroundWindow(hwnd) != 0;
            return 0;
        }
        1
    }

    let mut context = Context {
        pid,
        focused: false,
    };

    unsafe {
        EnumWindows(Some(callback), &mut context as *mut Context as LPARAM);
    }
    context.focused
}

#[cfg(not(windows))]
fn focus_window_for_pid(_pid: u32) -> bool {
    false
}

#[cfg(windows)]
mod platform {
    use super::{
        find_file_bounded, trim_wrapping_quotes, validate_codex_executable, CODEX_EXECUTABLE_NAME,
        MAX_DISCOVERY_DEPTH,
    };
    use std::{
        cmp::Ordering,
        ffi::OsStr,
        mem,
        os::windows::ffi::OsStrExt,
        path::{Path, PathBuf},
        ptr,
    };
    use windows_sys::Win32::{
        Foundation::{ERROR_MORE_DATA, ERROR_NO_MORE_ITEMS, ERROR_SUCCESS, HINSTANCE, HWND},
        System::{
            Environment::ExpandEnvironmentStringsW,
            Registry::{
                RegCloseKey, RegEnumKeyExW, RegOpenKeyExW, RegQueryValueExW, HKEY,
                HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE, KEY_READ, REG_EXPAND_SZ, REG_SZ,
            },
        },
        UI::Controls::Dialogs::{
            CommDlgExtendedError, GetOpenFileNameW, OFN_EXPLORER, OFN_FILEMUSTEXIST,
            OFN_NOCHANGEDIR, OFN_PATHMUSTEXIST, OPENFILENAMEW,
        },
    };

    const PACKAGES_REGISTRY_PATH: &str = r"Software\Classes\Local Settings\Software\Microsoft\Windows\CurrentVersion\AppModel\Repository\Packages";
    const CODEX_PACKAGE_PREFIX: &str = "OpenAI.Codex";
    const PACKAGE_ROOT_VALUE_NAMES: [&str; 4] = [
        "PackageRootFolder",
        "PackageRoot",
        "Path",
        "InstallLocation",
    ];
    const UNINSTALL_REGISTRY_PATHS: [&str; 2] = [
        r"Software\Microsoft\Windows\CurrentVersion\Uninstall",
        r"Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
    ];

    #[derive(Clone, Copy)]
    enum RegistryHive {
        CurrentUser,
        LocalMachine,
    }

    impl RegistryHive {
        fn handle(self) -> HKEY {
            match self {
                Self::CurrentUser => HKEY_CURRENT_USER,
                Self::LocalMachine => HKEY_LOCAL_MACHINE,
            }
        }

        fn label(self) -> &'static str {
            match self {
                Self::CurrentUser => "HKCU",
                Self::LocalMachine => "HKLM",
            }
        }
    }

    pub fn find_codex_executable_from_registry() -> Result<PathBuf, String> {
        let packages_key = RegistryKey::open(RegistryHive::CurrentUser, PACKAGES_REGISTRY_PATH)
            .map_err(|error| {
                format!(
                    "Failed to open Windows AppModel package registry path '{}': {error}",
                    PACKAGES_REGISTRY_PATH
                )
            })?;

        let mut package_names = packages_key.enumerate_subkeys().map_err(|error| {
            format!(
                "Failed to enumerate Windows AppModel package registry path '{}': {error}",
                PACKAGES_REGISTRY_PATH
            )
        })?;

        package_names.retain(|name| name.starts_with(CODEX_PACKAGE_PREFIX));
        package_names.sort_by(compare_package_names_descending);

        let mut scanned_packages = 0_usize;
        let mut roots_without_exe = Vec::new();

        for package_name in package_names {
            scanned_packages += 1;
            let package_key_path = format!("{PACKAGES_REGISTRY_PATH}\\{package_name}");
            let package_key = match RegistryKey::open(RegistryHive::CurrentUser, &package_key_path)
            {
                Ok(key) => key,
                Err(_) => continue,
            };

            let package_root = match query_package_root(&package_key) {
                Some(root) => root,
                None => continue,
            };

            if let Some(executable) = find_codex_executable_under_root(&package_root) {
                return validate_codex_executable(&executable);
            }

            roots_without_exe.push(package_root);
        }

        if scanned_packages == 0 {
            return Err(format!(
                "Codex is not registered under HKCU\\{}. Install Codex or set a custom Codex.exe path.",
                PACKAGES_REGISTRY_PATH
            ));
        }

        let roots = roots_without_exe
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<String>>()
            .join(", ");

        if roots.is_empty() {
            Err(format!(
                "Found {scanned_packages} OpenAI.Codex package registry entrie(s), but none exposed a package root path."
            ))
        } else {
            Err(format!(
                "Found {scanned_packages} OpenAI.Codex package registry entrie(s), but Codex.exe was not found under package root(s): {roots}."
            ))
        }
    }

    pub fn find_codex_executable_from_uninstall_registry() -> Vec<PathBuf> {
        let mut candidates = Vec::new();

        for hive in [RegistryHive::CurrentUser, RegistryHive::LocalMachine] {
            for path in UNINSTALL_REGISTRY_PATHS {
                let Ok(root_key) = RegistryKey::open(hive, path) else {
                    continue;
                };

                let Ok(subkeys) = root_key.enumerate_subkeys() else {
                    continue;
                };

                for subkey_name in subkeys {
                    let subkey_path = format!("{path}\\{subkey_name}");
                    let Ok(app_key) = RegistryKey::open(hive, &subkey_path) else {
                        continue;
                    };

                    if !registry_entry_mentions_codex(&app_key, &subkey_name) {
                        continue;
                    }

                    candidates.extend(uninstall_entry_candidates(&app_key));
                }
            }
        }

        candidates
    }

    pub fn browse_codex_executable() -> Result<Option<PathBuf>, String> {
        let mut file_buffer = vec![0_u16; 32_768];
        let filter = to_wide_filter(&[
            "Codex executable",
            CODEX_EXECUTABLE_NAME,
            "Executable files (*.exe)",
            "*.exe",
            "All files (*.*)",
            "*.*",
        ]);
        let title = to_wide(OsStr::new("Select Codex.exe"));
        let default_extension = to_wide(OsStr::new("exe"));

        let mut open_file_name = OPENFILENAMEW {
            lStructSize: mem::size_of::<OPENFILENAMEW>() as u32,
            hwndOwner: ptr::null_mut::<core::ffi::c_void>() as HWND,
            hInstance: ptr::null_mut::<core::ffi::c_void>() as HINSTANCE,
            lpstrFilter: filter.as_ptr(),
            lpstrCustomFilter: ptr::null_mut(),
            nMaxCustFilter: 0,
            nFilterIndex: 1,
            lpstrFile: file_buffer.as_mut_ptr(),
            nMaxFile: file_buffer.len() as u32,
            lpstrFileTitle: ptr::null_mut(),
            nMaxFileTitle: 0,
            lpstrInitialDir: ptr::null(),
            lpstrTitle: title.as_ptr(),
            Flags: OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR | OFN_PATHMUSTEXIST,
            nFileOffset: 0,
            nFileExtension: 0,
            lpstrDefExt: default_extension.as_ptr(),
            lCustData: 0,
            lpfnHook: None,
            lpTemplateName: ptr::null(),
            pvReserved: ptr::null_mut(),
            dwReserved: 0,
            FlagsEx: 0,
        };

        let selected = unsafe { GetOpenFileNameW(&mut open_file_name) };
        if selected == 0 {
            let extended_error = unsafe { CommDlgExtendedError() };
            if extended_error == 0 {
                return Ok(None);
            }

            return Err(format!(
                "Windows file picker failed with common dialog error code {extended_error}."
            ));
        }

        let length = match file_buffer.iter().position(|character| *character == 0) {
            Some(index) => index,
            None => file_buffer.len(),
        };

        let selected_path = String::from_utf16(&file_buffer[..length]).map_err(|error| {
            format!("Windows file picker returned invalid UTF-16 path data: {error}")
        })?;

        Ok(Some(PathBuf::from(selected_path)))
    }

    fn registry_entry_mentions_codex(app_key: &RegistryKey, subkey_name: &str) -> bool {
        if subkey_name.to_ascii_lowercase().contains("codex") {
            return true;
        }

        app_key
            .query_string_value("DisplayName")
            .map(|value| {
                let lower = value.to_ascii_lowercase();
                lower.contains("codex") || lower.contains("openai")
            })
            .unwrap_or(false)
    }

    fn uninstall_entry_candidates(app_key: &RegistryKey) -> Vec<PathBuf> {
        let mut candidates = Vec::new();

        for value_name in ["DisplayIcon", "InstallLocation", "UninstallString"] {
            let Ok(value) = app_key.query_string_value(value_name) else {
                continue;
            };

            if value_name == "InstallLocation" {
                let install_location = PathBuf::from(trim_wrapping_quotes(value.trim()));
                if install_location.as_os_str().is_empty() {
                    continue;
                }
                candidates.push(install_location.join(CODEX_EXECUTABLE_NAME));
                let mut visited = 0_usize;
                if let Some(found) = find_file_bounded(
                    &install_location,
                    CODEX_EXECUTABLE_NAME,
                    MAX_DISCOVERY_DEPTH,
                    &mut visited,
                ) {
                    candidates.push(found);
                }
                continue;
            }

            if let Some(path) = extract_executable_path(&value) {
                candidates.push(path);
            }
        }

        candidates
    }

    fn extract_executable_path(input: &str) -> Option<PathBuf> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return None;
        }

        if let Some(quoted) = extract_quoted_prefix(trimmed) {
            return Some(PathBuf::from(quoted));
        }

        let without_icon_index = strip_display_icon_index(trimmed);
        let lower = without_icon_index.to_ascii_lowercase();
        if let Some(index) = lower.find(".exe") {
            let end = index + 4;
            return Some(PathBuf::from(without_icon_index[..end].trim()));
        }

        Some(PathBuf::from(trim_wrapping_quotes(without_icon_index)))
    }

    fn extract_quoted_prefix(input: &str) -> Option<String> {
        let remainder = input.strip_prefix('"')?;
        let end = remainder.find('"')?;
        Some(remainder[..end].to_string())
    }

    fn strip_display_icon_index(input: &str) -> &str {
        let Some((path_part, suffix)) = input.rsplit_once(',') else {
            return input;
        };

        if suffix.trim().parse::<i32>().is_ok() {
            return path_part.trim();
        }

        input
    }

    fn query_package_root(package_key: &RegistryKey) -> Option<PathBuf> {
        for value_name in PACKAGE_ROOT_VALUE_NAMES {
            if let Ok(value) = package_key.query_string_value(value_name) {
                let path = PathBuf::from(value);
                if !path.as_os_str().is_empty() {
                    return Some(path);
                }
            }
        }
        None
    }

    fn find_codex_executable_under_root(root: &Path) -> Option<PathBuf> {
        let direct_candidates = [
            root.join("app").join(CODEX_EXECUTABLE_NAME),
            root.join(CODEX_EXECUTABLE_NAME),
        ];

        for candidate in direct_candidates {
            if candidate.is_file() {
                return Some(candidate);
            }
        }

        let mut visited = 0_usize;
        find_file_bounded(
            root,
            CODEX_EXECUTABLE_NAME,
            MAX_DISCOVERY_DEPTH,
            &mut visited,
        )
    }

    fn compare_package_names_descending(left: &String, right: &String) -> Ordering {
        let right_version = package_version(right);
        let left_version = package_version(left);
        right_version
            .cmp(&left_version)
            .then_with(|| right.cmp(left))
    }

    fn package_version(package_name: &str) -> [u64; 4] {
        let mut version = [0_u64; 4];
        let version_segment = match package_name.split('_').nth(1) {
            Some(segment) => segment,
            None => return version,
        };

        for (index, part) in version_segment.split('.').take(4).enumerate() {
            if let Ok(value) = part.parse::<u64>() {
                version[index] = value;
            }
        }

        version
    }

    struct RegistryKey {
        hive: RegistryHive,
        handle: HKEY,
    }

    impl RegistryKey {
        fn open(hive: RegistryHive, path: &str) -> Result<Self, String> {
            let mut handle: HKEY = ptr::null_mut();
            let path_wide = to_wide(OsStr::new(path));
            let status = unsafe {
                RegOpenKeyExW(hive.handle(), path_wide.as_ptr(), 0, KEY_READ, &mut handle)
            };

            if status != ERROR_SUCCESS {
                return Err(format!(
                    "{}\\{}: {}",
                    hive.label(),
                    path,
                    format_windows_error(status)
                ));
            }

            Ok(Self { hive, handle })
        }

        fn enumerate_subkeys(&self) -> Result<Vec<String>, String> {
            let mut result = Vec::new();
            let mut index = 0_u32;

            loop {
                let mut name_buffer = vec![0_u16; 512];
                let mut name_length = name_buffer.len() as u32;
                let status = unsafe {
                    RegEnumKeyExW(
                        self.handle,
                        index,
                        name_buffer.as_mut_ptr(),
                        &mut name_length,
                        ptr::null_mut(),
                        ptr::null_mut(),
                        ptr::null_mut(),
                        ptr::null_mut(),
                    )
                };

                if status == ERROR_NO_MORE_ITEMS {
                    break;
                }

                if status == ERROR_MORE_DATA {
                    return Err(format!(
                        "{} registry key name exceeded the supported buffer size.",
                        self.hive.label()
                    ));
                }

                if status != ERROR_SUCCESS {
                    return Err(format_windows_error(status));
                }

                let name = String::from_utf16_lossy(&name_buffer[..name_length as usize]);
                result.push(name);
                index += 1;
            }

            Ok(result)
        }

        fn query_string_value(&self, value_name: &str) -> Result<String, String> {
            let value_name_wide = to_wide(OsStr::new(value_name));
            let mut value_type = 0_u32;
            let mut data_length = 0_u32;
            let status = unsafe {
                RegQueryValueExW(
                    self.handle,
                    value_name_wide.as_ptr(),
                    ptr::null_mut(),
                    &mut value_type,
                    ptr::null_mut(),
                    &mut data_length,
                )
            };

            if status != ERROR_SUCCESS {
                return Err(format_windows_error(status));
            }

            if value_type != REG_SZ && value_type != REG_EXPAND_SZ {
                return Err(format!(
                    "Registry value '{}' is not a string value.",
                    value_name
                ));
            }

            if data_length == 0 {
                return Ok(String::new());
            }

            let mut data = vec![0_u8; data_length as usize];
            let status = unsafe {
                RegQueryValueExW(
                    self.handle,
                    value_name_wide.as_ptr(),
                    ptr::null_mut(),
                    &mut value_type,
                    data.as_mut_ptr(),
                    &mut data_length,
                )
            };

            if status != ERROR_SUCCESS {
                return Err(format_windows_error(status));
            }

            let raw = wide_string_from_bytes(&data[..data_length as usize])?;
            if value_type == REG_EXPAND_SZ {
                return expand_environment_string(&raw);
            }

            Ok(raw)
        }
    }

    impl Drop for RegistryKey {
        fn drop(&mut self) {
            unsafe {
                RegCloseKey(self.handle);
            }
        }
    }

    fn to_wide(value: &OsStr) -> Vec<u16> {
        value.encode_wide().chain(std::iter::once(0)).collect()
    }

    fn to_wide_filter(parts: &[&str]) -> Vec<u16> {
        let mut wide = Vec::new();
        for part in parts {
            wide.extend(OsStr::new(part).encode_wide());
            wide.push(0);
        }
        wide.push(0);
        wide
    }

    fn wide_string_from_bytes(bytes: &[u8]) -> Result<String, String> {
        if bytes.len() % 2 != 0 {
            return Err("Registry string value returned an invalid byte length.".to_string());
        }

        let mut wide = Vec::with_capacity(bytes.len() / 2);
        for chunk in bytes.chunks_exact(2) {
            wide.push(u16::from_le_bytes([chunk[0], chunk[1]]));
        }

        while wide.last().copied() == Some(0) {
            let last_index = wide.len() - 1;
            wide.remove(last_index);
        }

        String::from_utf16(&wide)
            .map_err(|error| format!("Registry string value contains invalid UTF-16 data: {error}"))
    }

    fn expand_environment_string(value: &str) -> Result<String, String> {
        let value_wide = to_wide(OsStr::new(value));
        let required =
            unsafe { ExpandEnvironmentStringsW(value_wide.as_ptr(), ptr::null_mut(), 0) };

        if required == 0 {
            return Err("Failed to compute expanded environment string size.".to_string());
        }

        let mut buffer = vec![0_u16; required as usize];
        let written = unsafe {
            ExpandEnvironmentStringsW(value_wide.as_ptr(), buffer.as_mut_ptr(), required)
        };

        if written == 0 || written > required {
            return Err("Failed to expand registry environment string.".to_string());
        }

        let content_length = written.saturating_sub(1) as usize;
        Ok(String::from_utf16_lossy(&buffer[..content_length]))
    }

    fn format_windows_error(code: u32) -> String {
        let io_error = std::io::Error::from_raw_os_error(code as i32);
        format!("Windows error {code}: {io_error}")
    }
}

#[cfg(not(windows))]
mod platform {
    use std::path::PathBuf;

    pub fn find_codex_executable_from_registry() -> Result<PathBuf, String> {
        Err(
            "Codex registry discovery is only supported on Windows. Set a custom Codex.exe path."
                .to_string(),
        )
    }

    pub fn find_codex_executable_from_uninstall_registry() -> Vec<PathBuf> {
        Vec::new()
    }

    pub fn browse_codex_executable() -> Result<Option<PathBuf>, String> {
        Err("The native Codex.exe picker is only supported on Windows.".to_string())
    }
}

```


## src-tauri/src/profile.rs

```rust
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Component, Path, PathBuf},
    sync::{Arc, RwLock},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use uuid::Uuid;

const APP_DIRECTORY_NAME: &str = "Aion";
const LEGACY_CONFIG_FILE_NAME: &str = "config.json";
const CONFIG_DIRECTORY_NAME: &str = "config";
const CONFIG_FILE_NAME: &str = "app.json";
const CONFIG_LOCK_FILE_NAME: &str = "config.lock";
const CONFIG_PROFILES_DIRECTORY_NAME: &str = "profiles";
const CONFIG_RUNTIME_DIRECTORY_NAME: &str = "runtime";
const CONFIG_LOGS_DIRECTORY_NAME: &str = "logs";
const CONFIG_SANDBOXES_DIRECTORY_NAME: &str = "sandboxes";
const DEFAULT_COLOR_TAG: &str = "#4F46E5";
const MAX_PROFILE_NAME_CHARS: usize = 80;
const MAX_EMAIL_CHARS: usize = 254;
const MAX_PROXY_HOST_PORT_CHARS: usize = 255;
const MAX_PROXY_USERNAME_CHARS: usize = 128;
const MAX_PROXY_PASSWORD_CHARS: usize = 512;
const MAX_TIMESTAMP_CHARS: usize = 64;
const WEEK_WINDOW_HOURS: f32 = 168.0;
const FIVE_HOUR_WINDOW_HOURS: f32 = 5.0;
const CONFIG_SCHEMA_VERSION: u32 = 1;
const CONFIG_LOCK_RETRIES: usize = 120;
const CONFIG_LOCK_RETRY_DELAY_MS: u64 = 25;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum ProcessStatus {
    Idle,
    Launching,
    Running,
    Stopping,
    Exited,
    Error(String),
}

impl ProcessStatus {
    pub fn is_running(&self) -> bool {
        matches!(self, Self::Running)
    }

    pub fn is_active(&self) -> bool {
        matches!(self, Self::Launching | Self::Running | Self::Stopping)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeState {
    pid: u32,
    status: ProcessStatus,
    is_focused: bool,
    started_at: String,
}

impl RuntimeState {
    pub fn idle() -> Self {
        Self {
            pid: 0,
            status: ProcessStatus::Idle,
            is_focused: false,
            started_at: String::new(),
        }
    }

    pub fn running(pid: u32, started_at: String) -> Self {
        Self {
            pid,
            status: ProcessStatus::Running,
            is_focused: false,
            started_at,
        }
    }

    pub fn pid(&self) -> u32 {
        self.pid
    }

    pub fn status(&self) -> &ProcessStatus {
        &self.status
    }

    pub fn is_focused(&self) -> bool {
        self.is_focused
    }

    pub fn started_at(&self) -> &str {
        &self.started_at
    }

    fn launching() -> Self {
        Self {
            pid: 0,
            status: ProcessStatus::Launching,
            is_focused: false,
            started_at: String::new(),
        }
    }

    fn stopping_from(current: &Self) -> Self {
        Self {
            pid: current.pid,
            status: ProcessStatus::Stopping,
            is_focused: current.is_focused,
            started_at: current.started_at.clone(),
        }
    }

    fn exited_from(current: &Self) -> Self {
        Self {
            pid: 0,
            status: ProcessStatus::Exited,
            is_focused: false,
            started_at: current.started_at.clone(),
        }
    }

    fn error(message: String) -> Self {
        Self {
            pid: 0,
            status: ProcessStatus::Error(message),
            is_focused: false,
            started_at: String::new(),
        }
    }

    fn with_focus(mut self, focused: bool) -> Self {
        self.is_focused = focused;
        self
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "PascalCase")]
pub enum RuntimeEvent {
    StatusChanged {
        id: String,
        from: ProcessStatus,
        to: ProcessStatus,
    },
    ProcessSpawned {
        id: String,
        pid: u32,
    },
    ProcessExited {
        id: String,
        pid: u32,
    },
    ProcessFailed {
        id: String,
        message: String,
    },
}

#[derive(Debug, Clone, Copy)]
enum RuntimeTransition {
    LaunchRequested,
    ProcessSpawned,
    FocusChanged,
    StopRequested,
    ProcessExited,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimePollingMetadata {
    pub launching_ms: u64,
    pub active_ms: u64,
    pub minimized_ms: u64,
}

impl Default for RuntimePollingMetadata {
    fn default() -> Self {
        Self {
            launching_ms: 500,
            active_ms: 2_000,
            minimized_ms: 15_000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeStatusView {
    pub id: String,
    pub status: ProcessStatus,
    pub pid: u32,
    pub is_focused: bool,
}

#[derive(Clone, Debug)]
pub struct RuntimeManager {
    inner: Arc<RwLock<HashMap<Uuid, RuntimeState>>>,
}

impl RuntimeManager {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn mark_launching(&self, profile_id: Uuid) -> Result<RuntimeEvent, String> {
        self.transition(
            profile_id,
            RuntimeState::launching(),
            RuntimeTransition::LaunchRequested,
        )
    }

    pub fn mark_running(
        &self,
        profile_id: Uuid,
        pid: u32,
        started_at: String,
    ) -> Result<Vec<RuntimeEvent>, String> {
        let event = self.transition(
            profile_id,
            RuntimeState::running(pid, started_at),
            RuntimeTransition::ProcessSpawned,
        )?;
        Ok(vec![
            event,
            RuntimeEvent::ProcessSpawned {
                id: profile_id.to_string(),
                pid,
            },
        ])
    }

    pub fn mark_stopping(&self, profile_id: Uuid) -> Result<RuntimeEvent, String> {
        let mut runtime = self
            .inner
            .write()
            .map_err(|_| "Aion runtime state lock was poisoned.".to_string())?;
        let current = runtime
            .get(&profile_id)
            .cloned()
            .unwrap_or_else(RuntimeState::idle);
        let next = RuntimeState::stopping_from(&current);
        apply_runtime_transition(
            &mut runtime,
            profile_id,
            current,
            next,
            RuntimeTransition::StopRequested,
        )
    }

    pub fn mark_error(&self, profile_id: Uuid, message: String) -> Result<RuntimeEvent, String> {
        self.transition(
            profile_id,
            RuntimeState::error(message.clone()),
            RuntimeTransition::ProcessExited,
        )
        .map(|event| match event {
            RuntimeEvent::StatusChanged { .. } => RuntimeEvent::ProcessFailed {
                id: profile_id.to_string(),
                message,
            },
            other => other,
        })
    }

    pub fn update_focus(&self, profile_id: Uuid, focused: bool) -> Result<RuntimeEvent, String> {
        let mut runtime = self
            .inner
            .write()
            .map_err(|_| "Aion runtime state lock was poisoned.".to_string())?;
        let current = runtime
            .get(&profile_id)
            .cloned()
            .unwrap_or_else(RuntimeState::idle);
        let next = current.clone().with_focus(focused);
        apply_runtime_transition(
            &mut runtime,
            profile_id,
            current,
            next,
            RuntimeTransition::FocusChanged,
        )
    }

    fn transition(
        &self,
        profile_id: Uuid,
        next: RuntimeState,
        transition: RuntimeTransition,
    ) -> Result<RuntimeEvent, String> {
        let mut runtime = self
            .inner
            .write()
            .map_err(|_| "Aion runtime state lock was poisoned.".to_string())?;
        let current = runtime
            .get(&profile_id)
            .cloned()
            .unwrap_or_else(RuntimeState::idle);
        apply_runtime_transition(&mut runtime, profile_id, current, next, transition)
    }

    pub fn remove(&self, profile_id: &Uuid) -> Result<(), String> {
        let mut runtime = self
            .inner
            .write()
            .map_err(|_| "Aion runtime state lock was poisoned.".to_string())?;
        runtime.remove(profile_id);
        Ok(())
    }

    pub fn merge_detected(
        &self,
        profile_ids: &[Uuid],
        detected: HashMap<Uuid, RuntimeState>,
    ) -> Result<Vec<RuntimeStatusView>, String> {
        let valid_ids: HashSet<Uuid> = profile_ids.iter().copied().collect();
        let mut runtime = self
            .inner
            .write()
            .map_err(|_| "Aion runtime state lock was poisoned.".to_string())?;

        runtime.retain(|profile_id, _| valid_ids.contains(profile_id));

        for profile_id in profile_ids {
            if let Some(state) = detected.get(profile_id) {
                let current = runtime
                    .get(profile_id)
                    .cloned()
                    .unwrap_or_else(RuntimeState::idle);
                apply_runtime_transition(
                    &mut runtime,
                    *profile_id,
                    current,
                    state.clone(),
                    RuntimeTransition::ProcessSpawned,
                )?;
                continue;
            }

            let current = runtime
                .get(profile_id)
                .cloned()
                .unwrap_or_else(RuntimeState::idle);
            let was_active = current.status().is_active();
            let previous_pid = current.pid();
            let next = if current.status().is_active() {
                RuntimeState::exited_from(&current)
            } else {
                RuntimeState::idle()
            };
            apply_runtime_transition(
                &mut runtime,
                *profile_id,
                current,
                next,
                RuntimeTransition::ProcessExited,
            )?;
            if was_active && previous_pid != 0 {
                let event = RuntimeEvent::ProcessExited {
                    id: profile_id.to_string(),
                    pid: previous_pid,
                };
                tracing::debug!(?event, "runtime process exit detected");
            }
        }

        Ok(profile_ids
            .iter()
            .map(|profile_id| runtime_status_view(*profile_id, runtime.get(profile_id)))
            .collect())
    }

    pub fn statuses_for(&self, profile_ids: &[Uuid]) -> Result<Vec<RuntimeStatusView>, String> {
        let runtime = self
            .inner
            .read()
            .map_err(|_| "Aion runtime state lock was poisoned.".to_string())?;
        Ok(profile_ids
            .iter()
            .map(|profile_id| runtime_status_view(*profile_id, runtime.get(profile_id)))
            .collect())
    }

    pub fn states_for(&self, profile_ids: &[Uuid]) -> Result<HashMap<Uuid, RuntimeState>, String> {
        let runtime = self
            .inner
            .read()
            .map_err(|_| "Aion runtime state lock was poisoned.".to_string())?;
        Ok(profile_ids
            .iter()
            .map(|profile_id| {
                (
                    *profile_id,
                    runtime
                        .get(profile_id)
                        .cloned()
                        .unwrap_or_else(RuntimeState::idle),
                )
            })
            .collect())
    }
}

impl Default for RuntimeManager {
    fn default() -> Self {
        Self::new()
    }
}

fn runtime_status_view(profile_id: Uuid, state: Option<&RuntimeState>) -> RuntimeStatusView {
    let runtime = state.cloned().unwrap_or_else(RuntimeState::idle);
    RuntimeStatusView {
        id: profile_id.to_string(),
        status: runtime.status().clone(),
        pid: runtime.pid(),
        is_focused: runtime.is_focused(),
    }
}

fn apply_runtime_transition(
    runtime: &mut HashMap<Uuid, RuntimeState>,
    profile_id: Uuid,
    current: RuntimeState,
    next: RuntimeState,
    transition: RuntimeTransition,
) -> Result<RuntimeEvent, String> {
    validate_runtime_transition(current.status(), next.status(), transition)?;
    let event = RuntimeEvent::StatusChanged {
        id: profile_id.to_string(),
        from: current.status().clone(),
        to: next.status().clone(),
    };
    let changed = current.pid() != next.pid()
        || current.status() != next.status()
        || current.is_focused() != next.is_focused();
    if changed {
        tracing::debug!(
            profile_id = %profile_id,
            ?transition,
            from = ?current.status(),
            to = ?next.status(),
            "runtime state transition"
        );
    }
    runtime.insert(profile_id, next);
    Ok(event)
}

fn validate_runtime_transition(
    from: &ProcessStatus,
    to: &ProcessStatus,
    transition: RuntimeTransition,
) -> Result<(), String> {
    let allowed = match transition {
        RuntimeTransition::LaunchRequested => {
            matches!(
                from,
                ProcessStatus::Idle | ProcessStatus::Exited | ProcessStatus::Error(_)
            ) && matches!(to, ProcessStatus::Launching)
        }
        RuntimeTransition::ProcessSpawned => {
            matches!(
                from,
                ProcessStatus::Idle
                    | ProcessStatus::Launching
                    | ProcessStatus::Running
                    | ProcessStatus::Exited
                    | ProcessStatus::Error(_)
            ) && matches!(to, ProcessStatus::Running)
        }
        RuntimeTransition::FocusChanged => from == to,
        RuntimeTransition::StopRequested => {
            matches!(from, ProcessStatus::Running | ProcessStatus::Launching)
                && matches!(to, ProcessStatus::Stopping)
        }
        RuntimeTransition::ProcessExited => {
            matches!(
                from,
                ProcessStatus::Idle
                    | ProcessStatus::Launching
                    | ProcessStatus::Running
                    | ProcessStatus::Stopping
                    | ProcessStatus::Exited
                    | ProcessStatus::Error(_)
            ) && matches!(
                to,
                ProcessStatus::Idle | ProcessStatus::Exited | ProcessStatus::Error(_)
            )
        }
    };

    if allowed {
        return Ok(());
    }

    Err(format!(
        "Invalid runtime transition from '{from:?}' to '{to:?}' via '{transition:?}'."
    ))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyConfig {
    #[serde(default, alias = "hostPort")]
    pub host_port: String,
    #[serde(default)]
    pub username: String,
    #[serde(default)]
    pub password: String,
}

impl ProxyConfig {
    pub fn has_credentials(&self) -> bool {
        !self.username.is_empty() || !self.password.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Profile {
    pub id: Uuid,
    pub name: String,
    #[serde(default)]
    pub email: String,
    pub color_tag: String,
    pub created_at: u64,
    pub last_launched: Option<u64>,
    #[serde(default)]
    pub usage_week_hours: f32,
    #[serde(default)]
    pub usage_5h_hours: f32,
    #[serde(default)]
    pub activated_at: Option<String>,
    #[serde(default)]
    pub expires_at: Option<String>,
    #[serde(default)]
    pub proxy: Option<ProxyConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileView {
    pub id: String,
    pub name: String,
    pub email: String,
    pub color_tag: String,
    pub created_at: u64,
    pub last_launched: Option<u64>,
    pub usage_week_hours: f32,
    pub usage_5h_hours: f32,
    pub activated_at: Option<String>,
    pub expires_at: Option<String>,
    pub user_data_dir: String,
    pub status: ProcessStatus,
    pub pid: u32,
    pub is_focused: bool,
    pub running: bool,
    pub proxy_enabled: bool,
    pub proxy_host_port: Option<String>,
    pub proxy_has_credentials: bool,
}

impl ProfileView {
    pub fn from_profile(
        profile: &Profile,
        user_data_dir: &Path,
        runtime_state: Option<&RuntimeState>,
    ) -> Self {
        let running = runtime_state
            .map(|state| state.status().is_running())
            .unwrap_or(false);
        let status = runtime_state
            .map(|state| state.status().clone())
            .unwrap_or(ProcessStatus::Idle);
        let pid = runtime_state.map(RuntimeState::pid).unwrap_or(0);
        let is_focused = runtime_state.map(RuntimeState::is_focused).unwrap_or(false);

        Self {
            id: profile.id.to_string(),
            name: profile.name.clone(),
            email: profile.email.clone(),
            color_tag: profile.color_tag.clone(),
            created_at: profile.created_at,
            last_launched: profile.last_launched,
            usage_week_hours: projected_usage_hours(
                profile.usage_week_hours,
                profile.last_launched,
                running,
                WEEK_WINDOW_HOURS,
            ),
            usage_5h_hours: projected_usage_hours(
                profile.usage_5h_hours,
                profile.last_launched,
                running,
                FIVE_HOUR_WINDOW_HOURS,
            ),
            activated_at: profile.activated_at.clone(),
            expires_at: profile.expires_at.clone(),
            user_data_dir: user_data_dir.to_string_lossy().into_owned(),
            status,
            pid,
            is_focused,
            running,
            proxy_enabled: profile.proxy.is_some(),
            proxy_host_port: profile.proxy.as_ref().map(|proxy| proxy.host_port.clone()),
            proxy_has_credentials: profile
                .proxy
                .as_ref()
                .map(ProxyConfig::has_credentials)
                .unwrap_or(false),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredConfig {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub custom_codex_path: Option<PathBuf>,
    #[serde(default)]
    pub profiles: Vec<Profile>,
}

impl Default for StoredConfig {
    fn default() -> Self {
        Self {
            schema_version: default_schema_version(),
            custom_codex_path: None,
            profiles: Vec::new(),
        }
    }
}

fn default_schema_version() -> u32 {
    CONFIG_SCHEMA_VERSION
}

pub trait ConfigStore: Send + Sync {
    fn sandboxes_root(&self) -> &Path;
    fn acquire_write_guard(&self) -> Result<ConfigWriteGuard, String>;
    fn read_config(&self) -> Result<StoredConfig, String>;
    fn write_config(&self, config: &StoredConfig) -> Result<(), String>;
}

#[derive(Debug, Clone)]
pub struct FileConfigStore {
    config_path: PathBuf,
    legacy_config_path: PathBuf,
    profile_config_dir: PathBuf,
    runtime_dir: PathBuf,
    logs_dir: PathBuf,
    sandboxes_root: PathBuf,
    lock_path: PathBuf,
}

impl FileConfigStore {
    pub fn default_store() -> Result<Self, String> {
        let app_dir = app_config_dir()?;
        let config_dir = app_dir.join(CONFIG_DIRECTORY_NAME);
        let profile_config_dir = config_dir.join(CONFIG_PROFILES_DIRECTORY_NAME);
        let runtime_dir = config_dir.join(CONFIG_RUNTIME_DIRECTORY_NAME);
        let logs_dir = config_dir.join(CONFIG_LOGS_DIRECTORY_NAME);
        let sandboxes_root = config_dir.join(CONFIG_SANDBOXES_DIRECTORY_NAME);

        for directory in [
            &profile_config_dir,
            &runtime_dir,
            &logs_dir,
            &sandboxes_root,
        ] {
            fs::create_dir_all(directory).map_err(|error| {
                format!(
                    "Failed to create Aion storage directory at '{}': {error}",
                    directory.display()
                )
            })?;
        }

        let store = Self {
            config_path: config_dir.join(CONFIG_FILE_NAME),
            legacy_config_path: app_dir.join(LEGACY_CONFIG_FILE_NAME),
            profile_config_dir,
            runtime_dir,
            logs_dir,
            sandboxes_root,
            lock_path: config_dir.join(CONFIG_LOCK_FILE_NAME),
        };
        store.recover_orphaned_tmp_files()?;
        Ok(store)
    }

    fn recover_orphaned_tmp_files(&self) -> Result<(), String> {
        for directory in [&self.profile_config_dir, &self.runtime_dir, &self.logs_dir] {
            remove_orphaned_tmp_files(directory)?;
        }
        Ok(())
    }
}

impl ConfigStore for FileConfigStore {
    fn sandboxes_root(&self) -> &Path {
        &self.sandboxes_root
    }

    fn acquire_write_guard(&self) -> Result<ConfigWriteGuard, String> {
        ConfigWriteGuard::acquire(&self.lock_path)
    }

    fn read_config(&self) -> Result<StoredConfig, String> {
        read_config_store(
            &self.config_path,
            &self.legacy_config_path,
            &self.profile_config_dir,
        )
    }

    fn write_config(&self, config: &StoredConfig) -> Result<(), String> {
        write_config_store(&self.config_path, &self.profile_config_dir, config)
    }
}

#[derive(Debug)]
pub struct ConfigWriteGuard {
    lock_path: PathBuf,
}

impl ConfigWriteGuard {
    fn acquire(lock_path: &Path) -> Result<Self, String> {
        let parent = lock_path.parent().ok_or_else(|| {
            format!(
                "Config lock path '{}' does not have a parent directory.",
                lock_path.display()
            )
        })?;

        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "Failed to create config lock directory '{}': {error}",
                parent.display()
            )
        })?;

        for attempt in 0..CONFIG_LOCK_RETRIES {
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(lock_path)
            {
                Ok(mut file) => {
                    let lock_content = format!(
                        "pid={};created_at={}\n",
                        std::process::id(),
                        iso_timestamp_now()?
                    );
                    file.write_all(lock_content.as_bytes()).map_err(|error| {
                        format!(
                            "Failed to write Aion config lock file '{}': {error}",
                            lock_path.display()
                        )
                    })?;
                    file.sync_all().map_err(|error| {
                        format!(
                            "Failed to sync Aion config lock file '{}': {error}",
                            lock_path.display()
                        )
                    })?;
                    return Ok(Self {
                        lock_path: lock_path.to_path_buf(),
                    });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                    if attempt + 1 == CONFIG_LOCK_RETRIES {
                        return Err(format!(
                            "Aion config is locked by another operation at '{}'. Retry after the current operation finishes.",
                            lock_path.display()
                        ));
                    }
                    thread::sleep(Duration::from_millis(CONFIG_LOCK_RETRY_DELAY_MS));
                }
                Err(error) => {
                    return Err(format!(
                        "Failed to acquire Aion config lock at '{}': {error}",
                        lock_path.display()
                    ));
                }
            }
        }

        Err(format!(
            "Aion config lock acquisition exhausted retries at '{}'.",
            lock_path.display()
        ))
    }
}

impl Drop for ConfigWriteGuard {
    fn drop(&mut self) {
        let remove_result = fs::remove_file(&self.lock_path);
        if let Err(error) = remove_result {
            if error.kind() != io::ErrorKind::NotFound {
                eprintln!(
                    "Aion could not remove config lock file '{}': {error}",
                    self.lock_path.display()
                );
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProfileManager {
    store: FileConfigStore,
}

impl ProfileManager {
    pub fn new_default() -> Result<Self, String> {
        let store = FileConfigStore::default_store()?;
        let manager = Self { store };
        manager.initialize_config()?;
        Ok(manager)
    }

    pub fn profiles(&self) -> Result<Vec<Profile>, String> {
        let config = self.load_valid_config()?;
        Ok(config.profiles)
    }

    pub fn profile_exists(&self, profile_id: &Uuid) -> Result<bool, String> {
        let config = self.load_valid_config()?;
        Ok(config
            .profiles
            .iter()
            .any(|profile| profile.id == *profile_id))
    }

    pub fn profile_directories(&self) -> Result<Vec<(Uuid, PathBuf)>, String> {
        let config = self.load_valid_config()?;
        Ok(config
            .profiles
            .iter()
            .map(|profile| (profile.id, self.profile_user_data_dir(&profile.id)))
            .collect())
    }

    pub fn profile_user_data_dir(&self, profile_id: &Uuid) -> PathBuf {
        self.store.sandboxes_root().join(profile_id.to_string())
    }

    pub fn custom_codex_path(&self) -> Result<Option<PathBuf>, String> {
        let config = self.load_valid_config()?;
        Ok(config.custom_codex_path)
    }

    pub fn create_profile(
        &self,
        name: String,
        email: String,
        color_tag: String,
        proxy: Option<ProxyConfig>,
    ) -> Result<Profile, String> {
        let normalized_name = normalize_profile_name(&name)?;
        let normalized_email = normalize_email(&email, true)?;
        let normalized_color = normalize_color_tag(&color_tag)?;
        let normalized_proxy = normalize_proxy_config(proxy)?;
        let _guard = self.store.acquire_write_guard()?;
        let mut config = self.load_valid_config()?;
        ensure_unique_profile_name(&config, &normalized_name, None)?;
        let timestamp = unix_timestamp_secs()?;
        let profile = Profile {
            id: Uuid::new_v4(),
            name: normalized_name,
            email: normalized_email,
            color_tag: normalized_color,
            created_at: timestamp,
            last_launched: None,
            usage_week_hours: 0.0,
            usage_5h_hours: 0.0,
            activated_at: None,
            expires_at: None,
            proxy: normalized_proxy,
        };

        let user_data_dir = self.profile_user_data_dir(&profile.id);
        fs::create_dir_all(&user_data_dir).map_err(|error| {
            format!(
                "Failed to create user data directory for profile '{}' at '{}': {error}",
                profile.name,
                user_data_dir.display()
            )
        })?;

        config.profiles.push(profile.clone());
        if let Err(error) = self.store.write_config(&config) {
            let cleanup_result = fs::remove_dir(&user_data_dir);
            if let Err(cleanup_error) = cleanup_result {
                return Err(format!(
                    "{error}. Additionally, Aion could not remove the newly-created empty directory '{}': {cleanup_error}",
                    user_data_dir.display()
                ));
            }
            return Err(error);
        }

        Ok(profile)
    }

    pub fn rename_profile(&self, profile_id: &Uuid, new_name: String) -> Result<Profile, String> {
        let normalized_name = normalize_profile_name(&new_name)?;
        let _guard = self.store.acquire_write_guard()?;
        let mut config = self.load_valid_config()?;
        ensure_unique_profile_name(&config, &normalized_name, Some(profile_id))?;
        let index = find_profile_index(&config, profile_id)?;
        config.profiles[index].name = normalized_name;
        let profile = config.profiles[index].clone();
        self.store.write_config(&config)?;
        Ok(profile)
    }

    pub fn clone_profile(
        &self,
        source_profile_id: &Uuid,
        requested_name: Option<String>,
    ) -> Result<Profile, String> {
        let _guard = self.store.acquire_write_guard()?;
        let mut config = self.load_valid_config()?;
        let source = find_profile(&config, source_profile_id)?.clone();
        let clone_name = match requested_name {
            Some(name) => normalize_profile_name(&name)?,
            None => unique_copy_name(&config, &source.name)?,
        };
        ensure_unique_profile_name(&config, &clone_name, None)?;

        let timestamp = unix_timestamp_secs()?;
        let cloned = Profile {
            id: Uuid::new_v4(),
            name: clone_name,
            email: source.email.clone(),
            color_tag: source.color_tag.clone(),
            created_at: timestamp,
            last_launched: None,
            usage_week_hours: 0.0,
            usage_5h_hours: 0.0,
            activated_at: source.activated_at.clone(),
            expires_at: source.expires_at.clone(),
            proxy: source.proxy.clone(),
        };

        let source_dir = self.profile_user_data_dir(&source.id);
        let target_dir = self.profile_user_data_dir(&cloned.id);

        if source_dir.exists() {
            copy_directory_recursively(&source_dir, &target_dir)?;
        } else {
            fs::create_dir_all(&target_dir).map_err(|error| {
                format!(
                    "Failed to create user data directory for cloned profile at '{}': {error}",
                    target_dir.display()
                )
            })?;
        }

        config.profiles.push(cloned.clone());
        if let Err(error) = self.store.write_config(&config) {
            let cleanup_result = fs::remove_dir_all(&target_dir);
            if let Err(cleanup_error) = cleanup_result {
                return Err(format!(
                    "{error}. Additionally, Aion could not remove incomplete cloned data at '{}': {cleanup_error}",
                    target_dir.display()
                ));
            }
            return Err(error);
        }

        Ok(cloned)
    }

    pub fn delete_profile(&self, profile_id: &Uuid) -> Result<(), String> {
        {
            let _guard = self.store.acquire_write_guard()?;
            let mut config = self.load_valid_config()?;
            let index = find_profile_index(&config, profile_id)?;
            config.profiles.remove(index);
            self.store.write_config(&config)?;
        }

        let user_data_dir = self.profile_user_data_dir(profile_id);
        if user_data_dir.exists() {
            fs::remove_dir_all(&user_data_dir).map_err(|error| {
                format!(
                    "Profile was removed from Aion config, but its user data directory '{}' could not be deleted: {error}",
                    user_data_dir.display()
                )
            })?;
        }

        Ok(())
    }

    pub fn ensure_profile_user_data_dir(&self, profile_id: &Uuid) -> Result<PathBuf, String> {
        let user_data_dir = self.profile_user_data_dir(profile_id);
        fs::create_dir_all(&user_data_dir).map_err(|error| {
            format!(
                "Failed to provision user data directory for profile '{}' at '{}': {error}",
                profile_id,
                user_data_dir.display()
            )
        })?;
        Ok(user_data_dir)
    }

    pub fn mark_profile_launched(&self, profile_id: &Uuid) -> Result<Profile, String> {
        let _guard = self.store.acquire_write_guard()?;
        let mut config = self.load_valid_config()?;
        let launched_at = unix_timestamp_secs()?;
        let index = find_profile_index(&config, profile_id)?;
        config.profiles[index].last_launched = Some(launched_at);
        let profile = config.profiles[index].clone();
        self.store.write_config(&config)?;
        Ok(profile)
    }

    pub fn set_custom_codex_path(&self, path: PathBuf) -> Result<PathBuf, String> {
        let _guard = self.store.acquire_write_guard()?;
        let mut config = self.load_valid_config()?;
        config.custom_codex_path = Some(path.clone());
        self.store.write_config(&config)?;
        Ok(path)
    }

    fn initialize_config(&self) -> Result<(), String> {
        let _guard = self.store.acquire_write_guard()?;
        let mut config = self.load_valid_config()?;
        ensure_all_profile_directories(self.store.sandboxes_root(), &config)?;
        config.schema_version = CONFIG_SCHEMA_VERSION;
        self.store.write_config(&config)
    }

    fn load_valid_config(&self) -> Result<StoredConfig, String> {
        let mut config = self.store.read_config()?;
        validate_loaded_config(&mut config)?;
        Ok(config)
    }
}

pub fn parse_profile_id(profile_id: &str) -> Result<Uuid, String> {
    let trimmed = profile_id.trim();
    if trimmed.is_empty() {
        return Err("Profile id cannot be empty.".to_string());
    }

    if trimmed.chars().any(char::is_control) || contains_parent_dir_component(Path::new(trimmed)) {
        return Err(format!(
            "Profile id '{}' contains unsafe characters.",
            trimmed
        ));
    }

    Uuid::parse_str(trimmed)
        .map_err(|error| format!("Profile id '{}' is not a valid UUID: {error}", trimmed))
}

pub fn iso_timestamp_now() -> Result<String, String> {
    let seconds = unix_timestamp_secs()?;
    Ok(iso_timestamp_from_unix(seconds))
}

pub fn iso_timestamp_from_unix(seconds: u64) -> String {
    let days = (seconds / 86_400) as i64;
    let seconds_of_day = seconds % 86_400;
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

fn civil_from_days(days_since_unix_epoch: i64) -> (i64, u32, u32) {
    let z = days_since_unix_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    let year = y + if m <= 2 { 1 } else { 0 };
    (year, m as u32, d as u32)
}

fn app_config_dir() -> Result<PathBuf, String> {
    let appdata = std::env::var_os("APPDATA").ok_or_else(|| {
        "The APPDATA environment variable is not available; Aion cannot locate AppData/Roaming storage.".to_string()
    })?;
    Ok(PathBuf::from(appdata).join(APP_DIRECTORY_NAME))
}

fn read_config_store(
    config_path: &Path,
    legacy_config_path: &Path,
    profile_config_dir: &Path,
) -> Result<StoredConfig, String> {
    let mut config = if config_path.exists() {
        read_config_file(config_path)?
    } else if legacy_config_path.exists() {
        read_config_file(legacy_config_path)?
    } else {
        StoredConfig::default()
    };

    let profile_documents = read_profile_documents(profile_config_dir)?;
    if !profile_documents.is_empty() {
        config.profiles = profile_documents;
    }

    Ok(config)
}

fn write_config_store(
    config_path: &Path,
    profile_config_dir: &Path,
    config: &StoredConfig,
) -> Result<(), String> {
    let app_config = StoredConfig {
        schema_version: CONFIG_SCHEMA_VERSION,
        custom_codex_path: config.custom_codex_path.clone(),
        profiles: Vec::new(),
    };

    sync_profile_documents(profile_config_dir, &config.profiles)?;
    atomic_write_json(config_path, &app_config)
}

fn read_config_file(config_path: &Path) -> Result<StoredConfig, String> {
    if !config_path.exists() {
        return Ok(StoredConfig::default());
    }

    let content = fs::read_to_string(config_path).map_err(|error| {
        format!(
            "Failed to read Aion config file at '{}': {error}",
            config_path.display()
        )
    })?;

    if content.trim().is_empty() {
        return Ok(StoredConfig::default());
    }

    match serde_json::from_str::<StoredConfig>(&content) {
        Ok(config) => Ok(config),
        Err(primary_error) => match serde_json::from_str::<Vec<Profile>>(&content) {
            Ok(profiles) => Ok(StoredConfig {
                schema_version: CONFIG_SCHEMA_VERSION,
                profiles,
                ..StoredConfig::default()
            }),
            Err(_) => Err(format!(
                "Failed to parse Aion config file at '{}': {primary_error}",
                config_path.display()
            )),
        },
    }
}

fn read_profile_documents(profile_config_dir: &Path) -> Result<Vec<Profile>, String> {
    if !profile_config_dir.exists() {
        return Ok(Vec::new());
    }

    let mut entries = fs::read_dir(profile_config_dir)
        .map_err(|error| {
            format!(
                "Failed to read Aion profile config directory '{}': {error}",
                profile_config_dir.display()
            )
        })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| {
            format!(
                "Failed to enumerate Aion profile config directory '{}': {error}",
                profile_config_dir.display()
            )
        })?;

    entries.sort_by_key(|entry| entry.file_name());
    let mut profiles = Vec::new();

    for entry in entries {
        let path = entry.path();
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();

        if file_name.ends_with(".tmp") {
            continue;
        }

        let is_json = path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.eq_ignore_ascii_case("json"))
            .unwrap_or(false);
        if !is_json {
            continue;
        }

        let content = fs::read_to_string(&path).map_err(|error| {
            format!(
                "Failed to read Aion profile document '{}': {error}",
                path.display()
            )
        })?;
        let profile = serde_json::from_str::<Profile>(&content).map_err(|error| {
            format!(
                "Failed to parse Aion profile document '{}': {error}",
                path.display()
            )
        })?;
        profiles.push(profile);
    }

    Ok(profiles)
}

fn sync_profile_documents(profile_config_dir: &Path, profiles: &[Profile]) -> Result<(), String> {
    fs::create_dir_all(profile_config_dir).map_err(|error| {
        format!(
            "Failed to create Aion profile config directory '{}': {error}",
            profile_config_dir.display()
        )
    })?;

    let expected: HashSet<String> = profiles
        .iter()
        .map(|profile| format!("{}.json", profile.id))
        .collect();

    if profile_config_dir.exists() {
        for entry_result in fs::read_dir(profile_config_dir).map_err(|error| {
            format!(
                "Failed to read Aion profile config directory '{}': {error}",
                profile_config_dir.display()
            )
        })? {
            let entry = entry_result.map_err(|error| {
                format!(
                    "Failed to enumerate Aion profile config directory '{}': {error}",
                    profile_config_dir.display()
                )
            })?;
            let path = entry.path();
            let file_name = entry.file_name().to_string_lossy().into_owned();
            let is_profile_json = path
                .extension()
                .and_then(|extension| extension.to_str())
                .map(|extension| extension.eq_ignore_ascii_case("json"))
                .unwrap_or(false);

            if is_profile_json && !expected.contains(&file_name) {
                fs::remove_file(&path).map_err(|error| {
                    format!(
                        "Failed to remove stale Aion profile document '{}': {error}",
                        path.display()
                    )
                })?;
            }
        }
    }

    for profile in profiles {
        let target = profile_config_dir.join(format!("{}.json", profile.id));
        atomic_write_profile_json(&target, profile)?;
    }

    Ok(())
}

fn validate_loaded_config(config: &mut StoredConfig) -> Result<(), String> {
    config.schema_version = CONFIG_SCHEMA_VERSION;

    let mut seen_ids = HashSet::new();
    for profile in &mut config.profiles {
        profile.name = normalize_profile_name(&profile.name)?;
        profile.email = normalize_email(&profile.email, false)?;
        profile.color_tag = normalize_color_tag(&profile.color_tag)?;
        profile.usage_week_hours =
            normalize_usage_hours(profile.usage_week_hours, WEEK_WINDOW_HOURS);
        profile.usage_5h_hours =
            normalize_usage_hours(profile.usage_5h_hours, FIVE_HOUR_WINDOW_HOURS);
        profile.activated_at = normalize_optional_timestamp(profile.activated_at.take())?;
        profile.expires_at = normalize_optional_timestamp(profile.expires_at.take())?;
        profile.proxy = normalize_proxy_config(profile.proxy.take())?;

        if !seen_ids.insert(profile.id) {
            return Err(format!(
                "Aion config contains duplicate profile id '{}'. Refusing to continue to avoid data corruption.",
                profile.id
            ));
        }
    }

    if matches!(
        config.custom_codex_path.as_ref(),
        Some(path) if path.as_os_str().is_empty()
    ) {
        config.custom_codex_path = None;
    }

    Ok(())
}

fn ensure_all_profile_directories(
    profiles_root: &Path,
    config: &StoredConfig,
) -> Result<(), String> {
    for profile in &config.profiles {
        let user_data_dir = profiles_root.join(profile.id.to_string());
        fs::create_dir_all(&user_data_dir).map_err(|error| {
            format!(
                "Failed to provision user data directory for profile '{}' at '{}': {error}",
                profile.name,
                user_data_dir.display()
            )
        })?;
    }
    Ok(())
}

fn find_profile<'a>(config: &'a StoredConfig, profile_id: &Uuid) -> Result<&'a Profile, String> {
    config
        .profiles
        .iter()
        .find(|profile| profile.id == *profile_id)
        .ok_or_else(|| format!("Profile '{}' was not found.", profile_id))
}

fn find_profile_index(config: &StoredConfig, profile_id: &Uuid) -> Result<usize, String> {
    config
        .profiles
        .iter()
        .position(|profile| profile.id == *profile_id)
        .ok_or_else(|| format!("Profile '{}' was not found.", profile_id))
}

fn ensure_unique_profile_name(
    config: &StoredConfig,
    candidate: &str,
    excluded_profile_id: Option<&Uuid>,
) -> Result<(), String> {
    let duplicate = config.profiles.iter().any(|profile| {
        profile.name.eq_ignore_ascii_case(candidate)
            && excluded_profile_id
                .map(|excluded| profile.id != *excluded)
                .unwrap_or(true)
    });

    if duplicate {
        return Err(format!(
            "A managed profile named '{}' already exists. Choose a unique profile name.",
            candidate
        ));
    }

    Ok(())
}

fn unique_copy_name(config: &StoredConfig, source_name: &str) -> Result<String, String> {
    let base = format!("{source_name} Copy");
    if !config
        .profiles
        .iter()
        .any(|profile| profile.name.eq_ignore_ascii_case(&base))
    {
        return normalize_profile_name(&base);
    }

    for index in 2..1000_u16 {
        let candidate = format!("{base} {index}");
        if !config
            .profiles
            .iter()
            .any(|profile| profile.name.eq_ignore_ascii_case(&candidate))
        {
            return normalize_profile_name(&candidate);
        }
    }

    Err("Aion could not generate a unique clone name. Please provide a custom name.".to_string())
}

fn contains_parent_dir_component(path: &Path) -> bool {
    path.components()
        .any(|component| matches!(component, Component::ParentDir))
}

fn normalize_profile_name(input: &str) -> Result<String, String> {
    let name = input.trim();
    if name.is_empty() {
        return Err("Profile name cannot be empty.".to_string());
    }

    if name.chars().count() > MAX_PROFILE_NAME_CHARS {
        return Err(format!(
            "Profile name cannot exceed {MAX_PROFILE_NAME_CHARS} characters."
        ));
    }

    if name.chars().any(char::is_control) {
        return Err("Profile name cannot contain control characters.".to_string());
    }

    Ok(name.to_string())
}

fn normalize_email(input: &str, required: bool) -> Result<String, String> {
    let email = input.trim();
    if email.is_empty() {
        if required {
            return Err("Profile email cannot be empty.".to_string());
        }
        return Ok(String::new());
    }

    if email.chars().count() > MAX_EMAIL_CHARS {
        return Err(format!(
            "Profile email cannot exceed {MAX_EMAIL_CHARS} characters."
        ));
    }

    if !email.is_ascii() {
        return Err("Profile email must contain only ASCII characters.".to_string());
    }

    if email
        .chars()
        .any(|character| character.is_ascii_control() || character.is_ascii_whitespace())
    {
        return Err("Profile email cannot contain whitespace or control characters.".to_string());
    }

    let mut parts = email.split('@');
    let local = parts.next().unwrap_or_default();
    let domain = parts.next().unwrap_or_default();
    if parts.next().is_some() || local.is_empty() || domain.is_empty() {
        return Err("Profile email must be a valid account email address.".to_string());
    }

    if local.starts_with('.') || local.ends_with('.') || local.contains("..") {
        return Err("Profile email local part is malformed.".to_string());
    }

    if domain.starts_with('.') || domain.ends_with('.') || domain.contains("..") {
        return Err("Profile email domain is malformed.".to_string());
    }

    if !domain.contains('.') {
        return Err("Profile email domain must include a top-level domain.".to_string());
    }

    let valid_domain = domain.split('.').all(|label| {
        !label.is_empty()
            && !label.starts_with('-')
            && !label.ends_with('-')
            && label
                .chars()
                .all(|character| character.is_ascii_alphanumeric() || character == '-')
    });

    if !valid_domain {
        return Err("Profile email domain contains invalid characters.".to_string());
    }

    Ok(email.to_string())
}

fn normalize_color_tag(input: &str) -> Result<String, String> {
    let color = input.trim();
    if color.is_empty() {
        return Ok(DEFAULT_COLOR_TAG.to_string());
    }

    if color.len() != 7 || !color.starts_with('#') {
        return Err("Profile color_tag must be a hex color in the form #RRGGBB.".to_string());
    }

    if !color[1..]
        .chars()
        .all(|character| character.is_ascii_hexdigit())
    {
        return Err("Profile color_tag contains invalid hex characters.".to_string());
    }

    Ok(format!("#{}", color[1..].to_ascii_uppercase()))
}

fn normalize_usage_hours(value: f32, cap_hours: f32) -> f32 {
    if !value.is_finite() || value.is_sign_negative() {
        return 0.0;
    }

    value.min(cap_hours)
}

fn normalize_proxy_config(input: Option<ProxyConfig>) -> Result<Option<ProxyConfig>, String> {
    let Some(proxy) = input else {
        return Ok(None);
    };

    let host_port = normalize_proxy_host_port(&proxy.host_port)?;
    let username =
        normalize_proxy_secret(&proxy.username, "Proxy username", MAX_PROXY_USERNAME_CHARS)?;
    let password =
        normalize_proxy_secret(&proxy.password, "Proxy password", MAX_PROXY_PASSWORD_CHARS)?;

    if host_port.is_empty() {
        if username.is_empty() && password.is_empty() {
            return Ok(None);
        }

        return Err("Proxy host:port is required when proxy credentials are provided.".to_string());
    }

    Ok(Some(ProxyConfig {
        host_port,
        username,
        password,
    }))
}

fn normalize_proxy_host_port(input: &str) -> Result<String, String> {
    let host_port = input.trim();
    if host_port.is_empty() {
        return Ok(String::new());
    }

    if host_port.chars().count() > MAX_PROXY_HOST_PORT_CHARS {
        return Err(format!(
            "Proxy host:port cannot exceed {MAX_PROXY_HOST_PORT_CHARS} characters."
        ));
    }

    if !host_port.is_ascii() {
        return Err("Proxy host:port must contain only ASCII characters.".to_string());
    }

    if host_port
        .chars()
        .any(|character| character.is_ascii_control() || character.is_ascii_whitespace())
    {
        return Err("Proxy host:port cannot contain whitespace or control characters.".to_string());
    }

    if host_port.contains("://")
        || host_port.contains('/')
        || host_port.contains('\\')
        || host_port.contains('@')
        || host_port.contains("..")
    {
        return Err("Proxy host:port must be a plain host:port value without schemes, paths, credentials, or traversal segments.".to_string());
    }

    let Some((host, port_text)) = host_port.rsplit_once(':') else {
        return Err("Proxy server must be provided as host:port.".to_string());
    };

    if host.is_empty() || port_text.is_empty() {
        return Err("Proxy host and port cannot be empty.".to_string());
    }

    let port = port_text
        .parse::<u16>()
        .map_err(|_| "Proxy port must be a number between 1 and 65535.".to_string())?;
    if port == 0 {
        return Err("Proxy port must be greater than 0.".to_string());
    }

    let valid_host = host.chars().all(|character| {
        character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_' | '[' | ']')
    });
    if !valid_host {
        return Err("Proxy host contains unsupported characters.".to_string());
    }

    Ok(host_port.to_string())
}

fn normalize_proxy_secret(input: &str, label: &str, max_chars: usize) -> Result<String, String> {
    let value = input.trim();
    if value.is_empty() {
        return Ok(String::new());
    }

    if value.chars().count() > max_chars {
        return Err(format!("{label} cannot exceed {max_chars} characters."));
    }

    if !value.is_ascii() {
        return Err(format!("{label} must contain only ASCII characters."));
    }

    if value.chars().any(char::is_control) {
        return Err(format!("{label} cannot contain control characters."));
    }

    Ok(value.to_string())
}

fn normalize_optional_timestamp(input: Option<String>) -> Result<Option<String>, String> {
    let Some(value) = input else {
        return Ok(None);
    };

    let timestamp = value.trim();
    if timestamp.is_empty() {
        return Ok(None);
    }

    if timestamp.chars().count() > MAX_TIMESTAMP_CHARS {
        return Err(format!(
            "Profile timestamp '{}' exceeds {MAX_TIMESTAMP_CHARS} characters.",
            timestamp
        ));
    }

    if timestamp.chars().any(char::is_control) {
        return Err(format!(
            "Profile timestamp '{}' contains control characters.",
            timestamp
        ));
    }

    Ok(Some(timestamp.to_string()))
}

fn projected_usage_hours(
    stored_hours: f32,
    last_launched: Option<u64>,
    running: bool,
    cap_hours: f32,
) -> f32 {
    let base = normalize_usage_hours(stored_hours, cap_hours);
    if !running {
        return base;
    }

    let Some(last_launched_at) = last_launched else {
        return base;
    };

    let Ok(now) = unix_timestamp_secs() else {
        return base;
    };

    let elapsed_hours = now.saturating_sub(last_launched_at) as f32 / 3600.0;
    normalize_usage_hours(base + elapsed_hours, cap_hours)
}

fn unix_timestamp_secs() -> Result<u64, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|error| format!("System clock is before the Unix epoch: {error}"))
}

fn unix_timestamp_nanos() -> Result<u128, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .map_err(|error| format!("System clock is before the Unix epoch: {error}"))
}

fn atomic_write_json<T>(target_path: &Path, value: &T) -> Result<(), String>
where
    T: Serialize,
{
    let parent = target_path.parent().ok_or_else(|| {
        format!(
            "Config path '{}' does not have a parent directory.",
            target_path.display()
        )
    })?;

    fs::create_dir_all(parent).map_err(|error| {
        format!(
            "Failed to create config directory '{}': {error}",
            parent.display()
        )
    })?;

    let json = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("Failed to serialize Aion config: {error}"))?;

    let target_file_name = target_path
        .file_name()
        .and_then(|name| name.to_str())
        .map(str::to_owned)
        .unwrap_or_else(|| CONFIG_FILE_NAME.to_string());
    let temp_name = format!(
        "{target_file_name}.tmp.{}.{}",
        std::process::id(),
        unix_timestamp_nanos()?
    );
    let temp_path = target_path.with_file_name(temp_name);

    write_all_and_sync(&temp_path, &json).map_err(|error| {
        format!(
            "Failed to write temporary Aion config file '{}': {error}",
            temp_path.display()
        )
    })?;

    if let Err(error) = replace_file_atomic(&temp_path, target_path) {
        let cleanup_result = fs::remove_file(&temp_path);
        if let Err(cleanup_error) = cleanup_result {
            return Err(format!(
                "Failed to atomically replace Aion config file '{}': {error}. Temporary file cleanup also failed for '{}': {cleanup_error}",
                target_path.display(),
                temp_path.display()
            ));
        }

        return Err(format!(
            "Failed to atomically replace Aion config file '{}': {error}",
            target_path.display()
        ));
    }

    Ok(())
}

fn atomic_write_profile_json<T>(target_path: &Path, value: &T) -> Result<(), String>
where
    T: Serialize,
{
    let parent = target_path.parent().ok_or_else(|| {
        format!(
            "Profile config path '{}' does not have a parent directory.",
            target_path.display()
        )
    })?;

    fs::create_dir_all(parent).map_err(|error| {
        format!(
            "Failed to create profile config directory '{}': {error}",
            parent.display()
        )
    })?;

    let json = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("Failed to serialize Aion profile document: {error}"))?;
    let target_file_name = target_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("Invalid profile config path '{}'.", target_path.display()))?;
    let temp_path = target_path.with_file_name(format!("{target_file_name}.tmp"));

    if temp_path.exists() {
        fs::remove_file(&temp_path).map_err(|error| {
            format!(
                "Failed to remove stale temporary profile config '{}': {error}",
                temp_path.display()
            )
        })?;
    }

    write_all_and_sync(&temp_path, &json).map_err(|error| {
        format!(
            "Failed to write temporary Aion profile config file '{}': {error}",
            temp_path.display()
        )
    })?;

    if let Err(error) = replace_file_atomic(&temp_path, target_path) {
        let cleanup_result = fs::remove_file(&temp_path);
        if let Err(cleanup_error) = cleanup_result {
            return Err(format!(
                "Failed to atomically replace Aion profile config '{}': {error}. Temporary cleanup also failed for '{}': {cleanup_error}",
                target_path.display(),
                temp_path.display()
            ));
        }

        return Err(format!(
            "Failed to atomically replace Aion profile config '{}': {error}",
            target_path.display()
        ));
    }

    Ok(())
}

fn write_all_and_sync(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.write_all(b"\n")?;
    file.flush()?;
    file.sync_all()?;
    Ok(())
}

fn remove_orphaned_tmp_files(directory: &Path) -> Result<(), String> {
    if !directory.exists() {
        return Ok(());
    }

    for entry_result in fs::read_dir(directory).map_err(|error| {
        format!(
            "Failed to scan Aion temporary file directory '{}': {error}",
            directory.display()
        )
    })? {
        let entry = entry_result.map_err(|error| {
            format!(
                "Failed to enumerate Aion temporary file directory '{}': {error}",
                directory.display()
            )
        })?;
        let path = entry.path();
        let is_tmp = path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.ends_with(".tmp"))
            .unwrap_or(false);

        if is_tmp && path.is_file() {
            fs::remove_file(&path).map_err(|error| {
                format!(
                    "Failed to clean orphaned Aion temporary file '{}': {error}",
                    path.display()
                )
            })?;
        }
    }

    Ok(())
}

fn copy_directory_recursively(source: &Path, target: &Path) -> Result<(), String> {
    let metadata = fs::symlink_metadata(source).map_err(|error| {
        format!(
            "Failed to inspect source profile data directory '{}': {error}",
            source.display()
        )
    })?;

    if metadata.file_type().is_symlink() {
        return Err(format!(
            "Refusing to clone profile data from symlinked directory '{}'.",
            source.display()
        ));
    }

    if !metadata.is_dir() {
        return Err(format!(
            "Source profile data path '{}' is not a directory.",
            source.display()
        ));
    }

    if target.exists() {
        return Err(format!(
            "Target clone directory '{}' already exists.",
            target.display()
        ));
    }

    fs::create_dir_all(target).map_err(|error| {
        format!(
            "Failed to create target clone directory '{}': {error}",
            target.display()
        )
    })?;

    copy_directory_contents(source, target)
}

fn copy_directory_contents(source: &Path, target: &Path) -> Result<(), String> {
    let entries = fs::read_dir(source).map_err(|error| {
        format!(
            "Failed to read source profile data directory '{}': {error}",
            source.display()
        )
    })?;

    for entry_result in entries {
        let entry = entry_result.map_err(|error| {
            format!(
                "Failed to read an entry from profile data directory '{}': {error}",
                source.display()
            )
        })?;
        let source_path = entry.path();
        let target_path = target.join(entry.file_name());
        let metadata = fs::symlink_metadata(&source_path).map_err(|error| {
            format!(
                "Failed to inspect profile data path '{}': {error}",
                source_path.display()
            )
        })?;

        if metadata.file_type().is_symlink() {
            return Err(format!(
                "Refusing to clone symlinked profile data path '{}'.",
                source_path.display()
            ));
        }

        if metadata.is_dir() {
            fs::create_dir_all(&target_path).map_err(|error| {
                format!(
                    "Failed to create cloned profile subdirectory '{}': {error}",
                    target_path.display()
                )
            })?;
            copy_directory_contents(&source_path, &target_path)?;
        } else if metadata.is_file() {
            fs::copy(&source_path, &target_path).map_err(|error| {
                format!(
                    "Failed to copy profile data file '{}' to '{}': {error}",
                    source_path.display(),
                    target_path.display()
                )
            })?;
        }
    }

    Ok(())
}

#[cfg(windows)]
fn replace_file_atomic(source: &Path, target: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    fn to_wide(path: &Path) -> Vec<u16> {
        path.as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect()
    }

    let source_wide = to_wide(source);
    let target_wide = to_wide(target);
    let moved = unsafe {
        MoveFileExW(
            source_wide.as_ptr(),
            target_wide.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };

    if moved == 0 {
        return Err(io::Error::last_os_error());
    }

    Ok(())
}

#[cfg(not(windows))]
fn replace_file_atomic(source: &Path, target: &Path) -> io::Result<()> {
    fs::rename(source, target)
}

```

