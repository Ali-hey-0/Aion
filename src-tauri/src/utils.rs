use serde::Serialize;
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Component, Path, PathBuf},
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
const WINDOW_TITLE_RETRY_COUNT: usize = 40;
const WINDOW_TITLE_RETRY_DELAY_MS: u64 = 250;
#[cfg(windows)]
const CHROMIUM_PROFILE_DIRECTORY_NAME: &str = "Chromium";
#[cfg(windows)]
const ELECTRON_USER_DATA_DIRECTORY_NAME: &str = "ElectronUserData";

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
        profile_id: Uuid,
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
    #[cfg(not(windows))]
    children: Arc<Mutex<HashMap<u32, TrackedChild>>>,
    #[cfg(windows)]
    jobs: Arc<Mutex<HashMap<Uuid, TrackedJob>>>,
}

#[cfg(not(windows))]
type TrackedChild = std::process::Child;

#[cfg(windows)]
#[derive(Debug)]
struct TrackedJob {
    pid: u32,
    _handle: JobHandle,
}

#[cfg(windows)]
#[derive(Debug)]
struct JobHandle(windows_sys::Win32::Foundation::HANDLE);

#[cfg(windows)]
unsafe impl Send for JobHandle {}

#[cfg(windows)]
impl Drop for JobHandle {
    fn drop(&mut self) {
        if self.0.is_null() {
            return;
        }

        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(self.0);
        }
        self.0 = std::ptr::null_mut();
    }
}

impl ProcessManager {
    #[cfg(windows)]
    fn track_with_job(
        &self,
        profile_id: Uuid,
        mut child: std::process::Child,
    ) -> Result<u32, String> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::HANDLE;
        use windows_sys::Win32::System::JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
            SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        };

        let pid = child.id();
        let process_handle = child.as_raw_handle() as HANDLE;
        let job_handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if job_handle.is_null() {
            let error = std::io::Error::last_os_error();
            let _ = child.kill();
            return Err(format!(
                "Failed to create a Win32 Job Object for Codex profile '{}': {error}",
                profile_id
            ));
        }

        let job_handle = JobHandle(job_handle);
        let mut limit_info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        limit_info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

        let set_result = unsafe {
            SetInformationJobObject(
                job_handle.0,
                JobObjectExtendedLimitInformation,
                &limit_info as *const _ as *const core::ffi::c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        };

        if set_result == 0 {
            let error = std::io::Error::last_os_error();
            let _ = child.kill();
            return Err(format!(
                "Failed to configure the Codex Job Object for profile '{}': {error}",
                profile_id
            ));
        }

        let assign_result = unsafe { AssignProcessToJobObject(job_handle.0, process_handle) };
        if assign_result == 0 {
            let error = std::io::Error::last_os_error();
            let _ = child.kill();
            return Err(format!(
                "Failed to assign Codex process {pid} to the profile Job Object '{}': {error}",
                profile_id
            ));
        }

        drop(child);

        let mut jobs = self
            .jobs
            .lock()
            .map_err(|_| "Aion process manager job lock was poisoned.".to_string())?;
        if let Some(previous) = jobs.remove(&profile_id) {
            tracing::debug!(
                profile_id = %profile_id,
                pid = previous.pid,
                "replacing stale Codex Job Object for profile"
            );
            drop(previous);
        }
        jobs.insert(
            profile_id,
            TrackedJob {
                pid,
                _handle: job_handle,
            },
        );

        Ok(pid)
    }

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

    #[cfg(windows)]
    fn release(&self, pid: u32) -> Result<(), String> {
        let mut jobs = self
            .jobs
            .lock()
            .map_err(|_| "Aion process manager job lock was poisoned.".to_string())?;
        jobs.retain(|_, job| job.pid != pid);
        Ok(())
    }

    #[cfg(not(windows))]
    fn release(&self, pid: u32) -> Result<(), String> {
        let mut children = self
            .children
            .lock()
            .map_err(|_| "Aion process manager lock was poisoned.".to_string())?;
        children.remove(&pid);
        Ok(())
    }

    #[cfg(windows)]
    fn close_jobs_for_profiles(
        &self,
        profile_dirs: &[(Uuid, PathBuf)],
    ) -> Result<Vec<KillResult>, String> {
        let mut jobs = self
            .jobs
            .lock()
            .map_err(|_| "Aion process manager job lock was poisoned.".to_string())?;
        let mut result = Vec::new();

        for (profile_id, _) in profile_dirs {
            if let Some(job) = jobs.remove(profile_id) {
                let pid = job.pid;
                drop(job);
                result.push(KillResult {
                    id: profile_id.to_string(),
                    pid,
                    killed: true,
                });
            }
        }

        Ok(result)
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
        profile_id: Uuid,
        codex_executable: &Path,
        user_data_dir: &Path,
        options: &LaunchOptions,
    ) -> Result<LaunchResult, String> {
        let executable = validate_codex_executable(codex_executable)?;
        let sandbox_path = user_data_dir;

        let safe_sandbox_path = validate_profile_sandbox_launch_path(sandbox_path)?;

        let pid = self.launch_isolated_codex_process(
            profile_id,
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
        #[cfg(windows)]
        let mut killed = self.process_manager.close_jobs_for_profiles(profile_dirs)?;
        #[cfg(not(windows))]
        let mut killed = Vec::new();

        let mut seen = killed
            .iter()
            .map(|result| (result.id.clone(), result.pid))
            .collect::<HashSet<_>>();

        for result in kill_profile_instances(profile_dirs) {
            if seen.insert((result.id.clone(), result.pid)) {
                killed.push(result);
            }
        }

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
        profile_id: Uuid,
        executable: &Path,
        absolute_sandbox_path: &Path,
        _proxy: Option<&ProxyConfig>,
    ) -> Result<u32, String> {
        use std::process::{Command, Stdio};

        prepare_codex_home_for_launch(absolute_sandbox_path)?;
        let layout = prepare_direct_spawn_profile_layout(absolute_sandbox_path)?;

        let mut command = Command::new(executable);
        command
            .arg(format!(
                "--user-data-dir={}",
                path_for_process_argument(&layout.chromium_dir)
            ))
            .env("CODEX_HOME", path_for_environment_value(&layout.root))
            .env(
                "CODEX_ELECTRON_USER_DATA_PATH",
                path_for_environment_value(&layout.electron_user_data_dir),
            )
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .stdin(Stdio::null());

        let child = command.spawn().map_err(|error| {
            format!(
                "Failed to launch Codex directly from '{}' with profile root '{}': {error}",
                executable.display(),
                layout.root.display()
            )
        })?;

        self.process_manager.track_with_job(profile_id, child)
    }

    #[cfg(not(windows))]
    fn launch_isolated_codex_process(
        &self,
        _profile_id: Uuid,
        executable: &Path,
        sandbox_path: &Path,
        proxy: Option<&ProxyConfig>,
    ) -> Result<u32, String> {
        use std::process::Command;

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

#[cfg(windows)]
fn validate_profile_sandbox_launch_path(sandbox_path: &Path) -> Result<PathBuf, String> {
    reject_path_traversal(sandbox_path, "profile sandbox directory")?;
    if !sandbox_path.is_absolute() {
        return Err(format!(
            "Profile sandbox directory '{}' must be an absolute path.",
            sandbox_path.display()
        ));
    }

    let parent = sandbox_path.parent().ok_or_else(|| {
        format!(
            "Profile sandbox directory '{}' does not have a parent directory.",
            sandbox_path.display()
        )
    })?;

    let safe_parent = canonicalize_existing_path(parent, "profile sandbox parent directory")?;
    ensure_directory_is_not_reparse_point(&safe_parent, "profile sandbox parent directory")?;

    if sandbox_path.exists() {
        let safe_sandbox_path =
            canonicalize_existing_path(sandbox_path, "profile sandbox directory")?;
        ensure_directory_is_not_reparse_point(&safe_sandbox_path, "profile sandbox directory")?;
        return Ok(safe_sandbox_path);
    }

    let leaf = sandbox_path.file_name().ok_or_else(|| {
        format!(
            "Profile sandbox directory '{}' does not have a valid directory name.",
            sandbox_path.display()
        )
    })?;
    Ok(safe_parent.join(leaf))
}

#[cfg(windows)]
fn prepare_codex_home_for_launch(codex_home: &Path) -> Result<(), String> {
    if !codex_home.exists() {
        return Ok(());
    }

    ensure_directory_is_not_reparse_point(codex_home, "Codex profile home")?;
    remove_legacy_profile_local_launcher(codex_home)?;

    let config_path = codex_home.join("config.toml");
    if config_path.exists() {
        repair_aion_minimal_codex_config(&config_path)?;
    }

    Ok(())
}

#[cfg(windows)]
#[derive(Debug)]
struct DirectSpawnProfileLayout {
    root: PathBuf,
    chromium_dir: PathBuf,
    electron_user_data_dir: PathBuf,
}

#[cfg(windows)]
fn prepare_direct_spawn_profile_layout(root: &Path) -> Result<DirectSpawnProfileLayout, String> {
    let root = PathBuf::from(path_for_environment_value(root));
    let chromium_dir = root.join(CHROMIUM_PROFILE_DIRECTORY_NAME);
    let electron_user_data_dir = root.join(ELECTRON_USER_DATA_DIRECTORY_NAME);
    let roaming_dir = root.join("AppData").join("Roaming");
    let local_dir = root.join("AppData").join("Local");
    let user_dir = root.join("User");

    for (label, directory) in [
        ("Codex profile root", &root),
        ("Chromium user data directory", &chromium_dir),
        ("Electron user data directory", &electron_user_data_dir),
        ("profile AppData roaming directory", &roaming_dir),
        ("profile AppData local directory", &local_dir),
        ("profile user directory", &user_dir),
    ] {
        fs::create_dir_all(directory).map_err(|error| {
            format!(
                "Failed to create {label} '{}': {error}",
                directory.display()
            )
        })?;
        ensure_directory_is_not_reparse_point(directory, label)?;
    }

    Ok(DirectSpawnProfileLayout {
        root,
        chromium_dir,
        electron_user_data_dir,
    })
}

#[cfg(windows)]
fn path_for_process_argument(path: &Path) -> String {
    normal_windows_path_text_from_str(&path.to_string_lossy())
}

#[cfg(windows)]
fn path_for_environment_value(path: &Path) -> String {
    normal_windows_path_text_from_str(&path.to_string_lossy())
}

#[cfg(windows)]
fn remove_legacy_profile_local_launcher(codex_home: &Path) -> Result<(), String> {
    let stale_launcher = codex_home.join("aion-launch.cmd");
    if !stale_launcher.exists() {
        return Ok(());
    }

    fs::remove_file(&stale_launcher).map_err(|error| {
        format!(
            "Failed to remove stale profile-local Codex launch script '{}': {error}",
            stale_launcher.display()
        )
    })
}

#[cfg(windows)]
fn repair_aion_minimal_codex_config(config_path: &Path) -> Result<(), String> {
    let content = fs::read_to_string(config_path).map_err(|error| {
        format!(
            "Failed to inspect Codex profile config '{}': {error}",
            config_path.display()
        )
    })?;

    if !is_aion_minimal_codex_config(&content) {
        return Ok(());
    }

    let backup_path = config_path.with_file_name("config.toml.aion-minimal.bak");
    if backup_path.exists() {
        fs::remove_file(&backup_path).map_err(|error| {
            format!(
                "Failed to replace stale Codex config backup '{}': {error}",
                backup_path.display()
            )
        })?;
    }

    fs::rename(config_path, &backup_path).map_err(|error| {
        format!(
            "Failed to move Aion-generated minimal Codex config '{}' to '{}': {error}",
            config_path.display(),
            backup_path.display()
        )
    })
}

#[cfg(windows)]
fn is_aion_minimal_codex_config(content: &str) -> bool {
    let meaningful_lines = content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect::<Vec<_>>();

    meaningful_lines.len() == 3
        && meaningful_lines[0] == "[mcp_servers.node_repl.env]"
        && meaningful_lines
            .iter()
            .any(|line| line.starts_with("CODEX_HOME ="))
        && meaningful_lines
            .iter()
            .any(|line| line.starts_with("NODE_REPL_TRUSTED_CODE_PATHS ="))
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

#[cfg(not(windows))]
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

#[cfg(not(windows))]
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

#[cfg(not(windows))]
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
