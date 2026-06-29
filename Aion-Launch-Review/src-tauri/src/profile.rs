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
const AION_PROFILES_DIRECTORY_NAME: &str = "CodexProfiles";
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
        let sandboxes_root = app_sandboxes_dir()?;

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
        let new_sandboxes_root = store.sandboxes_root().to_path_buf();

        #[cfg(windows)]
        {
            let legacy_roots = legacy_app_sandboxes_dirs();
            for old_root in legacy_roots {
                if old_root != new_sandboxes_root && old_root.exists() {
                    if let Ok(raw_config) = store.read_config() {
                        let ids: Vec<Uuid> = raw_config
                            .profiles
                            .iter()
                            .map(|profile| profile.id)
                            .collect();
                        try_migrate_sandboxes(&old_root, &new_sandboxes_root, &ids);
                    }
                }
            }
        }

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

        config.profiles.push(profile.clone());
        self.store.write_config(&config)?;
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
        }

        config.profiles.push(cloned.clone());
        if let Err(error) = self.store.write_config(&config) {
            if target_dir.exists() {
                let cleanup_result = fs::remove_dir_all(&target_dir);
                if let Err(cleanup_error) = cleanup_result {
                    return Err(format!(
                        "{error}. Additionally, Aion could not remove incomplete cloned data at '{}': {cleanup_error}",
                        target_dir.display()
                    ));
                }
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
        Ok(self.profile_user_data_dir(profile_id))
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

#[cfg(windows)]
fn app_sandboxes_dir() -> Result<PathBuf, String> {
    let userprofile = std::env::var_os("USERPROFILE").ok_or_else(|| {
        "USERPROFILE is not set; Aion cannot locate the user home directory for profile isolation."
            .to_string()
    })?;
    let root = PathBuf::from(&userprofile);
    if root.as_os_str().is_empty() {
        return Err("USERPROFILE is empty; cannot construct a safe profile path.".to_string());
    }
    if !root.is_absolute() {
        return Err(format!(
            "USERPROFILE '{}' is not an absolute path; Aion cannot construct a safe profile path.",
            root.display()
        ));
    }
    Ok(root.join(AION_PROFILES_DIRECTORY_NAME))
}

#[cfg(not(windows))]
fn app_sandboxes_dir() -> Result<PathBuf, String> {
    let app_dir = app_config_dir()?;
    Ok(app_dir
        .join(CONFIG_DIRECTORY_NAME)
        .join(CONFIG_SANDBOXES_DIRECTORY_NAME))
}

#[cfg(windows)]
fn legacy_app_sandboxes_dirs() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    if let Some(appdata) = std::env::var_os("APPDATA") {
        let root = PathBuf::from(appdata);
        if !root.as_os_str().is_empty() {
            roots.push(
                root.join(APP_DIRECTORY_NAME)
                    .join(CONFIG_DIRECTORY_NAME)
                    .join(CONFIG_SANDBOXES_DIRECTORY_NAME),
            );
        }
    }

    roots
}

#[cfg(windows)]
fn try_migrate_sandboxes(old_root: &Path, new_root: &Path, profile_ids: &[Uuid]) {
    for profile_id in profile_ids {
        let old_sandbox = old_root.join(profile_id.to_string());
        let new_sandbox = new_root.join(profile_id.to_string());
        if !old_sandbox.exists() || new_sandbox.exists() {
            continue;
        }
        if std::fs::rename(&old_sandbox, &new_sandbox).is_ok() {
            continue;
        }
        if copy_directory_recursively(&old_sandbox, &new_sandbox).is_ok() {
            let _ = std::fs::remove_dir_all(&old_sandbox);
        }
    }
}

#[cfg(not(windows))]
fn try_migrate_sandboxes(_old_root: &Path, _new_root: &Path, _profile_ids: &[Uuid]) {}

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
