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
pub fn stop_profile(
    profiles: State<'_, ProfileManager>,
    runtime: State<'_, RuntimeManager>,
    process_provider: State<'_, CodexProcessProvider>,
    profile_id: String,
) -> Result<Vec<KillResult>, String> {
    let parsed_profile_id = parse_profile_id(&profile_id)?;
    if !profiles.profile_exists(&parsed_profile_id)? {
        return Err(format!("Profile '{}' was not found.", profile_id));
    }

    let user_data_dir = profiles.profile_user_data_dir(&parsed_profile_id);
    runtime.mark_stopping(parsed_profile_id)?;

    let profile_dirs = vec![(parsed_profile_id, user_data_dir)];
    let killed = process_provider.kill_profile_instances(&profile_dirs)?;
    let detected = process_provider.detect_running_profiles(&profile_dirs)?;
    runtime.merge_detected(&[parsed_profile_id], detected)?;

    Ok(killed)
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
