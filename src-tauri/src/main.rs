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
            commands::stop_profile,
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
