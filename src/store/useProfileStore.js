import { useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { create } from "zustand";

const DEFAULT_POLLING_METADATA = {
  launchingMs: 500,
  activeMs: 2000,
  minimizedMs: 15000,
};
const DEFAULT_COLOR_TAG = "#4F46E5";
const WEEK_LIMIT_HOURS = 168;
const FIVE_HOUR_LIMIT_HOURS = 5;

const toNumber = (value, fallback = 0) => {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric >= 0 ? numeric : fallback;
};

const parseStatus = (value) => {
  if (value && typeof value === "object") {
    if (typeof value.kind === "string") {
      return {
        name: value.kind,
        message: value.message ? String(value.message) : "",
      };
    }

    if (Object.prototype.hasOwnProperty.call(value, "Error")) {
      return {
        name: "Error",
        message: value.Error ? String(value.Error) : "",
      };
    }
  }

  return { name: String(value ?? "Idle"), message: "" };
};

const normalizeStatus = (value) => {
  const status = parseStatus(value).name;
  if (status.toLowerCase() === "running") {
    return "Running";
  }
  if (status.toLowerCase() === "launching") {
    return "Launching";
  }
  if (status.toLowerCase() === "stopping") {
    return "Stopping";
  }
  if (status.toLowerCase() === "exited") {
    return "Exited";
  }
  if (status.toLowerCase() === "error") {
    return "Error";
  }
  return "Idle";
};

const statusIsRunning = (status) => normalizeStatus(status) === "Running";
const statusIsVolatile = (status) => ["Launching", "Stopping"].includes(normalizeStatus(status));

const normalizeProfile = (profile) => {
  const status = normalizeStatus(profile.status ?? (profile.running ? "Running" : "Idle"));
  const running = statusIsRunning(status) || Boolean(profile.running);
  const statusMessage = parseStatus(profile.status).message;
  const usageWeekHours = toNumber(
    profile.usage_week_hours ?? profile.usageWeekHours,
  );
  const usage5hHours = toNumber(profile.usage_5h_hours ?? profile.usage5hHours);

  return {
    id: String(profile.id ?? ""),
    name: String(profile.name ?? "Unnamed Profile"),
    email: String(profile.email ?? ""),
    colorTag: String(profile.color_tag ?? profile.colorTag ?? DEFAULT_COLOR_TAG),
    createdAt: toNumber(profile.created_at ?? profile.createdAt),
    lastLaunched:
      profile.last_launched ?? profile.lastLaunched ?? profile.lastLaunchedAt ?? null,
    usageWeekHours,
    usage5hHours,
    activatedAt: profile.activated_at ?? profile.activatedAt ?? null,
    expiresAt: profile.expires_at ?? profile.expiresAt ?? null,
    userDataDir: String(profile.user_data_dir ?? profile.userDataDir ?? ""),
    pid: toNumber(profile.pid, 0),
    isFocused: Boolean(profile.is_focused ?? profile.isFocused),
    running,
    status,
    statusMessage,
    proxyEnabled: Boolean(profile.proxy_enabled ?? profile.proxyEnabled),
    proxyHostPort: profile.proxy_host_port ?? profile.proxyHostPort ?? null,
    proxyHasCredentials: Boolean(
      profile.proxy_has_credentials ?? profile.proxyHasCredentials,
    ),
    runtimeUpdatedAt: Date.now(),
  };
};

const normalizeRuntimeStatus = (status) => {
  const normalizedStatus = normalizeStatus(status.status);
  const statusMessage = parseStatus(status.status).message;
  return {
    id: String(status.id ?? ""),
    status: normalizedStatus,
    pid: toNumber(status.pid, 0),
    isFocused: Boolean(status.is_focused ?? status.isFocused),
    running: statusIsRunning(normalizedStatus),
    statusMessage,
  };
};

const normalizePollingMetadata = (metadata) => ({
  launchingMs: toNumber(metadata?.launching_ms ?? metadata?.launchingMs, DEFAULT_POLLING_METADATA.launchingMs),
  activeMs: toNumber(metadata?.active_ms ?? metadata?.activeMs, DEFAULT_POLLING_METADATA.activeMs),
  minimizedMs: toNumber(metadata?.minimized_ms ?? metadata?.minimizedMs, DEFAULT_POLLING_METADATA.minimizedMs),
});

const normalizeDiscovery = (discovery) => ({
  path: discovery?.path ? String(discovery.path) : "",
  source: String(discovery?.source ?? "Unknown"),
  autoDetected: Boolean(discovery?.auto_detected ?? discovery?.autoDetected),
  message: String(discovery?.message ?? ""),
});

const sortProfiles = (profiles) =>
  [...profiles].sort((left, right) => {
    if (left.running !== right.running) {
      return left.running ? -1 : 1;
    }

    return left.name.localeCompare(right.name, undefined, {
      sensitivity: "base",
      numeric: true,
    });
  });

const selectedIdFrom = (profiles, currentSelectedId) => {
  if (profiles.length === 0) {
    return null;
  }

  if (currentSelectedId && profiles.some((profile) => profile.id === currentSelectedId)) {
    return currentSelectedId;
  }

  return profiles[0].id;
};

const getErrorMessage = (error, fallback) => {
  if (typeof error === "string" && error.trim()) {
    return error;
  }

  if (error instanceof Error && error.message.trim()) {
    return error.message;
  }

  return fallback;
};

const clampHours = (value, limit) => Math.max(0, Math.min(limit, value));

const projectRuntimeTelemetry = (profile, now) => {
  if (!profile.running) {
    return { ...profile, runtimeUpdatedAt: now };
  }

  const previous = toNumber(profile.runtimeUpdatedAt, now);
  const elapsedHours = Math.max(0, now - previous) / 3600000;
  if (elapsedHours === 0) {
    return { ...profile, runtimeUpdatedAt: now };
  }

  return {
    ...profile,
    usageWeekHours: clampHours(profile.usageWeekHours + elapsedHours, WEEK_LIMIT_HOURS),
    usage5hHours: clampHours(profile.usage5hHours + elapsedHours, FIVE_HOUR_LIMIT_HOURS),
    runtimeUpdatedAt: now,
  };
};

const mergeRuntimeStatuses = (profiles, runtimeStatuses) => {
  const now = Date.now();
  const runtimeById = new Map(runtimeStatuses.map((status) => [status.id, status]));
  return sortProfiles(
    profiles.map((profile) => {
      const projected = projectRuntimeTelemetry(profile, now);
      const runtime = runtimeById.get(profile.id);
      if (!runtime) {
        return projected;
      }

      return {
        ...projected,
        status: runtime.status,
        statusMessage: runtime.statusMessage,
        running: runtime.running,
        pid: runtime.pid,
        isFocused: runtime.isFocused,
        runtimeUpdatedAt: now,
      };
    }),
  );
};

export const useProfileStore = create((set, get) => ({
  profiles: [],
  selectedProfileId: null,
  loading: false,
  mutating: false,
  discoveringCodex: false,
  error: null,
  codexDiscovery: null,
  lastSyncedAt: null,
  pollHandle: null,
  pollingMetadata: DEFAULT_POLLING_METADATA,

  setSelectedProfileId: (profileId) => {
    set({ selectedProfileId: profileId });
  },

  clearError: () => {
    set({ error: null });
  },

  fetchProfiles: async ({ silent = false } = {}) => {
    if (!silent) {
      set({ loading: true, error: null });
    }

    try {
      const response = await invoke("list_profiles");
      const profiles = sortProfiles((Array.isArray(response) ? response : []).map(normalizeProfile));
      const selectedProfileId = selectedIdFrom(profiles, get().selectedProfileId);

      set({
        profiles,
        selectedProfileId,
        loading: false,
        error: null,
        lastSyncedAt: Date.now(),
      });

      return profiles;
    } catch (error) {
      const message = getErrorMessage(error, "Failed to load managed profiles.");
      set({ loading: false, error: message });
      throw new Error(message);
    }
  },

  fetchRuntimeStatuses: async () => {
    try {
      const response = await invoke("get_runtime_statuses");
      const runtimeStatuses = (Array.isArray(response) ? response : []).map(normalizeRuntimeStatus);
      const profiles = mergeRuntimeStatuses(get().profiles, runtimeStatuses);
      const selectedProfileId = selectedIdFrom(profiles, get().selectedProfileId);

      set({
        profiles,
        selectedProfileId,
        error: null,
        lastSyncedAt: Date.now(),
      });

      return runtimeStatuses;
    } catch (error) {
      const message = getErrorMessage(error, "Failed to refresh runtime statuses.");
      set({ error: message });
      throw new Error(message);
    }
  },

  fetchRuntimePollingMetadata: async () => {
    try {
      const metadata = normalizePollingMetadata(await invoke("get_runtime_polling_metadata"));
      set({ pollingMetadata: metadata });
      return metadata;
    } catch {
      set({ pollingMetadata: DEFAULT_POLLING_METADATA });
      return DEFAULT_POLLING_METADATA;
    }
  },

  discoverCodexPath: async () => {
    set({ discoveringCodex: true });

    try {
      const discovery = normalizeDiscovery(await invoke("discover_codex_path"));
      set({ codexDiscovery: discovery, discoveringCodex: false });
      return discovery;
    } catch (error) {
      const message = getErrorMessage(error, "Failed to auto-detect Codex.exe.");
      const discovery = {
        path: "",
        source: "Not Found",
        autoDetected: false,
        message,
      };
      set({ codexDiscovery: discovery, discoveringCodex: false });
      return discovery;
    }
  },

  browseCodexExecutable: async () => {
    set({ discoveringCodex: true, error: null });

    try {
      const selectedPath = await invoke("browse_codex_executable");
      set({ discoveringCodex: false });
      return selectedPath ? String(selectedPath) : "";
    } catch (error) {
      const message = getErrorMessage(error, "Failed to open Codex.exe picker.");
      set({ discoveringCodex: false, error: message });
      throw new Error(message);
    }
  },

  createProfile: async ({
    name,
    email,
    colorTag = DEFAULT_COLOR_TAG,
    customCodexPath = "",
    proxy = null,
  }) => {
    set({ mutating: true, error: null });

    try {
      const trimmedCustomPath = customCodexPath.trim();
      if (trimmedCustomPath) {
        await invoke("set_custom_codex_path", { codexPath: trimmedCustomPath });
      }

      const created = normalizeProfile(
        await invoke("create_profile", {
          name: name.trim(),
          email: email.trim(),
          colorTag,
          proxy,
        }),
      );

      const profiles = sortProfiles([...get().profiles.filter((profile) => profile.id !== created.id), created]);
      set({
        profiles,
        selectedProfileId: created.id,
        mutating: false,
        error: null,
        lastSyncedAt: Date.now(),
      });

      await get().fetchRuntimeStatuses().catch(() => {});
      return created;
    } catch (error) {
      const message = getErrorMessage(error, "Failed to create managed profile.");
      set({ mutating: false, error: message });
      throw new Error(message);
    }
  },

  deleteProfile: async (profileId) => {
    set({ mutating: true, error: null });

    try {
      await invoke("delete_profile", { profileId });
      const profiles = sortProfiles(get().profiles.filter((profile) => profile.id !== profileId));
      set({
        profiles,
        selectedProfileId: selectedIdFrom(profiles, get().selectedProfileId),
        mutating: false,
        error: null,
        lastSyncedAt: Date.now(),
      });

      await get().fetchRuntimeStatuses().catch(() => {});
    } catch (error) {
      const message = getErrorMessage(error, "Failed to delete managed profile.");
      set({ mutating: false, error: message });
      throw new Error(message);
    }
  },

  renameProfile: async (profileId, newName) => {
    set({ mutating: true, error: null });

    try {
      const renamed = normalizeProfile(
        await invoke("rename_profile", {
          profileId,
          newName: newName.trim(),
        }),
      );

      const profiles = sortProfiles(
        get().profiles.map((profile) => (profile.id === renamed.id ? renamed : profile)),
      );
      set({
        profiles,
        selectedProfileId: renamed.id,
        mutating: false,
        error: null,
        lastSyncedAt: Date.now(),
      });

      await get().fetchRuntimeStatuses().catch(() => {});
      return renamed;
    } catch (error) {
      const message = getErrorMessage(error, "Failed to rename managed profile.");
      set({ mutating: false, error: message });
      throw new Error(message);
    }
  },

  cloneProfile: async (profileId, newName = null) => {
    set({ mutating: true, error: null });

    try {
      const cloned = normalizeProfile(
        await invoke("clone_profile", {
          profileId,
          newName,
        }),
      );

      const profiles = sortProfiles([...get().profiles.filter((profile) => profile.id !== cloned.id), cloned]);
      set({
        profiles,
        selectedProfileId: cloned.id,
        mutating: false,
        error: null,
        lastSyncedAt: Date.now(),
      });

      await get().fetchRuntimeStatuses().catch(() => {});
      return cloned;
    } catch (error) {
      const message = getErrorMessage(error, "Failed to clone managed profile.");
      set({ mutating: false, error: message });
      throw new Error(message);
    }
  },

  launchProfile: async (profileId) => {
    set({ mutating: true, error: null });

    try {
      const launched = normalizeProfile(await invoke("launch_profile", { profileId }));
      const profiles = sortProfiles(
        get().profiles.map((profile) => (profile.id === launched.id ? launched : profile)),
      );

      set({
        profiles,
        selectedProfileId: launched.id,
        mutating: false,
        error: null,
        lastSyncedAt: Date.now(),
      });

      await get().fetchRuntimeStatuses().catch(() => {});
      return launched;
    } catch (error) {
      const message = getErrorMessage(error, "Failed to launch managed profile.");
      set({ mutating: false, error: message });
      throw new Error(message);
    }
  },

  focusProfile: async (profileId) => {
    set({ mutating: true, error: null });

    try {
      const focused = Boolean(await invoke("focus_profile", { profileId }));
      const profiles = get().profiles.map((profile) =>
        profile.id === profileId ? { ...profile, isFocused: focused } : profile,
      );
      set({ profiles, mutating: false, error: null, lastSyncedAt: Date.now() });
      await get().fetchRuntimeStatuses().catch(() => {});
      return focused;
    } catch (error) {
      const message = getErrorMessage(error, "Failed to focus managed profile window.");
      set({ mutating: false, error: message });
      throw new Error(message);
    }
  },

  stopProfile: async (profileId) => {
    set({ mutating: true, error: null });

    try {
      const response = await invoke("stop_profile", { profileId });
      await get().fetchRuntimeStatuses().catch(() => {});
      set({ mutating: false, error: null, lastSyncedAt: Date.now() });
      return Array.isArray(response) ? response : [];
    } catch (error) {
      const message = getErrorMessage(error, "Failed to stop managed Codex instance.");
      set({ mutating: false, error: message });
      throw new Error(message);
    }
  },

  launchAllProfiles: async () => {
    set({ mutating: true, error: null });

    try {
      const response = await invoke("launch_all_profiles");
      const profiles = sortProfiles((Array.isArray(response) ? response : []).map(normalizeProfile));
      set({
        profiles,
        selectedProfileId: selectedIdFrom(profiles, get().selectedProfileId),
        mutating: false,
        error: null,
        lastSyncedAt: Date.now(),
      });
      await get().fetchRuntimeStatuses().catch(() => {});
      return profiles;
    } catch (error) {
      const message = getErrorMessage(error, "Failed to launch all managed profiles.");
      set({ mutating: false, error: message });
      throw new Error(message);
    }
  },

  killAllActiveInstances: async () => {
    set({ mutating: true, error: null });

    try {
      const response = await invoke("kill_all_active_instances");
      await get().fetchRuntimeStatuses().catch(() => {});
      set({ mutating: false, error: null, lastSyncedAt: Date.now() });
      return Array.isArray(response) ? response : [];
    } catch (error) {
      const message = getErrorMessage(error, "Failed to kill active managed Codex instances.");
      set({ mutating: false, error: message });
      throw new Error(message);
    }
  },

  startPolling: () => {
    if (get().pollHandle) {
      return;
    }

    const pollingDelay = () => {
      const metadata = get().pollingMetadata ?? DEFAULT_POLLING_METADATA;
      if (typeof document !== "undefined" && document.hidden) {
        return metadata.minimizedMs;
      }

      return get().profiles.some((profile) => statusIsVolatile(profile.status))
        ? metadata.launchingMs
        : metadata.activeMs;
    };

    const tick = async () => {
      try {
        await get().fetchRuntimeStatuses();
      } catch {
        // Runtime polling errors are already stored in state.
      }

      if (!get().pollHandle) {
        return;
      }

      const nextHandle = window.setTimeout(tick, pollingDelay());
      set({ pollHandle: nextHandle });
    };

    const bootstrap = async () => {
      await get().fetchRuntimePollingMetadata().catch(() => {});
      await get().fetchProfiles({ silent: true }).catch(() => {});
      if (get().pollHandle) {
        await tick();
      }
    };

    const pollHandle = window.setTimeout(bootstrap, 0);
    set({ pollHandle });
  },

  stopPolling: () => {
    const { pollHandle } = get();
    if (pollHandle) {
      window.clearTimeout(pollHandle);
    }

    set({ pollHandle: null });
  },
}));

export const useProfilePolling = (enabled = true) => {
  const startPolling = useProfileStore((state) => state.startPolling);
  const stopPolling = useProfileStore((state) => state.stopPolling);

  useEffect(() => {
    if (!enabled) {
      stopPolling();
      return undefined;
    }

    startPolling();
    return () => {
      stopPolling();
    };
  }, [enabled, startPolling, stopPolling]);
};

export const selectSelectedProfile = (state) =>
  state.profiles.find((profile) => profile.id === state.selectedProfileId) ?? null;
