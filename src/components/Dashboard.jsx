import { useMemo, useState } from "react";
import {
  selectSelectedProfile,
  useProfilePolling,
  useProfileStore,
} from "../store/useProfileStore";
import ProfileCard from "./dashboard/ProfileCard";
import StatWidget from "./dashboard/StatWidget";
import ActionMenu from "./dashboard/ActionMenu";
import { AddProfileModal, DeleteModal, KillAllModal, RenameModal } from "./dashboard/ProfileModals";
import { Button, EmptyState, InlineAlert, MetaField, StatusPill } from "./dashboard/primitives";
import { formatUnixDate, getExpiryState } from "./dashboard/utils";

const modalProfileFrom = (profiles, modalProfileId, selectedProfile) =>
  profiles.find((profile) => profile.id === modalProfileId) ?? selectedProfile;

export default function Dashboard() {
  useProfilePolling(true);

  const profiles = useProfileStore((state) => state.profiles);
  const selectedProfile = useProfileStore(selectSelectedProfile);
  const selectedProfileId = useProfileStore((state) => state.selectedProfileId);
  const setSelectedProfileId = useProfileStore((state) => state.setSelectedProfileId);
  const loading = useProfileStore((state) => state.loading);
  const mutating = useProfileStore((state) => state.mutating);
  const error = useProfileStore((state) => state.error);
  const clearError = useProfileStore((state) => state.clearError);
  const lastSyncedAt = useProfileStore((state) => state.lastSyncedAt);
  const launchProfile = useProfileStore((state) => state.launchProfile);
  const focusProfile = useProfileStore((state) => state.focusProfile);
  const stopProfile = useProfileStore((state) => state.stopProfile);
  const cloneProfile = useProfileStore((state) => state.cloneProfile);
  const launchAllProfiles = useProfileStore((state) => state.launchAllProfiles);

  const [activeModal, setActiveModal] = useState(null);
  const [modalProfileId, setModalProfileId] = useState(null);
  const [notice, setNotice] = useState(null);

  const runningCount = profiles.filter((profile) => profile.running).length;
  const modalProfile = modalProfileFrom(profiles, modalProfileId, selectedProfile);
  const expiry = useMemo(() => getExpiryState(selectedProfile?.expiresAt), [selectedProfile?.expiresAt]);

  const openProfileModal = (modal, profile = selectedProfile) => {
    if (!profile) {
      return;
    }

    setSelectedProfileId(profile.id);
    setModalProfileId(profile.id);
    setActiveModal(modal);
  };

  const runProfileLaunch = async (profile = selectedProfile) => {
    if (!profile) {
      return;
    }

    setNotice(null);
    try {
      if (profile.running) {
        const focused = await focusProfile(profile.id);
        setNotice(focused ? "Focused the running Codex window." : "The process is running, but no visible window was found yet.");
      } else {
        await launchProfile(profile.id);
      }
    } catch (errorMessage) {
      setNotice(errorMessage.message);
    }
  };

  const runClone = async (profile) => {
    if (!profile) {
      return;
    }

    setNotice(null);
    try {
      await cloneProfile(profile.id);
    } catch (errorMessage) {
      setNotice(errorMessage.message);
    }
  };

  const runStop = async (profile) => {
    if (!profile) {
      return;
    }

    setNotice(null);
    try {
      const killed = await stopProfile(profile.id);
      setNotice(
        killed.length > 0
          ? "Stopped the selected Codex instance."
          : "No running process was found for this profile.",
      );
    } catch (errorMessage) {
      setNotice(errorMessage.message);
    }
  };

  const runLaunchAll = async () => {
    setNotice(null);
    try {
      await launchAllProfiles();
    } catch (errorMessage) {
      setNotice(errorMessage.message);
    }
  };

  return (
    <div className="h-screen overflow-hidden bg-[#050914] text-slate-100">
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(circle_at_top_right,rgba(79,70,229,0.10),transparent_30%),radial-gradient(circle_at_bottom_left,rgba(16,185,129,0.06),transparent_28%)]" />

      <div className="relative flex h-screen flex-col lg:flex-row">
        <aside className="flex max-h-[44vh] w-full shrink-0 bg-[#070c17]/90 p-5 lg:h-screen lg:max-h-none lg:w-80">
          <div className="flex min-h-0 w-full flex-col">
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-slate-800 text-sm font-semibold text-white">
                  AI
                </div>
                <div>
                  <h1 className="text-base font-semibold tracking-tight text-white">Aion</h1>
                  <p className="text-xs text-slate-500">Codex Profile Manager</p>
                </div>
              </div>
              <Button tone="primary" size="sm" onClick={() => setActiveModal("add")} className="lg:hidden">
                Add
              </Button>
            </div>

            <div className="mt-8 grid grid-cols-2 gap-3">
              <div className="rounded-3xl bg-slate-900/45 p-4">
                <p className="text-xs text-slate-500">Profiles</p>
                <p className="mt-2 text-2xl font-semibold text-white">{profiles.length}</p>
              </div>
              <div className="rounded-3xl bg-slate-900/45 p-4">
                <p className="text-xs text-slate-500">Running</p>
                <p className="mt-2 text-2xl font-semibold text-emerald-300">{runningCount}</p>
              </div>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-3">
              <Button onClick={runLaunchAll} disabled={mutating || profiles.length === 0}>
                Launch All
              </Button>
              <Button tone="danger" onClick={() => setActiveModal("killAll")} disabled={mutating || runningCount === 0}>
                Kill All
              </Button>
            </div>

            <div className="mt-6 min-h-0 flex-1 space-y-2 overflow-y-auto pr-1">
              {loading && profiles.length === 0 ? (
                <div className="space-y-3">
                  {[0, 1, 2].map((item) => (
                    <div key={item} className="h-28 animate-pulse rounded-3xl bg-slate-900/45" />
                  ))}
                </div>
              ) : null}

              {profiles.map((profile) => (
                <ProfileCard
                  key={profile.id}
                  profile={profile}
                  selected={selectedProfileId === profile.id}
                  disabled={mutating}
                  onSelect={setSelectedProfileId}
                  onClone={runClone}
                  onRename={(target) => openProfileModal("rename", target)}
                  onDelete={(target) => openProfileModal("delete", target)}
                  onStop={runStop}
                />
              ))}

              {!loading && profiles.length === 0 ? (
                <div className="rounded-3xl bg-slate-900/60 p-5 text-sm leading-6 text-slate-500">
                  No profiles yet. Add one to create an isolated Codex account space.
                </div>
              ) : null}
            </div>
          </div>
        </aside>

        <main className="flex min-h-0 min-w-0 flex-1 flex-col">
          <header className="shrink-0 px-6 py-5">
            <div className="mx-auto flex max-w-6xl items-center justify-between gap-4">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-600">Aion Control Plane</p>
                <h2 className="mt-1 text-xl font-semibold tracking-tight text-white">Managed Codex Instances</h2>
              </div>
              <Button tone="primary" onClick={() => setActiveModal("add")} className="hidden sm:inline-flex">
                Add Managed Profile
              </Button>
            </div>
          </header>

          <section className="min-h-0 flex-1 overflow-y-auto px-6 pb-6">
            <div className="mx-auto w-full max-w-6xl">
              {error ? (
                <div className="mb-5 flex items-start justify-between gap-4 rounded-3xl bg-rose-500/10 px-5 py-4 text-sm text-rose-100">
                  <span>{error}</span>
                  <button type="button" onClick={clearError} className="text-rose-200 hover:text-white">
                    Dismiss
                  </button>
                </div>
              ) : null}

              {notice ? (
                <div className="mb-5 animate-fade-in">
                  <InlineAlert tone="info">{notice}</InlineAlert>
                </div>
              ) : null}

              {!selectedProfile ? (
                <EmptyState onAddProfile={() => setActiveModal("add")} />
              ) : (
                <div className="space-y-5 animate-fade-in">
                  <section className="rounded-[32px] bg-slate-900/30 p-8">
                    <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
                      <div className="min-w-0">
                        <StatusPill running={selectedProfile.running} status={selectedProfile.status} />
                        <h3 className="mt-5 truncate text-4xl font-semibold tracking-tight text-white">
                          {selectedProfile.name}
                        </h3>
                        <p className="mt-2 truncate text-base text-slate-400">
                          {selectedProfile.email || "No account email"}
                        </p>
                        {selectedProfile.statusMessage ? (
                          <p className="mt-3 max-w-xl text-sm leading-6 text-rose-200/80">
                            {selectedProfile.statusMessage}
                          </p>
                        ) : null}
                      </div>

                      <div className="flex flex-wrap items-center gap-2">
                        <Button
                          tone={selectedProfile.running ? "success" : "primary"}
                          onClick={() => runProfileLaunch(selectedProfile)}
                          disabled={mutating}
                        >
                          {selectedProfile.running ? "Focus Window" : "Launch Profile"}
                        </Button>
                        <div className="group">
                          <ActionMenu
                            profile={selectedProfile}
                            disabled={mutating}
                            alwaysVisible
                            onClone={runClone}
                            onRename={(target) => openProfileModal("rename", target)}
                            onDelete={(target) => openProfileModal("delete", target)}
                            onStop={runStop}
                          />
                        </div>
                      </div>
                    </div>

                    <div className="mt-8 grid gap-4 md:grid-cols-4">
                      <MetaField label="Activated" value={selectedProfile.activatedAt || "Not configured"} />
                      <MetaField label="Expires" value={expiry.label} />
                      <MetaField label="Created" value={formatUnixDate(selectedProfile.createdAt)} />
                      <MetaField label="Last Launch" value={formatUnixDate(selectedProfile.lastLaunched)} />
                    </div>
                  </section>

                  <div className="grid gap-5 lg:grid-cols-2">
                    <StatWidget
                      label="Weekly Runtime"
                      value={selectedProfile.usageWeekHours}
                      limit={168}
                      accent="#8B5CF6"
                    />
                    <StatWidget
                      label="5-Hour Session Window"
                      value={selectedProfile.usage5hHours}
                      limit={5}
                      accent="#10B981"
                    />
                  </div>

                  <section className="grid gap-5 lg:grid-cols-2">
                    <div className="rounded-[28px] bg-slate-900/25 p-6">
                      <h4 className="text-sm font-semibold text-slate-200">Runtime details</h4>
                      <p className="mt-4 break-all font-mono text-xs leading-6 text-slate-500">
                        {selectedProfile.userDataDir}
                      </p>
                    </div>
                    <div className="rounded-[28px] bg-slate-900/25 p-6">
                      <h4 className="text-sm font-semibold text-slate-200">Network binding</h4>
                      <p className="mt-4 break-all font-mono text-xs leading-6 text-slate-500">
                        {selectedProfile.proxyEnabled
                          ? `${selectedProfile.proxyHostPort}${selectedProfile.proxyHasCredentials ? " with credentials" : ""}`
                          : "Proxy disabled"}
                      </p>
                    </div>
                  </section>
                </div>
              )}
            </div>
          </section>

          <footer className="shrink-0 px-6 py-4 text-center text-xs text-slate-700">
            {lastSyncedAt ? `Synced ${new Date(lastSyncedAt).toLocaleTimeString()}` : "Waiting for profile sync"}
          </footer>
        </main>
      </div>

      {activeModal === "add" ? <AddProfileModal onClose={() => setActiveModal(null)} /> : null}
      {activeModal === "rename" && modalProfile ? (
        <RenameModal profile={modalProfile} onClose={() => setActiveModal(null)} />
      ) : null}
      {activeModal === "delete" && modalProfile ? (
        <DeleteModal profile={modalProfile} onClose={() => setActiveModal(null)} />
      ) : null}
      {activeModal === "killAll" ? (
        <KillAllModal runningCount={runningCount} onClose={() => setActiveModal(null)} />
      ) : null}
    </div>
  );
}
