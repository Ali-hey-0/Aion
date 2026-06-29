import { useEffect, useMemo, useState } from "react";
import { useProfileStore } from "../../store/useProfileStore";
import { Button, InlineAlert, ModalShell, TextInput } from "./primitives";
import {
  COLOR_SWATCHES,
  DEFAULT_COLOR_TAG,
  classNames,
  profileNameError,
  validateEmail,
  validateProxyHostPort,
} from "./utils";

export function AddProfileModal({ onClose }) {
  const profiles = useProfileStore((state) => state.profiles);
  const createProfile = useProfileStore((state) => state.createProfile);
  const discoverCodexPath = useProfileStore((state) => state.discoverCodexPath);
  const browseCodexExecutable = useProfileStore((state) => state.browseCodexExecutable);
  const discoveringCodex = useProfileStore((state) => state.discoveringCodex);
  const codexDiscovery = useProfileStore((state) => state.codexDiscovery);
  const mutating = useProfileStore((state) => state.mutating);

  const [form, setForm] = useState({
    name: "",
    email: "",
    colorTag: DEFAULT_COLOR_TAG,
    customCodexPath: "",
    proxyHostPort: "",
    proxyUsername: "",
    proxyPassword: "",
  });
  const [formError, setFormError] = useState(null);
  const [touched, setTouched] = useState({});
  const [pathWasAutoFilled, setPathWasAutoFilled] = useState(false);

  useEffect(() => {
    let mounted = true;
    discoverCodexPath()
      .then((discovery) => {
        if (!mounted || !discovery.path) {
          return;
        }

        setForm((current) => {
          if (current.customCodexPath.trim()) {
            return current;
          }
          setPathWasAutoFilled(Boolean(discovery.autoDetected));
          return { ...current, customCodexPath: discovery.path };
        });
      })
      .catch(() => {});

    return () => {
      mounted = false;
    };
  }, [discoverCodexPath]);

  const updateField = (field, value) => {
    setTouched((current) => ({ ...current, [field]: true }));
    if (field === "customCodexPath") {
      setPathWasAutoFilled(false);
    }
    setForm((current) => ({ ...current, [field]: value }));
  };

  const errors = useMemo(() => {
    const proxyHasAnyValue =
      Boolean(form.proxyHostPort.trim()) ||
      Boolean(form.proxyUsername.trim()) ||
      Boolean(form.proxyPassword.trim());

    return {
      name: profileNameError(form.name, profiles),
      email: validateEmail(form.email),
      proxyHostPort: validateProxyHostPort(form.proxyHostPort, proxyHasAnyValue),
    };
  }, [form.email, form.name, form.proxyHostPort, form.proxyPassword, form.proxyUsername, profiles]);

  const visibleError = (field) => (touched[field] ? errors[field] : "");
  const canSubmit = !errors.name && !errors.email && !errors.proxyHostPort && !mutating;

  const browse = async () => {
    setFormError(null);
    try {
      const selectedPath = await browseCodexExecutable();
      if (selectedPath) {
        setForm((current) => ({ ...current, customCodexPath: selectedPath }));
        setPathWasAutoFilled(false);
      }
    } catch (error) {
      setFormError(error.message);
    }
  };

  const submit = async (event) => {
    event.preventDefault();
    setTouched({ name: true, email: true, proxyHostPort: true });
    setFormError(null);

    if (!canSubmit) {
      return;
    }

    const proxy = form.proxyHostPort.trim()
      ? {
          hostPort: form.proxyHostPort.trim(),
          username: form.proxyUsername.trim(),
          password: form.proxyPassword,
        }
      : null;

    try {
      await createProfile({
        name: form.name,
        email: form.email,
        colorTag: form.colorTag,
        customCodexPath: form.customCodexPath,
        proxy,
      });
      onClose();
    } catch (error) {
      setFormError(error.message);
    }
  };

  return (
    <ModalShell
      title="Add Managed Profile"
      description="Create one isolated Codex account space. Aion can auto-detect Codex.exe, or you can browse/paste it manually."
      onClose={onClose}
    >
      <form onSubmit={submit} className="mt-8 space-y-7">
        <div className="grid gap-5 md:grid-cols-2">
          <TextInput
            label="Profile name"
            value={form.name}
            onChange={(event) => updateField("name", event.target.value)}
            onBlur={() => setTouched((current) => ({ ...current, name: true }))}
            required
            maxLength={80}
            error={visibleError("name")}
            placeholder="Research Account"
          />
          <TextInput
            label="Account email"
            value={form.email}
            onChange={(event) => updateField("email", event.target.value)}
            onBlur={() => setTouched((current) => ({ ...current, email: true }))}
            required
            type="email"
            maxLength={254}
            error={visibleError("email")}
            placeholder="account@example.com"
          />
        </div>

        <div>
          <div className="mb-2 flex items-center justify-between gap-3">
            <span className="text-sm font-medium text-slate-300">Codex.exe path</span>
            {pathWasAutoFilled ? (
              <span className="rounded-full bg-emerald-500/12 px-3 py-1 text-xs font-medium text-emerald-200">
                Auto-detected
              </span>
            ) : null}
          </div>
          <div className="flex gap-2">
            <input
              value={form.customCodexPath}
              onChange={(event) => updateField("customCodexPath", event.target.value)}
              className="min-w-0 flex-1 rounded-xl bg-slate-900/80 px-4 py-3 font-mono text-xs text-slate-200 outline-none transition placeholder:text-slate-600 focus:bg-slate-900 focus:ring-2 focus:ring-indigo-500/50"
              placeholder="Auto-detected, browsed, or pasted Codex.exe path"
            />
            <Button onClick={browse} disabled={discoveringCodex}>
              {discoveringCodex ? "Searching" : "Browse"}
            </Button>
          </div>
          {codexDiscovery?.message ? (
            <p className="mt-2 text-xs leading-5 text-slate-500">
              {codexDiscovery.path
                ? `${codexDiscovery.source}: ${codexDiscovery.message}`
                : codexDiscovery.message}
            </p>
          ) : null}
        </div>

        <div className="rounded-[24px] bg-slate-900/35 p-5">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-sm font-semibold text-white">Network proxy</p>
              <p className="mt-1 text-xs leading-5 text-slate-500">
                Optional. Applied only when launching this profile.
              </p>
            </div>
            {form.proxyHostPort.trim() ? (
              <span className="rounded-full bg-sky-500/12 px-3 py-1 text-xs font-medium text-sky-200">
                Enabled
              </span>
            ) : null}
          </div>
          <div className="mt-5 grid gap-4 md:grid-cols-3">
            <TextInput
              label="Host:Port"
              value={form.proxyHostPort}
              onChange={(event) => updateField("proxyHostPort", event.target.value)}
              onBlur={() => setTouched((current) => ({ ...current, proxyHostPort: true }))}
              error={visibleError("proxyHostPort")}
              placeholder="127.0.0.1:7890"
            />
            <TextInput
              label="Username"
              value={form.proxyUsername}
              onChange={(event) => updateField("proxyUsername", event.target.value)}
              placeholder="Optional"
            />
            <TextInput
              label="Password"
              value={form.proxyPassword}
              onChange={(event) => updateField("proxyPassword", event.target.value)}
              type="password"
              placeholder="Optional"
            />
          </div>
        </div>

        <div>
          <span className="text-sm font-medium text-slate-300">Profile accent</span>
          <div className="mt-3 flex gap-2">
            {COLOR_SWATCHES.map((color) => (
              <button
                key={color}
                type="button"
                onClick={() => updateField("colorTag", color)}
                className={classNames(
                  "h-9 w-9 rounded-2xl transition duration-200 hover:scale-105",
                  form.colorTag === color && "shadow-[0_0_0_3px_rgba(255,255,255,0.22)]",
                )}
                style={{ backgroundColor: color }}
                aria-label={`Use ${color} accent`}
              />
            ))}
          </div>
        </div>

        {formError ? <InlineAlert>{formError}</InlineAlert> : null}

        <Button tone="primary" type="submit" className="w-full" disabled={!canSubmit}>
          {mutating ? "Creating Profile..." : "Create Profile"}
        </Button>
      </form>
    </ModalShell>
  );
}

export function RenameModal({ profile, onClose }) {
  const profiles = useProfileStore((state) => state.profiles);
  const renameProfile = useProfileStore((state) => state.renameProfile);
  const mutating = useProfileStore((state) => state.mutating);
  const [name, setName] = useState(profile?.name ?? "");
  const [formError, setFormError] = useState(null);
  const [touched, setTouched] = useState(false);
  const nameError = profileNameError(name, profiles, profile.id);

  const submit = async (event) => {
    event.preventDefault();
    setTouched(true);
    setFormError(null);
    if (nameError) {
      return;
    }

    try {
      await renameProfile(profile.id, name);
      onClose();
    } catch (error) {
      setFormError(error.message);
    }
  };

  return (
    <ModalShell title="Rename Profile" description={profile.email || "Update the display name."} onClose={onClose}>
      <form onSubmit={submit} className="mt-8 space-y-5">
        <TextInput
          label="New profile name"
          value={name}
          onChange={(event) => setName(event.target.value)}
          onBlur={() => setTouched(true)}
          required
          maxLength={80}
          error={touched ? nameError : ""}
        />
        {formError ? <InlineAlert>{formError}</InlineAlert> : null}
        <Button tone="primary" type="submit" className="w-full" disabled={mutating || Boolean(nameError)}>
          {mutating ? "Saving..." : "Save Name"}
        </Button>
      </form>
    </ModalShell>
  );
}

export function DeleteModal({ profile, onClose }) {
  const deleteProfile = useProfileStore((state) => state.deleteProfile);
  const mutating = useProfileStore((state) => state.mutating);
  const [formError, setFormError] = useState(null);

  const submit = async () => {
    setFormError(null);
    try {
      await deleteProfile(profile.id);
      onClose();
    } catch (error) {
      setFormError(error.message);
    }
  };

  return (
    <ModalShell
      title="Delete Instance"
      description="This removes the profile and its isolated Codex user data directory."
      onClose={onClose}
    >
      <div className="mt-8 rounded-[24px] bg-slate-900/60 p-5">
        <p className="text-base font-semibold text-white">{profile.name}</p>
        <p className="mt-1 truncate text-sm text-slate-500">{profile.email || "No email metadata"}</p>
      </div>
      {formError ? <div className="mt-5"><InlineAlert>{formError}</InlineAlert></div> : null}
      <div className="mt-7 flex justify-end gap-3">
        <Button tone="ghost" onClick={onClose}>Cancel</Button>
        <Button tone="danger" onClick={submit} disabled={mutating}>
          {mutating ? "Deleting..." : "Delete Instance"}
        </Button>
      </div>
    </ModalShell>
  );
}

export function KillAllModal({ runningCount, onClose }) {
  const killAllActiveInstances = useProfileStore((state) => state.killAllActiveInstances);
  const mutating = useProfileStore((state) => state.mutating);
  const [formError, setFormError] = useState(null);

  const submit = async () => {
    setFormError(null);
    try {
      await killAllActiveInstances();
      onClose();
    } catch (error) {
      setFormError(error.message);
    }
  };

  return (
    <ModalShell
      title="Kill All Active Instances"
      description="Aion terminates only Codex.exe processes whose user-data directory matches managed profiles."
      onClose={onClose}
    >
      <div className="mt-8">
        <InlineAlert tone="warning">
          {runningCount} managed Codex {runningCount === 1 ? "instance is" : "instances are"} currently running.
          Unsaved work inside those windows may be lost.
        </InlineAlert>
      </div>
      {formError ? <div className="mt-5"><InlineAlert>{formError}</InlineAlert></div> : null}
      <div className="mt-7 flex justify-end gap-3">
        <Button tone="ghost" onClick={onClose}>Cancel</Button>
        <Button tone="danger" onClick={submit} disabled={mutating}>
          {mutating ? "Terminating..." : "Kill Managed Instances"}
        </Button>
      </div>
    </ModalShell>
  );
}
