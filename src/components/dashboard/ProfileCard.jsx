import ActionMenu from "./ActionMenu";
import { StatusPill } from "./primitives";
import { classNames, formatHours, getInitials } from "./utils";

export default function ProfileCard({
  profile,
  selected,
  disabled,
  onSelect,
  onClone,
  onRename,
  onDelete,
  onStop,
}) {
  const handleSelect = () => {
    if (!disabled) {
      onSelect(profile.id);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      handleSelect();
    }
  };

  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      onClick={handleSelect}
      onKeyDown={handleKeyDown}
      className={classNames(
        "group w-full rounded-3xl p-4 text-left transition duration-200 outline-none",
        disabled ? "cursor-not-allowed opacity-70" : "cursor-pointer focus-visible:bg-slate-800/55",
        selected ? "bg-slate-800/55" : "hover:bg-slate-900/65",
      )}
    >
      <div className="flex items-start gap-3">
        <div
          className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl text-sm font-semibold text-white shadow-lg shadow-black/20"
          style={{
            background: `linear-gradient(135deg, ${profile.colorTag}, rgba(15,23,42,0.25))`,
          }}
        >
          {getInitials(profile)}
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 items-center justify-between gap-2">
            <p className="truncate text-[15px] font-semibold text-white">{profile.name}</p>
            <ActionMenu
              profile={profile}
              disabled={disabled}
              onClone={onClone}
              onRename={onRename}
              onDelete={onDelete}
              onStop={onStop}
            />
          </div>
          <p className="mt-1 truncate text-sm text-slate-500">{profile.email || "No email"}</p>
          {profile.statusMessage ? (
            <p className="mt-1 truncate text-xs text-rose-300/80">{profile.statusMessage}</p>
          ) : null}
        </div>
      </div>

      <div className="mt-4 flex items-center justify-between gap-3">
        <StatusPill running={profile.running} status={profile.status} />
        <span className="text-xs text-slate-500">{formatHours(profile.usage5hHours)}h / 5h</span>
      </div>
    </div>
  );
}
