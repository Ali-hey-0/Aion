import { classNames } from "./utils";

export const Button = ({
  tone = "secondary",
  size = "md",
  type = "button",
  className,
  children,
  ...props
}) => (
  <button
    type={type}
    {...props}
    className={classNames(
      "inline-flex items-center justify-center rounded-xl font-medium transition duration-200 disabled:cursor-not-allowed disabled:opacity-45",
      size === "sm" && "h-8 px-3 text-xs",
      size === "md" && "h-10 px-4 text-sm",
      tone === "primary" && "bg-slate-50 text-slate-950 hover:bg-white",
      tone === "secondary" && "bg-slate-800/60 text-slate-200 hover:bg-slate-800",
      tone === "ghost" && "bg-transparent text-slate-400 hover:bg-slate-800/55 hover:text-slate-100",
      tone === "success" && "bg-emerald-500/15 text-emerald-200 hover:bg-emerald-500/25",
      tone === "danger" && "bg-rose-500/15 text-rose-200 hover:bg-rose-500/25",
      className,
    )}
  >
    {children}
  </button>
);

export const InlineAlert = ({ tone = "danger", children }) => (
  <div
    className={classNames(
      "rounded-2xl px-4 py-3 text-sm leading-6",
      tone === "danger" && "bg-rose-500/10 text-rose-100",
      tone === "success" && "bg-emerald-500/10 text-emerald-100",
      tone === "info" && "bg-sky-500/10 text-sky-100",
      tone === "warning" && "bg-amber-500/10 text-amber-100",
    )}
  >
    {children}
  </div>
);

export const FieldError = ({ message }) =>
  message ? <p className="mt-2 text-xs font-medium text-rose-300">{message}</p> : null;

export const TextInput = ({ label, error, className, rightSlot, ...props }) => (
  <label className={classNames("block", className)}>
    <span className="text-sm font-medium text-slate-300">{label}</span>
    <div className="mt-2 flex gap-2">
      <input
        {...props}
        className={classNames(
          "min-w-0 flex-1 rounded-xl bg-slate-900/80 px-4 py-3 text-sm text-white outline-none transition placeholder:text-slate-600 focus:bg-slate-900 focus:ring-2 focus:ring-indigo-500/50",
          error && "bg-rose-950/40",
        )}
      />
      {rightSlot}
    </div>
    <FieldError message={error} />
  </label>
);

export const ModalShell = ({ title, description, children, onClose }) => (
  <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 px-4 backdrop-blur-xl animate-fade-in">
    <div className="max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-[28px] bg-[#080d18] p-8 shadow-2xl shadow-black/70">
      <div className="flex items-start justify-between gap-6">
        <div className="min-w-0">
          <h2 className="text-xl font-semibold tracking-tight text-white">{title}</h2>
          {description ? <p className="mt-2 max-w-xl text-sm leading-6 text-slate-400">{description}</p> : null}
        </div>
        <Button tone="ghost" size="sm" onClick={onClose}>
          Close
        </Button>
      </div>
      {children}
    </div>
  </div>
);

export const StatusPill = ({ running, status }) => {
  const label = status || (running ? "Running" : "Idle");
  const tone =
    label === "Running"
      ? "emerald"
      : label === "Launching"
        ? "indigo"
        : label === "Stopping"
          ? "amber"
          : label === "Error"
            ? "rose"
            : "slate";

  return (
    <span
      className={classNames(
        "inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium",
        tone === "emerald" && "bg-emerald-500/12 text-emerald-200",
        tone === "indigo" && "bg-indigo-500/12 text-indigo-200",
        tone === "amber" && "bg-amber-500/12 text-amber-200",
        tone === "rose" && "bg-rose-500/12 text-rose-200",
        tone === "slate" && "bg-slate-800/65 text-slate-400",
      )}
    >
      <span
        className={classNames(
          "h-1.5 w-1.5 rounded-full",
          tone === "emerald" && "bg-emerald-300",
          tone === "indigo" && "bg-indigo-300",
          tone === "amber" && "bg-amber-300",
          tone === "rose" && "bg-rose-300",
          tone === "slate" && "bg-slate-500",
        )}
      />
      {label}
    </span>
  );
};

export const MetaField = ({ label, value, mono = false }) => (
  <div className="min-w-0">
    <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-600">{label}</p>
    <p className={classNames("mt-1 truncate text-sm text-slate-300", mono && "font-mono text-xs text-slate-400")}>
      {value}
    </p>
  </div>
);

export const EmptyState = ({ onAddProfile }) => (
  <div className="flex min-h-[520px] items-center justify-center rounded-[32px] bg-slate-900/20 p-8 text-center animate-fade-in">
    <div>
      <p className="text-2xl font-semibold tracking-tight text-white">No profile selected</p>
      <p className="mt-3 max-w-md text-sm leading-6 text-slate-400">
        Create a managed profile to launch Codex with isolated account storage, proxy routing, and live runtime tracking.
      </p>
      <Button tone="primary" className="mt-7" onClick={onAddProfile}>
        Add Managed Profile
      </Button>
    </div>
  </div>
);
