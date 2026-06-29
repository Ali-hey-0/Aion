import { formatHours } from "./utils";

export default function StatWidget({ label, value, limit, accent }) {
  const numeric = Number(value);
  const safeValue = Number.isFinite(numeric) && numeric >= 0 ? numeric : 0;
  const ratio = Math.max(0, Math.min(100, (safeValue / limit) * 100));

  return (
    <div className="rounded-[28px] bg-slate-900/30 p-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm text-slate-500">{label}</p>
          <div className="mt-4 flex items-end gap-2">
            <span className="text-4xl font-semibold tracking-tight text-white">{formatHours(value)}</span>
            <span className="pb-1 text-sm text-slate-500">hours</span>
          </div>
        </div>
        <div
          className="h-10 w-10 rounded-2xl opacity-80"
          style={{ background: `linear-gradient(135deg, ${accent}, rgba(15,23,42,0.25))` }}
        />
      </div>

      <div className="mt-7 h-px overflow-hidden rounded-full bg-slate-800/70">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${ratio}%`, backgroundColor: accent }}
        />
      </div>
      <p className="mt-3 text-xs text-slate-500">{Math.round(ratio)}% of tracked window</p>
    </div>
  );
}
