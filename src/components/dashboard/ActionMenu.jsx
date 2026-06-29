import { useEffect, useRef, useState } from "react";
import { Button } from "./primitives";

export default function ActionMenu({
  profile,
  onClone,
  onRename,
  onDelete,
  onStop,
  disabled,
  alwaysVisible = false,
}) {
  const [open, setOpen] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    if (!open) {
      return undefined;
    }

    const close = (event) => {
      if (!menuRef.current?.contains(event.target)) {
        setOpen(false);
      }
    };

    window.addEventListener("pointerdown", close);
    return () => window.removeEventListener("pointerdown", close);
  }, [open]);

  const run = (handler) => (event) => {
    event.stopPropagation();
    setOpen(false);
    handler(profile);
  };

  return (
    <div ref={menuRef} className="relative" onClick={(event) => event.stopPropagation()}>
      <Button
        tone="ghost"
        size="sm"
        disabled={disabled}
        aria-label={`Open actions for ${profile.name}`}
        onClick={(event) => {
          event.stopPropagation();
          setOpen((current) => !current);
        }}
        className={`h-8 w-8 px-0 ${alwaysVisible ? "opacity-100" : "opacity-0 group-hover:opacity-100 group-focus-within:opacity-100"}`}
      >
        ...
      </Button>

      {open ? (
        <div className="absolute right-0 top-9 z-30 w-40 overflow-hidden rounded-2xl bg-slate-900 p-1 shadow-2xl shadow-black/50">
          {profile.running && onStop ? (
            <button
              type="button"
              onClick={run(onStop)}
              className="block w-full rounded-xl px-3 py-2 text-left text-sm text-amber-200 hover:bg-amber-500/15"
            >
              Stop Instance
            </button>
          ) : null}
          <button
            type="button"
            onClick={run(onClone)}
            className="block w-full rounded-xl px-3 py-2 text-left text-sm text-slate-300 hover:bg-slate-800 hover:text-white"
          >
            Clone
          </button>
          <button
            type="button"
            onClick={run(onRename)}
            className="block w-full rounded-xl px-3 py-2 text-left text-sm text-slate-300 hover:bg-slate-800 hover:text-white"
          >
            Rename
          </button>
          <button
            type="button"
            onClick={run(onDelete)}
            className="block w-full rounded-xl px-3 py-2 text-left text-sm text-rose-200 hover:bg-rose-500/15"
          >
            Delete
          </button>
        </div>
      ) : null}
    </div>
  );
}
