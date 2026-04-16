"use client";

import { useEffect, useId, useState } from "react";

/**
 * Minimal, styled HTML form controls.  Each input renders the
 * label + caption + invalid state in a consistent shape so the
 * `<SchemaForm />` can lay them out uniformly.  Controls are native
 * inputs (not HeroUI's React Aria compound components) for simplicity
 * and to keep the form tiny.
 */

export type BaseFieldProps = {
  label: string;
  caption?: string;
  error?: string;
  children: React.ReactNode;
};

export function Field({ label, caption, error, children }: BaseFieldProps) {
  const id = useId();
  return (
    <label htmlFor={id} className="flex flex-col gap-1.5">
      <span className="text-sm text-foreground/90">{label}</span>
      <div>
        <Slot id={id}>{children}</Slot>
      </div>
      {caption && (
        <span className="text-xs text-muted leading-snug">{caption}</span>
      )}
      {error && <span className="text-xs text-red-400">{error}</span>}
    </label>
  );
}

// Assigns the generated id to the first child form control so the
// label's htmlFor points to it.
function Slot({ id, children }: { id: string; children: React.ReactNode }) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const child = children as any;
  if (child && typeof child === "object" && "props" in child) {
    const merged = { ...child.props, id: child.props.id ?? id };
    return { ...child, props: merged };
  }
  return <>{children}</>;
}

const controlClass =
  "h-9 px-3 rounded-[calc(var(--radius)*0.6)] bg-surface/60 border border-separator text-sm " +
  "text-foreground/90 focus:outline-none focus:border-accent/60 focus:bg-surface transition-colors";

export function TextInput(
  props: React.InputHTMLAttributes<HTMLInputElement>,
) {
  return <input type="text" {...props} className={`${controlClass} w-full ${props.className ?? ""}`} />;
}

export function NumberInput({
  value,
  onValueChange,
  step,
  min,
  max,
  ...rest
}: {
  value: number;
  onValueChange: (n: number) => void;
  step?: number;
  min?: number;
  max?: number;
} & Omit<React.InputHTMLAttributes<HTMLInputElement>, "value" | "onChange" | "type">) {
  return (
    <input
      type="number"
      value={Number.isFinite(value) ? value : ""}
      step={step}
      min={min}
      max={max}
      onChange={(e) => {
        const v = e.target.value;
        if (v === "") return; // allow temporarily-empty input; schema default clamps on blur
        const n = Number(v);
        if (Number.isFinite(n)) onValueChange(n);
      }}
      {...rest}
      className={`${controlClass} w-full tabular-nums font-mono ${rest.className ?? ""}`}
    />
  );
}

export function SelectInput<T extends string>({
  value,
  onValueChange,
  options,
  ...rest
}: {
  value: T;
  onValueChange: (v: T) => void;
  options: readonly { value: string; label: string }[];
} & Omit<React.SelectHTMLAttributes<HTMLSelectElement>, "value" | "onChange">) {
  return (
    <select
      value={value}
      onChange={(e) => onValueChange(e.target.value as T)}
      {...rest}
      className={`${controlClass} w-full ${rest.className ?? ""}`}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

export function BoolSwitch({
  value,
  onValueChange,
}: {
  value: boolean;
  onValueChange: (v: boolean) => void;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={value}
      onClick={() => onValueChange(!value)}
      className={[
        "relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors",
        "focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/40",
        value ? "bg-accent" : "bg-default",
      ].join(" ")}
    >
      <span
        className={[
          "inline-block h-5 w-5 rounded-full bg-surface shadow transition-transform",
          value ? "translate-x-[1.375rem]" : "translate-x-0.5",
        ].join(" ")}
      />
    </button>
  );
}

export function IntListInput({
  value,
  onValueChange,
}: {
  value: number[];
  onValueChange: (v: number[]) => void;
}) {
  // Keep a local text buffer so separators in-flight (trailing comma,
  // spaces) survive while the user types.  We only sync *from* the
  // controlled `value` when it meaningfully diverges from what we'd
  // parse out of the buffer — otherwise typing "3, " would round-trip
  // back to "3, 3" and swallow the user's next keystroke.
  const [buffer, setBuffer] = useState(() => value.join(", "));

  useEffect(() => {
    const parsedFromBuffer = parseIntList(buffer);
    if (!sameNums(parsedFromBuffer, value)) {
      setBuffer(value.join(", "));
    }
  }, [value, buffer]);

  return (
    <input
      type="text"
      value={buffer}
      onChange={(e) => {
        const next = e.target.value;
        setBuffer(next);
        onValueChange(parseIntList(next));
      }}
      onBlur={() => setBuffer(value.join(", "))}
      placeholder="e.g. 3, 3, 7, 7, 15, 15, 0, 0"
      className={`${controlClass} w-full tabular-nums font-mono`}
    />
  );
}

function parseIntList(s: string): number[] {
  return s
    .split(/[,\s]+/)
    .filter(Boolean)
    .map((t) => Number(t))
    .filter((n) => Number.isFinite(n) && Number.isInteger(n));
}

function sameNums(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}
