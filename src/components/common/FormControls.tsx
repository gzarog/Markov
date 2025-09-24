import React from "react";

type NumLProps = {
  label: string;
  value: number;
  set: (value: number) => void;
  step?: number;
};

type SelectLProps = {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: string[];
};

export function Label({ children }: { children: React.ReactNode }) {
  return <div className="text-[10px] sm:text-xs text-gray-600 mb-0.5">{children}</div>;
}

export function NumL({ label, value, set, step }: NumLProps) {
  return (
    <div>
      <Label>{label}</Label>
      <input
        type="number"
        step={step || 1}
        className="border rounded px-2 py-1 w-full"
        value={value}
        onChange={(e) => set(Number(e.target.value))}
      />
    </div>
  );
}

export function SelectL({ label, value, onChange, options }: SelectLProps) {
  return (
    <div>
      <Label>{label}</Label>
      <select
        className="border rounded px-2 py-1 w-full"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </div>
  );
}
