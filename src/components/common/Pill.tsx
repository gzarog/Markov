import React from "react";

export function Pill({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center rounded-full border px-2 py-0.5 text-xs text-gray-600 bg-gray-50">
      {children}
    </span>
  );
}
