import { STATES, type StateKey } from "./types";

export type ConfidenceZone = {
  level: "Low" | "Medium" | "High";
  weight: number;
  maxProb: number;
  topState: StateKey;
};

/**
 * Map forecast distribution -> zone.
 * - High:   maxProb >= 0.60
 * - Medium: 0.45 <= maxProb < 0.60
 * - Low:    maxProb < 0.45
 * weight ∈ [0,1] used to scale sizing/targets (Low≈0, High≈1).
 */
export function zoneFromProbs(probs: number[]): ConfidenceZone {
  const maxProb = Math.max(...probs);
  const idx = probs.indexOf(maxProb);
  const topState = STATES[idx] ?? STATES[0];
  let level: ConfidenceZone["level"] = "Low";
  if (maxProb >= 0.6) level = "High";
  else if (maxProb >= 0.45) level = "Medium";
  const weight =
    level === "High"
      ? (maxProb - 0.6) / 0.4
      : level === "Medium"
      ? (maxProb - 0.45) / 0.15
      : 0;
  return {
    level,
    weight: Math.min(1, Math.max(0, weight)),
    maxProb,
    topState,
  };
}

/** Small helper for UI badges & colors */
export const confidenceBadge = {
  colorByState: {
    D: "#ef4444",
    R: "#f59e0b",
    B: "#64748b",
    U: "#22c55e",
  } satisfies Record<StateKey, string>,
  render(z: ConfidenceZone) {
    const bg =
      z.level === "High"
        ? "bg-green-100 text-green-800 border-green-200"
        : z.level === "Medium"
        ? "bg-amber-100 text-amber-800 border-amber-200"
        : "bg-slate-100 text-slate-700 border-slate-200";
    return (
      <span className={`inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded border ${bg}`}>
        <i
          className="inline-block w-2 h-2 rounded-full"
          style={{ background: confidenceBadge.colorByState[z.topState] }}
        />
        {z.level} confidence
      </span>
    );
  },
};
