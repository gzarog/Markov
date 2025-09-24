import { CandleRow } from "../../types/market";

export function stepsForHorizon(hours: number, intervalMinutes: number, rows: CandleRow[]) {
  const base = Math.max(1, Math.round((hours * 60) / Math.max(1, intervalMinutes)));
  if (rows.length < 30) return base;
  const lookback = Math.min(rows.length, 120);
  const recent = rows.slice(-lookback);
  const atrFracs = recent
    .map((row) => (row.atr14 ?? Math.max(1e-9, row.high - row.low)) / Math.max(1e-9, row.close))
    .sort((a, b) => a - b);
  const pctIndex = Math.max(0, Math.floor(0.8 * (atrFracs.length - 1)));
  const pct = atrFracs[pctIndex] ?? atrFracs[atrFracs.length - 1] ?? 0.01;
  const scale = Math.min(1.4, Math.max(0.7, 0.8 + 2 * (pct - 0.01)));
  return Math.max(1, Math.round(base * scale));
}
