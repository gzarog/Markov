import type { CandleRow } from "./types";

type Bias = "long" | "short" | "none";

export function stochRsiGate(row: CandleRow | undefined, bias: Bias) {
  if (!row || bias === "none") return false;
  const { stochK, stochD } = row;
  if (stochK == null || stochD == null || Number.isNaN(stochK) || Number.isNaN(stochD)) {
    return false;
  }
  if (bias === "long") {
    return (stochK > stochD && stochK < 40) || stochK < 20;
  }
  return (stochK < stochD && stochK > 60) || stochK > 80;
}

export function emaRetestGate(row: CandleRow | undefined, bias: Bias) {
  if (!row || bias === "none") return false;
  const { ema50, ma200, ma200_slope, close } = row;
  if (ema50 == null || ma200 == null || close == null) return false;
  if (bias === "long") {
    return close >= ema50 && (ma200_slope ?? 0) >= 0;
  }
  return close <= ema50 && (ma200_slope ?? 0) <= 0;
}

export function emaTrendGate(row: CandleRow | undefined, bias: Bias) {
  if (!row || bias === "none") return false;
  const { ema10, ema50 } = row;
  if (ema10 == null || ema50 == null || Number.isNaN(ema10) || Number.isNaN(ema50)) return false;
  return bias === "long" ? ema10 >= ema50 : ema10 <= ema50;
}

export function ma200BiasGate(row: CandleRow | undefined, bias: Bias) {
  if (!row || bias === "none") return false;
  const { ma200, ma200_slope, close } = row;
  if (ma200 == null || close == null || Number.isNaN(ma200) || Number.isNaN(close)) return false;
  if (bias === "long") {
    return close >= ma200 && (ma200_slope ?? 0) >= 0;
  }
  return close <= ma200 && (ma200_slope ?? 0) <= 0;
}

export function rsiGate(row: CandleRow | undefined, bias: Bias) {
  if (!row || bias === "none") return false;
  const r = row.rsi14 ?? row.rsi;
  if (r == null || Number.isNaN(r)) return false;
  const stochK = row.stochK ?? NaN;
  const stochD = row.stochD ?? NaN;
  if (bias === "long") {
    return r >= 50 || (r > 45 && stochK > stochD);
  }
  return r <= 50 || (r < 55 && stochK < stochD);
}

export function confluenceScore(row: CandleRow | undefined, bias: Bias) {
  if (!row || bias === "none") return 0;
  const checks = [
    { ok: stochRsiGate(row, bias), weight: 1 },
    { ok: emaRetestGate(row, bias), weight: 1.2 },
    { ok: emaTrendGate(row, bias), weight: 1.1 },
    { ok: ma200BiasGate(row, bias), weight: 1.2 },
    { ok: rsiGate(row, bias), weight: 1 },
  ];
  const totalWeight = checks.reduce((sum, item) => sum + item.weight, 0);
  if (totalWeight === 0) return 0;
  const scored = checks.reduce((sum, item) => sum + (item.ok ? item.weight : 0), 0);
  return scored / totalWeight;
}
