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
