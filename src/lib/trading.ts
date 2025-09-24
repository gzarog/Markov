import { CandleRow } from "../types/market";
import { isFiniteNum } from "./utils";

type StochMode = "hard" | "soft" | "off";

type RiskMode = "atr" | "conservative" | "aggressive";

type TradeSide = "long" | "short";

export function stochrsiFilterPass(
  kVal: number,
  side: TradeSide,
  mode: StochMode,
  lower: number,
  upper: number
) {
  if (!isFiniteNum(kVal)) return { ok: true, note: "StochRSI: insufficient history", penalty: 1 };
  if (mode === "off") return { ok: true, note: "StochRSI off", penalty: 1 };
  if (side === "long") {
    const cond = kVal <= lower;
    const note = `K=${kVal.toFixed(1)} <= ${lower}  ${cond ? "PASS" : "BLOCK"}`;
    return { ok: mode === "hard" ? cond : true, note, penalty: cond ? 1 : 0.75 };
  }
  const cond = kVal >= upper;
  const note = `K=${kVal.toFixed(1)} >= ${upper}  ${cond ? "PASS" : "BLOCK"}`;
  return { ok: mode === "hard" ? cond : true, note, penalty: cond ? 1 : 0.75 };
}

export function suggestTradeLevels(
  row: CandleRow,
  hist: CandleRow[],
  side: TradeSide,
  riskMode: RiskMode,
  rr: number,
  atrSL: number,
  atrTP: number,
  rrPenalty = 1
) {
  const close = row.close;
  const ema50 = row.ema50 ?? row.close;
  const atr = row.atr14 ?? NaN;
  const highs = hist.slice(-51, -1).map((r) => r.high);
  const lows = hist.slice(-51, -1).map((r) => r.low);
  const swingHigh = highs.length ? Math.max(...highs) : row.high;
  const swingLow = lows.length ? Math.min(...lows) : row.low;
  let entry: number;
  let stop: number;
  let target: number;
  const rrEff = (isFiniteNum(rr) ? rr : 0) * (isFiniteNum(rrPenalty) ? rrPenalty : 1);

  if (side === "short") {
    entry = isFiniteNum(ema50) && isFiniteNum(atr) ? Math.min(ema50, close + 0.3 * atr) : ema50;
    if (riskMode === "conservative") stop = isFiniteNum(atr) ? swingHigh + 0.2 * atr : swingHigh;
    else if (riskMode === "aggressive") stop = isFiniteNum(atr) ? ema50 + 0.5 * atr : ema50;
    else stop = isFiniteNum(atr) ? entry + (atrSL || 0) * atr : entry * 1.01;
    const risk = Math.max(1e-12, Math.abs(entry - stop));
    const tgt_rr = entry - rrEff * risk;
    const tgt_atr = isFiniteNum(atr) ? entry - (atrTP || 0) * atr : entry - 0.01 * entry;
    target = isFiniteNum(rr) ? tgt_rr : tgt_atr;
  } else {
    entry = isFiniteNum(ema50) && isFiniteNum(atr) ? Math.max(ema50, close - 0.3 * atr) : ema50;
    if (riskMode === "conservative") stop = isFiniteNum(atr) ? swingLow - 0.2 * atr : swingLow;
    else if (riskMode === "aggressive") stop = isFiniteNum(atr) ? ema50 - 0.5 * atr : ema50;
    else stop = isFiniteNum(atr) ? entry - (atrSL || 0) * atr : entry * 0.99;
    const risk = Math.max(1e-12, Math.abs(entry - stop));
    const tgt_rr = entry + rrEff * risk;
    const tgt_atr = isFiniteNum(atr) ? entry + (atrTP || 0) * atr : entry + 0.01 * entry;
    target = isFiniteNum(rr) ? tgt_rr : tgt_atr;
  }

  return { entry, stop, target };
}

export function positionSizeUSD(
  account: number,
  riskPct: number,
  entry: number,
  stop: number,
  leverage: number,
  takerBps: number,
  makerBps: number
) {
  if (![account, riskPct, entry, stop, leverage, takerBps, makerBps].every(isFiniteNum)) {
    return { notional: null as any, qty: null as any };
  }
  const feeFrac = (takerBps + takerBps) / 10000;
  const riskPerUnit = Math.abs(entry - stop) + feeFrac * entry;
  const riskAmt = account * (riskPct / 100);
  if (!(riskPerUnit > 0)) return { notional: null as any, qty: null as any };
  const qty = (riskAmt * leverage) / (riskPerUnit * entry);
  const notional = qty * entry;
  return { notional, qty };
}
