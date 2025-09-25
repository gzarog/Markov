import { CandleRow } from "../types/market";
import { clamp, isFiniteNum } from "./utils";

type StochMode = "hard" | "soft" | "off";

type RiskMode = "atr" | "conservative" | "aggressive";

type TradeSide = "long" | "short";

type ForecastContext = {
  bullish: number;
  bearish: number;
  base?: number;
  reversal?: number;
  confidence?: number;
};

export function stochrsiFilterPass(
  kVal: number,
  side: TradeSide,
  mode: StochMode,
  lower: number,
  upper: number,
  bias?: ForecastContext
) {
  if (!isFiniteNum(kVal)) return { ok: true, note: "StochRSI: insufficient history", penalty: 1 };
  if (mode === "off") return { ok: true, note: "StochRSI off", penalty: 1 };
  const tilt = clamp((bias?.bullish ?? 0) - (bias?.bearish ?? 0), -1, 1);
  const confidence = clamp(bias?.confidence ?? Math.abs(tilt), 0, 1);
  const adjust = tilt * confidence * 10;
  if (side === "long") {
    const dynLower = lower + adjust;
    const cond = kVal <= dynLower;
    const note = `K=${kVal.toFixed(1)} ≤ ${dynLower.toFixed(1)} (base ${lower})  ${cond ? "PASS" : "BLOCK"}`;
    const penalty = cond ? 1 : Math.max(0.55, 1 - confidence * (tilt > 0 ? 0.35 : 0.2));
    return { ok: mode === "hard" ? cond : true, note, penalty };
  }
  const dynUpper = upper - adjust;
  const cond = kVal >= dynUpper;
  const note = `K=${kVal.toFixed(1)} ≥ ${dynUpper.toFixed(1)} (base ${upper})  ${cond ? "PASS" : "BLOCK"}`;
  const penalty = cond ? 1 : Math.max(0.55, 1 - confidence * (tilt < 0 ? 0.35 : 0.2));
  return { ok: mode === "hard" ? cond : true, note, penalty };
}

export function suggestTradeLevels(
  row: CandleRow,
  hist: CandleRow[],
  side: TradeSide,
  riskMode: RiskMode,
  rr: number,
  atrSL: number,
  atrTP: number,
  rrPenalty = 1,
  probContext?: ForecastContext
) {
  const close = row.close;
  const ema50 = row.ema50 ?? row.close;
  const atr = row.atr14 ?? NaN;
  const highs = hist.slice(-51, -1).map((r) => r.high);
  const lows = hist.slice(-51, -1).map((r) => r.low);
  const swingHigh = highs.length ? Math.max(...highs) : row.high;
  const swingLow = lows.length ? Math.min(...lows) : row.low;
  const directionalEdge = clamp((probContext?.bullish ?? 0) - (probContext?.bearish ?? 0), -1, 1);
  const biasConfidence = clamp(probContext?.confidence ?? Math.abs(directionalEdge), 0, 1);
  const rrEffBase = (isFiniteNum(rr) ? rr : 0) * (isFiniteNum(rrPenalty) ? rrPenalty : 1);

  let entry: number;
  let stop: number;
  let target: number;

  if (side === "short") {
    const shortBias = -directionalEdge;
    entry = isFiniteNum(ema50) && isFiniteNum(atr) ? Math.min(ema50, close + 0.3 * atr) : ema50;
    if (riskMode === "conservative" || shortBias > 0.4) {
      stop = isFiniteNum(atr) ? swingHigh + 0.25 * atr : swingHigh;
    } else if (riskMode === "aggressive" && shortBias < 0.1) {
      stop = isFiniteNum(atr) ? ema50 + 0.4 * atr : ema50;
    } else {
      stop = isFiniteNum(atr) ? entry + (atrSL || 0) * atr * (1 + shortBias * 0.3) : entry * 1.01;
    }
    const risk = Math.max(1e-12, Math.abs(entry - stop));
    const rrBias = shortBias > 0 ? 1 + shortBias * 0.4 : 1 + shortBias * 0.2;
    const rrEff = rrEffBase * (isFiniteNum(rr) ? clamp(rrBias, 0.5, 1.6) : 1);
    const atrBias = 1 + shortBias * 0.3 * biasConfidence;
    const tgt_rr = entry - rrEff * risk;
    const tgt_atr = isFiniteNum(atr) ? entry - (atrTP || 0) * atr * atrBias : entry - 0.01 * entry;
    target = isFiniteNum(rr) ? tgt_rr : tgt_atr;
  } else {
    entry = isFiniteNum(ema50) && isFiniteNum(atr) ? Math.max(ema50, close - 0.3 * atr) : ema50;
    if (riskMode === "conservative" || directionalEdge < -0.4) {
      stop = isFiniteNum(atr) ? swingLow - 0.25 * atr : swingLow;
    } else if (riskMode === "aggressive" && directionalEdge > 0.1) {
      stop = isFiniteNum(atr) ? ema50 - 0.4 * atr : ema50;
    } else {
      stop = isFiniteNum(atr) ? entry - (atrSL || 0) * atr * (1 - directionalEdge * 0.3) : entry * 0.99;
    }
    const risk = Math.max(1e-12, Math.abs(entry - stop));
    const rrBias = directionalEdge > 0 ? 1 + directionalEdge * 0.4 : 1 + directionalEdge * 0.2;
    const rrEff = rrEffBase * (isFiniteNum(rr) ? clamp(rrBias, 0.5, 1.6) : 1);
    const atrBias = 1 + directionalEdge * 0.3 * biasConfidence;
    const tgt_rr = entry + rrEff * risk;
    const tgt_atr = isFiniteNum(atr) ? entry + (atrTP || 0) * atr * atrBias : entry + 0.01 * entry;
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
