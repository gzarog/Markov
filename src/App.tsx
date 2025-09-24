import React, { useEffect, useMemo, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  LineChart,
  Line,
  ResponsiveContainer,
  ReferenceLine,
  Area,
} from "recharts";

import { SectionCard } from "./components/common/SectionCard";
import { NumL, SelectL } from "./components/common/FormControls";
import { useBybitData } from "./hooks/useBybitData";
import { computeAll } from "./lib/compute";
import { isFiniteNum } from "./lib/utils";
import {
  buildMarkovWeighted,
  multiStepForecast,
  semiMarkovAdjustFirstRow,
  estimateDurations,
  computeRunLength,
  buildOrder2Counts,
  rowFromOrder2,
  PairKey,
} from "./lib/markov";
import {
  blendRows,
  buildFeatures,
  conditionedRow,
  defaultLogitModel,
  LogitModel,
} from "./lib/models/conditioning";
import { stepsForHorizon } from "./lib/models/horizons";
import { loadLogitModel, saveLogitModel } from "./lib/models/storage";
import { positionSizeUSD, stochrsiFilterPass, suggestTradeLevels, executionFilters } from "./lib/trading";
import { CandleRow, IDX, STATES, StateKey } from "./types/market";

function arrayClose(a: number[], b: number[], eps = 1e-6) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i] - b[i]) > eps) return false;
  }
  return true;
}

function matrixClose(a: number[][], b: number[][], eps = 1e-6) {
  if (a.length !== b.length) return false;
  for (let r = 0; r < a.length; r++) {
    if (a[r].length !== b[r].length) return false;
    for (let c = 0; c < a[r].length; c++) {
      if (Math.abs(a[r][c] - b[r][c]) > eps) return false;
    }
  }
  return true;
}

function logitModelsEqual(a: LogitModel | null | undefined, b: LogitModel | null | undefined) {
  if (!a || !b) return false;
  if (Math.abs(a.temperature - b.temperature) > 1e-6) return false;
  if (!matrixClose(a.W, b.W)) return false;
  if (!arrayClose(a.b, b.b)) return false;
  if (!arrayClose(a.mean, b.mean)) return false;
  if (!arrayClose(a.std, b.std)) return false;
  return true;
}

export default function App() {
  const [symbol, setSymbol] = useState("DOGEUSDT");
  const [interval, setIntervalStr] = useState("15");
  const [limit] = useState(1500);
  const [windowN, setWindowN] = useState(1200);
  const [halfLife, setHalfLife] = useState(300);
  const [smooth, setSmooth] = useState(0.5);
  const [horizons, setHorizons] = useState<number[]>([1, 2, 4, 6]);
  const [refreshSel, setRefreshSel] = useState(5);

  const [riskMode, setRiskMode] = useState<'atr' | 'conservative' | 'aggressive'>("atr");
  const [rr, setRR] = useState(2.0);
  const [atrSL, setAtrSL] = useState(1.2);
  const [atrTP, setAtrTP] = useState(1.8);
  const [stochMode, setStochMode] = useState<'hard' | 'soft' | 'off'>("hard");
  const [stochLo, setStochLo] = useState(20);
  const [stochHi, setStochHi] = useState(80);
  const [account, setAccount] = useState(500);
  const [riskPct, setRiskPct] = useState(2);
  const [leverage, setLeverage] = useState(20);
  const [takerBps, setTakerBps] = useState(7);
  const [makerBps, setMakerBps] = useState(1);

  const [filterCooldown, setFilterCooldown] = useState(2);
  const [filterMinRange, setFilterMinRange] = useState(0.5);
  const [filterMaxRange, setFilterMaxRange] = useState(3.5);
  const [filterMinBody, setFilterMinBody] = useState(0.15);
  const [macdGate, setMacdGate] = useState<'confirm' | 'off'>("confirm");
  const [macdDiffMin, setMacdDiffMin] = useState(0);
  const [macdHistSlopeMin, setMacdHistSlopeMin] = useState(0);
  const [bbLongMax, setBbLongMax] = useState(0.65);
  const [bbShortMin, setBbShortMin] = useState(0.35);
  const [bbWidthMinPct, setBbWidthMinPct] = useState(1.2);
  const [vwapLimitPct, setVwapLimitPct] = useState(0.8);

  const [persistedLogit, setPersistedLogit] = useState<LogitModel | null>(() => loadLogitModel());

  const refreshMs = useMemo(() => (refreshSel > 0 ? refreshSel * 60 * 1000 : null), [refreshSel]);

  const cfg = useMemo(
    () => ({ window: windowN, halfLife, smooth, horizons, intervalMinutes: Number(interval) }),
    [windowN, halfLife, smooth, horizons, interval]
  );

  const { data: candles, error, loading, refetch } = useBybitData({ symbol, interval, limit, refreshMs });

  const calc = useMemo(() => {
    if (!candles || candles.length < 210) return null as any;
    try {
      return computeAll(candles, cfg, { prevLogitModel: persistedLogit ?? undefined });
    } catch (e) {
      return { error: String(e) } as any;
    }
  }, [candles, cfg, persistedLogit]);

  useEffect(() => {
    const model = (calc as any)?.logitModel as LogitModel | undefined;
    if (!model) return;
    if (!logitModelsEqual(persistedLogit, model)) {
      saveLogitModel(model);
      setPersistedLogit(model);
    }
  }, [calc, persistedLogit]);

  const activeLogitModel = ((calc as any)?.logitModel as LogitModel | undefined) ?? persistedLogit ?? defaultLogitModel();

  const alignSeries = useMemo(() => {
    if (!calc) return [] as number[];
    const rows: CandleRow[] = calc.rows;
    const states = calc.states as StateKey[];
    const nearestSteps = horizons.length ? stepsForHorizon(horizons[0], Number(interval), rows) : 1;
    const vals = new Array(rows.length).fill(0) as number[];
    for (let t = 250; t < rows.length - nearestSteps; t++) {
      const histStates = states.slice(0, t + 1) as StateKey[];
      const { probs } = buildMarkovWeighted(histStates, windowN, smooth, halfLife);
      const cur = histStates[histStates.length - 1];
      const durations = estimateDurations(histStates);
      const runLen = computeRunLength(histStates);
      const baseRow = probs[IDX[cur]].slice();
      const durationRow = semiMarkovAdjustFirstRow(baseRow, histStates, { durations, runLength: runLen });
      const orderCounts = buildOrder2Counts(histStates);
      const pair: PairKey | null = histStates.length >= 2 ? (`${histStates[histStates.length - 2]}${cur}` as PairKey) : null;
      const orderRow = pair ? rowFromOrder2(orderCounts, pair) : null;
      const condRow = t >= 20 ? conditionedRow(buildFeatures(rows, t), activeLogitModel) : null;
      const components: number[][] = [durationRow];
      const weights: number[] = [0.6];
      if (orderRow) {
        components.push(orderRow);
        weights.push(0.25);
      }
      if (condRow) {
        components.push(condRow);
        weights.push(orderRow ? 0.15 : 0.4);
      }
      const blendedRow = blendRows(components, weights);
      const vec = multiStepForecast(probs, cur, { k: nearestSteps }, blendedRow)["k"];
      const up = vec[IDX["U"]] > vec[IDX["D"]];
      const kNow = rows[t].stochK ?? NaN;
      const pass = up ? kNow <= stochLo : kNow >= stochHi;
      if (pass) {
        vals[t] = up ? 1 : -1;
      }
    }
    return vals;
  }, [calc, halfLife, horizons, interval, smooth, stochHi, stochLo, windowN, activeLogitModel]);

  const oneStep = useMemo(() => {
    if (!calc) return null as any;
    const cur = calc.curState as StateKey;
    return {
      cur,
      raw: (calc.rowBase ?? []).slice(),
      duration: (calc.rowDuration ?? []).slice(),
      order2: calc.rowOrder2 ? calc.rowOrder2.slice() : null,
      conditioned: calc.rowConditioned ? calc.rowConditioned.slice() : null,
      blended: (calc.rowBlended ?? []).slice(),
    };
  }, [calc]);

  const tradeText = useMemo(() => {
    if (!calc) return "";
    const last: CandleRow = calc.rows[calc.rows.length - 1];
    const lines = ["Trade suggestions (EMA50 retest + StochRSI timing):"];
    const diffThreshold = isFiniteNum(macdDiffMin) ? Math.max(0, macdDiffMin) : 0;
    const slopeThreshold = isFiniteNum(macdHistSlopeMin) ? Math.max(0, macdHistSlopeMin) : 0;
    const longCap = isFiniteNum(bbLongMax) ? Math.min(1, Math.max(0, bbLongMax)) : 0.65;
    const shortFloor = isFiniteNum(bbShortMin) ? Math.min(1, Math.max(0, bbShortMin)) : 0.35;
    const widthMin = isFiniteNum(bbWidthMinPct) ? Math.max(0, bbWidthMinPct) : 0;
    const vwapTol = isFiniteNum(vwapLimitPct) ? Math.max(0, Math.abs(vwapLimitPct)) : Number.POSITIVE_INFINITY;
    (["short", "long"] as const).forEach((side) => {
      const osc = stochrsiFilterPass(last.stochK ?? NaN, side as any, stochMode, stochLo, stochHi);
      lines.push(`- ${side.toUpperCase()} oscillator check: ${osc.note}`);
      if (!osc.ok) {
        lines.push("  BLOCKED by StochRSI.");
        return;
      }
      const exec = executionFilters(last, calc.rows as CandleRow[], {
        side: side as any,
        runLength: calc.runLength ?? 0,
        minRunLength: filterCooldown,
        minRangeRatio: filterMinRange,
        maxRangeRatio: filterMaxRange,
        minBody: filterMinBody,
      });
      if (!exec.ok) {
        lines.push(`  Execution filters: BLOCK (${exec.notes.join(", ") || "conditions not met"})`);
        lines.push("  BLOCKED by execution filters.");
        return;
      }
      lines.push("  Execution filters: PASS");

      const prev = calc.rows[calc.rows.length - 2] as CandleRow | undefined;
      const macdDiff = (last.macd ?? NaN) - (last.macdSignal ?? NaN);
      const prevHist = prev?.macdHist ?? NaN;
      const histSlope =
        isFiniteNum(last.macdHist ?? NaN) && isFiniteNum(prevHist) ? (last.macdHist ?? 0) - prevHist : NaN;
      const bandSpan = (last.bbUpper ?? NaN) - (last.bbLower ?? NaN);
      const bbPos =
        isFiniteNum(bandSpan) && bandSpan !== 0 ? (last.close - (last.bbLower ?? last.close)) / bandSpan : NaN;
      const bbWidthPct =
        isFiniteNum(bandSpan) && isFiniteNum(last.close) ? (bandSpan / last.close) * 100 : NaN;
      const vwapDeltaPct = isFiniteNum(last.vwap ?? NaN)
        ? ((last.close - (last.vwap ?? 0)) / Math.max(1e-9, last.vwap ?? 1)) * 100
        : NaN;
      const indicatorNotes: string[] = [];
      let indicatorOk = true;

      if (macdGate === "confirm") {
        if (isFiniteNum(macdDiff)) {
          const needed = diffThreshold;
          const pass = side === "long" ? macdDiff >= needed : macdDiff <= -needed;
          if (!pass) {
            const target = side === "long" ? needed : -needed;
            indicatorOk = false;
            indicatorNotes.push(`MACD diff ${macdDiff.toFixed(4)} vs ${target.toFixed(4)}`);
          }
        } else {
          indicatorOk = false;
          indicatorNotes.push("MACD diff unavailable");
        }
      }

      if (slopeThreshold > 0) {
        if (isFiniteNum(histSlope)) {
          if (side === "long" && histSlope < slopeThreshold) {
            indicatorOk = false;
            indicatorNotes.push(`MACD slope ${histSlope.toFixed(4)} < ${slopeThreshold.toFixed(4)}`);
          }
          if (side === "short" && histSlope > -slopeThreshold) {
            indicatorOk = false;
            indicatorNotes.push(`MACD slope ${histSlope.toFixed(4)} > ${(-slopeThreshold).toFixed(4)}`);
          }
        } else {
          indicatorOk = false;
          indicatorNotes.push("MACD slope unavailable");
        }
      }

      if (isFiniteNum(bbPos)) {
        if (side === "long" && bbPos > longCap) {
          indicatorOk = false;
          indicatorNotes.push(`BB pos ${(bbPos * 100).toFixed(1)}% > ${(longCap * 100).toFixed(1)}% cap`);
        }
        if (side === "short" && bbPos < shortFloor) {
          indicatorOk = false;
          indicatorNotes.push(`BB pos ${(bbPos * 100).toFixed(1)}% < ${(shortFloor * 100).toFixed(1)}% floor`);
        }
      } else {
        indicatorNotes.push("Bollinger unavailable");
      }

      if (widthMin > 0) {
        if (isFiniteNum(bbWidthPct)) {
          if (bbWidthPct < widthMin) {
            indicatorOk = false;
            indicatorNotes.push(`BB width ${bbWidthPct.toFixed(1)}% < ${widthMin.toFixed(1)}% minimum`);
          }
        } else {
          indicatorOk = false;
          indicatorNotes.push("Bollinger width unavailable");
        }
      }

      if (Number.isFinite(vwapTol)) {
        if (isFiniteNum(vwapDeltaPct)) {
          if (side === "long" && vwapDeltaPct > vwapTol) {
            indicatorOk = false;
            indicatorNotes.push(`VWAP delta ${vwapDeltaPct.toFixed(2)}% > ${vwapTol.toFixed(2)}%`);
          }
          if (side === "short" && vwapDeltaPct < -vwapTol) {
            indicatorOk = false;
            indicatorNotes.push(`VWAP delta ${vwapDeltaPct.toFixed(2)}% < ${(-vwapTol).toFixed(2)}%`);
          }
        } else {
          indicatorOk = false;
          indicatorNotes.push("VWAP delta unavailable");
        }
      }

      if (!indicatorOk) {
        const note = indicatorNotes.filter(Boolean).join(", ") || "conditions not met";
        lines.push(`  Indicators: BLOCK (${note})`);
        lines.push("  BLOCKED by indicator stack.");
        return;
      }

      const summaryParts: string[] = [];
      if (isFiniteNum(macdDiff)) summaryParts.push(`MACD diff ${macdDiff.toFixed(4)}`);
      if (isFiniteNum(histSlope)) summaryParts.push(`MACD slope ${histSlope.toFixed(4)}`);
      if (isFiniteNum(bbPos))
        summaryParts.push(`BB pos ${(Math.max(0, Math.min(1, bbPos)) * 100).toFixed(1)}%`);
      if (isFiniteNum(bbWidthPct)) summaryParts.push(`Band width ${bbWidthPct.toFixed(1)}%`);
      if (isFiniteNum(vwapDeltaPct)) summaryParts.push(`VWAP delta ${vwapDeltaPct.toFixed(2)}%`);
      if (isFiniteNum(last.rsi ?? NaN)) summaryParts.push(`RSI ${(last.rsi ?? NaN).toFixed(1)}`);
      lines.push(`  Indicators: PASS (${summaryParts.join(" | ")})`);

      const combinedPenalty = osc.penalty;
      const lv = suggestTradeLevels(
        last,
        calc.rows as CandleRow[],
        side as any,
        riskMode,
        rr,
        atrSL,
        atrTP,
        combinedPenalty
      );
      const sz = positionSizeUSD(account, riskPct, lv.entry, lv.stop, leverage, takerBps, makerBps);
      lines.push(
        `  Entry: ${lv.entry.toFixed(6)}  Stop: ${lv.stop.toFixed(6)}  Target: ${lv.target.toFixed(6)}`
      );
      if (isFiniteNum(sz.notional)) {
        lines.push(
          `  Sizing: ~${sz.notional.toFixed(2)} USDT notional  (~${sz.qty.toFixed(2)} coins)  @x${leverage}  fees~${(
            2 * takerBps
          ).toFixed(1)} bps total`
        );
      }
    });
    return lines.join("\n");
  }, [
    account,
    atrSL,
    atrTP,
    bbLongMax,
    bbShortMin,
    bbWidthMinPct,
    calc,
    filterCooldown,
    filterMaxRange,
    filterMinBody,
    filterMinRange,
    leverage,
    macdDiffMin,
    macdGate,
    macdHistSlopeMin,
    makerBps,
    riskMode,
    riskPct,
    rr,
    stochHi,
    stochLo,
    stochMode,
    takerBps,
    vwapLimitPct,
  ]);
  const footerText = useMemo(() => {
    const auto = refreshSel === 0 ? 'OFF' : `${refreshSel} min`;
    let txt = `Markov+RSI/StochRSI SPA | Auto refresh: ${auto} | ` + (loading ? 'Updating.' : 'Live');
    if (error) {
      txt += ` | Error: ${error}`;
    }
    return txt;
  }, [refreshSel, loading, error]);

  const topPanel = (
    <div className="w-full bg-white/80 backdrop-blur sticky top-0 z-10 border-b shadow-sm">
      <div className="max-w-screen-2xl mx-auto px-3 sm:px-4 py-2 sm:py-3">
        <div className="flex gap-2 overflow-x-auto no-scrollbar pb-2 sm:pb-0 sm:flex-wrap items-end">
          <div className="min-w-[130px]">
            <div className="text-[10px] sm:text-xs text-gray-600">Symbol</div>
            <input className="border rounded px-2 py-1 w-full" value={symbol} onChange={(e) => setSymbol(e.target.value.toUpperCase())} />
          </div>
          <div className="min-w-[130px]">
            <div className="text-[10px] sm:text-xs text-gray-600">Interval (min)</div>
            <select className="border rounded px-2 py-1 w-full" value={interval} onChange={(e) => setIntervalStr(e.target.value)}>
              {['1', '3', '5', '15', '30', '60', '240'].map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </div>
          <div className="min-w-[120px]">
            <div className="text-[10px] sm:text-xs text-gray-600">Window</div>
            <input type="number" className="border rounded px-2 py-1 w-full" value={windowN} onChange={(e) => setWindowN(Number(e.target.value))} />
          </div>
          <div className="min-w-[120px]">
            <div className="text-[10px] sm:text-xs text-gray-600">Half-life</div>
            <input type="number" className="border rounded px-2 py-1 w-full" value={halfLife} onChange={(e) => setHalfLife(Number(e.target.value))} />
          </div>
          <div className="min-w-[110px]">
            <div className="text-[10px] sm:text-xs text-gray-600">Smoothing</div>
            <input type="number" step={0.1} className="border rounded px-2 py-1 w-full" value={smooth} onChange={(e) => setSmooth(Number(e.target.value))} />
          </div>
          <div className="min-w-[150px]">
            <div className="text-[10px] sm:text-xs text-gray-600">Horizons (h)</div>
            <select
              multiple
              className="border rounded px-2 py-1 w-full"
              value={horizons.map(String)}
              onChange={(e) => {
                const vals = Array.from(e.target.selectedOptions).map((o) => Number(o.value));
                setHorizons(vals);
              }}
            >
              {[1, 2, 4, 6, 12, 24].map((h) => (
                <option key={h} value={String(h)}>
                  {h}
                </option>
              ))}
            </select>
          </div>
          <div className="min-w-[140px]">
            <div className="text-[10px] sm:text-xs text-gray-600">Auto Refresh</div>
            <select className="border rounded px-2 py-1 w-full" value={String(refreshSel)} onChange={(e) => setRefreshSel(Number(e.target.value))}>
              {[1, 3, 5, 10, 0].map((m) => (
                <option key={m} value={String(m)}>
                  {m === 0 ? 'Off' : `${m} min`}
                </option>
              ))}
            </select>
          </div>
          <div className="ml-auto flex items-end">
            <button onClick={refetch} className="bg-black text-white rounded px-3 py-1 whitespace-nowrap">
              Refresh
            </button>
          </div>
        </div>
      </div>

      <details className="sm:hidden border-t px-3 py-2">
        <summary className="text-sm font-medium cursor-pointer">Trade config</summary>
        <div className="grid grid-cols-2 gap-2 pt-2">
          <SelectL label="Risk mode" value={riskMode} onChange={(v) => setRiskMode(v as any)} options={['atr', 'conservative', 'aggressive']} />
          <NumL label="RR" value={rr} set={setRR} step={0.1} />
          <NumL label="ATR SL" value={atrSL} set={setAtrSL} step={0.1} />
          <NumL label="ATR TP" value={atrTP} set={setAtrTP} step={0.1} />
          <SelectL label="StochRSI mode" value={stochMode} onChange={(v) => setStochMode(v as any)} options={['hard', 'soft', 'off']} />
          <NumL label="StochRSI <= (long)" value={stochLo} set={setStochLo} />
          <NumL label="StochRSI >= (short)" value={stochHi} set={setStochHi} />
          <NumL label="Account" value={account} set={setAccount} />
          <NumL label="Risk %" value={riskPct} set={setRiskPct} />
          <div className="col-span-2 grid grid-cols-3 gap-2">
            <NumL label="Lev" value={leverage} set={setLeverage} />
            <NumL label="Taker bps" value={takerBps} set={setTakerBps} />
            <NumL label="Maker bps" value={makerBps} set={setMakerBps} />
          </div>
          <div className="col-span-2 grid grid-cols-2 gap-2">
            <NumL label="Filter cooldown" value={filterCooldown} set={setFilterCooldown} />
            <NumL label="Min range ratio" value={filterMinRange} set={setFilterMinRange} step={0.1} />
            <NumL label="Max range ratio" value={filterMaxRange} set={setFilterMaxRange} step={0.1} />
            <NumL label="Min body fraction" value={filterMinBody} set={setFilterMinBody} step={0.05} />
          </div>
          <div className="col-span-2 grid grid-cols-2 gap-2">
            <SelectL label="MACD gate" value={macdGate} onChange={(v) => setMacdGate(v as 'confirm' | 'off')} options={['confirm', 'off']} />
            <NumL label="MACD diff min" value={macdDiffMin} set={setMacdDiffMin} step={0.0001} />
            <NumL label="MACD slope min" value={macdHistSlopeMin} set={setMacdHistSlopeMin} step={0.0001} />
            <NumL label="BB long max (0-1)" value={bbLongMax} set={setBbLongMax} step={0.01} />
            <NumL label="BB short min (0-1)" value={bbShortMin} set={setBbShortMin} step={0.01} />
            <NumL label="BB width min (%)" value={bbWidthMinPct} set={setBbWidthMinPct} step={0.1} />
            <NumL label="VWAP tolerance (%)" value={vwapLimitPct} set={setVwapLimitPct} step={0.1} />
          </div>
        </div>
      </details>

      <div className="hidden sm:block border-t">
        <div className="max-w-screen-2xl mx-auto px-4 py-2 grid grid-cols-4 lg:grid-cols-8 gap-3">
          <SelectL label="Risk mode" value={riskMode} onChange={(v) => setRiskMode(v as any)} options={['atr', 'conservative', 'aggressive']} />
          <NumL label="RR" value={rr} set={setRR} step={0.1} />
          <NumL label="ATR SL" value={atrSL} set={setAtrSL} step={0.1} />
          <NumL label="ATR TP" value={atrTP} set={setAtrTP} step={0.1} />
          <SelectL label="StochRSI mode" value={stochMode} onChange={(v) => setStochMode(v as any)} options={['hard', 'soft', 'off']} />
          <NumL label="StochRSI <= (long)" value={stochLo} set={setStochLo} />
          <NumL label="StochRSI >= (short)" value={stochHi} set={setStochHi} />
          <div className="grid grid-cols-3 gap-2 col-span-4 lg:col-span-2">
            <NumL label="Account" value={account} set={setAccount} />
            <NumL label="Risk %" value={riskPct} set={setRiskPct} />
            <NumL label="Lev" value={leverage} set={setLeverage} />
          </div>
          <div className="grid grid-cols-2 gap-2 col-span-4 lg:col-span-2">
            <NumL label="Taker bps" value={takerBps} set={setTakerBps} />
            <NumL label="Maker bps" value={makerBps} set={setMakerBps} />
          </div>
          <div className="grid grid-cols-2 gap-2 col-span-4 lg:col-span-2">
            <NumL label="Filter cooldown" value={filterCooldown} set={setFilterCooldown} />
            <NumL label="Min range ratio" value={filterMinRange} set={setFilterMinRange} step={0.1} />
            <NumL label="Max range ratio" value={filterMaxRange} set={setFilterMaxRange} step={0.1} />
            <NumL label="Min body fraction" value={filterMinBody} set={setFilterMinBody} step={0.05} />
          </div>
          <div className="grid grid-cols-3 gap-2 col-span-4 lg:col-span-3">
            <SelectL label="MACD gate" value={macdGate} onChange={(v) => setMacdGate(v as 'confirm' | 'off')} options={['confirm', 'off']} />
            <NumL label="MACD diff min" value={macdDiffMin} set={setMacdDiffMin} step={0.0001} />
            <NumL label="MACD slope min" value={macdHistSlopeMin} set={setMacdHistSlopeMin} step={0.0001} />
            <NumL label="BB long max (0-1)" value={bbLongMax} set={setBbLongMax} step={0.01} />
            <NumL label="BB short min (0-1)" value={bbShortMin} set={setBbShortMin} step={0.01} />
            <NumL label="BB width min (%)" value={bbWidthMinPct} set={setBbWidthMinPct} step={0.1} />
            <NumL label="VWAP tolerance (%)" value={vwapLimitPct} set={setVwapLimitPct} step={0.1} />
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      {topPanel}

      <div className="max-w-screen-2xl mx-auto px-3 sm:px-4 lg:px-6 py-4 grid gap-4 lg:gap-6 xl:gap-8">
        {!calc || (calc as any).error ? (
          <SectionCard title="Status">
            <div className="text-sm text-gray-500">
              {loading ? 'Loading data...' : (calc as any)?.error || 'No data'}
            </div>
          </SectionCard>
        ) : null}

        <SectionCard title="Markov Forecast (horizons)">
          {!calc ? (
            <div className="text-sm text-gray-500">{loading ? 'Loading...' : 'No forecast'}</div>
          ) : (
            <div className="chart-300">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={STATES.map((s) => ({
                    state: s,
                    ...Object.fromEntries(
                      Object.entries(calc.forecasts).map(([k, v]) => [k, (v as number[])[IDX[s]] ?? 0])
                    ),
                  }))}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="state" />
                  <YAxis domain={[0, 1]} tickFormatter={(v: number) => (v * 100).toFixed(0) + '%'} />
                  <Tooltip formatter={(v: any) => (Number(v) * 100).toFixed(1) + '%'} />
                  <Legend />
                  {Object.keys(calc.forecasts).map((k, i) => (
                    <Bar key={k} dataKey={k} fill={`hsl(${(i * 60) % 360},70%,50%)`} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </SectionCard>

        <SectionCard
          title="Current state next-step probabilities"
          right={oneStep && <span className="text-xs text-gray-500">state: <b>{oneStep.cur}</b></span>}
        >
          {!calc || !oneStep ? (
            <div className="text-sm text-gray-500">{loading ? 'Loading...' : 'No data'}</div>
          ) : (
            <div className="h-[200px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={STATES.map((s) => ({
                    state: s,
                    raw: (oneStep.raw ?? [])[IDX[s]] ?? 0,
                    duration: (oneStep.duration ?? [])[IDX[s]] ?? 0,
                    blended: (oneStep.blended ?? [])[IDX[s]] ?? 0,
                  }))}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="state" />
                  <YAxis domain={[0, 1]} tickFormatter={(v: number) => (v * 100).toFixed(0) + '%'} />
                  <Tooltip formatter={(v: any) => (Number(v) * 100).toFixed(1) + '%'} />
                  <Legend />
                  <Bar dataKey="raw" fill="#94a3b8" name="raw" />
                  <Bar dataKey="duration" fill="#0ea5e9" name="duration" />
                  <Bar dataKey="blended" fill="#fb923c" name="blended" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </SectionCard>

        <SectionCard title="Trade suggestions (EMA50 retest + StochRSI timing)">
          {!calc ? (
            <div className="text-sm text-gray-500">{loading ? 'Loading...' : 'No data'}</div>
          ) : (
            <pre className="whitespace-pre-wrap text-[12px] leading-6 font-mono">{tradeText}</pre>
          )}
        </SectionCard>

        <SectionCard title="RSI & StochRSI - shaded alignment (Markov + StochRSI)" className="xl:col-span-2">
          {!calc ? (
            <div className="text-sm text-gray-500">{loading ? 'Loading...' : 'No data'}</div>
          ) : (
            <div className="chart-380">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={calc.rows.map((r: CandleRow, i: number) => ({
                    t: r.time,
                    RSI: r.rsi,
                    K: r.stochK,
                    D: r.stochD,
                    align: alignSeries[i],
                  }))}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="t" tickFormatter={(t: any) => new Date(t).toLocaleString()} minTickGap={48} />
                  <YAxis domain={[0, 100]} />
                  <Tooltip labelFormatter={(t: any) => new Date(t).toLocaleString()} />
                  <Legend />
                  <ReferenceLine y={80} stroke="#999" strokeDasharray="3 3" />
                  <ReferenceLine y={50} stroke="#999" strokeDasharray="3 3" />
                  <ReferenceLine y={20} stroke="#999" strokeDasharray="3 3" />
                  <Area
                    type="monotone"
                    dataKey={(d: any) => (d.align === 1 ? 100 : undefined)}
                    isAnimationActive={false}
                    dot={false}
                    stroke="none"
                    fill="rgba(16,185,129,0.12)"
                  />
                  <Area
                    type="monotone"
                    dataKey={(d: any) => (d.align === -1 ? 100 : undefined)}
                    isAnimationActive={false}
                    dot={false}
                    stroke="none"
                    fill="rgba(239,68,68,0.12)"
                  />
                  <Line type="monotone" dataKey="RSI" stroke="#2563eb" dot={false} />
                  <Line type="monotone" dataKey="K" stroke="#f59e0b" dot={false} />
                  <Line type="monotone" dataKey="D" stroke="#16a34a" strokeDasharray="4 3" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </SectionCard>
      </div>

      <footer className="text-center text-xs text-gray-500 py-6">{footerText}</footer>
    </div>
  );
}
