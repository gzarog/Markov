import React, { useMemo, useState } from "react";
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
  buildTransitionCounts,
  smoothCountsToProbs,
  blendMatrices,
  forecastDistribution,
  distEntropy,
} from "./lib/markov";
import { computeIndicators, bucketVolSession } from "./lib/features";
import {
  stochRsiGate,
  emaRetestGate,
  emaTrendGate,
  ma200BiasGate,
  rsiGate,
  confluenceScore,
} from "./lib/entries";
import { entropySizing, edgeAfterCosts } from "./lib/sizing";
import { atrStopTake } from "./lib/exits";
import { zoneFromProbs, confidenceBadge, type ConfidenceZone } from "./lib/zones";
import { positionSizeUSD, stochrsiFilterPass, suggestTradeLevels } from "./lib/trading";
import { CandleRow, IDX, STATES, StateKey } from "./types/market";

// NOTE: This file uses Tailwind utility classes for layout/responsiveness.
// If you haven't set up Tailwind yet, I can give you the 60-second setup steps.
export default function App(){
  // --- Controls ---
  const [symbol, setSymbol] = useState("DOGEUSDT");
  const [interval, setIntervalStr] = useState("15");
  const [limit] = useState(1500);
  const [windowN, setWindowN] = useState(1200);
  const [halfLife, setHalfLife] = useState(300);
  const [smooth, setSmooth] = useState(0.5);
  const [horizons, setHorizons] = useState<number[]>([1,2,4,6]);
  const [order, setOrder] = useState(1);
  const [dirichlet, setDirichlet] = useState(2);
  const [autoHalfLife, setAutoHalfLife] = useState(true);
  const [confidenceGate, setConfidenceGate] = useState(0.45);
  const [refreshSel, setRefreshSel] = useState(5); // minutes

  // Trade panel
  const [riskMode, setRiskMode] = useState<'atr'|'conservative'|'aggressive'>("atr");
  const [rr, setRR] = useState(2.0);
  const [atrSL, setAtrSL] = useState(1.2);
  const [atrTP, setAtrTP] = useState(1.8);
  const [stochMode, setStochMode] = useState<'hard'|'soft'|'off'>("hard");
  const [stochLo, setStochLo] = useState(20);
  const [stochHi, setStochHi] = useState(80);
  const [account, setAccount] = useState(500);
  const [riskPct, setRiskPct] = useState(2);
  const [leverage, setLeverage] = useState(20);
  const [takerBps, setTakerBps] = useState(7);
  const [makerBps, setMakerBps] = useState(1);
  const [fundingBpsPer8h, setFundingBpsPer8h] = useState(1);
  const [useVolConditioning, setUseVolConditioning] = useState(true);

  const refreshMs = useMemo(()=> refreshSel>0? refreshSel*60*1000 : null, [refreshSel]);

  const cfg = useMemo(()=>({
    window: windowN,
    halfLife,
    smooth,
    horizons,
    intervalMinutes: Number(interval),
    order,
    dirichlet,
    autoHalfLife,
    confidenceGate,
  }), [windowN, halfLife, smooth, horizons, interval, order, dirichlet, autoHalfLife, confidenceGate]);

  const {data: candles, error, loading, refetch} = useBybitData({symbol, interval, limit, refreshMs});

  const calc = useMemo(()=>{
    if(!candles || candles.length<210) return null as any;
    try{ return computeAll(candles, cfg); }catch(e){ return {error: String(e)} as any; }
  }, [candles, cfg]);

  const enrichedRows = useMemo(() => {
    if (!calc || !calc.rows?.length) return [] as CandleRow[];
    const withState = (calc.rows as CandleRow[]).map((row, index) => ({
      ...row,
      state:
        (calc.states?.[index] as StateKey | undefined) ??
        (row.state as StateKey | undefined) ??
        (calc.learnedStates?.[index] as StateKey | undefined) ??
        (row.learnedState as StateKey | undefined) ??
        (calc.ruleStates?.[index] as StateKey | undefined) ??
        (row.ruleState as StateKey | undefined),
    }));
    return computeIndicators(withState);
  }, [calc]);

  const volMatrices = useMemo(() => {
    if (!enrichedRows.length) return null as null | { matGlobal: number[][]; matByBucket: Record<string, number[][]> };
    const countsGlobalRaw = buildTransitionCounts(enrichedRows, { decay: 0.995 });
    const globalCounts = Array.isArray(countsGlobalRaw)
      ? countsGlobalRaw
      : (countsGlobalRaw as Record<string, number[][]>).global ?? Array.from({ length: STATES.length }, () => Array(STATES.length).fill(0));
    const matGlobal = smoothCountsToProbs(globalCounts, 0.75);
    const countsByBucketRaw = buildTransitionCounts(enrichedRows, {
      decay: 0.995,
      bucketFn: (row) => bucketVolSession(row),
    });
    const matByBucket: Record<string, number[][]> = {};
    if (!Array.isArray(countsByBucketRaw)) {
      Object.entries(countsByBucketRaw).forEach(([key, counts]) => {
        matByBucket[key] = smoothCountsToProbs(counts, 0.75);
      });
    }
    return { matGlobal, matByBucket };
  }, [enrichedRows]);

  type ForecastEntry = { probs: number[]; entropy: number; bias: StateKey };
  type VolPanelState = {
    forecasts: Record<string, ForecastEntry>;
    zones: Record<string, ConfidenceZone>;
    bucketKey: string;
    agreement: { upVotes: number; downVotes: number };
    suggestion: any;
  };

  const volPanel = useMemo<VolPanelState | null>(() => {
    if (!calc || !enrichedRows.length || !volMatrices?.matGlobal) return null;

    const currentRow = enrichedRows[enrichedRows.length - 1];
    const startState = (calc.curState ?? currentRow?.state ?? "B") as StateKey;
    const bucketKey = currentRow ? bucketVolSession(currentRow) : "global";
    const bucketMat =
      useVolConditioning && volMatrices.matByBucket && bucketKey in volMatrices.matByBucket
        ? volMatrices.matByBucket[bucketKey]
        : undefined;
    const mat = blendMatrices(bucketMat, volMatrices.matGlobal, useVolConditioning ? 0.6 : 0);
    const intervalMinutes = Number(interval) || 1;
    const barsPerHour = Math.max(1, Math.round(60 / intervalMinutes));
    const horizonSteps: Record<string, number> = {
      "2h": 2 * barsPerHour,
      "4h": 4 * barsPerHour,
      "6h": 6 * barsPerHour,
    };
    const forecasts: Record<string, ForecastEntry> = Object.fromEntries(
      Object.entries(horizonSteps).map(([label, steps]) => {
        const probs = forecastDistribution({ mat, startState, steps });
        const max = Math.max(...probs);
        const biasIndex = probs.findIndex((value) => value === max);
        const bias = (STATES[biasIndex] ?? "B") as StateKey;
        return [label, { probs, entropy: distEntropy(probs), bias }];
      })
    );

    const zones: Record<string, ConfidenceZone> = {};
    Object.entries(forecasts).forEach(([label, forecast]) => {
      zones[label] = zoneFromProbs(forecast.probs);
    });

    const votes = { upVotes: 0, downVotes: 0 };
    Object.values(forecasts).forEach((forecast) => {
      if (forecast.bias === "U") votes.upVotes++;
      if (forecast.bias === "D") votes.downVotes++;
    });

    let suggestion: any = null;
    const primary = forecasts["2h"] ?? Object.values(forecasts)[0];
    const primaryZone = primary ? zones["2h"] ?? zoneFromProbs(primary.probs) : null;
    const fallbackZone: ConfidenceZone =
      primaryZone ?? { level: "Low", weight: 0, maxProb: 0, topState: "B" };
    if (currentRow && primary) {
      const upP = primary.probs[IDX["U"]];
      const downP = primary.probs[IDX["D"]];
      let bias: "long" | "short" | "none" = "none";
      if (upP - downP > 0.05) bias = "long";
      else if (downP - upP > 0.05) bias = "short";

      if (bias === "none") {
        suggestion = { reason: "No clear probabilistic bias" };
      } else if (votes.upVotes < 2 && votes.downVotes < 2) {
        suggestion = { reason: "Low agreement across horizons" };
      } else {
        const gates = {
          stoch: stochRsiGate(currentRow, bias),
          emaRetest: emaRetestGate(currentRow, bias),
          emaTrend: emaTrendGate(currentRow, bias),
          ma200Bias: ma200BiasGate(currentRow, bias),
          rsi: rsiGate(currentRow, bias),
        } as const;
        const conf = confluenceScore(currentRow, bias);
        const confCut =
          fallbackZone.level === "High" ? 0.5 : fallbackZone.level === "Medium" ? 0.55 : 0.6;
        if (conf < confCut) {
          suggestion = { reason: `Low confluence (${(conf * 100).toFixed(0)}%)`, gates };
        } else if (!Number.isFinite(currentRow.atr14) || !Number.isFinite(currentRow.close)) {
          suggestion = { reason: "Insufficient indicator coverage" };
        } else {
          const rv = Number.isFinite(currentRow.rv) ? (currentRow.rv as number) : 0;
          const approxEdge = (upP - downP) * Math.max(rv, 1e-4);
          const taker = (takerBps ?? 0) / 10000;
          const maker = (makerBps ?? 0) / 10000;
          const fundingCost = (fundingBpsPer8h / 10000) * (2 / 8);
          const netEdge = edgeAfterCosts(approxEdge, maker, taker, fundingCost);
          if (netEdge <= 0) {
            suggestion = { reason: "Edge does not clear costs" };
          } else {
            const entropySize = entropySizing(primary.entropy, 1);
            const zoneBoost = 0.9 + 0.4 * fallbackZone.weight;
            const size = entropySize * (0.5 + 0.5 * conf) * zoneBoost;
            const entry = currentRow.close;
            const { sl, tp } = atrStopTake(entry, currentRow.atr14 ?? 0, bias, {
              s: 1.2 - 0.2 * conf,
              t: 1.8 + 0.8 * conf + 0.4 * fallbackZone.weight,
            });
            suggestion = {
              side: bias,
              entry,
              stop: sl,
              take: tp,
              size,
              netEdge,
              horizon: "2h",
              details: { upP, downP, bucketKey, confluence: conf, gates, zone: fallbackZone },
            };
          }
        }
      }
    }

    return { forecasts, zones, bucketKey, agreement: votes, suggestion };
  }, [calc, enrichedRows, volMatrices, useVolConditioning, interval, takerBps, makerBps, fundingBpsPer8h]);

  // alignment shading timeline
  const alignSeries = useMemo(()=>{
    if(!calc) return [] as number[];
    const rows:CandleRow[] = calc.rows; const states:string[] = calc.states;
    const steps = horizons.map(h=> Math.round((h*60)/Number(interval)));
    const nearest = steps.length? steps[0]: 1;
    const vals = new Array(rows.length).fill(0) as number[];
    for(let t=250; t<rows.length-nearest; t++){
      const histStates = states.slice(0,t+1);
      const {probs} = buildMarkovWeighted(histStates, {
        window: windowN,
        smoothing: smooth,
        halfLife,
        order,
        dirichletStrength: dirichlet,
      });
      const cur = histStates[histStates.length-1] as StateKey;
      const prow = semiMarkovAdjustFirstRow(probs[IDX[cur]], histStates, { smoothing: 0.35 });
      const vec = multiStepForecast(probs, cur, {k: nearest}, prow)["k"];
      const up = vec[IDX['U']] > vec[IDX['D']];
      const kNow = rows[t].stochK ?? NaN;
      const pass = up ? (kNow<=stochLo) : (kNow>=stochHi);
      if(pass){ vals[t] = up? +1 : -1; }
    }
    return vals;
  }, [calc, horizons, interval, windowN, smooth, halfLife, order, dirichlet, stochLo, stochHi]);

// One-step forecast (current state row), with and without semi-Markov adjustment
const oneStep = useMemo(() => {
  if (!calc) return null as any;
  const cur = calc.curState as StateKey;
  const raw = (calc as any).probs[IDX[cur]].slice() as number[];
  const adj = semiMarkovAdjustFirstRow(raw, (calc as any).states as string[]);
  return { cur, raw, adj };
}, [calc]);


  // Trade suggestions text (pre-formatted)
  const tradeText = useMemo(()=>{
    if(!calc) return "";
    const last:CandleRow = calc.rows[calc.rows.length-1];
    const bias = calc.forecastBias;
    const biasText = `Bias → U ${(bias.bullish*100).toFixed(1)}% | D ${(bias.bearish*100).toFixed(1)}% | conf ${(bias.confidence*100).toFixed(0)}%`;
    const lines = ["Trade suggestions (EMA50 retest + StochRSI timing):", biasText];
    (["short","long"] as const).forEach((side)=>{
      const f = stochrsiFilterPass(last.stochK ?? NaN, side as any, stochMode, stochLo, stochHi, bias);
      lines.push(`- ${side.toUpperCase()} oscillator check: ${f.note}`);
      if(!f.ok){ lines.push("  → BLOCKED by StochRSI."); return; }
      const lv = suggestTradeLevels(last, calc.rows as CandleRow[], side as any, riskMode, rr, atrSL, atrTP, f.penalty, bias);
      const sz = positionSizeUSD(account, riskPct, lv.entry, lv.stop, leverage, takerBps, makerBps);
      lines.push(`  Entry: ${lv.entry.toFixed(6)}  Stop: ${lv.stop.toFixed(6)}  Target: ${lv.target.toFixed(6)}`);
      if(isFiniteNum(sz.notional)){ lines.push(`  Sizing: ~${sz.notional.toFixed(2)} USDT notional  (~${(sz.qty).toFixed(2)} coins)  @x${leverage}  fees≈${(2*takerBps).toFixed(1)} bps total`); }
    });
    return lines.join('\n');
  }, [calc, stochMode, stochLo, stochHi, riskMode, rr, atrSL, atrTP, account, riskPct, leverage, takerBps, makerBps]);

  const footerText = useMemo(()=>{
    const auto = refreshSel===0 ? 'OFF' : (String(refreshSel)+' min');
    let txt = 'Markov+RSI/StochRSI SPA • Auto refresh: '+auto+' • ' + (loading? 'Updating…':'Live');
    if(error){ txt += ' • Error: '+error; }
    return txt;
  }, [refreshSel, loading, error]);

  // ======= Responsive Top Controls: compact on mobile, grid on desktop =======
  const topPanel = (
    <div className="w-full bg-white/80 backdrop-blur sticky top-0 z-10 border-b shadow-sm">
      {/* row 1: primary controls, horizontally scrollable on mobile */}
      <div className="max-w-screen-2xl mx-auto px-3 sm:px-4 py-2 sm:py-3">
        <div className="flex gap-2 overflow-x-auto no-scrollbar pb-2 sm:pb-0 sm:flex-wrap items-end">
          <div className="min-w-[130px]"><div className="text-[10px] sm:text-xs text-gray-600">Symbol</div>
            <input className="border rounded px-2 py-1 w-full" value={symbol} onChange={e=>setSymbol(e.target.value.toUpperCase())}/>
          </div>
          <div className="min-w-[130px]"><div className="text-[10px] sm:text-xs text-gray-600">Interval (min)</div>
            <select className="border rounded px-2 py-1 w-full" value={interval} onChange={e=>setIntervalStr(e.target.value)}>
              {['1','3','5','15','30','60','240'].map(v=> <option key={v} value={v}>{v}</option>)}
            </select>
          </div>
          <div className="min-w-[120px]"><div className="text-[10px] sm:text-xs text-gray-600">Window</div>
            <input type="number" className="border rounded px-2 py-1 w-full" value={windowN} onChange={e=>setWindowN(Number(e.target.value))}/>
          </div>
          <div className="min-w-[120px]"><div className="text-[10px] sm:text-xs text-gray-600">Half-life</div>
            <input type="number" className="border rounded px-2 py-1 w-full" value={halfLife} onChange={e=>setHalfLife(Number(e.target.value))}/>
          </div>
          <div className="min-w-[110px]"><div className="text-[10px] sm:text-xs text-gray-600">Smoothing</div>
            <input type="number" step="0.1" className="border rounded px-2 py-1 w-full" value={smooth} onChange={e=>setSmooth(Number(e.target.value))}/>
          </div>
          <div className="min-w-[100px]"><div className="text-[10px] sm:text-xs text-gray-600">Order</div>
            <input type="number" className="border rounded px-2 py-1 w-full" min={1} step={1} value={order} onChange={e=>setOrder(Math.max(1, Number(e.target.value)))} />
          </div>
          <div className="min-w-[120px]"><div className="text-[10px] sm:text-xs text-gray-600">Dirichlet</div>
            <input type="number" step="0.5" className="border rounded px-2 py-1 w-full" value={dirichlet} onChange={e=>setDirichlet(Number(e.target.value))}/>
          </div>
          <div className="min-w-[140px]"><div className="text-[10px] sm:text-xs text-gray-600">Half-life mode</div>
            <select className="border rounded px-2 py-1 w-full" value={autoHalfLife ? "auto" : "manual"} onChange={e=>setAutoHalfLife(e.target.value === "auto")}>
              <option value="auto">Auto</option>
              <option value="manual">Manual</option>
            </select>
          </div>
          <div className="min-w-[130px]"><div className="text-[10px] sm:text-xs text-gray-600">Confidence gate</div>
            <input type="number" step="0.05" min={0} max={1} className="border rounded px-2 py-1 w-full" value={confidenceGate} onChange={e=>setConfidenceGate(Math.min(1, Math.max(0, Number(e.target.value))))}/>
          </div>
          <div className="min-w-[150px]"><div className="text-[10px] sm:text-xs text-gray-600">Horizons (h)</div>
            <select multiple className="border rounded px-2 py-1 w-full" value={horizons.map(String)} onChange={e=>{
              const vals=Array.from(e.target.selectedOptions).map(o=>Number(o.value)); setHorizons(vals);
            }}>
              {[1,2,4,6,12,24].map(h=> <option key={h} value={String(h)}>{h}</option>)}
            </select>
          </div>
          <div className="min-w-[140px]"><div className="text-[10px] sm:text-xs text-gray-600">Auto Refresh</div>
            <select className="border rounded px-2 py-1 w-full" value={String(refreshSel)} onChange={e=>setRefreshSel(Number(e.target.value))}>
              {[1,3,5,10,0].map(m=> <option key={m} value={String(m)}>{m===0? 'Off': (m + ' min')}</option>)}
            </select>
          </div>
          <div className="ml-auto flex items-end">
            <button onClick={refetch} className="bg-black text-white rounded px-3 py-1 whitespace-nowrap">Refresh</button>
          </div>
        </div>
      </div>

      {/* row 2: collapsible advanced trade config on small screens */}
      <details className="sm:hidden border-t px-3 py-2">
        <summary className="text-sm font-medium cursor-pointer">Trade config</summary>
        <div className="grid grid-cols-2 gap-2 pt-2">
          <SelectL label="Risk mode" value={riskMode} onChange={v=>setRiskMode(v as any)} options={['atr','conservative','aggressive']} />
          <NumL label="RR" value={rr} set={setRR} step={0.1} />
          <NumL label="ATR SL" value={atrSL} set={setAtrSL} step={0.1} />
          <NumL label="ATR TP" value={atrTP} set={setAtrTP} step={0.1} />
          <SelectL label="StochRSI mode" value={stochMode} onChange={v=>setStochMode(v as any)} options={['hard','soft','off']} />
          <NumL label="StochRSI ≤ (long)" value={stochLo} set={setStochLo} />
          <NumL label="StochRSI ≥ (short)" value={stochHi} set={setStochHi} />
          <NumL label="Account" value={account} set={setAccount} />
          <NumL label="Risk %" value={riskPct} set={setRiskPct} />
          <div className="col-span-2 grid grid-cols-3 gap-2">
            <NumL label="Lev" value={leverage} set={setLeverage} />
            <NumL label="Taker bps" value={takerBps} set={setTakerBps} />
            <NumL label="Maker bps" value={makerBps} set={setMakerBps} />
          </div>
          <NumL label="Funding bps (8h)" value={fundingBpsPer8h} set={setFundingBpsPer8h} step={0.5} />
          <NumL label="Order" value={order} set={(value)=>setOrder(Math.max(1, value))} />
          <NumL label="Dirichlet" value={dirichlet} set={setDirichlet} step={0.5} />
          <SelectL label="Half-life mode" value={autoHalfLife ? 'auto' : 'manual'} onChange={v=>setAutoHalfLife(v === 'auto')} options={['auto','manual']} />
          <NumL label="Conf gate" value={confidenceGate} set={(value)=>setConfidenceGate(Math.min(1, Math.max(0, value)))} step={0.05} />
        </div>
      </details>

      {/* row 2 desktop: full grid */}
      <div className="hidden sm:block border-t">
        <div className="max-w-screen-2xl mx-auto px-4 py-2 grid grid-cols-4 lg:grid-cols-8 gap-3">
          <SelectL label="Risk mode" value={riskMode} onChange={v=>setRiskMode(v as any)} options={['atr','conservative','aggressive']} />
          <NumL label="RR" value={rr} set={setRR} step={0.1} />
          <NumL label="ATR SL" value={atrSL} set={setAtrSL} step={0.1} />
          <NumL label="ATR TP" value={atrTP} set={setAtrTP} step={0.1} />
          <SelectL label="StochRSI mode" value={stochMode} onChange={v=>setStochMode(v as any)} options={['hard','soft','off']} />
          <NumL label="StochRSI ≤ (long)" value={stochLo} set={setStochLo} />
          <NumL label="StochRSI ≥ (short)" value={stochHi} set={setStochHi} />
          <div className="grid grid-cols-3 gap-2 col-span-4 lg:col-span-2">
            <NumL label="Account" value={account} set={setAccount} />
            <NumL label="Risk %" value={riskPct} set={setRiskPct} />
            <NumL label="Lev" value={leverage} set={setLeverage} />
          </div>
          <div className="grid grid-cols-3 gap-2 col-span-4 lg:col-span-2">
            <NumL label="Taker bps" value={takerBps} set={setTakerBps} />
            <NumL label="Maker bps" value={makerBps} set={setMakerBps} />
            <NumL label="Funding bps (8h)" value={fundingBpsPer8h} set={setFundingBpsPer8h} step={0.5} />
          </div>
          <NumL label="Order" value={order} set={(value)=>setOrder(Math.max(1, value))} />
          <NumL label="Dirichlet" value={dirichlet} set={setDirichlet} step={0.5} />
          <SelectL label="Half-life mode" value={autoHalfLife ? 'auto' : 'manual'} onChange={v=>setAutoHalfLife(v === 'auto')} options={['auto','manual']} />
          <NumL label="Conf gate" value={confidenceGate} set={(value)=>setConfidenceGate(Math.min(1, Math.max(0, value)))} step={0.05} />
        </div>
      </div>
    
    </div>

  );

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      {topPanel}
      {/* Main grid becomes 1col on mobile, 2col on lg, thirds on 2xl */}
     <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
  {/* Heatmap */}
  <SectionCard title="Transition Probabilities">
    {!calc ? (
      <div className="text-sm text-gray-500">{loading ? "Loading…" : "Not enough data"}</div>
    ) : (
      <div className="grid grid-cols-5 gap-1 text-center">
        <div></div>
        {STATES.map((s) => (
          <div key={s} className="text-xs font-medium text-gray-500">
            {s}
          </div>
        ))}
        {STATES.map((r, ri) => (
          <React.Fragment key={r}>
            <div className="text-xs font-medium text-gray-500 self-center">{r}</div>
            {STATES.map((c, ci) => {
              const v = (calc as any).probs[ri][ci];
              const hue = 220;
              const light = 100 - Math.round(v * 80);
              return (
                <div
                  key={c}
                  className="rounded-md p-2 text-sm"
                  style={{ background: `hsl(${hue},70%,${light}%)` }}
                >
                  {v.toFixed(2)}
                </div>
              );
            })}
          </React.Fragment>
        ))}
      </div>
    )}
  </SectionCard>

  {/* Forecast bars (multi-step horizons) */}
  <SectionCard title="Markov Forecast (horizons)">
    {!calc ? (
      <div className="text-sm text-gray-500">{loading ? "Loading…" : "No forecast"}</div>
    ) : (
      <div className="chart-300">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={STATES.map((s) => ({
              state: s,
              ...Object.fromEntries(
                Object.entries((calc as any).forecasts).map(([k, v]: any) => [k, v[IDX[s]]])
              ),
            }))}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="state" />
            <YAxis domain={[0, 1]} tickFormatter={(v: number) => (v * 100).toFixed(0) + "%"} />
            <Tooltip formatter={(v: any) => (Number(v) * 100).toFixed(1) + "%"} />
            <Legend />
            {Object.keys((calc as any).forecasts).map((k, i) => (
              <Bar key={k} dataKey={k} fill={`hsl(${(i * 60) % 360},70%,50%)`} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    )}
  </SectionCard>

  <SectionCard
    title="Volatility-conditioned outlook"
    right={
      <label className="inline-flex items-center gap-1">
        <input
          type="checkbox"
          className="accent-slate-600"
          checked={useVolConditioning}
          onChange={(e) => setUseVolConditioning(e.target.checked)}
        />
        <span className="text-xs">Vol conditioning</span>
      </label>
    }
  >
    {!calc || !volPanel ? (
      <div className="text-sm text-gray-500">{loading ? "Loading…" : "No bucketed forecast"}</div>
    ) : (
      <div className="space-y-4">
        <div className="text-xs text-gray-500">
          Bucket: <b>{volPanel.bucketKey}</b> · Agreement ↑<b>{volPanel.agreement.upVotes}</b> / ↓<b>{volPanel.agreement.downVotes}</b>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {Object.entries(volPanel.forecasts).map(([label, forecast]) => {
            const zone = volPanel.zones[label] ?? zoneFromProbs(forecast.probs);
            return (
              <div key={label} className="border rounded-lg p-2 bg-slate-50">
                <div className="text-xs uppercase tracking-wide text-gray-500">{label}</div>
                <div className="text-sm mt-1">
                  Bias: <b>{forecast.bias}</b>
                </div>
                <div className="text-xs text-gray-500">Entropy: {forecast.entropy.toFixed(3)}</div>
                <div className="mt-2 h-[80px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={STATES.map((state) => ({ state, p: forecast.probs[IDX[state]] }))}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="state" />
                      <YAxis domain={[0, 1]} hide />
                      <Tooltip formatter={(v: any) => (Number(v) * 100).toFixed(1) + "%"} />
                      <Bar dataKey="p" fill="#0ea5e9" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-3">
                  <div className="flex h-3 w-full overflow-hidden rounded">
                    {STATES.map((state) => {
                      const prob = forecast.probs[IDX[state]];
                      return (
                        <div
                          key={state}
                          style={{
                            width: `${(prob * 100).toFixed(2)}%`,
                            background: confidenceBadge.colorByState[state],
                          }}
                          title={`${state} ${(prob * 100).toFixed(1)}%`}
                        />
                      );
                    })}
                  </div>
                  <div className="mt-1 text-xs flex items-center gap-2 text-gray-600">
                    {confidenceBadge.render(zone)}
                    <span className="text-[11px]">
                      maxP {(zone.maxProb * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
        <div className="border rounded-lg p-3 bg-white shadow-inner">
          {volPanel.suggestion && volPanel.suggestion.side ? (
            <div className="grid gap-1 text-sm">
              <div>
                <span className="text-xs uppercase text-gray-500 mr-1">Side</span>
                <b className={volPanel.suggestion.side === "long" ? "text-green-600" : "text-red-600"}>
                  {volPanel.suggestion.side.toUpperCase()}
                </b>
                <span className="ml-2 text-xs text-gray-500">
                  {volPanel.suggestion.horizon} • bucket {volPanel.suggestion.details.bucketKey}
                </span>
              </div>
              <div>
                Entry: <b>{volPanel.suggestion.entry.toFixed(6)}</b>
              </div>
              <div>
                Stop: <b>{volPanel.suggestion.stop.toFixed(6)}</b>
              </div>
              <div>
                Take: <b>{volPanel.suggestion.take.toFixed(6)}</b>
              </div>
              <div>
                Size (rel): <b>{volPanel.suggestion.size.toFixed(2)}</b>
              </div>
              <div>
                Edge after costs: <b>{(volPanel.suggestion.netEdge * 100).toFixed(2)}%</b>
              </div>
              <div className="text-xs text-gray-500">
                P(U): {(volPanel.suggestion.details.upP * 100).toFixed(1)}% · P(D): {(volPanel.suggestion.details.downP * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-gray-500 flex items-center gap-2">
                {confidenceBadge.render(volPanel.suggestion.details.zone)}
                <span>
                  Zone <b>{volPanel.suggestion.details.zone.level}</b> — top {volPanel.suggestion.details.zone.topState} {(volPanel.suggestion.details.zone.maxProb * 100).toFixed(0)}%
                </span>
              </div>
              <div className="text-xs text-gray-500">
                Confluence {(volPanel.suggestion.details.confluence * 100).toFixed(0)}% — gates:
                <span className="ml-1">
                  stoch {volPanel.suggestion.details.gates.stoch ? "✓" : "×"}, emaRetest {volPanel.suggestion.details.gates.emaRetest ? "✓" : "×"},
                  emaTrend {volPanel.suggestion.details.gates.emaTrend ? "✓" : "×"}, ma200 {volPanel.suggestion.details.gates.ma200Bias ? "✓" : "×"}, rsi {volPanel.suggestion.details.gates.rsi ? "✓" : "×"}
                </span>
              </div>
            </div>
          ) : (
            <div className="text-sm text-gray-500">
              {volPanel.suggestion?.reason ?? "No trade suggestion"}
            </div>
          )}
        </div>
      </div>
    )}
  </SectionCard>

  {/* Diagnostics */}
  <SectionCard
    title="Model diagnostics"
    right={calc && calc.metrics ? <span className="subtle-text">samples: <b>{calc.metrics.count}</b></span> : undefined}
  >
    {!calc ? (
      <div className="text-sm text-gray-500">{loading ? "Loading…" : "No diagnostics"}</div>
    ) : (
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-3 text-sm">
        <div>
          Half-life used: <b>{calc.model.halfLifeUsed}</b> {calc.model.halfLifeUsed !== halfLife ? "(auto)" : ""}
        </div>
        <div>
          Order/context: <b>{calc.model.order}</b> / {calc.model.contextDepth}
        </div>
        <div>
          Dirichlet strength: <b>{calc.model.dirichletStrength.toFixed(2)}</b>
        </div>
        <div>
          Log-likelihood: <b>{isFiniteNum(calc.metrics.logLikelihood) ? calc.metrics.logLikelihood.toFixed(4) : "n/a"}</b>
        </div>
        <div>
          Brier: <b>{isFiniteNum(calc.metrics.brier) ? calc.metrics.brier.toFixed(4) : "n/a"}</b>
        </div>
        <div>
          Accuracy: <b>{isFiniteNum(calc.metrics.accuracy) ? (calc.metrics.accuracy * 100).toFixed(1) + "%" : "n/a"}</b>
        </div>
        <div>
          Current regime: <b>{calc.curState}</b>
        </div>
        <div>
          Rule vs learned: <b>{calc.ruleStates?.[calc.rows.length-1]}</b> → <b>{calc.learnedStates?.[calc.rows.length-1] ?? "-"}</b>
        </div>
        <div>
          Learned confidence: <b>{isFiniteNum(calc.rows[calc.rows.length-1]?.learnedConfidence) ? (calc.rows[calc.rows.length-1].learnedConfidence! * 100).toFixed(0) + "%" : "n/a"}</b>
        </div>
        <div>
          Quantum expected move: <b>{calc.quantum.expectedMove.toFixed(3)}</b>
        </div>
      </div>
    )}
  </SectionCard>

  {/* NEW: Current state → next-step probabilities (raw vs semi-Markov adjusted) */}
  <SectionCard
    title="Current state → next-step probabilities"
    right={oneStep && <span className="subtle-text">state: <b>{oneStep.cur}</b></span>}
  >
    {!calc || !oneStep ? (
      <div className="text-sm text-gray-500">{loading ? "Loading…" : "No data"}</div>
    ) : (
      <div className="h-[200px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={STATES.map((s) => ({
              state: s,
              raw: oneStep.raw[IDX[s]],
              adjusted: oneStep.adj[IDX[s]],
            }))}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="state" />
            <YAxis domain={[0, 1]} tickFormatter={(v: number) => (v * 100).toFixed(0) + "%"} />
            <Tooltip formatter={(v: any) => (Number(v) * 100).toFixed(1) + "%"} />
            <Legend />
            <Bar dataKey="raw" fill="#94a3b8" name="raw" />
            <Bar dataKey="adjusted" fill="#0ea5e9" name="semi-Markov" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    )}
  </SectionCard>

  {/* Trade Suggestions */}
  <SectionCard title="Trade suggestions (EMA50 retest + StochRSI timing)">
    {!calc ? (
      <div className="text-sm text-gray-500">{loading ? "Loading…" : "No data"}</div>
    ) : (
      <pre className="whitespace-pre-wrap text-[12px] leading-6 font-mono">{tradeText}</pre>
    )}
  </SectionCard>

  {/* RSI & StochRSI — now moved to the bottom; full width on xl */}
  <SectionCard title="RSI & StochRSI — shaded alignment (Markov + StochRSI)" className="xl:col-span-2">
    {!calc ? (
      <div className="text-sm text-gray-500">{loading ? "Loading…" : "No data"}</div>
    ) : (
      <div className="chart-380">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={(calc as any).rows.map((r: any, i: number) => ({
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

