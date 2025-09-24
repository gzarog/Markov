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
import { buildMarkovWeighted, multiStepForecast, semiMarkovAdjustFirstRow } from "./lib/markov";
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
          <div className="grid grid-cols-2 gap-2 col-span-4 lg:col-span-2">
            <NumL label="Taker bps" value={takerBps} set={setTakerBps} />
            <NumL label="Maker bps" value={makerBps} set={setMakerBps} />
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

