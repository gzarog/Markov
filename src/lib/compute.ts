import { CandleRow, IDX, StateKey } from "../types/market";
import { atr14, ema, movingAverage, rsi14, stochRsi } from "./indicators";
import { buildMarkovWeighted, multiStepForecast, semiMarkovAdjustFirstRow } from "./markov";
import { labelState } from "./stateClassifier";

export function computeAll(
  candles: CandleRow[],
  cfg: { window: number; halfLife: number; smooth: number; horizons: number[]; intervalMinutes: number }
) {
  const close = candles.map((c) => c.close);
  const high = candles.map((c) => c.high);
  const low = candles.map((c) => c.low);
  const ema10 = ema(close, 10);
  const ema50 = ema(close, 50);
  const ma200 = movingAverage(close, 200);
  const ma200_slope = ma200.map((value, index) => (index ? value - (ma200[index - 1] ?? value) : NaN));
  const atr = atr14(high, low, close);
  const rsi = rsi14(close);
  const st = stochRsi(rsi, 14, 3, 3);

  const rows = candles.map((c, i) => ({
    time: c.time,
    open: c.open,
    high: c.high,
    low: c.low,
    close: c.close,
    volume: c.volume,
    ema10: ema10[i],
    ema50: ema50[i],
    ma200: ma200[i],
    ma200_slope: ma200_slope[i],
    atr14: atr[i],
    rsi: rsi[i],
    stochK: st.k[i],
    stochD: st.d[i],
  })) as CandleRow[];

  const states = rows.map((r) => labelState(r));
  const { counts, probs } = buildMarkovWeighted(states.filter(Boolean), cfg.window, cfg.smooth, cfg.halfLife);
  const curState = states[states.length - 1] as StateKey;
  const firstAdj = semiMarkovAdjustFirstRow(probs[IDX[curState]], states.filter(Boolean));
  const steps = Object.fromEntries(cfg.horizons.map((h) => [h + "h", Math.round((h * 60) / cfg.intervalMinutes)]));
  const forecasts = multiStepForecast(probs, curState, steps, firstAdj);

  return { rows, states, probs, counts, curState, forecasts };
}
