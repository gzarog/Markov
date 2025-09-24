import { CandleRow, IDX, StateKey } from "../types/market";
import { atr14, ema, movingAverage, rsi14, stochRsi } from "./indicators";
import {
  buildMarkovWeighted,
  buildOrder2Counts,
  computeRunLength,
  estimateDurations,
  multiStepForecast,
  rowFromOrder2,
  semiMarkovAdjustFirstRow,
  PairKey,
} from "./markov";
import { buildFeatures, conditionedRow, blendRows } from "./models/conditioning";
import { LOGIT_MODEL } from "./models/modelConfig";
import { stepsForHorizon } from "./models/horizons";
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

  const states = rows.map((r) => labelState(r)) as StateKey[];
  const { counts, probs } = buildMarkovWeighted(states, cfg.window, cfg.smooth, cfg.halfLife);
  const curState = states[states.length - 1];
  const durations = estimateDurations(states);
  const runLength = computeRunLength(states);

  const rowBase = probs[IDX[curState]].slice();
  const rowDuration = semiMarkovAdjustFirstRow(rowBase, states, { durations, runLength });

  const order2Counts = buildOrder2Counts(states);
  const lastPair: PairKey | null = states.length >= 2 ? `${states[states.length - 2]}${curState}` as PairKey : null;
  const rowOrder2 = lastPair ? rowFromOrder2(order2Counts, lastPair) : null;

  const features = rows.length > 20 ? buildFeatures(rows, rows.length - 1) : null;
  const rowConditioned = features ? conditionedRow(features, LOGIT_MODEL) : null;

  const components: number[][] = [rowDuration];
  const weights: number[] = [0.6];
  if (rowOrder2) {
    components.push(rowOrder2);
    weights.push(0.25);
  }
  if (rowConditioned) {
    components.push(rowConditioned);
    weights.push(rowOrder2 ? 0.15 : 0.4);
  }
  const rowBlended = blendRows(components, weights);

  const steps = Object.fromEntries(
    cfg.horizons.map((h) => [h + "h", stepsForHorizon(h, cfg.intervalMinutes, rows)])
  );
  const forecasts = multiStepForecast(probs, curState, steps, rowBlended);

  return {
    rows,
    states,
    probs,
    counts,
    curState,
    forecasts,
    rowBase,
    rowDuration,
    rowOrder2,
    rowConditioned,
    rowBlended,
    durations,
    runLength,
    steps,
  };
}
