import { CandleRow, IDX, StateKey, STATES } from "../types/market";
import { atr14, ema, movingAverage, rollingStd, rsi14, stochRsi } from "./indicators";
import {
  MarkovBuildResult,
  buildMarkovWeighted,
  expectedDirectionalMove,
  multiStepForecast,
  semiMarkovAdjustFirstRow,
  stateProbabilityVector,
} from "./markov";
import { inferRegimeStates } from "./regimeModel";
import { labelState } from "./stateClassifier";
import { clamp, isFiniteNum } from "./utils";

type ComputeConfig = {
  window: number;
  halfLife: number;
  smooth: number;
  horizons: number[];
  intervalMinutes: number;
  order?: number;
  dirichlet?: number;
  autoHalfLife?: boolean;
  confidenceGate?: number;
};

type ForecastBias = {
  bullish: number;
  bearish: number;
  base: number;
  reversal: number;
  confidence: number;
};

type TransitionMetrics = {
  logLikelihood: number;
  brier: number;
  accuracy: number;
  count: number;
};

const STATE_COUNT = STATES.length;

function computeReturn(close: number[], lookback: number) {
  const out = new Array(close.length).fill(NaN) as number[];
  if (lookback <= 0) return out;
  for (let i = lookback; i < close.length; i++) {
    if (close[i - lookback] !== 0) {
      out[i] = close[i] / close[i - lookback] - 1;
    }
  }
  return out;
}

function computeLogReturns(close: number[]) {
  return close.map((value, index) => (index === 0 ? NaN : Math.log(value / close[index - 1])));
}

function deriveForecastBias(vector?: number[]): ForecastBias {
  if (!vector) {
    return { bullish: 0.25, bearish: 0.25, base: 0.25, reversal: 0.25, confidence: 0 };
  }
  const bull = vector[IDX["U"]] ?? 0;
  const bear = vector[IDX["D"]] ?? 0;
  const base = vector[IDX["B"]] ?? 0;
  const rev = vector[IDX["R"]] ?? 0;
  const diff = Math.abs(bull - bear);
  return {
    bullish: bull,
    bearish: bear,
    base,
    reversal: rev,
    confidence: clamp(diff, 0, 1),
  };
}

function scoreTransitionModel(states: StateKey[], probs: number[][]): TransitionMetrics {
  let logLikelihood = 0;
  let brier = 0;
  let accuracy = 0;
  let count = 0;
  for (let i = 0; i < states.length - 1; i++) {
    const from = IDX[states[i]];
    const to = IDX[states[i + 1]];
    if (from === undefined || to === undefined) continue;
    const row = probs[from];
    const prob = Math.max(row[to], 1e-12);
    logLikelihood += Math.log(prob);
    let stepBrier = 0;
    let bestIdx = 0;
    for (let j = 0; j < STATE_COUNT; j++) {
      const actual = j === to ? 1 : 0;
      const diff = row[j] - actual;
      stepBrier += diff * diff;
      if (row[j] > row[bestIdx]) bestIdx = j;
    }
    if (bestIdx === to) accuracy++;
    brier += stepBrier;
    count++;
  }
  if (count === 0) return { logLikelihood: NaN, brier: NaN, accuracy: NaN, count };
  return {
    logLikelihood: logLikelihood / count,
    brier: brier / count,
    accuracy: accuracy / count,
    count,
  };
}

export function computeAll(
  candles: CandleRow[],
  cfg: ComputeConfig
) {
  const close = candles.map((c) => c.close);
  const high = candles.map((c) => c.high);
  const low = candles.map((c) => c.low);
  const volume = candles.map((c) => (isFiniteNum(c.volume) ? (c.volume as number) : NaN));
  const ema10 = ema(close, 10);
  const ema50 = ema(close, 50);
  const ma200 = movingAverage(close, 200);
  const ma200_slope = ma200.map((value, index) => (index ? value - (ma200[index - 1] ?? value) : NaN));
  const atr = atr14(high, low, close);
  const rsi = rsi14(close);
  const st = stochRsi(rsi, 14, 3, 3);
  const logReturns = computeLogReturns(close);
  const vol14 = rollingStd(logReturns, 14);
  const vol30 = rollingStd(logReturns, 30);
  const volMean = movingAverage(volume, 30);
  const volStd = rollingStd(volume, 30);
  const barsPerHour = Math.max(1, Math.round(60 / cfg.intervalMinutes));
  const ret1 = computeReturn(close, barsPerHour);
  const ret4 = computeReturn(close, Math.max(1, barsPerHour * 4));
  const ret12 = computeReturn(close, Math.max(1, barsPerHour * 12));
  const emaHTFFast = ema(close, Math.max(5 * barsPerHour, 10));
  const emaHTFSlow = ema(close, Math.max(12 * barsPerHour, 20));

  const rows = candles.map((c, i) => {
    const price = c.close;
    const atrNorm = isFiniteNum(atr[i]) && price !== 0 ? atr[i] / price : NaN;
    const emaTrendFast = isFiniteNum(ema10[i]) && isFiniteNum(ema50[i]) && price !== 0 ? (ema10[i] - ema50[i]) / price : NaN;
    const emaTrendSlow = isFiniteNum(ema50[i]) && isFiniteNum(ma200[i]) && price !== 0 ? (ema50[i] - ma200[i]) / price : NaN;
    const emaTrendHtf = isFiniteNum(emaHTFFast[i]) && isFiniteNum(emaHTFSlow[i]) && price !== 0 ? (emaHTFFast[i] - emaHTFSlow[i]) / price : NaN;
    let htfState: StateKey | undefined;
    if (isFiniteNum(emaTrendHtf)) {
      if (emaTrendHtf > 0.0015) htfState = "U";
      else if (emaTrendHtf < -0.0015) htfState = "D";
      else if (emaTrendHtf >= 0) htfState = "R";
      else htfState = "B";
    }
    const volZ =
      isFiniteNum(volume[i]) && isFiniteNum(volMean[i]) && isFiniteNum(volStd[i]) && volStd[i] !== 0
        ? (volume[i]! - volMean[i]) / volStd[i]
        : NaN;
    return {
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
      rsi14: rsi[i],
      stochK: st.k[i],
      stochD: st.d[i],
      logRet: logReturns[i],
      realizedVol14: vol14[i],
      realizedVol30: vol30[i],
      ret1: ret1[i],
      ret4: ret4[i],
      ret12: ret12[i],
      emaTrendFast,
      emaTrendSlow,
      emaTrendHtf,
      atrNorm,
      volumeNorm: volZ,
      htfState,
    } as CandleRow;
  });

  const ruleStates = rows.map((row) => labelState(row));
  const regime = inferRegimeStates(rows, ruleStates, {
    confidenceGate: cfg.confidenceGate ?? 0.45,
    smoothPasses: 2,
  });
  const states = regime.states;
  rows.forEach((row, index) => {
    row.ruleState = ruleStates[index];
    row.learnedState = regime.learnedStates[index] ?? undefined;
    row.learnedConfidence = regime.confidence[index];
    row.state = states[index];
  });

  const window = cfg.window;
  const smoothing = cfg.smooth;
  const dirichlet = cfg.dirichlet ?? 0;
  const order = cfg.order ?? 1;

  const builder = (halfLife: number) =>
    buildMarkovWeighted(states, {
      window,
      smoothing,
      halfLife,
      order,
      dirichletStrength: dirichlet,
    });

  const halfLifeCandidates = cfg.autoHalfLife
    ? Array.from(
        new Set(
          [cfg.halfLife * 0.5, cfg.halfLife, cfg.halfLife * 1.5, cfg.halfLife * 2]
            .map((value) => Math.max(10, Math.round(value)))
            .filter((value) => Number.isFinite(value))
        )
      )
    : [cfg.halfLife];

  let bestBuild: MarkovBuildResult | null = null;
  let bestMetrics: TransitionMetrics | null = null;
  let bestHalfLife = cfg.halfLife;

  halfLifeCandidates.forEach((halfLife) => {
    const result = builder(halfLife);
    const metrics = scoreTransitionModel(states as StateKey[], result.probs);
    if (!bestMetrics || (isFiniteNum(metrics.logLikelihood) && metrics.logLikelihood > (bestMetrics.logLikelihood ?? -Infinity))) {
      bestBuild = result;
      bestMetrics = metrics;
      bestHalfLife = halfLife;
    }
  });

  const build = bestBuild ?? builder(cfg.halfLife);
  const metrics = bestMetrics ?? scoreTransitionModel(states as StateKey[], build.probs);

  const curState = states[states.length - 1] as StateKey;
  const firstAdj = semiMarkovAdjustFirstRow(build.probs[IDX[curState]], states, {
    smoothing: 0.4,
    minSamples: 3,
  });
  const steps = Object.fromEntries(cfg.horizons.map((h) => [h + "h", Math.round((h * 60) / cfg.intervalMinutes)]));
  const forecasts = multiStepForecast(build.probs, curState, steps, firstAdj);

  const primaryHorizon = cfg.horizons[0] ?? 1;
  const primaryVector = forecasts[primaryHorizon + "h"];
  const bias = deriveForecastBias(primaryVector);
  const currentRow = firstAdj ?? build.probs[IDX[curState]];
  const quantumVector = Array.from(stateProbabilityVector(currentRow));
  const expectedMove = expectedDirectionalMove(currentRow);

  return {
    rows,
    states,
    ruleStates,
    learnedStates: regime.learnedStates,
    probs: build.probs,
    counts: build.counts,
    curState,
    forecasts,
    firstRowAdjusted: firstAdj,
    metrics,
    forecastBias: bias,
    model: {
      halfLifeUsed: bestHalfLife,
      order: build.options.order,
      dirichletStrength: build.options.dirichletStrength,
      contextDepth: build.contextDepth,
      baseDistribution: build.baseDistribution,
    },
    quantum: {
      stateVector: quantumVector,
      expectedMove,
    },
    regime,
  };
}
