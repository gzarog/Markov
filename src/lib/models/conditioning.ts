import { CandleRow, STATES } from "../../types/market";

export type FeatureVector = number[];
export type LogitWeights = number[][];
export type LogitBias = number[];
export type LogitModel = {
  W: LogitWeights;
  b: LogitBias;
  temperature: number;
};

const FEATURE_DIM = 10;

function safe(value: number) {
  return Number.isFinite(value) ? value : 0;
}

function range(row: CandleRow) {
  return Math.max(1e-9, row.high - row.low);
}

function clampIndex<T>(arr: T[], index: number) {
  if (index < 0) return arr[0];
  if (index >= arr.length) return arr[arr.length - 1];
  return arr[index];
}

function sliceSafe<T>(arr: T[], start: number, end: number) {
  const s = Math.max(0, start);
  const e = Math.max(s + 1, end);
  return arr.slice(s, e);
}

export function defaultLogitModel(): LogitModel {
  return {
    W: Array.from({ length: STATES.length }, () => Array(FEATURE_DIM).fill(0)),
    b: Array(STATES.length).fill(0),
    temperature: 1.2,
  };
}

export function buildFeatures(rows: CandleRow[], index: number): FeatureVector {
  const take = (offset: number) => clampIndex(rows, index - offset);
  const current = take(0);
  const close = current.close;
  const prevClose = take(1).close;
  const ret1 = safe(close / Math.max(1e-9, prevClose) - 1);
  const ret5 = index >= 5 ? safe(close / Math.max(1e-9, take(5).close) - 1) : 0;

  const atr14 = current.atr14 ?? range(current);
  const volK = safe(atr14 / Math.max(1e-9, close));

  const rangeNow = range(current);
  const body = safe(Math.abs(close - current.open) / rangeNow);
  const wickUpper = safe((current.high - Math.max(current.open, close)) / rangeNow);
  const wickLower = safe((Math.min(current.open, close) - current.low) / rangeNow);

  const window20 = sliceSafe(rows, index - 19, index + 1);
  const highs20 = window20.map((row) => row.high);
  const lows20 = window20.map((row) => row.low);
  const hi20 = Math.max(...highs20);
  const lo20 = Math.min(...lows20);
  const donchPos = safe((close - lo20) / Math.max(1e-9, hi20 - lo20));

  const returnsWindow = window20
    .map((row, i, arr) => (i ? Math.log(row.close / Math.max(1e-9, arr[i - 1].close)) : 0))
    .slice(1);
  const mu = returnsWindow.length
    ? returnsWindow.reduce((acc, value) => acc + value, 0) / returnsWindow.length
    : 0;
  const std = returnsWindow.length > 1
    ? Math.sqrt(
        returnsWindow.reduce((acc, value) => acc + (value - mu) ** 2, 0) /
          Math.max(1, returnsWindow.length - 1)
      )
    : 0;

  const histWindow = sliceSafe(rows, index - 60, index + 1);
  const chunkStd: number[] = [];
  if (histWindow.length >= 20) {
    for (let start = 0; start + 20 <= histWindow.length; start += 20) {
      const chunk = histWindow.slice(start, start + 20);
      const chunkReturns = chunk
        .map((row, i, arr) => (i ? Math.log(row.close / Math.max(1e-9, arr[i - 1].close)) : 0))
        .slice(1);
      if (!chunkReturns.length) continue;
      const mean = chunkReturns.reduce((acc, value) => acc + value, 0) / chunkReturns.length;
      const sigma = Math.sqrt(
        chunkReturns.reduce((acc, value) => acc + (value - mean) ** 2, 0) /
          Math.max(1, chunkReturns.length - 1)
      );
      if (Number.isFinite(sigma)) chunkStd.push(sigma);
    }
  }
  chunkStd.sort((a, b) => a - b);
  const medianChunkStd = chunkStd.length
    ? chunkStd[Math.floor(chunkStd.length / 2)]
    : std;
  const volStdZ = safe((std - medianChunkStd) / Math.max(1e-9, Math.abs(medianChunkStd)));

  const ma200Now = current.ma200 ?? close;
  const back = take(10);
  const backValue = back.ma200 ?? back.close ?? close;
  const maSlope = safe((ma200Now - backValue) / 10);

  const trWindow = window20.map(range).sort((a, b) => a - b);
  const trMedian = trWindow[Math.floor(trWindow.length / 2)] ?? rangeNow;
  const trCompression = safe(rangeNow / Math.max(1e-9, trMedian));

  return [ret1, ret5, volK, body, wickUpper, wickLower, trCompression, donchPos, volStdZ, maSlope];
}

export function conditionedRow(features: FeatureVector, model: LogitModel) {
  const logits = STATES.map((_, j) => {
    const weights = model.W[j] ?? [];
    const dot = features.reduce((acc, value, f) => acc + value * (weights[f] ?? 0), 0);
    return (model.b[j] ?? 0) + dot;
  });
  const temperature = Math.max(1e-6, model.temperature ?? 1);
  const scaled = logits.map((value) => value / temperature);
  const maxLogit = Math.max(...scaled);
  const exps = scaled.map((value) => Math.exp(value - maxLogit));
  const Z = exps.reduce((acc, value) => acc + value, 0);
  return exps.map((value) => value / Z);
}

export function blendRows(rows: number[][], weights: number[]) {
  const total = Math.max(1e-9, weights.reduce((acc, value) => acc + value, 0));
  const normalized = weights.map((value) => value / total);
  const out = Array(STATES.length).fill(0);
  rows.forEach((row, idx) => {
    const w = normalized[idx] ?? 0;
    row.forEach((value, j) => {
      out[j] += w * value;
    });
  });
  const sum = out.reduce((acc, value) => acc + value, 0);
  return sum > 0 ? out.map((value) => value / sum) : Array(STATES.length).fill(1 / STATES.length);
}

export function blendTwoRows(base: number[], conditioned: number[], alpha: number) {
  return base.map((value, idx) => alpha * value + (1 - alpha) * (conditioned[idx] ?? 0));
}
