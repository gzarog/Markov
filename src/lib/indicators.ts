import { isFiniteNum } from "./utils";

export function rollingMean(arr: number[], n: number) {
  const out = new Array(arr.length).fill(NaN) as number[];
  let sum = 0;
  const queue: number[] = [];
  for (let i = 0; i < arr.length; i++) {
    const value = arr[i];
    if (!isFiniteNum(value)) {
      queue.push(0);
    } else {
      queue.push(value);
      sum += value;
    }
    if (queue.length > n) {
      const removed = queue.shift()!;
      sum -= removed;
    }
    if (queue.length === n) {
      out[i] = sum / n;
    }
  }
  return out;
}

export function ema(arr: number[], span: number) {
  if (!arr.length) return [] as number[];
  const k = 2 / (span + 1);
  let prev = arr[0];
  const out = [prev];
  for (let i = 1; i < arr.length; i++) {
    const value = arr[i];
    prev = (isFiniteNum(value) ? value : prev) * k + prev * (1 - k);
    out.push(prev);
  }
  return out;
}

export function movingAverage(arr: number[], n: number) {
  return rollingMean(arr, n);
}

export function atr14(high: number[], low: number[], close: number[]) {
  const trueRange: number[] = [];
  for (let i = 0; i < high.length; i++) {
    if (i === 0) {
      trueRange.push(high[i] - low[i]);
      continue;
    }
    const prevClose = close[i - 1];
    trueRange.push(
      Math.max(
        high[i] - low[i],
        Math.abs(high[i] - prevClose),
        Math.abs(low[i] - prevClose)
      )
    );
  }
  return rollingMean(trueRange, 14);
}

export function rsi14(close: number[]) {
  const delta = close.map((c, i) => (i === 0 ? 0 : c - close[i - 1]));
  const gains = delta.map((d) => Math.max(0, d));
  const losses = delta.map((d) => Math.max(0, -d));
  const avgGain = rollingMean(gains, 14);
  const avgLoss = rollingMean(losses, 14);
  const out: number[] = [];
  for (let i = 0; i < close.length; i++) {
    const g = avgGain[i];
    const l = avgLoss[i];
    if (!isFiniteNum(g) || !isFiniteNum(l) || l === 0) {
      out.push(NaN);
    } else {
      const rs = g / l;
      out.push(100 - 100 / (1 + rs));
    }
  }
  return out;
}

export function stochRsi(rsi: number[], lengthStoch = 14, k = 3, d = 3) {
  const minArr: number[] = [];
  const maxArr: number[] = [];
  for (let i = 0; i < rsi.length; i++) {
    const start = Math.max(0, i - lengthStoch + 1);
    const slice = rsi.slice(start, i + 1).filter((x) => isFiniteNum(x));
    const min = slice.length ? Math.min(...slice) : NaN;
    const max = slice.length ? Math.max(...slice) : NaN;
    minArr.push(min);
    maxArr.push(max);
  }
  const raw = rsi.map((value, i) =>
    !isFiniteNum(value) || !isFiniteNum(minArr[i]) || !isFiniteNum(maxArr[i]) || maxArr[i] === minArr[i]
      ? NaN
      : (value - minArr[i]) / (maxArr[i] - minArr[i])
  );
  const kLine = rollingMean(raw.map((value) => value * 100), k);
  const dLine = rollingMean(kLine, d);
  return { k: kLine, d: dLine };
}

export function rollingStd(arr: number[], n: number) {
  const out = new Array(arr.length).fill(NaN) as number[];
  for (let i = 0; i < arr.length; i++) {
    if (i + 1 < n) continue;
    let sum = 0;
    let count = 0;
    for (let k = i - n + 1; k <= i; k++) {
      const value = arr[k];
      if (!isFiniteNum(value)) continue;
      sum += value;
      count++;
    }
    if (count !== n) continue;
    const mean = sum / count;
    let variance = 0;
    for (let k = i - n + 1; k <= i; k++) {
      const value = arr[k];
      variance += (value - mean) ** 2;
    }
    out[i] = Math.sqrt(variance / n);
  }
  return out;
}

export function macd(close: number[], fast = 12, slow = 26, signal = 9) {
  if (!close.length) {
    return { macd: [], signal: [], hist: [] };
  }
  const fastEma = ema(close, fast);
  const slowEma = ema(close, slow);
  const macdLine = close.map((_, i) => (fastEma[i] ?? 0) - (slowEma[i] ?? 0));
  const signalLine = ema(macdLine, signal);
  const hist = macdLine.map((value, i) => value - (signalLine[i] ?? 0));
  return { macd: macdLine, signal: signalLine, hist };
}

export function bollingerBands(close: number[], period = 20, mult = 2) {
  const basis = movingAverage(close, period);
  const std = rollingStd(close, period);
  const upper = close.map((_, i) => {
    if (!isFiniteNum(basis[i]) || !isFiniteNum(std[i])) return NaN;
    return (basis[i] ?? 0) + mult * (std[i] ?? 0);
  });
  const lower = close.map((_, i) => {
    if (!isFiniteNum(basis[i]) || !isFiniteNum(std[i])) return NaN;
    return (basis[i] ?? 0) - mult * (std[i] ?? 0);
  });
  const width = upper.map((value, i) => {
    if (!isFiniteNum(value) || !isFiniteNum(lower[i])) return NaN;
    return value - (lower[i] ?? 0);
  });
  return { upper, lower, basis, width };
}

export function cumulativeVwap(typical: number[], volume: number[]) {
  const out = new Array(typical.length).fill(NaN) as number[];
  let pvSum = 0;
  let volSum = 0;
  for (let i = 0; i < typical.length; i++) {
    const price = typical[i];
    const vol = volume[i];
    if (!isFiniteNum(price) || !isFiniteNum(vol) || vol <= 0) {
      out[i] = i > 0 ? out[i - 1] : NaN;
      continue;
    }
    pvSum += price * vol;
    volSum += vol;
    out[i] = volSum > 0 ? pvSum / volSum : NaN;
  }
  return out;
}
