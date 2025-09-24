import { atr14, ema, movingAverage, rsi14, stochRsi } from "./indicators";
import type { CandleRow } from "./types";

function rollingStats(values: number[], window: number) {
  const means: number[] = [];
  const stds: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const start = Math.max(0, i - window + 1);
    const slice = values.slice(start, i + 1);
    const len = slice.length || 1;
    const mean = slice.reduce((acc, value) => acc + value, 0) / len;
    const variance = slice.reduce((acc, value) => acc + Math.pow(value - mean, 2), 0) / len;
    const std = Math.sqrt(Math.max(variance, 0));
    means.push(mean);
    stds.push(std);
  }
  return { means, stds };
}

export function computeIndicators(rows: CandleRow[]): CandleRow[] {
  if (!rows.length) return [];
  const closes = rows.map((row) => row.close);
  const highs = rows.map((row) => row.high);
  const lows = rows.map((row) => row.low);

  const ema10 = ema(closes, 10);
  const ema50 = ema(closes, 50);
  const ma200 = movingAverage(closes, 200);
  const atr = atr14(highs, lows, closes);
  const rsiSeries = rsi14(closes);
  const { k: stochK, d: stochD } = stochRsi(rsiSeries);

  const logReturns = closes.map((close, index) => (index === 0 ? 0 : Math.log(close / closes[index - 1])));
  const { means: retMeans, stds: retStd } = rollingStats(logReturns, 20);

  const volSeries = retStd
    .filter((value) => Number.isFinite(value))
    .slice(-500)
    .sort((a, b) => a - b);
  const q1 = volSeries[Math.floor(volSeries.length * 0.33)] ?? 0;
  const q2 = volSeries[Math.floor(volSeries.length * 0.66)] ?? 0;

  return rows.map((row, index) => {
    const rv = retStd[index];
    const zret = rv > 0 ? (logReturns[index] - retMeans[index]) / rv : 0;
    const time = row.time instanceof Date ? row.time : new Date(row.time);
    const hour = Number.isFinite(time.getUTCHours()) ? time.getUTCHours() : 0;
    const session: "ASIA" | "EU" | "US" = hour < 7 ? "ASIA" : hour < 15 ? "EU" : "US";
    const bucket: "low" | "mid" | "high" = rv <= q1 ? "low" : rv <= q2 ? "mid" : "high";
    const ma = ma200[index];
    const prevMa = index > 0 ? ma200[index - 1] : ma;
    const slope = Number.isFinite(ma) && Number.isFinite(prevMa) ? ma - prevMa : undefined;

    const rsiValue = Number.isFinite(rsiSeries[index]) ? rsiSeries[index] : row.rsi;

    return {
      ...row,
      ema10: ema10[index],
      ema50: ema50[index],
      ma200: ma,
      ma200_slope: slope,
      atr14: atr[index],
      rsi: rsiValue,
      rsi14: rsiValue,
      stochK: stochK[index],
      stochD: stochD[index],
      rv,
      zret,
      session,
      volBucket: bucket,
    };
  });
}

export function bucketVolSession(row: CandleRow) {
  const bucket = row.volBucket ?? "mid";
  const session = row.session ?? "EU";
  return `${bucket}|${session}`;
}
