export const isFiniteNum = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);

export const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

export function sum(values: number[]) {
  return values.reduce((acc, value) => acc + (isFiniteNum(value) ? value : 0), 0);
}

export function mean(values: number[]) {
  const filtered = values.filter(isFiniteNum);
  if (!filtered.length) return NaN;
  return sum(filtered) / filtered.length;
}

export function variance(values: number[]) {
  const filtered = values.filter(isFiniteNum);
  if (filtered.length < 2) return NaN;
  const mu = mean(filtered);
  const sq = filtered.reduce((acc, value) => acc + Math.pow(value - mu, 2), 0);
  return sq / (filtered.length - 1);
}

export function stddev(values: number[]) {
  const v = variance(values);
  return isFiniteNum(v) ? Math.sqrt(Math.max(v, 0)) : NaN;
}

export function softmax(values: number[]) {
  if (!values.length) return values.slice();
  const max = Math.max(...values.map((value) => (isFiniteNum(value) ? value : -Infinity)));
  const exps = values.map((value) => (isFiniteNum(value) ? Math.exp(value - max) : 0));
  const total = sum(exps);
  if (total === 0) return Array(values.length).fill(1 / values.length);
  return exps.map((value) => value / total);
}

export function normalize(values: number[], total = 1) {
  const s = sum(values);
  if (!isFiniteNum(s) || s === 0) return Array(values.length).fill(total / values.length);
  return values.map((value) => (isFiniteNum(value) ? (value / s) * total : total / values.length));
}

export function zScore(value: number, mu: number, sigma: number) {
  if (!isFiniteNum(value) || !isFiniteNum(mu) || !isFiniteNum(sigma) || sigma === 0) return 0;
  return (value - mu) / sigma;
}

export function safeLog(value: number, eps = 1e-12) {
  return Math.log(Math.max(value, eps));
}
