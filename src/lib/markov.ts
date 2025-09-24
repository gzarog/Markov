import { IDX, StateKey, STATES } from "../types/market";

const STATE_COUNT = STATES.length;

type DurationMap = Record<StateKey, number[]>;
export type PairKey = `${StateKey}${StateKey}`;

function normalizeRow(row: number[]) {
  const sum = row.reduce((acc, value) => acc + value, 0);
  if (sum <= 0) {
    const uniform = 1 / STATE_COUNT;
    return Array(STATE_COUNT).fill(uniform);
  }
  return row.map((value) => value / sum);
}

export function decayWeights(length: number, halfLife: number) {
  if (halfLife <= 0) return Array(length).fill(1) as number[];
  const lambda = Math.log(2) / halfLife;
  const weights: number[] = [];
  for (let offset = length - 1; offset >= 0; offset--) {
    weights.push(Math.exp(-lambda * offset));
  }
  const max = Math.max(...weights);
  return weights.map((value) => value / max);
}

export function buildMarkovWeighted(states: string[], window: number, smoothing: number, halfLife: number) {
  const filtered = window && states.length > window ? states.slice(-window) : states.slice();
  const fromStates = filtered.slice(0, -1);
  const toStates = filtered.slice(1);
  const length = toStates.length;
  const weights = decayWeights(length, halfLife);
  const counts = Array.from({ length: STATE_COUNT }, () => Array(STATE_COUNT).fill(0)) as number[][];

  for (let i = 0; i < length; i++) {
    const from = IDX[fromStates[i] as StateKey];
    const to = IDX[toStates[i] as StateKey];
    if (from !== undefined && to !== undefined) {
      counts[from][to] += weights[i];
    }
  }

  const probs = counts.map((row) => row.map((value) => value + smoothing));
  for (let i = 0; i < STATE_COUNT; i++) {
    const sum = probs[i].reduce((acc, value) => acc + value, 0);
    for (let j = 0; j < STATE_COUNT; j++) {
      probs[i][j] = sum ? probs[i][j] / sum : 1 / STATE_COUNT;
    }
  }

  return { counts, probs };
}

export function computeRunLength(states: StateKey[]) {
  if (!states.length) return 0;
  const cur = states[states.length - 1];
  let run = 1;
  for (let i = states.length - 2; i >= 0; i--) {
    if (states[i] === cur) run++;
    else break;
  }
  return run;
}

export function estimateDurations(states: StateKey[]): DurationMap {
  const out: DurationMap = { D: [], R: [], B: [], U: [] };
  if (!states.length) return out;
  let current = states[0];
  let length = 1;
  for (let i = 1; i < states.length; i++) {
    if (states[i] === current) {
      length++;
    } else {
      out[current].push(length);
      current = states[i];
      length = 1;
    }
  }
  out[current].push(length);
  (Object.keys(out) as StateKey[]).forEach((k) => {
    if (!out[k].length) out[k].push(1);
  });
  return out;
}

function survivalProb(samples: number[], runLength: number) {
  if (!samples.length) return 0.5;
  const greater = samples.filter((value) => value > runLength).length;
  return (greater + 1) / (samples.length + 2);
}

export function semiMarkovAdjustFirstRow(
  row: number[],
  states: StateKey[],
  opts: { durations?: DurationMap; runLength?: number } = {}
) {
  if (!states.length) return row;
  const cur = states[states.length - 1];
  const base = row.slice();
  const run = opts.runLength ?? computeRunLength(states);
  const durations = opts.durations;

  if (durations) {
    const stayIdx = IDX[cur];
    const stayBase = base[stayIdx] ?? 0;
    const survival = survivalProb(durations[cur] ?? [], run);
    const boostedStay = Math.min(0.995, Math.max(0.005, survival * Math.max(0.05, stayBase)));
    const exitsSum = base.reduce((acc, value, idx) => (idx === stayIdx ? acc : acc + value), 0);
    const scaled = base.map((value, idx) => {
      if (idx === stayIdx) return boostedStay;
      if (exitsSum <= 0) return (1 - boostedStay) / (STATE_COUNT - 1);
      return value * ((1 - boostedStay) / exitsSum);
    });
    return normalizeRow(scaled);
  }

  // fallback to legacy heuristic if no durations provided
  let runLegacy = 1;
  for (let i = states.length - 2; i >= 0; i--) {
    if (states[i] === cur) runLegacy++;
    else break;
  }
  const runs: number[] = [];
  let streak = 0;
  for (let i = 0; i < states.length; i++) {
    if (states[i] === cur) {
      streak++;
    } else if (streak > 0) {
      runs.push(streak);
      streak = 0;
    }
  }
  if (streak > 0) runs.push(streak);
  if (!runs.length) return normalizeRow(base);
  const sorted = [...runs].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  if (runLegacy < median) {
    const boost = Math.min(0.15, 0.03 * (median - runLegacy));
    const curIndex = IDX[cur];
    base[curIndex] = Math.min(1, base[curIndex] + boost);
    const others = Array.from({ length: STATE_COUNT }, (_, index) => index).filter((index) => index !== curIndex);
    const remainder = 1 - base[curIndex];
    const restSum = others.reduce((acc, index) => acc + row[index], 0);
    others.forEach((index) => {
      base[index] = restSum ? (row[index] * remainder) / restSum : remainder / (STATE_COUNT - 1);
    });
  }
  return normalizeRow(base);
}

export function buildOrder2Counts(states: StateKey[]): Record<PairKey, number[]> {
  const table: Partial<Record<PairKey, number[]>> = {};
  for (let i = 1; i + 1 < states.length; i++) {
    const key = `${states[i - 1]}${states[i]}` as PairKey;
    if (!table[key]) table[key] = Array(STATE_COUNT).fill(0);
    const nextIdx = IDX[states[i + 1]];
    table[key]![nextIdx] += 1;
  }
  return table as Record<PairKey, number[]>;
}

export function rowFromOrder2(table: Record<PairKey, number[]>, pair: PairKey, minCount = 12) {
  const arr = table[pair];
  if (!arr) return null;
  const total = arr.reduce((acc, value) => acc + value, 0);
  if (total < minCount) return null;
  const smoothed = arr.map((value) => value + 0.5);
  return normalizeRow(smoothed);
}

function matMul(A: number[][], B: number[][]) {
  const result = Array.from({ length: STATE_COUNT }, () => Array(STATE_COUNT).fill(0)) as number[][];
  for (let r = 0; r < STATE_COUNT; r++) {
    for (let c = 0; c < STATE_COUNT; c++) {
      for (let m = 0; m < STATE_COUNT; m++) {
        result[r][c] += A[r][m] * B[m][c];
      }
    }
  }
  return result;
}

export function matPow(matrix: number[][], k: number) {
  let result = Array.from({ length: STATE_COUNT }, (_, index) =>
    Array.from({ length: STATE_COUNT }, (_, j) => (index === j ? 1 : 0))
  ) as number[][];
  for (let i = 0; i < k; i++) {
    result = matMul(result, matrix);
  }
  return result;
}

export function vecMul(vector: number[], matrix: number[][]) {
  const out = Array(STATE_COUNT).fill(0) as number[];
  for (let j = 0; j < STATE_COUNT; j++) {
    for (let k = 0; k < STATE_COUNT; k++) {
      out[j] += vector[k] * matrix[k][j];
    }
  }
  return out;
}

export function multiStepForecast(
  transition: number[][],
  currentState: StateKey,
  steps: Record<string, number>,
  firstRowAdj?: number[]
) {
  const v0 = Array(STATE_COUNT).fill(0) as number[];
  v0[IDX[currentState]] = 1;
  const matrix = transition.map((row) => row.slice());
  const result: Record<string, number[]> = {};

  Object.entries(steps).forEach(([name, k]) => {
    if (k <= 0) {
      result[name] = v0.slice();
      return;
    }

    if (firstRowAdj) {
      const withAdj = matrix.map((row) => row.slice());
      withAdj[IDX[currentState]] = firstRowAdj.slice();
      let vec = vecMul(v0, matPow(withAdj, 1));
      if (k > 1) {
        vec = vecMul(vec, matPow(matrix, k - 1));
      }
      result[name] = vec;
    } else {
      result[name] = vecMul(v0, matPow(matrix, k));
    }
  });

  return result;
}
