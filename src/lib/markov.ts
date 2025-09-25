import { IDX, StateKey, STATES } from "../types/market";
import type { CandleRow } from "../types/market";

const STATE_COUNT = STATES.length;

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

type BuildOptions = {
  window?: number;
  smoothing?: number;
  halfLife?: number;
  order?: number;
  dirichletStrength?: number;
};

export type MarkovBuildResult = {
  counts: number[][];
  probs: number[][];
  weights: number[];
  contextDepth: number;
  baseDistribution: number[];
  options: Required<BuildOptions>;
};

export function buildMarkovWeighted(states: (StateKey | string)[], options: BuildOptions): MarkovBuildResult {
  const { window = states.length, smoothing = 0, halfLife = 0, order = 1, dirichletStrength = 0 } = options;
  const filtered = window && states.length > window ? states.slice(-window) : states.slice();
  if (filtered.length < 2) {
    const counts = Array.from({ length: STATE_COUNT }, () => Array(STATE_COUNT).fill(0)) as number[][];
    const probs = Array.from({ length: STATE_COUNT }, () => Array(STATE_COUNT).fill(1 / STATE_COUNT)) as number[][];
    return {
      counts,
      probs,
      weights: [],
      contextDepth: Math.max(0, order - 1),
      baseDistribution: Array(STATE_COUNT).fill(1 / STATE_COUNT),
      options: { window, smoothing, halfLife, order, dirichletStrength },
    };
  }

  const fromStates = filtered.slice(0, -1) as StateKey[];
  const toStates = filtered.slice(1) as StateKey[];
  const length = toStates.length;
  const weights = decayWeights(length, halfLife);
  const counts = Array.from({ length: STATE_COUNT }, () => Array(STATE_COUNT).fill(0)) as number[][];

  const orderDepth = Math.max(1, order);
  const contextDepth = Math.max(0, orderDepth - 1);

  const stateFreq = Array(STATE_COUNT).fill(0);
  filtered.forEach((state) => {
    const idx = IDX[state as StateKey];
    if (idx !== undefined) stateFreq[idx]++;
  });
  const totalFreq = stateFreq.reduce((acc, value) => acc + value, 0);
  const baseDistribution = totalFreq
    ? stateFreq.map((value) => value / totalFreq)
    : Array(STATE_COUNT).fill(1 / STATE_COUNT);

  const latestContext = contextDepth
    ? filtered.slice(filtered.length - 1 - contextDepth, filtered.length - 1)
    : [];

  for (let i = 0; i < length; i++) {
    const from = IDX[fromStates[i]];
    const to = IDX[toStates[i]];
    if (from === undefined || to === undefined) continue;
    let weight = weights[i];
    if (contextDepth) {
      const contextStart = i - contextDepth;
      if (contextStart >= 0) {
        const contextSlice = filtered.slice(contextStart, contextStart + contextDepth);
        let matches = 0;
        for (let j = 0; j < contextDepth; j++) {
          if (contextSlice[j] === latestContext[j]) matches++;
          else break;
        }
        if (matches === contextDepth) weight *= 1.5 + 0.5 * contextDepth;
        else if (matches > 0) weight *= 1 + matches / (contextDepth * 2);
        else weight *= 1;
      }
    }
    counts[from][to] += weight;
  }

  const probs = counts.map((row) => {
    const prior = baseDistribution.map((value) => value * dirichletStrength + smoothing);
    const withPrior = row.map((value, index) => value + prior[index]);
    const sum = withPrior.reduce((acc, value) => acc + value, 0);
    if (sum === 0) return Array(STATE_COUNT).fill(1 / STATE_COUNT);
    return withPrior.map((value) => value / sum);
  });

  return {
    counts,
    probs,
    weights,
    contextDepth,
    baseDistribution,
    options: { window, smoothing, halfLife, order: orderDepth, dirichletStrength },
  };
}

type SemiMarkovOptions = {
  smoothing?: number;
  minSamples?: number;
};

function extractRuns(states: (StateKey | string)[], state: StateKey) {
  const runs: number[] = [];
  let streak = 0;
  for (let i = 0; i < states.length; i++) {
    if (states[i] === state) {
      streak++;
    } else if (streak > 0) {
      runs.push(streak);
      streak = 0;
    }
  }
  if (streak > 0) runs.push(streak);
  return runs;
}

export function semiMarkovAdjustFirstRow(row: number[], states: (StateKey | string)[], opts: SemiMarkovOptions = {}) {
  const cur = states[states.length - 1] as StateKey;
  if (!cur) return row;

  const curIdx = IDX[cur];
  const runs = extractRuns(states, cur);
  if (runs.length < (opts.minSamples ?? 2)) return row;

  let currentRun = 1;
  for (let i = states.length - 2; i >= 0; i--) {
    if (states[i] === cur) currentRun++;
    else break;
  }

  const survivors = runs.filter((length) => length >= currentRun).length;
  const exits = runs.filter((length) => length === currentRun).length;
  const hazard = survivors ? exits / survivors : 0;
  const continueProb = Math.max(0, Math.min(1, 1 - hazard));
  const targetStay = currentRun > 1 ? Math.max(0.05, continueProb) : continueProb;

  const adjusted = row.slice();
  const smoothing = opts.smoothing ?? 0.35;
  adjusted[curIdx] = row[curIdx] * (1 - smoothing) + targetStay * smoothing;
  adjusted[curIdx] = Math.max(0, Math.min(1, adjusted[curIdx]));

  const remainder = 1 - adjusted[curIdx];
  const otherSum = row.reduce((acc, value, index) => (index === curIdx ? acc : acc + value), 0);
  for (let i = 0; i < STATE_COUNT; i++) {
    if (i === curIdx) continue;
    adjusted[i] = otherSum ? (row[i] / otherSum) * remainder : remainder / (STATE_COUNT - 1);
  }

  return adjusted;
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

export function stateProbabilityVector(probs: number[]) {
  const vector = new Float64Array(STATE_COUNT);
  let norm = 0;
  for (let i = 0; i < STATE_COUNT; i++) {
    const value = Math.max(0, probs[i]);
    vector[i] = Math.sqrt(value);
    norm += vector[i] * vector[i];
  }
  norm = Math.sqrt(norm);
  if (norm === 0) {
    const uniform = Math.sqrt(1 / STATE_COUNT);
    for (let i = 0; i < STATE_COUNT; i++) vector[i] = uniform;
    return vector;
  }
  for (let i = 0; i < STATE_COUNT; i++) vector[i] /= norm;
  return vector;
}

export function expectedDirectionalMove(probs: number[]) {
  const payoff: Record<StateKey, number> = { D: -1, R: 0.3, B: 0, U: 1 };
  return probs.reduce((acc, value, index) => acc + value * payoff[STATES[index]], 0);
}

function zeros(rows: number, cols: number) {
  return Array.from({ length: rows }, () => Array(cols).fill(0));
}

export function buildTransitionCounts(
  rows: CandleRow[],
  opts: { decay?: number; bucketFn?: (row: CandleRow) => string | undefined }
): Record<string, number[][]> | number[][] {
  const decay = opts.decay ?? 1;
  const byBucket = typeof opts.bucketFn === "function";
  const globalCounts = zeros(STATE_COUNT, STATE_COUNT);
  const bucketMap: Record<string, number[][]> = {};

  if (!rows.length) {
    return byBucket ? { global: globalCounts } : globalCounts;
  }

  for (let i = 1; i < rows.length; i++) {
    const prev = rows[i - 1].state as StateKey | undefined;
    const curr = rows[i].state as StateKey | undefined;
    if (!prev || !curr) continue;
    const from = IDX[prev];
    const to = IDX[curr];
    if (from === undefined || to === undefined) continue;
    const age = rows.length - 1 - i;
    const weight = Math.pow(decay, Math.max(age, 0));
    globalCounts[from][to] += weight;
    if (byBucket) {
      const key = opts.bucketFn!(rows[i]) ?? "global";
      if (!bucketMap[key]) bucketMap[key] = zeros(STATE_COUNT, STATE_COUNT);
      bucketMap[key][from][to] += weight;
    }
  }

  return byBucket ? { global: globalCounts, ...bucketMap } : globalCounts;
}

export function smoothCountsToProbs(counts: number[][], alpha = 0.75) {
  return counts.map((row) => {
    const total = row.reduce((acc, value) => acc + value, 0);
    return row.map((value) => (value + alpha) / (total + alpha * STATE_COUNT));
  });
}

export function blendMatrices(bucket: number[][] | undefined, global: number[][], beta = 0.6) {
  if (!bucket) return global;
  const out = zeros(STATE_COUNT, STATE_COUNT);
  for (let i = 0; i < STATE_COUNT; i++) {
    for (let j = 0; j < STATE_COUNT; j++) {
      out[i][j] = beta * bucket[i][j] + (1 - beta) * global[i][j];
    }
  }
  return out;
}

export function stepDistribution(dist: number[], mat: number[][]) {
  const next = Array(STATE_COUNT).fill(0);
  for (let i = 0; i < STATE_COUNT; i++) {
    for (let j = 0; j < STATE_COUNT; j++) {
      next[j] += dist[i] * mat[i][j];
    }
  }
  return next;
}

export function forecastDistribution(params: { mat: number[][]; startState: StateKey; steps: number }) {
  let dist = Array(STATE_COUNT).fill(0);
  dist[IDX[params.startState]] = 1;
  for (let k = 0; k < params.steps; k++) {
    dist = stepDistribution(dist, params.mat);
  }
  return dist;
}

export function distEntropy(probabilities: number[]) {
  const base = Math.log(STATE_COUNT);
  return base === 0
    ? 0
    : -probabilities.reduce((acc, value) => {
        if (value <= 0) return acc;
        return acc + value * Math.log(value);
      }, 0) / base;
}
