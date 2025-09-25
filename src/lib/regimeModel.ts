import { CandleRow, StateKey } from "../types/market";
import { clamp, isFiniteNum, mean, stddev } from "./utils";

type FeatureDescriptor = {
  key: string;
  label: string;
  extractor: (row: CandleRow) => number;
  post?: (value: number) => number;
};

type FeatureRow = {
  index: number;
  vector: number[];
  original: number[];
};

type KMeansResult = {
  centroids: number[][];
  assignments: number[];
  distances: number[];
};

type ClusterProfile = {
  idx: number;
  count: number;
  means: number[];
  trendScore: number;
  longScore: number;
  reversalScore: number;
  baseScore: number;
  rsi: number;
};

type RegimeInferenceOptions = {
  confidenceGate?: number;
  smoothPasses?: number;
};

type RegimeInferenceResult = {
  states: StateKey[];
  ruleStates: StateKey[];
  learnedStates: (StateKey | null)[];
  confidence: number[];
  clusters: ClusterProfile[];
};

const FEATURE_DESCRIPTORS: FeatureDescriptor[] = [
  { key: "ret1", label: "1h return", extractor: (row) => row.ret1 ?? NaN },
  { key: "ret4", label: "4h return", extractor: (row) => row.ret4 ?? NaN },
  { key: "ret12", label: "12h return", extractor: (row) => row.ret12 ?? NaN },
  { key: "emaFast", label: "EMA trend fast", extractor: (row) => row.emaTrendFast ?? NaN },
  { key: "emaSlow", label: "EMA trend slow", extractor: (row) => row.emaTrendSlow ?? NaN },
  { key: "emaHTF", label: "EMA trend HTF", extractor: (row) => row.emaTrendHtf ?? NaN },
  {
    key: "rsi",
    label: "RSI",
    extractor: (row) => row.rsi ?? NaN,
    post: (value) => (isFiniteNum(value) ? (value - 50) / 50 : NaN),
  },
  { key: "atrNorm", label: "ATR/Price", extractor: (row) => row.atrNorm ?? NaN },
  { key: "vol14", label: "Realized Vol (14)", extractor: (row) => row.realizedVol14 ?? NaN },
  { key: "volumeNorm", label: "Volume z", extractor: (row) => row.volumeNorm ?? NaN },
];

function buildFeatureRows(rows: CandleRow[]): FeatureRow[] {
  const features: FeatureRow[] = [];
  rows.forEach((row, index) => {
    const vector = FEATURE_DESCRIPTORS.map((descriptor) => {
      const raw = descriptor.extractor(row);
      return descriptor.post ? descriptor.post(raw) : raw;
    });
    if (vector.every(isFiniteNum)) {
      features.push({ index, vector, original: vector.slice() });
    }
  });
  return features;
}

function standardize(features: FeatureRow[]) {
  if (!features.length) return { standardized: [] as number[][], means: [] as number[], stds: [] as number[] };
  const dims = features[0].vector.length;
  const means: number[] = [];
  const stds: number[] = [];
  for (let d = 0; d < dims; d++) {
    const values = features.map((feature) => feature.vector[d]);
    means[d] = mean(values);
    const sd = stddev(values);
    stds[d] = isFiniteNum(sd) && sd > 0 ? sd : 1;
  }
  const standardized = features.map((feature) =>
    feature.vector.map((value, dim) => (value - means[dim]) / stds[dim])
  );
  return { standardized, means, stds };
}

function euclideanSquared(a: number[], b: number[]) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

function initCentroids(data: number[][], k: number) {
  const centroids: number[][] = [];
  if (!data.length) return centroids;
  const first = data[Math.floor(data.length / 2)];
  centroids.push(first.slice());
  while (centroids.length < k) {
    let bestIndex = 0;
    let bestDist = -Infinity;
    for (let i = 0; i < data.length; i++) {
      const point = data[i];
      let minDist = Infinity;
      for (const centroid of centroids) {
        const dist = euclideanSquared(point, centroid);
        if (dist < minDist) minDist = dist;
      }
      if (minDist > bestDist) {
        bestDist = minDist;
        bestIndex = i;
      }
    }
    centroids.push(data[bestIndex].slice());
  }
  return centroids;
}

function kMeans(data: number[][], k: number, maxIter = 60): KMeansResult {
  if (!data.length) return { centroids: [], assignments: [], distances: [] };
  const centroids = initCentroids(data, k);
  const assignments = new Array(data.length).fill(0);
  const distances = new Array(data.length).fill(0);

  for (let iter = 0; iter < maxIter; iter++) {
    let moved = false;
    for (let i = 0; i < data.length; i++) {
      const point = data[i];
      let best = 0;
      let bestDist = Infinity;
      for (let c = 0; c < centroids.length; c++) {
        const dist = euclideanSquared(point, centroids[c]);
        if (dist < bestDist) {
          bestDist = dist;
          best = c;
        }
      }
      if (assignments[i] !== best) {
        moved = true;
        assignments[i] = best;
      }
      distances[i] = Math.sqrt(Math.max(bestDist, 0));
    }

    if (!moved && iter > 1) break;

    const sums = Array.from({ length: k }, () => new Array(data[0].length).fill(0));
    const counts = new Array(k).fill(0);

    for (let i = 0; i < data.length; i++) {
      const cluster = assignments[i];
      counts[cluster]++;
      const point = data[i];
      for (let d = 0; d < point.length; d++) {
        sums[cluster][d] += point[d];
      }
    }

    for (let c = 0; c < k; c++) {
      if (counts[c] === 0) {
        centroids[c] = data[Math.floor(Math.random() * data.length)].slice();
        continue;
      }
      centroids[c] = sums[c].map((value) => value / counts[c]);
    }
  }

  return { centroids, assignments, distances };
}

function computeClusterProfiles(
  features: FeatureRow[],
  assignments: number[],
  confidence: number[]
): ClusterProfile[] {
  const dims = FEATURE_DESCRIPTORS.length;
  const stats = Array.from({ length: Math.max(...assignments) + 1 }, (_, idx) => ({
    idx,
    count: 0,
    sums: new Array(dims).fill(0),
  }));

  assignments.forEach((cluster, i) => {
    const stat = stats[cluster];
    stat.count++;
    for (let d = 0; d < dims; d++) {
      stat.sums[d] += features[i].original[d];
    }
  });

  return stats.map((stat) => {
    const means = stat.sums.map((value) => (stat.count ? value / stat.count : 0));
    const ret4 = means[FEATURE_DESCRIPTORS.findIndex((descriptor) => descriptor.key === "ret4")];
    const ret1 = means[FEATURE_DESCRIPTORS.findIndex((descriptor) => descriptor.key === "ret1")];
    const ret12 = means[FEATURE_DESCRIPTORS.findIndex((descriptor) => descriptor.key === "ret12")];
    const emaFast = means[FEATURE_DESCRIPTORS.findIndex((descriptor) => descriptor.key === "emaFast")];
    const emaSlow = means[FEATURE_DESCRIPTORS.findIndex((descriptor) => descriptor.key === "emaSlow")];
    const emaHTF = means[FEATURE_DESCRIPTORS.findIndex((descriptor) => descriptor.key === "emaHTF")];
    const rsiIndex = FEATURE_DESCRIPTORS.findIndex((descriptor) => descriptor.key === "rsi");
    const rsi = means[rsiIndex] * 50 + 50;

    const trendScore = (ret4 ?? 0) + 0.5 * (ret1 ?? 0) + (emaFast ?? 0) + 0.5 * (emaSlow ?? 0);
    const longScore = (emaSlow ?? 0) + 0.5 * (emaHTF ?? 0) + (ret12 ?? 0);
    const reversalScore = (emaFast ?? 0) - (emaSlow ?? 0) + (ret4 ?? 0);
    const baseScore = -Math.abs(ret1 ?? 0) - Math.abs(ret4 ?? 0) - Math.abs(emaFast ?? 0);

    return {
      idx: stat.idx,
      count: stat.count,
      means,
      trendScore,
      longScore,
      reversalScore,
      baseScore,
      rsi,
    };
  });
}

function assignStatesFromClusters(profiles: ClusterProfile[]) {
  const assignments: Record<number, StateKey> = {};
  if (!profiles.length) return assignments;

  const sortedByTrend = [...profiles].sort((a, b) => a.trendScore - b.trendScore);
  const downCluster = sortedByTrend[0];
  const upCluster = sortedByTrend[sortedByTrend.length - 1];
  assignments[downCluster.idx] = "D";
  assignments[upCluster.idx] = "U";

  const remaining = profiles.filter((profile) => !assignments[profile.idx]);
  if (remaining.length) {
    remaining.sort((a, b) => b.reversalScore - a.reversalScore);
    const reversal = remaining.shift();
    if (reversal) assignments[reversal.idx] = "R";
  }
  const rest = profiles.filter((profile) => !assignments[profile.idx]);
  rest.forEach((profile) => {
    if (!assignments[profile.idx]) assignments[profile.idx] = "B";
  });
  return assignments;
}

function smoothStates(states: StateKey[], passes: number) {
  let current = states.slice();
  for (let pass = 0; pass < passes; pass++) {
    const next = current.slice();
    for (let i = 1; i < current.length - 1; i++) {
      if (current[i - 1] === current[i + 1] && current[i] !== current[i - 1]) {
        next[i] = current[i - 1];
      }
    }
    current = next;
  }
  return current;
}

export function inferRegimeStates(
  rows: CandleRow[],
  ruleStates: StateKey[],
  options: RegimeInferenceOptions = {}
): RegimeInferenceResult {
  const { confidenceGate = 0.4, smoothPasses = 2 } = options;
  const features = buildFeatureRows(rows);
  if (!features.length) {
    return {
      states: ruleStates.slice(),
      ruleStates,
      learnedStates: ruleStates.map(() => null),
      confidence: ruleStates.map(() => 0),
      clusters: [],
    };
  }

  const { standardized } = standardize(features);
  const k = Math.min(4, new Set(ruleStates).size || 4);
  const { assignments, distances } = kMeans(standardized, k);

  const maxDist = Math.max(...distances.map((value) => (isFiniteNum(value) ? value : 0)));
  const minDist = Math.min(...distances.map((value) => (isFiniteNum(value) ? value : 0)));
  const confidence = distances.map((distance) => {
    if (!isFiniteNum(distance)) return 0;
    if (maxDist === minDist) return 1;
    const scaled = (distance - minDist) / (maxDist - minDist);
    return clamp(Math.exp(-scaled * 3), 0, 1);
  });

  const clusterProfiles = computeClusterProfiles(features, assignments, confidence);
  const clusterMap = assignStatesFromClusters(clusterProfiles);

  const learnedStates: (StateKey | null)[] = rows.map(() => null);
  const learnedConfidence: number[] = rows.map(() => 0);
  assignments.forEach((cluster, idx) => {
    const rowIndex = features[idx].index;
    const state = clusterMap[cluster];
    learnedStates[rowIndex] = state ?? null;
    learnedConfidence[rowIndex] = confidence[idx] ?? 0;
  });

  const combinedStates: StateKey[] = rows.map((_, index) => {
    const rule = ruleStates[index];
    const learned = learnedStates[index];
    const conf = learnedConfidence[index];
    const htf = rows[index].htfState;
    if (!learned) return rule;
    if (learned === rule) return learned;
    if (conf >= confidenceGate + 0.2) return learned;
    if (conf <= confidenceGate * 0.5) return rule;
    if (htf === "U" && learned === "D") return conf > confidenceGate ? "R" : rule;
    if (htf === "D" && learned === "U") return conf > confidenceGate ? "B" : rule;
    return conf >= confidenceGate ? learned : rule;
  });

  const smoothed = smoothStates(combinedStates, smoothPasses);

  return {
    states: smoothed,
    ruleStates,
    learnedStates,
    confidence: learnedConfidence,
    clusters: clusterProfiles,
  };
}
