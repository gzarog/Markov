import { CandleRow, StateKey, STATES } from "../../types/market";
import { buildFeatures, defaultLogitModel, LogitModel } from "./conditioning";
import { isFiniteNum } from "../utils";

const FEATURE_DIM = 10;
const STATE_DIM = STATES.length;

function softmax(logits: number[]) {
  const max = Math.max(...logits);
  const exps = logits.map((z) => Math.exp(z - max));
  const Z = exps.reduce((acc, value) => acc + value, 0);
  return exps.map((value) => value / Z);
}

export function fitLogitModel(rows: CandleRow[], states: StateKey[]): LogitModel {
  const model = defaultLogitModel();
  if (rows.length < 120 || states.length !== rows.length) {
    return model;
  }

  const samples: { x: number[]; label: number }[] = [];
  for (let i = 25; i < rows.length - 1; i++) {
    const next = states[i + 1];
    const idx = STATES.indexOf(next);
    if (idx < 0) continue;
    const feat = buildFeatures(rows, i);
    if (feat.some((v) => !isFiniteNum(v))) continue;
    samples.push({ x: feat, label: idx });
  }

  if (samples.length < 40) {
    return model;
  }

  const mean = Array(FEATURE_DIM).fill(0);
  const std = Array(FEATURE_DIM).fill(0);
  samples.forEach(({ x }) => {
    x.forEach((value, i) => {
      mean[i] += value;
    });
  });
  for (let i = 0; i < FEATURE_DIM; i++) mean[i] /= samples.length;
  samples.forEach(({ x }) => {
    x.forEach((value, i) => {
      std[i] += (value - mean[i]) ** 2;
    });
  });
  for (let i = 0; i < FEATURE_DIM; i++) {
    std[i] = Math.sqrt(std[i] / Math.max(1, samples.length - 1)) || 1;
  }

  const norm = samples.map(({ x, label }) => ({
    x: x.map((value, i) => (value - mean[i]) / std[i]),
    label,
  }));

  const W = Array.from({ length: STATE_DIM }, () => Array(FEATURE_DIM).fill(0));
  const b = Array(STATE_DIM).fill(0);
  const lr = 0.12;
  const lambda = 1e-3;
  const iterations = 80;

  for (let iter = 0; iter < iterations; iter++) {
    const gradW = Array.from({ length: STATE_DIM }, () => Array(FEATURE_DIM).fill(0));
    const gradB = Array(STATE_DIM).fill(0);
    norm.forEach(({ x, label }) => {
      const logits = b.map((bias, j) => {
        let z = bias;
        for (let f = 0; f < FEATURE_DIM; f++) {
          z += W[j][f] * x[f];
        }
        return z;
      });
      const probs = softmax(logits);
      for (let j = 0; j < STATE_DIM; j++) {
        const diff = probs[j] - (j === label ? 1 : 0);
        gradB[j] += diff;
        for (let f = 0; f < FEATURE_DIM; f++) {
          gradW[j][f] += diff * x[f];
        }
      }
    });
    const invN = 1 / norm.length;
    for (let j = 0; j < STATE_DIM; j++) {
      b[j] -= lr * gradB[j] * invN;
      for (let f = 0; f < FEATURE_DIM; f++) {
        const reg = lambda * W[j][f];
        W[j][f] -= lr * (gradW[j][f] * invN + reg);
      }
    }
  }

  const logitsList: number[][] = [];
  const labels: number[] = [];
  norm.forEach(({ x, label }) => {
    const logits = b.map((bias, j) => {
      let z = bias;
      for (let f = 0; f < FEATURE_DIM; f++) z += W[j][f] * x[f];
      return z;
    });
    logitsList.push(logits);
    labels.push(label);
  });

  const temps = [0.6, 0.8, 1, 1.2, 1.4, 1.6];
  let bestTemp = 1;
  let bestScore = Number.POSITIVE_INFINITY;
  temps.forEach((temp) => {
    let score = 0;
    logitsList.forEach((logits, idx) => {
      const scaled = logits.map((value) => value / temp);
      const probs = softmax(scaled);
      probs.forEach((p, j) => {
        score += (p - (j === labels[idx] ? 1 : 0)) ** 2;
      });
    });
    if (score < bestScore) {
      bestScore = score;
      bestTemp = temp;
    }
  });

  return {
    W,
    b,
    temperature: bestTemp,
    mean,
    std,
  };
}
