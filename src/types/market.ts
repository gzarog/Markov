export const STATES = ["D", "R", "B", "U"] as const; // Down, Reversal, Base, Up
export type StateKey = (typeof STATES)[number];

export const IDX: Record<StateKey, number> = STATES.reduce((acc, state, index) => {
  acc[state] = index;
  return acc;
}, {} as Record<StateKey, number>);

export type CandleRow = {
  time: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
  ema10?: number;
  ema50?: number;
  ma200?: number;
  ma200_slope?: number;
  atr14?: number;
  rsi?: number;
  stochK?: number;
  stochD?: number;
  ret1?: number;
  ret4?: number;
  ret12?: number;
  logRet?: number;
  realizedVol14?: number;
  realizedVol30?: number;
  emaTrendFast?: number;
  emaTrendSlow?: number;
  emaTrendHtf?: number;
  atrNorm?: number;
  volumeNorm?: number;
  htfState?: StateKey;
  ruleState?: StateKey;
  learnedState?: StateKey;
  learnedConfidence?: number;
};
