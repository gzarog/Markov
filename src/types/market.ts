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
  macd?: number;
  macdSignal?: number;
  macdHist?: number;
  bbUpper?: number;
  bbLower?: number;
  bbBasis?: number;
  bbWidth?: number;
  vwap?: number;
};
