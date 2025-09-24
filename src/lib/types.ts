import { IDX, STATES } from "../types/market";
import type { CandleRow as MarketCandleRow, StateKey as MarketStateKey } from "../types/market";

export { STATES };
export type StateKey = MarketStateKey;
export type CandleRow = MarketCandleRow;
export const stateIdx: Record<StateKey, number> = IDX;

export type TransitionCounts = number[][];
export type TransitionMatrix = number[][];
