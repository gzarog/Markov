import { CandleRow } from "../types/market";
import { isFiniteNum } from "./utils";

export function labelState(row: CandleRow) {
  if (!isFiniteNum(row.ma200!)) return "B";
  const below200 = row.close < (row.ma200 ?? Infinity);
  const emaBull = (row.ema10 ?? -Infinity) >= (row.ema50 ?? Infinity * -1);
  const atr = row.atr14 ?? NaN;
  const near50 = isFiniteNum(atr) && Math.abs(row.close - (row.ema50 ?? row.close)) < 0.5 * atr;
  const rsiMid = isFiniteNum(row.rsi!) && row.rsi! >= 40 && row.rsi! <= 55;
  const slopeDown = (row.ma200_slope ?? 0) < 0;
  const slopeUp = (row.ma200_slope ?? 0) >= 0;
  if (row.close > (row.ma200 ?? Infinity) && emaBull && slopeUp) return "U";
  if (below200 && !emaBull && slopeDown) return "D";
  if (below200 && emaBull) return "R";
  if (near50 && rsiMid) return "B";
  return below200 ? "D" : "U";
}
