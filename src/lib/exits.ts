export function atrStopTake(
  entry: number,
  atr: number,
  bias: "long" | "short",
  params: { s: number; t: number }
) {
  if (bias === "long") {
    return { sl: entry - params.s * atr, tp: entry + params.t * atr };
  }
  return { sl: entry + params.s * atr, tp: entry - params.t * atr };
}
