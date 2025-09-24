export function entropySizing(entropy01: number, base: number) {
  const scaled = 1 - Math.max(0, Math.min(1, entropy01));
  const clamped = Math.max(0.15, Math.min(1, scaled));
  return base * clamped;
}

export function edgeAfterCosts(edge: number, maker: number, taker: number, funding: number) {
  return edge - (Math.abs(maker) + Math.abs(taker)) - Math.abs(funding);
}
