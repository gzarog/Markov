export const isFiniteNum = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);
