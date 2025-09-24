import { LogitModel } from "./conditioning";

const STORAGE_KEY = "markov_logit_model";

export function saveLogitModel(model: LogitModel) {
  if (typeof window === "undefined" || !window.localStorage) return;
  try {
    const payload = JSON.stringify(model);
    window.localStorage.setItem(STORAGE_KEY, payload);
  } catch (err) {
    // ignore storage errors
  }
}

export function loadLogitModel(): LogitModel | null {
  if (typeof window === "undefined" || !window.localStorage) return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!isValidLogitModel(parsed)) return null;
    return parsed;
  } catch (err) {
    return null;
  }
}

function isValidLogitModel(value: any): value is LogitModel {
  if (!value || typeof value !== "object") return false;
  const { W, b, temperature, mean, std } = value as Partial<LogitModel>;
  if (!Array.isArray(W) || !Array.isArray(b) || !Array.isArray(mean) || !Array.isArray(std)) return false;
  if (typeof temperature !== "number" || !Number.isFinite(temperature)) return false;
  if (!W.every((row) => Array.isArray(row) && row.every((x) => Number.isFinite(x)))) return false;
  if (!b.every((x) => Number.isFinite(x))) return false;
  if (!mean.every((x) => Number.isFinite(x))) return false;
  if (!std.every((x) => Number.isFinite(x))) return false;
  return true;
}

export function clearLogitModel() {
  if (typeof window === "undefined" || !window.localStorage) return;
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch (err) {
    // ignore
  }
}
