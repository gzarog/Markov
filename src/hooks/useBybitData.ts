import { useCallback, useEffect, useRef, useState } from "react";

import { bybitKlineUrl } from "../lib/api";
import { CandleRow } from "../types/market";

export function useBybitData({
  symbol,
  interval,
  limit,
  refreshMs,
}: {
  symbol: string;
  interval: string | number;
  limit: number;
  refreshMs: number | null;
}) {
  const [data, setData] = useState<CandleRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const tickRef = useRef(0);

  const fetchNow = useCallback(async () => {
    try {
      setLoading(true);
      const res = await fetch(bybitKlineUrl({ symbol, interval, limit }));
      const js = await res.json();
      if (js.retCode !== 0) throw new Error(JSON.stringify(js));
      const list = [...js.result.list].reverse();
      const candles: CandleRow[] = list.map((r: any) => ({
        time: new Date(Number(r[0])),
        open: +r[1],
        high: +r[2],
        low: +r[3],
        close: +r[4],
        volume: +r[5],
      }));
      setData(candles);
      setError(null);
    } catch (e: any) {
      setError(String(e?.message || e));
    } finally {
      setLoading(false);
      tickRef.current++;
    }
  }, [interval, limit, symbol]);

  useEffect(() => {
    fetchNow();
  }, [fetchNow]);

  useEffect(() => {
    if (!refreshMs) return;
    const id = setInterval(fetchNow, refreshMs);
    return () => clearInterval(id);
  }, [fetchNow, refreshMs]);

  return { data, error, loading, refetch: fetchNow, tick: tickRef.current };
}
