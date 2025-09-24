export const bybitKlineUrl = ({
  symbol,
  interval,
  limit,
}: {
  symbol: string;
  interval: string | number;
  limit: number;
}) => {
  const params = new URLSearchParams({
    category: "linear",
    symbol: symbol.toUpperCase(),
    interval: String(interval),
    limit: String(limit),
  });
  return `https://api.bybit.com/v5/market/kline?${params.toString()}`;
};
