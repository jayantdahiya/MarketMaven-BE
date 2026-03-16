'use client';

import Link from 'next/link';
import useSWR from 'swr';
import { Card, CardContent } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { SignalBadge } from '@/components/forecast/signal-badge';
import { ConfidenceBar } from '@/components/forecast/confidence-bar';
import {
  fetchTickerForecast,
  fetchMetrics,
  swrKeys,
} from '@/lib/api';
import {
  formatReturn,
  formatSigned,
  modelLabel,
  modelShortLabel,
  TICKER_NAMES,
  staggerDelay,
} from '@/lib/utils';
import { useUIMode } from '@/hooks/use-ui-mode';
import { ChevronRight, AlertCircle } from 'lucide-react';

interface TickerCardProps {
  ticker: string;
  index: number;
}

function TickerCardSkeleton() {
  return (
    <div className="rounded-xl border border-[#1a1a24] bg-[#111118] p-5">
      <div className="flex items-start justify-between">
        <div className="space-y-1.5">
          <Skeleton className="h-5 w-14" />
          <Skeleton className="h-3.5 w-28" />
        </div>
        <Skeleton className="h-6 w-14 rounded-full" />
      </div>
      <div className="mt-5 space-y-3">
        <Skeleton className="h-2 w-full rounded-full" />
        <div className="flex justify-between">
          <Skeleton className="h-3 w-20" />
          <Skeleton className="h-3 w-12" />
        </div>
      </div>
    </div>
  );
}

export function TickerCard({ ticker, index }: TickerCardProps) {
  const { isQuant } = useUIMode();

  const { data: forecast, error: forecastError } = useSWR(
    swrKeys.forecast(ticker, 'lstm_baseline', 1),
    () => fetchTickerForecast(ticker, 'lstm_baseline', 1),
    { refreshInterval: 5 * 60 * 1000 },
  );

  const { data: metrics } = useSWR(
    isQuant ? swrKeys.metrics() : null,
    () => fetchMetrics(),
    { refreshInterval: 5 * 60 * 1000 },
  );

  const loading = !forecast && !forecastError;
  const prediction = forecast?.predictions?.[0];
  const companyName = TICKER_NAMES[ticker] ?? ticker;

  if (loading) return <TickerCardSkeleton />;

  if (forecastError) {
    return (
      <div className="rounded-xl border border-[#23232e] bg-[#111118] p-5">
        <div className="flex items-center gap-2 text-[#6b6b80]">
          <AlertCircle className="h-4 w-4 shrink-0" />
          <div>
            <p className="text-sm font-medium text-[#e8e8f0]">{ticker}</p>
            <p className="text-xs">Unable to load forecast</p>
          </div>
        </div>
      </div>
    );
  }

  const returnColor =
    prediction && prediction.predicted_return > 0
      ? 'text-[#f0c040]'
      : 'text-[#9090a8]';

  return (
    <Link href={`/forecast/${ticker}`} className="block group">
      <Card
        className="card-glow border-[#1a1a24] bg-[#111118] transition-all animate-fade-up cursor-pointer"
        style={{ animationDelay: staggerDelay(index) }}
      >
        <CardContent className="p-5">
          {/* Header row */}
          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-base font-semibold tracking-tight text-[#e8e8f0]">
                  {ticker}
                </span>
                {isQuant && forecast && (
                  <span className="inline-flex items-center rounded-full border border-[#f0c040]/30 bg-[#f0c040]/10 px-1.5 py-0.5 font-mono text-[10px] font-medium text-[#f0c040]">
                    {modelShortLabel(forecast.model)}
                  </span>
                )}
                <ChevronRight className="h-3.5 w-3.5 text-[#6b6b80] opacity-0 -translate-x-1 group-hover:opacity-100 group-hover:translate-x-0 transition-all duration-200" />
              </div>
              <p className="mt-0.5 text-xs text-[#6b6b80]">{companyName}</p>
            </div>
            {prediction && (
              <SignalBadge signal={prediction.signal} size="sm" />
            )}
          </div>

          {/* Predicted return */}
          {prediction && (
            <div className="mt-4">
              {!isQuant ? (
                // Retail mode: clean confidence bar + predicted return
                <>
                  <div className="mb-2 flex items-baseline justify-between">
                    <span className="text-[11px] uppercase tracking-widest text-[#6b6b80]">
                      Predicted Return
                    </span>
                    <span className={`font-mono text-sm font-medium tabular-nums ${returnColor}`}>
                      {formatReturn(prediction.predicted_return)}
                    </span>
                  </div>
                  <ConfidenceBar value={prediction.confidence} />
                  <p className="mt-2 text-[11px] text-[#6b6b80]">
                    Model confidence · {modelLabel(forecast!.model)}
                  </p>
                </>
              ) : (
                // Quant mode: data-dense rows
                <div className="space-y-2 mt-3">
                  <DataRow
                    label="Pred Return"
                    value={formatReturn(prediction.predicted_return)}
                    valueClass={returnColor}
                  />
                  <DataRow
                    label="Confidence"
                    value={`${(prediction.confidence * 100).toFixed(1)}%`}
                  />
                  <DataRow
                    label="Model"
                    value={modelLabel(forecast!.model)}
                  />
                  {metrics && (
                    <>
                      <div className="my-2 border-t border-[#1a1a24]" />
                      <DataRow
                        label="Dir Acc"
                        value={`${(metrics.directional_accuracy * 100).toFixed(1)}%`}
                      />
                      <DataRow
                        label="Sharpe"
                        value={formatSigned(metrics.sharpe, 3)}
                        valueClass={
                          metrics.sharpe > 0
                            ? 'text-[#22c55e]'
                            : 'text-[#ef4444]'
                        }
                      />
                      <DataRow
                        label="MAE"
                        value={metrics.mae.toFixed(5)}
                      />
                    </>
                  )}
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </Link>
  );
}

function DataRow({
  label,
  value,
  valueClass = 'text-[#e8e8f0]',
}: {
  label: string;
  value: string;
  valueClass?: string;
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-[11px] uppercase tracking-widest text-[#6b6b80]">
        {label}
      </span>
      <span className={`font-mono text-xs font-medium tabular-nums ${valueClass}`}>
        {value}
      </span>
    </div>
  );
}
