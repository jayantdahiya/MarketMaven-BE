'use client';

import { useState } from 'react';
import { useUIMode } from '@/hooks/use-ui-mode';
import { TickerCard } from '@/components/dashboard/ticker-card';
import { ModelComparisonTable } from '@/components/dashboard/model-comparison-table';
import useSWR from 'swr';
import { fetchMetrics, swrKeys } from '@/lib/api';
import { Skeleton } from '@/components/ui/skeleton';
import { ChevronDown, ChevronRight } from 'lucide-react';
import type { ForecastModel, MetricsSummaryResponse } from '@/types/api';

const TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY'];
const COMPARISON_MODELS: ForecastModel[] = ['lstm_baseline', 'cnn_transformer'];

function MetricsBar() {
  const { data: metrics, isLoading } = useSWR(
    swrKeys.metrics(),
    () => fetchMetrics(),
  );

  if (isLoading) {
    return (
      <div className="flex gap-6">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="h-8 w-24" />
        ))}
      </div>
    );
  }

  if (!metrics) return null;

  const items = [
    { label: 'Dir Acc', value: `${(metrics.directional_accuracy * 100).toFixed(1)}%` },
    {
      label: 'Sharpe',
      value: metrics.sharpe.toFixed(3),
      color: metrics.sharpe > 0 ? '#22c55e' : '#ef4444',
    },
    { label: 'Sortino', value: metrics.sortino.toFixed(3) },
    { label: 'MAE', value: metrics.mae.toFixed(5) },
    { label: 'RMSE', value: metrics.rmse.toFixed(5) },
  ];

  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
      {items.map(({ label, value, color }) => (
        <div key={label} className="flex items-baseline gap-1.5">
          <span className="text-[10px] uppercase tracking-widest text-[#6b6b80]">
            {label}
          </span>
          <span
            className="font-mono text-sm font-medium tabular-nums"
            style={{ color: color ?? '#e8e8f0' }}
          >
            {value}
          </span>
        </div>
      ))}
      {metrics.model && (
        <span className="ml-auto text-[11px] text-[#6b6b80]">
          {metrics.model}
        </span>
      )}
    </div>
  );
}

function ModelComparisonPanel() {
  const [open, setOpen] = useState(false);

  const { data: lstmMetrics, isLoading: lstmLoading } = useSWR(
    swrKeys.metrics('lstm_baseline'),
    () => fetchMetrics('lstm_baseline'),
  );

  const { data: cnnMetrics, isLoading: cnnLoading } = useSWR(
    swrKeys.metrics('cnn_transformer'),
    () => fetchMetrics('cnn_transformer'),
  );

  const isLoading = lstmLoading || cnnLoading;
  const metricsMap: Partial<Record<ForecastModel, MetricsSummaryResponse>> = {};
  if (lstmMetrics) metricsMap['lstm_baseline'] = lstmMetrics;
  if (cnnMetrics) metricsMap['cnn_transformer'] = cnnMetrics;
  const hasData = !!lstmMetrics || !!cnnMetrics;

  return (
    <div className="rounded-lg border border-[#1a1a24] bg-[#111118]">
      {/* Toggle header */}
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between px-4 py-3 text-left"
        aria-expanded={open}
      >
        <span className="text-[10px] uppercase tracking-widest text-[#6b6b80]">
          Model Comparison · LSTM vs CNN-Transformer
        </span>
        {open ? (
          <ChevronDown className="h-3.5 w-3.5 text-[#6b6b80]" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 text-[#6b6b80]" />
        )}
      </button>

      {open && (
        <div className="border-t border-[#1a1a24] px-4 pb-4 pt-3">
          {isLoading ? (
            <div className="space-y-2">
              {Array.from({ length: 6 }).map((_, i) => (
                <div key={i} className="flex items-center justify-between gap-4">
                  <Skeleton className="h-3 w-24" />
                  <Skeleton className="h-3 w-16" />
                  <Skeleton className="h-3 w-16" />
                  <Skeleton className="h-3 w-16" />
                </div>
              ))}
            </div>
          ) : hasData ? (
            <ModelComparisonTable models={COMPARISON_MODELS} metrics={metricsMap} />
          ) : (
            <p className="text-xs text-[#6b6b80]">
              Could not load metrics for one or more models. Ensure both checkpoints are available.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export default function DashboardPage() {
  const { isQuant, mode } = useUIMode();

  const today = new Date().toLocaleDateString('en-US', {
    weekday: 'long',
    month: 'long',
    day: 'numeric',
  });

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="space-y-1">
        <div className="flex items-end justify-between">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight text-[#e8e8f0]">
              Signal Dashboard
            </h1>
            <p className="mt-1 text-sm text-[#6b6b80]">{today}</p>
          </div>
          <div className="flex items-center gap-2 rounded-full border border-[#23232e] bg-[#111118] px-3 py-1">
            <span className="h-1.5 w-1.5 rounded-full bg-[#22c55e]" />
            <span className="text-[11px] text-[#6b6b80]">
              {mode === 'quant' ? 'Advanced view' : 'Simple view'}
            </span>
          </div>
        </div>

        {/* Quant mode: global metrics bar */}
        {isQuant && (
          <div className="mt-4 rounded-lg border border-[#1a1a24] bg-[#111118] px-4 py-3">
            <div className="mb-2 flex items-center gap-2">
              <span className="text-[10px] uppercase tracking-widest text-[#6b6b80]">
                Model Evaluation · LSTM Baseline
              </span>
            </div>
            <MetricsBar />
          </div>
        )}

        {/* Quant mode: model comparison panel */}
        {isQuant && <ModelComparisonPanel />}
      </div>

      {/* Retail mode intro */}
      {!isQuant && (
        <div className="rounded-xl border border-[#1a1a24] bg-[#111118] px-5 py-4">
          <p className="text-sm leading-relaxed text-[#9090a8]">
            These signals show what our AI model predicts for each stock{' '}
            <span className="text-[#e8e8f0]">tomorrow</span>. A{' '}
            <span className="text-[#f0c040] font-medium">Long</span> signal means
            the model expects a positive return. A{' '}
            <span className="text-[#9090a8] font-medium">Flat</span> signal means
            the model expects little to no movement. Click any card for details.
          </p>
        </div>
      )}

      {/* Ticker grid */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
        {TICKERS.map((ticker, i) => (
          <TickerCard key={ticker} ticker={ticker} index={i} />
        ))}
      </div>

      {/* Quant footnote */}
      {isQuant && (
        <p className="text-[11px] text-[#6b6b80]">
          Metrics sourced from{' '}
          <code className="font-mono text-[#9090a8]">GET /metrics/latest</code>.
          Signals from{' '}
          <code className="font-mono text-[#9090a8]">POST /forecast/daily</code>{' '}
          · model: lstm_baseline · horizon: 1d. Auto-refreshes every 5 min.
        </p>
      )}
    </div>
  );
}

