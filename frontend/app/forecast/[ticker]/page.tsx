'use client';

import { useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import useSWR from 'swr';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { fetchTickerForecast, fetchMetrics, swrKeys } from '@/lib/api';
import {
  formatReturn,
  formatSigned,
  modelLabel,
  modelShortLabel,
  TICKER_NAMES,
} from '@/lib/utils';
import { useUIMode } from '@/hooks/use-ui-mode';
import { SignalBadge } from '@/components/forecast/signal-badge';
import { ConfidenceBar } from '@/components/forecast/confidence-bar';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { ArrowLeft, RefreshCw, GitCompare } from 'lucide-react';
import type { ForecastModel, ForecastHorizon, DailyForecastPoint } from '@/types/api';

const MODELS: { value: ForecastModel; label: string }[] = [
  { value: 'lstm_baseline', label: 'LSTM Baseline' },
  { value: 'cnn_transformer', label: 'CNN-Transformer' },
  { value: 'mamba_ssm', label: 'Mamba SSM' },
  { value: 'prophet', label: 'Prophet' },
];

const HORIZONS: { value: ForecastHorizon; label: string }[] = [
  { value: 1, label: '1 Day' },
  { value: 5, label: '5 Days' },
  { value: 10, label: '10 Days' },
];

// The secondary model shown when compare mode is on
const COMPARE_MODEL: ForecastModel = 'cnn_transformer';

// Custom chart tooltip
function ChartTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ value: number; name: string; color: string; payload: { date: string } }>;
}) {
  if (!active || !payload?.length) return null;
  const date = payload[0]?.payload?.date;
  return (
    <div className="rounded-lg border border-[#23232e] bg-[#111118] px-3 py-2 shadow-xl">
      <p className="mb-1 text-[11px] text-[#6b6b80]">{date}</p>
      {payload.map((p) => (
        <p
          key={p.name}
          className="font-mono text-xs font-semibold tabular-nums"
          style={{ color: p.color }}
        >
          {p.name}: {formatReturn(p.value)}
        </p>
      ))}
    </div>
  );
}

// A single stat card
function StatCard({
  label,
  value,
  valueColor,
  mono = true,
  highlight = false,
}: {
  label: string;
  value: string;
  valueColor?: string;
  mono?: boolean;
  highlight?: boolean;
}) {
  return (
    <div
      className={`rounded-lg border bg-[#111118] p-4 ${
        highlight ? 'border-[#f0c040]/50 ring-1 ring-[#f0c040]/40' : 'border-[#1a1a24]'
      }`}
    >
      <p className="text-[10px] uppercase tracking-widest text-[#6b6b80]">{label}</p>
      <p
        className={`mt-1 text-sm font-semibold ${mono ? 'font-mono tabular-nums' : ''}`}
        style={{ color: valueColor ?? '#e8e8f0' }}
      >
        {value}
      </p>
    </div>
  );
}

// Two-column comparison stat grid
function CompareStatGrid({
  primaryModel,
  primaryPrediction,
  comparePrediction,
  horizon,
}: {
  primaryModel: ForecastModel;
  primaryPrediction: DailyForecastPoint;
  comparePrediction: DailyForecastPoint | null;
  horizon: ForecastHorizon;
}) {
  const primaryHigher =
    !comparePrediction ||
    primaryPrediction.confidence >= comparePrediction.confidence;

  const colClass = 'space-y-2 flex-1';

  function modelColumnHeader(model: ForecastModel) {
    return (
      <p className="mb-2 text-center text-[10px] uppercase tracking-widest text-[#f0c040]">
        {modelShortLabel(model)}
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-4 sm:flex-row">
      {/* Primary model column */}
      <div className={colClass}>
        {modelColumnHeader(primaryModel)}
        <StatCard
          label="Pred Return"
          value={formatReturn(primaryPrediction.predicted_return)}
          valueColor={primaryPrediction.predicted_return >= 0 ? '#f0c040' : '#ef4444'}
          highlight={primaryHigher}
        />
        <StatCard
          label="Confidence"
          value={`${(primaryPrediction.confidence * 100).toFixed(2)}%`}
          highlight={primaryHigher}
        />
        <StatCard
          label="Signal"
          value={primaryPrediction.signal.toUpperCase()}
          valueColor={primaryPrediction.signal === 'long' ? '#f0c040' : '#9090a8'}
          highlight={primaryHigher}
        />
        <StatCard
          label="Horizon"
          value={`${horizon}d`}
          highlight={primaryHigher}
        />
      </div>

      {/* Divider */}
      <div className="hidden w-px self-stretch bg-[#1a1a24] sm:block" />

      {/* Compare model column */}
      <div className={colClass}>
        {modelColumnHeader(COMPARE_MODEL)}
        {comparePrediction ? (
          <>
            <StatCard
              label="Pred Return"
              value={formatReturn(comparePrediction.predicted_return)}
              valueColor={comparePrediction.predicted_return >= 0 ? '#f0c040' : '#ef4444'}
              highlight={!primaryHigher}
            />
            <StatCard
              label="Confidence"
              value={`${(comparePrediction.confidence * 100).toFixed(2)}%`}
              highlight={!primaryHigher}
            />
            <StatCard
              label="Signal"
              value={comparePrediction.signal.toUpperCase()}
              valueColor={comparePrediction.signal === 'long' ? '#f0c040' : '#9090a8'}
              highlight={!primaryHigher}
            />
            <StatCard
              label="Horizon"
              value={`${horizon}d`}
              highlight={!primaryHigher}
            />
          </>
        ) : (
          <div className="space-y-2">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-[60px] rounded-lg" />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function ForecastPage() {
  const params = useParams();
  const ticker = (params.ticker as string).toUpperCase();
  const { isQuant } = useUIMode();

  const [model, setModel] = useState<ForecastModel>('lstm_baseline');
  const [horizon, setHorizon] = useState<ForecastHorizon>(1);
  const [compareMode, setCompareMode] = useState(false);

  // Primary model SWR
  const swrKey = swrKeys.forecast(ticker, model, horizon);
  const { data: forecast, error, isLoading, mutate } = useSWR(
    swrKey,
    () => fetchTickerForecast(ticker, model, horizon),
    { refreshInterval: 5 * 60 * 1000 },
  );

  // Compare model SWR — only active when compareMode is on and primary model is not already CNN-T
  const showCompare = isQuant && compareMode && model !== COMPARE_MODEL;
  const compareSwrKey = showCompare
    ? swrKeys.forecast(ticker, COMPARE_MODEL, horizon)
    : null;
  const { data: compareForecast } = useSWR(
    compareSwrKey,
    () => fetchTickerForecast(ticker, COMPARE_MODEL, horizon),
    { refreshInterval: 5 * 60 * 1000 },
  );

  const { data: metrics } = useSWR(
    swrKeys.metrics(),
    () => fetchMetrics(),
  );

  const prediction = forecast?.predictions?.[0];
  const comparePrediction = compareForecast?.predictions?.[0] ?? null;
  const companyName = TICKER_NAMES[ticker] ?? ticker;

  // Build chart data — current prediction(s) + synthetic "recent" history
  const chartData = prediction
    ? [
        { date: 'T-4', primary: 0 },
        { date: 'T-3', primary: 0 },
        { date: 'T-2', primary: 0 },
        { date: 'T-1', primary: 0 },
        { date: 'Today', primary: 0 },
        {
          date: 'Tomorrow',
          primary: prediction.predicted_return,
          ...(showCompare && comparePrediction
            ? { compare: comparePrediction.predicted_return }
            : {}),
          forecast: true,
        },
      ]
    : [];

  const primaryColor = prediction
    ? prediction.predicted_return >= 0
      ? '#f0c040'
      : '#ef4444'
    : '#f0c040';

  const compareColor = comparePrediction
    ? comparePrediction.predicted_return >= 0
      ? '#60a5fa'
      : '#f97316'
    : '#60a5fa';

  return (
    <div className="space-y-8">
      {/* Back nav */}
      <div className="flex items-center gap-3">
        <Link href="/">
          <Button
            variant="ghost"
            size="sm"
            className="gap-1.5 text-[#6b6b80] hover:text-[#e8e8f0] -ml-2"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Dashboard
          </Button>
        </Link>
        <span className="text-[#23232e]">/</span>
        <span className="text-sm text-[#9090a8]">{ticker}</span>
      </div>

      {/* Ticker header */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="font-mono text-3xl font-bold tracking-tight text-[#e8e8f0]">
              {ticker}
            </h1>
            {prediction && !isLoading && (
              <SignalBadge signal={prediction.signal} size="lg" />
            )}
          </div>
          <p className="mt-1 text-sm text-[#6b6b80]">{companyName}</p>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-2">
          {isQuant && (
            <>
              <Select
                value={model}
                onValueChange={(v) => {
                  setModel(v as ForecastModel);
                  // If user selects CNN-T manually, turn off compare mode
                  if (v === COMPARE_MODEL) setCompareMode(false);
                }}
              >
                <SelectTrigger className="w-44 border-[#23232e] bg-[#111118] text-xs text-[#e8e8f0]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {MODELS.map((m) => (
                    <SelectItem key={m.value} value={m.value} className="text-xs">
                      {m.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              {/* Compare toggle — only when primary is not already CNN-T */}
              {model !== COMPARE_MODEL && (
                <Button
                  variant={compareMode ? 'default' : 'outline'}
                  size="sm"
                  className={`gap-1.5 text-xs ${
                    compareMode
                      ? 'border-[#f0c040]/40 bg-[#f0c040]/10 text-[#f0c040] hover:bg-[#f0c040]/20'
                      : 'border-[#23232e] bg-[#111118] text-[#6b6b80] hover:text-[#e8e8f0]'
                  }`}
                  onClick={() => setCompareMode((c) => !c)}
                >
                  <GitCompare className="h-3 w-3" />
                  {compareMode ? 'Comparing' : 'Compare'}
                </Button>
              )}
            </>
          )}

          <Select
            value={String(horizon)}
            onValueChange={(v) => setHorizon(Number(v) as ForecastHorizon)}
          >
            <SelectTrigger className="w-28 border-[#23232e] bg-[#111118] text-xs text-[#e8e8f0]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {HORIZONS.map((h) => (
                <SelectItem key={h.value} value={String(h.value)} className="text-xs">
                  {h.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Button
            variant="outline"
            size="sm"
            className="border-[#23232e] bg-[#111118] text-[#6b6b80] hover:text-[#e8e8f0] text-xs gap-1.5"
            onClick={() => mutate()}
          >
            <RefreshCw className="h-3 w-3" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Main content */}
      {isLoading ? (
        <div className="space-y-4">
          <Skeleton className="h-48 w-full rounded-xl" />
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-20 rounded-xl" />
            ))}
          </div>
        </div>
      ) : error ? (
        <div className="rounded-xl border border-[#23232e] bg-[#111118] p-8 text-center">
          <p className="text-sm text-[#6b6b80]">
            Could not load forecast for{' '}
            <span className="font-mono text-[#e8e8f0]">{ticker}</span>.
          </p>
          <p className="mt-1 text-xs text-[#6b6b80]">
            The backend may not have a trained checkpoint for this model.
          </p>
        </div>
      ) : prediction ? (
        <div className="space-y-6">
          {/* Prediction overview */}
          {!isQuant ? (
            // Retail: plain-English card
            <div className="rounded-xl border border-[#1a1a24] bg-[#111118] p-6">
              <p className="text-base leading-relaxed text-[#9090a8]">
                The{' '}
                <span className="text-[#e8e8f0] font-medium">
                  {modelLabel(model)}
                </span>{' '}
                model predicts a{' '}
                <span
                  className={
                    prediction.predicted_return >= 0
                      ? 'text-[#f0c040] font-semibold'
                      : 'text-[#ef4444] font-semibold'
                  }
                >
                  {prediction.predicted_return >= 0 ? 'positive' : 'negative'}
                </span>{' '}
                return for{' '}
                <span className="font-mono text-[#e8e8f0]">{ticker}</span>{' '}
                over the next {horizon === 1 ? 'trading day' : `${horizon} days`}.
                The signal is{' '}
                <span
                  className={
                    prediction.signal === 'long'
                      ? 'text-[#f0c040] font-semibold'
                      : 'text-[#9090a8] font-semibold'
                  }
                >
                  {prediction.signal === 'long' ? 'LONG' : 'FLAT'}
                </span>
                .
              </p>
              <div className="mt-5 space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-[#6b6b80]">Predicted return</span>
                  <span
                    className={`font-mono font-semibold tabular-nums ${
                      prediction.predicted_return >= 0
                        ? 'text-[#f0c040]'
                        : 'text-[#ef4444]'
                    }`}
                  >
                    {formatReturn(prediction.predicted_return)}
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-[#6b6b80]">Confidence</span>
                  <span className="font-mono text-[#e8e8f0] tabular-nums">
                    {Math.round(prediction.confidence * 100)}%
                  </span>
                </div>
                <div className="mt-3">
                  <ConfidenceBar value={prediction.confidence} />
                </div>
              </div>
            </div>
          ) : showCompare ? (
            // Quant + compare mode: two-column stat grid
            <CompareStatGrid
              primaryModel={model}
              primaryPrediction={prediction}
              comparePrediction={comparePrediction}
              horizon={horizon}
            />
          ) : (
            // Quant single model: stat grid
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <StatCard
                label="Pred Return"
                value={formatReturn(prediction.predicted_return)}
                valueColor={
                  prediction.predicted_return >= 0 ? '#f0c040' : '#ef4444'
                }
              />
              <StatCard
                label="Confidence"
                value={`${(prediction.confidence * 100).toFixed(2)}%`}
              />
              <StatCard
                label="Model"
                value={modelLabel(model)}
                mono={false}
              />
              <StatCard
                label="Horizon"
                value={`${horizon}d`}
              />
              {metrics && (
                <>
                  <StatCard
                    label="Dir Accuracy"
                    value={`${(metrics.directional_accuracy * 100).toFixed(1)}%`}
                  />
                  <StatCard
                    label="Sharpe"
                    value={formatSigned(metrics.sharpe, 3)}
                    valueColor={metrics.sharpe > 0 ? '#22c55e' : '#ef4444'}
                  />
                  <StatCard
                    label="Sortino"
                    value={formatSigned(metrics.sortino, 3)}
                    valueColor={metrics.sortino > 0 ? '#22c55e' : '#ef4444'}
                  />
                  <StatCard
                    label="Max Drawdown"
                    value={`${(metrics.max_drawdown * 100).toFixed(2)}%`}
                    valueColor="#ef4444"
                  />
                </>
              )}
            </div>
          )}

          {/* Chart */}
          <div className="rounded-xl border border-[#1a1a24] bg-[#111118] p-5">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <h3 className="text-sm font-medium text-[#e8e8f0]">
                  Return Forecast
                </h3>
                <p className="text-xs text-[#6b6b80]">
                  Predicted log return ·{' '}
                  {showCompare
                    ? `${modelShortLabel(model)} vs ${modelShortLabel(COMPARE_MODEL)}`
                    : modelLabel(model)}
                </p>
              </div>
            </div>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={chartData}
                  margin={{ top: 8, right: 8, left: -20, bottom: 0 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="#1a1a24"
                    vertical={false}
                  />
                  <XAxis
                    dataKey="date"
                    tick={{ fill: '#6b6b80', fontSize: 11, fontFamily: 'DM Mono, monospace' }}
                    axisLine={{ stroke: '#23232e' }}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fill: '#6b6b80', fontSize: 11, fontFamily: 'DM Mono, monospace' }}
                    axisLine={false}
                    tickLine={false}
                    tickFormatter={(v: number) => `${(v * 100).toFixed(1)}%`}
                  />
                  <Tooltip content={<ChartTooltip />} />
                  <ReferenceLine
                    y={0}
                    stroke="#23232e"
                    strokeDasharray="4 4"
                  />
                  {showCompare && <Legend
                    formatter={(value: string) => (
                      <span style={{ color: '#9090a8', fontSize: '11px' }}>{value}</span>
                    )}
                  />}
                  {/* Primary model line */}
                  <Line
                    type="monotone"
                    dataKey="primary"
                    name={modelShortLabel(model)}
                    stroke={primaryColor}
                    strokeWidth={2}
                    dot={(props: { cx?: number; cy?: number; payload: { forecast?: boolean } }) => {
                      const { cx, cy, payload } = props;
                      if (!payload.forecast || cx == null || cy == null) return <g key={`dot-primary-${cx}-${cy}`} />;
                      return (
                        <circle
                          key={`dot-primary-${cx}-${cy}`}
                          cx={cx}
                          cy={cy}
                          r={5}
                          fill={primaryColor}
                          stroke="#09090f"
                          strokeWidth={2}
                        />
                      );
                    }}
                    activeDot={{ r: 4, stroke: '#09090f', strokeWidth: 2 }}
                  />
                  {/* Compare model line — dashed, blue */}
                  {showCompare && (
                    <Line
                      type="monotone"
                      dataKey="compare"
                      name={modelShortLabel(COMPARE_MODEL)}
                      stroke={compareColor}
                      strokeWidth={2}
                      strokeDasharray="5 3"
                      dot={(props: { cx?: number; cy?: number; payload: { forecast?: boolean } }) => {
                        const { cx, cy, payload } = props;
                        if (!payload.forecast || cx == null || cy == null) return <g key={`dot-compare-${cx}-${cy}`} />;
                        return (
                          <circle
                            key={`dot-compare-${cx}-${cy}`}
                            cx={cx}
                            cy={cy}
                            r={5}
                            fill={compareColor}
                            stroke="#09090f"
                            strokeWidth={2}
                          />
                        );
                      }}
                      activeDot={{ r: 4, stroke: '#09090f', strokeWidth: 2 }}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="mt-3 text-[11px] text-[#6b6b80]">
              {showCompare
                ? `Solid line: ${modelShortLabel(model)} · dashed line: ${modelShortLabel(COMPARE_MODEL)}. Historical values are baseline zero.`
                : 'The chart shows the predicted return for tomorrow. Historical values are baseline zero — only the forecast point is meaningful.'}
            </p>
          </div>

          {/* Quant: full metrics table */}
          {isQuant && metrics && (
            <div className="rounded-xl border border-[#1a1a24] bg-[#111118] p-5">
              <h3 className="mb-4 text-sm font-medium text-[#e8e8f0]">
                Evaluation Metrics
              </h3>
              <table className="w-full">
                <thead>
                  <tr className="border-b border-[#1a1a24]">
                    {['Metric', 'Value', 'Description'].map((h) => (
                      <th
                        key={h}
                        className="pb-2 text-left text-[10px] uppercase tracking-widest text-[#6b6b80]"
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1a1a24]">
                  {[
                    {
                      metric: 'MAE',
                      value: metrics.mae.toFixed(6),
                      desc: 'Mean Absolute Error of log returns',
                    },
                    {
                      metric: 'RMSE',
                      value: metrics.rmse.toFixed(6),
                      desc: 'Root Mean Squared Error',
                    },
                    {
                      metric: 'Directional Acc',
                      value: `${(metrics.directional_accuracy * 100).toFixed(2)}%`,
                      desc: 'Fraction of correct direction predictions',
                    },
                    {
                      metric: 'Sharpe',
                      value: formatSigned(metrics.sharpe, 4),
                      desc: 'Annualised Sharpe ratio of signal returns',
                    },
                    {
                      metric: 'Sortino',
                      value: formatSigned(metrics.sortino, 4),
                      desc: 'Sortino ratio (downside deviation only)',
                    },
                    {
                      metric: 'Max Drawdown',
                      value: `${(metrics.max_drawdown * 100).toFixed(2)}%`,
                      desc: 'Peak-to-trough equity drawdown',
                    },
                  ].map(({ metric, value, desc }) => (
                    <tr key={metric}>
                      <td className="py-2.5 text-xs text-[#e8e8f0]">{metric}</td>
                      <td className="py-2.5 font-mono text-xs tabular-nums text-[#f0c040]">
                        {value}
                      </td>
                      <td className="py-2.5 text-xs text-[#6b6b80]">{desc}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}
