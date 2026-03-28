'use client';

import { Fragment, useState } from 'react';
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
import {
  fetchTickerForecast,
  fetchMetrics,
  fetchHistoricalReturns,
  swrKeys,
} from '@/lib/api';
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
import { Separator } from '@/components/ui/separator';
import {
  Tooltip as UITooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { ArrowLeft, RefreshCw, GitCompare, Info } from 'lucide-react';
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

// The two secondary models shown in compare mode
const COMPARE_MODELS: [ForecastModel, ForecastModel] = ['cnn_transformer', 'mamba_ssm'];

// Metric descriptions for tooltips
const METRIC_DESCRIPTIONS: Record<string, string> = {
  MAE: 'Mean Absolute Error — average magnitude of prediction errors on log returns',
  RMSE: 'Root Mean Squared Error — penalises large errors more than MAE',
  'Directional Acc': 'Fraction of days where the model correctly predicted the sign of the return',
  'Dir Accuracy': 'Fraction of days where the model correctly predicted the sign of the return',
  Sharpe: 'Annualised Sharpe ratio of signal-following returns (higher is better)',
  Sortino: 'Like Sharpe but only penalises downside volatility',
  'Max Drawdown': 'Largest peak-to-trough equity decline during the backtest period',
  Confidence: 'Model certainty score — derived from the magnitude of the predicted return relative to historical distribution',
  'Pred Return': 'Predicted log return for the next trading period',
};

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

// Labelled metric value with optional tooltip
function MetricLabel({
  label,
  description,
}: {
  label: string;
  description?: string;
}) {
  const desc = description ?? METRIC_DESCRIPTIONS[label];
  if (!desc) {
    return (
      <p className="text-[10px] uppercase tracking-widest text-[#6b6b80]">{label}</p>
    );
  }
  return (
    <UITooltip>
      <TooltipTrigger asChild>
        <p className="flex cursor-default items-center gap-1 text-[10px] uppercase tracking-widest text-[#6b6b80]">
          {label}
          <Info className="h-2.5 w-2.5 shrink-0 opacity-50" />
        </p>
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-56">
        {desc}
      </TooltipContent>
    </UITooltip>
  );
}

// A single stat card
function StatCard({
  label,
  value,
  valueColor,
  mono = true,
  highlight = false,
  tooltipDescription,
}: {
  label: string;
  value: string;
  valueColor?: string;
  mono?: boolean;
  highlight?: boolean;
  tooltipDescription?: string;
}) {
  return (
    <div
      className={`rounded-lg border bg-[#111118] p-4 ${
        highlight ? 'border-[#f0c040]/50 ring-1 ring-[#f0c040]/40' : 'border-[#1a1a24]'
      }`}
    >
      <MetricLabel label={label} description={tooltipDescription} />
      <p
        className={`mt-1 text-sm font-semibold ${mono ? 'font-mono tabular-nums' : ''}`}
        style={{ color: valueColor ?? '#e8e8f0' }}
      >
        {value}
      </p>
    </div>
  );
}

// Three-column comparison stat grid: LSTM / CNN-T / Mamba
function CompareStatGrid({
  primaryModel,
  primaryPrediction,
  compareA,
  compareB,
  horizon,
}: {
  primaryModel: ForecastModel;
  primaryPrediction: DailyForecastPoint;
  compareA: DailyForecastPoint | null;
  compareB: DailyForecastPoint | null;
  horizon: ForecastHorizon;
}) {
  // Determine which column has the highest confidence
  const confidences = [
    primaryPrediction.confidence,
    compareA?.confidence ?? -1,
    compareB?.confidence ?? -1,
  ];
  const maxIdx = confidences.indexOf(Math.max(...confidences));

  const columns: Array<{
    model: ForecastModel;
    prediction: DailyForecastPoint | null;
    highlight: boolean;
  }> = [
    { model: primaryModel, prediction: primaryPrediction, highlight: maxIdx === 0 },
    { model: COMPARE_MODELS[0], prediction: compareA, highlight: maxIdx === 1 },
    { model: COMPARE_MODELS[1], prediction: compareB, highlight: maxIdx === 2 },
  ];

  return (
    <div className="flex flex-col gap-4 sm:flex-row">
      {columns.map((col, idx) => (
        <>
          {idx > 0 && (
            <Separator
              key={`sep-${col.model}`}
              orientation="vertical"
              className="hidden self-stretch sm:block"
            />
          )}
          <div key={col.model} className="flex-1 space-y-2">
            <p className="mb-2 text-center text-[10px] uppercase tracking-widest text-[#f0c040]">
              {modelShortLabel(col.model)}
            </p>
            {col.prediction ? (
              <>
                <StatCard
                  label="Pred Return"
                  value={formatReturn(col.prediction.predicted_return)}
                  valueColor={col.prediction.predicted_return >= 0 ? '#f0c040' : '#ef4444'}
                  highlight={col.highlight}
                />
                <StatCard
                  label="Confidence"
                  value={`${(col.prediction.confidence * 100).toFixed(2)}%`}
                  highlight={col.highlight}
                />
                <StatCard
                  label="Signal"
                  value={col.prediction.signal.toUpperCase()}
                  valueColor={col.prediction.signal === 'long' ? '#f0c040' : '#9090a8'}
                  highlight={col.highlight}
                />
                <StatCard label="Horizon" value={`${horizon}d`} highlight={col.highlight} />
              </>
            ) : (
              <div className="space-y-2">
                {Array.from({ length: 4 }).map((_, i) => (
                  <Skeleton key={i} className="h-[60px] rounded-lg" />
                ))}
              </div>
            )}
          </div>
        </>
      ))}
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
  const { data: forecast, error, isLoading, mutate } = useSWR(
    swrKeys.forecast(ticker, model, horizon),
    () => fetchTickerForecast(ticker, model, horizon),
    { refreshInterval: 5 * 60 * 1000 },
  );

  // Compare model SWRs — only active when compareMode is on
  const showCompare = isQuant && compareMode;
  const { data: compareForecastA } = useSWR(
    showCompare ? swrKeys.forecast(ticker, COMPARE_MODELS[0], horizon) : null,
    () => fetchTickerForecast(ticker, COMPARE_MODELS[0], horizon),
    { refreshInterval: 5 * 60 * 1000 },
  );
  const { data: compareForecastB } = useSWR(
    showCompare ? swrKeys.forecast(ticker, COMPARE_MODELS[1], horizon) : null,
    () => fetchTickerForecast(ticker, COMPARE_MODELS[1], horizon),
    { refreshInterval: 5 * 60 * 1000 },
  );

  // Historical returns — quant mode fetches rolling vol too
  const { data: historicalReturns } = useSWR(
    swrKeys.historicalReturns(ticker, isQuant, 504),
    () => fetchHistoricalReturns(ticker, isQuant, 504),
    { refreshInterval: 0 },
  );

  const { data: metrics } = useSWR(
    swrKeys.metrics(),
    () => fetchMetrics(),
  );

  const prediction = forecast?.predictions?.[0];
  const comparePredictionA = compareForecastA?.predictions?.[0] ?? null;
  const comparePredictionB = compareForecastB?.predictions?.[0] ?? null;
  const companyName = TICKER_NAMES[ticker] ?? ticker;

  // Build chart data from real historical returns, capped at last 60 points for readability
  const chartData = (() => {
    const recent = historicalReturns?.returns.slice(-60) ?? [];
    const points = recent.map((pt) => ({
      date: pt.date,
      primary: pt.log_return,
      vol: undefined as number | undefined,
    }));

    // Merge rolling volatility if present (quant mode)
    if (isQuant && historicalReturns?.rolling_volatility) {
      const volMap = new Map(historicalReturns.rolling_volatility.map((v) => [v.date, v.log_return]));
      for (const pt of points) {
        const v = volMap.get(pt.date);
        if (v !== undefined) pt.vol = v;
      }
    }

    // Append forecast point
    if (prediction) {
      points.push({
        date: 'Forecast',
        primary: prediction.predicted_return,
        vol: undefined,
      });
    }

    return points;
  })();

  const primaryColor = prediction
    ? prediction.predicted_return >= 0
      ? '#f0c040'
      : '#ef4444'
    : '#f0c040';

  const compareColorA = comparePredictionA
    ? comparePredictionA.predicted_return >= 0 ? '#60a5fa' : '#f97316'
    : '#60a5fa';

  const compareColorB = '#a78bfa';

  const hasHistory = historicalReturns && historicalReturns.returns.length > 0;

  return (
    <TooltipProvider delayDuration={300}>
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
                <Separator className="my-5" />
                <div className="space-y-2">
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
              // Quant + compare mode: three-column stat grid
              <CompareStatGrid
                primaryModel={model}
                primaryPrediction={prediction}
                compareA={comparePredictionA}
                compareB={comparePredictionB}
                horizon={horizon}
              />
            ) : (
              // Quant single model: stat grid
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <StatCard
                  label="Pred Return"
                  value={formatReturn(prediction.predicted_return)}
                  valueColor={prediction.predicted_return >= 0 ? '#f0c040' : '#ef4444'}
                />
                <StatCard
                  label="Confidence"
                  value={`${(prediction.confidence * 100).toFixed(2)}%`}
                />
                <StatCard label="Model" value={modelLabel(model)} mono={false} />
                <StatCard label="Horizon" value={`${horizon}d`} />
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
                    {hasHistory ? 'Historical Returns + Forecast' : 'Return Forecast'}
                  </h3>
                  <p className="text-xs text-[#6b6b80]">
                    {hasHistory
                      ? `Daily log return · last 60 sessions · ${modelLabel(model)} forecast`
                      : `Predicted log return · ${modelLabel(model)}`}
                    {isQuant && historicalReturns?.rolling_volatility && ' · dashed: 30-day rolling vol'}
                  </p>
                </div>
              </div>
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={chartData}
                    margin={{ top: 8, right: 8, left: -20, bottom: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#1a1a24" vertical={false} />
                    <XAxis
                      dataKey="date"
                      tick={{ fill: '#6b6b80', fontSize: 10, fontFamily: 'DM Mono, monospace' }}
                      axisLine={{ stroke: '#23232e' }}
                      tickLine={false}
                      interval="preserveStartEnd"
                    />
                    <YAxis
                      tick={{ fill: '#6b6b80', fontSize: 10, fontFamily: 'DM Mono, monospace' }}
                      axisLine={false}
                      tickLine={false}
                      tickFormatter={(v: number) => `${(v * 100).toFixed(1)}%`}
                    />
                    <Tooltip content={<ChartTooltip />} />
                    <ReferenceLine y={0} stroke="#23232e" strokeDasharray="4 4" />
                    {showCompare && (
                      <Legend
                        formatter={(value: string) => (
                          <span style={{ color: '#9090a8', fontSize: '11px' }}>{value}</span>
                        )}
                      />
                    )}
                    {/* Historical + forecast returns */}
                    <Line
                      type="monotone"
                      dataKey="primary"
                      name={modelShortLabel(model)}
                      stroke={primaryColor}
                      strokeWidth={1.5}
                      dot={false}
                      activeDot={{ r: 3, stroke: '#09090f', strokeWidth: 2 }}
                    />
                    {/* Rolling volatility (quant mode) */}
                    {isQuant && historicalReturns?.rolling_volatility && (
                      <Line
                        type="monotone"
                        dataKey="vol"
                        name="30d Vol"
                        stroke="#6b6b80"
                        strokeWidth={1}
                        strokeDasharray="4 2"
                        dot={false}
                        activeDot={false}
                      />
                    )}
                    {/* Compare model A */}
                    {showCompare && comparePredictionA && (
                      <Line
                        type="monotone"
                        dataKey="compareA"
                        name={modelShortLabel(COMPARE_MODELS[0])}
                        stroke={compareColorA}
                        strokeWidth={1.5}
                        strokeDasharray="5 3"
                        dot={false}
                        activeDot={{ r: 3, stroke: '#09090f', strokeWidth: 2 }}
                      />
                    )}
                    {/* Compare model B */}
                    {showCompare && comparePredictionB && (
                      <Line
                        type="monotone"
                        dataKey="compareB"
                        name={modelShortLabel(COMPARE_MODELS[1])}
                        stroke={compareColorB}
                        strokeWidth={1.5}
                        strokeDasharray="2 2"
                        dot={false}
                        activeDot={{ r: 3, stroke: '#09090f', strokeWidth: 2 }}
                      />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <p className="mt-3 text-[11px] text-[#6b6b80]">
                {hasHistory
                  ? 'Gold line: actual daily log returns. Last data point is the model forecast for the next session.'
                  : 'Only the forecast point is meaningful — historical data unavailable.'}
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
                    <tr>
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
                  <tbody>
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
                    ].map(({ metric, value, desc }, rowIdx) => (
                      <Fragment key={metric}>
                        {rowIdx > 0 && (
                          <tr key={`sep-${metric}`}>
                            <td colSpan={3} className="p-0">
                              <Separator />
                            </td>
                          </tr>
                        )}
                        <tr>
                          <td className="py-2.5 text-xs text-[#e8e8f0]">
                            <MetricLabel label={metric} description={METRIC_DESCRIPTIONS[metric]} />
                          </td>
                          <td className="py-2.5 font-mono text-xs tabular-nums text-[#f0c040]">
                            {value}
                          </td>
                          <td className="py-2.5 text-xs text-[#6b6b80]">{desc}</td>
                        </tr>
                      </Fragment>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        ) : null}
      </div>
    </TooltipProvider>
  );
}
