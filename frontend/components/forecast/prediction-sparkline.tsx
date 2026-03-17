'use client';

import {
  LineChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';
import type { PredictionPoint } from '@/types/api';
import { formatReturn } from '@/lib/utils';

interface SparklineTooltipProps {
  active?: boolean;
  payload?: Array<{
    value: number;
    name: string;
    color: string;
    payload: { date: string };
  }>;
}

function SparklineTooltip({ active, payload }: SparklineTooltipProps) {
  if (!active || !payload?.length) return null;
  const date = payload[0]?.payload?.date;
  return (
    <div className="rounded-md border border-[#23232e] bg-[#111118] px-2.5 py-1.5 shadow-xl">
      <p className="mb-0.5 text-[10px] text-[#6b6b80]">{date}</p>
      {payload.map((p) => (
        <p
          key={p.name}
          className="font-mono text-[11px] tabular-nums"
          style={{ color: p.color }}
        >
          {p.name}: {formatReturn(p.value)}
        </p>
      ))}
    </div>
  );
}

interface PredictionSparklineProps {
  predictions: PredictionPoint[];
  /** Height of the sparkline container in px. Default: 72. */
  height?: number;
}

/**
 * Compact sparkline showing predicted vs actual log-returns for a backtest
 * window. Predicted line is gold, actual line is muted blue.
 * No axes are rendered to keep it minimal — only a zero reference line.
 */
export function PredictionSparkline({
  predictions,
  height = 72,
}: PredictionSparklineProps) {
  if (!predictions.length) return null;

  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={predictions} margin={{ top: 4, right: 4, left: 4, bottom: 4 }}>
          <Tooltip content={<SparklineTooltip />} />
          <ReferenceLine y={0} stroke="#23232e" strokeDasharray="3 3" />
          <Line
            type="monotone"
            dataKey="predicted_return"
            name="Predicted"
            stroke="#f0c040"
            strokeWidth={1.5}
            dot={false}
            activeDot={{ r: 3, stroke: '#09090f', strokeWidth: 1.5 }}
          />
          <Line
            type="monotone"
            dataKey="actual_return"
            name="Actual"
            stroke="#60a5fa"
            strokeWidth={1.5}
            dot={false}
            activeDot={{ r: 3, stroke: '#09090f', strokeWidth: 1.5 }}
            strokeOpacity={0.6}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
