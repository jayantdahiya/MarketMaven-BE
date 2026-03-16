import { formatSigned, modelShortLabel } from '@/lib/utils';
import type { ForecastModel, MetricsSummaryResponse } from '@/types/api';

interface ModelComparisonTableProps {
  models: ForecastModel[];
  metrics: Partial<Record<ForecastModel, MetricsSummaryResponse>>;
}

interface MetricRow {
  key: keyof MetricsSummaryResponse;
  label: string;
  /** Higher is better? Used to colour the delta. */
  higherIsBetter: boolean;
  format: (v: number) => string;
}

const ROWS: MetricRow[] = [
  {
    key: 'mae',
    label: 'MAE',
    higherIsBetter: false,
    format: (v) => v.toFixed(6),
  },
  {
    key: 'rmse',
    label: 'RMSE',
    higherIsBetter: false,
    format: (v) => v.toFixed(6),
  },
  {
    key: 'directional_accuracy',
    label: 'Dir Accuracy',
    higherIsBetter: true,
    format: (v) => `${(v * 100).toFixed(2)}%`,
  },
  {
    key: 'sharpe',
    label: 'Sharpe',
    higherIsBetter: true,
    format: (v) => formatSigned(v, 3),
  },
  {
    key: 'sortino',
    label: 'Sortino',
    higherIsBetter: true,
    format: (v) => formatSigned(v, 3),
  },
  {
    key: 'max_drawdown',
    label: 'Max Drawdown',
    higherIsBetter: false,
    format: (v) => `${(v * 100).toFixed(2)}%`,
  },
];

/** Delta between second model and first model (second − first). */
function DeltaCell({
  first,
  second,
  row,
}: {
  first: number | undefined;
  second: number | undefined;
  row: MetricRow;
}) {
  if (first === undefined || second === undefined) {
    return <td className="py-2 text-right font-mono text-xs tabular-nums text-[#6b6b80]">—</td>;
  }
  const delta = second - first;
  // For lower-is-better metrics, a negative delta is good (second model is better).
  const isGood = row.higherIsBetter ? delta > 0 : delta < 0;
  const sign = delta >= 0 ? '+' : '';
  const colorClass = delta === 0 ? 'text-[#9090a8]' : isGood ? 'text-[#22c55e]' : 'text-[#ef4444]';
  return (
    <td className={`py-2 text-right font-mono text-xs tabular-nums ${colorClass}`}>
      {`${sign}${delta.toFixed(4)}`}
    </td>
  );
}

export function ModelComparisonTable({ models, metrics }: ModelComparisonTableProps) {
  const [baseModel, compareModel] = models;

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[480px]">
        <thead>
          <tr className="border-b border-[#1a1a24]">
            <th className="pb-2 text-left text-[10px] uppercase tracking-widest text-[#6b6b80]">
              Metric
            </th>
            {models.map((m) => (
              <th
                key={m}
                className="pb-2 text-right text-[10px] uppercase tracking-widest text-[#f0c040]"
              >
                {modelShortLabel(m)}
              </th>
            ))}
            {models.length >= 2 && (
              <th className="pb-2 text-right text-[10px] uppercase tracking-widest text-[#6b6b80]">
                Δ ({modelShortLabel(models[1])} − {modelShortLabel(models[0])})
              </th>
            )}
          </tr>
        </thead>
        <tbody className="divide-y divide-[#1a1a24]">
          {ROWS.map((row) => {
            const baseVal = metrics[baseModel]?.[row.key] as number | undefined;
            const cmpVal = compareModel ? (metrics[compareModel]?.[row.key] as number | undefined) : undefined;

            return (
              <tr key={row.key}>
                <td className="py-2 text-[11px] uppercase tracking-widest text-[#6b6b80]">
                  {row.label}
                </td>
                <td className="py-2 text-right font-mono text-xs tabular-nums text-[#e8e8f0]">
                  {baseVal !== undefined ? row.format(baseVal) : '—'}
                </td>
                {compareModel && (
                  <td className="py-2 text-right font-mono text-xs tabular-nums text-[#e8e8f0]">
                    {cmpVal !== undefined ? row.format(cmpVal) : '—'}
                  </td>
                )}
                {compareModel && (
                  <DeltaCell first={baseVal} second={cmpVal} row={row} />
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
