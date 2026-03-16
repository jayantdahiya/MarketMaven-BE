'use client';

import { clamp } from '@/lib/utils';

interface ConfidenceBarProps {
  value: number; // 0–1
  showLabel?: boolean;
}

export function ConfidenceBar({ value, showLabel = true }: ConfidenceBarProps) {
  const pct = clamp(value * 100, 0, 100);

  return (
    <div className="flex items-center gap-2">
      <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-[#18181f]">
        <div
          className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-[#f0c040]/60 to-[#f0c040] transition-all duration-700 ease-out"
          style={{ width: `${pct}%` }}
        />
      </div>
      {showLabel && (
        <span className="w-9 text-right font-mono text-xs text-[#9090a8] tabular-nums">
          {Math.round(pct)}%
        </span>
      )}
    </div>
  );
}
