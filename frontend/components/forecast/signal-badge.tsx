'use client';

import { cn } from '@/lib/utils';
import type { ForecastSignal } from '@/types/api';

interface SignalBadgeProps {
  signal: ForecastSignal;
  size?: 'sm' | 'md' | 'lg';
}

export function SignalBadge({ signal, size = 'md' }: SignalBadgeProps) {
  const isLong = signal === 'long';

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full font-semibold uppercase tracking-widest',
        size === 'sm' && 'px-2 py-0.5 text-[10px]',
        size === 'md' && 'px-3 py-1 text-xs',
        size === 'lg' && 'px-4 py-1.5 text-sm',
        isLong
          ? 'bg-[#f0c040]/12 text-[#f0c040] border border-[#f0c040]/25 signal-long'
          : 'bg-[#6b6b80]/10 text-[#9090a8] border border-[#6b6b80]/20',
      )}
    >
      <span
        className={cn(
          'block rounded-full',
          size === 'sm' && 'h-1 w-1',
          size === 'md' && 'h-1.5 w-1.5',
          size === 'lg' && 'h-2 w-2',
          isLong ? 'bg-[#f0c040]' : 'bg-[#6b6b80]',
        )}
      />
      {isLong ? 'Long' : 'Flat'}
    </span>
  );
}
