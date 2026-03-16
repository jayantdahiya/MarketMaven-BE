'use client';

import { useEffect } from 'react';
import { AlertTriangle } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ErrorProps {
  error: Error & { digest?: string };
  reset: () => void;
}

export default function Error({ error, reset }: ErrorProps) {
  useEffect(() => {
    // Surface in browser console for debugging — never silently swallow
    console.error('[MarketMaven error boundary]', error);
  }, [error]);

  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center gap-6 px-6">
      {/* Icon */}
      <div className="flex h-14 w-14 items-center justify-center rounded-full border border-[#ef4444]/20 bg-[#ef4444]/10">
        <AlertTriangle className="h-6 w-6 text-[#ef4444]" />
      </div>

      {/* Copy */}
      <div className="text-center">
        <p className="text-xs font-medium uppercase tracking-widest text-[#ef4444]">
          Something went wrong
        </p>
        <h1 className="mt-2 text-xl font-semibold text-[#e8e8f0]">
          An unexpected error occurred
        </h1>
        {error.message && (
          <p className="mt-2 font-mono text-xs text-[#6b6b80]">
            {error.message}
          </p>
        )}
        {error.digest && (
          <p className="mt-1 font-mono text-[10px] text-[#6b6b80]/60">
            digest: {error.digest}
          </p>
        )}
      </div>

      {/* Action */}
      <Button
        onClick={reset}
        variant="outline"
        size="sm"
        className="border-[#23232e] text-[#e8e8f0] hover:bg-[#18181f] hover:text-[#f0c040]"
      >
        Try again
      </Button>
    </div>
  );
}
