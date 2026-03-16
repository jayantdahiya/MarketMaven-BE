import Link from 'next/link';
import { TrendingUp } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function NotFound() {
  return (
    <div className="flex min-h-[70vh] flex-col items-center justify-center gap-6 px-6 text-center">
      {/* Amber accent number */}
      <div className="relative select-none">
        <span className="font-mono text-[8rem] font-bold leading-none tracking-tighter text-[#f0c040]/10">
          404
        </span>
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl border border-[#f0c040]/20 bg-[#f0c040]/10">
            <TrendingUp className="h-6 w-6 text-[#f0c040]" />
          </div>
        </div>
      </div>

      {/* Copy */}
      <div className="space-y-2">
        <h1 className="text-xl font-semibold text-[#e8e8f0]">Page not found</h1>
        <p className="max-w-sm text-sm text-[#6b6b80]">
          This ticker doesn&apos;t exist in our universe. Head back to the
          dashboard to find what you&apos;re looking for.
        </p>
      </div>

      {/* Action */}
      <Button
        asChild
        size="sm"
        className="bg-[#f0c040] text-[#09090f] hover:bg-[#f0c040]/90"
      >
        <Link href="/">Back to Dashboard</Link>
      </Button>
    </div>
  );
}
