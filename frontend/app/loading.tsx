import { Skeleton } from '@/components/ui/skeleton';

export default function Loading() {
  return (
    <main className="mx-auto max-w-7xl px-6 py-8">
      {/* Page header skeleton */}
      <div className="mb-8 flex items-center justify-between">
        <div className="space-y-2">
          <Skeleton className="h-7 w-48" />
          <Skeleton className="h-4 w-72" />
        </div>
        {/* Metrics bar skeleton */}
        <div className="hidden items-center gap-6 md:flex">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="space-y-1 text-right">
              <Skeleton className="h-3 w-16" />
              <Skeleton className="h-5 w-20" />
            </div>
          ))}
        </div>
      </div>

      {/* Ticker grid skeleton */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <div
            key={i}
            className="rounded-xl border border-[#1a1a24] bg-[#111118] p-5 space-y-4"
          >
            {/* Card header */}
            <div className="flex items-center justify-between">
              <div className="space-y-1.5">
                <Skeleton className="h-5 w-14" />
                <Skeleton className="h-3 w-28" />
              </div>
              <Skeleton className="h-6 w-16 rounded-full" />
            </div>
            {/* Confidence bar area */}
            <div className="space-y-1.5">
              <div className="flex justify-between">
                <Skeleton className="h-3 w-20" />
                <Skeleton className="h-3 w-10" />
              </div>
              <Skeleton className="h-1.5 w-full rounded-full" />
            </div>
            {/* Stats row */}
            <div className="grid grid-cols-3 gap-2 pt-1">
              {Array.from({ length: 3 }).map((_, j) => (
                <div key={j} className="space-y-1">
                  <Skeleton className="h-3 w-12" />
                  <Skeleton className="h-4 w-16" />
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </main>
  );
}
