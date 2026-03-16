'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Switch } from '@/components/ui/switch';
import { useUIMode } from '@/hooks/use-ui-mode';
import { cn } from '@/lib/utils';
import { TrendingUp, BarChart2, Activity, LogIn } from 'lucide-react';

export function Navbar() {
  const { setMode, isQuant } = useUIMode();
  const pathname = usePathname();

  const navItems = [
    { href: '/', label: 'Dashboard', icon: BarChart2 },
  ];

  return (
    <header className="sticky top-0 z-50 border-b border-[#1a1a24] bg-[#09090f]/90 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-6">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-[#f0c040]/10 border border-[#f0c040]/20 group-hover:bg-[#f0c040]/15 transition-colors">
            <TrendingUp className="h-4 w-4 text-[#f0c040]" />
          </div>
          <span className="text-sm font-semibold tracking-tight text-[#e8e8f0]">
            Market<span className="text-[#f0c040]">Maven</span>
          </span>
        </Link>

        {/* Nav Links */}
        <nav className="hidden items-center gap-1 md:flex">
          {navItems.map(({ href, label, icon: Icon }) => (
            <Link
              key={href}
              href={href}
              className={cn(
                'flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors',
                pathname === href
                  ? 'bg-[#f0c040]/10 text-[#f0c040]'
                  : 'text-[#6b6b80] hover:text-[#e8e8f0] hover:bg-[#18181f]',
              )}
            >
              <Icon className="h-3.5 w-3.5" />
              {label}
            </Link>
          ))}
        </nav>

        {/* Right side: Sign in + Mode Toggle */}
        <div className="flex items-center gap-3">
          {/* Sign in link */}
          <Link
            href="/auth/login"
            className="hidden items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium text-[#6b6b80] transition-colors hover:bg-[#18181f] hover:text-[#e8e8f0] md:flex"
          >
            <LogIn className="h-3.5 w-3.5" />
            Sign in
          </Link>

          {/* Mode toggle */}
          <div className="flex items-center gap-2 rounded-full border border-[#23232e] bg-[#111118] px-3 py-1.5">
            <Activity className="h-3 w-3 text-[#6b6b80]" />
            <span className={cn(
              'text-xs transition-colors',
              !isQuant ? 'text-[#e8e8f0] font-medium' : 'text-[#6b6b80]'
            )}>
              Simple
            </span>
            <Switch
              checked={isQuant}
              onCheckedChange={(checked) => setMode(checked ? 'quant' : 'retail')}
              className="h-4 w-7 data-[state=checked]:bg-[#f0c040]"
            />
            <span className={cn(
              'text-xs transition-colors',
              isQuant ? 'text-[#f0c040] font-medium' : 'text-[#6b6b80]'
            )}>
              Advanced
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
