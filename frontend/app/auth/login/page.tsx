'use client';

import { useState, type FormEvent } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { TrendingUp, Eye, EyeOff, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { loginUser } from '@/lib/api';

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: FormEvent<HTMLFormElement>): Promise<void> {
    e.preventDefault();
    setError(null);
    setIsLoading(true);
    try {
      await loginUser({ email, password });
      router.push('/');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="flex min-h-[80vh] items-center justify-center px-6">
      <div className="w-full max-w-sm">
        {/* Logo mark */}
        <div className="mb-8 flex flex-col items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-[#f0c040]/20 bg-[#f0c040]/10">
            <TrendingUp className="h-5 w-5 text-[#f0c040]" />
          </div>
          <div className="text-center">
            <h1 className="text-lg font-semibold text-[#e8e8f0]">Sign in</h1>
            <p className="mt-0.5 text-xs text-[#6b6b80]">
              Welcome back to MarketMaven
            </p>
          </div>
        </div>

        {/* Card */}
        <div className="rounded-xl border border-[#1a1a24] bg-[#111118] p-6 shadow-xl">
          <form onSubmit={handleSubmit} className="space-y-4" noValidate>
            {/* Email */}
            <div className="space-y-1.5">
              <label
                htmlFor="email"
                className="block text-xs font-medium text-[#9090a8]"
              >
                Email address
              </label>
              <input
                id="email"
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="w-full rounded-md border border-[#23232e] bg-[#09090f] px-3 py-2 text-sm text-[#e8e8f0] placeholder-[#6b6b80] outline-none transition-colors focus:border-[#f0c040]/40 focus:ring-1 focus:ring-[#f0c040]/20 disabled:opacity-50"
                disabled={isLoading}
              />
            </div>

            {/* Password */}
            <div className="space-y-1.5">
              <label
                htmlFor="password"
                className="block text-xs font-medium text-[#9090a8]"
              >
                Password
              </label>
              <div className="relative">
                <input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  autoComplete="current-password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full rounded-md border border-[#23232e] bg-[#09090f] px-3 py-2 pr-9 text-sm text-[#e8e8f0] placeholder-[#6b6b80] outline-none transition-colors focus:border-[#f0c040]/40 focus:ring-1 focus:ring-[#f0c040]/20 disabled:opacity-50"
                  disabled={isLoading}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((v) => !v)}
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 text-[#6b6b80] hover:text-[#9090a8] transition-colors"
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                >
                  {showPassword ? (
                    <EyeOff className="h-3.5 w-3.5" />
                  ) : (
                    <Eye className="h-3.5 w-3.5" />
                  )}
                </button>
              </div>
            </div>

            {/* Error */}
            {error && (
              <p className="rounded-md border border-[#ef4444]/20 bg-[#ef4444]/10 px-3 py-2 text-xs text-[#ef4444]">
                {error}
              </p>
            )}

            {/* Submit */}
            <Button
              type="submit"
              disabled={isLoading || !email || !password}
              className="w-full bg-[#f0c040] text-[#09090f] hover:bg-[#f0c040]/90 disabled:opacity-40"
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  Signing in…
                </>
              ) : (
                'Sign in'
              )}
            </Button>
          </form>
        </div>

        {/* Footer link */}
        <p className="mt-4 text-center text-xs text-[#6b6b80]">
          Don&apos;t have an account?{' '}
          <Link
            href="/auth/signup"
            className="text-[#f0c040] hover:underline underline-offset-4"
          >
            Sign up
          </Link>
        </p>
      </div>
    </div>
  );
}
