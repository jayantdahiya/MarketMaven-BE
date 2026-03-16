-- Idempotent migration for ticker API backing table.
-- Safe to run multiple times.

-- Ensure schema exists (public usually exists already).
create schema if not exists public;

-- Create backing table used by GET /tickers.
create table if not exists public.tickers (
    id bigint generated always as identity primary key,
    symbol text not null unique,
    name text,
    is_active boolean not null default true,
    created_at timestamptz not null default now()
);

-- Enable RLS (required when using anon/publishable key in the API).
alter table public.tickers enable row level security;

-- Allow anon users to read tickers if policy does not already exist.
do $$
begin
    if not exists (
        select 1
        from pg_policies
        where schemaname = 'public'
          and tablename = 'tickers'
          and policyname = 'anon can read tickers'
    ) then
        create policy "anon can read tickers"
        on public.tickers
        for select
        to anon
        using (true);
    end if;
end
$$;

-- Seed default symbols only if missing.
insert into public.tickers (symbol, name)
values
    ('AAPL', 'Apple Inc.'),
    ('MSFT', 'Microsoft Corporation'),
    ('GOOGL', 'Alphabet Inc.'),
    ('AMZN', 'Amazon.com, Inc.'),
    ('SPY', 'SPDR S&P 500 ETF Trust')
on conflict (symbol) do nothing;

-- Ask PostgREST to reload schema cache so the table is immediately visible to API.
notify pgrst, 'reload schema';
