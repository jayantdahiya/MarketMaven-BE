# AGENTS.md — Agentic Coding Guidelines for MarketMaven Frontend

## Overview

Next.js 16 + React 19 frontend for the MarketMaven AI stock forecasting platform. Renders LSTM/CNN-Transformer/Mamba model signals from the FastAPI backend. Uses Tailwind CSS v4, Radix UI primitives, shadcn-style components, SWR for data fetching, and TypeScript strict mode throughout.

---

## Build & Dev Commands

```bash
# Package management (uses pnpm, not npm)
pnpm install          # Install dependencies
pnpm add <package>    # Add a dependency

# Development
pnpm dev              # Dev server on localhost:3000 (Turbopack)
pnpm build            # Production build
pnpm start            # Serve production build
pnpm lint             # Run ESLint (Next.js + TypeScript rules)
```

**No test runner is configured** — there are currently no test files in this repo. Do not add a test framework unless explicitly requested.

---

## Architecture

```
app/                     # Next.js App Router pages
  layout.tsx             # Root layout: UIModeProvider, Navbar, footer
  page.tsx               # Dashboard (ticker grid + metrics bar)
  forecast/[ticker]/     # Per-ticker forecast detail page
components/
  dashboard/             # Dashboard-specific: TickerCard
  forecast/              # Signal/confidence UI: SignalBadge, ConfidenceBar
  layout/                # Navbar
  ui/                    # Primitive components (Button, Card, Badge, etc.)
hooks/
  use-ui-mode.tsx        # UIModeContext: 'retail' | 'quant' toggle, localStorage
lib/
  api.ts                 # All fetch calls to FastAPI backend + SWR keys
  utils.ts               # cn(), formatReturn(), modelLabel(), TICKER_NAMES, etc.
types/
  api.ts                 # TypeScript types mirroring backend Pydantic schemas
```

- `NEXT_PUBLIC_API_URL` env var controls backend base URL (defaults to `http://localhost:8000`)
- See `.env.local.example` for required environment variables
- SWR refreshes forecasts every 5 minutes; disable with `refreshInterval: 0` when needed

---

## Code Style

### TypeScript
- **Strict mode** is enabled (`"strict": true` in tsconfig). No `any` without justification.
- Use `type` imports where possible: `import type { Foo } from '...'`
- Prefer union types over enums: `'retail' | 'quant'` not `enum UIMode`
- Use `interface` for object shapes (props, API types), `type` for unions/aliases
- Return types are required on all exported functions

```ts
// Correct
export function formatReturn(value: number): string { ... }

// Incorrect
export function formatReturn(value) { ... }
```

### Imports
- Always use the `@/` path alias (maps to repo root): `import { cn } from '@/lib/utils'`
- Order: React/Next → third-party → `@/` local imports
- Use `import type` for type-only imports

```ts
import React, { useState } from 'react';
import useSWR from 'swr';
import type { DailyForecastResponse } from '@/types/api';
import { fetchForecast, swrKeys } from '@/lib/api';
import { cn } from '@/lib/utils';
```

### Naming Conventions
- **Components**: `PascalCase` files and exports — `TickerCard.tsx` → `export function TickerCard`
- **Hooks**: `use-kebab-case.tsx` file, `useCamelCase` export
- **Utilities/helpers**: `camelCase` functions in `lib/utils.ts`
- **Types**: `PascalCase` — `DailyForecastResponse`, `ForecastSignal`
- **Constants**: `UPPER_SNAKE_CASE` — `TICKER_NAMES`, `STORAGE_KEY`
- **Files**: `kebab-case` for all files and directories

### React / Next.js
- Mark client components with `'use client'` at the top; omit for Server Components
- Prefer named exports over default exports for components (except page/layout files, which Next.js requires as default)
- Use `React.forwardRef` and set `.displayName` on forwarded-ref components (see `button.tsx`)
- Guard against hydration mismatches with `mounted` state pattern (see `use-ui-mode.tsx`)
- Always provide `key` props when mapping; prefer semantic keys over array indices

### Styling
- **Tailwind CSS v4** with `@theme inline` for design tokens in `globals.css`
- Use design tokens via hex literals from `globals.css`; do not hardcode arbitrary colors
- Use the `cn()` helper (`clsx` + `tailwind-merge`) for conditional class merging
- Use `cva` (class-variance-authority) for variant-driven component APIs (see `button.tsx`)
- Component variants: `size`, `variant` props following shadcn conventions
- Dark-only design; no light mode. Background `#09090f`, surface `#111118`, foreground `#e8e8f0`

### Data Fetching (SWR)
- All API calls go through `lib/api.ts` — do not call `fetch` directly in components
- Use `swrKeys.*` for stable, deduplicated SWR cache keys
- Pass `null` as the SWR key to conditionally disable fetching: `useSWR(isQuant ? key : null, ...)`
- Handle `isLoading`, `error`, and empty-data states explicitly — always render a skeleton or error UI

### Error Handling
- Throw descriptive errors from `lib/api.ts`: `throw new Error(\`API \${res.status}: \${text}\`)`
- In components, check `error` from `useSWR` and render a user-visible fallback (see `TickerCard`)
- Never silently swallow errors; catch blocks should at minimum include a comment explaining why

### Component Patterns
- Co-locate small sub-components in the same file (e.g., `DataRow`, `TickerCardSkeleton`)
- Export only the primary component; keep helpers unexported
- Props interfaces are always named `<ComponentName>Props`

```tsx
interface SignalBadgeProps {
  signal: ForecastSignal;
  size?: 'sm' | 'md' | 'lg';
}

export function SignalBadge({ signal, size = 'md' }: SignalBadgeProps) { ... }
```

### Types (`types/api.ts`)
- This file mirrors FastAPI Pydantic schemas **exactly** — field names use `snake_case` to match the backend JSON
- Add new types here when the backend adds new response shapes; do not define ad-hoc inline types in components

---

## Documentation (`docs/`)

The `docs/` folder contains three LLM-oriented markdown files. **Read them before writing or
modifying code** — they are the authoritative source of truth for design decisions, constraints,
and planned work.

| File | Purpose |
|---|---|
| `docs/FRONTEND_GUIDE.md` | Master context file: stack versions, Tailwind v4 rules, color palette, typography, component conventions, API layer details, hard rules. Read this first. |
| `docs/IMPLEMENTATION_PHASES.md` | Roadmap tied to backend model phases (0–4). Describes what to build next for each phase, with concrete component names, prop shapes, and backend endpoint dependencies. |
| `docs/TASKS.md` | Granular task checklist. Shows what is done (`[x]`), in progress (`[~]`), todo (`[ ]`), or cancelled (`[-]`). Update this as work is completed. |

### Key points from the docs

- **Hard rules** from `FRONTEND_GUIDE.md` that agents must follow:
  - Never use `npm`/`yarn` — pnpm only.
  - Never create `tailwind.config.js` — all theme config belongs in `app/globals.css`.
  - Never call `fetch()` directly in components — always go through `lib/api.ts`.
  - Never introduce colors without adding them to `@theme inline {}` first.
  - `font-mono tabular-nums` on every rendered numeric value.
  - Keep `components/ui/` pure — no SWR calls, no `useUIMode()`, no business logic.
  - Every data-displaying component must implement both retail and quant views.

- **Current phase**: Phase 0 (LSTM baseline). Phases 1–4 are blocked on backend model milestones.
  Check `docs/IMPLEMENTATION_PHASES.md` for what each phase requires before starting work.

- **Task hygiene**: when completing a task, mark it `[x]` in `docs/TASKS.md`. When starting one,
  mark it `[~]`. New tasks discovered during implementation go at the bottom of the relevant
  phase section.

---

## ESLint
- Config: `eslint.config.mjs` — Next.js core-web-vitals + TypeScript rules (ESLint v9 flat config)
- Run `pnpm lint` before committing; the CI will fail on lint errors
- `next/` paths and build output are ignored automatically

---

## Environment

Copy `.env.local.example` → `.env.local`. Key variable:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```
