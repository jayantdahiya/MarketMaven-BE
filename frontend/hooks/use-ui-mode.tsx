'use client';

import React, {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react';
import type { UIMode } from '@/types/api';

interface UIModeContextValue {
  mode: UIMode;
  setMode: (mode: UIMode) => void;
  isQuant: boolean;
}

const UIModeContext = createContext<UIModeContextValue>({
  mode: 'retail',
  setMode: () => {},
  isQuant: false,
});

const STORAGE_KEY = 'marketmaven-ui-mode';

function readStoredMode(): UIMode {
  try {
    const stored = localStorage.getItem(STORAGE_KEY) as UIMode | null;
    if (stored === 'retail' || stored === 'quant') return stored;
  } catch {
    // localStorage unavailable (SSR / private browsing)
  }
  return 'retail';
}

export function UIModeProvider({ children }: { children: ReactNode }) {
  // mounted tracks hydration; mode is corrected from localStorage on first client render
  const [mounted, setMounted] = useState(false);
  const [mode, setModeState] = useState<UIMode>('retail');

  useEffect(() => {
    // Runs once after hydration — safe to read localStorage here.
    // The setState calls below are intentional: this is the standard Next.js
    // hydration-safe pattern for reading client-only storage after mount.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setModeState(readStoredMode());
    setMounted(true);
  }, []);

  const setMode = (next: UIMode) => {
    setModeState(next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // ignore
    }
  };

  // Avoid hydration mismatch — render with default 'retail' until mounted
  if (!mounted) {
    return (
      <UIModeContext.Provider value={{ mode: 'retail', setMode, isQuant: false }}>
        {children}
      </UIModeContext.Provider>
    );
  }

  return (
    <UIModeContext.Provider value={{ mode, setMode, isQuant: mode === 'quant' }}>
      {children}
    </UIModeContext.Provider>
  );
}

export function useUIMode() {
  return useContext(UIModeContext);
}
