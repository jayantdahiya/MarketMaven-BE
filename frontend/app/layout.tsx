import type { Metadata } from 'next';
import './globals.css';
import { UIModeProvider } from '@/hooks/use-ui-mode';
import { Navbar } from '@/components/layout/navbar';

export const metadata: Metadata = {
  title: 'MarketMaven — AI Stock Forecasting',
  description:
    'Deep learning powered stock signal dashboard. Real-time forecasts from LSTM, CNN-Transformer, and Mamba SSM models.',
  icons: { icon: '/favicon.ico' },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
      </head>
      <body className="antialiased min-h-screen bg-[#09090f]">
        <UIModeProvider>
          <Navbar />
          <main className="mx-auto max-w-7xl px-6 py-8">{children}</main>
          <footer className="mt-16 border-t border-[#1a1a24] py-6">
            <div className="mx-auto max-w-7xl px-6">
              <p className="text-xs text-[#6b6b80]">
                MarketMaven — AI signals are for informational purposes only.
                Not financial advice.
              </p>
            </div>
          </footer>
        </UIModeProvider>
      </body>
    </html>
  );
}
