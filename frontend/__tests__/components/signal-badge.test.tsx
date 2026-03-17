import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { SignalBadge } from '@/components/forecast/signal-badge';

describe('SignalBadge', () => {
  it('renders "Long" text for long signal', () => {
    render(<SignalBadge signal="long" />);
    expect(screen.getByText('Long')).toBeInTheDocument();
  });

  it('renders "Flat" text for flat signal', () => {
    render(<SignalBadge signal="flat" />);
    expect(screen.getByText('Flat')).toBeInTheDocument();
  });

  it('applies gold colour class for long signal', () => {
    const { container } = render(<SignalBadge signal="long" />);
    const badge = container.firstElementChild as HTMLElement;
    expect(badge.className).toContain('text-[#f0c040]');
  });

  it('applies muted colour class for flat signal', () => {
    const { container } = render(<SignalBadge signal="flat" />);
    const badge = container.firstElementChild as HTMLElement;
    expect(badge.className).toContain('text-[#9090a8]');
  });

  it('applies sm size classes', () => {
    const { container } = render(<SignalBadge signal="long" size="sm" />);
    const badge = container.firstElementChild as HTMLElement;
    expect(badge.className).toContain('text-[10px]');
  });

  it('applies lg size classes', () => {
    const { container } = render(<SignalBadge signal="long" size="lg" />);
    const badge = container.firstElementChild as HTMLElement;
    expect(badge.className).toContain('text-sm');
  });

  it('defaults to md size', () => {
    const { container } = render(<SignalBadge signal="long" />);
    const badge = container.firstElementChild as HTMLElement;
    expect(badge.className).toContain('text-xs');
  });
});
