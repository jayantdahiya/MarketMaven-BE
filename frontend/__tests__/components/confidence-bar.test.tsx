import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ConfidenceBar } from '@/components/forecast/confidence-bar';

describe('ConfidenceBar', () => {
  it('renders the percentage label by default', () => {
    render(<ConfidenceBar value={0.75} />);
    expect(screen.getByText('75%')).toBeInTheDocument();
  });

  it('hides the label when showLabel=false', () => {
    render(<ConfidenceBar value={0.75} showLabel={false} />);
    expect(screen.queryByText('75%')).not.toBeInTheDocument();
  });

  it('clamps value above 1 to 100%', () => {
    render(<ConfidenceBar value={1.5} />);
    expect(screen.getByText('100%')).toBeInTheDocument();
  });

  it('clamps value below 0 to 0%', () => {
    render(<ConfidenceBar value={-0.1} />);
    expect(screen.getByText('0%')).toBeInTheDocument();
  });

  it('renders the fill bar with correct width style', () => {
    const { container } = render(<ConfidenceBar value={0.6} showLabel={false} />);
    const fill = container.querySelector('[style*="width: 60%"]') as HTMLElement;
    expect(fill).not.toBeNull();
  });

  it('rounds 0.756 to 76%', () => {
    render(<ConfidenceBar value={0.756} />);
    expect(screen.getByText('76%')).toBeInTheDocument();
  });
});
