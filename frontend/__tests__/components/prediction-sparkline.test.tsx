import { describe, it, expect, vi } from 'vitest';
import { render } from '@testing-library/react';
import { PredictionSparkline } from '@/components/forecast/prediction-sparkline';
import type { PredictionPoint } from '@/types/api';

// recharts uses ResizeObserver and SVG measurement which are not available in jsdom.
// Stub ResizeObserver so recharts ResponsiveContainer renders without errors.
vi.stubGlobal(
  'ResizeObserver',
  class {
    observe() {}
    unobserve() {}
    disconnect() {}
  },
);

const SAMPLE_PREDICTIONS: PredictionPoint[] = [
  { date: '2024-01-02', predicted_return: 0.01, actual_return: 0.008, signal: 'long' },
  { date: '2024-01-03', predicted_return: -0.005, actual_return: -0.003, signal: 'flat' },
  { date: '2024-01-04', predicted_return: 0.02, actual_return: 0.015, signal: 'long' },
];

describe('PredictionSparkline', () => {
  it('renders nothing when predictions array is empty', () => {
    const { container } = render(<PredictionSparkline predictions={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders a container div with correct default height', () => {
    const { container } = render(
      <PredictionSparkline predictions={SAMPLE_PREDICTIONS} />,
    );
    const wrapper = container.firstElementChild as HTMLElement;
    expect(wrapper).not.toBeNull();
    expect(wrapper.style.height).toBe('72px');
  });

  it('applies custom height prop', () => {
    const { container } = render(
      <PredictionSparkline predictions={SAMPLE_PREDICTIONS} height={120} />,
    );
    const wrapper = container.firstElementChild as HTMLElement;
    expect(wrapper.style.height).toBe('120px');
  });

  it('renders a recharts wrapper div when given data', () => {
    const { container } = render(
      <PredictionSparkline predictions={SAMPLE_PREDICTIONS} />,
    );
    // recharts renders a div with class "recharts-responsive-container" inside the height wrapper
    const rechartsDiv = container.querySelector('.recharts-responsive-container');
    expect(rechartsDiv).not.toBeNull();
  });
});
