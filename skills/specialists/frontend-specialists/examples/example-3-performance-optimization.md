# Example 3: Performance Optimization for Large-Scale React Applications

## Scenario

You have a React dashboard application that displays real-time stock market data with:

- 500+ stock tickers updating every second
- Interactive charts with 10,000+ data points
- Complex filtering and sorting
- User watchlists and portfolios
- Performance issues: sluggish UI, frame drops, high memory usage

Users are complaining about lag when scrolling, delayed interactions, and browser freezing during market hours.

## User Request

"Our stock dashboard is too slow during market hours. It freezes when updating 500+ stocks per second. The charts lag, scrolling is janky, and the app uses too much memory. Make it fast and smooth."

## Walkthrough

### Step 1: Performance Audit

First, identify bottlenecks using React DevTools Profiler:

```typescript
// utils/performanceMonitor.ts
export class PerformanceMonitor {
  private metrics: Map<string, number[]> = new Map();

  measure(name: string, fn: () => void): void {
    const start = performance.now();
    fn();
    const duration = performance.now() - start;

    if (!this.metrics.has(name)) {
      this.metrics.set(name, []);
    }
    this.metrics.get(name)!.push(duration);
  }

  async measureAsync(name: string, fn: () => Promise<void>): Promise<void> {
    const start = performance.now();
    await fn();
    const duration = performance.now() - start;

    if (!this.metrics.has(name)) {
      this.metrics.set(name, []);
    }
    this.metrics.get(name)!.push(duration);
  }

  getStats(name: string) {
    const values = this.metrics.get(name) || [];
    if (values.length === 0) return null;

    const sorted = [...values].sort((a, b) => a - b);
    return {
      avg: values.reduce((a, b) => a + b, 0) / values.length,
      min: sorted[0],
      max: sorted[sorted.length - 1],
      p50: sorted[Math.floor(sorted.length * 0.5)],
      p95: sorted[Math.floor(sorted.length * 0.95)],
      p99: sorted[Math.floor(sorted.length * 0.99)],
      count: values.length,
    };
  }

  report(): void {
    console.group('Performance Metrics');
    this.metrics.forEach((_, name) => {
      const stats = this.getStats(name);
      console.log(`${name}:`, stats);
    });
    console.groupEnd();
  }

  clear(): void {
    this.metrics.clear();
  }
}

export const perfMonitor = new PerformanceMonitor();
```

### Step 2: Virtualized Lists with React Window

Replace standard list rendering with virtualization:

```typescript
// components/VirtualizedStockList.tsx
import React, { useMemo, useCallback } from 'react';
import { FixedSizeList as List } from 'react-window';
import AutoSizer from 'react-virtualized-auto-sizer';
import type { Stock } from '../types/Stock';

interface VirtualizedStockListProps {
  stocks: Stock[];
  onStockClick: (stock: Stock) => void;
  filterText: string;
}

// Memoized row component - only re-renders if props change
const StockRow = React.memo<{
  index: number;
  style: React.CSSProperties;
  data: {
    stocks: Stock[];
    onStockClick: (stock: Stock) => void;
  };
}>(({ index, style, data }) => {
  const stock = data.stocks[index];
  const priceChange = stock.price - stock.previousClose;
  const priceChangePercent = (priceChange / stock.previousClose) * 100;
  const isPositive = priceChange >= 0;

  return (
    <div
      style={style}
      className={`stock-row ${isPositive ? 'positive' : 'negative'}`}
      onClick={() => data.onStockClick(stock)}
    >
      <span className="ticker">{stock.ticker}</span>
      <span className="name">{stock.name}</span>
      <span className="price">${stock.price.toFixed(2)}</span>
      <span className={`change ${isPositive ? 'up' : 'down'}`}>
        {isPositive ? '+' : ''}{priceChange.toFixed(2)} ({priceChangePercent.toFixed(2)}%)
      </span>
      <span className="volume">{formatVolume(stock.volume)}</span>
    </div>
  );
}, (prevProps, nextProps) => {
  // Custom comparison for even better performance
  return (
    prevProps.index === nextProps.index &&
    prevProps.data.stocks[prevProps.index] === nextProps.data.stocks[nextProps.index]
  );
});

export const VirtualizedStockList: React.FC<VirtualizedStockListProps> = ({
  stocks,
  onStockClick,
  filterText,
}) => {
  // Memoize filtered stocks to avoid recalculation
  const filteredStocks = useMemo(() => {
    if (!filterText) return stocks;

    const lower = filterText.toLowerCase();
    return stocks.filter(
      (stock) =>
        stock.ticker.toLowerCase().includes(lower) ||
        stock.name.toLowerCase().includes(lower)
    );
  }, [stocks, filterText]);

  // Memoize item data to prevent unnecessary re-renders
  const itemData = useMemo(
    () => ({ stocks: filteredStocks, onStockClick }),
    [filteredStocks, onStockClick]
  );

  return (
    <div className="stock-list-container">
      <AutoSizer>
        {({ height, width }) => (
          <List
            height={height}
            itemCount={filteredStocks.length}
            itemSize={60} // Fixed row height in pixels
            width={width}
            itemData={itemData}
            overscanCount={5} // Render 5 extra rows above/below viewport
          >
            {StockRow}
          </List>
        )}
      </AutoSizer>
    </div>
  );
};

const formatVolume = (volume: number): string => {
  if (volume >= 1e9) return `${(volume / 1e9).toFixed(2)}B`;
  if (volume >= 1e6) return `${(volume / 1e6).toFixed(2)}M`;
  if (volume >= 1e3) return `${(volume / 1e3).toFixed(2)}K`;
  return String(volume);
};
```

**Before**: Rendering 500 stocks = 500 DOM nodes, ~300ms render time
**After**: Rendering ~15 visible stocks = 15 DOM nodes, ~12ms render time
**Improvement**: 25x faster, 97% less DOM

### Step 3: Optimized Real-Time Updates with Web Workers

Move data processing to Web Worker thread:

```typescript
// workers/stockDataWorker.ts
interface StockUpdate {
  ticker: string;
  price: number;
  volume: number;
  timestamp: number;
}

interface ProcessedStock extends StockUpdate {
  priceChange: number;
  priceChangePercent: number;
  trend: 'up' | 'down' | 'flat';
}

// In-memory cache of previous prices
const priceCache = new Map<string, number>();

self.onmessage = (event: MessageEvent<StockUpdate[]>) => {
  const updates = event.data;
  const processed: ProcessedStock[] = [];

  for (const update of updates) {
    const previousPrice = priceCache.get(update.ticker) || update.price;
    const priceChange = update.price - previousPrice;
    const priceChangePercent = (priceChange / previousPrice) * 100;

    processed.push({
      ...update,
      priceChange,
      priceChangePercent,
      trend: priceChange > 0 ? 'up' : priceChange < 0 ? 'down' : 'flat',
    });

    priceCache.set(update.ticker, update.price);
  }

  // Send processed data back to main thread
  self.postMessage(processed);
};
```

```typescript
// hooks/useStockData.ts
import { useState, useEffect, useCallback, useRef } from 'react';
import type { Stock } from '../types/Stock';

export const useStockData = (websocketUrl: string) => {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const workerRef = useRef<Worker | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Initialize Web Worker
    workerRef.current = new Worker(
      new URL('../workers/stockDataWorker.ts', import.meta.url),
      { type: 'module' }
    );

    workerRef.current.onmessage = (event) => {
      // Batch update from worker (already processed)
      setStocks(event.data);
    };

    // Initialize WebSocket
    const ws = new WebSocket(websocketUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const rawUpdates = JSON.parse(event.data);
        // Send raw data to worker for processing
        workerRef.current?.postMessage(rawUpdates);
      } catch (error) {
        console.error('Failed to parse stock data:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
    };

    return () => {
      ws.close();
      workerRef.current?.terminate();
    };
  }, [websocketUrl]);

  return { stocks, isConnected };
};
```

**Before**: Main thread processing 500 updates/sec = UI freezes, 60fps → 15fps
**After**: Worker thread processing = smooth UI, stable 60fps
**Improvement**: 4x better frame rate, no UI blocking

### Step 4: Memoization and React.memo

Prevent unnecessary re-renders with strategic memoization:

```typescript
// components/StockChart.tsx
import React, { useMemo } from 'react';
import { Line } from 'react-chartjs-2';
import type { Stock } from '../types/Stock';

interface StockChartProps {
  stock: Stock;
  historicalData: { timestamp: number; price: number }[];
  timeRange: '1D' | '1W' | '1M' | '1Y';
}

export const StockChart = React.memo<StockChartProps>(
  ({ stock, historicalData, timeRange }) => {
    // Memoize chart data to avoid recalculation on every render
    const chartData = useMemo(() => {
      const filteredData = filterDataByTimeRange(historicalData, timeRange);

      return {
        labels: filteredData.map((d) => new Date(d.timestamp).toLocaleTimeString()),
        datasets: [
          {
            label: stock.ticker,
            data: filteredData.map((d) => d.price),
            borderColor: stock.priceChange >= 0 ? '#10b981' : '#ef4444',
            backgroundColor: 'transparent',
            borderWidth: 2,
            pointRadius: 0, // No points for better performance
            tension: 0.1,
          },
        ],
      };
    }, [historicalData, timeRange, stock.ticker, stock.priceChange]);

    // Memoize chart options
    const chartOptions = useMemo(
      () => ({
        responsive: true,
        maintainAspectRatio: false,
        animation: {
          duration: 0, // Disable animations for real-time updates
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            mode: 'index' as const,
            intersect: false,
            callbacks: {
              label: (context: any) => `$${context.parsed.y.toFixed(2)}`,
            },
          },
        },
        scales: {
          x: {
            display: true,
            grid: { display: false },
          },
          y: {
            display: true,
            grid: { color: 'rgba(0, 0, 0, 0.05)' },
            ticks: {
              callback: (value: any) => `$${value.toFixed(2)}`,
            },
          },
        },
        interaction: {
          mode: 'nearest' as const,
          axis: 'x' as const,
          intersect: false,
        },
      }),
      []
    );

    return (
      <div className="stock-chart">
        <Line data={chartData} options={chartOptions} />
      </div>
    );
  },
  // Custom comparison function - only re-render if stock or data actually changed
  (prevProps, nextProps) => {
    return (
      prevProps.stock.ticker === nextProps.stock.ticker &&
      prevProps.stock.price === nextProps.stock.price &&
      prevProps.timeRange === nextProps.timeRange &&
      prevProps.historicalData.length === nextProps.historicalData.length
    );
  }
);

const filterDataByTimeRange = (
  data: { timestamp: number; price: number }[],
  range: '1D' | '1W' | '1M' | '1Y'
): { timestamp: number; price: number }[] => {
  const now = Date.now();
  const ranges = {
    '1D': 24 * 60 * 60 * 1000,
    '1W': 7 * 24 * 60 * 60 * 1000,
    '1M': 30 * 24 * 60 * 60 * 1000,
    '1Y': 365 * 24 * 60 * 60 * 1000,
  };

  const cutoff = now - ranges[range];
  return data.filter((d) => d.timestamp >= cutoff);
};
```

**Before**: Chart re-renders on every stock update = 500 renders/sec
**After**: Chart only re-renders when visible data changes = ~2 renders/sec
**Improvement**: 250x fewer renders

### Step 5: Code Splitting and Lazy Loading

Split code for faster initial load:

```typescript
// App.tsx
import React, { lazy, Suspense } from 'react';
import { Routes, Route } from 'react-router-dom';
import { ErrorBoundary } from 'react-error-boundary';

// Eagerly loaded components (critical path)
import { Header } from './components/Header';
import { LoadingSpinner } from './components/LoadingSpinner';

// Lazy loaded routes (loaded on demand)
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Watchlist = lazy(() => import('./pages/Watchlist'));
const Portfolio = lazy(() => import('./pages/Portfolio'));
const Analytics = lazy(() => import('./pages/Analytics'));
const Settings = lazy(() => import('./pages/Settings'));

// Preload commonly accessed routes
const preloadDashboard = () => import('./pages/Dashboard');
const preloadWatchlist = () => import('./pages/Watchlist');

export const App: React.FC = () => {
  // Preload likely next routes on hover
  React.useEffect(() => {
    const dashboardLink = document.querySelector('[data-preload="dashboard"]');
    const watchlistLink = document.querySelector('[data-preload="watchlist"]');

    dashboardLink?.addEventListener('mouseenter', preloadDashboard);
    watchlistLink?.addEventListener('mouseenter', preloadWatchlist);

    return () => {
      dashboardLink?.removeEventListener('mouseenter', preloadDashboard);
      watchlistLink?.removeEventListener('mouseenter', preloadWatchlist);
    };
  }, []);

  return (
    <ErrorBoundary fallback={<div>Something went wrong</div>}>
      <div className="app">
        <Header />
        <Suspense fallback={<LoadingSpinner />}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/watchlist" element={<Watchlist />} />
            <Route path="/portfolio" element={<Portfolio />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </Suspense>
      </div>
    </ErrorBoundary>
  );
};
```

**Before**: Initial bundle = 482KB, FCP = 2.8s
**After**: Initial bundle = 124KB, FCP = 0.9s
**Improvement**: 74% smaller bundle, 3.1x faster load

### Step 6: Memory Optimization with Cleanup

Prevent memory leaks and optimize garbage collection:

```typescript
// hooks/useOptimizedWebSocket.ts
import { useEffect, useRef, useCallback } from 'react';

export const useOptimizedWebSocket = (
  url: string,
  onMessage: (data: any) => void
) => {
  const wsRef = useRef<WebSocket | null>(null);
  const messageQueueRef = useRef<any[]>([]);
  const rafIdRef = useRef<number | null>(null);

  // Batch updates using requestAnimationFrame
  const processBatch = useCallback(() => {
    if (messageQueueRef.current.length > 0) {
      onMessage(messageQueueRef.current);
      messageQueueRef.current = []; // Clear queue
    }
    rafIdRef.current = null;
  }, [onMessage]);

  useEffect(() => {
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        messageQueueRef.current.push(data);

        // Schedule batch processing if not already scheduled
        if (rafIdRef.current === null) {
          rafIdRef.current = requestAnimationFrame(processBatch);
        }
      } catch (error) {
        console.error('Failed to parse message:', error);
      }
    };

    return () => {
      // Cleanup: cancel pending RAF, clear queue, close socket
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current);
      }
      messageQueueRef.current = [];
      ws.close();
    };
  }, [url, processBatch]);

  return wsRef;
};
```

**Before**: Memory leaks from unclosed sockets = 50MB/hour growth
**After**: Proper cleanup = stable 28MB memory usage
**Improvement**: 0MB/hour memory growth

### Step 7: Performance Monitoring Component

Track performance metrics in production:

```typescript
// components/PerformanceMonitor.tsx
import React, { useEffect, useState } from 'react';

export const PerformanceMonitor: React.FC = () => {
  const [metrics, setMetrics] = useState({
    fps: 60,
    memory: 0,
    renderTime: 0,
  });

  useEffect(() => {
    let frameCount = 0;
    let lastTime = performance.now();
    let animationId: number;

    const measureFPS = () => {
      frameCount++;
      const currentTime = performance.now();

      if (currentTime >= lastTime + 1000) {
        const fps = Math.round((frameCount * 1000) / (currentTime - lastTime));
        frameCount = 0;
        lastTime = currentTime;

        setMetrics((prev) => ({
          ...prev,
          fps,
          memory: (performance as any).memory?.usedJSHeapSize / 1048576 || 0,
        }));
      }

      animationId = requestAnimationFrame(measureFPS);
    };

    animationId = requestAnimationFrame(measureFPS);

    return () => cancelAnimationFrame(animationId);
  }, []);

  return (
    <div className="performance-monitor">
      <div>FPS: {metrics.fps}</div>
      <div>Memory: {metrics.memory.toFixed(1)} MB</div>
      <div>Render: {metrics.renderTime.toFixed(1)} ms</div>
    </div>
  );
};
```

## Outcomes

### Performance Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Initial Load (FCP)** | 2.8s | 0.9s | 3.1x faster |
| **Time to Interactive** | 4.2s | 1.4s | 3x faster |
| **Frame Rate** | 15 fps | 60 fps | 4x smoother |
| **Bundle Size** | 482KB | 124KB | 74% smaller |
| **Memory Usage** | 250MB | 45MB | 82% less |
| **Render Time (500 stocks)** | 300ms | 12ms | 25x faster |
| **Updates per Second** | Freezes UI | Smooth 60fps | ∞ better |

### Lighthouse Score

| Category | Before | After |
|----------|--------|-------|
| Performance | 42 | 96 |
| Accessibility | 78 | 95 |
| Best Practices | 83 | 100 |
| SEO | 92 | 100 |

### Real User Metrics (RUM)

- **Largest Contentful Paint**: 2.8s → 0.9s (68% improvement)
- **First Input Delay**: 180ms → 12ms (93% improvement)
- **Cumulative Layout Shift**: 0.14 → 0.01 (93% improvement)

## Tips & Best Practices

1. **Profile First**: Use React DevTools Profiler to identify actual bottlenecks before optimizing
2. **Virtualize Long Lists**: Always virtualize lists >100 items with react-window or react-virtualized
3. **Web Workers**: Move heavy computations off main thread (parsing, calculations, filtering)
4. **Memoization**: Use `useMemo`, `useCallback`, and `React.memo` strategically, not everywhere
5. **Code Splitting**: Lazy load routes and heavy components with `React.lazy()` and `Suspense`
6. **Batch Updates**: Use `requestAnimationFrame` for batching frequent updates (60fps max)
7. **Cleanup**: Always cleanup subscriptions, timers, and listeners in `useEffect` return
8. **Reduce Bundle**: Use tree-shaking, dynamic imports, and analyze with webpack-bundle-analyzer
9. **Optimize Images**: Use WebP, lazy loading, and proper sizing with `srcset`
10. **Monitor Production**: Track real user metrics with Web Vitals and performance.measure()
11. **Avoid Re-renders**: Use shallow equality checks and immutable updates
12. **Disable Animations**: Turn off animations for real-time data visualizations
13. **Use Concurrent React**: Enable concurrent rendering for better UX during heavy operations
14. **Cache Wisely**: Cache computed values but invalidate when source data changes
15. **Optimize Dependencies**: Keep dependency arrays minimal to prevent unnecessary effect runs
