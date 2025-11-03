---
name: react-specialist
description: Modern React development specialist for React 18+ with hooks, context, suspense, server components (Next.js 13+), state management (Redux/Zustand/Jotai), performance optimization (React.memo, useMemo, useCallback), and component library development. Use when building React applications, optimizing rendering performance, implementing complex state management, or creating reusable component libraries.
category: Frontend Specialists
complexity: Medium
triggers: ["react", "react 18", "hooks", "next.js", "redux", "zustand", "react performance", "server components", "suspense", "react testing library"]
---

# React Specialist

Expert React development for modern, performant, and maintainable frontend applications.

## Purpose

Provide comprehensive React expertise including React 18+ features (concurrent rendering, suspense, server components), performance optimization, state management patterns, and production-grade component architecture. Ensures React applications follow best practices and leverage the latest React capabilities.

## When to Use This Skill

- Building React applications with modern patterns
- Optimizing React performance (re-renders, bundle size, lazy loading)
- Implementing complex state management (global state, server state)
- Creating reusable component libraries with TypeScript
- Migrating to React 18+ or Next.js App Router
- Setting up React testing with React Testing Library and Jest
- Implementing accessibility (a11y) in React components

## Prerequisites

**Required**: JavaScript ES6+, TypeScript basics, HTML/CSS, npm/yarn/pnpm

**Agent Assignments**: `coder` (implementation), `tester` (React Testing Library), `mobile-dev` (React Native if needed)

## Core Workflows

### Workflow 1: Next.js 13+ App Router with Server Components

**Step 1: Initialize Next.js Project**

```bash
npx create-next-app@latest my-app --typescript --tailwind --app --no-src-dir
cd my-app
pnpm install
```

**Step 2: Create Server Component (RSC)**

```tsx
// app/users/page.tsx (Server Component by default)
import { Suspense } from 'react';
import { UserList } from './user-list';
import { UserSkeleton } from './user-skeleton';

async function getUsers() {
  const res = await fetch('https://api.example.com/users', {
    next: { revalidate: 60 } // ISR: revalidate every 60s
  });
  return res.json();
}

export default async function UsersPage() {
  const users = await getUsers();

  return (
    <main>
      <h1>Users</h1>
      <Suspense fallback={<UserSkeleton />}>
        <UserList users={users} />
      </Suspense>
    </main>
  );
}
```

**Step 3: Create Client Component with Interactivity**

```tsx
// app/users/user-list.tsx
'use client'; // Marks as Client Component

import { useState } from 'react';

interface User {
  id: number;
  name: string;
  email: string;
}

export function UserList({ users }: { users: User[] }) {
  const [filter, setFilter] = useState('');

  const filtered = users.filter(u =>
    u.name.toLowerCase().includes(filter.toLowerCase())
  );

  return (
    <div>
      <input
        type="text"
        placeholder="Filter users..."
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        className="border p-2 mb-4"
      />
      <ul>
        {filtered.map(user => (
          <li key={user.id}>{user.name} ({user.email})</li>
        ))}
      </ul>
    </div>
  );
}
```

**Step 4: Implement Server Actions**

```tsx
// app/actions.ts
'use server';

import { revalidatePath } from 'next/cache';

export async function createUser(formData: FormData) {
  const name = formData.get('name') as string;
  const email = formData.get('email') as string;

  await fetch('https://api.example.com/users', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, email }),
  });

  revalidatePath('/users'); // Revalidate users page
}
```

### Workflow 2: State Management with Zustand

**Step 1: Install Zustand**

```bash
pnpm add zustand
```

**Step 2: Create Type-Safe Store**

```tsx
// stores/user-store.ts
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

interface User {
  id: number;
  name: string;
}

interface UserState {
  users: User[];
  addUser: (user: User) => void;
  removeUser: (id: number) => void;
  clearUsers: () => void;
}

export const useUserStore = create<UserState>()(
  devtools(
    persist(
      (set) => ({
        users: [],
        addUser: (user) => set((state) => ({
          users: [...state.users, user]
        })),
        removeUser: (id) => set((state) => ({
          users: state.users.filter(u => u.id !== id)
        })),
        clearUsers: () => set({ users: [] }),
      }),
      { name: 'user-storage' }
    )
  )
);
```

**Step 3: Use Store in Components**

```tsx
// components/user-manager.tsx
'use client';

import { useUserStore } from '@/stores/user-store';

export function UserManager() {
  const users = useUserStore((state) => state.users);
  const addUser = useUserStore((state) => state.addUser);
  const removeUser = useUserStore((state) => state.removeUser);

  return (
    <div>
      <button onClick={() => addUser({ id: Date.now(), name: 'New User' })}>
        Add User
      </button>
      <ul>
        {users.map(user => (
          <li key={user.id}>
            {user.name}
            <button onClick={() => removeUser(user.id)}>Remove</button>
          </li>
        ))}
      </ul>
    </div>
  );
}
```

### Workflow 3: Performance Optimization

**Step 1: Memoize Expensive Computations**

```tsx
import { useMemo } from 'react';

function DataTable({ data }: { data: Item[] }) {
  // ✅ Memoize expensive filtering/sorting
  const sorted = useMemo(() => {
    return data.sort((a, b) => a.value - b.value);
  }, [data]);

  return <table>{/* render sorted */}</table>;
}
```

**Step 2: Prevent Unnecessary Re-renders**

```tsx
import { memo, useCallback } from 'react';

interface ChildProps {
  onAction: () => void;
}

// ✅ Memoize component
const Child = memo(function Child({ onAction }: ChildProps) {
  return <button onClick={onAction}>Action</button>;
});

function Parent() {
  // ✅ Stable callback reference
  const handleAction = useCallback(() => {
    console.log('Action triggered');
  }, []);

  return <Child onAction={handleAction} />;
}
```

**Step 3: Code Splitting with Lazy Loading**

```tsx
import { lazy, Suspense } from 'react';

const HeavyComponent = lazy(() => import('./heavy-component'));

function App() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <HeavyComponent />
    </Suspense>
  );
}
```

**Step 4: Optimize Re-renders with React DevTools Profiler**

```tsx
import { Profiler } from 'react';

function onRenderCallback(
  id: string,
  phase: 'mount' | 'update',
  actualDuration: number
) {
  console.log(`${id} (${phase}) took ${actualDuration}ms`);
}

function App() {
  return (
    <Profiler id="App" onRender={onRenderCallback}>
      <MyComponent />
    </Profiler>
  );
}
```

### Workflow 4: Testing with React Testing Library

**Step 1: Write Component Tests**

```tsx
// __tests__/user-list.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { UserList } from '@/components/user-list';

describe('UserList', () => {
  const mockUsers = [
    { id: 1, name: 'Alice', email: 'alice@example.com' },
    { id: 2, name: 'Bob', email: 'bob@example.com' },
  ];

  it('renders user list', () => {
    render(<UserList users={mockUsers} />);
    expect(screen.getByText('Alice')).toBeInTheDocument();
    expect(screen.getByText('Bob')).toBeInTheDocument();
  });

  it('filters users by name', () => {
    render(<UserList users={mockUsers} />);
    const input = screen.getByPlaceholderText('Filter users...');

    fireEvent.change(input, { target: { value: 'alice' } });

    expect(screen.getByText('Alice')).toBeInTheDocument();
    expect(screen.queryByText('Bob')).not.toBeInTheDocument();
  });
});
```

**Step 2: Test Hooks with `renderHook`**

```tsx
import { renderHook, act } from '@testing-library/react';
import { useCounter } from '@/hooks/use-counter';

describe('useCounter', () => {
  it('increments counter', () => {
    const { result } = renderHook(() => useCounter(0));

    expect(result.current.count).toBe(0);

    act(() => {
      result.current.increment();
    });

    expect(result.current.count).toBe(1);
  });
});
```

## Best Practices

**1. Server Components by Default (Next.js 13+)**
```tsx
// ✅ Server Component (default in app/ directory)
async function ServerComponent() {
  const data = await fetchData();
  return <div>{data}</div>;
}

// Only use 'use client' when needed
'use client';
function ClientComponent() {
  const [state, setState] = useState(0);
  return <button onClick={() => setState(state + 1)}>{state}</button>;
}
```

**2. Avoid Prop Drilling, Use Context Strategically**
```tsx
// ✅ GOOD: Context for global UI state
const ThemeContext = createContext<'light' | 'dark'>('light');

// ❌ BAD: Don't pass props through 5+ levels
<A><B><C><D><E prop={value} /></D></C></B></A>
```

**3. Custom Hooks for Reusable Logic**
```tsx
// hooks/use-fetch.ts
function useFetch<T>(url: string) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(url)
      .then(res => res.json())
      .then(setData)
      .finally(() => setLoading(false));
  }, [url]);

  return { data, loading };
}
```

**4. TypeScript for Props**
```tsx
interface ButtonProps {
  variant: 'primary' | 'secondary';
  size?: 'sm' | 'md' | 'lg';
  children: React.ReactNode;
  onClick?: () => void;
}

function Button({ variant, size = 'md', children, onClick }: ButtonProps) {
  return <button onClick={onClick}>{children}</button>;
}
```

**5. Accessibility (a11y)**
```tsx
// ✅ GOOD: Semantic HTML, ARIA labels, keyboard support
<button
  aria-label="Close modal"
  onClick={onClose}
  onKeyDown={(e) => e.key === 'Escape' && onClose()}
>
  <CloseIcon aria-hidden="true" />
</button>

// ❌ BAD: Non-semantic, no ARIA
<div onClick={onClose}>
  <CloseIcon />
</div>
```

## Quality Criteria

- ✅ TypeScript strict mode enabled
- ✅ No prop drilling beyond 3 levels
- ✅ Performance: Lighthouse score ≥90
- ✅ Test coverage ≥80%
- ✅ Accessibility: WCAG 2.1 AA compliance
- ✅ Bundle size: <200KB initial load (gzipped)

## Common Patterns

**Pattern 1: Compound Components**
```tsx
interface TabsContextValue {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

const TabsContext = createContext<TabsContextValue | null>(null);

function Tabs({ children }: { children: React.ReactNode }) {
  const [activeTab, setActiveTab] = useState('tab1');
  return (
    <TabsContext.Provider value={{ activeTab, setActiveTab }}>
      {children}
    </TabsContext.Provider>
  );
}

function TabList({ children }: { children: React.ReactNode }) {
  return <div role="tablist">{children}</div>;
}

function Tab({ value, children }: { value: string; children: React.ReactNode }) {
  const { activeTab, setActiveTab } = useContext(TabsContext)!;
  return (
    <button
      role="tab"
      aria-selected={activeTab === value}
      onClick={() => setActiveTab(value)}
    >
      {children}
    </button>
  );
}

// Usage
<Tabs>
  <TabList>
    <Tab value="tab1">Tab 1</Tab>
    <Tab value="tab2">Tab 2</Tab>
  </TabList>
</Tabs>
```

**Pattern 2: Render Props**
```tsx
interface DataFetcherProps<T> {
  url: string;
  children: (data: T | null, loading: boolean) => React.ReactNode;
}

function DataFetcher<T>({ url, children }: DataFetcherProps<T>) {
  const { data, loading } = useFetch<T>(url);
  return <>{children(data, loading)}</>;
}

// Usage
<DataFetcher<User[]> url="/api/users">
  {(users, loading) => loading ? <Spinner /> : <UserList users={users!} />}
</DataFetcher>
```

## Troubleshooting

**Issue**: "Hydration mismatch" errors in Next.js
**Solution**: Ensure server and client render the same initial HTML. Use `suppressHydrationWarning` for time-dependent content.

**Issue**: Slow performance with large lists
**Solution**: Use virtualization (react-window or react-virtual) to render only visible items.

**Issue**: State updates not batching in React 18
**Solution**: Use `flushSync` sparingly; React 18 auto-batches by default.

## Related Skills

- `typescript-specialist`: TypeScript patterns
- `wcag-accessibility`: Accessibility compliance
- `testing-quality`: Advanced testing strategies
- `docker-containerization`: Containerizing Next.js apps

## Tools

- **Frameworks**: Next.js, Remix, Gatsby
- **State**: Zustand, Jotai, Redux Toolkit, TanStack Query
- **Testing**: React Testing Library, Jest, Vitest, Playwright
- **Performance**: React DevTools Profiler, Lighthouse

## MCP Tools

- `mcp__flow-nexus__sandbox_create` with `template: "react"` for isolated testing
- `mcp__playwright__browser_snapshot` for visual testing
- `mcp__memory-mcp__memory_store` for persisting React patterns

## Success Metrics

- Component creation: 10-15 minutes
- Feature with tests: 1-2 hours
- Lighthouse performance: ≥90
- Test coverage: ≥80%
- Bundle size: <200KB (gzipped)

---

**Skill Version**: 1.0.0
**Last Updated**: 2025-11-02
**Maintained By**: react-specialist agent
