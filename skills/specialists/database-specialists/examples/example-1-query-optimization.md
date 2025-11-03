# Example 1: Query Optimization for E-Commerce Order System

## Scenario

An e-commerce platform experiences severe performance degradation during peak hours. The `get_user_orders` API endpoint takes 8-12 seconds to respond, causing cart abandonment and customer complaints. The database contains:

- **orders** table: 5M rows
- **order_items** table: 25M rows
- **products** table: 100K rows
- **users** table: 2M rows

The problematic query retrieves order history with product details for a user's dashboard.

---

## Initial Query (Slow)

```sql
-- Original query: 8-12 seconds average execution time
SELECT
    o.order_id,
    o.order_date,
    o.total_amount,
    o.status,
    oi.quantity,
    oi.unit_price,
    p.product_name,
    p.category,
    p.image_url,
    u.email,
    u.first_name,
    u.last_name
FROM orders o
LEFT JOIN order_items oi ON o.order_id = oi.order_id
LEFT JOIN products p ON oi.product_id = p.product_id
LEFT JOIN users u ON o.user_id = u.user_id
WHERE o.user_id = 12345
ORDER BY o.order_date DESC;
```

---

## Step 1: EXPLAIN ANALYZE Diagnosis

```sql
EXPLAIN ANALYZE
SELECT
    o.order_id,
    o.order_date,
    o.total_amount,
    o.status,
    oi.quantity,
    oi.unit_price,
    p.product_name,
    p.category,
    p.image_url,
    u.email,
    u.first_name,
    u.last_name
FROM orders o
LEFT JOIN order_items oi ON o.order_id = oi.order_id
LEFT JOIN products p ON oi.product_id = p.product_id
LEFT JOIN users u ON o.user_id = u.user_id
WHERE o.user_id = 12345
ORDER BY o.order_date DESC;
```

**Output reveals:**
```
Seq Scan on orders  (cost=0.00..125000.00 rows=50 width=256) (actual time=125.432..8234.521 rows=87 loops=1)
  Filter: (user_id = 12345)
  Rows Removed by Filter: 4999913
Hash Join  (cost=15234.00..325000.00 rows=500 width=512) (actual time=8234.623..11245.832 rows=143 loops=1)
  Hash Cond: (oi.product_id = p.product_id)
  -> Seq Scan on order_items oi  (cost=0.00..285000.00 rows=25000000 width=32)
Planning Time: 2.345 ms
Execution Time: 11247.892 ms
```

**Problems identified:**
1. ❌ Sequential scan on 5M row `orders` table (no index on `user_id`)
2. ❌ Hash join scanning entire 25M row `order_items` table
3. ❌ Unnecessary `users` table join (user data already known)
4. ❌ Fetching all columns including large BLOBs (`image_url`)

---

## Step 2: Index Creation Strategy

```sql
-- Create composite index for user_id + order_date (supports WHERE + ORDER BY)
CREATE INDEX CONCURRENTLY idx_orders_user_date
ON orders(user_id, order_date DESC);

-- Create index on order_items for efficient join
CREATE INDEX CONCURRENTLY idx_order_items_order_id
ON order_items(order_id)
INCLUDE (product_id, quantity, unit_price);

-- Create index on products for join optimization
CREATE INDEX CONCURRENTLY idx_products_id
ON products(product_id)
INCLUDE (product_name, category);

-- Analyze tables to update statistics
ANALYZE orders;
ANALYZE order_items;
ANALYZE products;
```

**Index rationale:**
- `CONCURRENTLY` prevents table locking during creation
- `INCLUDE` clause adds covering index data (PostgreSQL 11+)
- Composite index on `(user_id, order_date DESC)` enables index-only scan
- Covering indexes eliminate table lookups

---

## Step 3: Query Refactoring

```sql
-- Optimized query: <200ms average execution time
WITH user_orders AS (
    SELECT
        order_id,
        order_date,
        total_amount,
        status
    FROM orders
    WHERE user_id = 12345
    ORDER BY order_date DESC
    LIMIT 50  -- Pagination: only fetch recent orders
)
SELECT
    uo.order_id,
    uo.order_date,
    uo.total_amount,
    uo.status,
    COALESCE(
        json_agg(
            json_build_object(
                'quantity', oi.quantity,
                'unit_price', oi.unit_price,
                'product_name', p.product_name,
                'category', p.category
            ) ORDER BY oi.line_number
        ) FILTER (WHERE oi.order_item_id IS NOT NULL),
        '[]'::json
    ) AS items
FROM user_orders uo
LEFT JOIN order_items oi ON uo.order_id = oi.order_id
LEFT JOIN products p ON oi.product_id = p.product_id
GROUP BY uo.order_id, uo.order_date, uo.total_amount, uo.status
ORDER BY uo.order_date DESC;
```

**Optimizations applied:**
1. ✅ CTE isolates user order filtering (uses covering index)
2. ✅ `LIMIT 50` implements pagination (fetch only visible orders)
3. ✅ Removed unnecessary `users` table join
4. ✅ Excluded large `image_url` BLOB (fetch separately if needed)
5. ✅ JSON aggregation reduces result set size (1 row per order vs 1 row per item)
6. ✅ `FILTER` clause handles NULL gracefully

---

## Step 4: Verification with EXPLAIN ANALYZE

```sql
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
WITH user_orders AS (
    SELECT order_id, order_date, total_amount, status
    FROM orders
    WHERE user_id = 12345
    ORDER BY order_date DESC
    LIMIT 50
)
SELECT
    uo.order_id,
    uo.order_date,
    uo.total_amount,
    uo.status,
    COALESCE(json_agg(json_build_object(...)) FILTER (...), '[]'::json) AS items
FROM user_orders uo
LEFT JOIN order_items oi ON uo.order_id = oi.order_id
LEFT JOIN products p ON oi.product_id = p.product_id
GROUP BY uo.order_id, uo.order_date, uo.total_amount, uo.status
ORDER BY uo.order_date DESC;
```

**Optimized execution plan:**
```json
{
  "Plan": {
    "Node Type": "Group Aggregate",
    "Total Cost": 245.32,
    "Actual Total Time": 156.234,
    "Plans": [{
      "Node Type": "Index Scan",
      "Index Name": "idx_orders_user_date",
      "Scan Direction": "Backward",
      "Actual Rows": 50,
      "Buffers": {
        "Shared Hit Blocks": 12,
        "Shared Read Blocks": 3
      }
    }]
  },
  "Execution Time": 157.892
}
```

**Performance improvement:**
- Before: 11,247ms (11.2 seconds)
- After: 157ms (0.16 seconds)
- **71x faster** 🚀

---

## Step 5: Application-Level Optimization

```javascript
// Node.js API endpoint with caching
const Redis = require('ioredis');
const redis = new Redis();

async function getUserOrders(userId, page = 1, limit = 50) {
    const cacheKey = `user:${userId}:orders:page:${page}`;

    // Check cache first
    const cached = await redis.get(cacheKey);
    if (cached) {
        return JSON.parse(cached);
    }

    // Execute optimized query with parameterization
    const offset = (page - 1) * limit;
    const result = await db.query(`
        WITH user_orders AS (
            SELECT order_id, order_date, total_amount, status
            FROM orders
            WHERE user_id = $1
            ORDER BY order_date DESC
            LIMIT $2 OFFSET $3
        )
        SELECT
            uo.order_id,
            uo.order_date,
            uo.total_amount,
            uo.status,
            COALESCE(json_agg(...) FILTER (...), '[]'::json) AS items
        FROM user_orders uo
        LEFT JOIN order_items oi ON uo.order_id = oi.order_id
        LEFT JOIN products p ON oi.product_id = p.product_id
        GROUP BY uo.order_id, uo.order_date, uo.total_amount, uo.status
        ORDER BY uo.order_date DESC;
    `, [userId, limit, offset]);

    // Cache for 5 minutes
    await redis.setex(cacheKey, 300, JSON.stringify(result.rows));

    return result.rows;
}
```

---

## Outcomes & Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **P50 Latency** | 8,234ms | 157ms | 98.1% reduction |
| **P95 Latency** | 12,456ms | 289ms | 97.7% reduction |
| **P99 Latency** | 18,234ms | 412ms | 97.7% reduction |
| **Database CPU** | 85% avg | 12% avg | 85.9% reduction |
| **Cart Abandonment** | 23% | 4.2% | 81.7% reduction |
| **Customer Satisfaction** | 3.2/5 | 4.7/5 | +46.9% |

**Additional benefits:**
- ✅ Reduced database connection pool exhaustion
- ✅ Lower cloud infrastructure costs ($4,200/month savings)
- ✅ Eliminated need for read replica scaling
- ✅ Improved cache hit ratio (62% with Redis)

---

## Key Takeaways

### 1. **Always Start with EXPLAIN ANALYZE**
Never guess at query performance. Use `EXPLAIN (ANALYZE, BUFFERS)` to identify:
- Sequential scans on large tables
- Missing indexes
- Inefficient join strategies
- High disk I/O

### 2. **Index Strategy Matters**
- Composite indexes support `WHERE` + `ORDER BY`
- Covering indexes (`INCLUDE` clause) eliminate table lookups
- Use `CREATE INDEX CONCURRENTLY` in production
- Monitor index bloat with `pg_stat_user_indexes`

### 3. **Pagination is Essential**
Fetching all rows is rarely necessary. Implement:
- `LIMIT` + `OFFSET` for simple pagination
- Keyset pagination for better performance at scale
- Cursor-based pagination for real-time data

### 4. **JSON Aggregation Reduces Network I/O**
Aggregating child records (order items) into JSON:
- Reduces result set size (1 row vs N rows)
- Simplifies application-level processing
- Improves cache efficiency

### 5. **Cache Strategically**
- Cache expensive queries with Redis/Memcached
- Use short TTLs (5-15 minutes) for frequently updated data
- Invalidate cache on writes (`SET order_id` triggers `DEL user:*:orders:*`)

### 6. **Avoid Anti-Patterns**
- ❌ `SELECT *` (fetch only needed columns)
- ❌ Joining unnecessary tables
- ❌ Fetching BLOBs in list queries
- ❌ Missing parameterization (SQL injection risk)

### 7. **Monitor Continuously**
Set up alerts for:
- Query execution time > 1 second
- Sequential scans on large tables
- Index bloat > 30%
- Connection pool utilization > 80%

---

## Tools Used

- **PostgreSQL EXPLAIN ANALYZE** - Query plan analysis
- **pg_stat_statements** - Query performance tracking
- **pgBadger** - Log analysis and visualization
- **Redis** - Application-level caching
- **Grafana + Prometheus** - Real-time monitoring
- **k6** - Load testing and benchmarking

---

## Next Steps

1. **Implement materialized views** for complex aggregations
2. **Partition large tables** by date range (monthly partitions)
3. **Add connection pooling** (PgBouncer) for better resource utilization
4. **Set up query performance alerts** in monitoring stack
5. **Schedule VACUUM ANALYZE** during off-peak hours
